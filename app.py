"""Streamlit UI for the PR reviewer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
import traceback

import streamlit as st

from core.crew import run_review
from core.reporter import compute_score, generate_report


# --------------------------------------------------------------- page setup

st.set_page_config(
    page_title="PR Reviewer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multi-Agent PR Reviewer")
st.caption("CrewAI · Ollama · Bandit · detect-secrets")


# --------------------------------------------------------------- sidebar inputs

with st.sidebar:
    st.header("Review settings")

    repo = st.text_input(
        "Repository URL or local path",
        placeholder="https://github.com/user/repo or /path/to/repo",
        help=(
            "Either a remote git URL (will be cloned to /tmp/pr-reviewer) "
            "or an absolute path to a local repo."
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        base_ref = st.text_input(
            "Base ref",
            value="",
            placeholder="HEAD~1",
            help="Leave blank to use HEAD~1 (the previous commit).",
        )
    with col2:
        head_ref = st.text_input(
            "Head ref",
            value="",
            placeholder="HEAD",
            help="Leave blank to use HEAD.",
        )

    st.markdown("---")
    st.subheader("Agents to run")
    # Checkboxes default to all enabled; user can drop any of them. Documentation
    # stays available even with skip-LLM since its AST checks don't need Ollama.
    enable_bug = st.checkbox("🐛 Bug Detection", value=True)
    enable_security = st.checkbox("🔒 Security", value=True)
    enable_performance = st.checkbox("⚡ Performance", value=True)
    enable_documentation = st.checkbox("📝 Documentation", value=True)

    st.markdown("---")
    st.subheader("LLM")
    skip_llm = st.checkbox(
        "Skip LLM (AST checks only)",
        value=False,
        help=(
            "Skip every LLM call. Only deterministic AST checks in the "
            "Documentation agent will run. Useful when Ollama isn't running "
            "or for a fast sanity check."
        ),
    )
    model = st.text_input(
        "Ollama model",
        value=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b"),
        disabled=skip_llm,
        help="Must be a model you've already pulled with `ollama pull <name>`.",
    )

    st.markdown("---")
    run_clicked = st.button("Run Review", type="primary", use_container_width=True)


# --------------------------------------------------------------- progress log helpers


def _icon_for(event: str) -> str:
    return {
        "prep": "⚙️",
        "agent_start": "▶️",
        "agent_done": "✅",
        "crew_start": "🚀",
        "done": "🏁",
    }.get(event, "·")


def _make_progress_callback(placeholder, events: list):
    """Build a callback that appends to `events` and re-renders the log.

    Streamlit's placeholder API is the simplest reliable way to get a
    live-updating log: we keep a list of events in the closure, append on
    each callback, and call `.markdown()` again with the full text. Each
    update replaces the previous render in place.
    """
    start = time.monotonic()

    def cb(event: str, detail: str) -> None:
        elapsed = time.monotonic() - start
        events.append((elapsed, event, detail))
        lines = [
            f"`{e:6.1f}s` {_icon_for(ev)} **{ev}** — {dt}"
            for e, ev, dt in events
        ]
        placeholder.markdown("\n\n".join(lines))

    return cb


# --------------------------------------------------------------- main pane

if not run_clicked:
    st.info(
        "Enter a repository URL or local path in the sidebar and click "
        "**Run Review** to start."
    )
    with st.expander("What does this tool do?"):
        st.markdown(
            """
            This is a multi-agent code reviewer. It checks out a repo, computes
            the diff between two refs, and dispatches it to specialist agents:

            - **Bug Agent** — logic errors, null derefs, swallowed exceptions
            - **Security Agent** — runs Bandit + detect-secrets, then the LLM
              triages results and looks for authorization / IDOR / SSRF issues
            - **Performance Agent** — N+1 queries, hoistable work in loops
            - **Documentation Agent** — AST-based coverage + naming + an LLM
              check for docstrings that no longer match their code
            - **Fix Agent** — generates corrected snippets for the findings

            Local models are slow. Expect a real review to take minutes, not
            seconds.
            """
        )
    st.stop()


# ---------- validate inputs ----------

if not repo.strip():
    st.error("Please enter a repository URL or local path in the sidebar.")
    st.stop()

enabled_agents = set()
if enable_bug:
    enabled_agents.add("bug")
if enable_security:
    enabled_agents.add("security")
if enable_performance:
    enabled_agents.add("performance")
if enable_documentation:
    enabled_agents.add("documentation")

if not enabled_agents:
    st.error("Enable at least one agent in the sidebar.")
    st.stop()

# Apply model override before any agent is constructed.
if model:
    os.environ["OLLAMA_MODEL"] = model


# ---------- run the pipeline ----------

st.subheader("Progress")
log_placeholder = st.empty()
events: list = []
callback = _make_progress_callback(log_placeholder, events)

with st.spinner("Reviewing…"):
    try:
        result = run_review(
            repo.strip(),
            base=base_ref.strip() or None,
            head=head_ref.strip() or None,
            skip_llm=skip_llm,
            enabled_agents=enabled_agents,
            progress_callback=callback,
        )
    except Exception as e:
        st.error(f"Review failed: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()


state = result.state
score = compute_score(state)


# ---------- header / summary ----------

st.markdown("---")
col_score, col_summary = st.columns([1, 3])
with col_score:
    # Color the score so it's glanceable.
    if score >= 90:
        st.metric("Score", f"{score}/100", delta="Clean", delta_color="normal")
    elif score >= 70:
        st.metric("Score", f"{score}/100", delta="Minor issues", delta_color="off")
    elif score >= 50:
        st.metric("Score", f"{score}/100", delta="Needs work", delta_color="inverse")
    else:
        st.metric("Score", f"{score}/100", delta="Significant issues", delta_color="inverse")

with col_summary:
    counts = {
        "Bugs": len(state.get("bug_findings", []) or []),
        "Security": len(state.get("security_findings", []) or []),
        "Performance": len(state.get("performance_findings", []) or []),
        # Subtract 1 for the coverage summary pseudo-finding, but only if it's there.
        "Docs": max(
            0,
            len(state.get("documentation_findings", []) or [])
            - sum(
                1
                for f in state.get("documentation_findings", []) or []
                if f.get("file") == "<summary>"
            ),
        ),
        "Fixes": len(state.get("fixes", []) or []),
    }
    cols = st.columns(len(counts))
    for col, (label, n) in zip(cols, counts.items()):
        col.metric(label, n)

    st.caption(
        f"Files changed: {len(state.get('changed_files', []))} · "
        f"Base: `{(state.get('base_ref') or '')[:8]}` · "
        f"Head: `{(state.get('head_ref') or '')[:8]}`"
    )


# ---------- tabs ----------

tabs = st.tabs(
    [
        "📋 Report",
        "🧠 Synthesis",
        "🐛 Bugs",
        "🔒 Security",
        "⚡ Performance",
        "📝 Docs",
        "🔧 Fixes",
        "📄 Diff",
    ]
)


def _render_findings(findings: list[dict], empty_msg: str) -> None:
    """Render a list of finding dicts as a sorted, severity-tagged list."""
    if not findings:
        st.success(empty_msg)
        return
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_f = sorted(
        findings,
        key=lambda f: (
            severity_order.get(f.get("severity", "info"), 5),
            f.get("file", ""),
            f.get("line") or 0,
        ),
    )
    badge = {
        "critical": "🔴 CRITICAL",
        "high": "🟠 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🔵 LOW",
        "info": "⚪ INFO",
    }
    for f in sorted_f:
        if f.get("file") == "<summary>":
            st.info(f.get("issue", ""))
            continue
        sev = f.get("severity", "info")
        loc = f.get("file", "?")
        if f.get("line"):
            loc = f"{loc}:{f['line']}"
        with st.expander(f"{badge.get(sev, sev.upper())} · `{loc}` · {f.get('issue', '')[:80]}"):
            st.write(f.get("issue", ""))
            if f.get("metadata"):
                with st.expander("Metadata"):
                    st.json(f["metadata"])


with tabs[0]:
    # Full Markdown report + download button.
    report_md = generate_report(state)
    st.markdown(report_md)
    st.download_button(
        "Download report.md",
        data=report_md,
        file_name="review.md",
        mime="text/markdown",
    )

with tabs[1]:
    synthesis = state.get("synthesis") or {}
    if not synthesis:
        st.info("Synthesis not available (no findings, or --skip-llm used).")
    else:
        if synthesis.get("narrative"):
            st.subheader("Summary")
            st.write(synthesis["narrative"])
        pf = synthesis.get("prioritised_findings") or []
        if pf:
            st.subheader("Prioritised Action List")
            severity_badge = {
                "critical": "🔴", "high": "🟠", "medium": "🟡",
                "low": "🔵", "info": "⚪",
            }
            for f in pf:
                rank = f.get("rank", "?")
                sev = str(f.get("severity", "info")).lower()
                badge = severity_badge.get(sev, "·")
                loc = f.get("file", "?")
                if f.get("line"):
                    loc = f"{loc}:{f['line']}"
                label = f"{rank}. {badge} `{loc}` — {f.get('issue', '')[:80]}"
                with st.expander(label):
                    st.write(f.get("issue", ""))
                    if f.get("action"):
                        st.info(f"**Fix:** {f['action']}")
        elif synthesis.get("narrative"):
            st.success("No prioritised findings.")

with tabs[2]:
    _render_findings(state.get("bug_findings", []) or [], "No bugs detected.")

with tabs[3]:
    _render_findings(
        state.get("security_findings", []) or [], "No security issues detected."
    )

with tabs[4]:
    _render_findings(
        state.get("performance_findings", []) or [],
        "No performance issues detected.",
    )

with tabs[5]:
    _render_findings(
        state.get("documentation_findings", []) or [],
        "No documentation issues detected.",
    )

with tabs[6]:
    fixes = state.get("fixes", []) or []
    if not fixes:
        st.info("No fix suggestions generated.")
    else:
        for fix in fixes:
            loc = fix.get("file", "?")
            if fix.get("line"):
                loc = f"{loc}:{fix['line']}"
            with st.expander(f"`{loc}` — {fix.get('original_issue', '')[:80]}"):
                if fix.get("explanation"):
                    st.write(fix["explanation"])
                code = fix.get("suggested_code") or fix.get("suggestion", "")
                if code:
                    st.code(str(code), language="python")

with tabs[7]:
    diff = state.get("diff", "")
    if not diff:
        st.info("No diff (empty or skipped).")
    else:
        st.code(diff, language="diff")
