"""Pipeline orchestration: prep → detection crew → fix crew.

Two crews instead of one is intentional: the Fix Agent needs the union
of detection findings as input, so it has to run after the others finish.
Two crews makes the data flow explicit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import git
from crewai import Crew, Process
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.ast_tools import extract_definitions, functions_touching_lines, run_static_bug_checks
from agents.bug_agent import build_bug_agent, build_bug_task
from agents.coordinator_agent import build_coordinator_agent, build_coordinator_task
from agents.documentation_agent import run_documentation_review
from agents.fix_agent import build_fix_agent, build_fix_task
from tools.git_tools import (
    FileChange,
    clone_or_open,
    get_changed_files,
    get_diff_text,
    read_file_at,
    resolve_refs,
)
from agents.performance_agent import build_performance_agent, build_performance_task
from agents.security_agent import build_security_agent, build_security_task
from agents.synthesis_agent import build_synthesis_agent, build_synthesis_task
from core.logger import ReviewLogger
from core.state import ReviewState

# Callback signature: (event_name: str, detail: str) -> None
# Streamlit hooks into this to show a live progress log.
ProgressCallback = "callable[[str, str], None] | None"


@dataclass
class PipelineResult:
    state: ReviewState


def run_review(
    repo_url_or_path: str,
    base: str | None = None,
    head: str | None = None,
    skip_llm: bool = False,
    enabled_agents: set[str] | None = None,
    progress_callback=None,
) -> PipelineResult:
    """Run the full multi-agent review pipeline.

    Parameters
    ----------
    skip_llm:
        When True, only deterministic Documentation checks run. Useful for
        smoke tests or environments without Ollama.
    enabled_agents:
        Subset of {"bug", "security", "performance", "documentation"}.
        Defaults to all four. Disabled agents are skipped entirely; the
        Fix Agent then only sees findings from the agents that ran.
    progress_callback:
        Optional callable taking (event_name, detail). Called at each
        pipeline milestone so a UI can show live progress.
    """
    # Default: every agent runs.
    enabled = enabled_agents or {"bug", "security", "performance", "documentation"}

    def _emit(event: str, detail: str = "") -> None:
        if progress_callback is not None:
            try:
                progress_callback(event, detail)
            except Exception:
                pass

    # ---------- 1. Prep ----------
    _emit("prep", "Cloning / opening repository")
    repo = clone_or_open(repo_url_or_path)
    base_sha, head_sha = resolve_refs(repo, base, head)

    _emit("prep", f"Computing diff {base_sha[:8]}..{head_sha[:8]}")
    diff_text = get_diff_text(repo, base_sha, head_sha)
    file_changes = get_changed_files(repo, base_sha, head_sha)

    _emit("prep", f"Extracting changed functions from {len(file_changes)} file(s)")
    changed_functions_source = _collect_changed_function_sources(
        repo, head_sha, base_sha, file_changes
    )

    # Start logger — records everything from here on.
    logger = ReviewLogger(repo_url_or_path, base_sha, head_sha)
    logger.log_event("prep", f"Repo: {repo.working_dir}")
    logger.log_event("prep", f"Comparing {base_sha[:8]} → {head_sha[:8]}")
    logger.log_event("prep", f"Changed files: {[fc.path for fc in file_changes]}")
    logger.log_diff(diff_text)

    state: ReviewState = {
        "repo_path": repo.working_dir,
        "base_ref": base_sha,
        "head_ref": head_sha,
        "diff": diff_text,
        "changed_files": [fc.path for fc in file_changes],
        "bug_findings": [],
        "security_findings": [],
        "performance_findings": [],
        "documentation_findings": [],
        "fixes": [],
        "synthesis": {},
    }

    # ---------- 2. Documentation Agent (direct path) ----------
    if "documentation" in enabled:
        _emit("agent_start", "Documentation Agent (AST + LLM mismatch check)")
        logger.log_event("agent_start", "Documentation Agent")
        doc_findings = run_documentation_review(
            repo, head_sha, file_changes, use_llm_mismatch_check=not skip_llm
        )
        state["documentation_findings"] = [f.to_dict() for f in doc_findings]
        logger.log_agent_output(
            "Documentation Agent",
            json.dumps([f.to_dict() for f in doc_findings], indent=2),
        )
        _emit("agent_done", f"Documentation Agent: {len(doc_findings)} finding(s)")
        logger.log_event("agent_done", f"Documentation Agent: {len(doc_findings)} finding(s)")

    # ---------- 2b. Static bug checks (deterministic, no LLM) ----------
    _emit("prep", "Running static bug checks (AST)")
    static_bug_findings: list[dict] = []
    for fc in file_changes:
        if fc.is_deleted:
            continue
        source = read_file_at(repo, head_sha, fc.path)
        if source is None:
            continue
        if not fc.path.endswith(".py"):
            continue
        static_bug_findings.extend(run_static_bug_checks(source, fc.path))
    for f in static_bug_findings:
        if f.get("category") == "documentation":
            state["documentation_findings"].append(f)
        else:
            state["bug_findings"].append(f)
    logger.log_static_checks(static_bug_findings)
    _emit("prep", f"Static checks: {len(static_bug_findings)} finding(s)")
    logger.log_event("prep", f"Static checks: {len(static_bug_findings)} finding(s)")

    if skip_llm:
        _emit("done", "Skipped LLM agents (--skip-llm)")
        logger.log_event("done", "Skipped LLM agents")
        logger.log_findings_summary(state)
        logger.close()
        return PipelineResult(state=state)

    # ---------- 3. Detection crew ----------
    # Build the crew with only the enabled LLM-driven detection agents.
    llm_detection_agents = enabled & {"bug", "security", "performance"}
    if llm_detection_agents:
        inputs = {
            "repo_path": repo.working_dir,
            "base_ref": base_sha,
            "head_ref": head_sha,
            "diff": _truncate_diff(diff_text),
            "changed_files": "\n".join(state["changed_files"]),
            "changed_functions_source": changed_functions_source,
        }

        _emit(
            "crew_start",
            f"Detection crew: {', '.join(sorted(llm_detection_agents))}",
        )
        logger.log_event("crew_start", f"Detection crew: {', '.join(sorted(llm_detection_agents))}")
        logger.log_agent_input("Detection Crew", inputs)
        detection_results = _run_detection_crew(
            inputs, llm_detection_agents, _emit
        )
        for cat in ("bug", "security", "performance"):
            findings = detection_results.get(cat, [])
            if findings:
                logger.log_agent_output(
                    f"{cat.capitalize()} Agent",
                    json.dumps(findings, indent=2),
                )
        # Merge LLM findings with static findings already in state (don't overwrite).
        state["bug_findings"] = state["bug_findings"] + detection_results.get("bug", [])
        state["security_findings"] = state["security_findings"] + detection_results.get("security", [])
        state["performance_findings"] = state["performance_findings"] + detection_results.get("performance", [])

    # ---------- 4. Fix crew ----------
    # Only send findings about surviving files — don't ask Fix Agent to patch deleted code.
    _surviving = set(state["changed_files"])
    all_findings = [
        f for f in (
            state["bug_findings"]
            + state["security_findings"]
            + state["performance_findings"]
            + [f for f in state["documentation_findings"] if f.get("file") != "<summary>"]
        )
        if f.get("file", "") in _surviving or f.get("file", "") in ("?", "")
    ]
    if all_findings:
        _emit("agent_start", f"Fix Agent ({len(all_findings)} finding(s) to fix)")
        fixes = _run_fix_crew(
            {
                "repo_path": repo.working_dir,
                "base_ref": base_sha,
                "head_ref": head_sha,
                "diff": _truncate_diff(diff_text),
                "changed_files": "\n".join(state["changed_files"]),
                "changed_functions_source": changed_functions_source,
                "all_findings_json": json.dumps(all_findings, indent=2),
            }
        )
        state["fixes"] = fixes
        logger.log_agent_output("Fix Agent", json.dumps(fixes, indent=2))
        _emit("agent_done", f"Fix Agent: {len(fixes)} suggestion(s)")
        logger.log_event("agent_done", f"Fix Agent: {len(fixes)} suggestion(s)")

    # ---------- 5. Synthesis Agent ----------
    # Hard-filter to surviving files only — deleted-file findings confuse the LLM.
    surviving_files = set(state["changed_files"])
    _raw_for_synthesis = [
        f for f in (
            state["bug_findings"]
            + state["security_findings"]
            + state["performance_findings"]
            + [f for f in state["documentation_findings"] if f.get("file") != "<summary>"]
        )
        if f.get("file", "") in surviving_files or f.get("file", "") in ("?", "")
    ]
    # Deduplicate deterministically: same (file, line, issue[:60]) → keep first occurrence.
    _seen_keys: set[tuple] = set()
    all_findings_for_synthesis: list[dict] = []
    for f in _raw_for_synthesis:
        key = (f.get("file", ""), f.get("line"), f.get("issue", "")[:60])
        if key not in _seen_keys:
            _seen_keys.add(key)
            all_findings_for_synthesis.append(f)
    if all_findings_for_synthesis:
        _emit("agent_start", f"Synthesis Agent ({len(all_findings_for_synthesis)} finding(s))")
        synthesis = _run_synthesis(
            {
                "all_findings_json": json.dumps(all_findings_for_synthesis, indent=2),
                "diff": _truncate_diff(diff_text, max_chars=6000),
                "changed_files": "\n".join(state["changed_files"]),
            },
            original_findings=all_findings_for_synthesis,
        )
        state["synthesis"] = synthesis
        logger.log_agent_output("Synthesis Agent", json.dumps(synthesis, indent=2))
        _emit("agent_done", "Synthesis Agent: consolidated report ready")
        logger.log_event("agent_done", "Synthesis Agent done")

    _emit("done", "Review complete")
    logger.log_event("done", "Review complete")
    logger.log_findings_summary(state)
    logger.close()
    print(f"\n[Log saved to: {logger.path}]", flush=True)
    return PipelineResult(state=state)


# --------------------------------------------------------------- helpers


def _collect_changed_function_sources(
    repo: git.Repo, head_sha: str, base_sha: str, file_changes: list[FileChange]
) -> str:
    """Single string with the source of every changed function.

    Strategy:
    - For modified files: prefer functions whose lines were added/changed.
      If none (e.g. pure deletion diff), fall back to ALL functions in the
      file so LLM agents still have context.
    - For deleted files: pull functions from the BASE commit so agents can
      reason about what was removed.
    """
    chunks: list[str] = []

    for fc in file_changes:
        if fc.is_deleted:
            # Deleted file — read from base so agents can see what was removed.
            source = read_file_at(repo, base_sha, fc.path)
            if source is None:
                continue
            defs = extract_definitions(source, filename=fc.path)
            for d in defs:
                chunks.append(
                    f"### {fc.path} [DELETED] :: {d.qualified_name} "
                    f"(lines {d.start_line}-{d.end_line})\n"
                    f"```python\n{d.source}\n```"
                )
            continue

        source = read_file_at(repo, head_sha, fc.path)
        if source is None:
            continue
        defs = extract_definitions(source, filename=fc.path)

        # Primary: functions touching added/modified lines.
        touched = functions_touching_lines(defs, fc.changed_lines)

        # Fallback: if diff was all deletions, include every function in the
        # file so agents aren't handed an empty context.
        if not touched and defs:
            touched = defs

        for d in touched:
            chunks.append(
                f"### {fc.path} :: {d.qualified_name} "
                f"(lines {d.start_line}-{d.end_line})\n"
                f"```python\n{d.source}\n```"
            )

    return "\n\n".join(chunks) if chunks else "(no changed functions extracted)"


def _truncate_diff(diff: str, max_chars: int = 12000) -> str:
    """Cap diff size sent to the LLM."""
    if len(diff) <= max_chars:
        return diff
    return diff[:max_chars] + "\n\n[... diff truncated ...]"


def _run_detection_crew(
    inputs: dict, enabled: set[str], emit
) -> dict[str, list[dict]]:
    """Coordinator + selected detection agents, sequential.

    The Coordinator always runs (it's cheap and produces the framing note).
    Specialist agents are added conditionally based on `enabled`.
    """
    coordinator = build_coordinator_agent()
    agents = [coordinator]
    tasks = [build_coordinator_task(coordinator)]

    # Track which slot in tasks_output corresponds to which category, so we
    # can parse results back to the right key after the crew finishes.
    slot_categories: list[str] = []  # index → category, parallel to agent specialists

    if "bug" in enabled:
        emit("agent_start", "Bug Agent")
        a = build_bug_agent()
        agents.append(a)
        tasks.append(build_bug_task(a))
        slot_categories.append("bug")
    if "security" in enabled:
        emit("agent_start", "Security Agent (with Bandit + detect-secrets)")
        a = build_security_agent()
        agents.append(a)
        tasks.append(build_security_task(a))
        slot_categories.append("security")
    if "performance" in enabled:
        emit("agent_start", "Performance Agent")
        a = build_performance_agent()
        agents.append(a)
        tasks.append(build_performance_task(a))
        slot_categories.append("performance")

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff(inputs=inputs)

    task_outputs = getattr(result, "tasks_output", []) or []
    parsed: dict[str, list[dict]] = {"bug": [], "security": [], "performance": []}

    # task_outputs[0] is the Coordinator's framing note. Specialists start at 1.
    for i, category in enumerate(slot_categories, start=1):
        if i < len(task_outputs):
            findings = _parse_findings(_to_string(task_outputs[i]), category)
            parsed[category] = findings
            emit("agent_done", f"{category.capitalize()} Agent: {len(findings)} finding(s)")
    return parsed


def _run_fix_crew(inputs: dict) -> list[dict]:
    fix = build_fix_agent()
    crew = Crew(
        agents=[fix],
        tasks=[build_fix_task(fix)],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff(inputs=inputs)
    task_outputs = getattr(result, "tasks_output", []) or []
    if not task_outputs:
        return []
    return _parse_fixes(_to_string(task_outputs[0]))


def _to_string(task_output) -> str:
    """Coerce a CrewAI TaskOutput to its raw string.

    Attribute name has shifted between CrewAI versions; try each.
    """
    for attr in ("raw", "raw_output", "output", "result"):
        if hasattr(task_output, attr):
            v = getattr(task_output, attr)
            if isinstance(v, str):
                return v
    return str(task_output)


_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", re.DOTALL)


def _parse_findings(raw: str, category: str) -> list[dict]:
    """Parse a JSON array of findings; stamp each with its category."""
    items = _extract_json_array(raw)
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item.setdefault("category", category)
        item.setdefault("severity", "low")
        item.setdefault("file", "?")
        item.setdefault("issue", "")
        out.append(item)
    return out


def _parse_fixes(raw: str) -> list[dict]:
    items = _extract_json_array(raw)
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("file", "original_issue", "explanation", "suggested_code"):
            if key in item and not isinstance(item[key], str):
                item[key] = str(item[key]) if item[key] is not None else ""
        out.append(item)
    return out


def _run_synthesis(inputs: dict, original_findings: list[dict] | None = None) -> dict:
    """Run the Synthesis Agent and return its parsed JSON output."""
    agent = build_synthesis_agent()
    crew = Crew(
        agents=[agent],
        tasks=[build_synthesis_task(agent)],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff(inputs=inputs)
    task_outputs = getattr(result, "tasks_output", []) or []
    if not task_outputs:
        return {}
    raw = _to_string(task_outputs[0])
    synthesis = _parse_synthesis(raw)
    # Enforce: restore original severities — never let the LLM change them.
    if original_findings and synthesis.get("prioritised_findings"):
        _restore_severities(synthesis["prioritised_findings"], original_findings)
    return synthesis


def _parse_synthesis(raw: str) -> dict:
    """Extract the synthesis JSON object from LLM output."""
    raw = raw.strip()
    # Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return _normalise_synthesis(parsed)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return _normalise_synthesis(parsed)
        except json.JSONDecodeError:
            pass
    return {}


def _restore_severities(
    prioritised: list[dict], originals: list[dict]
) -> None:
    """Overwrite any LLM-modified severity with the original deterministic value.

    Matches on (file, line, issue[:60]) — same key used for deduplication.
    """
    index: dict[tuple, str] = {}
    for f in originals:
        key = (f.get("file", ""), f.get("line"), f.get("issue", "")[:60])
        index[key] = f.get("severity", "low")

    for f in prioritised:
        key = (f.get("file", ""), f.get("line"), f.get("issue", "")[:60])
        if key in index:
            f["severity"] = index[key]


def _normalise_synthesis(d: dict) -> dict:
    """Ensure fields are the right types regardless of LLM quirks."""
    narrative = d.get("narrative", "")
    if not isinstance(narrative, str):
        narrative = str(narrative)
    findings = d.get("prioritised_findings", [])
    if not isinstance(findings, list):
        findings = []
    clean = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        for key in ("file", "issue", "action", "category", "severity"):
            if key in f and not isinstance(f[key], str):
                f[key] = str(f[key]) if f[key] is not None else ""
        clean.append(f)
    return {"narrative": narrative, "prioritised_findings": clean}


def _extract_json_array(raw: str) -> list:
    """Lenient JSON-array extraction. Return [] on failure rather than crash."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass
    m = _JSON_ARRAY_RE.search(raw)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []
