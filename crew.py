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

from ast_tools import extract_definitions, functions_touching_lines
from bug_agent import build_bug_agent, build_bug_task
from coordinator_agent import build_coordinator_agent, build_coordinator_task
from documentation_agent import run_documentation_review
from fix_agent import build_fix_agent, build_fix_task
from git_tools import (
    FileChange,
    clone_or_open,
    get_changed_files,
    get_diff_text,
    read_file_at,
    resolve_refs,
)
from performance_agent import build_performance_agent, build_performance_task
from security_agent import build_security_agent, build_security_task
from state import ReviewState

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
                # Never let UI callbacks break the pipeline.
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
        repo, head_sha, file_changes
    )

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
    }

    # ---------- 2. Documentation Agent (direct path) ----------
    if "documentation" in enabled:
        _emit("agent_start", "Documentation Agent (AST + LLM mismatch check)")
        doc_findings = run_documentation_review(
            repo, head_sha, file_changes, use_llm_mismatch_check=not skip_llm
        )
        state["documentation_findings"] = [f.to_dict() for f in doc_findings]
        _emit(
            "agent_done",
            f"Documentation Agent: {len(doc_findings)} finding(s)",
        )

    if skip_llm:
        _emit("done", "Skipped LLM agents (--skip-llm)")
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
        detection_results = _run_detection_crew(
            inputs, llm_detection_agents, _emit
        )
        state["bug_findings"] = detection_results.get("bug", [])
        state["security_findings"] = detection_results.get("security", [])
        state["performance_findings"] = detection_results.get("performance", [])

    # ---------- 4. Fix crew ----------
    all_findings = (
        state["bug_findings"]
        + state["security_findings"]
        + state["performance_findings"]
        + [
            f
            for f in state["documentation_findings"]
            if f.get("file") != "<summary>"
        ]
    )
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
        _emit("agent_done", f"Fix Agent: {len(fixes)} suggestion(s)")

    _emit("done", "Review complete")
    return PipelineResult(state=state)


# --------------------------------------------------------------- helpers


def _collect_changed_function_sources(
    repo: git.Repo, head_sha: str, file_changes: list[FileChange]
) -> str:
    """Single string with the source of every changed function."""
    chunks: list[str] = []
    for fc in file_changes:
        if fc.is_deleted:
            continue
        source = read_file_at(repo, head_sha, fc.path)
        if source is None:
            continue
        defs = extract_definitions(source, filename=fc.path)
        touched = functions_touching_lines(defs, fc.changed_lines)
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
        verbose=False,
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
        verbose=False,
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
    return [item for item in items if isinstance(item, dict)]


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
