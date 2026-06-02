"""Documentation & Maintainability Agent.

Hybrid:
  - Deterministic AST checks: coverage %, missing docstrings, undocumented
    parameters, naming violations.
  - One LLM-driven check: does the docstring still describe what the
    function actually does?

Two forms:
  1. `run_documentation_review()`: plain Python entry point. Used by
     crew.py because most of the work is deterministic and going through
     the agent loop would waste tokens.
  2. `build_documentation_agent()` / `build_documentation_task()`: CrewAI
     form, kept for completeness in case you want it inside the Crew.
     
     Changed File
      ↓
    Read Source Code
        ↓
    Extract Functions
        ↓
    Check Missing Docs
        ↓
    Check Parameters
        ↓
    Check Naming
        ↓
    Optional LLM Check
        ↓
    Return Findings
"""

from __future__ import annotations

import json
import os
import re
import sys

import git
from crewai import Agent, Task

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from tools.ast_tools import (
    Definition,
    ExtractFunctionContextTool,
    check_naming,
    check_param_documentation,
    documentation_coverage,
    extract_definitions,
    functions_touching_lines,
)
from core.config import CONFIG
from tools.git_tools import FileChange, ReadFileAtRefTool, read_file_at
from tools.llm import get_llm
from core.state import Finding


_MISMATCH_PROMPT = """You are a senior code reviewer checking if a Python docstring actively CONTRADICTS its function's behaviour.

Function source:
```python
{source}
```

Docstring:
\"\"\"{docstring}\"\"\"

Rules — flag mismatch=true ONLY if the docstring says something that is FACTUALLY WRONG about the code:
  - States the function returns X but it actually returns Y (different type or value)
  - States a side-effect that does NOT happen (e.g. "deletes files" but code never calls os.remove)
  - States NO side-effect but the code clearly has one (e.g. docstring silent on deletion but code removes files)
  - Describes the function as a simple write/save/delete action but the code returns True/False to
    signal success or failure — this is a silent failure pattern the caller cannot know about
  - Describes completely different behaviour from what the code does

Do NOT flag mismatch=true for:
  - The docstring being short or incomplete (missing parameter descriptions, no return type mentioned)
  - Minor wording differences or paraphrasing
  - The docstring not capturing every implementation detail
  - The docstring being a high-level summary that is still broadly accurate

Respond with ONLY a JSON object on a single line, no prose, no markdown fences:
{{"mismatch": true|false, "reason": "<one sentence describing the factual contradiction if true, empty string if false>"}}"""


# --------------------------------------------------------------- direct path


def run_documentation_review(
    repo: git.Repo,
    head_sha: str,
    file_changes: list[FileChange],
    use_llm_mismatch_check: bool = True,
) -> list[Finding]:
    """Run all documentation checks. Returns a list of Finding objects.

    Bypassing the CrewAI agent loop here is deliberate: the deterministic
    checks don't need it, and we only invoke the LLM for the single
    semantic check that benefits from it.
    """
    findings: list[Finding] = []
    total_documented = 0
    total_public = 0

    llm = get_llm() if use_llm_mismatch_check else None

    for fc in file_changes:
        if fc.is_deleted:
            continue
        source = read_file_at(repo, head_sha, fc.path)
        if source is None:
            continue

        defs = extract_definitions(source, filename=fc.path)
        if not defs:
            continue

        documented, total, _ = documentation_coverage(defs)
        total_documented += documented
        total_public += total

        touched = functions_touching_lines(defs, fc.changed_lines)

        findings.extend(_missing_docstrings(fc.path, touched))
        findings.extend(_missing_param_docs(fc.path, touched))
        findings.extend(_naming_violations(fc.path, touched))
        if llm is not None:
            findings.extend(_mismatched_docstrings(fc.path, touched, llm))

    findings.insert(0, _coverage_summary(total_documented, total_public))
    return findings


def _missing_docstrings(path: str, defs: list[Definition]) -> list[Finding]:
    out: list[Finding] = []
    for d in defs:
        if not d.is_public or d.has_docstring:
            continue
        # A long function without a docstring is worse than a one-liner.
        span = d.end_line - d.start_line
        severity = "medium" if span >= 5 else "low"
        out.append(
            Finding(
                category="documentation",
                severity=severity,
                file=path,
                line=d.start_line,
                end_line=d.end_line,
                issue=f"Public {d.kind} '{d.qualified_name}' has no docstring",
                metadata={"qualified_name": d.qualified_name, "kind": d.kind},
            )
        )
    return out


def _missing_param_docs(path: str, defs: list[Definition]) -> list[Finding]:
    out: list[Finding] = []
    for d in defs:
        if d.kind == "class":
            continue
        undocumented = check_param_documentation(d)
        if undocumented:
            out.append(
                Finding(
                    category="documentation",
                    severity="low",
                    file=path,
                    line=d.start_line,
                    end_line=d.end_line,
                    issue=(
                        f"Docstring of '{d.qualified_name}' does not describe "
                        f"parameter(s): {', '.join(undocumented)}"
                    ),
                    metadata={
                        "qualified_name": d.qualified_name,
                        "undocumented_params": undocumented,
                    },
                )
            )
    return out


def _naming_violations(path: str, defs: list[Definition]) -> list[Finding]:
    out: list[Finding] = []
    for d, reason in check_naming(defs):
        out.append(
            Finding(
                category="documentation",
                severity="low",
                file=path,
                line=d.start_line,
                end_line=d.end_line,
                issue=reason,
                metadata={"qualified_name": d.qualified_name},
            )
        )
    return out


def _mismatched_docstrings(
    path: str, defs: list[Definition], llm
) -> list[Finding]:
    out: list[Finding] = []
    for d in defs:
        if d.kind == "class" or not d.has_docstring:
            continue
        # Skip one-liner docstrings — they're deliberately high-level summaries.
        # The LLM cannot reliably distinguish "incomplete" from "wrong" for these.
        docstring = d.docstring or ""
        if len(docstring.strip().splitlines()) == 1 and len(docstring.strip()) < 80:
            continue
        src = _truncate(d.source, CONFIG.max_context_lines)
        prompt = _MISMATCH_PROMPT.format(source=src, docstring=docstring)
        try:
            raw = llm.call(
                [
                    {"role": "system", "content": "You output only valid JSON."},
                    {"role": "user", "content": prompt},
                ]
            )
            parsed = _parse_json_response(raw)
        except Exception as e:
            # Never let a flaky LLM kill the review.
            out.append(
                Finding(
                    category="documentation",
                    severity="info",
                    file=path,
                    line=d.start_line,
                    issue=(
                        f"Docstring mismatch check skipped for "
                        f"'{d.qualified_name}': {e}"
                    ),
                    metadata={"qualified_name": d.qualified_name},
                )
            )
            continue

        if parsed.get("mismatch"):
            out.append(
                Finding(
                    category="documentation",
                    severity="medium",
                    file=path,
                    line=d.start_line,
                    end_line=d.end_line,
                    issue=(
                        f"Docstring of '{d.qualified_name}' may no longer "
                        f"match the implementation: {parsed.get('reason', '')}"
                    ),
                    metadata={"qualified_name": d.qualified_name},
                )
            )
    return out


def _coverage_summary(documented: int, total: int) -> Finding:
    pct = (documented / total) if total else 1.0
    severity = "info" if pct >= CONFIG.doc_coverage_warn_threshold else "medium"
    return Finding(
        category="documentation",
        severity=severity,
        file="<summary>",
        issue=(
            f"Documentation coverage on changed files: "
            f"{documented}/{total} public functions documented ({pct:.0%})"
        ),
        metadata={
            "documented": documented,
            "total": total,
            "coverage": round(pct, 4),
        },
    )


def _truncate(source: str, max_lines: int) -> str:
    lines = source.splitlines()
    if len(lines) <= max_lines:
        return source
    half = max_lines // 2
    return "\n".join(lines[:half] + ["    # ... (truncated) ..."] + lines[-half:])


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_response(raw: str) -> dict:
    """Pull a JSON object out of an LLM response. Lenient about fences/preamble."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        raise ValueError(f"No JSON object in LLM response: {raw[:200]!r}")
    return json.loads(m.group(0))


# --------------------------------------------------------------- CrewAI form


def build_documentation_agent() -> Agent:
    """CrewAI form of the Documentation Agent. Optional; crew.py uses the direct path."""
    return Agent(
        role="Documentation & Maintainability Reviewer",
        goal=(
            "Evaluate documentation coverage, missing docstrings, outdated "
            "docstrings, and naming-convention violations."
        ),
        backstory=(
            "You are a maintainer who has been burned too many times by "
            "out-of-date docstrings. You care about readers six months from now."
        ),
        llm=get_llm(),
        tools=[ExtractFunctionContextTool(), ReadFileAtRefTool()],
        allow_delegation=False,
        verbose=False,
    )


def build_documentation_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Examine the changed functions and report documentation issues.\n\n"
            "Repository path: {repo_path}\n"
            "Head ref: {head_ref}\n\n"
            "Changed functions:\n{changed_functions_source}\n\n"
            "Look for:\n"
            "  - Public functions/classes without docstrings\n"
            "  - Docstrings that no longer match the implementation\n"
            "  - Parameters added without docstring updates\n"
            "  - PEP 8 naming-convention violations"
        ),
        expected_output=(
            "A JSON array of finding objects with keys: 'severity', 'file', "
            "'line', 'issue'. Return ONLY the JSON array."
        ),
        agent=agent,
    )
