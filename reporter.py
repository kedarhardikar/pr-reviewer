"""Markdown report generator + 0-100 score."""

from __future__ import annotations

from collections import Counter, defaultdict

from state import ReviewState


_SEVERITY_PENALTY = {
    "critical": 20,
    "high": 10,
    "medium": 4,
    "low": 1,
    "info": 0,
}

_CATEGORY_TITLES = {
    "bug": "Bug Findings",
    "security": "Security Findings",
    "performance": "Performance Findings",
    "documentation": "Documentation Findings",
}

_CATEGORY_KEYS = {
    "bug": "bug_findings",
    "security": "security_findings",
    "performance": "performance_findings",
    "documentation": "documentation_findings",
}


def compute_score(state: ReviewState) -> int:
    """0-100, lower for more/worse findings."""
    score = 100
    for key in _CATEGORY_KEYS.values():
        for f in state.get(key, []) or []:
            if f.get("file") == "<summary>":
                continue
            score -= _SEVERITY_PENALTY.get(f.get("severity", "info"), 0)
    return max(0, score)


def generate_report(state: ReviewState) -> str:
    score = compute_score(state)
    lines: list[str] = []
    lines.append("# Pull Request Review")
    lines.append("")
    lines.append(f"**Score:** {score}/100")
    lines.append("")
    lines.extend(_executive_summary(state, score))
    lines.append("")

    critical = _collect_critical(state)
    if critical:
        lines.append("## Critical Issues")
        lines.append("")
        for f in critical:
            lines.append(_format_finding(f))
        lines.append("")

    for category, key in _CATEGORY_KEYS.items():
        findings = state.get(key) or []
        if not findings:
            continue
        lines.append(f"## {_CATEGORY_TITLES[category]}")
        lines.append("")
        summaries = [f for f in findings if f.get("file") == "<summary>"]
        regular = [f for f in findings if f.get("file") != "<summary>"]
        for s in summaries:
            lines.append(f"_{s['issue']}_")
            lines.append("")
        if regular:
            for f in _sorted_by_severity(regular):
                lines.append(_format_finding(f))
            lines.append("")

    fixes = state.get("fixes") or []
    if fixes:
        lines.append("## Suggested Fixes")
        lines.append("")
        for fix in fixes:
            lines.append(
                f"**{fix.get('file', '?')}"
                f"{':' + str(fix.get('line')) if fix.get('line') else ''}** "
                f"— {fix.get('original_issue', '')}"
            )
            if fix.get("explanation"):
                lines.append("")
                lines.append(fix["explanation"])
            snippet = fix.get("suggested_code") or fix.get("suggestion", "")
            if snippet:
                lines.append("")
                lines.append("```python")
                lines.append(snippet)
                lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(state: ReviewState, score: int) -> list[str]:
    counter: Counter[str] = Counter()
    by_severity: dict[str, int] = defaultdict(int)
    for key in _CATEGORY_KEYS.values():
        for f in state.get(key, []) or []:
            if f.get("file") == "<summary>":
                continue
            counter[key] += 1
            by_severity[f.get("severity", "info")] += 1

    out = ["## Executive Summary", ""]
    out.append(f"- Files changed: {len(state.get('changed_files', []))}")
    total_findings = sum(counter.values())
    out.append(f"- Total findings: {total_findings}")
    if total_findings:
        bits = []
        for sev in ("critical", "high", "medium", "low", "info"):
            n = by_severity.get(sev, 0)
            if n:
                bits.append(f"{n} {sev}")
        out.append(f"- By severity: {', '.join(bits)}")
    if score == 100:
        out.append("- No issues detected.")
    elif score >= 80:
        out.append("- Minor issues only.")
    elif score >= 50:
        out.append("- Several issues worth addressing before merge.")
    else:
        out.append("- Significant issues — address before merge.")
    return out


def _collect_critical(state: ReviewState) -> list[dict]:
    out: list[dict] = []
    for key in _CATEGORY_KEYS.values():
        for f in state.get(key, []) or []:
            if (
                f.get("severity") in ("critical", "high")
                and f.get("file") != "<summary>"
            ):
                out.append(f)
    return _sorted_by_severity(out)


_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _sorted_by_severity(findings: list[dict]) -> list[dict]:
    return sorted(
        findings,
        key=lambda f: (
            _SEVERITY_ORDER.get(f.get("severity", "info"), 5),
            f.get("file", ""),
            f.get("line") or 0,
        ),
    )


def _format_finding(f: dict) -> str:
    sev = f.get("severity", "info").upper()
    file = f.get("file", "?")
    line = f.get("line")
    loc = f"{file}:{line}" if line else file
    return f"- **[{sev}]** `{loc}` — {f.get('issue', '')}"
