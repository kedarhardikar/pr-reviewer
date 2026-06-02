"""Review logger — writes a structured .log file for every run.

Records:
  - Which repo and commits are being compared
  - The full git diff
  - Every agent's inputs and raw outputs
  - Static check results
  - Final findings summary
"""

from __future__ import annotations

import datetime
import os
import textwrap
from pathlib import Path


_LOG_DIR = os.getenv("PR_REVIEWER_LOG_DIR", "logs")
_SEP = "=" * 80
_SUBSEP = "-" * 60


class ReviewLogger:
    """One logger per review run. Call close() or use as context manager."""

    def __init__(self, repo_url: str, base_ref: str, head_ref: str) -> None:
        Path(_LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slug(repo_url)
        self._path = os.path.join(_LOG_DIR, f"review_{slug}_{ts}.log")
        self._f = open(self._path, "w", encoding="utf-8")
        self._write_header(repo_url, base_ref, head_ref, ts)

    # ------------------------------------------------------------------ public

    @property
    def path(self) -> str:
        return self._path

    def log_diff(self, diff: str) -> None:
        self._section("GIT DIFF")
        self._f.write(diff if diff.strip() else "(empty diff)\n")
        self._f.write("\n")

    def log_static_checks(self, findings: list[dict]) -> None:
        self._section("STATIC CHECKS")
        if not findings:
            self._f.write("No findings.\n")
        for f in findings:
            self._f.write(
                f"  [{f.get('severity','?').upper()}] {f.get('file','?')}:"
                f"{f.get('line','?')} — {f.get('issue','')}\n"
            )
        self._f.write("\n")

    def log_agent_input(self, agent_name: str, inputs: dict) -> None:
        self._subsection(f"AGENT INPUT — {agent_name}")
        for k, v in inputs.items():
            val = str(v)
            if len(val) > 2000:
                val = val[:2000] + "\n... [truncated]"
            self._f.write(f"  {k}:\n")
            for line in val.splitlines():
                self._f.write(f"    {line}\n")
        self._f.write("\n")

    def log_agent_output(self, agent_name: str, raw_output: str) -> None:
        self._subsection(f"AGENT OUTPUT — {agent_name}")
        out = raw_output.strip() if raw_output else "(no output)"
        if len(out) > 4000:
            out = out[:4000] + "\n... [truncated]"
        for line in out.splitlines():
            self._f.write(f"  {line}\n")
        self._f.write("\n")

    def log_findings_summary(self, state: dict) -> None:
        self._section("FINDINGS SUMMARY")
        categories = {
            "bug": "Bug",
            "security": "Security",
            "performance": "Performance",
            "documentation": "Documentation",
        }
        keys = {
            "bug": "bug_findings",
            "security": "security_findings",
            "performance": "performance_findings",
            "documentation": "documentation_findings",
        }
        total = 0
        for cat, label in categories.items():
            findings = [
                f for f in (state.get(keys[cat]) or [])
                if f.get("file") != "<summary>"
            ]
            if not findings:
                continue
            self._f.write(f"\n  {label} ({len(findings)}):\n")
            for f in findings:
                self._f.write(
                    f"    [{f.get('severity','?').upper()}] "
                    f"{f.get('file','?')}:{f.get('line','?')} — "
                    f"{f.get('issue','')[:120]}\n"
                )
            total += len(findings)
        self._f.write(f"\n  Total: {total} finding(s)\n\n")

    def log_event(self, event: str, detail: str = "") -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._f.write(f"[{ts}] {event}: {detail}\n")
        self._f.flush()

    def close(self) -> None:
        self._f.write(f"\n{_SEP}\n")
        self._f.write(f"Log written to: {self._path}\n")
        self._f.close()

    def __enter__(self) -> "ReviewLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ----------------------------------------------------------------- private

    def _write_header(
        self, repo_url: str, base_ref: str, head_ref: str, ts: str
    ) -> None:
        self._f.write(f"{_SEP}\n")
        self._f.write("PR REVIEWER — RUN LOG\n")
        self._f.write(f"{_SEP}\n")
        self._f.write(f"Timestamp : {ts}\n")
        self._f.write(f"Repo      : {repo_url}\n")
        self._f.write(f"Base ref  : {base_ref}\n")
        self._f.write(f"Head ref  : {head_ref}\n")
        self._f.write(f"Log file  : {self._path}\n")
        self._f.write(f"{_SEP}\n\n")
        self._f.flush()

    def _section(self, title: str) -> None:
        self._f.write(f"\n{_SEP}\n{title}\n{_SEP}\n")

    def _subsection(self, title: str) -> None:
        self._f.write(f"\n{_SUBSEP}\n{title}\n{_SUBSEP}\n")


def _slug(url: str) -> str:
    """Turn a URL or path into a safe filename fragment."""
    import re
    s = re.sub(r"[^\w\-]", "_", url.split("/")[-1] or "repo")
    return s[:30]
