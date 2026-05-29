"""Shared state structures passed between agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict

Severity = Literal["critical", "high", "medium", "low", "info"]
Category = Literal["bug", "security", "performance", "documentation"]


@dataclass
class Finding:
    """A single issue surfaced by an agent.

    Findings are the universal currency of the system: every agent emits
    them, the report generator consumes them.
    """

    category: Category
    severity: Severity
    file: str
    issue: str
    line: int | None = None
    end_line: int | None = None
    suggestion: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fix:
    """A code suggestion produced by the Fix Agent for a specific finding."""

    file: str
    line: int | None
    original_issue: str
    explanation: str
    suggested_code: str

    def to_dict(self) -> dict:
        return asdict(self)


class ReviewState(TypedDict, total=False):
    """The blackboard passed through the pipeline."""

    repo_path: str
    base_ref: str | None
    head_ref: str | None
    diff: str
    changed_files: list[str]

    bug_findings: list[dict]
    security_findings: list[dict]
    performance_findings: list[dict]
    documentation_findings: list[dict]

    fixes: list[dict]
    final_report: str
    score: int
