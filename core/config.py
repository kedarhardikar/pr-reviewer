"""Configuration. All values overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "ollama-local")
    doc_coverage_warn_threshold: float = float(os.getenv("DOC_COVERAGE_WARN", "0.70"))
    workspace_dir: str = os.getenv("PR_REVIEWER_WORKSPACE", os.path.join(os.path.expanduser("~"), ".pr-reviewer-workspace"))
    max_context_lines: int = int(os.getenv("PR_REVIEWER_MAX_CTX", "80"))
    llm_temperature: float = float(os.getenv("PR_REVIEWER_TEMP", "0.1"))


CONFIG = Config()
