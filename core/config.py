"""Configuration. All values overridable via environment variables."""
from __future__ import annotations

import os
# Kill litellm prompt caching before it initialises — prevents
# cache_breakpoint injection that Groq rejects.
os.environ["LITELLM_CACHE"] = "False"
os.environ["LITELLM_LOCAL_CACHE"] = "False"

from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

import os


load_dotenv()


@dataclass(frozen=True)
class Config:
    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Ollama (kept as fallback)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "ollama-local")

    # Shared
    doc_coverage_warn_threshold: float = float(os.getenv("DOC_COVERAGE_WARN", "0.70"))
    workspace_dir: str = os.getenv("PR_REVIEWER_WORKSPACE", "/tmp/pr-reviewer")
    max_context_lines: int = int(os.getenv("PR_REVIEWER_MAX_CTX", "80"))
    llm_temperature: float = float(os.getenv("PR_REVIEWER_TEMP", "0.1"))


CONFIG = Config()