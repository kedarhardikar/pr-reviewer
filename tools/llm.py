"""LLM client wrapper for CrewAI.

Uses Groq's OpenAI-compatible endpoint via CrewAI's native openai provider.
This avoids litellm entirely — no cache_breakpoint injection, no provider
routing issues.
"""

from __future__ import annotations

from crewai import LLM
from core.config import CONFIG

# Groq's OpenAI-compatible base URL
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_llm() -> LLM:
    if CONFIG.groq_api_key:
        print(f"[LLM] Using Groq (OpenAI-compat) — model: {CONFIG.groq_model}")
        return LLM(
            model=f"openai/{CONFIG.groq_model}",
            base_url=_GROQ_BASE_URL,
            api_key=CONFIG.groq_api_key,
            temperature=CONFIG.llm_temperature,
        )

    print(f"[LLM] Using local Ollama — model: {CONFIG.ollama_model}")
    return LLM(
        model=f"ollama/{CONFIG.ollama_model}",
        base_url=CONFIG.ollama_base_url,
        api_key=CONFIG.ollama_api_key,
        temperature=CONFIG.llm_temperature,
    )