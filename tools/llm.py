"""LLM client wrapper for CrewAI.

Centralised so every agent gets the same Ollama-backed model. Swap
providers (e.g. Groq) here in one place.
"""

from __future__ import annotations

from crewai import LLM

from core.config import CONFIG


def get_llm() -> LLM:
    """CrewAI LLM configured for local Ollama.

    Uses litellm's `ollama/<model>` provider against the native Ollama
    API on :11434.
    """
    return LLM(
        model=f"ollama/{CONFIG.ollama_model}",
        base_url=CONFIG.ollama_base_url,
        api_key=CONFIG.ollama_api_key,
        temperature=CONFIG.llm_temperature,
    )
