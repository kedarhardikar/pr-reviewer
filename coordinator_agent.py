"""Coordinator Agent.

Routes work to the specialists and produces a short framing note before
they run. Most prep work (git, AST extraction) happens in crew.py
*before* the Crew starts, so the Coordinator only reasons about routing.
"""

from __future__ import annotations

from crewai import Agent, Task

from llm import get_llm


def build_coordinator_agent() -> Agent:
    return Agent(
        role="Code Review Coordinator",
        goal=(
            "Orchestrate a thorough pull request review by ensuring each "
            "specialist focuses on the parts of the diff most relevant to "
            "their expertise."
        ),
        backstory=(
            "You are a pragmatic engineering lead who has led code review "
            "for a busy team for years. You know when to defer to specialists "
            "and never duplicate their findings."
        ),
        llm=get_llm(),
        allow_delegation=True,
        verbose=False,
    )


def build_coordinator_task(agent: Agent) -> Task:
    """The Coordinator's framing task. Inputs interpolated via Crew.kickoff."""
    return Task(
        description=(
            "You are reviewing the following pull request.\n\n"
            "Repository: {repo_path}\n"
            "Base ref: {base_ref}\n"
            "Head ref: {head_ref}\n\n"
            "Changed files:\n{changed_files}\n\n"
            "Specialist agents (Bug, Security, Performance, Documentation) "
            "will examine the diff in turn. Your job is to flag which areas "
            "warrant the most attention. Do not duplicate their work."
        ),
        expected_output=(
            "A short coordination note (3-5 sentences) describing which "
            "areas of the diff most warrant attention from each specialist."
        ),
        agent=agent,
    )
