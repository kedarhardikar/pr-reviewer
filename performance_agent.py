"""Performance Agent. Pure LLM-driven."""

from __future__ import annotations

from crewai import Agent, Task

from ast_tools import ExtractFunctionContextTool
from git_tools import ReadFileAtRefTool
from llm import get_llm


def build_performance_agent() -> Agent:
    return Agent(
        role="Performance Reviewer",
        goal=(
            "Identify performance issues introduced by the diff: N+1 queries, "
            "redundant work inside loops, inefficient data structures."
        ),
        backstory=(
            "You are a backend engineer who has spent years profiling and "
            "fixing slow code. You spot the database call inside a `for` loop, "
            "the regex compilation that should be hoisted. You only flag "
            "issues likely to matter in realistic workloads — micro-optimisations "
            "that won't show up in a profile aren't worth reporting."
        ),
        llm=get_llm(),
        tools=[ExtractFunctionContextTool(), ReadFileAtRefTool()],
        allow_delegation=False,
        verbose=False,
    )


def build_performance_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Review the following diff for performance issues.\n\n"
            "Repository path: {repo_path}\n"
            "Head ref: {head_ref}\n\n"
            "Diff:\n```diff\n{diff}\n```\n\n"
            "Changed functions:\n{changed_functions_source}\n\n"
            "Look for:\n"
            "  - N+1 database queries (query inside loop over previous results)\n"
            "  - Repeated DB/API calls that could be batched\n"
            "  - Expensive operations inside loops that could be hoisted "
            "    (regex compilation, file opens, network calls)\n"
            "  - Quadratic algorithms where linear would do\n"
            "  - Loading entire datasets when streaming would suffice\n"
            "  - Redundant recomputation of the same value\n\n"
            "Skip micro-optimisations that wouldn't show up in a profile."
        ),
        expected_output=(
            "A JSON array of finding objects with keys: 'severity' "
            "(high/medium/low), 'file', 'line', 'issue'. Return ONLY the "
            "JSON array. If no issues found, return []."
        ),
        agent=agent,
    )
