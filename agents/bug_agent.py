"""Bug Agent. Pure LLM-driven; tool access for pulling context."""

from __future__ import annotations

from crewai import Agent, Task

from tools.ast_tools import ExtractFunctionContextTool
from tools.git_tools import ReadFileAtRefTool
from tools.llm import get_llm


def build_bug_agent() -> Agent:
    return Agent(
        role="Bug Detection Specialist",
        goal=(
            "Identify logic errors, runtime exceptions, missing validation, "
            "off-by-one errors, swallowed exceptions, and dangerous "
            "assumptions in the diff."
        ),
        backstory=(
            "You are a senior engineer who has spent a decade debugging "
            "production incidents. You spot subtle bugs: an unguarded None, "
            "a loop bound off by one, an exception silently dropped. You "
            "flag only real issues — when in doubt, leave it out."
        ),
        llm=get_llm(),
        tools=[ExtractFunctionContextTool(), ReadFileAtRefTool()],
        allow_delegation=False,
        verbose=False,
    )


def build_bug_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Review the following diff for bugs. Focus only on lines that "
            "were added or modified.\n\n"
            "Repository path: {repo_path}\n"
            "Head ref: {head_ref}\n\n"
            "Diff:\n```diff\n{diff}\n```\n\n"
            "Changed functions:\n{changed_functions_source}\n\n"
            "Look for:\n"
            "  - None / null dereferences\n"
            "  - Missing input validation\n"
            "  - Off-by-one errors\n"
            "  - Unhandled or swallowed exceptions\n"
            "  - Incorrect boolean logic\n"
            "  - Resource leaks (unclosed files, connections)\n"
            "  - Race conditions and concurrency issues\n"
            "  - Dangerous assumptions about input shape\n\n"
            "Use `extract_function_context` for full function bodies, "
            "`read_file_at_ref` for surrounding context."
        ),
        expected_output=(
            "A JSON array of finding objects with keys: 'severity' "
            "(critical/high/medium/low), 'file', 'line' (int or null), "
            "'issue'. Return ONLY the JSON array, no prose, no fences. "
            "If no bugs found, return []."
        ),
        agent=agent,
    )
