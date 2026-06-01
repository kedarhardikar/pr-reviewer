"""Fix Agent.

Runs after detection. Receives the union of findings and produces
concrete code suggestions.
"""

from __future__ import annotations

from crewai import Agent, Task
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.ast_tools import ExtractFunctionContextTool
from tools.git_tools import ReadFileAtRefTool
from tools.llm import get_llm


def build_fix_agent() -> Agent:
    return Agent(
        role="Code Fix Author",
        goal=(
            "Generate concrete, copy-pasteable code corrections for the "
            "findings produced by the other specialists."
        ),
        backstory=(
            "You are a senior engineer who writes the fix when someone else "
            "writes the bug report. You produce minimal, targeted diffs — "
            "never a rewrite when one line will do. Your fixes compile, "
            "follow existing style, and don't introduce new issues."
        ),
        llm=get_llm(),
        tools=[ExtractFunctionContextTool(), ReadFileAtRefTool()],
        allow_delegation=False,
        verbose=False,
    )


def build_fix_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Generate code fixes for the following findings.\n\n"
            "Repository path: {repo_path}\n"
            "Head ref: {head_ref}\n\n"
            "Findings:\n{all_findings_json}\n\n"
            "Changed functions:\n{changed_functions_source}\n\n"
            "For each finding with a clear fix:\n"
            "  - Produce a minimal corrected snippet (just the changing "
            "    lines plus a little context).\n"
            "  - Use `extract_function_context` if you need to see more.\n"
            "  - Skip findings where no clear fix exists (e.g. 'consider "
            "    refactoring').\n"
            "  - Do not invent issues that weren't in the input."
        ),
        expected_output=(
            "A JSON array of fix objects with keys: 'file', 'line' (int or "
            "null), 'original_issue' (copy from finding), 'explanation' "
            "(one sentence), 'suggested_code' (the corrected snippet). "
            "Return ONLY the JSON array. If no fixes warranted, return []."
        ),
        agent=agent,
    )
