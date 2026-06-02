"""Synthesis Agent.

Runs last. Receives all findings from every detection agent and produces:
  - De-duplicated, cross-referenced findings
  - A prioritised action list (what to fix first)
  - A short narrative paragraph summarising the overall PR health
"""

from __future__ import annotations

import json

from crewai import Agent, Task

import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.llm import get_llm


def build_synthesis_agent() -> Agent:
    return Agent(
        role="Chief Code Reviewer",
        goal=(
            "Read every finding from the Bug, Security, Performance, and "
            "Documentation agents, remove duplicates, cross-reference related "
            "issues, and produce a single prioritised action list with a "
            "narrative summary a developer can act on immediately."
        ),
        backstory=(
            "You are the engineering lead who signs off on PRs. You have seen "
            "the specialist reports and now you must consolidate them into a "
            "clear, ranked list of what matters most. You call out when two "
            "agents flagged the same root cause, and you elevate severity when "
            "a bug also has a security dimension."
        ),
        llm=get_llm(),
        allow_delegation=False,
        verbose=False,
    )


def build_synthesis_task(agent: Agent) -> Task:
    return Task(
        description=(
            "You have received the following findings from specialist agents.\n\n"
            "All findings (JSON):\n{all_findings_json}\n\n"
            "Your job:\n"
            "1. Write a 3-5 sentence narrative paragraph summarising the overall "
            "   code quality and the single most important thing to fix.\n"
            "2. Produce a prioritised action list ordered: critical → high → medium → low. "
            "   Within the same severity, order by impact (bugs before docs).\n\n"
            "STRICT RULES — breaking these makes the output useless:\n"
            "  - NEVER change a finding's severity. Copy the 'severity' field exactly "
            "    as it appears in the input JSON.\n"
            "  - NEVER invent new findings not present in the input.\n"
            "  - NEVER reference files not in the input findings.\n"
            "  - Each finding in 'prioritised_findings' must copy 'file', 'line', "
            "    'severity', 'category', 'issue' verbatim from the input.\n"
            "  - Add only one new field: 'action' (one-sentence fix instruction).\n\n"
            "Return ONLY a JSON object with two keys:\n"
            "  'narrative': string\n"
            "  'prioritised_findings': array of finding objects with keys "
            "  'rank' (int), 'severity', 'file', 'line', 'category', 'issue', 'action'.\n"
            "Return ONLY the JSON. No prose outside it."
        ),
        expected_output=(
            "A JSON object with 'narrative' (string) and 'prioritised_findings' "
            "(array of ranked finding objects)."
        ),
        agent=agent,
    )
