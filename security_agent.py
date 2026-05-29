"""Security Agent — hybrid static + LLM.

Strategy:
  1. Bandit and detect-secrets run as LLM-callable tools. They reliably
     catch the obvious: SQL injection via string formatting, hardcoded
     keys, unsafe subprocess, pickle.load on untrusted input, etc.
  2. The LLM triages those results (dedup, suppress false positives) AND
     looks for things static tools miss: authorization bypasses, IDOR,
     missing access control, unsafe redirects.

Static alone produces false positives at scale; LLM alone misses easy
stuff. Combining them gives reach and precision.
"""

from __future__ import annotations

from crewai import Agent, Task

from ast_tools import ExtractFunctionContextTool
from git_tools import ReadFileAtRefTool
from llm import get_llm
from static_analysis import BanditScanTool, SecretsScanTool


def build_security_agent() -> Agent:
    return Agent(
        role="Security Reviewer",
        goal=(
            "Identify security vulnerabilities: injection, secret exposure, "
            "unsafe deserialization, broken access control, and similar."
        ),
        backstory=(
            "You are a security engineer for an internet-facing application. "
            "You use Bandit and detect-secrets as a first pass and then apply "
            "judgement for things they can't see — authorization bugs, unsafe "
            "trust boundaries. You suppress false positives aggressively; a "
            "noisy review is an ignored review."
        ),
        llm=get_llm(),
        tools=[
            BanditScanTool(),
            SecretsScanTool(),
            ExtractFunctionContextTool(),
            ReadFileAtRefTool(),
        ],
        allow_delegation=False,
        verbose=False,
    )


def build_security_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Review the following diff for security issues.\n\n"
            "Repository path: {repo_path}\n"
            "Head ref: {head_ref}\n\n"
            "Diff:\n```diff\n{diff}\n```\n\n"
            "Changed functions:\n{changed_functions_source}\n\n"
            "Workflow:\n"
            "  1. Run `bandit_scan` and `secrets_scan` on each changed "
            "     function. Collect findings.\n"
            "  2. For each static finding, decide if it's real or a false "
            "     positive given the context.\n"
            "  3. Also look for issues static tools miss:\n"
            "     - Authorization bypasses (new endpoint without auth check)\n"
            "     - IDOR (insecure direct object references)\n"
            "     - SSRF / unsafe URL fetches\n"
            "     - Path traversal in user-derived file paths\n"
            "     - Unsafe redirects\n"
            "     - Logging of sensitive data\n\n"
            "Do NOT re-report stylistic Bandit warnings without real impact."
        ),
        expected_output=(
            "A JSON array of finding objects with keys: 'severity', "
            "'file', 'line', 'issue'. Return ONLY the JSON array. "
            "If no issues found, return []."
        ),
        agent=agent,
    )
