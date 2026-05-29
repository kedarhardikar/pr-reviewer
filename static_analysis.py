"""Static-analysis tools.

These make the Security Agent reliable. LLMs miss obvious things like
hardcoded credentials and SQL injection patterns; Bandit and
detect-secrets catch those deterministically. The LLM's job is then to
triage results and catch the subtler issues these tools miss.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# --------------------------------------------------------------- Bandit


class BanditScanInput(BaseModel):
    source: str = Field(description="Python source code to scan.")
    file_hint: str = Field(
        default="snippet.py",
        description="Filename hint for the report (cosmetic).",
    )


class BanditScanTool(BaseTool):
    """Run Bandit against a Python source snippet."""

    name: str = "bandit_scan"
    description: str = (
        "Runs the Bandit security linter on Python source. Returns a JSON "
        "string with findings; each has 'severity', 'line_number', "
        "'issue_text', and 'test_id' (Bandit rule code)."
    )
    args_schema: type[BaseModel] = BanditScanInput

    def _run(self, source: str, file_hint: str = "snippet.py") -> str:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / file_hint
            p.write_text(source)
            try:
                proc = subprocess.run(
                    ["bandit", "-f", "json", "-q", str(p)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except FileNotFoundError:
                return json.dumps(
                    {"error": "Bandit not installed. `pip install bandit`."}
                )
            except subprocess.TimeoutExpired:
                return json.dumps({"error": "Bandit scan timed out."})

            # Bandit exits non-zero when it finds issues; that's not a failure.
            try:
                data = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError:
                return json.dumps(
                    {"error": "Bandit output was not valid JSON.", "stderr": proc.stderr}
                )

            results = []
            for r in data.get("results", []):
                results.append(
                    {
                        "severity": r.get("issue_severity", "").lower(),
                        "confidence": r.get("issue_confidence", "").lower(),
                        "line_number": r.get("line_number"),
                        "issue_text": r.get("issue_text"),
                        "test_id": r.get("test_id"),
                        "test_name": r.get("test_name"),
                    }
                )
            return json.dumps({"findings": results})


# --------------------------------------------------------------- detect-secrets


class SecretsScanInput(BaseModel):
    source: str = Field(description="Source code to scan for secrets.")
    file_hint: str = Field(
        default="snippet.py",
        description="Filename hint (some plugins are extension-aware).",
    )


class SecretsScanTool(BaseTool):
    """Run detect-secrets against a source snippet.

    Catches hardcoded API keys, AWS keys, base64 secrets, high-entropy
    strings, etc.
    """

    name: str = "secrets_scan"
    description: str = (
        "Scans source code for hardcoded secrets, API keys, and credentials. "
        "Returns a JSON string with findings; each has 'type' and 'line_number'."
    )
    args_schema: type[BaseModel] = SecretsScanInput

    def _run(self, source: str, file_hint: str = "snippet.py") -> str:
        try:
            from detect_secrets import SecretsCollection
            from detect_secrets.settings import default_settings
        except ImportError:
            return json.dumps(
                {"error": "detect-secrets not installed. `pip install detect-secrets`."}
            )

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / file_hint
            p.write_text(source)

            secrets = SecretsCollection()
            with default_settings():
                secrets.scan_file(str(p))

            findings = []
            for _file, secret in secrets:
                findings.append(
                    {"type": secret.type, "line_number": secret.line_number}
                )
            return json.dumps({"findings": findings})
