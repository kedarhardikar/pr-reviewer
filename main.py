"""Run the PR reviewer.

Usage:
    python main.py <repo_path_or_url>
    python main.py <repo_path_or_url> --base <ref> --head <ref>
    python main.py <repo_path_or_url> --skip-llm
    python main.py <repo_path_or_url> -o review.md

Prerequisites:
    pip install crewai crewai-tools GitPython pydantic click bandit detect-secrets
    ollama pull qwen2.5-coder:7b
    (ollama serve must be running on localhost:11434)
"""

from __future__ import annotations

import argparse
import sys

from crew import run_review
from reporter import compute_score, generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent PR reviewer.")
    parser.add_argument("repo", help="Path to local repo or git URL.")
    parser.add_argument("--base", default=None, help="Base ref (default: HEAD~1).")
    parser.add_argument("--head", default=None, help="Head ref (default: HEAD).")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-driven checks (AST checks only, no Ollama needed).",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Write report to file (default: stdout)."
    )
    args = parser.parse_args()

    result = run_review(
        args.repo, base=args.base, head=args.head, skip_llm=args.skip_llm
    )
    report = generate_report(result.state)
    score = compute_score(result.state)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output} (score: {score}/100)")
    else:
        print(report)

    sys.exit(0 if score >= 70 else 1)


if __name__ == "__main__":
    main()
