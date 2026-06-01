"""Git operations: plain functions plus CrewAI tools.

The pipeline uses the plain functions (no LLM-tool indirection needed).
The CrewAI BaseTool wrappers are for agents that want to read files at
specific refs while reasoning.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import git
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from core.config import CONFIG


@dataclass
class FileChange:
    """Per-file summary of what the diff touched."""

    path: str
    changed_lines: set[int]  # 1-based, post-change line numbers
    is_new: bool
    is_deleted: bool


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


# --------------------------------------------------------------- plain functions


def clone_or_open(repo_url_or_path: str, workspace: str | None = None) -> git.Repo:
    """Open a local repo or clone a remote one into the workspace."""
    workspace = workspace or CONFIG.workspace_dir

    if os.path.isdir(repo_url_or_path) and os.path.isdir(
        os.path.join(repo_url_or_path, ".git")
    ):
        return git.Repo(repo_url_or_path)

    parsed = urlparse(repo_url_or_path)
    if not parsed.scheme and not parsed.netloc:
        raise ValueError(f"Not a valid repo path or URL: {repo_url_or_path!r}")

    os.makedirs(workspace, exist_ok=True)
    name = Path(parsed.path).stem or "repo"
    target = os.path.join(workspace, name)

    if os.path.isdir(target):
        repo = git.Repo(target)
        try:
            repo.remotes.origin.fetch()
        except git.GitCommandError:
            pass  # offline / detached; tolerate
        return repo

    return git.Repo.clone_from(repo_url_or_path, target)


def resolve_refs(
    repo: git.Repo, base: str | None, head: str | None
) -> tuple[str, str]:
    """Resolve base/head refs to commit SHAs. Defaults: HEAD~1..HEAD."""
    head_ref = head or "HEAD"
    head_sha = repo.commit(head_ref).hexsha

    if base is None:
        try:
            base_sha = repo.commit(f"{head_ref}~1").hexsha
        except git.BadName as e:
            raise ValueError(
                "No base ref given and HEAD has no parent — pass --base explicitly."
            ) from e
    else:
        base_sha = repo.commit(base).hexsha

    return base_sha, head_sha


def get_diff_text(repo: git.Repo, base: str, head: str) -> str:
    """Full unified diff between base and head."""
    return repo.git.diff(base, head, "--unified=3")


def get_changed_files(
    repo: git.Repo,
    base: str,
    head: str,
    extensions: tuple[str, ...] = (".py",),
) -> list[FileChange]:
    """List files changed between base and head, filtered by extension."""
    diff_index = repo.commit(base).diff(head, create_patch=True)
    results: list[FileChange] = []

    for d in diff_index:
        path = d.b_path or d.a_path
        if path is None:
            continue
        if extensions and not path.endswith(extensions):
            continue

        results.append(
            FileChange(
                path=path,
                changed_lines=_parse_added_lines(d.diff),
                is_new=d.new_file,
                is_deleted=d.deleted_file,
            )
        )
    return results


def _parse_added_lines(patch_bytes: bytes | str) -> set[int]:
    """Return new-side line numbers that were added or modified.

    Walks hunk-by-hunk: a hunk header gives the starting line on the new
    side; we advance for context and added lines, skip deletions.
    """
    if isinstance(patch_bytes, bytes):
        patch = patch_bytes.decode("utf-8", errors="replace")
    else:
        patch = patch_bytes

    added: set[int] = set()
    new_line: int | None = None

    for line in patch.splitlines():
        m = _HUNK_RE.match(line)
        if m:
            new_line = int(m.group(1))
            continue
        if new_line is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            added.add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass  # deletion — does not advance new-side counter
        elif line.startswith("\\"):
            pass  # "\ No newline at end of file"
        else:
            new_line += 1

    return added


def read_file_at(repo: git.Repo, ref: str, path: str) -> str | None:
    """Return file contents at a given ref, or None if missing there."""
    try:
        return repo.git.show(f"{ref}:{path}")
    except git.GitCommandError:
        return None


def cleanup_workspace() -> None:
    """Wipe the cached clones."""
    if os.path.isdir(CONFIG.workspace_dir):
        shutil.rmtree(CONFIG.workspace_dir)


# --------------------------------------------------------------- CrewAI tools


class ReadFileInput(BaseModel):
    repo_path: str = Field(description="Absolute path to the local git repo.")
    ref: str = Field(description="Git ref (commit SHA, branch, or tag).")
    file_path: str = Field(description="Repo-relative file path.")


class ReadFileAtRefTool(BaseTool):
    """Read a file's contents at a specific git ref."""

    name: str = "read_file_at_ref"
    description: str = (
        "Reads the contents of a file at a specific git ref. "
        "Returns the file source as a string, or 'FILE_NOT_FOUND' if missing."
    )
    args_schema: type[BaseModel] = ReadFileInput

    def _run(self, repo_path: str, ref: str, file_path: str) -> str:
        repo = git.Repo(repo_path)
        content = read_file_at(repo, ref, file_path)
        return content if content is not None else "FILE_NOT_FOUND"


class GetDiffInput(BaseModel):
    repo_path: str = Field(description="Absolute path to the local git repo.")
    base: str = Field(description="Base ref.")
    head: str = Field(description="Head ref.")


class GetDiffTool(BaseTool):
    """Return the unified diff between two refs."""

    name: str = "get_diff"
    description: str = "Returns the unified diff between two git refs as a string."
    args_schema: type[BaseModel] = GetDiffInput

    def _run(self, repo_path: str, base: str, head: str) -> str:
        repo = git.Repo(repo_path)
        return get_diff_text(repo, base, head)
