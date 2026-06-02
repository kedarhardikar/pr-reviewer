"""Python AST inspection.

Deterministic checks (no LLM):
  - extract_definitions: every function/class with docstring, params, span
  - functions_touching_lines: filter to those overlapping changed lines
  - check_naming: PEP 8 naming violations
  - check_param_documentation: param coverage in docstrings
  - documentation_coverage: documented / total ratio

Plus ExtractFunctionContextTool for agents that need to pull a function's
source while reasoning.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import git
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.git_tools import read_file_at


_SNAKE_CASE_RE = re.compile(r"^_?_?[a-z][a-z0-9_]*$")
_PASCAL_CASE_RE = re.compile(r"^_?[A-Z][a-zA-Z0-9]*$")


@dataclass
class Definition:
    kind: str  # "function" | "method" | "class"
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    docstring: str | None
    params: list[str]
    returns_something: bool
    is_public: bool
    source: str

    @property
    def has_docstring(self) -> bool:
        return bool(self.docstring and self.docstring.strip())


def extract_definitions(source: str, filename: str = "<source>") -> list[Definition]:
    """Walk the AST and return every function, method, and class."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    out: list[Definition] = []

    def _walk(node: ast.AST, qual_prefix: str, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [
                    a.arg for a in child.args.args if a.arg not in ("self", "cls")
                ]
                qname = f"{qual_prefix}{child.name}" if qual_prefix else child.name
                out.append(
                    Definition(
                        kind="method" if in_class else "function",
                        name=child.name,
                        qualified_name=qname,
                        start_line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                        docstring=ast.get_docstring(child),
                        params=params,
                        returns_something=_function_returns_value(child),
                        is_public=not child.name.startswith("_"),
                        source=_slice_source(source_lines, child.lineno, child.end_lineno),
                    )
                )
                _walk(child, qual_prefix=f"{qname}.", in_class=False)

            elif isinstance(child, ast.ClassDef):
                qname = f"{qual_prefix}{child.name}" if qual_prefix else child.name
                out.append(
                    Definition(
                        kind="class",
                        name=child.name,
                        qualified_name=qname,
                        start_line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                        docstring=ast.get_docstring(child),
                        params=[],
                        returns_something=False,
                        is_public=not child.name.startswith("_"),
                        source=_slice_source(source_lines, child.lineno, child.end_lineno),
                    )
                )
                _walk(child, qual_prefix=f"{qname}.", in_class=True)
            else:
                _walk(child, qual_prefix=qual_prefix, in_class=in_class)

    _walk(tree, qual_prefix="", in_class=False)
    return out


def _function_returns_value(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function has at least one `return <expr>`.

    Returns inside nested functions are excluded.
    """
    nested_func_ids = {
        id(n)
        for n in ast.walk(func)
        if n is not func and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def _scan(node: ast.AST) -> bool:
        for child in ast.iter_child_nodes(node):
            if id(child) in nested_func_ids:
                continue
            if isinstance(child, ast.Return) and child.value is not None:
                return True
            if _scan(child):
                return True
        return False

    return _scan(func)


def _slice_source(lines: list[str], start: int, end: int | None) -> str:
    if end is None:
        end = start
    return "\n".join(lines[start - 1 : end])


def functions_touching_lines(
    defs: list[Definition], changed_lines: set[int]
) -> list[Definition]:
    """Filter defs to those whose line range overlaps any changed line."""
    if not changed_lines:
        return []
    return [
        d
        for d in defs
        if any(d.start_line <= ln <= d.end_line for ln in changed_lines)
    ]


def documentation_coverage(defs: list[Definition]) -> tuple[int, int, float]:
    """Coverage over *public* functions and methods. Classes excluded."""
    public_callables = [
        d for d in defs if d.is_public and d.kind in ("function", "method")
    ]
    total = len(public_callables)
    documented = sum(1 for d in public_callables if d.has_docstring)
    pct = (documented / total) if total else 1.0
    return documented, total, pct


def check_naming(defs: list[Definition]) -> list[tuple[Definition, str]]:
    """Return (definition, reason) for each naming violation."""
    issues: list[tuple[Definition, str]] = []
    for d in defs:
        if d.kind == "class":
            if not _PASCAL_CASE_RE.match(d.name):
                issues.append((d, f"Class '{d.name}' should use PascalCase"))
        else:
            if not _SNAKE_CASE_RE.match(d.name):
                issues.append(
                    (d, f"{d.kind.capitalize()} '{d.name}' should use snake_case")
                )
            elif d.is_public and len(d.name) == 1:
                issues.append(
                    (
                        d,
                        f"{d.kind.capitalize()} '{d.name}' has a "
                        "non-descriptive single-letter name",
                    )
                )

    for d in defs:
        if d.kind in ("function", "method") and d.is_public:
            # x/y/z get a pass for math-ish APIs.
            bad_params = [
                p for p in d.params if len(p) == 1 and p not in ("x", "y", "z")
            ]
            if bad_params:
                issues.append(
                    (d, f"Parameters {bad_params} have non-descriptive names")
                )
    return issues


def check_param_documentation(d: Definition) -> list[str]:
    """Return param names that look undocumented.

    Heuristic: every parameter should appear as a word in the docstring.
    Works across Google/NumPy/Sphinx styles via substring matching.
    """
    if not d.has_docstring or not d.params:
        return []
    doc = d.docstring or ""
    missing = []
    for p in d.params:
        if not re.search(rf"\b{re.escape(p)}\b", doc):
            missing.append(p)
    return missing


# --------------------------------------------------------------- static bug checks

import builtins as _builtins_module
_PYTHON_BUILTINS = frozenset(dir(_builtins_module))

# Subset of builtins commonly shadowed by accident — these are the ones we flag.
_RISKY_SHADOW_NAMES = frozenset({
    "format", "list", "dict", "set", "tuple", "map", "filter",
    "zip", "open", "next", "iter", "input", "print", "type", "id",
    "sum", "min", "max", "len", "range", "sorted", "reversed",
    "enumerate", "any", "all", "hash", "round", "abs",
})

_DB_CALL_RE = re.compile(
    r"\b(sqlite3\.connect|conn\.execute|cursor\.execute|conn\.commit)\s*\("
)


def run_static_bug_checks(source: str, filename: str) -> list[dict]:
    """Deterministic AST + regex checks that catch common bugs.

    Returns a list of finding dicts compatible with the pipeline's
    bug_findings format.
    """
    findings: list[dict] = []
    findings.extend(_check_duplicate_imports(source, filename))
    findings.extend(_check_shadowed_builtins(source, filename))
    findings.extend(_check_db_without_try(source, filename))
    findings.extend(_check_missing_return_documented(source, filename))
    return findings


def _check_duplicate_imports(source: str, filename: str) -> list[dict]:
    """Flag modules imported more than once at module level."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    seen: dict[str, int] = {}  # module_name → first line
    findings: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name in seen:
                    findings.append({
                        "category": "bug",
                        "severity": "low",
                        "file": filename,
                        "line": node.lineno,
                        "issue": (
                            f"Module '{name}' is imported twice "
                            f"(first at line {seen[name]}, again at line {node.lineno}). "
                            "Remove the duplicate."
                        ),
                    })
                else:
                    seen[name] = node.lineno

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                key = f"{mod}.{alias.name}"
                if key in seen:
                    findings.append({
                        "category": "bug",
                        "severity": "low",
                        "file": filename,
                        "line": node.lineno,
                        "issue": (
                            f"'{alias.name}' from '{mod}' is imported twice "
                            f"(first at line {seen[key]}, again at line {node.lineno}). "
                            "Remove the duplicate."
                        ),
                    })
                else:
                    seen[key] = node.lineno

    return findings


def _check_shadowed_builtins(source: str, filename: str) -> list[dict]:
    """Flag assignments that shadow Python builtins (e.g. format = ...)."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    findings: list[dict] = []

    for node in ast.walk(tree):
        # Regular assignment: x = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _RISKY_SHADOW_NAMES:
                    findings.append({
                        "category": "bug",
                        "severity": "medium",
                        "file": filename,
                        "line": node.lineno,
                        "issue": (
                            f"Variable '{target.id}' shadows the Python builtin "
                            f"'{target.id}'. Rename it to avoid subtle bugs."
                        ),
                    })
        # Function argument named after a builtin
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg in _RISKY_SHADOW_NAMES:
                    findings.append({
                        "category": "bug",
                        "severity": "low",
                        "file": filename,
                        "line": node.lineno,
                        "issue": (
                            f"Parameter '{arg.arg}' in '{node.name}' shadows "
                            f"the Python builtin '{arg.arg}'. Rename it."
                        ),
                    })

    return findings


def _check_db_without_try(source: str, filename: str) -> list[dict]:
    """Flag sqlite3 operations that aren't inside a try/except block."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    findings: list[dict] = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Collect line numbers of all nodes inside try blocks within this func.
        try_covered: set[int] = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        try_covered.add(child.lineno)

        # Find DB calls outside try blocks.
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            line = getattr(node, "lineno", None)
            if line is None or line in try_covered:
                continue

            # sqlite3.connect(...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sqlite3"
            ):
                findings.append({
                    "category": "bug",
                    "severity": "medium",
                    "file": filename,
                    "line": line,
                    "issue": (
                        f"sqlite3.connect() in '{func.name}' is not inside a "
                        "try/except block. A DB error will crash the caller. "
                        "Wrap in try/except and ensure conn.close() runs in finally."
                    ),
                })

            # cursor.execute(...) or conn.execute(...)
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("cursor", "conn")
            ):
                findings.append({
                    "category": "bug",
                    "severity": "low",
                    "file": filename,
                    "line": line,
                    "issue": (
                        f"DB execute() call in '{func.name}' is not inside a "
                        "try/except block. Failed queries will raise uncaught exceptions."
                    ),
                })

    # Deduplicate — keep only first per function+issue-type.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for f in findings:
        key = (f["file"], f["line"], f["issue"][:40])
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped


_ACTION_VERBS = re.compile(
    r"\b(save|insert|store|write|update|delete|remove|clear|send|push|commit)\b",
    re.IGNORECASE,
)
_RETURN_WORDS = re.compile(r"\b(return|returns|→|->|:returns?:)\b", re.IGNORECASE)


def _returns_bool(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the function has at least one `return True` or `return False`."""
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, bool):
                return True
    return False


def _check_missing_return_documented(source: str, filename: str) -> list[dict]:
    """Flag public functions whose return value isn't documented.

    Two sub-cases:
    1. Generic: returns something but docstring never mentions it.
    2. Silent-failure: returns True/False AND docstring only describes a
       write/save/delete action — callers won't know to check the result.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    findings: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        docstring = ast.get_docstring(node)
        if not docstring:
            continue  # Missing docstring caught elsewhere.
        if not _function_returns_value(node):
            continue

        has_return_mention = bool(_RETURN_WORDS.search(docstring))

        if _returns_bool(node) and _ACTION_VERBS.search(docstring) and not has_return_mention:
            # Silent-failure pattern: function describes a write op but returns
            # True/False to signal success — callers won't know to check it.
            findings.append({
                "category": "bug",
                "severity": "medium",
                "file": filename,
                "line": node.lineno,
                "issue": (
                    f"'{node.name}' returns True/False to signal success or failure "
                    "but the docstring only describes the write operation. Callers have "
                    "no indication they must check the return value. Document the return "
                    "value and consider raising an exception instead of silent failure."
                ),
            })
        elif not has_return_mention:
            findings.append({
                "category": "documentation",
                "severity": "low",
                "file": filename,
                "line": node.lineno,
                "issue": (
                    f"'{node.name}' returns a value but its docstring "
                    "does not mention what it returns."
                ),
            })
    return findings


# --------------------------------------------------------------- CrewAI tool


class ExtractFunctionInput(BaseModel):
    repo_path: str = Field(description="Absolute path to the local git repo.")
    ref: str = Field(description="Git ref to read from.")
    file_path: str = Field(description="Repo-relative file path.")
    function_name: str = Field(
        description="Function/method name, e.g. 'my_func' or 'MyClass.my_method'."
    )


class ExtractFunctionContextTool(BaseTool):
    """Extract the source of a single function/method from a file at a ref."""

    name: str = "extract_function_context"
    description: str = (
        "Returns the source code of a specific function or method from a "
        "file at a given git ref. Use when you need to see the full body."
    )
    args_schema: type[BaseModel] = ExtractFunctionInput

    def _run(
        self, repo_path: str, ref: str, file_path: str, function_name: str
    ) -> str:
        repo = git.Repo(repo_path)
        source = read_file_at(repo, ref, file_path)
        if source is None:
            return "FILE_NOT_FOUND"
        defs = extract_definitions(source, filename=file_path)
        for d in defs:
            if d.qualified_name == function_name or d.name == function_name:
                return d.source
        return "FUNCTION_NOT_FOUND"
