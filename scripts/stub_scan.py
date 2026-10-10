"""Scan of ``src/`` for stand-ins, stubs and toy code (R-NFR-004, CLAUDE.md rules 1 and 2).

Three families of rules, all applied to every file under ``src/`` (``.py`` for the syntax rules,
every text file for the keyword rule):

``keyword``
    ``TODO``, ``FIXME``, ``XXX``, ``NotImplementedError``, ``stub``, ``placeholder``, ``mock``,
    ``fake``, ``dummy``, ``简化``, ``示例``, ``暂不``, ``以后再`` - in identifiers, comments and
    strings alike.  Identifiers are split at underscores and camel-case humps first, so
    ``fake_clock`` and ``FakeClock`` are found as well as ``fake clock``; matching ignores case.
``empty-body`` / ``constant-return``
    A function or method whose whole body is ``pass``, ``...``, a docstring, ``return`` /
    ``return None`` or a return of a literal constant (a number, a string, ``True`` / ``False``, an
    empty list / dict / tuple / set).  Members of a ``Protocol``, ``@abstractmethod`` and
    ``@overload`` are exempt by definition.
``tests-import``
    ``src/`` imports ``tests`` (or ``scripts``) - directly, through ``importlib`` / ``__import__``
    with a literal name - or can reach such a module from ``twin.cli``: no ``twin`` command may
    depend on a fake model.

A finding that is a legitimate use is listed in ``scripts/stub_scan_allowlist.toml`` with the file,
the symbol (syntax rules) or an excerpt of the line (keyword rule) and the reason.  An entry that
matches nothing, or has no reason, is an error itself, so the list cannot rot.

Usage::

    uv run python scripts/stub_scan.py                  # exit code 1 on any finding
    uv run python scripts/stub_scan.py --show-allowed   # also list what the allowlist covers
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIRNAME = "src"
PACKAGE = "twin"
DEFAULT_ALLOWLIST = Path(__file__).resolve().with_name("stub_scan_allowlist.toml")
SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".enc", ".db"}
FORBIDDEN_IMPORT_ROOTS = ("tests", "scripts")
MIN_REASON_CHARS = 12

RULE_KEYWORD = "keyword"
RULE_EMPTY_BODY = "empty-body"
RULE_CONSTANT_RETURN = "constant-return"
RULE_TESTS_IMPORT = "tests-import"
SYNTAX_RULES = (RULE_EMPTY_BODY, RULE_CONSTANT_RETURN)
ALL_RULES = (RULE_KEYWORD, RULE_EMPTY_BODY, RULE_CONSTANT_RETURN, RULE_TESTS_IMPORT)

# Words are matched after identifiers were split into words (see ``split_identifiers``).
_WORDS = re.compile(
    r"\b(?:todo|fixme|xxx|stubs?|stubbed|stubbing|placeholders?|mock\w*|fake\w*|dummy|dummies)\b",
    re.IGNORECASE,
)
_NOT_IMPLEMENTED = re.compile(r"NotImplementedError")
_CHINESE = re.compile("简化|示例|暂不|以后再")
_HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


@dataclass(frozen=True)
class Finding:
    """One thing the scan objects to."""

    path: str
    line: int
    rule: str
    detail: str
    symbol: str = ""
    excerpt: str = ""

    def render(self) -> str:
        where = f" in {self.symbol}" if self.symbol else ""
        return f"{self.path}:{self.line}: {self.rule}{where}: {self.detail}"


@dataclass(frozen=True)
class AllowEntry:
    """One justified exception of the allowlist."""

    rule: str
    file: str
    reason: str
    symbol: str = ""
    contains: str = ""
    index: int = 0

    def covers(self, finding: Finding) -> bool:
        if finding.rule != self.rule or finding.path != self.file:
            return False
        if self.symbol and finding.symbol != self.symbol:
            return False
        return not self.contains or self.contains in finding.excerpt


@dataclass
class Allowlist:
    entries: list[AllowEntry] = field(default_factory=list)

    def problems(self) -> list[str]:
        """Entries that cannot be right: unknown rule, no reason, no way to tell what they cover."""
        found: list[str] = []
        for entry in self.entries:
            label = f"allowlist entry {entry.index} ({entry.rule} {entry.file})"
            if entry.rule not in ALL_RULES:
                found.append(f"{label}: unknown rule")
            if len(entry.reason.strip()) < MIN_REASON_CHARS:
                found.append(f"{label}: the reason is missing or too short")
            if not entry.file:
                found.append(f"{label}: no file")
            if entry.rule in SYNTAX_RULES and not entry.symbol:
                found.append(f"{label}: a syntax rule needs the symbol (Class.method)")
            if entry.rule == RULE_KEYWORD and not entry.contains:
                found.append(f"{label}: the keyword rule needs an excerpt of the line")
        return found


def load_allowlist(path: Path) -> Allowlist:
    """Read ``[[allow]]`` tables from a TOML file (a missing file is an empty list)."""
    if not path.is_file():
        return Allowlist()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = []
    for index, table in enumerate(data.get("allow", []), start=1):
        entries.append(
            AllowEntry(
                rule=str(table.get("rule", "")),
                file=str(table.get("file", "")),
                reason=str(table.get("reason", "")),
                symbol=str(table.get("symbol", "")),
                contains=str(table.get("contains", "")),
                index=index,
            )
        )
    return Allowlist(entries)


# ---------------------------------------------------------------------------------- keywords


def split_identifiers(text: str) -> str:
    """``fake_clock`` and ``FakeClock`` as ``fake clock`` (so word boundaries apply)."""
    return _HUMP.sub(" ", text.replace("_", " "))


def keyword_hits(line: str) -> list[str]:
    """The forbidden words in one line of text (empty when it is clean)."""
    hits = [match.group(0) for match in _WORDS.finditer(split_identifiers(line))]
    hits += _NOT_IMPLEMENTED.findall(line)
    hits += _CHINESE.findall(line)
    return hits


def scan_keywords(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        hits = keyword_hits(line)
        if hits:
            findings.append(
                Finding(
                    path,
                    number,
                    RULE_KEYWORD,
                    f"{', '.join(sorted(set(hits)))}: {line.strip()[:100]}",
                    excerpt=line.strip(),
                )
            )
    return findings


# --------------------------------------------------------------------------------- ast rules


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_ellipsis(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and node.value.value is Ellipsis
    )


_EMPTY_BUILTINS = {"list", "dict", "set", "tuple", "frozenset"}


def is_literal(node: ast.expr | None) -> bool:
    """Is ``node`` a fixed value: a constant, a signed number or an empty container?"""
    if node is None:
        return True  # a bare ``return``
    if isinstance(node, ast.Constant):
        return True
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub | ast.UAdd)
        and isinstance(node.operand, ast.Constant)
    ):
        return True
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _EMPTY_BUILTINS
        and not node.args
        and not node.keywords
    )


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute):
            names.add(target.attr)
        elif isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _is_protocol(node: ast.ClassDef) -> bool:
    for base in node.bases:
        target = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(target, ast.Name) and target.id == "Protocol":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "Protocol":
            return True
    return False


def _classify(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, int, str] | None:
    """``(rule, line, detail)`` when the body of ``node`` is nothing but a stand-in."""
    body = [stmt for stmt in node.body if not _is_docstring(stmt)]
    if not body:
        return RULE_EMPTY_BODY, node.lineno, "the body is only a docstring"
    returns: list[ast.Return] = []
    for stmt in body:
        if isinstance(stmt, ast.Pass) or _is_ellipsis(stmt):
            continue
        if isinstance(stmt, ast.Return) and is_literal(stmt.value):
            returns.append(stmt)
            continue
        return None
    if returns:
        last = returns[-1]
        value = "None" if last.value is None else ast.unparse(last.value)
        return RULE_CONSTANT_RETURN, last.lineno, f"always returns the constant {value}"
    return RULE_EMPTY_BODY, node.lineno, "the body is only pass / ..."


def _walk_functions(
    body: Sequence[ast.stmt], scope: tuple[str, ...], in_protocol: bool
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, bool]]:
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield from _walk_functions(node.body, (*scope, node.name), _is_protocol(node))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            name = ".".join((*scope, node.name))
            yield name, node, in_protocol
            yield from _walk_functions(node.body, (*scope, node.name), False)
        else:
            for child_body in _child_bodies(node):
                yield from _walk_functions(child_body, scope, in_protocol)


def _child_bodies(node: ast.stmt) -> Iterator[list[ast.stmt]]:
    for attribute in ("body", "orelse", "finalbody"):
        child = getattr(node, attribute, None)
        if isinstance(child, list) and child and isinstance(child[0], ast.stmt):
            yield child
    for handler in getattr(node, "handlers", []):
        yield handler.body
    for case in getattr(node, "cases", []):
        yield case.body


def scan_functions(path: str, source: str) -> list[Finding]:
    """Functions of one module whose whole body is a stand-in (see the module description)."""
    tree = ast.parse(source)
    lines = source.splitlines()
    findings: list[Finding] = []
    for name, node, in_protocol in _walk_functions(tree.body, (), False):
        if in_protocol:
            continue
        if _decorator_names(node) & {"abstractmethod", "overload"}:
            continue
        verdict = _classify(node)
        if verdict is None:
            continue
        rule, line, detail = verdict
        excerpt = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        findings.append(Finding(path, line, rule, detail, symbol=name, excerpt=excerpt))
    return findings


# -------------------------------------------------------------------------- import rule


def module_name(path: Path, src_root: Path) -> str:
    """``src/twin/a/b.py`` as ``twin.a.b`` (``__init__`` stands for its package)."""
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_modules(tree: ast.Module, current: str, *, is_package: bool) -> list[str]:
    """Absolute module names a module imports (relative imports resolved)."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package = current.split(".")
                if not is_package:
                    package = package[:-1]
                package = package[: len(package) - (node.level - 1)]
                base = ".".join([*package, base] if base else package)
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Call):
            target = node.func
            called = (
                target.attr
                if isinstance(target, ast.Attribute)
                else target.id
                if isinstance(target, ast.Name)
                else ""
            )
            if (
                called in {"import_module", "__import__"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.append(node.args[0].value)
    return names


def _is_forbidden(name: str) -> bool:
    return name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS


def scan_imports(sources: dict[str, tuple[str, str]], entry: str = "twin.cli") -> list[Finding]:
    """The imports of ``tests`` / ``scripts``, and which of them ``entry`` can reach.

    ``sources`` maps a module name to ``(display path, source text)``.
    """
    graph: dict[str, list[str]] = {}
    findings: list[Finding] = []
    for module, (display, text) in sources.items():
        tree = ast.parse(text)
        is_package = display.endswith("__init__.py")
        imports = imported_modules(tree, module, is_package=is_package)
        graph[module] = [name for name in imports if name in sources]
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = node.func
                called = (
                    target.attr
                    if isinstance(target, ast.Attribute)
                    else target.id
                    if isinstance(target, ast.Name)
                    else ""
                )
                if called in {"import_module", "__import__"}:
                    names = [node.args[0].value]
            for name in names:
                if _is_forbidden(name):
                    findings.append(
                        Finding(
                            display,
                            getattr(node, "lineno", 1),
                            RULE_TESTS_IMPORT,
                            f"imports {name}: production code must not depend on test support",
                            symbol=module,
                            excerpt=name,
                        )
                    )
    chain = _reachable(graph, entry)
    for finding in list(findings):
        module = finding.symbol
        if module in chain:
            findings.append(
                Finding(
                    finding.path,
                    finding.line,
                    RULE_TESTS_IMPORT,
                    "reachable from the command line: " + " -> ".join(chain[module]),
                    symbol=module,
                    excerpt=finding.excerpt,
                )
            )
    return findings


def _reachable(graph: dict[str, list[str]], entry: str) -> dict[str, list[str]]:
    """Module -> import path from ``entry`` (breadth first)."""
    if entry not in graph:
        return {}
    paths = {entry: [entry]}
    queue: deque[str] = deque([entry])
    while queue:
        current = queue.popleft()
        for name in graph.get(current, []):
            if name not in paths:
                paths[name] = [*paths[current], name]
                queue.append(name)
    return paths


# ------------------------------------------------------------------------------ the scan


def source_files(src_root: Path) -> list[Path]:
    """Every text file under ``src/`` (the scan never follows symlinks out of the tree)."""
    files: list[Path] = []
    for path in sorted(src_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in SKIP_DIRS for part in path.parts) or path.suffix in SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def scan_tree(root: Path) -> tuple[list[Finding], int]:
    """All findings under ``root/src`` and the number of files looked at."""
    src_root = root / SRC_DIRNAME
    findings: list[Finding] = []
    modules: dict[str, tuple[str, str]] = {}
    files = source_files(src_root)
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        display = path.relative_to(root).as_posix()
        findings.extend(scan_keywords(display, text))
        if path.suffix == ".py":
            findings.extend(scan_functions(display, text))
            modules[module_name(path, src_root)] = (display, text)
    findings.extend(scan_imports(modules))
    return findings, len(files)


def apply_allowlist(
    findings: Iterable[Finding], allowlist: Allowlist
) -> tuple[list[Finding], list[Finding], list[str]]:
    """``(remaining, allowed, stale entries)``."""
    used: set[int] = set()
    remaining: list[Finding] = []
    allowed: list[Finding] = []
    for finding in findings:
        match = next((e for e in allowlist.entries if e.covers(finding)), None)
        if match is None:
            remaining.append(finding)
        else:
            allowed.append(finding)
            used.add(match.index)
    stale = [
        f"allowlist entry {entry.index} ({entry.rule} {entry.file}) matches nothing: remove it"
        for entry in allowlist.entries
        if entry.index not in used
    ]
    return remaining, allowed, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--show-allowed", action="store_true", help="also list allowed findings")
    args = parser.parse_args(argv)

    allowlist = load_allowlist(args.allowlist)
    findings, count = scan_tree(args.root)
    remaining, allowed, stale = apply_allowlist(findings, allowlist)
    problems = [*allowlist.problems(), *stale]
    if args.show_allowed:
        for finding in allowed:
            sys.stdout.write(f"allowed  {finding.render()}\n")
    if remaining or problems:
        sys.stdout.write("stub scan FAILED: stand-ins or toy code in src/\n")
        for finding in remaining:
            sys.stdout.write(f"  {finding.render()}\n")
        for problem in problems:
            sys.stdout.write(f"  {problem}\n")
        return 1
    sys.stdout.write(
        f"stub scan passed ({count} files, {len(allowed)} findings covered by "
        f"{len(allowlist.entries)} allowlist entries)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
