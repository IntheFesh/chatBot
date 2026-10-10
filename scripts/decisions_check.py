"""Audit of ``docs/DECISIONS.md`` (R-SCOPE-009, round 16).

Checks, all of them on the files as they are:

1. **Numbering** - every decision row ``| D-nnn |`` has a unique number (gaps are fine, repeats are
   not), and every ``D-nnn`` that any document, source file, test, script or workflow mentions is
   a decision that exists.  (A range such as ``D-490..D-497`` is checked at both ends.)
2. **Trade-off table** (section "取舍记录") - it has a row for every trade-off point that the SPEC
   names in R-SCOPE-009, and every row gives the order taken, the reason (the priority it serves)
   and at least one test.
3. **Test nodes** - every ``tests/...py::name`` cited anywhere in the file (also the short form
   ``::name`` that continues the file of the previous citation in the same row) exists:
   ``pytest --collect-only`` finds it, or - for a bare file - the file exists.  A test that was
   renamed or deleted since a decision cited it is a failure of this check.

Usage::

    uv run python scripts/decisions_check.py
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECISION_ROW = re.compile(r"^\|\s*(D-\d{3})\s*\|", re.MULTILINE)
REFERENCE = re.compile(r"(?<![\w-])D-(\d{3})(?!\d)")
TRADEOFF_HEADING = re.compile(r"^##\s+\d+\.\s*取舍记录", re.MULTILINE)
NEXT_HEADING = re.compile(r"^##\s", re.MULTILINE)
NODE = re.compile(
    r"(?<![\w/.\-])(?P<file>tests/[\w./\-]+\.py|test_\w+\.py)"
    r"(?P<parts>(?:::\w+\*?(?:\[[^\]\s`]*\])?)*)"
    r"|(?<![\w/\].])(?P<short>::[A-Za-z_]\w*\*?(?:\[[^\]\s`]*\])?)"
)
REFERENCE_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".ps1", ".sh", ".txt", ".mako"}
SKIP_PARTS = {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules", "data"}
SPEC_POINTS = re.compile(r"R-SCOPE-009\*\*.*?至少[:：](?P<points>[^）)]+)[）)]", re.DOTALL)


@dataclass(frozen=True)
class NodeRef:
    """A test cited in the document: the node id (or bare file) and where."""

    node: str
    line: int
    is_file: bool


@dataclass(frozen=True)
class TradeOff:
    """One row of the trade-off table."""

    point: str
    rounds: str
    order: str
    reason: str
    tests: str
    line: int


# ----------------------------------------------------------------------------- numbering


def decision_ids(text: str) -> list[tuple[str, int]]:
    """``(D-nnn, line)`` of every decision row, in document order."""
    return [
        (match.group(1), text.count("\n", 0, match.start()) + 1)
        for match in DECISION_ROW.finditer(text)
    ]


def duplicate_ids(ids: Iterable[tuple[str, int]]) -> list[str]:
    seen: dict[str, int] = {}
    problems: list[str] = []
    for ident, line in ids:
        if ident in seen:
            problems.append(f"{ident} is defined twice (lines {seen[ident]} and {line})")
        else:
            seen[ident] = line
    return problems


def references(text: str) -> list[tuple[str, int]]:
    """Every ``D-nnn`` mentioned in ``text``, with its line."""
    found: list[tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        found.extend((f"D-{match.group(1)}", number) for match in REFERENCE.finditer(line))
    return found


def reference_files(root: Path) -> list[Path]:
    """The files that may mention a decision: documents, code, tests, scripts, workflows."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.relative_to(root).parts):
            continue
        if path.suffix in REFERENCE_SUFFIXES:
            files.append(path)
    return files


def unknown_references(root: Path, known: set[str]) -> list[str]:
    problems: list[str] = []
    for path in reference_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for ident, line in references(text):
            if ident not in known:
                problems.append(
                    f"{path.relative_to(root).as_posix()}:{line}: {ident} is mentioned "
                    "but docs/DECISIONS.md has no such decision"
                )
    return problems


# ------------------------------------------------------------------------ trade-off table


def spec_points(spec_text: str) -> list[str]:
    """The trade-off points that R-SCOPE-009 says must be listed."""
    match = SPEC_POINTS.search(spec_text)
    if match is None:
        return []
    return [part.strip() for part in match.group("points").split("、") if part.strip()]


def tradeoff_section(text: str) -> tuple[str, int]:
    """The text of the trade-off section and the line it starts on."""
    start = TRADEOFF_HEADING.search(text)
    if start is None:
        return "", 0
    end = NEXT_HEADING.search(text, start.end())
    body = text[start.end() : end.start() if end else len(text)]
    return body, text.count("\n", 0, start.start()) + 1


def tradeoff_rows(text: str) -> list[TradeOff]:
    body, first_line = tradeoff_section(text)
    rows: list[TradeOff] = []
    for offset, line in enumerate(body.splitlines()):
        if not line.startswith("|") or line.startswith("| ---") or "取舍点" in line.split("|")[1]:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            rows.append(TradeOff(cells[0] if cells else "", "", "", "", "", first_line + offset))
            continue
        rows.append(TradeOff(*cells, line=first_line + offset))
    return rows


def check_tradeoffs(text: str, required: Iterable[str]) -> list[str]:
    rows = tradeoff_rows(text)
    problems: list[str] = []
    if not rows:
        return ["docs/DECISIONS.md has no trade-off table (section '取舍记录')"]
    for point in required:
        if not any(point in row.point for row in rows):
            problems.append(f"trade-off point missing from the table: {point}")
    for row in rows:
        name = row.point or f"line {row.line}"
        where = f"trade-off '{name}' (line {row.line})"
        if not (row.order and row.reason and row.tests):
            problems.append(f"{where}: needs the order taken, the reason and the tests (5 cells)")
            continue
        if not [ref for ref in node_refs(row.tests) if not ref.is_file]:
            problems.append(f"{where}: names no test node (tests/...py::test_name)")
    return problems


# --------------------------------------------------------------------------- test nodes


def test_file_index(root: Path) -> dict[str, str]:
    """Bare file name -> path relative to ``root``, for the test files whose name is unique."""
    seen: dict[str, list[str]] = {}
    tests = root / "tests"
    if tests.is_dir():
        for path in sorted(tests.rglob("test_*.py")):
            seen.setdefault(path.name, []).append(path.relative_to(root).as_posix())
    return {name: paths[0] for name, paths in seen.items() if len(paths) == 1}


def node_refs(text: str, index: dict[str, str] | None = None) -> list[NodeRef]:
    """The tests cited in ``text``; the short ``::name`` continues the last file of its line.

    A bare ``test_x.py::name`` is resolved through ``index`` (see :func:`test_file_index`); one
    that names no unique file is kept as written and reported later as a missing file.
    """
    found: list[NodeRef] = []
    for number, line in enumerate(text.splitlines(), start=1):
        last_file: str | None = None
        for match in NODE.finditer(line):
            if match.group("file"):
                name = match.group("file")
                parts = match.group("parts")
                if not name.startswith("tests/"):
                    if not parts:
                        continue  # a file named in prose
                    name = (index or {}).get(name, name)
                last_file = name
                found.append(NodeRef(last_file + parts, number, is_file=not parts))
            elif last_file is not None:
                found.append(NodeRef(last_file + match.group("short"), number, is_file=False))
    return found


def is_collected(node: str, collected: set[str]) -> bool:
    """Exact node id, a parametrised test (``node[...]``), a class prefix, or ``name*``.

    A trailing ``*`` is a prefix: ``test_redo_*`` cites every test whose name starts so, and at
    least one has to exist.
    """
    if node.endswith("*"):
        prefix = node[:-1]
        return any(item.startswith(prefix) for item in collected)
    return (
        node in collected
        or any(item.startswith(node + "[") for item in collected)
        or any(item.startswith(node + "::") for item in collected)
    )


def collect_nodes(root: Path, files: Iterable[str]) -> set[str]:
    """Node ids ``pytest --collect-only -q`` reports for the given test files."""
    names = sorted({name for name in files if (root / name).is_file()})
    if not names:
        return set()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *names],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    ids = {line.strip() for line in result.stdout.splitlines() if "::" in line}
    if result.returncode not in (0, 5) and not ids:
        raise RuntimeError(
            f"pytest collection failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}"
        )
    return ids


def defined_names(source: str) -> set[str]:
    """Dotted names of the functions, classes and variables a module defines at its top levels."""
    names: set[str] = set()

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(prefix + node.name)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, ast.Assign):
                names.update(prefix + t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(prefix + node.target.id)
            elif isinstance(node, ast.If | ast.Try):
                visit(node.body, prefix)
                visit(node.orelse, prefix)

    visit(ast.parse(source).body, "")
    return names


def is_test_module(file: str) -> bool:
    """Does pytest collect tests from this file (``test_*.py``)?"""
    return Path(file).name.startswith("test_")


def is_test_name(symbol: str) -> bool:
    """Is the last name of a dotted node a test (``test_x``, ``TestX``) rather than a helper?"""
    return symbol.rsplit(".", 1)[-1].startswith(("test", "Test"))


def is_defined(symbol: str, names: set[str]) -> bool:
    """The module defines ``symbol`` (``name*`` stands for every name that starts so)."""
    if symbol.endswith("*"):
        return any(name.startswith(symbol[:-1]) for name in names)
    return symbol in names


def check_nodes(
    refs: Iterable[NodeRef],
    root: Path,
    collector: Callable[[Path, Iterable[str]], set[str]] = collect_nodes,
) -> list[str]:
    """Problems for cited tests that do not exist.

    A test (``test_x``) in a test module has to be found by ``pytest --collect-only``.  A fixture
    or helper that a decision mentions (``test_x.py::no_bot_reader``, ``tests/support/x.py::name``,
    ``tests/conftest.py::name``) has to be defined in its module, since pytest collects nothing
    for those.
    """
    cited = list(refs)
    problems: list[str] = []
    files = {ref.node.split("::")[0] for ref in cited}
    for name in sorted(files):
        if not (root / name).is_file():
            problems.append(f"test file does not exist: {name}")
    collected = collector(root, {name for name in files if is_test_module(name)})
    definitions: dict[str, set[str]] = {}
    reported: set[str] = set()
    for ref in cited:
        file, _, rest = ref.node.partition("::")
        if ref.is_file or not (root / file).is_file() or ref.node in reported:
            continue
        symbol = rest.split("[")[0].replace("::", ".")
        if is_test_module(file) and is_test_name(symbol):
            found, how = is_collected(ref.node, collected), "pytest --collect-only"
        else:
            if file not in definitions:
                definitions[file] = defined_names((root / file).read_text(encoding="utf-8"))
            found, how = is_defined(symbol, definitions[file]), "the definitions of the module"
        if not found:
            reported.add(ref.node)
            problems.append(f"line {ref.line}: not found by {how}: {ref.node}")
    return problems


# ----------------------------------------------------------------------------- the check


def check(
    root: Path, collector: Callable[[Path, Iterable[str]], set[str]] = collect_nodes
) -> tuple[list[str], dict[str, int]]:
    """All problems of ``docs/DECISIONS.md`` under ``root`` and some counts."""
    decisions = (root / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    spec = (root / "docs" / "SPEC.md").read_text(encoding="utf-8")
    ids = decision_ids(decisions)
    problems = duplicate_ids(ids)
    problems += unknown_references(root, {ident for ident, _ in ids})
    required = spec_points(spec)
    if not required:
        problems.append("docs/SPEC.md: cannot find the trade-off points listed by R-SCOPE-009")
    problems += check_tradeoffs(decisions, required)
    refs = node_refs(decisions, test_file_index(root))
    problems += check_nodes(refs, root, collector)
    stats = {
        "decisions": len(ids),
        "tradeoffs": len(tradeoff_rows(decisions)),
        "required_points": len(required),
        "cited_tests": len({ref.node for ref in refs if not ref.is_file}),
    }
    return problems, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    args = parser.parse_args(argv)
    problems, stats = check(args.root, collect_nodes)
    sys.stdout.write(
        f"decisions_check: {stats['decisions']} decisions, {stats['tradeoffs']} trade-off rows "
        f"({stats['required_points']} required by the SPEC), {stats['cited_tests']} cited tests\n"
    )
    if problems:
        sys.stdout.write(f"{len(problems)} problem(s):\n")
        for problem in problems:
            sys.stdout.write(f"  - {problem}\n")
        return 1
    sys.stdout.write("OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
