"""Requirement traceability check (run at the end of every round).

Reads ``docs/SPEC.md`` (requirement ids ``R-<MODULE>-<nnn>``) and table 2 of
``docs/TRACEABILITY.md`` and verifies:

1. every SPEC requirement has exactly one row in table 2 (and no row is unknown);
2. a row whose status is "已实现" names an implementation location and tests;
3. every ``path::symbol`` in a filled "实现位置" cell exists (Python symbols are
   looked up with ``ast``; other files only need to exist);
4. every test in a filled "测试" cell is found by ``pytest --collect-only -q``;
5. with ``--round NN``: for the requirements owned by that round, the *last*
   owning round must have set the status to "已实现"; earlier owning rounds must
   have recorded a partial note ``（部分：第 NN 轮）`` in "实现位置".

Usage::

    uv run python scripts/trace_check.py --round 00
    uv run python scripts/trace_check.py            # global consistency only
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_ID = re.compile(r"^\s*-\s+\*\*(R-[A-Z]+-\d{3})\*\*", re.MULTILINE)
PARTIAL = re.compile(r"[（(]部分[:：]\s*第\s*(\d{2}[a-z]?)\s*轮[)）]")
NOTE = re.compile(r"（[^）]*）")  # any full-width parenthetical is a free-text remark
IMPLEMENTED = "已实现"
PENDING = "待实现"


@dataclass(frozen=True)
class Row:
    req_id: str
    summary: str
    rounds: tuple[str, ...]
    status: str
    impl: str
    tests: str
    line: int


def round_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d{2})([a-z]?)", value)
    if not match:
        raise ValueError(f"invalid round id {value!r}")
    return int(match.group(1)), match.group(2)


def parse_spec_ids(text: str) -> list[str]:
    """Requirement ids defined in the SPEC, in document order, without duplicates."""
    seen: dict[str, None] = {}
    for match in SPEC_ID.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)


def parse_table(text: str) -> list[Row]:
    """Rows of table 2 (the section starting at '## 表 2')."""
    marker = text.find("## 表 2")
    if marker < 0:
        raise ValueError("TRACEABILITY.md has no '## 表 2' section")
    rows: list[Row] = []
    for offset, line in enumerate(text[marker:].splitlines()):
        if not line.startswith("| R-"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            raise ValueError(f"table 2 row has {len(cells)} cells instead of 6: {line[:60]}")
        req_id, summary, rounds, status, impl, tests = cells
        parsed_rounds = tuple(part.strip() for part in rounds.split(",") if part.strip())
        for item in parsed_rounds:
            round_key(item)
        rows.append(
            Row(
                req_id,
                summary,
                parsed_rounds,
                status,
                impl,
                tests,
                text[:marker].count("\n") + offset + 1,
            )
        )
    return rows


def split_locations(cell: str) -> list[str]:
    """``path::symbol; path`` -> ['path::symbol', 'path'] without partial notes or remarks."""
    cleaned = NOTE.sub("", PARTIAL.sub("", cell))
    return [part.strip() for part in cleaned.split(";") if part.strip()]


def partial_rounds(cell: str) -> set[str]:
    return set(PARTIAL.findall(cell))


def python_symbols(source: str) -> set[str]:
    """Dotted names of module-level and class-level definitions in ``source``."""
    names: set[str] = set()

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(prefix + node.name)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(prefix + target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(prefix + node.target.id)
            elif isinstance(node, ast.If | ast.Try):
                visit(node.body, prefix)
                visit(getattr(node, "orelse", []), prefix)

    visit(ast.parse(source).body, "")
    return names


def check_location(root: Path, location: str) -> str | None:
    """Return an error message if ``location`` does not exist."""
    path_part, _, symbol = location.partition("::")
    path = root / path_part
    if not path.is_file():
        return f"file not found: {path_part}"
    if symbol and path.suffix == ".py":
        try:
            symbols = python_symbols(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            return f"cannot parse {path_part}: {exc}"
        if symbol not in symbols:
            return f"symbol not found: {path_part}::{symbol}"
    return None


def collect_tests(root: Path) -> set[str]:
    """Node ids reported by ``pytest --collect-only -q``."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
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


def is_collected(name: str, collected: set[str]) -> bool:
    return name in collected or any(node.startswith(name + "[") for node in collected)


def check(
    spec_text: str,
    table_text: str,
    collected: set[str],
    root: Path,
    round_id: str | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Return ``(problems, stats)``."""
    problems: list[str] = []
    spec_ids = parse_spec_ids(spec_text)
    rows = parse_table(table_text)

    by_id: dict[str, Row] = {}
    for row in rows:
        if row.req_id in by_id:
            problems.append(f"{row.req_id}: duplicate row (line {row.line})")
        by_id[row.req_id] = row
    for req_id in spec_ids:
        if req_id not in by_id:
            problems.append(f"{req_id}: defined in SPEC but has no row in table 2")
    spec_set = set(spec_ids)
    for row in rows:
        if row.req_id not in spec_set:
            problems.append(f"{row.req_id}: row exists but the SPEC does not define it")
        if row.status not in (IMPLEMENTED, PENDING):
            problems.append(
                f"{row.req_id}: status must be {IMPLEMENTED} or {PENDING}, got {row.status!r}"
            )

    def validate_filled(row: Row) -> None:
        for location in split_locations(row.impl):
            error = check_location(root, location)
            if error:
                problems.append(f"{row.req_id}: 实现位置 {error}")
        for name in split_locations(row.tests):
            if not is_collected(name, collected):
                problems.append(f"{row.req_id}: test not collected by pytest: {name}")

    for row in rows:
        if row.status == IMPLEMENTED:
            if not split_locations(row.impl):
                problems.append(f"{row.req_id}: marked {IMPLEMENTED} without 实现位置")
            if not split_locations(row.tests):
                problems.append(f"{row.req_id}: marked {IMPLEMENTED} without tests")
        validate_filled(row)

    stats = {"spec": len(spec_ids), "rows": len(rows), "owned": 0, "implemented": 0, "partial": 0}
    if round_id is not None:
        round_key(round_id)
        for row in rows:
            if round_id not in row.rounds:
                continue
            stats["owned"] += 1
            last = max(row.rounds, key=round_key)
            if round_id == last:
                if row.status != IMPLEMENTED:
                    problems.append(
                        f"{row.req_id}: round {round_id} is its last owner "
                        f"but status is {row.status}"
                    )
                else:
                    stats["implemented"] += 1
            elif row.status == IMPLEMENTED:
                stats["implemented"] += 1
            elif round_id in partial_rounds(row.impl) and split_locations(row.impl):
                stats["partial"] += 1
            else:
                problems.append(
                    f"{row.req_id}: shared with later rounds; 实现位置 needs "
                    f"a location and the note （部分：第 {round_id} 轮）"
                )
    return problems, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--round", dest="round_id", help="round id, e.g. 00 or 09b")
    parser.add_argument("--spec", type=Path, default=ROOT / "docs" / "SPEC.md")
    parser.add_argument("--trace", type=Path, default=ROOT / "docs" / "TRACEABILITY.md")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    collected = collect_tests(args.root)
    problems, stats = check(
        args.spec.read_text(encoding="utf-8"),
        args.trace.read_text(encoding="utf-8"),
        collected,
        args.root,
        args.round_id,
    )
    scope = f"round {args.round_id}" if args.round_id else "global"
    sys.stdout.write(
        f"trace_check ({scope}): {stats['spec']} requirements in SPEC, {stats['rows']} rows; "
        f"owned by round={stats['owned']}, implemented={stats['implemented']}, "
        f"partial={stats['partial']}, collected tests={len(collected)}\n"
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
