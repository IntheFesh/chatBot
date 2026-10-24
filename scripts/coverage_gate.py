"""Coverage gate (R-NFR-004): total >= 85 % and every sub-package >= 75 %.

Reads the JSON report written by ``pytest --cov=src/twin --cov-report=json`` and
groups ``src/twin/<package>/...`` by package; modules directly under ``src/twin``
form the "(top-level)" group.  Packages without executable statements are
reported but cannot fail.

A package that has packages of its own (``ops/backup``, ``channel/ilink``, ``storage/migrations``
...) is gated twice: as a whole, as before, and every directory below it on its own, with the same
threshold (round 17, D-740) - a big, well-covered parent no longer hides a weak child.  The total
is the sum of the top-level groups only, so nothing is counted twice.  ``--no-nested`` switches
the second gate off.

Usage::

    uv run python scripts/coverage_gate.py [--json coverage.json] [--total 85] [--package 75]
        [--no-nested]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TOP_LEVEL = "(top-level)"


def package_of(path: str) -> str:
    """``src/twin/ops/jobs.py`` -> ``ops``; ``src/twin/cli.py`` -> ``(top-level)``."""
    parts = Path(path.replace("\\", "/")).parts
    if "twin" not in parts:
        return TOP_LEVEL
    rest = parts[parts.index("twin") + 1 :]
    return rest[0] if len(rest) > 1 else TOP_LEVEL


def nested_packages_of(path: str) -> list[str]:
    """The directories below a package that hold ``path``: ``src/twin/ops/backup/x.py`` ->
    ``["ops/backup"]``; one more level down adds ``ops/backup/sub``.  Empty for a module that
    sits directly in its package (or in ``src/twin``)."""
    parts = Path(path.replace("\\", "/")).parts
    if "twin" not in parts:
        return []
    rest = parts[parts.index("twin") + 1 :]
    directories = rest[:-1]
    return ["/".join(directories[: depth + 1]) for depth in range(1, len(directories))]


def summarise(
    report: dict[str, Any], *, nested: bool = False
) -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
    """``({group: (covered, statements)}, (covered_total, statements_total))``.

    The groups are the packages; with ``nested`` also every directory below a package (named
    ``package/sub``).  The total counts each statement once, whatever the groups are.
    """
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for path, data in report["files"].items():
        summary = data["summary"]
        covered, statements = int(summary["covered_lines"]), int(summary["num_statements"])
        names = [package_of(path)]
        if nested:
            names += nested_packages_of(path)
        for name in names:
            groups[name][0] += covered
            groups[name][1] += statements
        total[0] += covered
        total[1] += statements
    per_group = {name: (cov, stmts) for name, (cov, stmts) in groups.items()}
    return per_group, (total[0], total[1])


def percent(covered: int, statements: int) -> float:
    return 100.0 if statements == 0 else 100.0 * covered / statements


def evaluate(
    report: dict[str, Any], total_min: float, package_min: float, *, nested: bool = True
) -> tuple[list[str], list[str]]:
    """Return ``(report_lines, failures)``."""
    per_package, (cov_total, stmt_total) = summarise(report, nested=nested)
    lines: list[str] = []
    failures: list[str] = []
    for name in sorted(per_package):
        covered, statements = per_package[name]
        pct = percent(covered, statements)
        mark = "ok"
        if statements and pct < package_min:
            mark = "FAIL"
            failures.append(f"package {name}: {pct:.1f}% < {package_min:.0f}%")
        lines.append(f"  {name:<24} {pct:6.1f}%  ({covered}/{statements}) {mark}")
    total_pct = percent(cov_total, stmt_total)
    if total_pct < total_min:
        failures.append(f"total: {total_pct:.1f}% < {total_min:.0f}%")
    lines.append(f"  {'TOTAL':<24} {total_pct:6.1f}%  ({cov_total}/{stmt_total})")
    return lines, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "coverage.json")
    parser.add_argument("--total", type=float, default=85.0)
    parser.add_argument("--package", type=float, default=75.0)
    parser.add_argument(
        "--no-nested",
        action="store_true",
        help="gate the packages only, not the directories below them",
    )
    args = parser.parse_args(argv)
    if not args.json.is_file():
        sys.stderr.write(
            f"{args.json} not found; run `pytest --cov=src/twin --cov-report=json` first\n"
        )
        return 2
    report = json.loads(args.json.read_text(encoding="utf-8"))
    lines, failures = evaluate(report, args.total, args.package, nested=not args.no_nested)
    sys.stdout.write("coverage by package:\n" + "\n".join(lines) + "\n")
    if failures:
        sys.stdout.write("coverage gate FAILED:\n" + "\n".join(f"  - {f}" for f in failures) + "\n")
        return 1
    sys.stdout.write(
        f"coverage gate passed (total >= {args.total:.0f}%, each package >= {args.package:.0f}%)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
