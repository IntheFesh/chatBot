"""Coverage gate (R-NFR-004): total >= 85 % and every sub-package >= 75 %.

Reads the JSON report written by ``pytest --cov=src/twin --cov-report=json`` and
groups ``src/twin/<package>/...`` by package; modules directly under ``src/twin``
form the "(top-level)" group.  Packages without executable statements are
reported but cannot fail.

Usage::

    uv run python scripts/coverage_gate.py [--json coverage.json] [--total 85] [--package 75]
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


def summarise(report: dict[str, Any]) -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
    """``({package: (covered, statements)}, (covered_total, statements_total))``."""
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, data in report["files"].items():
        summary = data["summary"]
        group = groups[package_of(path)]
        group[0] += int(summary["covered_lines"])
        group[1] += int(summary["num_statements"])
    per_package = {name: (cov, stmts) for name, (cov, stmts) in groups.items()}
    total = (
        sum(cov for cov, _ in per_package.values()),
        sum(stmts for _, stmts in per_package.values()),
    )
    return per_package, total


def percent(covered: int, statements: int) -> float:
    return 100.0 if statements == 0 else 100.0 * covered / statements


def evaluate(
    report: dict[str, Any], total_min: float, package_min: float
) -> tuple[list[str], list[str]]:
    """Return ``(report_lines, failures)``."""
    per_package, (cov_total, stmt_total) = summarise(report)
    lines: list[str] = []
    failures: list[str] = []
    for name in sorted(per_package):
        covered, statements = per_package[name]
        pct = percent(covered, statements)
        mark = "ok"
        if statements and pct < package_min:
            mark = "FAIL"
            failures.append(f"package {name}: {pct:.1f}% < {package_min:.0f}%")
        lines.append(f"  {name:<14} {pct:6.1f}%  ({covered}/{statements}) {mark}")
    total_pct = percent(cov_total, stmt_total)
    if total_pct < total_min:
        failures.append(f"total: {total_pct:.1f}% < {total_min:.0f}%")
    lines.append(f"  {'TOTAL':<14} {total_pct:6.1f}%  ({cov_total}/{stmt_total})")
    return lines, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "coverage.json")
    parser.add_argument("--total", type=float, default=85.0)
    parser.add_argument("--package", type=float, default=75.0)
    args = parser.parse_args(argv)
    if not args.json.is_file():
        sys.stderr.write(
            f"{args.json} not found; run `pytest --cov=src/twin --cov-report=json` first\n"
        )
        return 2
    report = json.loads(args.json.read_text(encoding="utf-8"))
    lines, failures = evaluate(report, args.total, args.package)
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
