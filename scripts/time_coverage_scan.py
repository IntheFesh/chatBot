"""Which modules that deal with time have a test that can move the clock (R-NFR-005).

A development tool, not a ``twin`` command.  "All time-related logic has tests on an injected clock,
DST switch days and time zone switch days included": this script keeps the first half of that
sentence honest by reading the source, not by running anything.

* A module of ``src/twin`` is **time-related** when it imports ``zoneinfo``, imports from
  ``twin.clock`` (``Clock``, ``SystemClock``, ``get_clock`` ...), or imports
  ``twin.schedule.time_service`` (``TimeService``).
* It is **covered** when at least one test file imports it - directly (``import twin.x.y``,
  ``from twin.x.y import Name``, ``from twin.x import y``) or through the name a package
  re-exports from it (``from twin.x import Name`` where ``twin/x/__init__.py`` takes ``Name`` from
  ``twin.x.y``) - and that file can move time: it names a clock (``ManualClock``, ``LifeClock``,
  a ``clock`` parameter or fixture, ``set_active_clock``), uses ``time_machine``, or builds fixed
  aware instants (``datetime(2026, 11, 1, ..., tzinfo=...)``, ``ZoneInfo(...)``) that it hands to
  the code as ``now``.  Importing is the relation; the file's own control of time is what makes it
  a *time* test.  (Being imported by a module that a test imports does not count: a deep module
  would otherwise be "covered" by the whole suite.)
* The scan also looks for the two kinds of day the requirement names: tests of a daylight saving
  switch (a test whose name says dst, daylight, spring forward, clock change, a 25 or 23 hour day)
  and tests of a time zone switch (a name with timezone switch, switch_timezone, zone switched).

Usage::

    uv run python scripts/time_coverage_scan.py          # the table; exit 1 if there is a gap
    uv run python scripts/time_coverage_scan.py --list   # every time-related module, with its tests
    uv run python scripts/time_coverage_scan.py --json out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "twin"
CLOCK_MODULES = frozenset({"twin.clock"})
TIME_SERVICE_MODULES = frozenset({"twin.schedule.time_service"})
ZONE_MODULES = frozenset({"zoneinfo"})
# what makes a test file one that can move time: an injected clock, or fixed instants that are
# handed to the code as ``now`` (the pure functions of the schedule take no clock at all)
CLOCK_MARKERS = re.compile(
    r"\b(ManualClock|LifeClock|time_machine|set_active_clock|use_clock|FakeClock)\b"
    r"|\b[Cc]lock\b|\bclock_?\w*\b"
)
FIXED_INSTANTS = re.compile(
    r"\bdatetime\(\s*20\d\d\b[^)]*\b(tzinfo|UTC|ZoneInfo)"  # datetime(2026, 11, 1, tzinfo=UTC)
    r"|\bZoneInfo\(|\butc\(\s*20\d\d\b"  # the helper of the memory tests: utc(2026, 11, 1)
)
DST_NAME = re.compile(
    r"dst|daylight|spring_forward|clock_chang|clocks_(go|move|change)|25_hour|23_hour|"
    r"fall_back_(an_hour|to_standard)",
    re.I,
)
ZONE_NAME = re.compile(
    r"(time_?zone|zone)_(switch|switched|switching)|switch(es|ed|ing)?_(the_)?(bot_)?(time_?)?zone",
    re.I,
)


@dataclass
class Module:
    """A module of the package."""

    name: str
    path: str
    reasons: list[str] = field(default_factory=list)  # why it is time-related; empty if it is not
    tests: list[str] = field(default_factory=list)  # test files that cover it

    @property
    def time_related(self) -> bool:
        return bool(self.reasons)

    @property
    def covered(self) -> bool:
        return bool(self.tests)

    @property
    def subpackage(self) -> str:
        parts = self.name.split(".")
        return parts[1] if len(parts) > 2 else "(top level)"


@dataclass
class ScanResult:
    modules: list[Module]
    dst_tests: list[str]
    zone_tests: list[str]

    @property
    def related(self) -> list[Module]:
        return [module for module in self.modules if module.time_related]

    @property
    def gaps(self) -> list[Module]:
        return [module for module in self.related if not module.covered]

    @property
    def ok(self) -> bool:
        return not self.gaps and bool(self.dst_tests) and bool(self.zone_tests)


# ------------------------------------------------------------------------------- reading


def module_name(path: Path, src: Path) -> str:
    relative = path.relative_to(src).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def imported_modules(tree: ast.Module) -> list[tuple[str, tuple[str, ...]]]:
    """``(module, names)`` for every import of a tree; ``names`` is empty for ``import x``."""
    found: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, tuple(alias.name for alias in node.names)))
    return found


def reasons_for(tree: ast.Module) -> list[str]:
    """Why a module is time-related (see the module description); empty if it is not."""
    reasons: set[str] = set()
    for imported, names in imported_modules(tree):
        if imported in ZONE_MODULES:
            reasons.add("zoneinfo")
        elif imported in CLOCK_MODULES:
            reasons.add("Clock")
        elif imported in TIME_SERVICE_MODULES or (
            imported == "twin.schedule" and "time_service" in names
        ):
            reasons.add("TimeService")
    return sorted(reasons)


def package_exports(
    modules: dict[str, ast.Module], packages: set[str]
) -> dict[str, dict[str, str]]:
    """For each package: the names its ``__init__`` takes from submodules, and from which one."""
    exports: dict[str, dict[str, str]] = {}
    for package in packages:
        tree = modules[package]
        names: dict[str, str] = {}
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.level == 0
                and node.module.startswith(package + ".")
            ):
                for alias in node.names:
                    names[alias.asname or alias.name] = node.module
        exports[package] = names
    return exports


def modules_a_test_imports(
    tree: ast.Module, known: set[str], exports: dict[str, dict[str, str]]
) -> set[str]:
    """The modules of the package that one test file imports (directly or by a re-exported name)."""
    reached: set[str] = set()
    for imported, names in imported_modules(tree):
        if imported != PACKAGE and not imported.startswith(PACKAGE + "."):
            continue
        if imported in known:
            reached.add(imported)
        for name in names:
            submodule = f"{imported}.{name}"
            if submodule in known:
                reached.add(submodule)  # from twin.x import y  (y is a module)
            source = exports.get(imported, {}).get(name)
            if source is not None and source in known:
                reached.add(source)  # from twin.x import Name  (re-exported from twin.x.y)
    # ``import twin.x.y`` also reaches the packages above it, but a package that only re-exports
    # is not covered by that: it has no logic of its own
    return reached


def test_functions(tree: ast.Module) -> list[str]:
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test")
    ]


# ------------------------------------------------------------------------------- the scan


def scan(root: Path = ROOT, source: str = "src", tests: str = "tests") -> ScanResult:
    """Read ``root/source`` and ``root/tests`` and decide which modules are covered."""
    src = root / source
    sources = {
        module_name(path, src): path
        for path in sorted((src / PACKAGE).rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    trees = {name: tree for name, path in sources.items() if (tree := parse(path)) is not None}
    known = set(trees)
    packages = {name for name, path in sources.items() if path.name == "__init__.py"}
    exports = package_exports(trees, packages & known)
    modules = {
        name: Module(name, sources[name].relative_to(root).as_posix(), reasons_for(trees[name]))
        for name in sorted(trees)
    }
    dst: list[str] = []
    zone: list[str] = []
    for path in sorted((root / tests).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = parse(path)
        if tree is None:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(root).as_posix()
        movable = CLOCK_MARKERS.search(text) is not None or FIXED_INSTANTS.search(text) is not None
        if path.name.startswith("test_"):
            for name in test_functions(tree):
                if DST_NAME.search(name):
                    dst.append(f"{relative}::{name}")
                if ZONE_NAME.search(name):
                    zone.append(f"{relative}::{name}")
            if movable:
                for reached in modules_a_test_imports(tree, known, exports):
                    modules[reached].tests.append(relative)
    return ScanResult(list(modules.values()), dst, zone)


# ----------------------------------------------------------------------------------- output


def render(result: ScanResult, *, everything: bool = False) -> str:
    by_package: dict[str, list[Module]] = defaultdict(list)
    for module in result.related:
        by_package[module.subpackage].append(module)
    lines = [
        f"time-related modules: {len(result.related)} of {len(result.modules)}; "
        f"without a test that moves the clock: {len(result.gaps)}",
        "",
        f"{'subpackage':<14}{'modules':>8}{'covered':>9}",
    ]
    for package in sorted(by_package):
        group = by_package[package]
        lines.append(f"{package:<14}{len(group):>8}{sum(1 for m in group if m.covered):>9}")
    if everything:
        lines.append("")
        for module in result.related:
            lines.append(f"{module.name} [{', '.join(module.reasons)}]")
            lines.extend(f"    {test}" for test in module.tests[:6])
            if not module.tests:
                lines.append("    -- no test --")
    lines.append("")
    if result.gaps:
        lines.append("gaps (a test must import the module and use an injected clock):")
        lines.extend(
            f"  {module.name}  ({module.path})  [{', '.join(module.reasons)}]"
            for module in result.gaps
        )
    else:
        lines.append("gaps: none")
    lines.append("")
    lines.append(f"daylight saving switch tests: {len(result.dst_tests)}")
    lines.extend(f"  {name}" for name in result.dst_tests[:5])
    lines.append(f"time zone switch tests: {len(result.zone_tests)}")
    lines.extend(f"  {name}" for name in result.zone_tests[:5])
    if not result.dst_tests:
        lines.append("MISSING: no test of a daylight saving switch day")
    if not result.zone_tests:
        lines.append("MISSING: no test of a time zone switch day")
    lines.append("")
    lines.append("result: " + ("OK" if result.ok else "GAPS"))
    return "\n".join(lines)


def to_json(result: ScanResult) -> str:
    return json.dumps(
        {
            "modules": [asdict(m) for m in result.related],
            "gaps": [m.name for m in result.gaps],
            "dst_tests": result.dst_tests,
            "zone_tests": result.zone_tests,
            "ok": result.ok,
        },
        ensure_ascii=False,
        indent=2,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="every time-related module")
    parser.add_argument("--json", type=Path, default=None, help="write the findings here")
    parser.add_argument("--root", type=Path, default=ROOT, help="the repository (this one)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = scan(args.root)
    print(render(result, everything=args.list))
    if args.json is not None:
        args.json.write_text(to_json(result) + "\n", encoding="utf-8")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
