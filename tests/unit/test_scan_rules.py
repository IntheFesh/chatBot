"""Repository-wide rules enforced by scanning the source (CLAUDE.md rules 1, 2, 5, 8)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "twin"
SOURCES = sorted(SRC.rglob("*.py"))
CLOCK_FILE = SRC / "clock.py"

FORBIDDEN_CALLS = {
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("datetime", "today"),
    ("date", "today"),
    ("time", "time"),
    ("time", "time_ns"),
    ("time", "monotonic"),
    ("time", "perf_counter"),
    ("time", "sleep"),
    ("asyncio", "sleep"),
}
FORBIDDEN_FROM_IMPORTS = {
    "time": {"time", "time_ns", "monotonic", "perf_counter", "sleep"},
    "asyncio": {"sleep"},
}
STUB_WORDS = re.compile(
    r"TODO|FIXME|\bXXX\b|\bstub\b|placeholder|\bmock|\bfake|\bdummy|NotImplementedError|简化|示例",
    re.IGNORECASE,
)


def test_sources_were_found() -> None:
    assert len(SOURCES) > 40


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(SRC)))
def test_src_never_imports_the_tests_package(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "tests" for a in node.names), path
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "tests", path


def test_no_direct_clock_access_outside_the_clock_module() -> None:
    offenders: list[str] = []
    for path in SOURCES:
        if path == CLOCK_FILE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func.value
                if isinstance(target, ast.Name) and (target.id, node.func.attr) in FORBIDDEN_CALLS:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {target.id}.{node.func.attr}()")
            if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_FROM_IMPORTS:
                bad = {a.name for a in node.names} & FORBIDDEN_FROM_IMPORTS[node.module]
                if bad:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} from {node.module} import {bad}")
    assert not offenders, "use twin.clock instead:\n" + "\n".join(offenders)


def test_the_scan_itself_catches_violations(tmp_path: Path) -> None:
    sample = ast.parse("import time\nfrom datetime import datetime\nx = datetime.now()\ny = time.time()\n")
    calls = [
        (n.func.value.id, n.func.attr)  # type: ignore[union-attr]
        for n in ast.walk(sample)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert set(calls) <= FORBIDDEN_CALLS and len(calls) == 2


def test_no_stub_or_toy_markers_in_production_code() -> None:
    offenders = []
    for path in SOURCES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if STUB_WORDS.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:80]}")
    assert not offenders, "\n".join(offenders)


def is_protocol_or_abstract(node: ast.FunctionDef | ast.AsyncFunctionDef, owner: ast.ClassDef | None) -> bool:
    if owner is not None:
        bases = {ast.unparse(base) for base in owner.bases}
        if any(base.startswith("Protocol") or "ABC" in base for base in bases):
            return True
    return any("abstractmethod" in ast.unparse(d) or "overload" in ast.unparse(d) for d in node.decorator_list)


def test_no_function_has_an_empty_body() -> None:
    offenders = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for owner in [None, *[n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]]:
            scope = tree.body if owner is None else owner.body
            for node in scope:
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                body = [
                    s for s in node.body
                    if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
                ]
                empty = all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis) for s in body)
                if (not body or empty) and not is_protocol_or_abstract(node, owner):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}")
    assert not offenders, "\n".join(offenders)


def test_application_code_logs_through_the_structured_logger_only() -> None:
    allowed = {SRC / "ops" / "logging.py"}
    offenders = [
        str(path.relative_to(ROOT))
        for path in SOURCES
        if path not in allowed and "logging.getLogger(" in path.read_text(encoding="utf-8")
    ]
    assert not offenders


def test_no_print_in_production_code() -> None:
    offenders = []
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders


def test_gitignore_covers_the_private_paths() -> None:
    lines = {
        line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    for entry in ("data/", "*.db", "*.enc", "exports/", "models/", "backups/", ".env"):
        assert entry in lines, f".gitignore must list {entry}"
