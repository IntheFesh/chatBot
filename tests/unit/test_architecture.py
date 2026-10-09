"""Package layout, interfaces and import structure (R-ARCH-002)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Protocol

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGES = [
    "config",
    "storage",
    "channel",
    "llm",
    "ingest",
    "profile",
    "stickers",
    "retrieval",
    "memory",
    "engine",
    "schedule",
    "commands",
    "learning",
    "ops",
    "training",
    "eval",
]


def test_package_layout_matches_claude_md_section_4() -> None:
    for package in PACKAGES:
        assert (SRC / "twin" / package / "__init__.py").is_file(), package
    assert (SRC / "twin" / "cli.py").is_file() and (SRC / "twin" / "app.py").is_file()
    for directory in (
        "docs",
        "prompts",
        "scripts",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "tests/support",
        "config/lists",
        "training",
    ):
        assert (ROOT / directory).is_dir(), directory
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for package in PACKAGES:
        assert package in claude


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("twin.clock", "Clock"),
        ("twin.app", "Component"),
        ("twin.ops.jobs", "OffPeakPolicy"),
        ("twin.ops.alerts", "AlertSink"),
        ("twin.ops.power", "PowerManager"),
        ("twin.ops.winapi", "Win32"),
        ("twin.config.secrets", "CredentialBackend"),
    ],
)
def test_module_boundaries_are_explicit_protocols(module: str, name: str) -> None:
    interface = getattr(importlib.import_module(module), name)
    assert Protocol in interface.__mro__ or getattr(interface, "_is_protocol", False)


def module_name(path: Path) -> str:
    relative = path.relative_to(SRC).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def runtime_imports(path: Path, known: set[str]) -> set[str]:
    """``twin.*`` modules imported at run time (``if TYPE_CHECKING`` blocks excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                visit(node.orelse)
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    names = [alias.name for alias in child.names]
                elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                    names = [child.module] + [
                        f"{child.module}.{alias.name}" for alias in child.names
                    ]
                else:
                    continue
                for name in names:
                    if name in known:
                        imported.add(name)

    visit(tree.body)
    return imported


def test_modules_form_an_acyclic_import_graph() -> None:
    files = sorted((SRC / "twin").rglob("*.py"))
    known = {module_name(p) for p in files}
    graph = {module_name(p): runtime_imports(p, known) - {module_name(p)} for p in files}

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cycle = " -> ".join([*stack[stack.index(node) :], node])
            raise AssertionError(f"import cycle: {cycle}")
        state[node] = 1
        stack.append(node)
        for dependency in sorted(graph[node]):
            visit(dependency)
        stack.pop()
        state[node] = 2

    for node in sorted(graph):
        visit(node)


def test_low_level_modules_do_not_depend_on_the_application_layers() -> None:
    files = sorted((SRC / "twin").rglob("*.py"))
    known = {module_name(p) for p in files}
    forbidden_for_storage = {
        "twin.app",
        "twin.cli",
        "twin.services",
        "twin.ops.jobs",
        "twin.ops.process_model",
    }
    for path in files:
        name = module_name(path)
        if name.startswith("twin.storage.") and not name.endswith(".cli"):
            assert not (runtime_imports(path, known) & forbidden_for_storage), name
    for lowest in ("twin.clock", "twin.storage.ids", "twin.storage.crypto", "twin.ops.winapi"):
        path = SRC / (lowest.replace(".", "/") + ".py")
        higher = {
            m
            for m in runtime_imports(path, known)
            if m.startswith(("twin.ops", "twin.config", "twin.app"))
        }
        assert higher <= ({"twin.ops.winapi"} if lowest == "twin.ops.winapi" else set()), lowest
