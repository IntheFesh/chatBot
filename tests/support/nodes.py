"""Check that the tests an audit cites exist, without importing them.

The audit tests of round 16 (outbound contacts, privacy, ...) pin a requirement to tests written
earlier.  If one of those is renamed or deleted the audit must notice.  ``pytest --collect-only``
is the authority (``scripts/trace_check.py`` and ``scripts/decisions_check.py`` ask it in CI); in a
unit test the cited modules are read as syntax trees instead, which needs no imports and costs
milliseconds.  A cited name counts if the module defines it (a function, or ``Class.method``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tests.support.scripts import load_script

ROOT = Path(__file__).resolve().parents[2]
_check = load_script("decisions_check")


def node_exists(node: str, root: Path = ROOT) -> bool:
    """Does ``tests/unit/test_x.py::name`` (or ``::Class::name``) exist in the file?"""
    file, _, rest = node.partition("::")
    path = root / file
    if not rest or not path.is_file():
        return False
    symbol = rest.split("[")[0].replace("::", ".")
    return symbol in _check.defined_names(path.read_text(encoding="utf-8"))


def missing_nodes(nodes: Iterable[str], root: Path = ROOT) -> list[str]:
    """The cited tests that do not exist."""
    return [node for node in nodes if not node_exists(node, root)]
