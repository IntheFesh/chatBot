"""The CI workflows install exactly the locked environment (round 18 item F13, D-908).

``uv sync --frozen`` installs ``uv.lock`` without looking at ``pyproject.toml``: a dependency
added to ``pyproject.toml`` without ``uv lock`` passes CI on the old lock.  ``--locked`` fails
when the two disagree, which is the check the CI is for.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def sync_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if re.search(r"\buv sync\b", line)]


def test_every_install_in_ci_checks_that_the_lock_matches_the_project() -> None:
    """R18-F13: every ``uv sync`` of a workflow is ``--locked`` (before: ``--frozen``)."""
    found = {path.name: sync_lines(path) for path in sorted(WORKFLOWS.glob("*.yml"))}
    assert found.get("ci.yml"), "the main workflow installs the environment"
    for name, lines in found.items():
        for line in lines:
            assert "--locked" in line and "--frozen" not in line, (name, line)


def test_the_ci_document_describes_the_same_install() -> None:
    text = (ROOT / "docs" / "CI.md").read_text(encoding="utf-8")
    assert "uv sync --locked" in text and "uv sync --frozen" not in text
