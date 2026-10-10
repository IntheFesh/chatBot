"""The tables R-STO-006 names exist: in the models and in a database migrated to the head.

The list is read from the SPEC itself, so a table added to the requirement without a model and a
migration fails here, and so does a migration that forgets one.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from twin import storage  # noqa: F401 - importing the package registers every model
from twin.storage import migrate
from twin.storage.models import Base

SPEC = Path(__file__).resolve().parents[2] / "docs" / "SPEC.md"
TABLES_TEXT = re.compile(r"\*\*R-STO-006\*\*[^：]*：(?P<tables>.*?)。各表由")


def spec_tables() -> list[str]:
    found = TABLES_TEXT.search(SPEC.read_text(encoding="utf-8"))
    assert found is not None, "R-STO-006 not found in the SPEC"
    return re.findall(r"`([a-z_]+)`", found.group("tables"))


def test_the_spec_names_the_tables_it_is_expected_to() -> None:
    names = spec_tables()
    assert len(names) == len(set(names)) >= 37
    # the tables of the rounds that share the requirement, named once each
    for table in ("proactive_candidates", "proactive_log", "ratings", "preference_pairs"):
        assert table in names
    for table in ("health_snapshots", "backup_records", "eval_runs", "eval_items"):
        assert table in names
    for table in ("consistency_findings", "consistency_fixes"):  # round 15
        assert table in names


@pytest.mark.parametrize("table", spec_tables())
def test_every_table_of_the_spec_is_a_model(table: str) -> None:
    assert table in Base.metadata.tables


def test_a_database_migrated_to_the_head_has_every_table_of_the_spec(tmp_path: Path) -> None:
    path = tmp_path / "head.db"
    migrate.upgrade(path)
    connection = sqlite3.connect(path)
    try:
        present = {
            name
            for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()
    assert set(spec_tables()) <= present
