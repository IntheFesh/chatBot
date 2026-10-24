"""R-STO-006 (round 11): the migration of ``preference_pairs``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.unit.test_migrations import table_names
from twin.storage import migrate


def test_round_11_migration_adds_the_table_with_its_constraints_and_goes_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0012_eval_tables")
    before = table_names(path)
    migrate.upgrade(path, "0013_preference_pairs")
    assert table_names(path) - before == {"preference_pairs"}
    assert "0013_preference_pairs" in migrate.revision_history()
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        columns = {
            row[1]: row[2] for row in connection.execute("PRAGMA table_info(preference_pairs)")
        }
        assert set(columns) == {
            "id",
            "feedback_id",
            "reply_id",
            "prompt_sample",
            "chosen",
            "rejected",
            "source",
            "template_version",
            "persona_version",
            "created_at",
            "updated_at",
        }
        for sealed in ("prompt_sample", "chosen", "rejected"):
            assert columns[sealed] == "BLOB"  # the three texts are stored sealed

        pair = (
            "INSERT INTO preference_pairs (id, feedback_id, reply_id, prompt_sample, chosen, "
            "rejected, source, template_version, persona_version, created_at, updated_at) "
            "VALUES ('{id}', {feedback}, 'r1', x'00', x'00', x'00', '{source}', 't', 'p', "
            "'2026-03-08', '2026-03-08')"
        )
        connection.execute(pair.format(id="p1", feedback="NULL", source="user_correction"))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(pair.format(id="p2", feedback="NULL", source="the_bots_own_words"))
        with pytest.raises(sqlite3.IntegrityError):  # a pair belongs to a feedback that exists
            connection.execute(
                pair.format(id="p3", feedback="'no-such-feedback'", source="user_correction")
            )
        connection.execute(
            "INSERT INTO feedback (id, type, reply_id, created_at, updated_at) "
            "VALUES ('f1', 'not_like', 'r1', '2026-03-08', '2026-03-08')"
        )
        connection.execute(pair.format(id="p4", feedback="'f1'", source="user_correction"))
        connection.execute("DELETE FROM feedback WHERE id = 'f1'")  # the pair outlives its verdict
        assert connection.execute(
            "SELECT feedback_id FROM preference_pairs WHERE id = 'p4'"
        ).fetchone() == (None,)
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(preference_pairs)")}
        assert {"ix_preference_pairs_reply_id", "ix_preference_pairs_created_at"} <= indexes
    finally:
        connection.close()
    migrate.downgrade(path, "0012_eval_tables")
    assert table_names(path) == before


def test_the_table_is_part_of_the_models_the_schema_check_compares() -> None:
    from twin.storage import models

    assert "preference_pairs" in models.Base.metadata.tables
