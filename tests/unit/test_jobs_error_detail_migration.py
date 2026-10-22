"""R-STO-006, R-STO-002 (round 17 item 44): the migration ``0018_jobs_error_detail``.

One new nullable column, ``jobs.last_error_detail`` (ciphertext), and the plain ``last_error`` of
the rows of before cut back to the type of the exception (D-714).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.unit.test_ops_migration import columns, insert
from twin.ops.backup.archive import database_key_ids
from twin.storage import migrate

BEFORE = "0017_eval_kinds"
AFTER = "0018_jobs_error_detail"
OLD_MESSAGE = "ValueError: 这是一句曾经被写进明文列的聊天内容"
FIXED_TEXT = "process stopped while the job was running; attempts exhausted"
NEW_STYLE = "ApiError status=503 retryable=true @ deepseek.py:612:call <- jobs.py:700:execute"


def add_job(connection: sqlite3.Connection, ident: str, **values: object) -> None:
    fields: dict[str, object] = {"type": "t", "status": "pending", "attempts": 0, "max_attempts": 3}
    insert(connection, "jobs", id=ident, **{**fields, **values})


def error_texts(path: Path) -> dict[str, str | None]:
    connection = sqlite3.connect(path)
    try:
        return dict(connection.execute("SELECT id, last_error FROM jobs").fetchall())
    finally:
        connection.close()


def old_database(tmp_path: Path) -> Path:
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    connection = sqlite3.connect(path)
    try:
        for ident, status, error in (
            ("A" * 26, "failed", OLD_MESSAGE),
            ("B" * 26, "failed", FIXED_TEXT),
            ("C" * 26, "pending", None),
            ("D" * 26, "pending", NEW_STYLE),
            ("E" * 26, "failed", "RuntimeError"),
        ):
            add_job(connection, ident, status=status, last_error=error)
        connection.commit()
    finally:
        connection.close()
    return path


def test_the_migration_is_in_the_history_after_the_eval_kinds() -> None:
    history = migrate.revision_history()
    assert history.index(AFTER) == history.index(BEFORE) + 1


def test_it_adds_one_nullable_sealed_column_and_changes_nothing_else(tmp_path: Path) -> None:
    path = old_database(tmp_path)
    old = columns(path, "jobs")
    migrate.upgrade(path, AFTER)
    new = columns(path, "jobs")
    assert set(new) - set(old) == {"last_error_detail"}
    kind, not_null, default = new["last_error_detail"]
    assert kind.upper() == "BLOB" and not not_null and default is None
    assert all(new[name] == old[name] for name in old)  # no column of before was touched
    connection = sqlite3.connect(path)
    try:
        values = connection.execute("SELECT last_error_detail FROM jobs").fetchall()
    finally:
        connection.close()
    assert values == [(None,)] * 5  # nothing is invented for the rows of before


def test_a_message_in_the_plain_column_of_before_is_cut_back_to_the_type(tmp_path: Path) -> None:
    path = old_database(tmp_path)
    migrate.upgrade(path, AFTER)
    texts = error_texts(path)
    assert texts["A" * 26] == "ValueError"  # the sentence is gone from the plain column
    assert texts["B" * 26] == FIXED_TEXT  # a fixed text has no ": " and stays as it is
    assert texts["C" * 26] is None
    assert texts["D" * 26] == NEW_STYLE  # what the new code writes is never cut
    assert texts["E" * 26] == "RuntimeError"


def test_the_migration_can_be_undone_and_done_again(tmp_path: Path) -> None:
    path = old_database(tmp_path)
    old = columns(path, "jobs")
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE jobs SET last_error_detail = ? WHERE id = ?", (b"sealed", "A" * 26)
        )
        connection.commit()
    finally:
        connection.close()
    migrate.downgrade(path, BEFORE)
    assert columns(path, "jobs") == old  # the column is gone, the rest as it was
    assert set(error_texts(path)) == {c * 26 for c in "ABCDE"}  # no job was lost
    migrate.upgrade(path, AFTER)
    assert "last_error_detail" in columns(path, "jobs")
    assert error_texts(path)["D" * 26] == NEW_STYLE


def test_the_constraints_and_indexes_of_the_queue_survive_the_downgrade(tmp_path: Path) -> None:
    """Dropping a column recreates the table in SQLite: its checks and indexes must come back."""
    path = old_database(tmp_path)
    migrate.upgrade(path, AFTER)
    migrate.downgrade(path, BEFORE)
    connection = sqlite3.connect(path)
    try:
        indexes = {row[1] for row in connection.execute('PRAGMA index_list("jobs")')}
        assert {"ix_jobs_claim", "ix_jobs_type", "ix_jobs_batch_id"} <= indexes
        with pytest.raises(sqlite3.IntegrityError):
            add_job(connection, "X" * 26, status="sleeping")
        with pytest.raises(sqlite3.IntegrityError):
            add_job(connection, "Y" * 26, attempts=-1)
    finally:
        connection.close()


def test_a_backup_of_the_schema_before_the_column_can_still_be_scanned_for_its_keys(
    tmp_path: Path,
) -> None:
    """The scan of a snapshot looks for the encrypted columns of the models: the older snapshot
    has no ``last_error_detail``, and SQLite reads a double-quoted unknown name as a string."""
    path = old_database(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("UPDATE jobs SET payload = ?", (b"\x01\x00\x00\x00\x07rest",))
        connection.commit()
    finally:
        connection.close()
    assert database_key_ids(path) == {7}  # the key id inside the sealed payload; no TypeError
    migrate.upgrade(path, AFTER)
    assert database_key_ids(path) == {7}
