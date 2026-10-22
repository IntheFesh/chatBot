"""R-STO-006 (round 17): the migration ``0019_sticker_first_frame`` - one column, one index.

``stickers.first_frame_sha256`` holds the digest of the first-frame PNG of a GIF sticker
(R-STK-007, D-697).  The migration only adds the column and its index; the rows that exist stay as
they are (no first frame until the backfill makes one), and the way down removes both again.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.unit.test_migrations import table_names
from tests.unit.test_ops_migration import columns, insert
from twin.storage import migrate

AFTER = "0019_sticker_first_frame"
INDEX = "ix_stickers_first_frame_sha256"


def before() -> str:
    """The revision this one follows (whatever batch of the round put its own in between)."""
    history = migrate.revision_history()
    return history[history.index(AFTER) - 1]


def indexes(path: Path, table: str) -> dict[str, list[str]]:
    connection = sqlite3.connect(path)
    try:
        found: dict[str, list[str]] = {}
        for _, name, *_rest in connection.execute(f'PRAGMA index_list("{table}")').fetchall():
            info = connection.execute(f'PRAGMA index_info("{name}")').fetchall()
            found[str(name)] = [str(row[2]) for row in info]
        return found
    finally:
        connection.close()


def test_the_revision_is_in_the_one_linear_history_after_the_eval_kinds() -> None:
    history = migrate.revision_history()
    assert len(history) == len(set(history))  # one chain, no branch
    assert AFTER in history and history.index(AFTER) > history.index("0017_eval_kinds")
    assert migrate.head_revision() in history


def test_it_adds_the_column_and_the_index_and_nothing_else(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, before())
    tables, old_columns, old_indexes = (
        table_names(path),
        columns(path, "stickers"),
        indexes(path, "stickers"),
    )
    assert "first_frame_sha256" not in old_columns
    migrate.upgrade(path, AFTER)
    new_columns = columns(path, "stickers")
    assert table_names(path) == tables
    assert set(new_columns) - set(old_columns) == {"first_frame_sha256"}
    kind, not_null, default = new_columns["first_frame_sha256"]
    assert kind.upper().startswith("VARCHAR(64)") and not not_null and default is None
    assert {name: new_columns[name] for name in old_columns} == old_columns
    new_indexes = indexes(path, "stickers")
    assert set(new_indexes) - set(old_indexes) == {INDEX}
    assert new_indexes[INDEX] == ["first_frame_sha256"]
    assert new_indexes["ix_stickers_sha256"] == ["sha256"]  # the neighbours are untouched


def test_the_rows_that_exist_have_no_first_frame_and_keep_everything_else(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, before())
    connection = sqlite3.connect(path)
    try:
        insert(connection, "stickers", md5="a" * 32, status="available", sha256="b" * 64)
        connection.commit()
    finally:
        connection.close()
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT md5, status, sha256, first_frame_sha256 FROM stickers"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("a" * 32, "available", "b" * 64, None)


def test_the_way_down_removes_the_column_and_the_index_and_keeps_the_rows(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        insert(
            connection,
            "stickers",
            md5="c" * 32,
            status="available",
            sha256="d" * 64,
            first_frame_sha256="e" * 64,
        )
        connection.commit()
    finally:
        connection.close()
    migrate.downgrade(path, before())
    assert "first_frame_sha256" not in columns(path, "stickers")
    assert INDEX not in indexes(path, "stickers")
    assert "ix_stickers_sha256" in indexes(path, "stickers")
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT md5, sha256 FROM stickers").fetchall() == [
            ("c" * 32, "d" * 64)
        ]
    finally:
        connection.close()
    migrate.upgrade(path, AFTER)  # and up again
    assert columns(path, "stickers")["first_frame_sha256"][0].upper().startswith("VARCHAR(64)")
