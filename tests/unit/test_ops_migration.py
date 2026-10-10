"""R-STO-006 (round 12): the migration of the operations tables, ``0015_ops_tables``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.unit.test_migrations import table_names
from twin.storage import migrate

BEFORE = "0013_preference_pairs"
AFTER = "0015_ops_tables"


def columns(path: Path, table: str) -> dict[str, tuple[str, bool, str | None]]:
    """``{name: (type, not null, default)}`` of a table."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    finally:
        connection.close()
    return {str(r[1]): (str(r[2]), bool(r[3]), r[4]) for r in rows}


def insert(connection: sqlite3.Connection, table: str, **values: object) -> None:
    """Insert a row, filling every other column that must have a value with a placeholder."""
    info = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    row: dict[str, object] = dict(values)
    for _, name, kind, not_null, default, _pk in info:
        if name in row or default is not None or not not_null:
            continue
        upper = str(kind).upper()
        if "INT" in upper or "BOOL" in upper:
            row[name] = 0
        elif "DATE" in upper:
            row[name] = "2026-01-01 00:00:00"
        elif "JSON" in upper:
            row[name] = "{}"
        elif "BLOB" in upper:
            row[name] = b""
        elif "FLOAT" in upper or "REAL" in upper:
            row[name] = 0.0
        else:
            row[name] = "x"
    names = ", ".join(f'"{n}"' for n in row)
    marks = ", ".join("?" for _ in row)
    connection.execute(f'INSERT INTO "{table}" ({names}) VALUES ({marks})', list(row.values()))


def test_the_migration_is_in_the_history_after_the_preference_pairs() -> None:
    history = migrate.revision_history()
    assert AFTER in history and history.index(AFTER) > history.index(BEFORE)


def test_it_adds_the_two_tables_and_the_delivery_columns_of_alerts(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    before = table_names(path)
    old = columns(path, "alerts")
    connection = sqlite3.connect(path)
    insert(
        connection, "alerts", id="A" * 26, category="old_category", severity="warning", title="t"
    )
    connection.commit()
    connection.close()

    migrate.upgrade(path, AFTER)
    assert table_names(path) - before == {"health_snapshots", "backup_records"}
    new = columns(path, "alerts")
    assert set(new) - set(old) == {
        "kind", "suppressed", "toast_state", "mail_state", "toast_at", "mail_at",
        "mail_attempts", "mail_next_at", "mail_error", "claimed_at",
    }  # fmt: skip
    assert all(new[name] == old[name] for name in old)  # nothing that was there changed
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT category, kind, suppressed, toast_state, mail_state, mail_attempts FROM alerts"
        ).fetchone()
        # the alert of before is kept, as an alert that nobody has been told about
        assert row == ("old_category", "alert", 0, "none", "none", 0)
    finally:
        connection.close()


def test_the_closed_vocabularies_are_enforced_by_the_database(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        insert(connection, "health_snapshots", id="H" * 26, status="ok", launch="task", checks="{}")
        for bad in (
            {"status": "great", "launch": "task"},
            {"status": "ok", "launch": "cron"},
        ):
            with pytest.raises(sqlite3.IntegrityError):
                insert(connection, "health_snapshots", id="I" * 26, checks="{}", **bad)
        insert(connection, "backup_records", id="B" * 26, kind="pre_restore", status="ok")
        for bad in ({"kind": "weekly", "status": "ok"}, {"kind": "daily", "status": "lost"}):
            with pytest.raises(sqlite3.IntegrityError):
                insert(connection, "backup_records", id="C" * 26, **bad)
        insert(connection, "alerts", id="D" * 26, category="c", severity="info", title="t")
        for bad in ({"kind": "note"}, {"toast_state": "shown"}, {"mail_state": "queued"}):
            with pytest.raises(sqlite3.IntegrityError):
                insert(
                    connection,
                    "alerts",
                    id="E" * 26,
                    category="c",
                    severity="info",
                    title="t",
                    **bad,
                )
    finally:
        connection.close()


def test_a_stability_run_can_be_stored_and_the_other_kinds_still_can(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        for kind in ("blind", "memory", "style", "gate", "stability"):
            insert(connection, "eval_runs", id=f"R{kind}".ljust(26, "0"), kind=kind, status="done")
        with pytest.raises(sqlite3.IntegrityError):
            insert(connection, "eval_runs", id="Z" * 26, kind="bogus", status="done")
    finally:
        connection.close()
    assert columns(path, "eval_runs")["kind"][0].upper().startswith("VARCHAR(12)")


def test_a_kind_another_round_added_is_kept(tmp_path: Path) -> None:
    """The constraint is read from the table, not written down: round 10 may add its own kind."""
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    connection = sqlite3.connect(path)
    sql = str(
        connection.execute("SELECT sql FROM sqlite_master WHERE name = 'eval_runs'").fetchone()[0]
    )
    assert "'gate'" in sql
    connection.execute("PRAGMA writable_schema = ON")
    connection.execute(
        "UPDATE sqlite_master SET sql = ? WHERE name = 'eval_runs'",
        (sql.replace("'gate'", "'gate', 'proactive'"),),
    )
    connection.commit()
    connection.close()

    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        for kind in ("proactive", "stability", "gate"):
            insert(connection, "eval_runs", id=f"K{kind}".ljust(26, "0"), kind=kind, status="done")
        connection.commit()
    finally:
        connection.close()

    migrate.downgrade(path, BEFORE)
    connection = sqlite3.connect(path)
    try:
        kinds = {r[0] for r in connection.execute("SELECT kind FROM eval_runs")}
        assert kinds == {"proactive", "gate"}  # only the stability runs went with the migration
        insert(connection, "eval_runs", id="Y" * 26, kind="proactive", status="done")
        with pytest.raises(sqlite3.IntegrityError):
            insert(connection, "eval_runs", id="W" * 26, kind="stability", status="done")
    finally:
        connection.close()


def test_the_migration_can_be_undone_and_done_again(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    old_alerts = columns(path, "alerts")
    old_eval = columns(path, "eval_runs")
    old_tables = table_names(path)
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    insert(
        connection, "alerts", id="A" * 26, category="c", severity="info", title="t", kind="recovery"
    )
    insert(connection, "eval_runs", id="S" * 26, kind="stability", status="done")
    connection.commit()
    connection.close()

    migrate.downgrade(path, BEFORE)
    assert table_names(path) == old_tables and columns(path, "alerts") == old_alerts
    assert columns(path, "eval_runs") == old_eval
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1  # kept
        assert connection.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 0
    finally:
        connection.close()
    migrate.upgrade(path, AFTER)
    assert "health_snapshots" in table_names(path)
    assert migrate.current_revision(path) == AFTER
