"""Alembic migrations and the start-up schema check (R-STO-001, R-STO-006, task C.6)."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from twin.storage import migrate
from twin.storage.migrate import SchemaOutdatedError, SchemaState
from twin.storage.models import Base

ROOT = Path(__file__).resolve().parents[2]


def table_names(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        connection.close()
    return {name for (name,) in rows} - {"alembic_version"}


def test_round_00_migration_creates_exactly_the_five_tables(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path)
    assert table_names(path) == {"settings", "jobs", "cost_ledger", "alerts", "channel_state"}
    assert migrate.revision_history()[:2] == ["0001", "0002"]


def test_round_01_migration_adds_ledger_accounts_and_keeps_old_rows(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0001")
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO cost_ledger (id, at, provider, model, purpose, cache_hit_tokens, "
        "cache_miss_tokens, completion_tokens, cost_usd, peak, thinking, latency_ms, "
        "created_at, updated_at) VALUES ('a', '2026-10-09 00:00:00', 'deepseek', 'm', 'reply', "
        "0, 1, 1, 0.5, 1, 0, 10, '2026-10-09 00:00:00', '2026-10-09 00:00:00')"
    )
    connection.commit()
    connection.close()
    migrate.upgrade(path)
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT account, batch_id, reasoning_tokens, image_count FROM cost_ledger"
        ).fetchone()
        assert row == ("daily", None, 0, 0)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE cost_ledger SET account = 'monthly' WHERE id = 'a'")
    finally:
        connection.close()
    migrate.downgrade(path, "0001")
    connection = sqlite3.connect(path)
    try:
        columns = {r[1] for r in connection.execute("PRAGMA table_info(cost_ledger)")}
    finally:
        connection.close()
    assert "account" not in columns and "batch_id" not in columns


def test_migration_and_models_are_in_sync(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path)
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection, opts={"compare_type": True})
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()


def test_every_table_has_utc_timestamps_and_documented_columns() -> None:
    for table in Base.metadata.sorted_tables:
        assert {"created_at", "updated_at"} <= set(table.columns.keys()), table.name
    jobs = Base.metadata.tables["jobs"].columns.keys()
    for column in (
        "id",
        "type",
        "payload",
        "priority",
        "status",
        "attempts",
        "max_attempts",
        "run_after",
        "offpeak_only",
        "deadline",
        "last_error",
        "created_at",
        "updated_at",
        "batch_id",
        "estimated_cost_usd",
        "requires_approval",
        "approved_at",
    ):
        assert column in jobs


def test_downgrade_removes_the_tables_and_upgrade_is_repeatable(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path)
    migrate.downgrade(path, "base")
    assert table_names(path) == set()
    migrate.upgrade(path)
    migrate.upgrade(path)  # already current: nothing happens
    assert len(table_names(path)) == 5


def test_status_transitions(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    missing = migrate.schema_status(path)
    assert missing.state is SchemaState.MISSING and not missing.ok
    assert "twin db upgrade" in missing.hint()

    sqlite3.connect(path).close()
    empty = migrate.schema_status(path)
    assert empty.state is SchemaState.EMPTY and "twin db upgrade" in empty.hint()

    migrate.upgrade(path)
    current = migrate.schema_status(path)
    assert current.ok and current.current == current.head == migrate.head_revision()
    assert "up to date" in current.hint()
    assert migrate.require_current_schema(path).ok

    connection = sqlite3.connect(path)
    connection.execute("UPDATE alembic_version SET version_num = '9999'")
    connection.commit()
    connection.close()
    ahead = migrate.schema_status(path)
    assert ahead.state is SchemaState.AHEAD and "newer than this program" in ahead.hint()
    with pytest.raises(SchemaOutdatedError, match="newer than this program"):
        migrate.require_current_schema(path)


def test_startup_check_tells_the_user_how_to_migrate(tmp_path: Path) -> None:
    with pytest.raises(SchemaOutdatedError) as info:
        migrate.require_current_schema(tmp_path / "none.db")
    assert "twin db upgrade" in str(info.value)
    assert info.value.status.state is SchemaState.MISSING


def test_outdated_state_is_reported_when_revisions_are_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "o.db"
    migrate.upgrade(path)
    monkeypatch.setattr(migrate, "head_revision", lambda: "9998")
    monkeypatch.setattr(migrate, "revision_history", lambda: ["0001", "0002", "9998"])
    status = migrate.schema_status(path)
    assert status.state is SchemaState.OUTDATED and "older than 9998" in status.hint()


def test_alembic_command_line_works_from_the_repository_root(tmp_path: Path) -> None:
    """`uv run alembic upgrade head` (CLAUDE.md section 5) finds the database via the settings."""
    home = tmp_path / "proj"
    home.mkdir()
    env = {**os.environ, "TWIN_HOME": str(home), "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ROOT / "alembic.ini"), "upgrade", "head"],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert table_names(home / "data" / "twin.db") >= {"settings", "jobs"}
