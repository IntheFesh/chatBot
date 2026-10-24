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
    migrate.upgrade(path, "0001")
    assert table_names(path) == {"settings", "jobs", "cost_ledger", "alerts", "channel_state"}
    assert migrate.revision_history()[:2] == ["0001", "0002"]


def test_round_03_migration_adds_the_import_tables_and_keeps_the_old_ones(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0002")
    before = table_names(path)
    migrate.upgrade(path, "0003_import_tables")
    assert table_names(path) - before == {
        "conversations",
        "messages",
        "media_assets",
        "stickers",
        "sticker_uses",
        "import_runs",
    }
    assert migrate.revision_history()[2] == "0003_import_tables"
    migrate.downgrade(path, "0002")
    assert table_names(path) == before


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
    assert table_names(path) == set(Base.metadata.tables)


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
    migrate.upgrade(path, "0002")
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


def test_round_04_migration_adds_the_profile_and_routine_tables(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0003_import_tables")
    before = table_names(path)
    migrate.upgrade(path, "0004_profile_activity_tables")
    assert table_names(path) - before == {
        "profile_versions",
        "activity_models",
        "routine_overrides",
    }
    assert migrate.revision_history()[3] == "0004_profile_activity_tables"
    connection = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO profile_versions (id, scope, reason, input_hash, her_messages, "
                "data_range, metrics, summary_rules, created_at, updated_at) VALUES "
                "('p', 'elsewhere', 'r', 'h', 0, x'00', x'00', x'00', '2026-01-01', '2026-01-01')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO routine_overrides (id, kind, params, enabled, created_at, "
                "updated_at) VALUES ('o', 'nap', x'00', 1, '2026-01-01', '2026-01-01')"
            )
    finally:
        connection.close()
    migrate.downgrade(path, "0003_import_tables")
    assert table_names(path) == before


INSERT_WINDOW = (
    "INSERT INTO example_windows (id, conversation_id, reply_block_ids, context_block_ids, "
    "context_turns, reply_reproducible, reply_at_utc, local_slot, day_type, holdout, signature, "
    "created_at, updated_at) VALUES ('w', 'c', '[]', '[]', 0, 1, '2026-01-01', {slot}, "
    "'{day_type}', 0, 's', '2026-01-01', '2026-01-01')"
)


def test_round_05_migration_adds_the_example_windows_table(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0004_profile_activity_tables")
    before = table_names(path)
    migrate.upgrade(path, "0005_example_windows")
    assert table_names(path) - before == {"example_windows"}
    assert migrate.revision_history()[4] == "0005_example_windows"
    connection = sqlite3.connect(path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(example_windows)")}
        # ids, times and flags only: no column can hold what she wrote
        assert columns == {
            "id",
            "conversation_id",
            "reply_block_ids",
            "context_block_ids",
            "context_turns",
            "reply_reproducible",
            "reply_at_utc",
            "local_slot",
            "day_type",
            "holdout",
            "signature",
            "embed_version",
            "created_at",
            "updated_at",
        }
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(INSERT_WINDOW.format(slot=96, day_type="workday"))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(INSERT_WINDOW.format(slot=5, day_type="someday"))
    finally:
        connection.close()
    migrate.downgrade(path, "0004_profile_activity_tables")
    assert table_names(path) == before


def test_round_06_migration_adds_the_persona_tables_and_the_sticker_columns(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0005_example_windows")
    before = table_names(path)
    connection = sqlite3.connect(path)
    try:
        old_columns = {row[1] for row in connection.execute("PRAGMA table_info(stickers)")}
        connection.execute(
            "INSERT INTO stickers (md5, status, attempts, her_uses, user_uses, created_at, "
            "updated_at) VALUES ('aaaa', 'available', 0, 3, 1, '2026-01-01', '2026-01-01')"
        )
        connection.commit()
    finally:
        connection.close()
    migrate.upgrade(path, "0006_persona_sticker_tags")
    assert table_names(path) - before == {"persona_cards", "prompt_templates"}
    assert migrate.revision_history()[5] == "0006_persona_sticker_tags"
    connection = sqlite3.connect(path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(stickers)")}
        assert columns - old_columns == {
            "vision_tags", "context_tags", "manual_tags", "tags", "description", "use_cases",
            "context_note", "tag_source", "origin", "disabled", "tagged_at", "context_tagged_at",
            "context_cutoff_at", "context_uses", "desc_vector_id", "desc_encoding",
        }  # fmt: skip
        # a sticker of the earlier rounds keeps its data and gets the defaults of the new columns
        row = connection.execute(
            "SELECT her_uses, origin, disabled, context_uses, tags FROM stickers WHERE md5='aaaa'"
        ).fetchone()
        assert row == (3, "import", 0, 0, None)
        assert {r[1] for r in connection.execute("PRAGMA table_info(persona_cards)")} >= {
            "id", "scope", "number", "parent_id", "reason", "profile_version_id",
            "template_version", "described_her_messages", "described_at", "content",
            "provenance", "created_at", "updated_at",
        }  # fmt: skip
        card = (
            "INSERT INTO persona_cards (id, scope, number, reason, content, created_at, updated_at)"
            " VALUES ('{id}', '{scope}', {number}, 'generate', x'00', '2026-01-01', '2026-01-01')"
        )
        connection.execute(card.format(id="c1", scope="live", number=1))
        with pytest.raises(sqlite3.IntegrityError):  # a scope of its own kind only
            connection.execute(card.format(id="c2", scope="elsewhere", number=1))
        with pytest.raises(sqlite3.IntegrityError):  # one number once per scope
            connection.execute(card.format(id="c3", scope="live", number=1))
        connection.execute(card.format(id="c4", scope="pre_holdout", number=1))
        template = (
            "INSERT INTO prompt_templates (id, name, version, content, content_sha256, source, "
            "created_at, updated_at) VALUES ('{id}', 'n', {version}, x'00', 'h', '{source}', "
            "'2026-01-01', '2026-01-01')"
        )
        connection.execute(template.format(id="t1", version=1, source="file"))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(template.format(id="t2", version=1, source="file"))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(template.format(id="t3", version=2, source="web"))
    finally:
        connection.close()
    migrate.downgrade(path, "0005_example_windows")
    assert table_names(path) == before
    connection = sqlite3.connect(path)
    try:
        assert {row[1] for row in connection.execute("PRAGMA table_info(stickers)")} == old_columns
        assert connection.execute("SELECT her_uses FROM stickers WHERE md5='aaaa'").fetchone() == (
            3,
        )
    finally:
        connection.close()
    migrate.upgrade(path, "0006_persona_sticker_tags")  # and up again
    assert table_names(path) - before == {"persona_cards", "prompt_templates"}


def test_round_07_migration_adds_the_memory_tables_and_their_constraints(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0006_persona_sticker_tags")
    before = table_names(path)
    migrate.upgrade(path, "0007_memory_tables")
    assert table_names(path) - before == {
        "facts",
        "daily_summaries",
        "lifeline_events",
        "followups",
        "memory_replay_days",
    }
    assert migrate.revision_history()[6] == "0007_memory_tables"
    connection = sqlite3.connect(path)
    try:
        fact = (
            "INSERT INTO facts (id, rev, number, subject, category, text, source, status, "
            "confidence, importance, known_at, recurrence, created_at, updated_at) VALUES "
            "('{id}', 1, {number}, '{subject}', 'life', x'00', '{source}', 'active', 0.8, "
            "{importance}, '2026-01-01', 'none', '2026-01-01', '2026-01-01')"
        )
        connection.execute(
            fact.format(id="f1", number=1, subject="her", source="real_record", importance=3)
        )
        with pytest.raises(sqlite3.IntegrityError):  # a subject of the closed vocabulary only
            connection.execute(
                fact.format(id="f2", number=2, subject="dog", source="real_record", importance=3)
            )
        with pytest.raises(sqlite3.IntegrityError):  # sources are the four of R-MEM-003
            connection.execute(
                fact.format(id="f3", number=3, subject="her", source="guess", importance=3)
            )
        with pytest.raises(sqlite3.IntegrityError):  # importance is 1 to 5
            connection.execute(
                fact.format(id="f4", number=4, subject="her", source="real_record", importance=6)
            )
        with pytest.raises(sqlite3.IntegrityError):  # the number the user sees is unique
            connection.execute(
                fact.format(id="f5", number=1, subject="her", source="real_record", importance=3)
            )
        summary = (
            "INSERT INTO daily_summaries (id, rev, scope, local_date, timezone, utc_start, "
            "utc_end, text, version, is_current, created_at, updated_at) VALUES ('{id}', 1, "
            "'{scope}', '2026-03-05', 'America/Chicago', '2026-03-05', '2026-03-06', x'00', "
            "{version}, 1, "
            "'2026-01-01', '2026-01-01')"
        )
        connection.execute(summary.format(id="s1", scope="real", version=1))
        connection.execute(summary.format(id="s2", scope="real", version=2))  # a recomputed day
        connection.execute(summary.format(id="s3", scope="bot", version=1))
        with pytest.raises(sqlite3.IntegrityError):  # one row per scope, day and version
            connection.execute(summary.format(id="s4", scope="real", version=1))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(summary.format(id="s5", scope="both", version=1))
    finally:
        connection.close()
    migrate.downgrade(path, "0006_persona_sticker_tags")
    assert table_names(path) == before


def test_round_08_migration_adds_the_plan_and_time_zone_tables(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0007_memory_tables")
    before = table_names(path)
    migrate.upgrade(path, "0008_daily_plans_timezone")
    assert table_names(path) - before == {"daily_plans", "timezone_history"}
    assert migrate.revision_history()[7] == "0008_daily_plans_timezone"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        plan = (
            "INSERT INTO daily_plans (id, local_date, timezone, day_type, plan, seed, "
            "effective_from, ends_at, reason, inputs_hash, created_at, updated_at) VALUES "
            "('{id}', '2026-03-08', 'America/Chicago', '{day_type}', x'00', 7, '{start}', "
            "'{end}', 'daily', 'abc', '2026-03-08', '2026-03-08')"
        )
        connection.execute(
            plan.format(
                id="p1", day_type="weekend", start="2026-03-08 06:00", end="2026-03-09 13:00"
            )
        )
        with pytest.raises(sqlite3.IntegrityError):  # the closed vocabulary of day types
            connection.execute(
                plan.format(id="p2", day_type="feast", start="2026-03-08", end="2026-03-09")
            )
        with pytest.raises(sqlite3.IntegrityError):  # a plan decides a stretch of time
            connection.execute(
                plan.format(id="p3", day_type="weekend", start="2026-03-09", end="2026-03-08")
            )
        change = (
            "INSERT INTO timezone_history (id, changed_at, from_timezone, to_timezone, source, "
            "plan_id, created_at, updated_at) VALUES ('{id}', '2026-03-08', '{old}', '{new}', "
            "'{source}', {plan}, '2026-03-08', '2026-03-08')"
        )
        connection.execute(
            change.format(
                id="t1", old="America/Chicago", new="Asia/Shanghai", source="cli", plan="'p1'"
            )
        )
        with pytest.raises(sqlite3.IntegrityError):  # a switch changes the zone
            connection.execute(
                change.format(
                    id="t2", old="Asia/Shanghai", new="Asia/Shanghai", source="cli", plan="NULL"
                )
            )
        with pytest.raises(sqlite3.IntegrityError):  # the routes of R-SCH-002
            connection.execute(
                change.format(
                    id="t3", old="Asia/Shanghai", new="Europe/Paris", source="web", plan="NULL"
                )
            )
        with pytest.raises(sqlite3.IntegrityError):  # the plan must exist
            connection.execute(
                change.format(
                    id="t4", old="Asia/Shanghai", new="Europe/Paris", source="app", plan="'nope'"
                )
            )
    finally:
        connection.close()
    migrate.downgrade(path, "0007_memory_tables")
    assert table_names(path) == before


def test_round_09_migration_adds_the_conversation_tables_and_their_constraints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0009_training_tables")
    before = table_names(path)
    migrate.upgrade(path, "0010_bot_turns_state_feedback")
    assert table_names(path) - before == {"bot_turns", "conversation_state", "feedback"}
    assert migrate.revision_history()[9] == "0010_bot_turns_state_feedback"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")

    def turn(name: str, **fields: str) -> str:
        values = {
            "direction": "'out'",
            "kind": "'text'",
            "external": "NULL",
            "index": "NULL",
            "backend": "NULL",
        } | fields
        return (
            "INSERT INTO bot_turns (id, at, direction, kind, text, is_command, external_id, "
            f"bubble_index, backend, created_at, updated_at) VALUES ('{name}', '2026-03-08', "
            f"{values['direction']}, {values['kind']}, x'00', 0, {values['external']}, "
            f"{values['index']}, {values['backend']}, '2026-03-08', '2026-03-08')"
        )

    try:
        connection.execute(turn("t1", direction="'in'", external="'ext-1'"))
        for name, bad in (
            ("t2", {"direction": "'sideways'"}),
            ("t3", {"kind": "'poem'"}),
            ("t4", {"index": "-1"}),
            ("t5", {"index": "0", "backend": "'oracle'"}),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(turn(name, **bad))
        # a channel message id is stored once per direction; other rows are free to have none
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(turn("t6", direction="'in'", external="'ext-1'"))
        for name in ("n1", "n2"):
            connection.execute(turn(name, index="0", backend="'deepseek'"))
        # the same id on the other side is another message
        connection.execute(turn("t7", external="'ext-1'", index="0", backend="'deepseek'"))
        state = (
            "INSERT INTO conversation_state (id, state, state_since, created_at, updated_at) "
            "VALUES ('{id}', '{state}', '2026-03-08', '2026-03-08', '2026-03-08')"
        )
        connection.execute(state.format(id="main", state="COLLECTING"))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(state.format(id="other", state="DREAMING"))
        feedback = (
            "INSERT INTO feedback (id, type, reply_id, bot_turn_id, created_at, updated_at) "
            "VALUES ('{id}', '{kind}', 'r1', {turn}, '2026-03-08', '2026-03-08')"
        )
        connection.execute(feedback.format(id="f1", kind="redo", turn="'t7'"))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(feedback.format(id="f2", kind="praise", turn="NULL"))
        with pytest.raises(sqlite3.IntegrityError):  # the rejected bubble must exist
            connection.execute(feedback.format(id="f3", kind="redo", turn="'nope'"))
        connection.execute("DELETE FROM bot_turns WHERE id = 't7'")
        found = connection.execute("SELECT bot_turn_id FROM feedback WHERE id = 'f1'").fetchone()
        assert found == (None,)  # the feedback outlives the row it was about
    finally:
        connection.close()
    migrate.downgrade(path, "0009_training_tables")
    assert table_names(path) == before
