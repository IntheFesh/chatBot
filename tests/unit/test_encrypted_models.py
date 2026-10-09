"""Encrypted columns and the SQLite settings (R-STO-001, R-STO-002, R-PRIV-004)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import StatementError

from tests.support.clock import ManualClock
from twin.storage.crypto import DecryptionError, KeyRing
from twin.storage.db import Database, ReadOnlyViolationError, WritePolicy, use_write_policy
from twin.storage.ids import is_valid_id
from twin.storage.models import Alert, ChannelState, Job, Setting
from twin.storage.types import EncryptedJSON, EncryptedText, UTCDateTime

SECRET_TEXT = "绝密正文-the-quick-brown-fox"


def raw_value(db: Database, sql: str, **params: object) -> bytes:
    with db.session() as session:
        row = session.execute(text(sql), params).fetchone()
    assert row is not None
    return bytes(row[0])


def test_json_round_trip_and_ciphertext_on_disk(db: Database) -> None:
    with db.transaction() as session:
        session.add(Setting(key="greeting", value={"say": SECRET_TEXT, "n": [1, 2, 3]}))
    with db.session() as session:
        row = session.get(Setting, "greeting")
        assert row is not None
        assert row.value == {"say": SECRET_TEXT, "n": [1, 2, 3]}
    blob = raw_value(db, "SELECT value FROM settings WHERE key = 'greeting'")
    assert SECRET_TEXT.encode() not in blob
    assert b"say" not in blob
    assert blob[0] == 1  # version byte
    assert db.path.read_bytes().find(SECRET_TEXT.encode()) == -1
    assert db.path.with_name(db.path.name + "-wal").read_bytes().find(SECRET_TEXT.encode()) == -1


def test_job_payload_is_sealed_and_primary_key_is_a_ulid(db: Database) -> None:
    with db.transaction() as session:
        job = Job(type="demo", payload={"text": SECRET_TEXT})
        session.add(job)
        job_id = job.id
    assert is_valid_id(job_id)
    assert SECRET_TEXT.encode() not in raw_value(db, "SELECT payload FROM jobs")
    with db.session() as session:
        loaded = session.get(Job, job_id)
        assert loaded is not None and loaded.payload == {"text": SECRET_TEXT}


def test_optional_encrypted_field_stores_null(db: Database) -> None:
    with db.transaction() as session:
        alert = Alert(category="x", title="no detail")
        session.add(alert)
        alert_id = alert.id
    with db.session() as session:
        loaded = session.get(Alert, alert_id)
        assert loaded is not None
        assert loaded.detail is None
        assert loaded.detail_ct is None


def test_each_write_of_the_same_value_produces_different_ciphertext(db: Database) -> None:
    with db.transaction() as session:
        session.add(Setting(key="a", value="same"))
        session.add(Setting(key="b", value="same"))
    first = raw_value(db, "SELECT value FROM settings WHERE key = 'a'")
    second = raw_value(db, "SELECT value FROM settings WHERE key = 'b'")
    assert first != second


def test_ciphertext_copied_to_another_row_fails_to_decrypt(db: Database) -> None:
    """AAD binds (table, primary key, column) - R-STO-002."""
    with db.transaction() as session:
        session.add(Setting(key="alpha", value="A"))
        session.add(Setting(key="beta", value="B"))
    with db.transaction() as session:
        session.execute(
            text("UPDATE settings SET value = (SELECT value FROM settings WHERE key='alpha') "
                 "WHERE key='beta'")
        )
    with db.session() as session:
        beta = session.get(Setting, "beta")
        assert beta is not None
        with pytest.raises(DecryptionError, match="authentication failed"):
            _ = beta.value
        alpha = session.get(Setting, "alpha")
        assert alpha is not None and alpha.value == "A"


def test_ciphertext_moved_between_columns_or_tables_fails(db: Database) -> None:
    with db.transaction() as session:
        session.add(Setting(key="same", value="in settings", history=["h"]))
        session.add(ChannelState(key="same", value="in channel"))
    with db.transaction() as session:
        session.execute(text("UPDATE settings SET history = value WHERE key='same'"))
        session.execute(
            text("UPDATE channel_state SET value = (SELECT value FROM settings WHERE key='same')")
        )
    with db.session() as session:
        setting = session.get(Setting, "same")
        channel = session.get(ChannelState, "same")
        assert setting is not None and channel is not None
        with pytest.raises(DecryptionError):
            _ = setting.history  # value blob moved into the history column
        with pytest.raises(DecryptionError):
            _ = channel.value  # blob moved from another table with the same key


def test_plaintext_cannot_be_written_to_an_encrypted_column(db: Database) -> None:
    with pytest.raises(StatementError, match="sealed values"), db.transaction() as session:
        row = Setting(key="raw")
        row.value_ct = b"plain bytes"  # type: ignore[assignment]
        session.add(row)


def test_primary_key_is_required_before_assigning_encrypted_fields(keyring_ring: KeyRing) -> None:
    row = Setting(key="temporary")
    row.key = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="primary key"):
        row.value = "x"


def test_unknown_constructor_argument_is_rejected(keyring_ring: KeyRing) -> None:
    with pytest.raises(TypeError, match="no attribute"):
        Setting(key="k", value=1, nonsense=True)


def test_required_encrypted_field_must_be_set(db: Database) -> None:
    row = Setting(key="unset")
    with pytest.raises(ValueError, match="has not been set"):
        _ = row.value


def test_column_types_exist_and_declare_blob_storage() -> None:
    from sqlalchemy import LargeBinary

    for column_type in (EncryptedText, EncryptedJSON):
        assert issubclass(column_type, EncryptedText | EncryptedJSON)
        assert isinstance(column_type().impl, LargeBinary)


def test_timestamps_are_utc_and_updated_at_moves(db: Database, clock: ManualClock) -> None:
    with db.transaction() as session:
        session.add(Setting(key="t", value=1))
    with db.session() as session:
        row = session.get(Setting, "t")
        assert row is not None
        assert row.created_at == clock.now_utc() == row.updated_at
        assert row.created_at.tzinfo is UTC
    clock.tick(60)
    with db.transaction() as session:
        row = session.get(Setting, "t")
        assert row is not None
        row.value = 2
    with db.session() as session:
        row = session.get(Setting, "t")
        assert row is not None
        assert row.updated_at == row.created_at + timedelta(seconds=60)


def test_utc_datetime_converts_offsets_and_rejects_naive(db: Database) -> None:
    shanghai = timezone(timedelta(hours=8))
    stamp = datetime(2026, 10, 9, 20, 0, tzinfo=shanghai)
    with db.transaction() as session:
        job = Job(type="t", payload={}, run_after=stamp)
        session.add(job)
        job_id = job.id
    with db.session() as session:
        loaded = session.get(Job, job_id)
        assert loaded is not None
        assert loaded.run_after == datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        assert loaded.run_after.utcoffset() == timedelta(0)
        raw = session.execute(text("SELECT run_after FROM jobs")).scalar_one()
    assert str(raw).startswith("2026-10-09 12:00:00")  # stored as naive UTC
    with pytest.raises(StatementError, match="naive datetime"), db.transaction() as session:
        session.add(Job(type="t", payload={}, run_after=datetime(2026, 1, 1)))  # noqa: DTZ001


def test_utc_datetime_type_passes_none_through() -> None:
    column_type = UTCDateTime()
    assert column_type.process_bind_param(None, None) is None  # type: ignore[arg-type]
    assert column_type.process_result_value(None, None) is None  # type: ignore[arg-type]


def test_sqlite_pragmas(db: Database) -> None:
    with db.session() as session:
        assert session.execute(text("PRAGMA journal_mode")).scalar_one() == "wal"
        assert session.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert session.execute(text("PRAGMA busy_timeout")).scalar_one() >= 5000


def test_write_transactions_roll_back_on_error(db: Database) -> None:
    with pytest.raises(RuntimeError), db.transaction() as session:
        session.add(Setting(key="doomed", value=1))
        session.flush()
        raise RuntimeError("boom")
    with db.session() as session:
        assert session.get(Setting, "doomed") is None


def test_read_only_policy_blocks_writes(db: Database) -> None:
    with use_write_policy(WritePolicy(read_only=True)):
        with pytest.raises(ReadOnlyViolationError, match="READ"):
            with db.transaction():
                pass
        with db.session() as session:  # reading is still allowed
            assert session.get(Setting, "anything") is None


def test_bump_policy_increments_state_version_in_the_same_transaction(db: Database) -> None:
    from twin.storage.state import read_state_version

    with db.session() as session:
        assert read_state_version(session) == 0
    with use_write_policy(WritePolicy(bump_state=True)):
        with db.transaction() as session:
            session.add(Setting(key="x", value=1))
        with db.transaction(bump_state=False) as session:
            session.add(Setting(key="y", value=1))
    with db.session() as session:
        assert read_state_version(session) == 1
    with pytest.raises(RuntimeError), db.transaction(bump_state=True):
        raise RuntimeError("rolled back, so no bump")
    with db.session() as session:
        assert read_state_version(session) == 1


async def test_arun_executes_in_a_worker_thread(db: Database) -> None:
    def write(session: object) -> str:
        from sqlalchemy.orm import Session

        assert isinstance(session, Session)
        session.add(Setting(key="threaded", value="ok"))
        return "written"

    assert await db.arun(write, write=True) == "written"
    assert await db.arun(lambda s: s.get(Setting, "threaded") is not None) is True
