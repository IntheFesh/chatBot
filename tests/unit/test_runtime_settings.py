"""Runtime settings in the database (R-CFG-003)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import TypeAdapter
from sqlalchemy import text

from tests.support.clock import ManualClock
from twin.config.loader import load_settings
from twin.config.runtime import (
    BACKEND_ACTIVE,
    BOT_TIMEZONE,
    PAUSED,
    PROACTIVE_DAILY_MAX,
    THINKING_CHAT,
    RuntimeSettings,
    SettingSpec,
    SettingValueError,
    register_setting,
    registered_settings,
)
from twin.storage.db import Database
from twin.storage.models import Setting
from twin.storage.settings_store import (
    MAX_HISTORY,
    delete_setting,
    get_history,
    get_setting,
    has_setting,
    put_setting,
)
from twin.storage.state import read_state_version


@pytest.fixture
def runtime(db: Database, clock: ManualClock) -> RuntimeSettings:
    return RuntimeSettings(db, load_settings(), clock)


def test_reads_fall_back_to_the_configured_value_before_initialisation(runtime: RuntimeSettings) -> None:
    assert runtime.get(BOT_TIMEZONE) == "America/Chicago"
    assert runtime.get(PAUSED) is False


def test_initialize_seeds_from_config_once(db: Database, clock: ManualClock) -> None:
    first = RuntimeSettings(db, load_settings(None, {"time": {"bot_timezone": "Asia/Shanghai"}}), clock)
    seeded = first.initialize()
    assert set(seeded) == set(registered_settings())
    assert first.get(BOT_TIMEZONE) == "Asia/Shanghai"
    assert first.initialize() == []  # idempotent
    # the config file only supplies the *initial* value: later config edits do not override
    later = RuntimeSettings(db, load_settings(None, {"time": {"bot_timezone": "America/Denver"}}), clock)
    assert later.get(BOT_TIMEZONE) == "Asia/Shanghai"
    with db.session() as session:
        assert read_state_version(session) == 0  # seeding is not a user change


def test_set_validates_and_persists_across_instances(
    runtime: RuntimeSettings, db: Database, clock: ManualClock
) -> None:
    assert runtime.set(BOT_TIMEZONE, "Asia/Shanghai", by="cli") is True
    assert runtime.set(BOT_TIMEZONE, "Asia/Shanghai") is False  # unchanged
    assert RuntimeSettings(db, load_settings(), clock).get(BOT_TIMEZONE) == "Asia/Shanghai"
    runtime.set(THINKING_CHAT, "auto")
    runtime.set(BACKEND_ACTIVE, "hybrid")
    runtime.set(PAUSED, True)
    runtime.set(PROACTIVE_DAILY_MAX, 3)
    assert runtime.snapshot()["thinking.chat"] == "auto"
    assert runtime.get(PROACTIVE_DAILY_MAX) == 3


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        (BOT_TIMEZONE, "Mars/Olympus"),
        (BOT_TIMEZONE, ""),
        (THINKING_CHAT, "sometimes"),
        (BACKEND_ACTIVE, "gpt"),
        (PAUSED, "maybe"),
        (PROACTIVE_DAILY_MAX, -1),
    ],
)
def test_invalid_values_are_rejected_and_not_stored(
    runtime: RuntimeSettings, spec: SettingSpec[object], value: object
) -> None:
    before = runtime.get(spec)
    with pytest.raises(SettingValueError, match=spec.key):
        runtime.set(spec, value)
    assert runtime.get(spec) == before


def test_every_change_keeps_who_when_old_and_new(runtime: RuntimeSettings, clock: ManualClock) -> None:
    runtime.set(BOT_TIMEZONE, "Asia/Shanghai", by="cli")
    clock.tick(3600)
    runtime.set(BOT_TIMEZONE, "America/Chicago", by="wechat:/时区")
    history = runtime.history(BOT_TIMEZONE)
    assert [(h.by, h.old, h.new) for h in history] == [
        ("cli", None, "Asia/Shanghai"),
        ("wechat:/时区", "Asia/Shanghai", "America/Chicago"),
    ]
    assert history[0].created and not history[1].created
    assert history[1].at > history[0].at
    assert history[1].at.startswith("2026-10-09T13:00:00")  # UTC, from the injected clock


def test_history_values_are_encrypted_on_disk(runtime: RuntimeSettings, db: Database) -> None:
    runtime.set(BOT_TIMEZONE, "Asia/Shanghai")
    runtime.set(BOT_TIMEZONE, "Pacific/Auckland")
    with db.session() as session:
        rows = session.execute(text("SELECT value, history FROM settings")).all()
    blob = b"".join(bytes(part) for row in rows for part in row if part)
    assert b"Asia/Shanghai" not in blob and b"Pacific/Auckland" not in blob


def test_set_bumps_the_state_version_in_the_same_transaction(
    runtime: RuntimeSettings, db: Database
) -> None:
    runtime.set(PAUSED, True)
    with db.session() as session:
        assert read_state_version(session) == 1
    runtime.set(PAUSED, True)  # unchanged: nothing to tell the application about
    runtime.set(PAUSED, False)
    with db.session() as session:
        assert read_state_version(session) == 2


def test_a_corrupted_stored_value_is_reported_clearly(runtime: RuntimeSettings, db: Database) -> None:
    with db.transaction(bump_state=False) as session:
        put_setting(session, THINKING_CHAT.key, "bogus", clock=db.clock)
    with pytest.raises(SettingValueError, match="stored value of thinking.chat"):
        runtime.get(THINKING_CHAT)


def test_register_setting_rejects_conflicting_keys_but_allows_reregistering_the_same_spec() -> None:
    assert register_setting(PAUSED) is PAUSED
    clash = SettingSpec("paused", TypeAdapter(bool), lambda s: True, "other")
    with pytest.raises(ValueError, match="already registered"):
        register_setting(clash)


# -------------------------------------------------------------- settings_store


def test_settings_store_basics(db: Database, clock: ManualClock) -> None:
    with db.transaction(bump_state=False) as session:
        assert put_setting(session, "k", {"a": 1}, clock=clock) is True
        assert put_setting(session, "k", {"a": 1}, clock=clock) is False
        assert put_setting(session, "k", {"a": 2}, clock=clock, by="tester") is True
        assert put_setting(session, "quiet", 1, clock=clock, record_history=False)
    with db.session() as session:
        assert get_setting(session, "k") == {"a": 2}
        assert get_setting(session, "missing", "dflt") == "dflt"
        assert has_setting(session, "k") and not has_setting(session, "missing")
        assert [h["by"] for h in get_history(session, "k")] == ["system", "tester"]
        assert get_history(session, "quiet") == [] and get_history(session, "missing") == []
    with db.transaction(bump_state=False) as session:
        assert delete_setting(session, "k") is True
        assert delete_setting(session, "k") is False
        assert session.get(Setting, "k") is None


def test_history_is_capped(db: Database, clock: ManualClock) -> None:
    with db.transaction(bump_state=False) as session:
        for value in range(MAX_HISTORY + 25):
            clock.tick(1)
            put_setting(session, "busy", value, clock=clock)
    with db.session() as session:
        history = get_history(session, "busy")
    assert len(history) == MAX_HISTORY
    assert history[-1]["new"] == MAX_HISTORY + 24
    assert history[-1]["at"] > history[0]["at"]
    assert timedelta(0) < clock.now_utc() - clock.now_utc().replace(year=2026, month=1)
