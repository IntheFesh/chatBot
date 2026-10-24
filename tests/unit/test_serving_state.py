"""The record other processes read about the model's server and tunnel (R-SRV-002, R-SRV-003)."""

from __future__ import annotations

from tests.support.clock import ManualClock
from twin.services import Services
from twin.serving.state import STATE_KEY, ServingStateStore
from twin.storage.settings_store import get_setting, put_setting
from twin.storage.state import read_state_version


def store_of(services: Services) -> ServingStateStore:
    return ServingStateStore(services.db, services.clock)


def test_there_is_no_record_before_anything_is_written(services: Services) -> None:
    assert store_of(services).read() == {}


def test_sections_are_replaced_one_by_one_and_stamped_with_the_time(
    services: Services, clock: ManualClock
) -> None:
    store = store_of(services)
    first = store.update({"server": {"state": "starting"}, "tunnel": {"state": "up"}})
    assert first["server"] == {"state": "starting"} and first["updated"].startswith("2026-10-09")
    clock.tick(60)
    second = store.update({"server": {"state": "ready", "restarts": 1}})
    assert second["server"] == {"state": "ready", "restarts": 1}
    assert second["tunnel"] == {"state": "up"}  # not named: not touched
    assert second["updated"] != first["updated"]
    assert store.read() == second


def test_none_removes_a_section(services: Services) -> None:
    store = store_of(services)
    store.update({"server": {"state": "ready"}, "warmup": {"first_token_ms": 300}})
    after = store.update({"server": None, "unknown": None})
    assert "server" not in after and after["warmup"] == {"first_token_ms": 300}


def test_writing_the_record_does_not_wake_the_application(services: Services) -> None:
    def version() -> int:
        with services.db.session() as session:
            return read_state_version(session)

    before = version()
    store_of(services).update({"server": {"state": "ready"}})
    store_of(services).mark_reminded("2026-10-09")
    assert version() == before  # it is output, not a request


def test_a_record_that_is_not_a_dictionary_reads_as_empty_and_is_replaced(
    services: Services,
) -> None:
    with services.db.transaction(bump_state=False) as session:
        put_setting(session, STATE_KEY, "garbage", clock=services.clock, by="test")
    store = store_of(services)
    assert store.read() == {}
    store.update({"tunnel": {"state": "backoff"}})
    with services.db.session() as session:
        stored = get_setting(session, STATE_KEY, None)
    assert isinstance(stored, dict) and stored["tunnel"] == {"state": "backoff"}


def test_the_day_of_the_last_reminder_is_remembered(services: Services) -> None:
    store = store_of(services)
    assert store.reminded_on() is None
    store.mark_reminded("2026-10-09")
    assert store.reminded_on() == "2026-10-09"
    store.mark_reminded("2026-10-10")
    assert store.reminded_on() == "2026-10-10"
