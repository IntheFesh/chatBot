"""Clock abstraction, ManualClock and ULID generation (task H, R-NFR-005 foundation)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tests.support.clock import ManualClock
from twin.clock import (
    Clock,
    SystemClock,
    ensure_aware,
    from_epoch,
    get_clock,
    now_utc,
    to_epoch,
    use_clock,
)
from twin.storage.ids import is_valid_id, new_id


def test_system_clock_returns_aware_utc_and_monotonic_time() -> None:
    clock = SystemClock()
    first = clock.now_utc()
    assert first.tzinfo is UTC
    assert clock.monotonic() <= clock.monotonic()
    assert isinstance(clock, Clock)


async def test_system_clock_sleep_never_negative() -> None:
    await SystemClock().sleep(-5)  # must return immediately instead of raising


def test_use_clock_swaps_and_restores_active_clock(clock: ManualClock) -> None:
    other = ManualClock(datetime(2030, 1, 1, tzinfo=UTC))
    with use_clock(other):
        assert get_clock() is other
        assert now_utc().year == 2030
    assert get_clock() is clock


def test_ensure_aware_rejects_naive_and_converts_offsets() -> None:
    with pytest.raises(ValueError, match="naive"):
        ensure_aware(datetime(2026, 1, 1, 12, 0))  # noqa: DTZ001 - the naive value is the test
    shanghai = timezone(timedelta(hours=8))
    assert ensure_aware(datetime(2026, 1, 1, 8, 0, tzinfo=shanghai)) == datetime(
        2026, 1, 1, 0, 0, tzinfo=UTC
    )


def test_epoch_round_trip() -> None:
    moment = datetime(2026, 10, 9, 1, 2, 3, tzinfo=UTC)
    assert from_epoch(to_epoch(moment)) == moment


def test_manual_clock_requires_aware_start() -> None:
    with pytest.raises(ValueError, match="aware"):
        ManualClock(datetime(2026, 1, 1))  # noqa: DTZ001


async def test_manual_clock_wakes_sleepers_in_order(clock: ManualClock) -> None:
    woke: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        woke.append(name)

    tasks = [
        asyncio.create_task(sleeper("late", 30)),
        asyncio.create_task(sleeper("early", 10)),
    ]
    await clock.settle()
    assert clock.pending_sleepers == 2
    start = clock.now_utc()
    await clock.advance(15)
    assert woke == ["early"]
    await clock.advance(20)
    assert woke == ["early", "late"]
    assert clock.now_utc() == start + timedelta(seconds=35)
    await asyncio.gather(*tasks)


async def test_manual_clock_set_time_and_tick(clock: ManualClock) -> None:
    before = clock.monotonic()
    clock.tick(5)
    assert clock.monotonic() == before + 5
    clock.set_time(clock.now_utc() + timedelta(hours=1))
    assert clock.monotonic() == before + 5 + 3600
    await clock.sleep(0)  # zero sleeps yield but do not wait


def test_ulid_is_valid_sortable_and_unique(clock: ManualClock) -> None:
    ids = []
    for _ in range(50):
        ids.append(new_id())
    assert len(set(ids)) == 50
    assert ids == sorted(ids)  # strictly monotonic within one millisecond too
    assert all(is_valid_id(item) for item in ids)
    clock.tick(1)
    assert new_id() > ids[-1]


def test_ulid_stays_monotonic_when_clock_goes_backwards(clock: ManualClock) -> None:
    first = new_id()
    clock.set_time(clock.now_utc() - timedelta(hours=1))
    assert new_id() > first


def test_is_valid_id_rejects_garbage() -> None:
    assert not is_valid_id("not-an-id")
    assert not is_valid_id("Z" * 26)  # first character beyond the 48-bit time range
    assert not is_valid_id("0" * 25 + "U")  # 'U' is not in the Crockford alphabet
