"""Waking from sleep: the clock-jump detector and the ``WM_POWERBROADCAST`` window (R-SCH-005).

Every platform detects a wake-up by the wall clock running ahead of the monotonic clock; Windows
also receives the system's own message in a hidden window.  The Windows calls go through the
``Win32`` protocol, so the logic is tested here with a recording double and, on Windows only, with
a real window.
"""

from __future__ import annotations

import sys
from datetime import timedelta

import pytest

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from tests.support.waiting import wait_until
from tests.support.win32 import FakeWin32
from twin.app import HealthStatus
from twin.ops.power_events import PowerEvent, PowerEventMonitor, clock_drift
from twin.ops.winapi import (
    PBT_APMRESUMEAUTOMATIC,
    PBT_APMRESUMESUSPEND,
    PBT_APMSUSPEND,
    WM_POWERBROADCAST,
)

TICK = 5.0


class Heard:
    def __init__(self) -> None:
        self.events: list[PowerEvent] = []

    async def __call__(self, event: PowerEvent) -> None:
        self.events.append(event)


async def started(clock: ManualClock, heard: Heard, **kwargs: object) -> PowerEventMonitor:
    monitor = PowerEventMonitor(clock, heard, tick_s=TICK, jump_ticks=2, **kwargs)  # type: ignore[arg-type]
    await monitor.start()
    return monitor


# ------------------------------------------------------------- the clock check


def test_the_drift_is_the_wall_clock_beyond_the_monotonic_clock() -> None:
    from datetime import UTC, datetime

    start = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    assert clock_drift(start, start + timedelta(seconds=5), 100.0, 105.0) == 0.0
    assert clock_drift(start, start + timedelta(hours=3, seconds=5), 100.0, 105.0) == 10800.0
    assert clock_drift(start, start - timedelta(hours=1), 100.0, 105.0) == -3605.0


async def test_a_wall_clock_that_jumps_three_hours_ahead_is_a_wake_up(clock: ManualClock) -> None:
    heard = Heard()
    monitor = await started(clock, heard, platform="linux")
    try:
        await wait_until(lambda: clock.pending_sleepers >= 1)
        await clock.advance(TICK)  # an ordinary tick
        assert heard.events == []
        clock.jump_wall(3 * 3600)  # the machine slept; the monotonic clock did not run
        await clock.advance(TICK)
        await wait_until(lambda: len(heard.events) == 1)
        event = heard.events[0]
        assert (event.kind, event.source) == ("resume", "clock_jump")
        assert event.gap_s == pytest.approx(3 * 3600, abs=1)
        assert event.at == clock.now_utc() and monitor.resumes == 1
        await clock.advance(TICK)  # and the next tick is quiet again
        assert len(heard.events) == 1
    finally:
        await monitor.stop()


async def test_a_jump_must_be_more_than_two_ticks_to_count(clock: ManualClock) -> None:
    heard = Heard()
    monitor = await started(clock, heard, platform="linux")
    try:
        await wait_until(lambda: clock.pending_sleepers >= 1)
        assert monitor.threshold_s == 2 * TICK
        clock.jump_wall(2 * TICK - 0.5)  # a hiccup of the scheduler, not a sleep
        await clock.advance(TICK)
        assert heard.events == []
        clock.jump_wall(2 * TICK + 1)
        await clock.advance(TICK)
        await wait_until(lambda: len(heard.events) == 1)
    finally:
        await monitor.stop()


async def test_a_wall_clock_set_back_is_reported_without_a_negative_gap(clock: ManualClock) -> None:
    heard = Heard()
    monitor = await started(clock, heard, platform="linux")
    try:
        await wait_until(lambda: clock.pending_sleepers >= 1)
        clock.jump_wall(-3600)
        await clock.advance(TICK)
        await wait_until(lambda: len(heard.events) == 1)
        assert heard.events[0].gap_s == 0.0 and heard.events[0].source == "clock_jump"
    finally:
        await monitor.stop()


async def test_two_reports_of_one_wake_up_are_one_event(clock: ManualClock) -> None:
    heard = Heard()
    monitor = await started(clock, heard, platform="linux")
    try:
        await wait_until(lambda: clock.pending_sleepers >= 1)
        assert await monitor.resumed("wm_powerbroadcast", 600.0) is True
        assert await monitor.resumed("clock_jump", 600.0) is False  # the same wake-up, a tick later
        clock.jump_wall(600)
        await clock.advance(TICK)  # the detector sees the jump as well
        assert len(heard.events) == 1 and heard.events[0].source == "wm_powerbroadcast"
        await clock.advance(120)  # a while later: another sleep is another event
        clock.jump_wall(900)
        await clock.advance(TICK)
        await wait_until(lambda: len(heard.events) == 2)
        assert heard.events[1].source == "clock_jump"
    finally:
        await monitor.stop()


async def test_only_windows_opens_a_window(clock: ManualClock) -> None:
    fake = FakeWin32()
    monitor = await started(clock, Heard(), platform="linux", win32=fake)
    try:
        assert fake.windows == {} and monitor.window_handle is None
        assert monitor.health().status is HealthStatus.OK
    finally:
        await monitor.stop()


# ------------------------------------------------------ WM_POWERBROADCAST (Windows)


async def test_the_hidden_window_receives_suspend_and_resume(clock: ManualClock) -> None:
    fake = FakeWin32()
    heard = Heard()
    monitor = await started(clock, heard, platform="win32", win32=fake)
    try:
        hwnd = monitor.window_handle
        assert hwnd is not None and hwnd in fake.windows
        assert fake.send_message(hwnd, WM_POWERBROADCAST, PBT_APMSUSPEND, 0) == 1
        assert heard.events == []  # a suspend alone is not an event
        clock.jump_wall(3 * 3600)  # the machine slept for three hours
        assert fake.send_message(hwnd, WM_POWERBROADCAST, PBT_APMRESUMEAUTOMATIC, 0) == 1
        await wait_until(lambda: len(heard.events) == 1)
        event = heard.events[0]
        assert event.source == "wm_powerbroadcast"
        assert event.gap_s == pytest.approx(3 * 3600, abs=1)
        assert fake.send_message(hwnd, 0x0001, 0, 0) == 0  # other messages are not ours
        assert fake.send_message(hwnd, WM_POWERBROADCAST, 0x000A, 0) == 1  # answered, ignored
        assert len(heard.events) == 1
    finally:
        await monitor.stop()
    assert fake.closed == [hwnd] and monitor.window_handle is None


async def test_the_user_waking_the_machine_by_hand_is_the_same_wake_up(clock: ManualClock) -> None:
    fake = FakeWin32()
    heard = Heard()
    monitor = await started(clock, heard, platform="win32", win32=fake)
    try:
        hwnd = monitor.window_handle
        assert hwnd is not None
        fake.send_message(hwnd, WM_POWERBROADCAST, PBT_APMSUSPEND, 0)
        clock.jump_wall(1800)
        fake.send_message(hwnd, WM_POWERBROADCAST, PBT_APMRESUMEAUTOMATIC, 0)
        fake.send_message(hwnd, WM_POWERBROADCAST, PBT_APMRESUMESUSPEND, 0)  # after the key press
        await wait_until(lambda: len(heard.events) >= 1)
        await clock.advance(TICK)  # and the clock check sees it too
        assert len(heard.events) == 1
    finally:
        await monitor.stop()


async def test_a_window_that_cannot_be_made_falls_back_to_the_clock(clock: ManualClock) -> None:
    fake = FakeWin32()
    fake.window_fails = True
    heard = Heard()
    monitor = await started(clock, heard, platform="win32", win32=fake)
    try:
        assert monitor.window_handle is None
        health = monitor.health()
        assert health.status is HealthStatus.DEGRADED and "clock" in health.detail
        await wait_until(lambda: clock.pending_sleepers >= 1)
        clock.jump_wall(7200)
        await clock.advance(TICK)
        await wait_until(lambda: len(heard.events) == 1)
    finally:
        await monitor.stop()


async def test_a_failing_wake_up_flow_is_logged_and_raised_as_an_alert(
    clock: ManualClock,
) -> None:
    fake = FakeWin32()
    alerts = RecordingAlerts()

    async def broken(event: PowerEvent) -> None:
        raise RuntimeError("the wake-up flow failed")

    monitor = PowerEventMonitor(
        clock, broken, tick_s=TICK, platform="win32", win32=fake, alerts=alerts
    )
    await monitor.start()
    try:
        hwnd = monitor.window_handle
        assert hwnd is not None
        fake.send_message(hwnd, WM_POWERBROADCAST, PBT_APMRESUMEAUTOMATIC, 0)
        await wait_until(lambda: alerts.categories() == ["power_resume_failed"])
        assert alerts.alerts[0].detail == {"source": "wm_powerbroadcast", "error": "RuntimeError"}
        assert monitor.resumes == 1 and monitor.health().status is HealthStatus.OK
    finally:
        await monitor.stop()


@pytest.mark.windows
async def test_a_real_hidden_window_receives_a_broadcast_message(clock: ManualClock) -> None:
    """On Windows: post a real WM_POWERBROADCAST to the real hidden window."""
    import asyncio

    from twin.ops.winapi import load_win32

    win32 = load_win32()
    heard = Heard()
    monitor = PowerEventMonitor(clock, heard, tick_s=TICK, platform=sys.platform, win32=win32)
    await monitor.start()
    try:
        hwnd = monitor.window_handle
        assert hwnd is not None
        answer = await asyncio.to_thread(
            win32.send_message, hwnd, WM_POWERBROADCAST, PBT_APMRESUMEAUTOMATIC, 0
        )
        assert answer == 1  # TRUE: the message was dealt with
        await wait_until(lambda: len(heard.events) == 1)
        assert heard.events[0].source == "wm_powerbroadcast"
    finally:
        await monitor.stop()
    assert monitor.window_handle is None
