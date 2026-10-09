"""Application lifecycle, supervised tasks and shutdown signals (R-ARCH-001, R-ARCH-004, R-OPS-002)."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from collections.abc import Sequence
from typing import Any

import pytest

from tests.support.clock import ManualClock
from tests.support.waiting import wait_until
from tests.support.win32 import FakeWin32
from twin.app import (
    Application,
    ComponentHealth,
    ComponentStartError,
    DependencyError,
    HealthStatus,
    ShutdownSignals,
    TaskSupervisor,
)
from twin.ops.alerts import DbAlertSink
from twin.ops.winapi import (
    CTRL_BREAK_EVENT,
    CTRL_C_EVENT,
    CTRL_CLOSE_EVENT,
    CTRL_LOGOFF_EVENT,
    CTRL_SHUTDOWN_EVENT,
)
from twin.storage.db import Database
from twin.storage.models import Alert


class Recorder:
    """Component double that records the order of lifecycle calls."""

    def __init__(
        self,
        name: str,
        log: list[str],
        depends_on: Sequence[str] = (),
        *,
        fail_start: bool = False,
        fail_stop: bool = False,
    ) -> None:
        self.name = name
        self.depends_on = depends_on
        self.log = log
        self.fail_start = fail_start
        self.fail_stop = fail_stop

    async def start(self) -> None:
        self.log.append(f"start {self.name}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} cannot start")

    async def stop(self) -> None:
        self.log.append(f"stop {self.name}")
        if self.fail_stop:
            raise RuntimeError(f"{self.name} cannot stop")

    def health(self) -> ComponentHealth:
        return ComponentHealth(HealthStatus.OK)


# -------------------------------------------------------------- application


async def test_components_start_in_dependency_order_and_stop_in_reverse() -> None:
    log: list[str] = []
    app = Application()
    app.register(Recorder("channel", log, ["engine"]))
    app.register(Recorder("engine", log, ["storage"]))
    app.register(Recorder("storage", log))
    app.register(Recorder("heartbeat", log))
    assert [c.name for c in app.start_order()] == ["storage", "engine", "channel", "heartbeat"]
    await app.start()
    await app.stop()
    assert log == [
        "start storage", "start engine", "start channel", "start heartbeat",
        "stop heartbeat", "stop channel", "stop engine", "stop storage",
    ]
    await app.stop()  # idempotent
    assert len(log) == 8


async def test_a_failing_start_rolls_back_the_components_already_started() -> None:
    log: list[str] = []
    app = Application()
    app.register(Recorder("a", log))
    app.register(Recorder("b", log, ["a"], fail_start=True))
    app.register(Recorder("c", log, ["b"]))
    with pytest.raises(ComponentStartError, match="'b' failed to start") as info:
        await app.start()
    assert info.value.component == "b"
    assert log == ["start a", "start b", "stop a"]


async def test_a_failing_stop_does_not_prevent_the_others_from_stopping() -> None:
    log: list[str] = []
    app = Application()
    app.register(Recorder("a", log))
    app.register(Recorder("b", log, ["a"], fail_stop=True))
    app.register(Recorder("c", log, ["b"]))
    await app.start()
    await app.stop()
    assert log[3:] == ["stop c", "stop b", "stop a"]


def test_dependency_errors_are_detected() -> None:
    log: list[str] = []
    duplicate = Application()
    duplicate.register(Recorder("a", log))
    with pytest.raises(DependencyError, match="twice"):
        duplicate.register(Recorder("a", log))

    unknown = Application()
    unknown.register(Recorder("a", log, ["ghost"]))
    with pytest.raises(DependencyError, match="unknown component 'ghost'"):
        unknown.start_order()

    cyclic = Application()
    cyclic.register(Recorder("a", log, ["b"]))
    cyclic.register(Recorder("b", log, ["c"]))
    cyclic.register(Recorder("c", log, ["a"]))
    with pytest.raises(DependencyError, match="cycle"):
        cyclic.start_order()


async def test_health_report_survives_a_crashing_health_check() -> None:
    class BadHealth(Recorder):
        def health(self) -> ComponentHealth:
            raise RuntimeError("probe failed")

    app = Application()
    app.register(Recorder("good", []))
    app.register(BadHealth("bad", []))
    report = app.health()
    assert report["good"].status is HealthStatus.OK
    assert report["bad"].status is HealthStatus.UNHEALTHY and "probe failed" in report["bad"].detail
    assert set(app.components) == {"good", "bad"}


async def test_run_waits_for_the_stop_event_then_shuts_down() -> None:
    log: list[str] = []
    app = Application()
    app.register(Recorder("a", log))
    stop = asyncio.Event()
    runner = asyncio.create_task(app.run(stop))
    await wait_until(lambda: log == ["start a"])
    assert not runner.done()
    stop.set()
    await asyncio.wait_for(runner, timeout=5)
    assert log == ["start a", "stop a"]


# ---------------------------------------------------------- task supervisor


async def test_a_crashing_task_is_restarted_with_exponential_backoff_and_alerts(
    db: Database, clock: ManualClock
) -> None:
    alerts = DbAlertSink(db, clock)
    supervisor = TaskSupervisor("demo", clock, alerts, backoff_base_s=1, backoff_cap_s=60)
    attempts: list[int] = []
    healthy = asyncio.Event()

    async def flaky() -> None:
        attempts.append(len(attempts) + 1)
        if len(attempts) <= 3:
            raise RuntimeError("transient failure")
        healthy.set()
        await asyncio.sleep(3600)

    supervisor.spawn("worker", flaky)
    await clock.settle()
    assert attempts == [1] and supervisor.crash_counts == {"worker": 1}
    assert supervisor.health().status is HealthStatus.DEGRADED
    await clock.advance(0.9)
    assert attempts == [1]  # still backing off (1 s)
    await clock.advance(0.2)
    assert attempts == [1, 2]
    await clock.advance(1.7)  # t = 2.8: the second backoff (2 s, from t = 1.0) is not over
    assert attempts == [1, 2]
    await clock.advance(0.4)  # t = 3.2
    assert attempts == [1, 2, 3]
    await clock.advance(4.0)  # third backoff is 4 s (t = 3.0 -> 7.0)
    await wait_until(healthy.is_set)
    assert clock.sleeps[:3] == [1, 2, 4]
    assert supervisor.crash_counts == {"worker": 3}
    with db.session() as session:
        rows = session.query(Alert).filter(Alert.category == "task_crashed").all()
        assert len(rows) == 3 and rows[0].detail["task"] == "worker"  # type: ignore[index]
    await supervisor.stop()


async def test_backoff_is_capped_and_repeated_crashes_mark_the_component_unhealthy(
    clock: ManualClock,
) -> None:
    supervisor = TaskSupervisor("demo", clock, None, backoff_base_s=10, backoff_cap_s=30, unhealthy_after=4)

    async def always_fails() -> None:
        raise RuntimeError("permanent failure")

    supervisor.spawn("bad", always_fails)
    for _ in range(6):
        await clock.advance(31)
    assert clock.sleeps[:5] == [10, 20, 30, 30, 30]
    health = supervisor.health()
    assert health.status is HealthStatus.UNHEALTHY
    assert "crashed" in health.detail and "permanent failure" in health.detail
    await supervisor.stop()


async def test_a_task_that_runs_stably_clears_its_crash_streak(clock: ManualClock) -> None:
    supervisor = TaskSupervisor("demo", clock, None, backoff_base_s=1, unhealthy_after=2, stable_after_s=60)
    runs = 0
    hold = asyncio.Event()

    async def sometimes() -> None:
        nonlocal runs
        runs += 1
        if runs == 1:
            raise RuntimeError("first run fails")
        await hold.wait()

    supervisor.spawn("t", sometimes)
    await clock.advance(2)
    assert runs == 2
    assert supervisor.health().status is HealthStatus.DEGRADED  # recovering
    await clock.advance(120)  # the second run is now stable
    assert supervisor.health().status is HealthStatus.OK
    await supervisor.stop()


async def test_restart_on_exit_and_normal_completion(clock: ManualClock) -> None:
    supervisor = TaskSupervisor("demo", clock, None, backoff_base_s=1)
    returns = 0

    async def returns_early() -> None:
        nonlocal returns
        returns += 1

    supervisor.spawn("loop", returns_early, restart_on_exit=True)
    await clock.advance(1.5)
    assert returns >= 2  # restarted after returning unexpectedly
    assert supervisor.crash_counts["loop"] >= 2

    once: list[str] = []

    async def one_shot() -> None:
        once.append("ran")

    supervisor.spawn("once", one_shot)
    await clock.settle()
    await clock.advance(10)
    assert once == ["ran"]
    supervisor.spawn("once", one_shot)  # finished tasks may be spawned again
    await clock.settle()
    assert once == ["ran", "ran"]
    await supervisor.stop()


async def test_spawning_a_running_task_twice_is_an_error(clock: ManualClock) -> None:
    supervisor = TaskSupervisor("demo", clock)

    async def idle() -> None:
        await asyncio.sleep(3600)

    supervisor.spawn("t", idle)
    with pytest.raises(ValueError, match="already running"):
        supervisor.spawn("t", idle)
    await supervisor.stop()


async def test_a_crash_in_one_component_does_not_disturb_another(clock: ManualClock) -> None:
    healthy_ticks = 0
    crashes = 0

    async def healthy_loop() -> None:
        nonlocal healthy_ticks
        while True:
            healthy_ticks += 1
            await clock.sleep(1)

    async def crashing_loop() -> None:
        nonlocal crashes
        crashes += 1
        raise RuntimeError("component B is broken")

    good = TaskSupervisor("good", clock)
    bad = TaskSupervisor("bad", clock, backoff_base_s=1, backoff_cap_s=2)
    good.spawn("loop", healthy_loop)
    bad.spawn("loop", crashing_loop)
    await clock.advance(10)
    assert healthy_ticks == 11  # kept running throughout
    assert crashes >= 4
    assert good.health().status is HealthStatus.OK
    assert bad.health().status is not HealthStatus.OK
    await good.stop()
    await bad.stop()


async def test_stop_waits_for_the_grace_period_before_cancelling(clock: ManualClock) -> None:
    supervisor = TaskSupervisor("demo", clock)
    finished = asyncio.Event()

    async def finishes_soon() -> None:
        await asyncio.sleep(0.05)
        finished.set()

    async def never() -> None:
        await asyncio.sleep(3600)

    supervisor.spawn("a", finishes_soon)
    supervisor.spawn("b", never)
    await clock.settle()
    await supervisor.stop(grace_s=2)
    assert finished.is_set()


# ----------------------------------------------------------------- signals


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handling")
async def test_posix_signals_set_the_stop_event() -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    signals = ShutdownSignals(loop, stop)
    signals.install()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(stop.wait(), timeout=5)
        assert signals.reason == "SIGTERM"
    finally:
        signals.uninstall()
    stop.clear()
    signals.install()
    try:
        os.kill(os.getpid(), signal.SIGINT)  # Ctrl+C: no KeyboardInterrupt, a graceful stop
        await asyncio.wait_for(stop.wait(), timeout=5)
    finally:
        signals.uninstall()


async def test_application_run_installs_and_removes_handlers() -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    signals = ShutdownSignals(loop, stop, platform="win32", win32=FakeWin32())
    app = Application()
    app.register(Recorder("a", []))
    previous = signal.getsignal(signal.SIGINT)
    runner = asyncio.create_task(app.run(stop, signals=signals))
    await wait_until(lambda: signal.getsignal(signal.SIGINT) != previous)
    signals.request_stop("test")
    await asyncio.wait_for(runner, timeout=5)
    assert signal.getsignal(signal.SIGINT) == previous
    assert signals._done.is_set()


async def test_windows_console_events_request_a_stop() -> None:
    loop = asyncio.get_running_loop()
    fake = FakeWin32()
    stop = asyncio.Event()
    signals = ShutdownSignals(loop, stop, platform="win32", win32=fake, close_grace_s=0.5)
    signals.install()
    try:
        assert fake.ctrl_handlers == [(signals.console_ctrl_handler, True)]
        assert signals.console_ctrl_handler(99) is False  # not a stop event
        assert not stop.is_set()
        for event in (CTRL_C_EVENT, CTRL_BREAK_EVENT):
            stop.clear()
            assert await asyncio.to_thread(signals.console_ctrl_handler, event) is True
            await asyncio.wait_for(stop.wait(), timeout=5)
        assert signals.reason == "console_ctrl_0"
    finally:
        signals.uninstall()
    assert fake.ctrl_handlers[-1] == (signals.console_ctrl_handler, False)


@pytest.mark.parametrize("event", [CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT])
async def test_windows_close_events_block_until_shutdown_finished(event: int) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    signals = ShutdownSignals(loop, stop, platform="win32", win32=FakeWin32(), close_grace_s=5)
    result: dict[str, Any] = {}

    def handler_thread() -> None:
        result["value"] = signals.console_ctrl_handler(event)
        result["returned"] = True

    thread = threading.Thread(target=handler_thread)
    thread.start()
    await asyncio.wait_for(stop.wait(), timeout=5)  # the stop request went through
    await asyncio.sleep(0.1)
    assert "returned" not in result  # Windows would kill the process on return: keep waiting
    signals.notify_done()
    thread.join(timeout=5)
    assert result == {"value": True, "returned": True}
