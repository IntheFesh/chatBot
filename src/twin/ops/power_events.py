"""Noticing that the machine slept: ``PowerEventMonitor`` (R-SCH-005).

The computer is set not to sleep (:mod:`twin.ops.power`), but it can still be put to sleep by
hand, by closing the lid, or by an update.  After the machine wakes the bot has to forget the
plans it made for a time that has passed: candidate messages are void, the day plan is checked
and the channel connects afresh.  This component notices the wake-up in two independent ways and
calls one handler:

``WM_POWERBROADCAST`` (Windows)
    a hidden window on a thread of its own receives ``PBT_APMSUSPEND`` and
    ``PBT_APMRESUMEAUTOMATIC`` / ``PBT_APMRESUMESUSPEND`` from the system.  The window is a
    top-level window that is never shown (message-only windows do not receive broadcasts).  The
    Windows calls are written against :class:`~twin.ops.winapi.Win32`.

A clock jump (all platforms)
    every ``schedule.power_tick_s`` seconds the difference between the wall clock and the
    monotonic clock is compared with the previous tick.  While the machine sleeps the wall clock
    goes on and the monotonic clock does not, so the difference grows by the time it slept; a
    change of more than ``schedule.power_jump_ticks`` ticks counts as a suspend that went by.
    The same check notices a wall clock that was set by hand.

Both paths end in :meth:`PowerEventMonitor.resumed`; two reports of one wake-up (the window message
and the next tick) are one event.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from twin.app import ComponentHealth, HealthStatus, TaskSupervisor
from twin.clock import Clock
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger
from twin.ops.winapi import (
    PBT_APMRESUMEAUTOMATIC,
    PBT_APMRESUMESUSPEND,
    PBT_APMSUSPEND,
    WM_POWERBROADCAST,
    Win32,
    load_win32,
)

log = get_logger("twin.power_events")

DEDUP_MIN_S = 30.0
WINDOW_START_TIMEOUT_S = 5.0
WINDOW_STOP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class PowerEvent:
    """The machine woke up (``resume``) after ``gap_s`` seconds away."""

    kind: Literal["resume"]
    source: Literal["wm_powerbroadcast", "clock_jump"]
    at: datetime
    gap_s: float


ResumeHandler = Callable[[PowerEvent], Awaitable[None]]


def clock_drift(
    wall_before: datetime, wall_after: datetime, mono_before: float, mono_after: float
) -> float:
    """Seconds the wall clock moved beyond the monotonic clock between two ticks."""
    return (wall_after - wall_before).total_seconds() - (mono_after - mono_before)


class PowerEventMonitor:
    """Detects waking from sleep and calls ``on_resume`` once per wake-up."""

    name = "power_events"
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        clock: Clock,
        on_resume: ResumeHandler,
        *,
        tick_s: float = 5.0,
        jump_ticks: int = 2,
        platform: str | None = None,
        win32: Win32 | None = None,
        alerts: AlertSink | None = None,
    ) -> None:
        self._clock = clock
        self._on_resume = on_resume
        self._tick_s = tick_s
        self._jump_ticks = jump_ticks
        self._platform = sys.platform if platform is None else platform
        self._win32 = win32
        self._alerts = alerts
        self._supervisor = TaskSupervisor(self.name, clock, alerts)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._hwnd: int | None = None
        self._window_error: str | None = None
        self._suspended_at: datetime | None = None
        self._last_resume: float | None = None
        self.resumes = 0
        self.last_event: PowerEvent | None = None

    # ------------------------------------------------------------------ detection

    @property
    def window_handle(self) -> int | None:
        """The handle of the hidden window (Windows only), once it exists."""
        return self._hwnd

    @property
    def threshold_s(self) -> float:
        return self._jump_ticks * self._tick_s

    async def resumed(
        self, source: Literal["wm_powerbroadcast", "clock_jump"], gap_s: float
    ) -> bool:
        """Report a wake-up; a second report of the same wake-up is ignored.  True if handled."""
        mono = self._clock.monotonic()
        window = max(DEDUP_MIN_S, 4 * self._tick_s)
        if self._last_resume is not None and mono - self._last_resume < window:
            return False
        self._last_resume = mono
        event = PowerEvent("resume", source, self._clock.now_utc(), max(0.0, gap_s))
        self.resumes += 1
        self.last_event = event
        log.info("machine_resumed", source=source, gap_s=round(event.gap_s))
        try:
            await self._on_resume(event)
        except Exception as exc:  # the window thread has nobody to raise to
            log.exception("resume_handler_failed", source=source)
            if self._alerts is not None:
                self._alerts.raise_alert(
                    "power_resume_failed",
                    "the wake-up flow failed; check the log",
                    severity="warning",
                    detail={"source": source, "error": type(exc).__name__},
                    dedup_key="power_resume_failed",
                )
        return True

    async def _watch(self) -> None:
        wall, mono = self._clock.now_utc(), self._clock.monotonic()
        while True:
            await self._clock.sleep(self._tick_s)
            now_wall, now_mono = self._clock.now_utc(), self._clock.monotonic()
            drift = clock_drift(wall, now_wall, mono, now_mono)
            wall, mono = now_wall, now_mono
            if abs(drift) > self.threshold_s:
                await self.resumed("clock_jump", drift)

    # --------------------------------------------------------- the Windows window

    def _on_message(self, message: int, wparam: int, _lparam: int) -> bool:
        """Runs on the window thread: hand the broadcast over to the event loop."""
        if message != WM_POWERBROADCAST:
            return False
        loop = self._loop
        if wparam == PBT_APMSUSPEND:
            self._suspended_at = self._clock.now_utc()  # noted at once: the system is going down
        elif wparam in (PBT_APMRESUMEAUTOMATIC, PBT_APMRESUMESUSPEND) and loop is not None:
            asyncio.run_coroutine_threadsafe(self._resumed_by_window(), loop)
        return True

    async def _resumed_by_window(self) -> None:
        gap = 0.0
        if self._suspended_at is not None:
            gap = (self._clock.now_utc() - self._suspended_at).total_seconds()
        self._suspended_at = None
        await self.resumed("wm_powerbroadcast", gap)

    def _window_thread(self, win32: Win32) -> None:
        try:
            hwnd = win32.create_message_window(self._on_message)
        except Exception as exc:
            self._window_error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
            return
        if hwnd is None:
            self._window_error = "the hidden window could not be created"
            self._ready.set()
            return
        self._hwnd = hwnd
        self._ready.set()
        try:
            win32.run_message_loop()
        finally:
            self._hwnd = None

    def _start_window(self) -> None:
        if self._platform != "win32":
            return
        win32 = self._win32 or load_win32()
        self._win32 = win32
        self._thread = threading.Thread(
            target=self._window_thread, args=(win32,), name="twin-power-window", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=WINDOW_START_TIMEOUT_S) or self._window_error:
            log.warning(
                "power_window_unavailable",
                reason=self._window_error or "timed out",
                fallback="clock",
            )

    def _stop_window(self) -> None:
        thread, hwnd, win32 = self._thread, self._hwnd, self._win32
        if thread is None:
            return
        if hwnd is not None and win32 is not None:
            win32.post_close(hwnd)
        thread.join(timeout=WINDOW_STOP_TIMEOUT_S)
        self._thread = None

    # --------------------------------------------------------------- Component

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._start_window)
        self._supervisor.spawn("clock_jump", self._watch, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()
        await asyncio.to_thread(self._stop_window)

    def health(self) -> ComponentHealth:
        supervised = self._supervisor.health()
        if supervised.status is not HealthStatus.OK:
            return supervised
        if self._platform == "win32" and self._window_error:
            return ComponentHealth(
                HealthStatus.DEGRADED,
                f"wake-up messages are not received ({self._window_error}); using the clock check",
            )
        return supervised
