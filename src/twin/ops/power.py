"""Keeping the machine awake while the bot runs (R-OPS-002).

On Windows :class:`WindowsPowerManager` calls
``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` from a dedicated
thread that stays alive until :meth:`stop` (the setting belongs to the calling
thread, so it must be set and cleared on the same one).  Other platforms get
:class:`NotNeededPowerManager`, a real implementation that records why nothing
needs to be done.  Wake-up handling after a manual sleep is round 08 (R-SCH-005).
"""

from __future__ import annotations

import sys
import threading
from typing import Protocol

from twin.ops.logging import get_logger
from twin.ops.winapi import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, Win32, load_win32

log = get_logger("twin.power")


class PowerError(RuntimeError):
    """The sleep inhibitor could not be installed."""


class PowerManager(Protocol):
    """Prevents system sleep for the lifetime of the application."""

    @property
    def active(self) -> bool: ...

    @property
    def description(self) -> str: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


WINDOWS_DESCRIPTION = "SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)"


def describe_power_strategy(platform: str | None = None) -> str:
    """What the sleep inhibitor does on ``platform`` (no side effects; used by doctor)."""
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        return WINDOWS_DESCRIPTION
    return NotNeededPowerManager(plat).description


class NotNeededPowerManager:
    """Used where the platform needs no sleep inhibitor: records that fact."""

    def __init__(self, platform: str | None = None) -> None:
        self._platform = sys.platform if platform is None else platform
        self._started = False

    @property
    def active(self) -> bool:
        return False

    @property
    def description(self) -> str:
        return f"sleep inhibitor not needed on platform '{self._platform}'"

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True
        log.info("power_manager", status="not_needed", platform=self._platform)

    def stop(self) -> None:
        self._started = False


class WindowsPowerManager:
    """``SetThreadExecutionState`` held by a background thread."""

    def __init__(self, win32: Win32 | None = None) -> None:
        self._win32 = win32 or load_win32()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: str | None = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def description(self) -> str:
        return WINDOWS_DESCRIPTION

    def _hold(self) -> None:
        previous = self._win32.set_thread_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if previous == 0:
            self._error = "SetThreadExecutionState failed"
            self._ready.set()
            return
        self._active = True
        self._ready.set()
        self._stop.wait()
        self._win32.set_thread_execution_state(ES_CONTINUOUS)
        self._active = False

    def start(self) -> None:
        if self._thread is not None:
            raise PowerError("the power manager is already started")
        self._thread = threading.Thread(target=self._hold, name="twin-power", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise PowerError("timed out while installing the sleep inhibitor")
        if self._error:
            raise PowerError(self._error)
        log.info("power_manager", status="inhibiting_sleep")

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=5.0)
        self._thread = None
        log.info("power_manager", status="released")


def default_power_manager(platform: str | None = None) -> PowerManager:
    """The right manager for ``platform`` (default: this machine)."""
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        return WindowsPowerManager()
    return NotNeededPowerManager(plat)
