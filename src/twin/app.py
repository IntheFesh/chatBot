"""Application lifecycle: components, supervised tasks, graceful shutdown.

R-ARCH-001 / R-ARCH-004 / R-OPS-002:

* a :class:`Component` has ``start``, ``stop`` and ``health``; the
  :class:`Application` starts components in dependency order and stops them in
  reverse, and a failing ``stop`` never prevents the others from stopping;
* background tasks of a component run under a :class:`TaskSupervisor`: an
  exception is logged, reported to the alert hook, and the task is restarted
  after an exponential backoff, without affecting other components; too many
  consecutive crashes mark the component unhealthy;
* :class:`ShutdownSignals` turns Ctrl+C / SIGTERM (POSIX) and console control
  events - Ctrl+C, Ctrl+Break, window close, logoff, shutdown (Windows) - into one
  stop event, and on Windows holds the close event open until shutdown finished.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import Any, Protocol

from twin.clock import Clock
from twin.ops.alerts import AlertSink
from twin.ops.jobs import describe_failure
from twin.ops.logging import get_logger
from twin.ops.winapi import (
    CTRL_BREAK_EVENT,
    CTRL_C_EVENT,
    CTRL_CLOSE_EVENT,
    CTRL_LOGOFF_EVENT,
    CTRL_SHUTDOWN_EVENT,
    Win32,
    load_win32,
)

log = get_logger("twin.app")


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class ComponentHealth:
    status: HealthStatus
    detail: str = ""


class Component(Protocol):
    """A unit of the application with a lifecycle."""

    name: str
    depends_on: Sequence[str]

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health(self) -> ComponentHealth: ...


class DependencyError(ValueError):
    """Component registration is inconsistent (duplicate, unknown dependency, cycle)."""


class ComponentStartError(RuntimeError):
    """A component failed to start; already started components were stopped again."""

    def __init__(self, name: str, cause: BaseException) -> None:
        self.component = name
        super().__init__(f"component '{name}' failed to start: {cause}")


# ------------------------------------------------------------ task supervisor


@dataclass
class _TaskState:
    name: str
    task: asyncio.Task[None] | None = None
    consecutive_crashes: int = 0
    total_crashes: int = 0
    running_since: float | None = None
    finished: bool = False
    last_error: str = ""


class TaskSupervisor:
    """Runs background coroutines of one component and restarts them when they crash."""

    def __init__(
        self,
        owner: str,
        clock: Clock,
        alerts: AlertSink | None = None,
        *,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 60.0,
        unhealthy_after: int = 5,
        stable_after_s: float = 60.0,
    ) -> None:
        self._owner = owner
        self._clock = clock
        self._alerts = alerts
        self._base = backoff_base_s
        self._cap = backoff_cap_s
        self._unhealthy_after = unhealthy_after
        self._stable_after = stable_after_s
        self._states: dict[str, _TaskState] = {}
        self._stopping = False

    @property
    def crash_counts(self) -> dict[str, int]:
        return {name: state.total_crashes for name, state in self._states.items()}

    def spawn(
        self,
        name: str,
        factory: Callable[[], Awaitable[None]],
        *,
        restart_on_exit: bool = False,
    ) -> None:
        """Start ``factory()`` as a supervised task named ``name``."""
        if name in self._states and not self._states[name].finished:
            raise ValueError(f"task '{name}' is already running under {self._owner}")
        state = _TaskState(name)
        self._states[name] = state
        state.task = asyncio.create_task(
            self._run(state, factory, restart_on_exit), name=f"{self._owner}:{name}"
        )

    async def _run(
        self, state: _TaskState, factory: Callable[[], Awaitable[None]], restart_on_exit: bool
    ) -> None:
        while not self._stopping:
            state.running_since = self._clock.monotonic()
            try:
                await factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._note_crash(state, describe_failure(exc), unexpected_exit=False)
            else:
                if not restart_on_exit or self._stopping:
                    state.finished = True
                    state.running_since = None
                    return
                self._note_crash(state, "task returned unexpectedly", unexpected_exit=True)
            state.running_since = None
            delay = min(self._cap, self._base * 2 ** (state.consecutive_crashes - 1))
            await self._clock.sleep(delay)

    def _note_crash(self, state: _TaskState, error: str, *, unexpected_exit: bool) -> None:
        ran = (
            self._clock.monotonic() - state.running_since
            if state.running_since is not None
            else 0.0
        )
        if ran >= self._stable_after:
            state.consecutive_crashes = 0
        state.consecutive_crashes += 1
        state.total_crashes += 1
        state.last_error = error
        log.error(
            "task_crashed",
            component=self._owner,
            task=state.name,
            error=error,
            consecutive=state.consecutive_crashes,
            unexpected_exit=unexpected_exit,
        )
        if self._alerts is not None:
            self._alerts.raise_alert(
                "task_crashed",
                f"{self._owner}/{state.name} crashed",
                severity="critical"
                if state.consecutive_crashes >= self._unhealthy_after
                else "warning",
                detail={"component": self._owner, "task": state.name, "error": error},
                dedup_key=f"task_crashed:{self._owner}:{state.name}",
            )

    def health(self) -> ComponentHealth:
        now = self._clock.monotonic()
        worst = ComponentHealth(HealthStatus.OK)
        for state in self._states.values():
            if state.consecutive_crashes == 0:
                continue
            stable_now = (
                state.running_since is not None and now - state.running_since >= self._stable_after
            )
            if state.consecutive_crashes >= self._unhealthy_after and not stable_now:
                return ComponentHealth(
                    HealthStatus.UNHEALTHY,
                    f"{state.name} crashed {state.consecutive_crashes} times in a row: "
                    f"{state.last_error}",
                )
            if not stable_now:
                worst = ComponentHealth(
                    HealthStatus.DEGRADED,
                    f"{state.name} restarted after a crash: {state.last_error}",
                )
        return worst

    async def stop(self, grace_s: float = 0.0) -> None:
        """Stop all tasks: wait up to ``grace_s`` for them to finish, then cancel."""
        self._stopping = True
        tasks = [state.task for state in self._states.values() if state.task is not None]
        if grace_s > 0 and tasks:
            await asyncio.wait(tasks, timeout=grace_s)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------- application


class Application:
    """Owns the components and their start/stop order."""

    def __init__(self, *, stop_timeout_s: float = 30.0) -> None:
        self._components: dict[str, Component] = {}
        self._started: list[Component] = []
        self._stop_timeout_s = stop_timeout_s

    def register(self, component: Component) -> None:
        if component.name in self._components:
            raise DependencyError(f"component '{component.name}' is registered twice")
        self._components[component.name] = component

    @property
    def components(self) -> dict[str, Component]:
        return dict(self._components)

    def start_order(self) -> list[Component]:
        """Dependency order (dependencies first); ties keep registration order."""
        for component in self._components.values():
            for dependency in component.depends_on:
                if dependency not in self._components:
                    raise DependencyError(
                        f"component '{component.name}' depends on unknown component '{dependency}'"
                    )
        ordered: list[Component] = []
        state: dict[str, int] = {}  # 1 = visiting, 2 = done

        def visit(component: Component, trail: tuple[str, ...]) -> None:
            if state.get(component.name) == 2:
                return
            if state.get(component.name) == 1:
                cycle = " -> ".join((*trail, component.name))
                raise DependencyError(f"dependency cycle: {cycle}")
            state[component.name] = 1
            for dependency in component.depends_on:
                visit(self._components[dependency], (*trail, component.name))
            state[component.name] = 2
            ordered.append(component)

        for component in self._components.values():
            visit(component, ())
        return ordered

    async def start(self) -> None:
        for component in self.start_order():
            try:
                await component.start()
            except Exception as exc:
                log.exception("component_start_failed", component=component.name)
                await self.stop()
                raise ComponentStartError(component.name, exc) from exc
            self._started.append(component)
            log.info("component_started", component=component.name)

    async def stop(self) -> None:
        """Stop started components in reverse order; failures are logged, not raised."""
        while self._started:
            component = self._started.pop()
            try:
                await asyncio.wait_for(component.stop(), timeout=self._stop_timeout_s)
            except Exception:
                log.exception("component_stop_failed", component=component.name)
            else:
                log.info("component_stopped", component=component.name)

    def health(self) -> dict[str, ComponentHealth]:
        report: dict[str, ComponentHealth] = {}
        for name, component in self._components.items():
            try:
                report[name] = component.health()
            except Exception as exc:
                report[name] = ComponentHealth(
                    HealthStatus.UNHEALTHY, f"health check failed: {exc}"
                )
        return report

    async def run(
        self,
        stop_event: asyncio.Event,
        *,
        signals: ShutdownSignals | None = None,
    ) -> None:
        """Start, wait for ``stop_event``, then shut down gracefully."""
        if signals is not None:
            signals.install()
        try:
            await self.start()
            log.info("application_running", components=sorted(self._components))
            await stop_event.wait()
            log.info("application_stopping")
        finally:
            await self.stop()
            if signals is not None:
                signals.notify_done()
                signals.uninstall()
        log.info("application_stopped")


# ------------------------------------------------------------------- signals

_BLOCKING_CTRL_EVENTS = frozenset({CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT})
_STOP_CTRL_EVENTS = frozenset({CTRL_C_EVENT, CTRL_BREAK_EVENT}) | _BLOCKING_CTRL_EVENTS
CLOSE_GRACE_S = 4.5  # Windows terminates a process ~5 s after CTRL_CLOSE_EVENT


@dataclass
class ShutdownSignals:
    """Converts OS shutdown requests into setting ``stop_event``."""

    loop: asyncio.AbstractEventLoop
    stop_event: asyncio.Event
    platform: str = field(default_factory=lambda: sys.platform)
    win32: Win32 | None = None
    close_grace_s: float = CLOSE_GRACE_S
    reason: str | None = field(default=None, init=False)
    _done: threading.Event = field(default_factory=threading.Event, init=False)
    _installed: list[Callable[[], object]] = field(default_factory=list, init=False)

    def request_stop(self, reason: str) -> None:
        """Thread-safe: ask the application to stop."""
        if self.reason is None:
            self.reason = reason
        self.loop.call_soon_threadsafe(self.stop_event.set)

    def notify_done(self) -> None:
        """Called when shutdown finished (releases a blocked Windows close event)."""
        self._done.set()

    def install(self) -> None:
        if self.platform == "win32":
            self._install_windows()
        else:
            self._install_posix()

    def uninstall(self) -> None:
        while self._installed:
            self._installed.pop()()

    # -- POSIX ---------------------------------------------------------------

    def _install_posix(self) -> None:
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            self.loop.add_signal_handler(sig, self.request_stop, name)
            self._installed.append(partial(self.loop.remove_signal_handler, sig))

    # -- Windows -------------------------------------------------------------

    def _on_python_signal(self, signum: int, _frame: Any) -> None:
        self.request_stop(signal.Signals(signum).name)

    def _install_windows(self) -> None:
        for name in ("SIGINT", "SIGBREAK", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous = signal.signal(sig, self._on_python_signal)
                self._installed.append(partial(signal.signal, sig, previous))
        win32 = self.win32 or load_win32()
        if win32.set_console_ctrl_handler(self.console_ctrl_handler, True):
            self._installed.append(
                partial(win32.set_console_ctrl_handler, self.console_ctrl_handler, False)
            )

    def console_ctrl_handler(self, ctrl_type: int) -> bool:
        """Windows console control callback (runs on a system thread)."""
        if ctrl_type not in _STOP_CTRL_EVENTS:
            return False
        self.request_stop(f"console_ctrl_{ctrl_type}")
        if ctrl_type in _BLOCKING_CTRL_EVENTS:
            # the process is killed when this returns: wait for the graceful shutdown
            self._done.wait(timeout=self.close_grace_s)
        return True
