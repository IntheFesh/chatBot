"""Starting, stopping and inspecting the service (``twin service``, R-OPS-001).

*Stopping* is done in the order that loses nothing: ``data/locks/supervisor.stop`` is written; the
supervisor sees it within a second, asks ``twin run`` to stop, the engine finishes the bubble it
is sending and saves its state, both processes end and release their instance locks.  Only if the
locks are still held after the grace time is the scheduled task ended by force (``schtasks /End``,
which kills the process).  A ``twin run`` that was started by hand in a terminal has no
supervisor to ask: it is reported, and the person presses Ctrl+C in its window.

*Status* is read from the locks (is the process alive?), the scheduled task (is it registered, as
whom, with which settings?) and the restart history the supervisor keeps.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from twin.ops.instance_lock import LOCK_RUN, LOCK_SUPERVISOR, locks_held_elsewhere
from twin.ops.supervise import STOP_FILE, RestartLog
from twin.ops.taskscheduler import TaskInfo, TaskScheduler, TaskSchedulerError
from twin.ops.winapi import Win32

STEP_S = 0.5
EXTRA_WAIT_S = 15.0
KILL_WAIT_S = 10.0


def stop_file_path(locks_dir: Path) -> Path:
    return locks_dir / STOP_FILE


def running(
    locks_dir: Path, *, platform: str | None = None, win32: Win32 | None = None
) -> tuple[bool, bool]:
    """``(supervisor alive, twin run alive)`` as the instance locks say."""
    held = locks_held_elsewhere(
        locks_dir, (LOCK_SUPERVISOR, LOCK_RUN), platform=platform, win32=win32
    )
    return LOCK_SUPERVISOR in held, LOCK_RUN in held


def default_waiter(seconds: float) -> None:
    """Wait (a thread-event wait: ``time.sleep`` belongs to the clock module)."""
    threading.Event().wait(seconds)


@dataclass(frozen=True)
class StopOutcome:
    """What ``twin service stop`` did."""

    was_running: bool
    graceful: bool = False
    forced: bool = False
    manual_run: bool = False
    still_running: bool = False


def stop_service(
    locks_dir: Path,
    *,
    scheduler: TaskScheduler | None,
    grace_s: float,
    waiter: Callable[[float], None] = default_waiter,
    platform: str | None = None,
    win32: Win32 | None = None,
) -> StopOutcome:
    """Stop the supervisor and ``twin run`` (see the module description)."""
    supervisor, run = running(locks_dir, platform=platform, win32=win32)
    if not supervisor and not run:
        stop_file_path(locks_dir).unlink(missing_ok=True)
        return StopOutcome(was_running=False)
    if not supervisor:
        return StopOutcome(was_running=True, manual_run=True, still_running=True)

    def alive() -> bool:
        return any(running(locks_dir, platform=platform, win32=win32))

    flag = stop_file_path(locks_dir)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("stop\n", encoding="utf-8")
    try:
        for _ in range(int((grace_s + EXTRA_WAIT_S) / STEP_S)):
            if not alive():
                return StopOutcome(was_running=True, graceful=True)
            waiter(STEP_S)
        forced = False
        if scheduler is not None:
            try:
                scheduler.end()
                forced = True
            except TaskSchedulerError:
                forced = False
            for _ in range(int(KILL_WAIT_S / STEP_S)):
                if not alive():
                    return StopOutcome(was_running=True, forced=forced)
                waiter(STEP_S)
        return StopOutcome(was_running=True, forced=forced, still_running=alive())
    finally:
        flag.unlink(missing_ok=True)


@dataclass(frozen=True)
class RestartSummary:
    last_24h: int
    total: int
    last_at: datetime | None
    last_reason: str | None


def summarize_restarts(log: RestartLog, now: datetime) -> RestartSummary:
    """How often the supervisor restarted ``twin run``."""
    entries = log.entries()
    cutoff = now - timedelta(hours=24)
    stamps = [datetime.fromisoformat(str(item["at"])) for item in entries if item.get("at")]
    recent = sum(1 for stamp in stamps if stamp >= cutoff)
    last = entries[-1] if entries else None
    return RestartSummary(
        last_24h=recent,
        total=len(entries),
        last_at=stamps[-1] if stamps else None,
        last_reason=str(last["reason"]) if last and last.get("reason") else None,
    )


@dataclass(frozen=True)
class ServiceStatus:
    """Everything ``twin service status`` shows."""

    task: TaskInfo | None
    task_state: str | None
    task_error: str | None
    supervisor_running: bool
    run_running: bool
    restarts: RestartSummary | None
    session: dict[str, object] | None


def read_status(
    locks_dir: Path,
    scheduler: TaskScheduler | None,
    restart_log: RestartLog | None,
    now: datetime,
    *,
    platform: str | None = None,
    win32: Win32 | None = None,
) -> ServiceStatus:
    supervisor, run = running(locks_dir, platform=platform, win32=win32)
    task = state = error = None
    if scheduler is not None:
        try:
            task = scheduler.info()
            state = scheduler.state() if task.registered else None
        except TaskSchedulerError as exc:
            error = str(exc)
    return ServiceStatus(
        task=task,
        task_state=state,
        task_error=error,
        supervisor_running=supervisor,
        run_running=run,
        restarts=summarize_restarts(restart_log, now) if restart_log is not None else None,
        session=restart_log.session() if restart_log is not None else None,
    )
