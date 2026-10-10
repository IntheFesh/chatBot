"""``twin supervise``: keeps ``twin run`` alive (R-OPS-001).

The scheduled task starts the supervisor when the user logs on.  The supervisor starts
``twin run`` in a child process (same Python environment) and:

* if the child **exits with a non-zero code** (a crash, a failed start-up, ``kill``), waits and
  starts it again: 5 seconds, then 10, 20, ... never more than 5 minutes (``ops.supervise``); a
  child that then ran for 30 minutes counts as stable and the next wait is 5 seconds again.  Every
  restart is logged, written to the restart history (``settings`` key ``supervise.history``, which
  the stability report reads) and reported as the alert ``process_restarted`` (the alert service
  rate-limits it);
* if the child **exits with code 0** (it was stopped on purpose), ends too: a normal exit is not
  restarted;
* on a **stop request** (Ctrl+C, ``SIGTERM``, the window being closed, or the file
  ``data/locks/supervisor.stop`` that ``twin service stop`` writes) asks the child to stop
  gracefully - it finishes the bubble it is sending and saves its state - waits up to
  ``ops.supervise.stop_grace_s`` and only then kills it.  A stop during a wait for a restart ends
  at once.

The supervisor holds the ``supervisor`` instance lock for all of this, also while it waits to
restart: the exclusive commands (restore, purge, rotation, migrations) are refused for as long as
a supervisor lives, which is exactly right - the child comes back in a few seconds.  On Windows the
child is put in a job object so that it is ended if the supervisor is killed.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from twin.clock import Clock
from twin.config.settings import SuperviseConfig
from twin.ops.alerts import AlertSink
from twin.ops.jobobject import ProcessJob
from twin.ops.logging import get_logger
from twin.ops.process_model import ExitCode
from twin.storage.db import Database
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.supervise")

HISTORY_KEY = "supervise.history"
SESSION_KEY = "supervise.session"
STOP_FILE = "supervisor.stop"
MAX_HISTORY = 200
STOP_POLL_S = 1.0
LAUNCH_ENV = "TWIN_LAUNCH"
EXIT_REASONS = {
    int(ExitCode.BUSY): "another instance holds the run lock",
    int(ExitCode.SCHEMA): "the database needs `twin db upgrade`",
    int(ExitCode.SECRETS): "a secret or key is missing",
    int(ExitCode.CONFIG): "the configuration is invalid",
    int(ExitCode.CONSENT): "the consent date is missing",
}


@dataclass(frozen=True)
class Restart:
    """One restart of the child."""

    at: datetime
    exit_code: int | None
    ran_s: float
    delay_s: float
    reason: str

    def to_json(self) -> dict[str, object]:
        return {
            "at": self.at.isoformat(),
            "exit_code": self.exit_code,
            "ran_s": round(self.ran_s, 1),
            "delay_s": self.delay_s,
            "reason": self.reason,
        }


def describe_exit(code: int | None) -> str:
    """Why the child ended, in words (a code, never output of the child)."""
    if code is None:
        return "the process ended without an exit code"
    return EXIT_REASONS.get(code) or f"exit code {code}"


class RestartLog:
    """The restart history in the ``settings`` table (the stability report reads it)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(self, restart: Restart) -> None:
        with self._db.transaction(bump_state=False) as session:
            history = list(get_setting(session, HISTORY_KEY) or [])
            history.append(restart.to_json())
            put_setting(
                session,
                HISTORY_KEY,
                history[-MAX_HISTORY:],
                clock=self._clock,
                by="supervise",
                record_history=False,
            )

    def start_session(self, launch: str) -> None:
        with self._db.transaction(bump_state=False) as session:
            put_setting(
                session,
                SESSION_KEY,
                {"started_at": self._clock.now_utc().isoformat(), "launch": launch},
                clock=self._clock,
                by="supervise",
                record_history=False,
            )

    def entries(self) -> list[dict[str, object]]:
        with self._db.session() as session:
            history = get_setting(session, HISTORY_KEY)
        return [dict(item) for item in history] if isinstance(history, list) else []

    def session(self) -> dict[str, object] | None:
        with self._db.session() as session:
            value = get_setting(session, SESSION_KEY)
        return dict(value) if isinstance(value, dict) else None


# ---------------------------------------------------------------- the child process


class Child(Protocol):
    """A started ``twin run``."""

    @property
    def pid(self) -> int | None: ...

    async def wait(self) -> int:
        """Wait for the process to end; returns its exit code."""
        ...

    def request_stop(self) -> None:
        """Ask it to stop gracefully."""
        ...

    def kill(self) -> None: ...


class Launcher(Protocol):
    async def start(self, env: Mapping[str, str]) -> Child: ...


class SubprocessChild:
    """An :mod:`asyncio` subprocess."""

    def __init__(self, process: asyncio.subprocess.Process, platform: str) -> None:
        self._process = process
        self._platform = platform

    @property
    def pid(self) -> int | None:
        return self._process.pid

    async def wait(self) -> int:
        return await self._process.wait()

    def request_stop(self) -> None:
        if self._process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            if self._platform == "win32":
                # the child has a process group of its own, so only it receives the event
                self._process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
            else:
                self._process.terminate()

    def kill(self) -> None:
        if self._process.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                self._process.kill()


class SubprocessLauncher:
    """Starts ``args`` as a child process of this one (no shell)."""

    def __init__(
        self, args: Sequence[str], *, cwd: Path | None = None, platform: str | None = None
    ):
        self._args = list(args)
        self._cwd = cwd
        self._platform = sys.platform if platform is None else platform

    async def start(self, env: Mapping[str, str]) -> Child:
        flags = 0
        if self._platform == "win32":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = await asyncio.create_subprocess_exec(
            *self._args, env=dict(env), cwd=self._cwd, creationflags=flags
        )
        return SubprocessChild(process, self._platform)


def child_arguments(global_options: Sequence[str]) -> list[str]:
    """The command line of ``twin run`` in this Python environment (UTF-8 mode on)."""
    return [sys.executable, "-X", "utf8", "-m", "twin", *global_options, "run"]


# -------------------------------------------------------------------------- the loop


class Supervisor:
    """Starts the child, restarts it, stops it (see the module description)."""

    def __init__(
        self,
        launcher: Launcher,
        clock: Clock,
        config: SuperviseConfig,
        *,
        env: Mapping[str, str],
        alerts: AlertSink | None = None,
        history: RestartLog | None = None,
        job: ProcessJob | None = None,
        stop_file: Path | None = None,
        launch: str = "manual",
    ) -> None:
        self._launcher = launcher
        self._clock = clock
        self._config = config
        self._env = dict(env)
        self._env[LAUNCH_ENV] = launch
        self._env.setdefault("PYTHONUTF8", "1")
        self._alerts = alerts
        self._history = history
        self._job = job
        self._stop_file = stop_file
        self.restarts = 0
        self.last_exit: int | None = None

    # -- waiting that a stop request interrupts -------------------------------------

    async def _sleep(self, seconds: float, stop: asyncio.Event) -> None:
        sleeper = asyncio.ensure_future(self._clock.sleep(seconds))
        stopper = asyncio.ensure_future(stop.wait())
        try:
            await asyncio.wait({sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sleeper, stopper):
                task.cancel()
            await asyncio.gather(sleeper, stopper, return_exceptions=True)

    async def _watch_stop_file(self, stop: asyncio.Event) -> None:
        path = self._stop_file
        if path is None:
            return
        while not stop.is_set():
            if path.exists():
                log.info("stop_requested", source="stop_file")
                stop.set()
                return
            await self._clock.sleep(STOP_POLL_S)

    async def _stop_child(self, child: Child) -> int | None:
        """Graceful stop, then the hard one after ``stop_grace_s``."""
        child.request_stop()
        try:
            async with asyncio.timeout(self._config.stop_grace_s):
                return await child.wait()
        except TimeoutError:
            log.warning("child_killed", after_s=self._config.stop_grace_s)
            child.kill()
            return await child.wait()

    # -- the main loop ------------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> int:
        """Run until stopped or until the child exits with code 0; returns the exit code."""
        if self._stop_file is not None:
            self._stop_file.unlink(missing_ok=True)  # a request made before this start is stale
        watcher = asyncio.ensure_future(self._watch_stop_file(stop))
        delay = self._config.backoff_start_s
        try:
            while not stop.is_set():
                started = self._clock.monotonic()
                child = await self._launcher.start(self._env)
                if self._job is not None and child.pid is not None:
                    self._job.assign(child.pid)
                log.info("child_started", pid=child.pid)
                code = await self._wait_for(child, stop)
                self.last_exit = code
                ran = self._clock.monotonic() - started
                if stop.is_set():
                    log.info("child_stopped", exit_code=code)
                    break
                if code == 0:
                    log.info("child_exited_normally", ran_s=round(ran))
                    return 0
                if ran >= self._config.stable_after_min * 60:
                    delay = self._config.backoff_start_s
                self._note_restart(code, ran, delay)
                await self._sleep(delay, stop)
                delay = min(delay * 2, self._config.backoff_max_s)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            if self._stop_file is not None:
                self._stop_file.unlink(missing_ok=True)
            if self._job is not None:
                self._job.close()
        return 0

    async def _wait_for(self, child: Child, stop: asyncio.Event) -> int | None:
        """The child's exit code; if a stop is requested first, the child is stopped."""
        exited = asyncio.ensure_future(child.wait())
        stopper = asyncio.ensure_future(stop.wait())
        try:
            await asyncio.wait({exited, stopper}, return_when=asyncio.FIRST_COMPLETED)
            if exited.done():
                return exited.result()
            return await self._stop_child(child)
        finally:
            for task in (exited, stopper):
                if not task.done():
                    task.cancel()
            await asyncio.gather(exited, stopper, return_exceptions=True)

    def _note_restart(self, code: int | None, ran: float, delay: float) -> None:
        self.restarts += 1
        reason = describe_exit(code)
        restart = Restart(self._clock.now_utc(), code, ran, delay, reason)
        log.error("child_exited", exit_code=code, ran_s=round(ran), restart_in_s=delay)
        if self._history is not None:
            try:
                self._history.add(restart)
            except Exception as exc:  # the record is secondary to restarting
                log.warning("restart_not_recorded", reason=type(exc).__name__)
        if self._alerts is not None:
            self._alerts.raise_alert(
                "process_restarted",
                f"twin run ended ({reason}) after {ran:.0f} s; restarting in {delay:.0f} s",
                severity="critical" if ran < 60 and self.restarts >= 5 else "warning",
                detail={"exit_code": code, "ran_s": round(ran), "restart_in_s": round(delay)},
                dedup_key="process_restarted",
            )
