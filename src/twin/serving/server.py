"""The managed ``llama-server`` process (R-SRV-002, CLAUDE.md section 7).

:class:`LlamaServerManager` keeps one server alive for as long as it is wanted:

* it starts the process with :func:`asyncio.create_subprocess_exec` (an argument list, no shell,
  no console window on Windows), standard output and error going to a log file of their own
  (``data/logs/llama-server.log``, rotated at start-up);
* it waits for ``GET /health`` to say the model is loaded (the server answers 503 while it reads
  the file) and gives up after ``style_model.serve.start_timeout_s``;
* then ``on_ready`` runs - the tokenizer comparison of :mod:`twin.serving.tokencheck`.  A hook that
  raises :class:`ServerBlocked` stops the server and keeps it stopped (a model whose tokens differ
  from the training must not be served);
* it keeps asking ``/health``; a server that dies, or stops answering three times in a row, is
  killed and started again after a pause that doubles from ``backoff_start_s`` to
  ``backoff_max_s`` and starts over once a server has run ``stable_after_s``;
* it is ended with the application: :meth:`stop` terminates the process and, on Windows, the job
  object of :mod:`twin.ops.jobobject` ends it even when the application is killed.

A server that is already listening on the port is **adopted** when it serves the very model file
(a ``twin model serve`` in another window) and refused when it serves something else or is not a
llama-server: the port is never taken over and never shared.

The manager only ever starts a program on this computer, on ``127.0.0.1``; the address check is in
:class:`~twin.serving.llamacpp.ServerSpec`.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Final

import httpx

from twin.clock import Clock
from twin.llm.style_client import StyleModelClient
from twin.ops.alerts import AlertSink
from twin.ops.jobobject import ProcessJob
from twin.ops.logging import get_logger
from twin.serving.llamacpp import ServeError, ServerSpec

log = get_logger("twin.serving.server")

LOG_KEEP: Final = 3
LOG_MAX_BYTES: Final = 10 * 1024 * 1024
POLL_S: Final = 0.5
STOP_GRACE_S: Final = 8.0
PORT_TIMEOUT_S: Final = 1.0
PROPS_TIMEOUT_S: Final = 3.0
TAIL_LINES: Final = 3


class ServerState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    BACKOFF = "backoff"  # crashed; waiting to start again
    BLOCKED = "blocked"  # refused by the tokenizer comparison; stays stopped
    EXTERNAL = "external"  # another process serves this model; only watched


class ServerBlocked(RuntimeError):
    """``on_ready`` refuses the model: the server is stopped and not started again."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ServerTimings:
    """How the supervision loop waits (all from ``style_model.serve`` and ``backend``)."""

    start_timeout_s: float = 180.0
    backoff_start_s: float = 2.0
    backoff_max_s: float = 60.0
    stable_after_s: float = 120.0
    health_interval_s: float = 30.0
    unhealthy_limit: int = 3
    stop_grace_s: float = STOP_GRACE_S


@dataclass(frozen=True)
class ServerSnapshot:
    """What ``/状态`` and ``twin model serve`` show about the process."""

    state: ServerState
    pid: int | None
    restarts: int
    started_at: datetime | None
    ready_at: datetime | None
    last_exit: int | None
    detail: str


def rotate_log(path: Path, *, keep: int = LOG_KEEP, max_bytes: int = LOG_MAX_BYTES) -> None:
    """Move a log that grew past ``max_bytes`` to ``.1`` (older ones to ``.2``, ...)."""
    if not path.is_file() or path.stat().st_size < max_bytes:
        return
    oldest = path.with_name(f"{path.name}.{keep}")
    oldest.unlink(missing_ok=True)
    for number in range(keep - 1, 0, -1):
        older = path.with_name(f"{path.name}.{number}")
        if older.is_file():
            older.replace(path.with_name(f"{path.name}.{number + 1}"))
    path.replace(path.with_name(f"{path.name}.1"))


def log_tail(path: Path, lines: int = TAIL_LINES) -> str:
    """The last lines of the server's log, for the message that says why it stopped."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    kept = [line.strip()[:200] for line in text.splitlines() if line.strip()]
    return " | ".join(kept[-lines:])


async def served_model_path(endpoint: str) -> str | None:
    """``model_path`` that a llama-server at ``endpoint`` reports in ``/props``."""
    try:
        async with httpx.AsyncClient(timeout=PROPS_TIMEOUT_S) as http:
            response = await http.get(endpoint.rstrip("/") + "/props")
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    path = data.get("model_path") if isinstance(data, dict) else None
    return path if isinstance(path, str) else None


def same_model_file(reported: str | None, expected: Path) -> bool:
    """Whether the path a server reports is the model file (compared by file name)."""
    if not reported:
        return False
    return Path(reported.replace("\\", "/")).name == expected.name


def _creation_flags(platform: str) -> int:
    """No console window for the child on Windows (the application runs without one)."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform == "win32" else 0


class LlamaServerManager:
    """Starts, watches, restarts and stops one ``llama-server`` (see the module description)."""

    def __init__(
        self,
        spec: ServerSpec,
        *,
        clock: Clock,
        client: StyleModelClient,
        log_path: Path,
        timings: ServerTimings | None = None,
        alerts: AlertSink | None = None,
        job: ProcessJob | None = None,
        on_ready: Callable[[], Awaitable[None]] | None = None,
        on_change: Callable[[ServerSnapshot], None] | None = None,
        cwd: Path | None = None,
        platform: str | None = None,
    ) -> None:
        self._spec = spec
        self._clock = clock
        self._client = client
        self._log_path = log_path
        self._timings = timings or ServerTimings()
        self._alerts = alerts
        self._job = job
        self._on_ready = on_ready
        self._on_change = on_change
        self._cwd = cwd
        self._platform = sys.platform if platform is None else platform
        self._state = ServerState.STOPPED
        self._detail = ""
        self._process: asyncio.subprocess.Process | None = None
        self._stop = asyncio.Event()
        self._unblock = asyncio.Event()
        self._restarts = 0
        self._started_at: datetime | None = None
        self._ready_at: datetime | None = None
        self._last_exit: int | None = None
        self._blocked: ServerBlocked | None = None

    # ------------------------------------------------------------------ state

    @property
    def spec(self) -> ServerSpec:
        return self._spec

    @property
    def state(self) -> ServerState:
        return self._state

    @property
    def blocked(self) -> ServerBlocked | None:
        return self._blocked

    def snapshot(self) -> ServerSnapshot:
        return ServerSnapshot(
            self._state,
            self._process.pid if self._process is not None else None,
            self._restarts,
            self._started_at,
            self._ready_at,
            self._last_exit,
            self._detail,
        )

    def _set(self, state: ServerState, detail: str = "") -> None:
        self._state, self._detail = state, detail
        if self._on_change is not None:
            self._on_change(self.snapshot())

    def unblock(self) -> None:
        """Try the model again after a refusal (the user fixed the cause)."""
        self._blocked = None
        self._unblock.set()

    # ------------------------------------------------------------- the waiting

    async def _sleep(self, seconds: float) -> bool:
        """Sleep ``seconds`` (interrupted by a stop request); ``True``: the full time passed."""
        sleeper = asyncio.ensure_future(self._clock.sleep(seconds))
        stopper = asyncio.ensure_future(self._stop.wait())
        try:
            await asyncio.wait({sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sleeper, stopper):
                task.cancel()
            await asyncio.gather(sleeper, stopper, return_exceptions=True)
        return not self._stop.is_set()

    # ------------------------------------------------------------ the process

    async def _port_in_use(self) -> bool:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(self._spec.host, self._spec.port), PORT_TIMEOUT_S
            )
        except (OSError, TimeoutError):
            return False
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
        return True

    async def _start_process(self) -> asyncio.subprocess.Process:
        if not self._spec.model.is_file():
            raise ServeError(f"the model file {self._spec.model} does not exist")
        rotate_log(self._log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        stream: IO[bytes] = self._log_path.open("ab")
        try:
            process = await asyncio.create_subprocess_exec(
                *self._spec.argv(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stream,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._cwd,
                creationflags=_creation_flags(self._platform),
            )
        except OSError as exc:
            raise ServeError(f"cannot start {self._spec.prefix[0]}: {exc}") from exc
        finally:
            stream.close()  # the child has its own handle; ours would lock the file on Windows
        if self._job is not None:
            # ``assign`` opens a job that nobody has opened yet; off Windows it does nothing.  Not
            # asking ``job.active`` first: a job handed over unopened would silently never
            # protect the child (D-600).
            self._job.assign(process.pid)
        return process

    async def _end_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), self._timings.stop_grace_s)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.kill()
                await process.wait()
        self._last_exit = process.returncode

    async def _await_ready(self, process: asyncio.subprocess.Process) -> str | None:
        """Poll ``/health`` until the model is loaded; ``None`` if so, else why not."""
        deadline = self._clock.monotonic() + self._timings.start_timeout_s
        while not self._stop.is_set():
            if process.returncode is not None:
                return f"the server exited with code {process.returncode}"
            health = await self._client.health()
            if health.ok:
                return None
            if self._clock.monotonic() >= deadline:
                return f"the model was not loaded after {self._timings.start_timeout_s:g} s"
            await self._sleep(POLL_S)
        return "stopped"

    async def _monitor(self, process: asyncio.subprocess.Process) -> str:
        """Watch a ready server until it dies or stops answering; returns why."""
        failures = 0
        waiter = asyncio.ensure_future(process.wait())
        try:
            while not self._stop.is_set():
                if waiter.done():
                    return f"the server exited with code {process.returncode}"
                await self._sleep(self._timings.health_interval_s)
                if self._stop.is_set():
                    break
                if waiter.done():
                    return f"the server exited with code {process.returncode}"
                health = await self._client.health()
                failures = 0 if health.ok else failures + 1
                if failures >= self._timings.unhealthy_limit:
                    return f"/health failed {failures} times in a row: {health.detail}"
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        return "stopped"

    # ----------------------------------------------------------------- adoption

    async def _adopt_or_refuse(self) -> bool:
        """If the port is taken: ``True`` when the model's own server is there, else an error."""
        if not await self._port_in_use():
            return False
        endpoint = f"http://{self._spec.host}:{self._spec.port}"
        health = await self._client.health()
        reported = await served_model_path(endpoint) if (health.ok or health.loading) else None
        if reported is None:
            raise ServeError(
                f"port {self._spec.port} is used by a program that is not a llama-server; "
                "stop it or change style_model.endpoint"
            )
        if not same_model_file(reported, self._spec.model):
            raise ServeError(
                f"port {self._spec.port} is used by a llama-server that serves another model "
                f"({Path(reported.replace(chr(92), '/')).name}); stop it first"
            )
        return True

    async def _watch_external(self) -> None:
        """A server of someone else serves our model: only look at it, never touch it."""
        self._set(ServerState.EXTERNAL, "a llama-server of another process serves this model")
        log.info("server_adopted", port=self._spec.port)
        failures = 0
        while await self._sleep(self._timings.health_interval_s):
            health = await self._client.health()
            failures = 0 if health.ok else failures + 1
            if failures >= self._timings.unhealthy_limit:
                log.warning("server_external_gone", port=self._spec.port)
                return

    # ------------------------------------------------------------- the main loop

    async def run(self) -> None:
        """Keep the server running until :meth:`stop` (run it under a task supervisor)."""
        delay = self._timings.backoff_start_s
        while not self._stop.is_set():
            if self._blocked is not None:
                self._unblock.clear()
                stopper = asyncio.ensure_future(self._stop.wait())
                unblock = asyncio.ensure_future(self._unblock.wait())
                await asyncio.wait({stopper, unblock}, return_when=asyncio.FIRST_COMPLETED)
                for task in (stopper, unblock):
                    task.cancel()
                await asyncio.gather(stopper, unblock, return_exceptions=True)
                continue
            if await self._adopt_or_refuse():
                await self._watch_external()
                continue
            started = self._clock.monotonic()
            reason = await self._run_once()
            if self._stop.is_set() or self._blocked is not None:
                continue
            self._restarts += 1
            tail = log_tail(self._log_path)
            log.warning("server_stopped", reason=reason, restarts=self._restarts, tail=tail)
            if self._clock.monotonic() - started >= self._timings.stable_after_s:
                delay = self._timings.backoff_start_s
            self._set(ServerState.BACKOFF, reason)
            await self._sleep(delay)
            delay = min(self._timings.backoff_max_s, delay * 2)
        self._set(ServerState.STOPPED)

    async def _run_once(self) -> str:
        """One life of the process: start, load, check, watch, end.  Returns why it ended."""
        self._set(ServerState.STARTING, "loading the model")
        self._started_at, self._ready_at = self._clock.now_utc(), None
        try:
            self._process = await self._start_process()
        except ServeError as exc:  # no program, no model file: waiting does not fix that
            self._blocked = ServerBlocked("start_failed", str(exc))
            self._set(ServerState.BLOCKED, str(exc))
            log.warning("server_blocked", reason="start_failed")
            return str(exc)
        log.info("server_started", pid=self._process.pid, port=self._spec.port)
        try:
            failed = await self._await_ready(self._process)
            if failed is not None:
                return failed
            if self._on_ready is not None:
                try:
                    await self._on_ready()
                except ServerBlocked as blocked:
                    self._blocked = blocked
                    self._set(ServerState.BLOCKED, blocked.reason)
                    log.warning("server_blocked", reason=blocked.reason)
                    return blocked.reason
            self._ready_at = self._clock.now_utc()
            self._set(ServerState.READY)
            return await self._monitor(self._process)
        finally:
            await self._end_process()

    async def stop(self) -> None:
        """End the server and the loop (the process is terminated, then killed after a grace)."""
        self._stop.set()
        await self._end_process()

    async def wait_state(self, states: set[ServerState], limit_s: float) -> bool:
        """Wait until the manager is in one of ``states`` (``False`` after ``limit_s``)."""
        deadline = self._clock.monotonic() + limit_s
        while self._state not in states:
            if self._clock.monotonic() >= deadline or self._stop.is_set():
                return self._state in states
            await self._clock.sleep(0.05)
        return True
