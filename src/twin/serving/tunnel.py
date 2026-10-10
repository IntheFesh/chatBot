"""The SSH tunnel to the rented instance's vLLM server (R-SRV-003).

vLLM listens on ``127.0.0.1:<remote_port>`` of the AutoDL instance only.  :class:`TunnelManager`
logs in with ``asyncssh`` (the login of ``twin train remote``: password from the credential store
or a key file, the host key saved by ``twin train remote connect``) and forwards
``127.0.0.1:<style_model.tunnel.local_port>`` on this computer to it, so
``StyleModelClient`` in ``vllm_completion`` mode talks to a local address.

* the local end listens on **127.0.0.1 only**; nothing on the network can reach the instance
  through this computer;
* a connection that breaks (the keep-alives of the login notice a dead link within about a minute)
  is logged in again after a pause that doubles from ``tunnel.backoff_start_s`` to
  ``tunnel.backoff_max_s`` and starts over once the tunnel has held a minute - the instance is
  billed by the hour, a reconnect that takes minutes costs replies, not just time;
* a refused login or a changed host key is **not** retried (waiting does not fix them): the tunnel
  stays ``blocked`` with the reason until it is asked to try again;
* when the local port is already served by a healthy tunnel of another process (``twin model
  tunnel start`` in a window of its own) that one is used and only watched.

While the tunnel is up the manager can also read how long the instance has been running
(``/proc/uptime`` over the same login), which the daily reminder of the cost shows.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

import asyncssh

from twin.clock import Clock
from twin.llm.style_client import StyleHealth
from twin.ops.logging import get_logger
from twin.training.remote.connection import (
    HostKeyError,
    LoginRefusedError,
    RemoteError,
    RemoteTarget,
    open_connection,
)
from twin.training.remote.transfer import run_command

log = get_logger("twin.serving.tunnel")

LISTEN_HOST: Final = "127.0.0.1"
REMOTE_HOST: Final = "127.0.0.1"
UPTIME_COMMAND: Final = "cat /proc/uptime"
STABLE_AFTER_S: Final = 60.0
WATCH_S: Final = 5.0
CLOSE_LIMIT_S: Final = 10.0


class TunnelState(StrEnum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    UP = "up"
    BACKOFF = "backoff"  # the connection broke; logging in again soon
    BLOCKED = "blocked"  # refused login or changed host key; waits for a new try
    EXTERNAL = "external"  # another process holds the tunnel; only watched


@dataclass(frozen=True)
class TunnelTimings:
    backoff_start_s: float = 2.0
    backoff_max_s: float = 60.0
    stable_after_s: float = STABLE_AFTER_S
    watch_s: float = WATCH_S  # how often an adopted tunnel is looked at


@dataclass(frozen=True)
class TunnelSnapshot:
    state: TunnelState
    local_port: int
    up_since: datetime | None
    reconnects: int
    detail: str


Connector = Callable[[RemoteTarget], Awaitable[asyncssh.SSHClientConnection]]
LocalHealth = Callable[[], Awaitable[StyleHealth]]


def parse_uptime(text: str) -> float | None:
    """Seconds since boot from the text of ``/proc/uptime`` (``"12345.67 98765.43"``)."""
    fields = text.split()
    try:
        value = float(fields[0])
    except (IndexError, ValueError):
        return None
    return value if value >= 0 else None


class TunnelManager:
    """Keeps the forward to the instance up (see the module description)."""

    def __init__(
        self,
        target: RemoteTarget,
        *,
        local_port: int,
        remote_port: int,
        clock: Clock,
        timings: TunnelTimings | None = None,
        connect: Connector | None = None,
        local_health: LocalHealth | None = None,
        on_change: Callable[[TunnelSnapshot], None] | None = None,
    ) -> None:
        self._target = target
        self._local_port = local_port
        self._remote_port = remote_port
        self._clock = clock
        self._timings = timings or TunnelTimings()
        self._connect: Connector = connect or _login
        self._local_health = local_health
        self._on_change = on_change
        self._state = TunnelState.STOPPED
        self._detail = ""
        self._connection: asyncssh.SSHClientConnection | None = None
        self._listener: asyncssh.SSHListener | None = None
        self._stop = asyncio.Event()
        self._retry = asyncio.Event()
        self._reconnects = 0
        self._up_since: datetime | None = None

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> TunnelState:
        return self._state

    @property
    def local_port(self) -> int:
        """The port the forward listens on (the real one when 0 was asked for)."""
        if self._listener is not None:
            return int(self._listener.get_port())
        return self._local_port

    def snapshot(self) -> TunnelSnapshot:
        return TunnelSnapshot(
            self._state, self.local_port, self._up_since, self._reconnects, self._detail
        )

    def _set(self, state: TunnelState, detail: str = "") -> None:
        self._state, self._detail = state, detail
        if self._on_change is not None:
            self._on_change(self.snapshot())

    def retry(self) -> None:
        """Try again after a refusal (the user fixed the password or the host key)."""
        self._retry.set()

    # ------------------------------------------------------------- the waiting

    async def _sleep(self, seconds: float) -> bool:
        sleeper = asyncio.ensure_future(self._clock.sleep(seconds))
        stopper = asyncio.ensure_future(self._stop.wait())
        try:
            await asyncio.wait({sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sleeper, stopper):
                task.cancel()
            await asyncio.gather(sleeper, stopper, return_exceptions=True)
        return not self._stop.is_set()

    async def _wait_for_retry(self) -> None:
        self._retry.clear()
        stopper = asyncio.ensure_future(self._stop.wait())
        retry = asyncio.ensure_future(self._retry.wait())
        try:
            await asyncio.wait({stopper, retry}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (stopper, retry):
                task.cancel()
            await asyncio.gather(stopper, retry, return_exceptions=True)

    # --------------------------------------------------------------- one life

    async def _close(self) -> None:
        """Close the login first (that ends the forwarded connections), then the listener.

        ``Server.wait_closed`` of Python 3.12 waits for every accepted connection, so each wait is
        bounded: a peer that never hangs up must not keep the application from stopping.
        """
        listener, self._listener = self._listener, None
        connection, self._connection = self._connection, None
        if listener is not None:
            listener.close()
        if connection is not None:
            connection.close()
            await _bounded(connection.wait_closed())
        if listener is not None:
            await _bounded(listener.wait_closed())
        self._up_since = None

    async def _adoptable(self) -> bool:
        """Is the local port already served by a healthy tunnel of another process?"""
        if self._local_health is None:
            return False
        health = await self._local_health()
        return health.ok

    async def _hold(self, connection: asyncssh.SSHClientConnection) -> str:
        """Hold a forward on ``connection`` until it breaks or the manager is stopped."""
        try:
            self._listener = await connection.forward_local_port(
                LISTEN_HOST, self._local_port, REMOTE_HOST, self._remote_port
            )
        except OSError as exc:  # the port is taken: by a tunnel of another process, or by a program
            if await self._adoptable():
                await self._watch_external()
                return "external"
            raise RemoteError(
                f"local port {self._local_port} is in use by something that is not a tunnel "
                f"to the instance ({exc})"
            ) from exc
        self._up_since = self._clock.now_utc()
        self._set(TunnelState.UP)
        log.info("tunnel_up", local_port=self.local_port, remote_port=self._remote_port)
        closed = asyncio.ensure_future(connection.wait_closed())
        stopper = asyncio.ensure_future(self._stop.wait())
        try:
            await asyncio.wait({closed, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (closed, stopper):
                task.cancel()
            await asyncio.gather(closed, stopper, return_exceptions=True)
        return "stopped" if self._stop.is_set() else "the connection to the instance was lost"

    async def _watch_external(self) -> None:
        self._set(TunnelState.EXTERNAL, "another process holds the tunnel")
        log.info("tunnel_adopted", local_port=self._local_port)
        failures = 0
        while await self._sleep(self._timings.watch_s):
            health = await self._local_health() if self._local_health else None
            failures = 0 if health is not None and health.ok else failures + 1
            if failures >= 3:
                return

    async def run(self) -> None:
        """Keep the tunnel up until :meth:`stop` (run it under a task supervisor)."""
        delay = self._timings.backoff_start_s
        while not self._stop.is_set():
            started = self._clock.monotonic()
            self._set(TunnelState.CONNECTING)
            reason = "stopped"
            try:
                self._connection = await self._connect(self._target)
                reason = await self._hold(self._connection)
            except (HostKeyError, LoginRefusedError) as exc:
                await self._close()
                self._set(TunnelState.BLOCKED, str(exc))
                log.warning("tunnel_blocked", reason=type(exc).__name__)
                await self._wait_for_retry()
                continue
            except (RemoteError, OSError, asyncssh.Error, TimeoutError) as exc:
                reason = str(exc) or type(exc).__name__
            finally:
                await self._close()
            if self._stop.is_set():
                break
            self._reconnects += 1
            if self._clock.monotonic() - started >= self._timings.stable_after_s:
                delay = self._timings.backoff_start_s
            self._set(TunnelState.BACKOFF, reason)
            log.warning("tunnel_lost", reason=reason, reconnects=self._reconnects)
            await self._sleep(delay)
            delay = min(self._timings.backoff_max_s, delay * 2)
        self._set(TunnelState.STOPPED)

    async def stop(self) -> None:
        self._stop.set()
        await self._close()

    # ------------------------------------------------------------- the instance

    async def instance_uptime(self) -> float | None:
        """How long the instance has been up (``/proc/uptime``); ``None`` if that cannot be read."""
        connection = self._connection
        if connection is None or self._state is not TunnelState.UP:
            return None
        try:
            done = await run_command(connection, UPTIME_COMMAND, limit_s=15.0)
        except RemoteError:
            return None
        return parse_uptime(done.stdout) if done.ok else None


async def _bounded(waiting: Awaitable[object]) -> None:
    """Wait for a close to finish, but not longer than ``CLOSE_LIMIT_S``; failures do not matter."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(waiting, CLOSE_LIMIT_S)


async def _login(target: RemoteTarget) -> asyncssh.SSHClientConnection:
    """The login of ``twin train remote``; the host key must have been trusted before."""
    return await open_connection(target, trust=None)
