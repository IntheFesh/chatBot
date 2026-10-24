"""A connection that comes back after it breaks (R-TRN-010).

Long steps run on the instance as background jobs (:mod:`twin.training.remote.jobs`), so a dropped
connection loses nothing there; what has to survive on this side is the *control* of them.
``RemoteSession`` holds the login, hands out the SFTP helper, and runs commands; when the
connection breaks, :meth:`RemoteSession.retrying` waits (a growing pause measured with the
application's clock), logs in again and repeats the action.  The actions it is used for are
repeatable by design: reading a log from an offset, asking for a status, starting a job by name,
and uploads and downloads that resume.

A refused login and a changed host key are not retried: waiting does not change them.  A file
whose sha256 does not match is transferred once more, and a second mismatch ends the step.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import asyncssh

from twin.clock import Clock
from twin.ops.logging import get_logger
from twin.training.remote.connection import (
    HashMismatchError,
    HostKeyError,
    LoginRefusedError,
    RemoteError,
    RemoteTarget,
    TrustPrompt,
    open_connection,
)
from twin.training.remote.transfer import RemoteFiles, RunResult, run_command

T = TypeVar("T")
log = get_logger("twin.training.remote")

BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 30.0)


class RemoteSession:
    """The login of one ``twin train remote`` invocation."""

    def __init__(
        self,
        target: RemoteTarget,
        clock: Clock,
        *,
        trust: TrustPrompt | None = None,
        python: str = "python3",
        backoff: tuple[float, ...] = BACKOFF_SECONDS,
    ) -> None:
        self.target = target
        self._clock = clock
        self._trust = trust
        self._python = python
        self._backoff = backoff
        self._connection: asyncssh.SSHClientConnection | None = None
        self._files: RemoteFiles | None = None
        self.reconnects = 0
        self.hash_tool: str | None = None

    @property
    def python(self) -> str:
        return self._python

    async def connection(self) -> asyncssh.SSHClientConnection:
        if self._connection is None:
            self._connection = await open_connection(self.target, trust=self._trust)
            self._files = None
        return self._connection

    async def files(self) -> RemoteFiles:
        connection = await self.connection()
        if self._files is None:
            self._files = RemoteFiles(connection, python=self._python, hash_tool=self.hash_tool)
        return self._files

    def use_hash_tool(self, path: str) -> None:
        self.hash_tool = path
        if self._files is not None:
            self._files.use_hash_tool(path)

    async def run(
        self, command: str, *, stdin: str | None = None, limit_s: float | None = 600.0
    ) -> RunResult:
        """Run a command once (no retry): use :meth:`retrying` for repeatable ones."""
        return await run_command(await self.connection(), command, stdin=stdin, limit_s=limit_s)

    async def drop(self) -> None:
        """Forget the connection (it broke, or the caller wants a fresh one)."""
        connection, self._connection, self._files = self._connection, None, None
        if connection is not None:
            connection.abort()
            await connection.wait_closed()

    async def retrying(self, action: Callable[[], Awaitable[T]], *, what: str = "the step") -> T:
        """Run ``action``; when the connection breaks, log in again and repeat it."""
        attempt = 0
        mismatches = 0
        while True:
            try:
                return await action()
            except (HostKeyError, LoginRefusedError):
                raise
            except HashMismatchError:
                # damage in transit may be gone on the second try; a second mismatch is final
                mismatches += 1
                if mismatches > 1:
                    raise
            except (RemoteError, OSError, asyncssh.Error, TimeoutError) as exc:
                if attempt >= len(self._backoff):
                    raise RemoteError(f"{what} failed after {attempt} reconnects: {exc}") from exc
                pause = self._backoff[attempt]
                attempt += 1
                self.reconnects += 1
                log.warning("remote_reconnect", what=what, attempt=attempt, pause_s=pause)
                await self.drop()
                await self._clock.sleep(pause)

    async def close(self) -> None:
        if self._files is not None:
            await self._files.close()
        await self.drop()

    async def __aenter__(self) -> RemoteSession:
        await self.connection()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
