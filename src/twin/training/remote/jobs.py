"""Long steps on the instance that survive a dropped connection (R-TRN-010).

A step such as ``train.sh`` runs for hours.  It is started through ``autodl/tools/remote_job.py``,
which detaches it from the SSH session and keeps its log, its exit code and a heartbeat in
``<workdir>/jobs/<name>/``.  Starting a name that is still running does nothing, so asking again
after a reconnect is safe.  :meth:`RemoteJobs.follow` then reads the log from an offset - kept
by the caller between invocations - until the job has exited, and goes on through reconnects.

The log is printed to the terminal through ``emit`` and never written to this program's log
files: it contains samples of the training data.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from twin.clock import Clock
from twin.training.layout import RemoteLayout
from twin.training.remote.connection import RemoteError
from twin.training.remote.session import RemoteSession

POLL_SECONDS = 2.0
TAIL_LIMIT = 1 << 18


@dataclass(frozen=True)
class JobStatus:
    state: str  # none, running, exited, lost
    exit_code: int | None
    log_size: int


@dataclass(frozen=True)
class JobOutcome:
    name: str
    state: str
    exit_code: int | None
    offset: int

    @property
    def ok(self) -> bool:
        return self.state == "exited" and self.exit_code == 0


class RemoteJobs:
    def __init__(
        self,
        session: RemoteSession,
        layout: RemoteLayout,
        clock: Clock,
        *,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._session = session
        self._layout = layout
        self._clock = clock
        self._poll = poll_seconds

    def _tool(self, *args: str) -> str:
        tool = f"{self._layout.tools}/remote_job.py"
        parts = [self._session.python, tool, *args]
        return " ".join(shlex.quote(part) for part in parts)

    async def _call(self, *args: str) -> dict[str, object]:
        command = self._tool(*args)

        async def action() -> dict[str, object]:
            result = await self._session.run(command)
            if not result.ok:
                raise RemoteError(f"remote_job.py failed: {result.stderr.strip()[:200]}")
            try:
                return json.loads(result.stdout)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                raise RemoteError("remote_job.py gave an unreadable answer") from None

        return await self._session.retrying(action, what="reading the job state")

    async def start(
        self, name: str, argv: Sequence[str], *, env: Mapping[str, str] | None = None
    ) -> JobStatus:
        """Start the job unless one of that name is running (then it just keeps running)."""
        args = ["start", "--dir", self._layout.jobs, "--name", name, "--cwd", self._layout.workdir]
        for key, value in (env or {}).items():
            args += ["--env", f"{key}={value}"]
        args += ["--", *argv]
        return _status(await self._call(*args))

    async def status(self, name: str) -> JobStatus:
        return _status(await self._call("status", "--dir", self._layout.jobs, "--name", name))

    async def stop(self, name: str) -> JobStatus:
        return _status(await self._call("stop", "--dir", self._layout.jobs, "--name", name))

    async def follow(
        self,
        name: str,
        *,
        offset: int = 0,
        emit: Callable[[str], None],
        on_offset: Callable[[int], None] | None = None,
    ) -> JobOutcome:
        """Print the job's log from ``offset`` until the job ends; returns how it ended."""
        while True:
            status = await self.status(name)
            while True:
                chunk = await self._call(
                    "tail",
                    "--dir",
                    self._layout.jobs,
                    "--name",
                    name,
                    "--offset",
                    str(offset),
                    "--limit",
                    str(TAIL_LIMIT),
                )
                data, new_offset = str(chunk["data"]), int(str(chunk["offset"]))
                if data:
                    emit(data)
                if new_offset != offset:
                    offset = new_offset
                    if on_offset:
                        on_offset(offset)
                if chunk["eof"]:
                    break
            if status.state != "running":
                return JobOutcome(name, status.state, status.exit_code, offset)
            await self._clock.sleep(self._poll)

    async def run(
        self,
        name: str,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        offset: int = 0,
        emit: Callable[[str], None],
        on_offset: Callable[[int], None] | None = None,
    ) -> JobOutcome:
        """Start (or re-attach to) the job and follow it to its end."""
        await self.start(name, argv, env=env)
        return await self.follow(name, offset=offset, emit=emit, on_offset=on_offset)


def _status(data: Mapping[str, object]) -> JobStatus:
    code = data.get("exit_code")
    return JobStatus(
        str(data["state"]), int(str(code)) if code is not None else None, int(str(data["log_size"]))
    )
