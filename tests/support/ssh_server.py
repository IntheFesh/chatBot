"""A real SSH server on localhost for the tests of the remote orchestration (R-TRN-010).

It is ``asyncssh``'s own server side: password or public key login, an SFTP subsystem confined to
a directory (the "instance's disk"), and command execution that runs the command line in a
subprocess with that directory as its working directory.  ``python3`` in a command line means
the interpreter running the tests, so the tools in ``training/autodl/tools`` run on Windows as well.

Hooks let a test break the connection in the middle of a transfer (``drop_after_bytes``) and
record what was written and which commands were run.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncssh

USER = "root"
PASSWORD = "instance-password-1"


@dataclass
class ServerState:
    """What a test can look at and steer."""

    root: Path
    chroot: bool = True
    password: str = PASSWORD
    authorized_key: asyncssh.SSHKey | None = None
    commands: list[str] = field(default_factory=list)
    bytes_written: dict[str, int] = field(default_factory=dict)
    drop_after_bytes: dict[str, int] = field(default_factory=dict)  # path suffix -> byte count
    fail_exec: bool = False  # refuse every command (hash tool fallback tests)
    logins: int = 0
    drops: int = 0


class _Auth(asyncssh.SSHServer):
    def __init__(self, state: ServerState) -> None:
        self._state = state

    def connection_made(self, conn: asyncssh.SSHServerConnection) -> None:
        self._state.logins += 1

    def begin_auth(self, username: str) -> bool:
        return True

    def password_auth_supported(self) -> bool:
        return True

    def validate_password(self, username: str, password: str) -> bool:
        return username == USER and password == self._state.password

    def public_key_auth_supported(self) -> bool:
        return self._state.authorized_key is not None

    def validate_public_key(self, username: str, key: asyncssh.SSHKey) -> bool:
        authorized = self._state.authorized_key
        return (
            username == USER
            and authorized is not None
            and key.export_public_key() == authorized.export_public_key()
        )


def _sftp_factory(state: ServerState) -> Any:
    class HookedSFTP(asyncssh.SFTPServer):
        def __init__(self, chan: asyncssh.SSHServerChannel[bytes]) -> None:
            super().__init__(chan, chroot=str(state.root).encode() if state.chroot else None)
            self._channel = chan

        def write(self, file_obj: Any, offset: int, data: bytes) -> int:
            written = super().write(file_obj, offset, data)
            name = os.fsdecode(getattr(file_obj, "name", b""))
            total = state.bytes_written.get(name, 0) + len(data)
            state.bytes_written[name] = total
            for suffix, limit in list(state.drop_after_bytes.items()):
                if name.endswith(suffix) and total >= limit:
                    del state.drop_after_bytes[suffix]
                    state.drops += 1
                    self._channel.get_connection().abort()
            return written  # type: ignore[no-any-return]

    return HookedSFTP


def _process_handler(state: ServerState) -> Any:
    async def handle(process: asyncssh.SSHServerProcess[str]) -> None:
        command = process.command or ""
        state.commands.append(command)
        if state.fail_exec:
            process.stderr.write("exec disabled\n")
            process.exit(127)
            return
        argv = shlex.split(command)
        if argv and argv[0] == "python3":
            argv[0] = sys.executable
        try:
            child = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=state.root,
            )
        except OSError as exc:
            process.stderr.write(f"{exc}\n")
            process.exit(127)
            return

        async def pump() -> None:
            """Forward the client's standard input until it closes it (or the command ends)."""
            assert child.stdin is not None
            try:
                while data := await process.stdin.read(65536):
                    child.stdin.write(data.encode("utf-8"))
                    await child.stdin.drain()
            except (ConnectionError, asyncssh.Error):
                pass
            finally:
                child.stdin.close()

        # communicate() would close the command's standard input at once; the client may still
        # be sending, so the streams are read separately and stdin is closed by pump() alone
        assert child.stdout is not None and child.stderr is not None
        feeder = asyncio.create_task(pump())
        out_task = asyncio.create_task(child.stdout.read())
        err_task = asyncio.create_task(child.stderr.read())
        await child.wait()
        out, err = await out_task, await err_task
        feeder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await feeder
        process.stdout.write(out.decode("utf-8", errors="replace"))
        process.stderr.write(err.decode("utf-8", errors="replace"))
        process.exit(child.returncode or 0)

    return handle


class LocalSSHServer:
    """``async with LocalSSHServer(tmp_path) as server`` - then connect to ``server.port``."""

    def __init__(self, root: Path, *, chroot: bool = True) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.state = ServerState(root=root, chroot=chroot)
        self.host_key = asyncssh.generate_private_key("ssh-ed25519")
        self._server: asyncssh.SSHAcceptor | None = None
        self.port = 0

    @property
    def root(self) -> Path:
        return self.state.root

    async def __aenter__(self) -> LocalSSHServer:
        self._server = await asyncssh.listen(
            "127.0.0.1",
            0,
            server_factory=lambda: _Auth(self.state),
            server_host_keys=[self.host_key],
            process_factory=_process_handler(self.state),
            sftp_factory=_sftp_factory(self.state),
            allow_scp=False,
        )
        self.port = self._server.get_port()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def fingerprint(self) -> str:
        return self.host_key.get_fingerprint("sha256")


class ThreadedSSHServer:
    """The same server in a thread with its own event loop, for tests that call ``asyncio.run``.

    ``chroot=False`` serves the real file system, so absolute paths mean the same for SFTP and
    for the commands (POSIX tests only).
    """

    def __init__(self, root: Path, *, chroot: bool = True) -> None:
        self._server = LocalSSHServer(root, chroot=chroot)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._server.__aenter__())
        self._ready.set()
        self._loop.run_forever()

    @property
    def server(self) -> LocalSSHServer:
        return self._server

    @property
    def port(self) -> int:
        return self._server.port

    @property
    def root(self) -> Path:
        return self._server.root

    def __enter__(self) -> ThreadedSSHServer:
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("the test SSH server did not start")
        return self

    def __exit__(self, *exc_info: object) -> None:
        future = asyncio.run_coroutine_threadsafe(self._server.__aexit__(), self._loop)
        future.result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)
        self._loop.close()
