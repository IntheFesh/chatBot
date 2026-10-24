"""Files to and from the instance: uploads that resume, downloads that verify (R-TRN-010).

Every transfer ends with a sha256 comparison, because the bytes cross the Internet and the
instance's disk is not redundant:

* ``upload`` writes to ``<remote>.part`` and moves the file into place only when the remote copy
  has the size and sha256 of the local file.  When the connection drops, the next call finds the
  ``.part`` file, checks that its bytes are the beginning of the local file (the hash of the first
  ``size`` bytes is compared) and sends only the rest.  A ``.part`` file that is not a prefix of
  the local file is discarded and the upload starts again.
* ``download`` does the same in the other direction and refuses a file whose sha256 differs from
  the one the manifest names; the file is moved to its final name only after that check.

The remote hash is computed on the instance by ``tools/file_hash.py`` (no data crosses the
network for it); when that tool is not there yet - while the tools themselves are uploaded - the
bytes are read back over SFTP.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import asyncssh

from twin.training.dataset_dir import file_sha256
from twin.training.remote.connection import HashMismatchError, RemoteError

CHUNK = 1 << 20
Progress = Callable[[int, int], None]
"""``(bytes done, bytes total)``."""


@dataclass(frozen=True)
class RunResult:
    stdout: str
    stderr: str
    status: int

    @property
    def ok(self) -> bool:
        return self.status == 0


@dataclass(frozen=True)
class TransferResult:
    path: str
    size: int
    sha256: str
    bytes_sent: int  # bytes that crossed the network in this call
    resumed_from: int  # offset a partial file was continued from (0: a fresh transfer)
    skipped: bool  # the file was already complete


async def run_command(
    connection: asyncssh.SSHClientConnection,
    command: str,
    *,
    stdin: str | None = None,
    limit_s: float | None = 600.0,
) -> RunResult:
    """Run ``command`` on the instance and wait for it; ``stdin`` is sent and closed."""
    try:
        done = await connection.run(command, input=stdin, check=False, timeout=limit_s)
    except (OSError, asyncssh.Error, TimeoutError) as exc:
        raise RemoteError(f"the command could not be run: {exc}") from exc
    status = done.exit_status if done.exit_status is not None else -1
    return RunResult(str(done.stdout or ""), str(done.stderr or ""), int(status))


def _prefix_sha256(path: Path, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        while remaining > 0:
            block = stream.read(min(CHUNK, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _local_size(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def _prepare_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_complete(path: Path, size: int, sha256: str) -> bool:
    return _local_size(path) == size and file_sha256(path) == sha256


def _read_block(stream: object, size: int) -> bytes:
    return stream.read(size)  # type: ignore[attr-defined, no-any-return]


class RemoteFiles:
    """SFTP transfers on one connection."""

    def __init__(
        self,
        connection: asyncssh.SSHClientConnection,
        *,
        python: str = "python3",
        hash_tool: str | None = None,
    ) -> None:
        self._connection = connection
        self._python = python
        self._hash_tool = hash_tool
        self._sftp: asyncssh.SFTPClient | None = None

    async def sftp(self) -> asyncssh.SFTPClient:
        if self._sftp is None:
            try:
                self._sftp = await self._connection.start_sftp_client()
            except (OSError, asyncssh.Error) as exc:
                raise RemoteError(f"cannot start SFTP: {exc}") from exc
        return self._sftp

    def use_hash_tool(self, path: str | None) -> None:
        """Enable hashing on the instance once ``tools/file_hash.py`` has been uploaded."""
        self._hash_tool = path

    async def makedirs(self, path: str) -> None:
        sftp = await self.sftp()
        await sftp.makedirs(path, exist_ok=True)

    async def size(self, path: str) -> int | None:
        sftp = await self.sftp()
        try:
            attrs = await sftp.stat(path)
        except asyncssh.SFTPNoSuchFile:
            return None
        return int(attrs.size or 0)

    async def remote_hash(
        self, path: str, length: int | None = None
    ) -> tuple[int | None, str | None]:
        """``(size, sha256 of the first length bytes or of all)``; ``(None, None)`` if absent."""
        if self._hash_tool:
            command = f"{self._python} {shlex.quote(self._hash_tool)} --path {shlex.quote(path)}"
            if length is not None:
                command += f" --length {length}"
            result = await run_command(self._connection, command, limit_s=3600.0)
            if result.ok:
                data = json.loads(result.stdout)
                return data["size"], data["sha256"]
        return await self._hash_over_sftp(path, length)

    async def _hash_over_sftp(self, path: str, length: int | None) -> tuple[int | None, str | None]:
        sftp = await self.sftp()
        total = await self.size(path)
        if total is None:
            return None, None
        remaining = total if length is None else min(length, total)
        digest = hashlib.sha256()
        async with sftp.open(path, "rb") as stream:
            offset = 0
            while remaining > 0:
                block = cast(bytes, await stream.read(min(CHUNK, remaining), offset))
                if not block:
                    break
                digest.update(block)
                offset += len(block)
                remaining -= len(block)
        return total, digest.hexdigest()

    async def read_text(self, path: str) -> str | None:
        sftp = await self.sftp()
        try:
            async with sftp.open(path, "rb") as stream:
                return cast(bytes, await stream.read()).decode("utf-8")
        except asyncssh.SFTPNoSuchFile:
            return None

    async def write_bytes(self, path: str, data: bytes, *, executable: bool = False) -> None:
        """Write a small file (scripts, environment files) and check its size."""
        sftp = await self.sftp()
        await sftp.makedirs(posixpath.dirname(path), exist_ok=True)
        async with sftp.open(path, "wb") as stream:
            await stream.write(data)
        if await self.size(path) != len(data):
            raise RemoteError(f"{path} was not written completely")
        if executable:
            await sftp.chmod(path, 0o755)

    async def upload(
        self, local: Path, remote: str, *, progress: Progress | None = None
    ) -> TransferResult:
        """Upload ``local`` to ``remote``, resuming an earlier partial attempt (see above)."""
        sftp = await self.sftp()
        size = await asyncio.to_thread(_local_size, local)
        if size is None:
            raise RemoteError(f"{local} does not exist")
        digest = await asyncio.to_thread(file_sha256, local)
        await sftp.makedirs(posixpath.dirname(remote), exist_ok=True)
        existing, existing_hash = await self.remote_hash(remote)
        if existing == size and existing_hash == digest:
            return TransferResult(remote, size, digest, 0, 0, True)

        part = remote + ".part"
        offset = await self._resumable_offset(local, part, size)
        sent = 0
        try:
            async with sftp.open(part, "r+b" if offset else "wb") as target:
                source = await asyncio.to_thread(local.open, "rb")
                try:
                    await asyncio.to_thread(source.seek, offset)
                    position = offset
                    while position < size:
                        block = await asyncio.to_thread(_read_block, source, CHUNK)
                        if not block:
                            break
                        await target.write(block, position)
                        position += len(block)
                        sent += len(block)
                        if progress:
                            progress(position, size)
                finally:
                    await asyncio.to_thread(source.close)
        except (OSError, asyncssh.Error) as exc:
            raise RemoteError(f"the upload of {local.name} was interrupted: {exc}") from exc
        done_size, done_hash = await self.remote_hash(part)
        if done_size != size or done_hash != digest:
            raise HashMismatchError(f"{remote} does not match the local file after the upload")
        if existing is not None:
            await sftp.remove(remote)
        await sftp.rename(part, remote)
        return TransferResult(remote, size, digest, sent, offset, False)

    async def _resumable_offset(self, local: Path, part: str, size: int) -> int:
        have = await self.size(part)
        if not have or have > size:
            return 0
        remote_size, remote_digest = await self.remote_hash(part, have)
        if remote_size is None or remote_digest is None:
            return 0
        prefix = await asyncio.to_thread(_prefix_sha256, local, have)
        return have if prefix == remote_digest else 0

    async def download(
        self,
        remote: str,
        local: Path,
        *,
        size: int,
        sha256: str,
        progress: Progress | None = None,
    ) -> TransferResult:
        """Download ``remote`` to ``local`` and verify it against ``size`` and ``sha256``."""
        sftp = await self.sftp()
        await asyncio.to_thread(_prepare_directory, local)
        if await asyncio.to_thread(_is_complete, local, size, sha256):
            return TransferResult(remote, size, sha256, 0, 0, True)
        part = local.with_name(local.name + ".part")
        offset = 0
        have = await asyncio.to_thread(_local_size, part) or 0
        if 0 < have <= size:
            _, remote_digest = await self.remote_hash(remote, have)
            if remote_digest == await asyncio.to_thread(_prefix_sha256, part, have):
                offset = have
        received = 0
        try:
            async with sftp.open(remote, "rb") as source:
                target = await asyncio.to_thread(part.open, "r+b" if offset else "wb")
                try:
                    await asyncio.to_thread(target.seek, offset)
                    position = offset
                    while position < size:
                        block = cast(
                            bytes, await source.read(min(CHUNK, size - position), position)
                        )
                        if not block:
                            break
                        await asyncio.to_thread(target.write, block)
                        position += len(block)
                        received += len(block)
                        if progress:
                            progress(position, size)
                finally:
                    await asyncio.to_thread(target.close)
        except (OSError, asyncssh.Error) as exc:
            raise RemoteError(f"the download of {remote} was interrupted: {exc}") from exc
        if not await asyncio.to_thread(_is_complete, part, size, sha256):
            await asyncio.to_thread(lambda: part.unlink(missing_ok=True))
            raise HashMismatchError(
                f"{remote} does not match its sha256 after the download; it was discarded"
            )
        await asyncio.to_thread(os.replace, part, local)
        return TransferResult(remote, size, sha256, received, offset, False)

    async def close(self) -> None:
        if self._sftp is not None:
            self._sftp.exit()
            await self._sftp.wait_closed()
            self._sftp = None
