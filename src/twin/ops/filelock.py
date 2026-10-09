"""Advisory cross-process file lock (POSIX ``flock``, Windows ``msvcrt.locking``).

Used for the non-Windows single-instance locks (R-OPS-002, task G.1) and to
serialise read-modify-write cycles of the encrypted secrets file.  The lock is
tied to an open file descriptor, so it disappears automatically if the process
dies; nothing is left behind that could make a later start-up believe an
instance is still running.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import TracebackType
from typing import Self


class FileLock:
    """An exclusive advisory lock on ``path`` (the file is created if needed)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, *, blocking: bool = False) -> bool:
        """Take the lock.  Returns ``False`` if it is held elsewhere and not ``blocking``."""
        if self._fd is not None:
            raise RuntimeError(f"lock {self.path} is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if not _lock_fd(fd, blocking):
                os.close(fd)
                return False
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return True

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            _unlock_fd(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> Self:
        self.acquire(blocking=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


if sys.platform == "win32":  # pragma: win32-only
    import msvcrt

    def _lock_fd(fd: int, blocking: bool) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        while True:
            try:
                msvcrt.locking(fd, mode, 1)
            except OSError:
                if blocking:
                    continue  # LK_LOCK gives up after ~10 s; keep waiting
                return False
            return True

    def _unlock_fd(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_fd(fd: int, blocking: bool) -> bool:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            return False
        return True

    def _unlock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
