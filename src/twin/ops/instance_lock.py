"""Single-instance locks (R-OPS-002, R-ARCH-006.1).

Two independent locks exist: ``run`` (held by ``twin run``) and ``supervisor``
(held by ``twin supervise``, round 12).  Different names never conflict.

* Windows: a named mutex ``Local\\wechat-twin-<name>`` (``CreateMutexW``); the
  operating system removes it when the owning process dies.
* Other platforms: an advisory ``flock`` on ``<locks_dir>/<name>.lock``.

``is_held_elsewhere()`` is a non-destructive probe used by EXCLUSIVE commands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

from twin.ops.filelock import FileLock
from twin.ops.winapi import ERROR_ALREADY_EXISTS, Win32, load_win32

LOCK_RUN = "run"
LOCK_SUPERVISOR = "supervisor"
ALL_LOCKS = (LOCK_RUN, LOCK_SUPERVISOR)


class InstanceAlreadyRunningError(RuntimeError):
    """Another instance holds the lock."""

    def __init__(self, name: str) -> None:
        self.lock_name = name
        super().__init__(
            f"another '{name}' instance is already running (lock '{name}' is held). "
            "Stop it first, or use the running application."
        )


class _Backend(Protocol):
    def try_acquire(self) -> bool: ...

    def release(self) -> None: ...


class _MutexBackend:
    def __init__(self, win32: Win32, name: str) -> None:
        self._win32 = win32
        self._mutex_name = f"Local\\wechat-twin-{name}"
        self._handle: int | None = None

    def try_acquire(self) -> bool:
        handle, error = self._win32.create_mutex(self._mutex_name)
        if handle is None:
            raise OSError(f"CreateMutexW failed for {self._mutex_name} (error {error})")
        if error == ERROR_ALREADY_EXISTS:
            self._win32.close_handle(handle)
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is not None:
            self._win32.close_handle(self._handle)
            self._handle = None


class _FileBackend:
    def __init__(self, path: Path) -> None:
        self._lock = FileLock(path)

    def try_acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()


class InstanceLock:
    """A named, process-wide lock."""

    def __init__(
        self,
        name: str,
        *,
        locks_dir: Path,
        platform: str | None = None,
        win32: Win32 | None = None,
    ) -> None:
        self.name = name
        plat = sys.platform if platform is None else platform
        self._backend: _Backend
        if plat == "win32":
            self._backend = _MutexBackend(win32 or load_win32(), name)
        else:
            self._backend = _FileBackend(locks_dir / f"{name}.lock")
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> bool:
        """Take the lock; ``False`` if another process holds it."""
        if self._held:
            raise RuntimeError(f"lock '{self.name}' is already held by this object")
        self._held = self._backend.try_acquire()
        return self._held

    def acquire_or_raise(self) -> None:
        if not self.acquire():
            raise InstanceAlreadyRunningError(self.name)

    def release(self) -> None:
        if self._held:
            self._backend.release()
            self._held = False

    def is_held_elsewhere(self) -> bool:
        """Probe without keeping the lock (never true for a lock this object holds)."""
        if self._held:
            return False
        if self.acquire():
            self.release()
            return False
        return True

    def __enter__(self) -> InstanceLock:
        self.acquire_or_raise()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def locks_held_elsewhere(
    locks_dir: Path,
    names: tuple[str, ...] = ALL_LOCKS,
    *,
    platform: str | None = None,
    win32: Win32 | None = None,
) -> list[str]:
    """Names of the given locks that some other process currently holds."""
    return [
        name
        for name in names
        if InstanceLock(
            name, locks_dir=locks_dir, platform=platform, win32=win32
        ).is_held_elsewhere()
    ]
