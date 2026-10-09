"""Single-instance locks (R-OPS-002, R-ARCH-006.1) and the file lock beneath them."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.support.win32 import FakeWin32
from twin.ops.filelock import FileLock
from twin.ops.instance_lock import (
    ALL_LOCKS,
    LOCK_RUN,
    LOCK_SUPERVISOR,
    InstanceAlreadyRunningError,
    InstanceLock,
    locks_held_elsewhere,
)
from twin.ops.winapi import load_win32


def test_second_instance_with_the_same_name_cannot_start(tmp_path: Path) -> None:
    first = InstanceLock(LOCK_RUN, locks_dir=tmp_path)
    second = InstanceLock(LOCK_RUN, locks_dir=tmp_path)
    assert first.acquire() is True and first.held
    assert second.acquire() is False and not second.held
    with pytest.raises(InstanceAlreadyRunningError, match="'run' instance is already running"):
        second.acquire_or_raise()
    first.release()
    assert second.acquire() is True
    second.release()


def test_differently_named_locks_do_not_conflict(tmp_path: Path) -> None:
    run = InstanceLock(LOCK_RUN, locks_dir=tmp_path)
    supervisor = InstanceLock(LOCK_SUPERVISOR, locks_dir=tmp_path)
    assert run.acquire() and supervisor.acquire()
    run.release()
    supervisor.release()


def test_probe_is_non_destructive(tmp_path: Path) -> None:
    holder = InstanceLock(LOCK_RUN, locks_dir=tmp_path)
    probe = InstanceLock(LOCK_RUN, locks_dir=tmp_path)
    assert probe.is_held_elsewhere() is False
    assert holder.acquire()
    assert probe.is_held_elsewhere() is True
    assert probe.is_held_elsewhere() is True  # probing twice changes nothing
    assert holder.is_held_elsewhere() is False  # a lock we hold ourselves is not "elsewhere"
    holder.release()
    assert probe.is_held_elsewhere() is False


def test_context_manager_and_double_acquire(tmp_path: Path) -> None:
    with InstanceLock(LOCK_RUN, locks_dir=tmp_path) as lock:
        assert lock.held
        with pytest.raises(RuntimeError, match="already held"):
            lock.acquire()
        with pytest.raises(InstanceAlreadyRunningError):
            InstanceLock(LOCK_RUN, locks_dir=tmp_path).acquire_or_raise()
    assert not lock.held
    lock.release()  # releasing twice is harmless


def test_locks_held_elsewhere_lists_the_busy_ones(tmp_path: Path) -> None:
    assert locks_held_elsewhere(tmp_path) == []
    supervisor = InstanceLock(LOCK_SUPERVISOR, locks_dir=tmp_path)
    assert supervisor.acquire()
    assert locks_held_elsewhere(tmp_path) == [LOCK_SUPERVISOR]
    assert locks_held_elsewhere(tmp_path, (LOCK_RUN,)) == []
    run = InstanceLock(LOCK_RUN, locks_dir=tmp_path)
    assert run.acquire()
    assert locks_held_elsewhere(tmp_path) == list(ALL_LOCKS)[:2] or set(
        locks_held_elsewhere(tmp_path)
    ) == {LOCK_RUN, LOCK_SUPERVISOR}


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX file lock in a child process")
def test_lock_is_held_across_processes_and_freed_when_the_holder_dies(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys, time
        from pathlib import Path
        from twin.ops.instance_lock import InstanceLock
        lock = InstanceLock("run", locks_dir=Path(sys.argv[1]))
        assert lock.acquire()
        print("locked", flush=True)
        time.sleep(60)
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        probe = InstanceLock(LOCK_RUN, locks_dir=tmp_path)
        assert probe.is_held_elsewhere() is True
        assert probe.acquire() is False
    finally:
        child.kill()
        child.wait()
    probe = InstanceLock(LOCK_RUN, locks_dir=tmp_path)
    assert probe.acquire() is True  # the OS dropped the dead process's lock
    probe.release()


# ---------------------------------------------- Windows logic via the test double


def test_windows_backend_uses_named_mutexes_and_detects_existing_ones(tmp_path: Path) -> None:
    registry: dict[str, int] = {}
    one = InstanceLock(LOCK_RUN, locks_dir=tmp_path, platform="win32", win32=FakeWin32(registry))
    two = InstanceLock(LOCK_RUN, locks_dir=tmp_path, platform="win32", win32=FakeWin32(registry))
    assert one.acquire() is True
    assert two.acquire() is False  # CreateMutexW reported ERROR_ALREADY_EXISTS
    assert registry == {"Local\\wechat-twin-run": 1}  # the probing handle was closed again
    assert two.is_held_elsewhere() is True
    one.release()
    assert registry == {"Local\\wechat-twin-run": 0}
    assert two.acquire() is True
    two.release()
    assert not list(tmp_path.iterdir())  # no lock file on Windows


def test_windows_mutex_names_differ_per_lock(tmp_path: Path) -> None:
    registry: dict[str, int] = {}
    run = InstanceLock(LOCK_RUN, locks_dir=tmp_path, platform="win32", win32=FakeWin32(registry))
    supervisor = InstanceLock(
        LOCK_SUPERVISOR, locks_dir=tmp_path, platform="win32", win32=FakeWin32(registry)
    )
    assert run.acquire() and supervisor.acquire()
    assert set(registry) == {"Local\\wechat-twin-run", "Local\\wechat-twin-supervisor"}


def test_windows_mutex_creation_failure_is_an_error(tmp_path: Path) -> None:
    fake = FakeWin32()
    fake.mutex_fails = True
    lock = InstanceLock(LOCK_RUN, locks_dir=tmp_path, platform="win32", win32=fake)
    with pytest.raises(OSError, match="CreateMutexW failed"):
        lock.acquire()


@pytest.mark.skipif(sys.platform == "win32", reason="the real binding exists only on Windows")
def test_the_real_windows_binding_is_unavailable_elsewhere() -> None:
    with pytest.raises(RuntimeError, match="only available on Windows"):
        load_win32()


@pytest.mark.windows
def test_real_named_mutex_blocks_a_second_instance(tmp_path: Path) -> None:
    name = "test-" + tmp_path.name[-8:]
    first = InstanceLock(name, locks_dir=tmp_path)
    second = InstanceLock(name, locks_dir=tmp_path)
    assert first.acquire() is True
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()


@pytest.mark.windows
def test_real_win32_binding_sets_execution_state_and_console() -> None:
    api = load_win32()
    from twin.ops.winapi import ES_CONTINUOUS, ES_SYSTEM_REQUIRED

    assert api.set_thread_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    assert api.set_thread_execution_state(ES_CONTINUOUS) != 0


# ------------------------------------------------------------------ file lock


def test_file_lock_basics(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "x.lock"
    first = FileLock(path)
    second = FileLock(path)
    assert first.acquire() and first.held
    assert second.acquire(blocking=False) is False
    with pytest.raises(RuntimeError, match="already held"):
        first.acquire()
    first.release()
    with second:
        assert second.held
    assert not second.held
