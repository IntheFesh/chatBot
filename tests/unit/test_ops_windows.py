"""Round 12 against the real Windows: Task Scheduler, job objects, powercfg (R-OPS-001, R-OPS-009).

These run only on Windows (``pytest.mark.windows``; CI skips them elsewhere).  They touch the
machine in small ways and clean up after themselves: a scheduled task with a name of its own that
is registered and removed again, a job object, a child process that is killed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from twin.config.loader import parse_overrides
from twin.ops.doctor import CheckStatus, DoctorContext, check_power_plan, sleep_timeouts
from twin.ops.jobobject import ProcessJob
from twin.ops.service_cli import global_options, task_spec
from twin.ops.taskscheduler import SubprocessRunner, TaskScheduler, TaskSpec, decode_output
from twin.ops.winapi import load_win32
from twin.services import CliContext

pytestmark = pytest.mark.windows


def current_user() -> str:
    domain, name = os.environ.get("USERDOMAIN", ""), os.environ["USERNAME"]
    return f"{domain}\\{name}" if domain else name


@pytest.fixture
def scheduler() -> Iterator[TaskScheduler]:
    """A scheduled task of the tests' own, removed at the end whatever happened."""
    real = TaskScheduler(SubprocessRunner(), name=f"wechat-twin-test-{uuid.uuid4().hex[:8]}")
    try:
        yield real
    finally:
        real.uninstall()


def spec() -> TaskSpec:
    return TaskSpec(
        command=os.environ.get("COMSPEC", "cmd.exe"),
        arguments="/c exit 0",
        working_dir=str(Path.cwd()),
        user_id=current_user(),
    )


def test_windows_accepts_the_task_definition_and_reads_back_what_was_registered(
    scheduler: TaskScheduler,
) -> None:
    assert scheduler.info().registered is False
    scheduler.install(spec())
    info = scheduler.info()
    assert info.registered and info.interactive and info.time_limit == "PT0S"
    assert info.multiple_instances == "IgnoreNew" and info.on_battery_allowed is True
    assert info.restart_interval == "PT1M" and info.enabled is True and not info.runs_uv_sync
    assert info.arguments == "/c exit 0"
    assert scheduler.state() is not None
    scheduler.install(spec())  # registering again replaces the task (/F)
    assert scheduler.uninstall() is True
    assert scheduler.uninstall() is False


def windows_argv(command_line: str) -> list[str]:
    """How Windows itself splits a command line into the arguments a program receives."""
    import ctypes
    from ctypes import wintypes

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    count = ctypes.c_int()
    argv = shell32.CommandLineToArgvW(command_line, ctypes.byref(count))
    if not argv:
        raise OSError(ctypes.get_last_error(), "CommandLineToArgvW failed")
    try:
        return [str(argv[i]) for i in range(count.value)]
    finally:
        kernel32.LocalFree(argv)


def test_windows_splits_the_task_arguments_into_the_options_that_were_given(
    tmp_path: Path,
) -> None:
    """The child of the supervisor must see the data folder and config the user chose.

    The options go through JSON and ``list2cmdline`` into the task's arguments; Windows splits
    them again when it starts the program (spaces, quotes and backslashes in the folder names).
    """
    config = tmp_path / "my config" / "twin.yaml"
    data = tmp_path / '数据 & "备份" dir' / "data"
    context = CliContext(
        config_path=config,
        overrides={"paths": {"data_dir": str(data)}},
        log_level="DEBUG",
    )
    options = global_options(context)
    spec = task_spec(tmp_path, options)
    received = windows_argv(f"twin.exe {spec.arguments}")[1:]  # [0] is the program
    assert received == [*options, "supervise", "--from-task"]
    sets = [received[i + 1] for i, item in enumerate(received) if item == "--set"]
    assert parse_overrides(sets) == {"paths": {"data_dir": str(data)}}
    assert received[received.index("--config") + 1] == str(config)


def test_the_task_can_be_started_by_hand(scheduler: TaskScheduler) -> None:
    scheduler.install(spec())
    scheduler.start()
    scheduler.end()  # (it may have finished already: ending a task that is not running fails)


def test_a_task_that_does_not_exist_is_reported_as_such() -> None:
    absent = TaskScheduler(SubprocessRunner(), name="wechat-twin-test-does-not-exist")
    assert absent.info().registered is False and absent.state() is None


def test_powercfg_answers_in_a_form_the_doctor_can_read() -> None:
    result = SubprocessRunner().run(
        ["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"]
    )
    assert result.returncode == 0
    assert sleep_timeouts(decode_output(result.stdout)) is not None
    check = check_power_plan(DoctorContext(None, platform="win32"))
    assert check.status in (CheckStatus.OK, CheckStatus.WARN)


def sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def wait_gone(process: subprocess.Popen[bytes], seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    return False


def test_closing_the_job_ends_the_child_that_was_put_in_it() -> None:
    job = ProcessJob(load_win32())
    assert job.open()
    child = sleeper()
    try:
        assert job.assign(child.pid)
        assert child.poll() is None
        job.close()
        assert wait_gone(child)
    finally:
        child.kill()


def test_a_child_outside_the_job_is_not_touched() -> None:
    job = ProcessJob(load_win32())
    assert job.open()
    inside, outside = sleeper(), sleeper()
    try:
        assert job.assign(inside.pid)
        job.close()
        assert wait_gone(inside)
        assert outside.poll() is None
    finally:
        inside.kill()
        outside.kill()


PARENT = textwrap.dedent(
    """
    import subprocess, sys, time
    from twin.ops.jobobject import ProcessJob

    assert ProcessJob().adopt_current_process()
    beat = (
        "import pathlib, sys, time\\n"
        "while True:\\n"
        "    pathlib.Path(sys.argv[1]).write_text(str(time.time()))\\n"
        "    time.sleep(0.1)\\n"
    )
    subprocess.Popen([sys.executable, "-c", beat, sys.argv[1]])
    print("started", flush=True)
    time.sleep(120)
    """
)


def test_when_a_process_that_adopted_itself_dies_its_children_die_with_it(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT, str(heartbeat)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert parent.stdout is not None and parent.stdout.readline().strip() == "started"
        deadline = time.monotonic() + 10
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert heartbeat.exists()  # the grandchild is alive and writing
        first = heartbeat.read_text()
        time.sleep(0.5)
        assert heartbeat.read_text() != first
        parent.kill()
        parent.wait(timeout=10)
        time.sleep(1.5)  # the system ends the job's processes when the last handle is gone
        last = heartbeat.read_text()
        time.sleep(1.0)
        assert heartbeat.read_text() == last  # nobody writes any more: the grandchild is gone
    finally:
        parent.kill()
