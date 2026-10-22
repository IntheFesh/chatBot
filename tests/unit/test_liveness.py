"""Is the worker that claimed a job still there? (R-ARCH-003; round 17 item 27, D-712).

Three layers: the parsing of the worker id, the two platform answers (a ``/proc`` tree on POSIX, the
:class:`~twin.ops.winapi.Win32` calls on Windows - both are tested on every platform with a fake
tree and ``FakeWin32``), and real processes started by the test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.support.win32 import FakeWin32
from twin.ops import liveness
from twin.ops.liveness import START_SLACK, ProcessProbe, proc_started_at, worker_pid

CLAIMED = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
BOOT = int(CLAIMED.timestamp()) - 86_400


@pytest.mark.parametrize(
    ("worker_id", "pid"),
    [
        ("4242-1a2b3c4d", 4242),
        ("7-x", 7),
        ("dead-process", None),
        ("the-running-app", None),
        ("", None),
        (None, None),
        ("0-abcd", None),
        ("-5-abcd", None),
        ("12a-abcd", None),
        ("٣٤-abcd", None),  # Arabic-Indic digits are not a pid
    ],
)
def test_the_pid_is_the_part_before_the_first_dash(worker_id: str | None, pid: int | None) -> None:
    assert worker_pid(worker_id) == pid


# ------------------------------------------------------------------ a /proc tree


def write_proc(
    root: Path, pid: int, *, comm: str, start_ticks: int, boot: int | None = BOOT
) -> None:
    """The two files of a ``/proc`` tree that the start time is read from."""
    folder = root / str(pid)
    folder.mkdir(parents=True, exist_ok=True)
    after_name = ["S", "1"] + ["0"] * 17 + [str(start_ticks)] + ["0"] * 29  # state is field 3
    (folder / "stat").write_text(f"{pid} ({comm}) " + " ".join(after_name) + "\n", encoding="utf-8")
    lines = ["cpu  1 2 3 4"] + ([f"btime {boot}"] if boot is not None else []) + ["processes 99"]
    (root / "stat").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def hundred_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(liveness, "_clock_ticks", lambda: 100)


def test_the_start_time_is_the_boot_time_plus_the_ticks_since_boot(
    tmp_path: Path, hundred_ticks: None
) -> None:
    write_proc(tmp_path, 4242, comm="python", start_ticks=360_000)
    assert proc_started_at(4242, tmp_path) == datetime.fromtimestamp(BOOT + 3600, UTC)


def test_a_program_name_with_spaces_and_brackets_does_not_shift_the_fields(
    tmp_path: Path, hundred_ticks: None
) -> None:
    write_proc(tmp_path, 4243, comm="my (odd) prog 9", start_ticks=100)
    assert proc_started_at(4243, tmp_path) == datetime.fromtimestamp(BOOT + 1, UTC)


def test_an_unreadable_tree_gives_no_start_time(tmp_path: Path, hundred_ticks: None) -> None:
    assert proc_started_at(1, tmp_path) is None  # no such process
    write_proc(tmp_path, 5, comm="x", start_ticks=1, boot=None)
    assert proc_started_at(5, tmp_path) is None  # no boot time
    (tmp_path / "6").mkdir()
    (tmp_path / "6" / "stat").write_text("6 (x) S 1\n", encoding="utf-8")
    assert proc_started_at(6, tmp_path) is None  # too short
    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "stat").write_text("garbage\n", encoding="utf-8")
    assert proc_started_at(7, tmp_path) is None


def test_the_clock_ticks_come_from_the_system_or_default_to_one_hundred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert liveness._clock_ticks() > 0

    def broken(name: str) -> int:
        raise ValueError(name)

    monkeypatch.setattr(os, "sysconf", broken, raising=False)
    assert liveness._clock_ticks() == liveness.DEFAULT_CLOCK_TICKS
    monkeypatch.delattr(os, "sysconf", raising=False)  # Windows has none
    assert liveness._clock_ticks() == liveness.DEFAULT_CLOCK_TICKS


# ------------------------------------------------------------------- the answers


def test_the_posix_answer_is_the_answer_of_signal_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = ProcessProbe(platform="linux")
    outcomes: dict[int, BaseException | None] = {
        1: None,
        2: ProcessLookupError(),
        3: PermissionError(),
        4: OSError("odd"),
    }

    def kill(pid: int, sig: int) -> None:
        assert sig == 0  # a real signal would end the process
        outcome = outcomes[pid]
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(liveness.os, "kill", kill)
    assert [probe.exists(pid) for pid in (1, 2, 3, 4)] == [True, False, True, False]


def test_the_windows_answer_comes_from_the_win32_calls() -> None:
    api = FakeWin32()
    api.processes = {4242: CLAIMED.timestamp() - 60}
    probe = ProcessProbe(platform="win32", win32=api)
    assert probe.exists(4242) and not probe.exists(4243)
    assert probe.started_at(4242) == CLAIMED - timedelta(seconds=60)
    assert probe.started_at(4243) is None


def test_a_windows_worker_that_is_alive_and_older_than_the_claim_is_the_owner() -> None:
    api = FakeWin32()
    api.processes = {4242: CLAIMED.timestamp() - 60}
    probe = ProcessProbe(platform="win32", win32=api, own_pid=1)
    assert probe.owner_alive("4242-aaaa", CLAIMED)


def test_a_posix_worker_is_judged_by_its_proc_entry(
    tmp_path: Path, hundred_ticks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(liveness.os, "kill", lambda pid, sig: None)  # every pid exists
    write_proc(tmp_path, 4242, comm="python", start_ticks=100 * (86_400 - 60))
    write_proc(tmp_path, 4243, comm="python", start_ticks=100 * (86_400 + 600))
    probe = ProcessProbe(platform="linux", proc_root=tmp_path, own_pid=1)
    assert probe.owner_alive("4242-aaaa", CLAIMED)  # started a minute before the claim
    assert not probe.owner_alive("4243-aaaa", CLAIMED)  # started ten minutes after it
    assert probe.owner_alive("4244-aaaa", CLAIMED)  # no /proc entry (macOS): the pid counts


def test_a_pid_that_belongs_to_a_younger_process_is_not_the_owner() -> None:
    """The machine was restarted, or the owner died: the pid now belongs to another program."""
    api = FakeWin32()
    api.processes = {4242: CLAIMED.timestamp() + START_SLACK.total_seconds() + 1}
    probe = ProcessProbe(platform="win32", win32=api, own_pid=1)
    assert not probe.owner_alive("4242-aaaa", CLAIMED)
    api.processes[4242] = CLAIMED.timestamp() + START_SLACK.total_seconds() - 1
    assert probe.owner_alive("4242-aaaa", CLAIMED)  # the clocks of two programs differ a little


def test_what_cannot_be_the_owner_is_not() -> None:
    api = FakeWin32()
    api.processes = {4242: CLAIMED.timestamp() - 5, 77: None}
    probe = ProcessProbe(platform="win32", win32=api, own_pid=4242)
    assert not probe.owner_alive("4242-aaaa", CLAIMED)  # a worker of this very process is gone
    assert not probe.owner_alive("dead-process", CLAIMED)  # no pid in the id
    assert not probe.owner_alive(None, CLAIMED)
    assert not probe.owner_alive("4300-aaaa", CLAIMED)  # no such process
    assert probe.owner_alive("77-aaaa", CLAIMED)  # alive, start time unreadable: do not guess
    assert probe.owner_alive("77-aaaa", None)  # a claim without a time: only the pid counts


# --------------------------------------------------------------- real processes


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield process
    finally:
        process.kill()
        process.wait()


def test_a_real_process_is_alive_until_it_ends_and_is_older_than_a_claim_made_after_it(
    sleeper: subprocess.Popen[bytes],
) -> None:
    probe = ProcessProbe()
    claimed = datetime.now(UTC)
    assert probe.exists(sleeper.pid)
    started = probe.started_at(sleeper.pid)
    assert started is not None and started <= claimed + START_SLACK
    assert probe.owner_alive(f"{sleeper.pid}-aaaa", claimed)
    sleeper.kill()
    sleeper.wait()
    assert not probe.owner_alive(f"{sleeper.pid}-aaaa", claimed)


def test_a_real_process_that_started_after_the_claim_is_not_its_owner(
    sleeper: subprocess.Popen[bytes],
) -> None:
    """A claim from an hour ago: the process started a moment ago cannot have made it."""
    claimed = datetime.now(UTC) - timedelta(hours=1)
    assert ProcessProbe().exists(sleeper.pid)
    assert not ProcessProbe().owner_alive(f"{sleeper.pid}-aaaa", claimed)


def test_this_process_is_never_the_owner_of_a_job_that_needs_recovering() -> None:
    assert not ProcessProbe().owner_alive(f"{os.getpid()}-aaaa", datetime.now(UTC))
