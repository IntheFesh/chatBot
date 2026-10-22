"""Is the process that claimed a job still there? (R-ARCH-003, round 17 item 27, D-712).

A job that is ``running`` carries ``locked_by = "<pid>-<8 hex>"`` of the worker that claimed it.
After a crash that worker is gone and the job must go back to ``pending``; but a *second* process
(``twin jobs run --until-idle --force`` beside the application, or an application that starts
while such a worker is still busy) must not take over a job whose worker is alive and working.

The primary protection is not here: a foreground worker holds the ``run`` lock for as long as it
works (:func:`twin.ops.foreground.foreground_run_lock`), so ``twin run`` cannot start beside it.
This module is the second line, for the rows that nobody holds a lock for.  A pid alone proves
too little - the operating system gives a pid to another program once its owner has died, and a
machine that was restarted after a power loss has handed most low pids out again - so a live pid
counts only if the process is **not younger than the claim**: a process that started after the
job was claimed cannot be the one that claimed it.

* POSIX: ``os.kill(pid, 0)`` for existence; ``/proc/<pid>/stat`` and ``/proc/stat`` for the start
  time where the system has them (elsewhere the start time is unknown and a live pid counts).
* Windows: ``OpenProcess`` + ``GetExitCodeProcess`` and ``GetProcessTimes`` behind the
  :class:`~twin.ops.winapi.Win32` protocol, which the tests replace.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from twin.ops.winapi import Win32, load_win32

START_SLACK = timedelta(seconds=5)  # the clocks of the database and of the system differ a little
DEFAULT_CLOCK_TICKS = 100  # CLK_TCK of every Linux that matters


def worker_pid(worker_id: str | None) -> int | None:
    """The pid in a worker id (``"<pid>-<8 hex>"``); ``None`` for anything else."""
    head = (worker_id or "").split("-", 1)[0]
    pid = int(head) if head.isascii() and head.isdigit() else 0
    return pid if pid > 0 else None


def _clock_ticks() -> int:
    sysconf = getattr(os, "sysconf", None)  # not on Windows
    try:
        return int(sysconf("SC_CLK_TCK")) if sysconf is not None else DEFAULT_CLOCK_TICKS
    except (ValueError, OSError):
        return DEFAULT_CLOCK_TICKS


def proc_started_at(pid: int, proc_root: Path) -> datetime | None:
    """When ``pid`` started, from a ``/proc`` tree; ``None`` if the tree has no such process."""
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
        system = (proc_root / "stat").read_text(encoding="utf-8", errors="replace")
        # the name of the program (field 2) may hold spaces and brackets: count from its end
        fields = stat[stat.rindex(")") + 1 :].split()
        started_ticks = int(fields[19])  # field 22 of proc(5), counting from field 3 at index 0
        boot_lines = (line for line in system.splitlines() if line.startswith("btime"))
        boot = int(next(boot_lines).split()[1])
    except (OSError, ValueError, IndexError, StopIteration):
        return None
    return datetime.fromtimestamp(boot + started_ticks / _clock_ticks(), UTC)


@dataclass
class ProcessProbe:
    """Existence and age of processes, on the platform given (the tests give another one)."""

    platform: str | None = None
    win32: Win32 | None = None
    own_pid: int | None = None
    proc_root: Path = field(default_factory=lambda: Path("/proc"))

    def _is_windows(self) -> bool:
        return (sys.platform if self.platform is None else self.platform) == "win32"

    def _api(self) -> Win32:
        return self.win32 if self.win32 is not None else load_win32()

    def exists(self, pid: int) -> bool:
        if self._is_windows():
            return self._api().process_alive(pid)
        try:
            os.kill(pid, 0)  # signal 0 only checks; never use it on Windows (it ends the process)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # another user's process: there, and not ours
        except OSError:
            return False
        return True

    def started_at(self, pid: int) -> datetime | None:
        if self._is_windows():
            seconds = self._api().process_started_at(pid)
            return None if seconds is None else datetime.fromtimestamp(seconds, UTC)
        return proc_started_at(pid, self.proc_root)

    def owner_alive(self, worker_id: str | None, claimed_at: datetime | None) -> bool:
        """Whether the worker that claimed a job at ``claimed_at`` is still working on it.

        ``False`` for a worker id without a pid, for this very process (the worker of this
        process that claimed the job has been abandoned: a restart inside one process, as the
        tests do, or a cancelled task), for a pid that is not there, and for a pid whose process
        started after the claim (the pid was handed to another program).
        """
        pid = worker_pid(worker_id)
        if pid is None or pid == (os.getpid() if self.own_pid is None else self.own_pid):
            return False
        if not self.exists(pid):
            return False
        if claimed_at is None:
            return True
        started = self.started_at(pid)
        return started is None or started <= claimed_at + START_SLACK
