"""Children that cannot outlive their parent: a Windows job object (R-OPS-001).

``twin supervise`` starts ``twin run``; ``twin run`` starts ``llama-server`` (round 14) and holds an
SSH tunnel.  If the parent is killed - from the task manager, by a crash, by ``schtasks /End`` -
nothing would tell the children, and a ``llama-server`` that keeps the GPU and its port would make
the next start fail.  A *job object* created with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` ends every
process in it when its last handle is closed, and the operating system closes the handles of a
dead process: so the children die with their parent, however it died.

Two ways to use it:

* :meth:`ProcessJob.adopt_current_process` puts **this** process in the job; every process it
  starts afterwards is in the job too (``twin run`` does this, so its ``llama-server`` is covered);
  the handle is never closed explicitly - closing it would end this process;
* :meth:`ProcessJob.assign` puts one **child** in the job by its process id (``twin supervise`` does
  this for ``twin run``); :meth:`ProcessJob.close` then ends whatever is left of it.

Jobs may be nested on Windows 8 and later, so a ``twin run`` that is already in the job of its
supervisor can have a job of its own.  Elsewhere there is nothing to do and nothing is done.
"""

from __future__ import annotations

import sys

from twin.ops.logging import get_logger
from twin.ops.winapi import Win32, load_win32

log = get_logger("twin.jobobject")


class ProcessJob:
    """A kill-on-close job object (inactive off Windows)."""

    def __init__(self, win32: Win32 | None = None, platform: str | None = None) -> None:
        self._platform = sys.platform if platform is None else platform
        self._win32 = win32
        self._job: int | None = None
        self._adopted = False

    @property
    def active(self) -> bool:
        return self._job is not None

    def _api(self) -> Win32:
        if self._win32 is None:
            self._win32 = load_win32()
        return self._win32

    def open(self) -> bool:
        """Create the job; ``False`` where there are no job objects or it could not be made."""
        if self._job is not None:
            return True
        if self._platform != "win32":
            return False
        job = self._api().create_kill_on_close_job()
        if job is None:
            log.warning("job_object_unavailable")
            return False
        self._job = job
        return True

    def adopt_current_process(self) -> bool:
        """Put this process in the job: its children are then ended with it."""
        if not self.open() or self._job is None:
            return False
        api = self._api()
        if not api.assign_to_job(self._job, api.current_process_handle()):
            log.warning("job_object_adopt_failed")
            return False
        self._adopted = True
        return True

    def assign(self, pid: int) -> bool:
        """Put the process ``pid`` (a child) in the job."""
        if not self.open() or self._job is None:
            return False
        api = self._api()
        handle = api.open_process(pid)
        if handle is None:
            log.warning("job_object_open_process_failed")
            return False
        try:
            assigned = api.assign_to_job(self._job, handle)
        finally:
            api.close_handle(handle)
        if not assigned:
            log.warning("job_object_assign_failed")
        return assigned

    def close(self) -> None:
        """End the processes in the job.  Does nothing if this process is one of them: the
        operating system closes the handle when the process ends."""
        job, self._job = self._job, None
        if job is not None and not self._adopted:
            self._api().close_handle(job)
