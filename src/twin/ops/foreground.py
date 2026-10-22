"""Running selected job types in the foreground of a CLI command (R-ARCH-006).

``twin import --foreground`` and ``twin stickers download --foreground`` run their jobs
here when the application is not running.  Only the job types of the given registry are
executed, so a foreground import does not also start every other queued job.

A foreground worker holds the ``run`` lock for as long as it works (:func:`foreground_run_lock`,
D-711).  That is what makes "the application is not running" true for the whole run and not just
at the moment it was looked at: ``twin run`` cannot start beside the worker, and a worker that
cannot get the lock does nothing - in particular it does not return the application's ``running``
jobs to ``pending`` and execute them a second time.  The lock is released by the operating system
when the process dies, so a crashed worker never blocks the application.

The lock is for HEAVY commands.  An EXCLUSIVE command already holds ``run`` for its whole body
(D-710) and must not call this.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager

from twin.llm.runtime import activate_offpeak_policy
from twin.ops.instance_lock import LOCK_RUN, InstanceAlreadyRunningError, InstanceLock
from twin.ops.jobs import HandlerRegistry, JobQueue, RunSummary, Worker
from twin.ops.winapi import Win32
from twin.services import Services

TICK_SECONDS = 0.5


class ApplicationRunningError(InstanceAlreadyRunningError):
    """The ``run`` lock is held, so the application is running (or another foreground worker).

    An :class:`~twin.ops.instance_lock.InstanceAlreadyRunningError`, which the command wrapper
    turns into the exit code BUSY with this message.
    """

    def __init__(self) -> None:
        super().__init__(
            LOCK_RUN,
            "the application is running and executes the queued jobs itself; "
            "nothing was run here (a job that was queued stays queued)",
        )


@contextmanager
def foreground_run_lock(
    services: Services, *, platform: str | None = None, win32: Win32 | None = None
) -> Iterator[None]:
    """Hold the ``run`` lock while a CLI process executes jobs; refuse if it is taken (D-711)."""
    lock = InstanceLock(
        LOCK_RUN, locks_dir=services.paths.locks_dir, platform=platform, win32=win32
    )
    if not lock.acquire():
        raise ApplicationRunningError
    try:
        yield
    finally:
        lock.release()


async def run_jobs_until_idle(
    services: Services,
    registry: HandlerRegistry,
    *,
    on_tick: Callable[[], Awaitable[None]] | None = None,
    tick_seconds: float = TICK_SECONDS,
) -> RunSummary:
    """Run the queued jobs of ``registry``'s types to completion; ``on_tick`` runs periodically.

    Raises :class:`ApplicationRunningError` (exit code BUSY) without doing anything if the
    ``run`` lock is held by another process.
    """
    with foreground_run_lock(services):
        activate_offpeak_policy(services)
        worker = Worker(
            JobQueue(services.db, services.clock),
            registry,
            services.clock,
            services=services,
            alerts=services.alerts,
            concurrency=1,
        )
        await worker.recover()
        task = asyncio.ensure_future(worker.run_until_idle())
        try:
            while not task.done():
                if on_tick is not None:
                    await on_tick()
                await asyncio.wait({task}, timeout=tick_seconds)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if on_tick is not None:
            await on_tick()
        return task.result()
