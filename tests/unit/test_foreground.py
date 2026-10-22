"""Foreground job runs hold the ``run`` lock (R-ARCH-006, R-ARCH-003; round 17 item 27, D-711)."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack

import pytest

from tests.support.clock import ManualClock
from twin.ops.foreground import ApplicationRunningError, foreground_run_lock, run_jobs_until_idle
from twin.ops.instance_lock import LOCK_RUN, LOCK_SUPERVISOR, InstanceLock
from twin.ops.jobs import HandlerRegistry, JobContext, JobQueue
from twin.ops.process_model import (
    CliError,
    CommandSpec,
    ExitCode,
    _guard_locks,
    app_is_running,
    get_spec,
    iter_commands,
)
from twin.services import Services


def can_take(services: Services, name: str) -> bool:
    lock = InstanceLock(name, locks_dir=services.paths.locks_dir)
    taken = lock.acquire()
    lock.release()
    return taken


def hold(services: Services, name: str) -> InstanceLock:
    lock = InstanceLock(name, locks_dir=services.paths.locks_dir)
    assert lock.acquire()
    return lock


def run_spec() -> CommandSpec:
    from twin.cli import app

    spec = get_spec(dict(iter_commands(app))["run"])
    assert spec is not None
    return spec


async def test_a_foreground_run_beside_the_application_does_nothing_and_leaves_its_job_alone(
    services: Services, clock: ManualClock
) -> None:
    """R17-27: the app has a job ``running``; a foreground run must not take it over.

    Before the fix ``run_jobs_until_idle`` recovered every ``running`` row at its start, so the
    application's job went back to ``pending`` and was executed a second time here.
    """
    ran: list[str] = []
    registry = HandlerRegistry()

    async def handler(ctx: JobContext) -> None:
        ran.append(ctx.job.id)

    registry.register("demo", handler)
    queue = JobQueue(services.db, services.clock)
    busy = queue.enqueue("demo", {"n": 1})
    claimed = queue.claim_next({"demo"}, offpeak_allowed=True, worker_id="4242-app")
    assert claimed is not None and claimed.id == busy
    waiting = queue.enqueue("demo", {"n": 2})
    application = hold(services, LOCK_RUN)  # `twin run` of another process
    try:
        with pytest.raises(ApplicationRunningError) as info:
            await run_jobs_until_idle(services, registry)
    finally:
        application.release()
    assert "application is running" in str(info.value) and info.value.lock_name == LOCK_RUN
    assert ran == []  # nothing was executed here, not even the job nobody was running
    job = queue.get(busy)
    assert job is not None and (job.status, job.attempts) == ("running", 1)
    other = queue.get(waiting)
    assert other is not None and other.status == "pending"


async def test_the_foreground_run_holds_the_lock_until_it_has_finished(
    services: Services, clock: ManualClock
) -> None:
    seen: dict[str, object] = {}
    registry = HandlerRegistry()

    async def handler(ctx: JobContext) -> None:
        seen["free_inside"] = can_take(services, LOCK_RUN)
        seen["app_running_inside"] = app_is_running(services)
        # `twin run` is refused with BUSY, and so is a second foreground run
        with ExitStack() as stack:
            try:
                _guard_locks(stack, run_spec(), services.paths.locks_dir)
            except CliError as exc:
                seen["twin_run_code"] = exc.code
        seen["supervisor_free"] = can_take(services, LOCK_SUPERVISOR)

    registry.register("demo", handler)
    JobQueue(services.db, services.clock).enqueue("demo", {})
    summary = await run_jobs_until_idle(services, registry)
    assert summary.done == 1
    assert seen == {
        "free_inside": False,
        "app_running_inside": True,
        "twin_run_code": ExitCode.BUSY,
        "supervisor_free": True,  # only `run` is held: the supervisor lock is another story
    }
    assert can_take(services, LOCK_RUN)  # given back


async def test_the_lock_is_given_back_when_a_job_fails_and_when_the_run_is_cancelled(
    services: Services,
) -> None:
    registry = HandlerRegistry()

    async def boom(ctx: JobContext) -> None:
        raise ValueError("no")

    registry.register("boom", boom)
    JobQueue(services.db, services.clock).enqueue("boom", {}, max_attempts=1)
    summary = await run_jobs_until_idle(services, registry)
    assert summary.failed == 1 and can_take(services, LOCK_RUN)

    with pytest.raises(RuntimeError, match="stop here"), foreground_run_lock(services):
        assert not can_take(services, LOCK_RUN)
        raise RuntimeError("stop here")
    assert can_take(services, LOCK_RUN)


async def test_cancelling_the_run_gives_the_lock_back(
    services: Services,
) -> None:
    started = asyncio.Event()
    registry = HandlerRegistry()

    async def handler(ctx: JobContext) -> None:
        started.set()
        await asyncio.Event().wait()  # until cancelled

    registry.register("demo", handler)
    JobQueue(services.db, services.clock).enqueue("demo", {})
    task = asyncio.create_task(run_jobs_until_idle(services, registry, tick_seconds=0.01))
    await asyncio.wait_for(started.wait(), timeout=10)
    assert not can_take(services, LOCK_RUN)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert can_take(services, LOCK_RUN)


def test_the_lock_helper_refuses_while_the_application_holds_the_lock(services: Services) -> None:
    application = hold(services, LOCK_RUN)
    try:
        with pytest.raises(ApplicationRunningError), foreground_run_lock(services):
            raise AssertionError("must not get in")
    finally:
        application.release()
    with foreground_run_lock(services):
        assert not can_take(services, LOCK_RUN)
