"""The ``import`` job and how it behaves inside the worker (R-ARCH-006, R-IMP-006)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from tests.support.clock import ManualClock
from tests.support.ingest import make_export, start_import
from tests.support.waiting import wait_until
from twin.ingest.importer import ImportRunner, RunOutcome
from twin.ingest.jobs import IMPORT_JOB, handle_import, import_job_ids, queue_import
from twin.ingest.runs import get_run
from twin.ops.components import build_application
from twin.ops.jobs import (
    HANDLER_MODULES,
    HandlerRegistry,
    JobContext,
    JobQueue,
    Worker,
    default_registry,
    load_handlers,
)
from twin.services import Services


def registry() -> HandlerRegistry:
    handlers = HandlerRegistry()
    handlers.register(IMPORT_JOB, handle_import)
    return handlers


def worker(services: Services) -> Worker:
    return Worker(
        JobQueue(services.db, services.clock),
        registry(),
        services.clock,
        services=services,
        alerts=services.alerts,
    )


def test_the_handlers_of_this_round_are_loaded_with_the_others() -> None:
    assert {"twin.ingest.jobs", "twin.ingest.captions", "twin.stickers.download"} <= set(
        HANDLER_MODULES
    )
    load_handlers()
    for job_type in ("import", "image_caption", "sticker_download"):
        assert default_registry.has(job_type)


def test_one_job_is_queued_per_run(services: Services, tmp_path: Path) -> None:
    run_id = start_import(services, make_export(tmp_path, target_messages=10))
    first = queue_import(services, run_id)
    assert queue_import(services, run_id) == first
    assert import_job_ids(services, run_id) == [first]
    job = JobQueue(services.db, services.clock).get(first)
    assert job is not None and job.payload == {"run_id": run_id} and not job.offpeak_only


async def test_the_worker_runs_an_import_job_to_the_end(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=60)
    run_id = start_import(services, export)
    job_id = queue_import(services, run_id)
    summary = await worker(services).run_until_idle()
    assert summary.done == 1 and summary.failed == 0
    run = get_run(services.db, run_id)
    assert run is not None and run.status == "done" and run.processed == 60
    job = JobQueue(services.db, services.clock).get(job_id)
    assert job is not None and job.status == "done"
    assert import_job_ids(services, run_id) == []


async def test_a_failed_import_fails_the_job_with_a_safe_message(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=60)
    data = export.messages_path.read_bytes()
    export.messages_path.write_bytes(data[: len(data) // 2])
    run_id = start_import(services, export)
    job_id = queue_import(services, run_id)
    summary = await worker(services).run_until_idle()
    assert summary.retried == 1 and summary.failed == 0  # it will be tried again
    job = JobQueue(services.db, services.clock).get(job_id)
    assert job is not None and job.last_error is not None
    assert "ImportFailure" in job.last_error
    for sentence in export.texts:
        assert sentence not in job.last_error


async def test_a_shutdown_stops_the_import_at_a_batch_boundary_and_hands_the_job_back(
    services: Services, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = make_export(tmp_path, target_messages=30)
    run_id = start_import(services, export)
    started = threading.Event()
    stop_seen = threading.Event()

    def slow_run(self: ImportRunner, rid: str, stop: threading.Event | None = None) -> RunOutcome:
        started.set()
        assert stop is not None and stop.wait(10), "the handler must ask the import to stop"
        stop_seen.set()
        view = get_run(services.db, rid)
        assert view is not None
        return RunOutcome("interrupted", view)

    monkeypatch.setattr(ImportRunner, "run", slow_run)
    queue = JobQueue(services.db, services.clock)
    job_id = queue_import(services, run_id)
    claimed = queue.claim_next({IMPORT_JOB}, offpeak_allowed=True, worker_id="w1")
    assert claimed is not None and claimed.id == job_id
    context = JobContext(job=claimed, services=services, clock=services.clock)
    task = asyncio.create_task(handle_import(context))
    await wait_until(started.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stop_seen.is_set()


async def test_an_interrupted_run_is_deferred_not_failed(
    services: Services, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = make_export(tmp_path, target_messages=30)
    run_id = start_import(services, export)

    def interrupted(
        self: ImportRunner, rid: str, stop: threading.Event | None = None
    ) -> RunOutcome:
        view = get_run(services.db, rid)
        assert view is not None
        return RunOutcome("interrupted", view)

    monkeypatch.setattr(ImportRunner, "run", interrupted)
    job_id = queue_import(services, run_id)
    summary = await worker(services).run_until_idle()
    assert summary.deferred == 1 and summary.failed == 0 and summary.done == 0
    job = JobQueue(services.db, services.clock).get(job_id)
    assert job is not None and job.status == "pending" and job.attempts == 0


async def test_the_handler_needs_the_services_container() -> None:
    from datetime import UTC, datetime

    from tests.support.clock import ManualClock
    from twin.ops.jobs import JobView

    now = datetime(2026, 10, 9, tzinfo=UTC)
    job = JobView(
        id="j",
        type=IMPORT_JOB,
        payload={"run_id": "x"},
        priority=1,
        status="running",
        attempts=1,
        max_attempts=3,
        run_after=now,
        offpeak_only=False,
        deadline=None,
        last_error=None,
        batch_id=None,
        estimated_cost_usd=None,
        requires_approval=False,
        approved_at=None,
        approved_usd=None,
        started_at=None,
        finished_at=None,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(RuntimeError, match="services"):
        await handle_import(JobContext(job=job, services=None, clock=ManualClock()))


async def test_the_running_application_executes_a_queued_import(
    services: Services, tmp_path: Path, clock: ManualClock
) -> None:
    """R-ARCH-006.3: with the application running, ``twin import`` only queues; the app imports."""
    export = make_export(tmp_path, target_messages=80)
    run_id = start_import(services, export)
    application, _watcher = build_application(services)
    await application.start()
    try:
        # state watcher, heartbeat and job worker are all waiting for their next turn
        await wait_until(lambda: clock.pending_sleepers >= 3)
        queue_import(services, run_id)
        await clock.advance(3)  # the worker polls every two seconds of the application clock

        def finished() -> bool:
            run = get_run(services.db, run_id)
            return run is not None and run.status == "done"

        await wait_until(finished, limit_s=60)
    finally:
        await application.stop()
    run = get_run(services.db, run_id)
    assert run is not None and run.processed == 80 and run.hooks
    assert import_job_ids(services, run_id) == []
