"""The ``import`` job: runs an import run inside the job queue (R-ARCH-006, R-IMP-006).

``twin import`` creates the run and queues this job; the running application (or
``twin jobs run --until-idle`` / ``twin import --foreground``) executes it.  The import
itself is synchronous work, so it runs in a worker thread; a shutdown asks it to stop at the
next batch boundary, after which the run is ``interrupted`` and the job goes back to the
queue (the attempt is not counted) to continue from the saved progress.
"""

from __future__ import annotations

import asyncio
import threading

from twin.ingest.importer import ImportRunner
from twin.ops.jobs import JobContext, JobDeferred, JobQueue, job_handler
from twin.services import Services

IMPORT_JOB = "import"


@job_handler(IMPORT_JOB)
async def handle_import(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("the import job needs the services container")
    run_id = str(ctx.job.payload["run_id"])
    stop = threading.Event()
    work = asyncio.ensure_future(asyncio.to_thread(ImportRunner(services).run, run_id, stop))
    try:
        outcome = await asyncio.shield(work)
    except asyncio.CancelledError:
        stop.set()  # finish the current batch, save progress, then stop
        await asyncio.gather(work, return_exceptions=True)
        raise
    if outcome.status != "done":
        raise JobDeferred(
            "the import was interrupted and continues from its saved progress", retry_in_s=5.0
        )


def import_job_ids(services: Services, run_id: str) -> list[str]:
    """Pending or running ``import`` jobs of ``run_id``."""
    queue = JobQueue(services.db, services.clock)
    found = []
    for status in ("pending", "running"):
        for job in queue.list_jobs(status=status, job_type=IMPORT_JOB):
            if job.payload.get("run_id") == run_id:
                found.append(job.id)
    return found


def queue_import(services: Services, run_id: str) -> str:
    """Queue the ``import`` job for ``run_id`` (once)."""
    existing = import_job_ids(services, run_id)
    if existing:
        return existing[0]
    return JobQueue(services.db, services.clock).enqueue(
        IMPORT_JOB, {"run_id": run_id}, priority=50, max_attempts=3
    )
