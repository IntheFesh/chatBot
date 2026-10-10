"""Queueing the index run (R-ARCH-003, R-IMP-011, R-RET-006).

One job type, ``retrieval_index``, syncs the example windows and encodes what is missing.  The
import hook, ``twin retrieval rebuild`` and the hold-out re-split only need to put the job in
the queue; nothing here imports the embedding model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from twin.ops.jobs import JobQueue
from twin.profile.queue import Queued
from twin.retrieval.indexer import INDEX_JOB

if TYPE_CHECKING:
    from twin.services import Services

INDEX_PRIORITY = 70


def queue_retrieval_index(
    services: Services,
    *,
    mode: str = "update",
    full: bool = False,
    reason: str = "manual",
    run_id: str | None = None,
) -> Queued:
    """Queue an index run unless an identical one is already waiting."""
    if mode not in ("update", "rebuild"):
        raise ValueError("mode must be update or rebuild")
    queue = JobQueue(services.db, services.clock)
    for job in queue.list_jobs(status="pending", job_type=INDEX_JOB, limit=200):
        if job.payload.get("mode") == mode and bool(job.payload.get("full")) == full:
            return Queued(job.id, True)
    payload = {"mode": mode, "full": full, "reason": reason, "run_id": run_id}
    job_id = queue.enqueue(INDEX_JOB, payload, priority=INDEX_PRIORITY, max_attempts=3)
    return Queued(job_id, False)
