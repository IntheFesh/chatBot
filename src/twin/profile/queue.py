"""Queueing the profile and routine recomputation (R-ARCH-003, R-IMP-011).

One job type, ``profile_rebuild``, recomputes the style profile and the activity model of the
named scope(s).  Nothing here imports the heavy computation: the import hook, ``twin profile
rebuild`` and the hold-out re-split only need to put the job in the queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from twin.ops.jobs import JobQueue

if TYPE_CHECKING:
    from twin.services import Services

PROFILE_JOB = "profile_rebuild"
SCOPE_CHOICES = ("live", "pre_holdout", "all")
PROFILE_PRIORITY = 60


@dataclass(frozen=True)
class Queued:
    job_id: str
    already_queued: bool


def queue_profile_rebuild(
    services: Services,
    *,
    scope: str = "all",
    reason: str = "manual",
    run_id: str | None = None,
    force: bool = False,
) -> Queued:
    """Queue a recomputation unless an identical one is already waiting."""
    if scope not in SCOPE_CHOICES:
        raise ValueError(f"scope must be one of {', '.join(SCOPE_CHOICES)}")
    queue = JobQueue(services.db, services.clock)
    for job in queue.list_jobs(status="pending", job_type=PROFILE_JOB, limit=200):
        if job.payload.get("scope") == scope and bool(job.payload.get("force")) == force:
            return Queued(job.id, True)
    payload = {"scope": scope, "reason": reason, "run_id": run_id, "force": force}
    job_id = queue.enqueue(PROFILE_JOB, payload, priority=PROFILE_PRIORITY, max_attempts=3)
    return Queued(job_id, False)
