"""The weekly job of the learning: consolidate the rules of ``[不要这样]`` (R-LRN-003).

``learning_rules``
    runs :class:`~twin.learning.rules.RuleConsolidator` once.  It is an off-peak job (R-LLM-007):
    it waits for the cheap hours and is forced after a day.  A budget that does not allow the call
    or an open circuit hands the job back to the queue; any other failure is retried by the queue
    and leaves the card and the feedback as they were.

:func:`queue_if_due` is what the learning component calls every hour: when a week
(``learning.rules_interval_days``) has passed since the last time, and there is feedback nobody has
used yet, it queues the job.  :func:`queue_rules_job` queues it now (``twin persona rules
consolidate``).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from twin.engine.feedback import FeedbackStore
from twin.engine.turns import BotTurnStore
from twin.learning.rules import RuleConsolidator
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.runtime import build_llm_runtime
from twin.ops.jobs import JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.services import Services
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.learning.jobs")

RULES_JOB = "learning_rules"
RULES_PRIORITY = 85
LAST_QUEUED_KEY = "learning.rules.last_queued"
FORCE_AFTER = timedelta(days=1)


def _waiting(services: Services) -> bool:
    queue = JobQueue(services.db, services.clock)
    return any(
        queue.list_jobs(status=s, job_type=RULES_JOB, limit=1) for s in ("pending", "running")
    )


def queue_rules_job(services: Services) -> str | None:
    """Queue a consolidation now; ``None`` if one is already waiting."""
    if _waiting(services):
        return None
    now = services.clock.now_utc()
    job_id = JobQueue(services.db, services.clock).enqueue(
        RULES_JOB,
        {},
        priority=RULES_PRIORITY,
        offpeak_only=True,
        deadline=now + FORCE_AFTER,
        max_attempts=3,
    )
    with services.db.transaction(bump_state=False) as session:
        put_setting(
            session,
            LAST_QUEUED_KEY,
            now.isoformat(),
            clock=services.clock,
            by="learning",
            record_history=False,
        )
    return job_id


def last_queued(services: Services) -> datetime | None:
    with services.db.session() as session:
        raw = get_setting(session, LAST_QUEUED_KEY)
    try:
        return datetime.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def queue_if_due(services: Services) -> str | None:
    """Queue the weekly consolidation when it is due and there is something to learn from."""
    now = services.clock.now_utc()
    last = last_queued(services)
    interval = timedelta(days=services.settings.learning.rules_interval_days)
    if last is not None and now - last < interval:
        return None
    if not FeedbackStore(services.db, services.clock).unprocessed():
        return None
    return queue_rules_job(services)


@job_handler(RULES_JOB)
async def handle_rules(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("the learning job needs the services container")
    runtime = build_llm_runtime(services)
    consolidator = RuleConsolidator(
        services,
        runtime.client,
        FeedbackStore(services.db, services.clock),
        BotTurnStore(services.db, services.clock),
    )
    try:
        await consolidator.run()
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()
