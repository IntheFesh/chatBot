"""The weekly consistency audit as an off-peak job (R-EVAL-004, R-LLM-007).

``eval_consistency``
    runs :func:`~twin.eval.consistency_audit.run_audit` once.  It is an off-peak job like the
    other offline work (summaries, the weekly consolidation of the rules): it waits for the cheap
    hours and is forced after a day.  A budget that does not allow the call or an open circuit
    hands the job back to the queue; any other failure is retried by the queue.  The audit only
    *finds* contradictions: when it has found some, an ``info`` alert says that they wait for the
    user (``twin eval consistency --review``); the live memory is not touched and no message is
    sent (see :mod:`twin.eval.consistency_audit`).

:func:`queue_consistency_job` is what the weekly task of the operations scheduler
(:func:`twin.ops.scheduler.consistency_task`, every :data:`CONSISTENCY_RULE`) and
``twin eval consistency --queue`` call.  It queues nothing while an audit is already waiting.
"""

from __future__ import annotations

from datetime import timedelta

from twin.eval.consistency_audit import run_audit
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.runtime import build_llm_runtime
from twin.ops.jobs import JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.ops.recurring import Weekly
from twin.services import Services

log = get_logger("twin.eval.consistency")

CONSISTENCY_JOB = "eval_consistency"
CONSISTENCY_PRIORITY = 90
FORCE_AFTER = timedelta(days=1)
CONSISTENCY_RULE = Weekly(weekday=0, hour=4, minute=20)  # Monday, 04:20 on the bot's clock
REVIEW_ALERT = "consistency_review"


def audit_is_waiting(services: Services) -> bool:
    """True while an audit job is queued or running."""
    queue = JobQueue(services.db, services.clock)
    return any(
        queue.list_jobs(status=state, job_type=CONSISTENCY_JOB, limit=1)
        for state in ("pending", "running")
    )


def queue_consistency_job(services: Services, *, days: int | None = None) -> str | None:
    """Queue an audit for the off-peak hours; ``None`` if one is already waiting."""
    if audit_is_waiting(services):
        return None
    now = services.clock.now_utc()
    window = days if days is not None else services.settings.eval.consistency_days
    return JobQueue(services.db, services.clock).enqueue(
        CONSISTENCY_JOB,
        {"days": window},
        priority=CONSISTENCY_PRIORITY,
        offpeak_only=True,
        deadline=now + FORCE_AFTER,
        max_attempts=3,
    )


@job_handler(CONSISTENCY_JOB)
async def handle_consistency(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("the consistency audit needs the services container")
    days = int(ctx.job.payload.get("days") or services.settings.eval.consistency_days)
    runtime = build_llm_runtime(services)
    try:
        outcome = await run_audit(services, runtime.client, days=days)
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()
    if outcome.findings:
        services.alerts.raise_alert(
            REVIEW_ALERT,
            f"the consistency audit found {outcome.findings} contradiction(s) for you to decide: "
            "twin eval consistency --review",
            severity="info",
            detail={"run_id": outcome.run.id, "findings": outcome.findings},
            dedup_key=REVIEW_ALERT,
        )
