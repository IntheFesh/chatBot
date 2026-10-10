"""The schedule's job: draw the life line of a day when she wakes (R-MEM-005).

``lifeline_generate``
    queued by the schedule component at the moment she wakes (:func:`queue_lifeline`), runs
    :class:`~twin.memory.lifeline_gen.LifelineGenerator` for the plan of that day.  It is an
    ordinary job: it survives a restart, is retried when the model fails twice in a row and goes
    back to the queue when the budget or the circuit breaker holds the calls back.  A plan that was
    replaced in the meantime (the time zone changed) is looked up again by its date and zone; if
    nothing is left of it the job has nothing to do - the new plan queues its own.
"""

from __future__ import annotations

import asyncio
from datetime import date

from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.runtime import build_llm_runtime
from twin.llm.types import DAILY
from twin.memory.lifeline_gen import LifelineGenerator
from twin.memory.memory import Memory
from twin.ops.jobs import JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.schedule.plan_model import DailyPlan
from twin.schedule.service import schedule_kit
from twin.services import Services

log = get_logger("twin.schedule.jobs")

LIFELINE_JOB = "lifeline_generate"
LIFELINE_PRIORITY = 60
LIFELINE_ATTEMPTS = 4


def queue_lifeline(services: Services, plan: DailyPlan) -> str:
    """Queue the life line of ``plan``'s day; returns the job id."""
    queue = JobQueue(services.db, services.clock)
    return queue.enqueue(
        LIFELINE_JOB,
        {"date": plan.local_date.isoformat(), "plan_id": plan.id},
        priority=LIFELINE_PRIORITY,
        max_attempts=LIFELINE_ATTEMPTS,
    )


@job_handler(LIFELINE_JOB)
async def handle_lifeline_generate(ctx: JobContext) -> None:
    if ctx.services is None:
        raise RuntimeError("the life line job needs the services container")
    services = ctx.services
    kit = schedule_kit(services)
    store = kit.planner.store
    day = date.fromisoformat(str(ctx.job.payload["date"]))
    queued = await asyncio.to_thread(store.get, str(ctx.job.payload["plan_id"]))
    if queued is None:
        log.info("lifeline_job_skipped", reason="plan_gone", day=day.isoformat())
        return
    plan = await asyncio.to_thread(store.current_for, queued.local_date, queued.timezone)
    if plan is None:
        log.info("lifeline_job_skipped", reason="plan_replaced", day=day.isoformat())
        return
    runtime = build_llm_runtime(services)
    generator = LifelineGenerator(Memory(services), runtime.client, kit.time)
    try:
        result = await generator.generate(plan, DAILY)
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()
    await asyncio.to_thread(store.mark_lifeline_done, plan.id, services.clock.now_utc())
    log.info(
        "lifeline_job_done",
        day=day.isoformat(),
        events=result.events,
        drafts=result.drafts,
        corrected=result.corrected,
    )
