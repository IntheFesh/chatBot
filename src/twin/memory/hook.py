"""After an import: replay the new days into the memory (R-IMP-011, R-MEM-010, R-MEM-011).

The import brings new messages - more history, or earlier days that were missing.  The hook finds
the local days that have not been replayed (or whose messages changed) and queues them as a
one-time batch (:func:`twin.memory.replay.plan_replay`).  Two cases:

* the first replay, or a large increment: the batch waits for ``twin jobs approve <batch>``
  (the report gives the estimate);
* an *incremental* replay whose estimate is below ``memory.replay_auto_approve_ratio`` (10 %) of
  ``budget.one_time_usd``: approved at once.

While those days are replayed, a real fact that contradicts something the bot made up replaces
it, and a life line entry it contradicts is marked invalid (R-MEM-011): that happens in the
writer (:mod:`twin.memory.writer`), so no separate pass is needed here.

The backfill command is ``twin memory replay start``.
"""

from __future__ import annotations

from twin.ingest.hooks import HookContext, HookResult, post_import_hook
from twin.memory.replay import ReplayPlan, plan_replay
from twin.ops.logging import get_logger

log = get_logger("twin.memory.hook")


def describe_plan(plan: ReplayPlan) -> str:
    """One line about a queued replay (counts and money only)."""
    days = len(plan.estimate.days)
    batches = ", ".join(batch.batch_id for batch in plan.batches)
    if plan.approved:
        return (
            f"{days} new day(s) in {plan.jobs} job(s), estimated ${plan.estimated_usd:.2f}; "
            f"approved automatically ({batches})"
        )
    return (
        f"{days} day(s) in {plan.jobs} job(s), estimated ${plan.estimated_usd:.2f}; "
        f"waiting for `twin jobs approve <batch>`: {batches}"
    )


@post_import_hook(
    "memory_replay",
    backfill_command="memory replay start",
    description="replay new days of the records into the memory (summaries, facts, follow-ups)",
)
def queue_memory_replay(context: HookContext) -> HookResult:
    services = context.services
    if not (services.settings.memory.fact_extraction or services.settings.memory.daily_summary):
        return HookResult("skipped", "memory.fact_extraction and memory.daily_summary are off")
    plan = plan_replay(services, auto_approve=True, reason="import")
    if not plan.estimate.days:
        if plan.already_queued:
            return HookResult("skipped", f"{plan.already_queued} day(s) already queued")
        return HookResult("skipped", "every day of the records has been replayed")
    return HookResult("queued", describe_plan(plan), jobs=plan.jobs)
