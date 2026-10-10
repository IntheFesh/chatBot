"""When the persona card is refreshed on its own: after an import and after a re-split.

**After an import** (R-IMP-011) the hook queues the free part immediately - a new version of
each scope with fresh ``[自动-统计规则]`` (once the profile of the same import is computed) - and
checks, per scope, whether the automatic description is due: never generated, or her messages
have grown by ``persona.regen_ratio`` (10 %) since it was.  A due description is queued as a
one-time batch, off-peak, waiting for ``twin jobs approve <batch>`` (R-LLM-014).  The backfill
command is ``twin persona regenerate``.

**After a re-split of the hold-out** (R-TRN-013) the pre-holdout card describes a period that is
no longer the right one: its statistics are queued again and a new description of that scope is
planned (also waiting for approval).
"""

from __future__ import annotations

from twin.ingest.hooks import HookContext, HookResult, post_import_hook
from twin.profile.holdout import Holdout, on_holdout_change
from twin.profile.persona.jobs import plan_generation, queue_refresh
from twin.profile.persona.refresh import description_due
from twin.services import Services

SCOPES = ("live", "pre_holdout")


@post_import_hook(
    "persona",
    backfill_command="persona regenerate",
    description="refresh the persona card's statistics; queue a new description when it is due",
)
def queue_persona(context: HookContext) -> HookResult:
    services = context.services
    if context.inserted == 0 and context.changed == 0 and not context.first_import:
        return HookResult("skipped", "no new messages, so the persona card is unchanged")
    queue_refresh(services, scope="all", reason="import")
    due = [item for item in (description_due(services, scope) for scope in SCOPES) if item.due]
    if not due:
        return HookResult(
            "queued", "statistics rules of the persona card will be refreshed", jobs=1
        )
    plan = plan_generation(services, [item.scope for item in due], reason="import")
    if plan.batch_id is None:
        return HookResult(
            "queued", "statistics refresh queued; a description is already waiting", jobs=1
        )
    reasons = "; ".join(f"{item.scope}: {item.reason}" for item in due)
    return HookResult(
        "queued",
        f"statistics refresh and a new description ({reasons}), estimated "
        f"${plan.estimated_usd:.2f}; waiting for `twin jobs approve {plan.batch_id}`",
        jobs=1 + len(plan.scopes),
    )


@on_holdout_change("persona")
def persona_after_resplit(services: Services, previous: Holdout | None, current: Holdout) -> str:
    queue_refresh(services, scope="pre_holdout", reason="resplit")
    plan = plan_generation(services, ["pre_holdout"], reason="resplit")
    if plan.batch_id is None:
        return "pre-holdout statistics refresh queued; a description is already waiting"
    return (
        "pre-holdout statistics refresh queued; a new pre-holdout description is planned "
        f"(${plan.estimated_usd:.2f}) and waits for `twin jobs approve {plan.batch_id}`"
    )
