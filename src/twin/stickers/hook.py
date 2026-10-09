"""When stickers are tagged on their own: after an import, after a download, after a re-split.

**After an import** (R-IMP-011) the hook looks for stickers that have a stored file but no tags,
and for stickers she used often enough before the hold-out cutoff to be judged by their use
(R-STK-003).  The first tagging of the library is a one-time batch waiting for
``twin jobs approve <batch>`` (R-LLM-014); later, small amounts go on the daily account.  The
backfill command is ``twin stickers tag-all``.  A sticker that is still being downloaded cannot
be looked at yet: the download job runs the same planning when it has finished.

**After a re-split of the hold-out** (R-TRN-013) the corrections made for the old cutoff may have
used messages that are held out now.  They are dropped at once and made again for the new cutoff.
"""

from __future__ import annotations

from twin.ingest.hooks import HookContext, HookResult, post_import_hook
from twin.profile.holdout import Holdout, on_holdout_change
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.stickers.tag_jobs import TagPlan, plan_tagging


def describe_plan(plan: TagPlan) -> str:
    """One line about a tagging plan (no sticker content)."""
    corrections = f", {plan.contexts} correction(s) from her use" if plan.contexts else ""
    if plan.mode == "batch":
        batches = ", ".join(plan.batch_ids)
        return (
            f"{plan.stickers} sticker(s){corrections} in {plan.jobs} job(s), estimated "
            f"${plan.estimated_usd:.2f}; waiting for `twin jobs approve <batch>`: {batches}"
        )
    return f"{plan.stickers} sticker(s){corrections} in {plan.jobs} job(s) queued off-peak"


@post_import_hook(
    "sticker_tag",
    backfill_command="stickers tag-all",
    description="tag new stickers by picture and by her use; the first tagging needs approval",
)
def queue_sticker_tagging(context: HookContext) -> HookResult:
    plan = plan_tagging(context.services, batch=None)
    if plan.mode == "none":
        if plan.already_queued:
            return HookResult("skipped", f"{plan.already_queued} sticker(s) already queued")
        return HookResult("skipped", "no sticker is waiting to be tagged")
    return HookResult("queued", describe_plan(plan), jobs=plan.jobs)


@on_holdout_change("stickers")
def stickers_after_resplit(services: Services, previous: Holdout | None, current: Holdout) -> str:
    cleared = StickerCatalog(services).clear_stale_context(current.cutoff)
    plan = plan_tagging(services, batch=False)
    queued = describe_plan(plan) if plan.mode != "none" else "nothing to queue"
    return f"{cleared} sticker correction(s) made for the old cutoff dropped; {queued}"
