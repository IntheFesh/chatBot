"""The post-import hooks of round 03: picture descriptions and sticker downloads (R-IMP-011).

Both only queue work; the work itself is done by jobs (`image_caption`, `sticker_download`),
and both have a backfill command that does the same for data that is already imported.
"""

from __future__ import annotations

from twin.ingest.captions import plan_caption_batches
from twin.ingest.hooks import HookContext, HookResult, post_import_hook
from twin.stickers.download import queue_sticker_download


@post_import_hook(
    "image_caption",
    backfill_command="images caption-backfill",
    description="queue descriptions for pictures of the last N days (needs `twin jobs approve`)",
)
def queue_image_captions(context: HookContext) -> HookResult:
    services = context.services
    days = services.settings.ingest.caption_recent_days
    plan = plan_caption_batches(services, days=days)
    if plan.images == 0:
        if plan.already_queued:
            return HookResult("skipped", f"{plan.already_queued} picture(s) already queued")
        return HookResult("skipped", f"no undescribed pictures in the last {days} days")
    batches = ", ".join(batch.batch_id for batch in plan.batches)
    return HookResult(
        "queued",
        f"{plan.images} picture(s) of the last {days} days in {len(plan.batches)} batch(es), "
        f"estimated ${plan.estimated_usd:.2f}; waiting for `twin jobs approve <batch>`: {batches}",
        jobs=plan.jobs,
    )


@post_import_hook(
    "sticker_download",
    backfill_command="stickers download",
    description="download the stickers whose files are not in the export",
)
def queue_sticker_downloads(context: HookContext) -> HookResult:
    queued = queue_sticker_download(context.services)
    if queued.already_queued:
        return HookResult("queued", f"a download job is already waiting ({queued.pending} pending)")
    if queued.job_id is None:
        return HookResult("skipped", "no sticker is waiting for a download")
    return HookResult("queued", f"{queued.pending} sticker(s) to download", jobs=1)
