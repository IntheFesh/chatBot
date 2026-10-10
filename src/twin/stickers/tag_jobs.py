"""Planning and running the sticker tagging jobs (R-STK-003, R-LLM-014, R-IMP-011).

One job type, ``sticker_tag``, does three things for a few stickers (``stickers.tag_job_size``):
tags the ones never tagged by their picture, makes the context correction of those she used
often enough before the hold-out cutoff, and finally brings the description vectors up to date.
Every step saves its result at once, so a job that stops half way continues where it was.

*Who pays.*  The first tagging of the whole library is a one-time batch (R-LLM-014): the jobs are
queued waiting for ``twin jobs approve <batch>``, priced in advance, recorded on the ``one_time``
account.  Later runs - a new sticker after an import, a correction that became due - are small
and go on the daily account, off-peak, without waiting for anybody.  ``twin stickers tag-all``
always uses a batch, because the person asks for it and wants to see the price first.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sqlalchemy import func, select

from twin.llm.errors import ApiError, BudgetDeniedError, CircuitOpenError, ImageError, LlmError
from twin.llm.onetime import BatchItem
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY, LedgerTag
from twin.ops.jobs import JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.profile.holdout import HoldoutError, holdout_cutoff
from twin.profile.prompt_templates import STICKER_CONTEXT, STICKER_TAG, TemplateStore
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.stickers.tagging import (
    CONTEXT_COMPLETION_TOKENS,
    VISION_COMPLETION_TOKENS,
    StickerTagger,
    context_due,
    her_use_counts_before,
)
from twin.stickers.vectors import StickerVectors
from twin.storage.chat_models import Sticker
from twin.storage.ids import new_id

log = get_logger("twin.stickers.tag_jobs")

TAG_JOB = "sticker_tag"
BATCH_PRIORITY = 130
DAILY_PRIORITY = 140
DEFAULT_SIDE = 512
CONTEXT_PROMPT_CHARS = 5 * 7 * 126


@dataclass(frozen=True)
class TagWork:
    """The stickers that need work, in the order they are done (hers first)."""

    vision: list[str]
    context: list[str]
    stale_vectors: int

    @property
    def stickers(self) -> list[str]:
        return list(dict.fromkeys([*self.vision, *self.context]))

    @property
    def empty(self) -> bool:
        return not self.vision and not self.context and not self.stale_vectors


@dataclass
class TagPlan:
    """What :func:`plan_tagging` queued."""

    mode: str  # "batch", "daily" or "none"
    stickers: int = 0
    contexts: int = 0
    jobs: int = 0
    estimated_usd: float = 0.0
    batch_ids: list[str] = field(default_factory=list)
    already_queued: int = 0


def queued_stickers(queue: JobQueue) -> set[str]:
    """Stickers that wait in a pending or running tagging job."""
    found: set[str] = set()
    for status in ("pending", "running"):
        for job in queue.list_jobs(status=status, job_type=TAG_JOB, limit=100_000):
            found.update(str(m) for m in job.payload.get("vision", []))
            found.update(str(m) for m in job.payload.get("context", []))
    return found


def jobs_waiting(queue: JobQueue) -> int:
    """How many tagging jobs are pending or running."""
    return sum(
        len(queue.list_jobs(status=status, job_type=TAG_JOB, limit=100_000))
        for status in ("pending", "running")
    )


def find_work(services: Services, catalog: StickerCatalog | None = None) -> TagWork:
    """The stickers that have no tags yet, are due a correction, or have a stale vector."""
    library = catalog or StickerCatalog(services)
    records = {r.md5: r for r in library.records(status="available")}
    vision = [m for m, r in records.items() if not r.vision_tags and r.sha256]
    context: list[str] = []
    try:
        cutoff = holdout_cutoff(services)
    except HoldoutError:
        cutoff = None
    if cutoff is not None:
        minimum = services.settings.stickers.context_min_uses
        for md5, uses in her_use_counts_before(services, cutoff, minimum).items():
            record = records.get(md5)
            if record is not None and record.sha256 and context_due(minimum, record, cutoff, uses):
                context.append(md5)
    with services.db.session() as session:
        stale_vectors = int(
            session.scalar(
                select(func.count())
                .select_from(Sticker)
                .where(
                    Sticker.her_uses > 0,
                    Sticker.description_ct.is_not(None),
                    Sticker.desc_encoding.is_(None),
                )
            )
            or 0
        )
    order = {md5: index for index, md5 in enumerate(records)}  # her most used first
    return TagWork(
        sorted(vision, key=order.__getitem__),
        sorted(context, key=lambda m: order.get(m, len(order))),
        stale_vectors,
    )


def never_tagged(services: Services) -> bool:
    """True before any sticker has been tagged: the first tagging of the library."""
    with services.db.session() as session:
        return (
            session.scalar(select(Sticker.md5).where(Sticker.tagged_at.is_not(None)).limit(1))
            is None
        )


def _estimate(
    services: Services,
    runtime: LlmRuntime,
    catalog: StickerCatalog,
    vision: list[str],
    context: list[str],
) -> dict[str, float]:
    """Price per sticker (picture call plus correction call), peak prices, upper bound."""
    templates = TemplateStore(services.db, services.clock)
    vocabulary = "、".join(catalog.vocabulary.tags)
    estimator = runtime.estimator
    model = services.settings.deepseek.vision_model
    tag_prompt = estimator.estimate_messages(
        templates.active(STICKER_TAG).render(vocabulary=vocabulary)
    )
    context_prompt = estimator.estimate_messages(
        templates.active(STICKER_CONTEXT).render(vocabulary=vocabulary, uses="")
    )
    context_extra = estimator.estimate_text("字" * CONTEXT_PROMPT_CHARS)
    prices: dict[str, float] = {}
    wants_picture, wants_context = set(vision), set(context)
    for md5 in dict.fromkeys([*vision, *context]):
        record = catalog.get(md5)
        side = (
            (record.width or DEFAULT_SIDE, record.height or DEFAULT_SIDE)
            if record
            else (DEFAULT_SIDE,) * 2
        )
        items = []
        if md5 in wants_picture:
            items.append(BatchItem(model, tag_prompt, VISION_COMPLETION_TOKENS, (side,)))
        if md5 in wants_context:
            items.append(
                BatchItem(model, context_prompt + context_extra, CONTEXT_COMPLETION_TOKENS, (side,))
            )
        prices[md5] = runtime.batches.estimate(items).total_usd
    return prices


def plan_tagging(
    services: Services,
    *,
    batch: bool | None = None,
    runtime: LlmRuntime | None = None,
) -> TagPlan:
    """Queue the tagging work that is not queued yet (see the module description)."""
    catalog = StickerCatalog(services)
    work = find_work(services, catalog)
    queue = JobQueue(services.db, services.clock)
    waiting = queued_stickers(queue)
    vision = [m for m in work.vision if m not in waiting]
    context = [m for m in work.context if m not in waiting]
    skipped = len(work.stickers) - len(set(vision) | set(context))
    if not vision and not context:
        if not work.stale_vectors or jobs_waiting(queue):
            return TagPlan("none", already_queued=skipped)
        # nothing to look at, but some description has no vector: a job that only indexes
        queue.enqueue(
            TAG_JOB,
            {"vision": [], "context": [], "batch_id": None},
            priority=DAILY_PRIORITY,
            offpeak_only=True,
        )
        return TagPlan("daily", jobs=1)
    use_batch = never_tagged(services) if batch is None else batch
    size = services.settings.stickers.tag_job_size
    ordered = list(dict.fromkeys([*vision, *context]))
    chunks = [ordered[i : i + size] for i in range(0, len(ordered), size)]
    pictures, corrections = set(vision), set(context)
    payloads = [
        {
            "vision": [m for m in chunk if m in pictures],
            "context": [m for m in chunk if m in corrections],
        }
        for chunk in chunks
    ]
    plan = TagPlan("batch" if use_batch else "daily", len(ordered), len(context), len(payloads))
    plan.already_queued = skipped
    if not use_batch:
        for payload in payloads:
            queue.enqueue(
                TAG_JOB, {**payload, "batch_id": None}, priority=DAILY_PRIORITY, offpeak_only=True
            )
        return plan
    llm = runtime or build_llm_runtime(services)
    prices = _estimate(services, llm, catalog, vision, context)
    estimates = [sum(prices.get(m, 0.0) for m in {*p["vision"], *p["context"]}) for p in payloads]
    stamp = f"{services.clock.now_utc():%Y%m%d%H%M%S}-{new_id()[-4:].lower()}"
    limit = llm.batches.limit_usd
    groups: list[list[int]] = [[]]
    running = 0.0
    for index, estimate in enumerate(estimates):
        if groups[-1] and running + estimate > limit:
            groups.append([])
            running = 0.0
        groups[-1].append(index)
        running += estimate
    for number, group in enumerate(groups, start=1):
        batch_id = f"stickers-{stamp}-{number}"
        llm.batches.enqueue(
            batch_id,
            TAG_JOB,
            [{**payloads[i], "batch_id": batch_id} for i in group],
            [estimates[i] for i in group],
            priority=BATCH_PRIORITY,
            offpeak_only=True,
        )
        plan.batch_ids.append(batch_id)
    plan.estimated_usd = sum(estimates)
    return plan


# --------------------------------------------------------------------- running


async def run_tagging(
    services: Services,
    tagger: StickerTagger,
    vision: list[str],
    context: list[str],
    tag: LedgerTag,
) -> tuple[int, int, int]:
    """Do the work of one job; returns ``(tagged, corrected, failures)``."""
    tagged = corrected = failures = 0
    for md5 in vision:
        try:
            tagged += int(await tagger.tag_sticker(md5, tag))
        except ImageError:
            log.warning("sticker_picture_unreadable", md5=md5)
        except (BudgetDeniedError, CircuitOpenError):
            raise
        except LlmError:
            failures += 1
    for md5 in context:
        try:
            corrected += int(await tagger.correct_with_context(md5, tag))
        except ImageError:
            log.warning("sticker_picture_unreadable", md5=md5)
        except (BudgetDeniedError, CircuitOpenError):
            raise
        except LlmError:
            failures += 1
    return tagged, corrected, failures


@job_handler(TAG_JOB)
async def handle_sticker_tag(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("sticker_tag needs the services container")
    payload = ctx.job.payload
    batch_id = payload.get("batch_id")
    tag = LedgerTag("one_time", str(batch_id)) if batch_id else DAILY
    vision = [str(m) for m in payload.get("vision", [])]
    context = [str(m) for m in payload.get("context", [])]
    runtime = build_llm_runtime(services)
    tagger = StickerTagger(services, runtime.client)
    try:
        tagged, corrected, failures = await run_tagging(services, tagger, vision, context, tag)
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()
    synced = await asyncio.to_thread(StickerVectors(services).sync)
    log.info(
        "stickers_tagged",
        tagged=tagged,
        corrected=corrected,
        failures=failures,
        vectors=synced.encoded,
    )
    if failures:
        raise ApiError(f"{failures} sticker(s) could not be tagged yet; the job will retry")
