"""Image descriptions (R-IMP-012, R-LLM-014, R-LLM-004).

One short, objective Chinese sentence per picture, written by the DeepSeek vision model,
passed through :func:`twin.llm.redaction.redact` and stored encrypted in
``media_assets.caption``.

* **History** (``twin images caption-backfill`` and the post-import hook): pictures of the
  last ``ingest.caption_recent_days`` days are queued as *one-time batches* (R-LLM-014).  The
  command shows the estimated cost and the jobs wait for ``twin jobs approve <batch>``; the
  batch's spending is recorded on the ``one_time`` account.  The jobs are off-peak only.
* **On demand** (:meth:`CaptionService.get_caption`): a picture the person just sent is
  described synchronously, with a timeout (``wait=True``); a historic picture found by
  retrieval returns what is stored or ``None`` and queues a description job
  (``wait=False``).  The reply path never waits for history.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select

from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import ApiError, BudgetDeniedError, CircuitOpenError, ImageError, LlmError
from twin.llm.images import ImageInput
from twin.llm.onetime import BatchItem
from twin.llm.redaction import redact
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY, ChatMessage, LedgerTag, Purpose
from twin.ops.jobs import JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.services import Services
from twin.storage.chat_models import MediaAsset, Message

log = get_logger("twin.ingest.captions")

CAPTION_JOB = "image_caption"
CAPTION_JOB_SIZE = 25
CAPTION_MAX_CHARS = 200
CAPTION_MAX_TOKENS = 120
CAPTION_PROMPT_TOKENS = 80
UNKNOWN_IMAGE_SIDE = 1024
SYSTEM_PROMPT = (
    "你是图片描述助手。用一句客观的中文描述图片里能看到的内容。"
    "只描述画面本身，不要猜测或说出人物的身份、姓名、关系，不要评价，不要加多余的话。"
)
USER_PROMPT = "请用一句话描述这张图片。"


def caption_messages() -> list[ChatMessage]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]


def clean_caption(text: str) -> str:
    """One line, no wrapping quotes, redacted (phone numbers, addresses, ... replaced)."""
    line = " ".join(text.split()).strip("\"'“”‘’「」")
    if len(line) > CAPTION_MAX_CHARS:
        line = line[: CAPTION_MAX_CHARS - 1] + "…"
    return redact(line).text


@dataclass(frozen=True)
class AssetInfo:
    id: str
    sha256: str | None
    status: str
    kind: str
    caption: str | None


@dataclass(frozen=True)
class PlannedBatch:
    batch_id: str
    jobs: int
    images: int
    estimated_usd: float


@dataclass(frozen=True)
class CaptionPlan:
    days: int
    images: int  # pictures that got a job now
    already_queued: int
    batches: list[PlannedBatch] = field(default_factory=list)

    @property
    def estimated_usd(self) -> float:
        return sum(batch.estimated_usd for batch in self.batches)

    @property
    def jobs(self) -> int:
        return sum(batch.jobs for batch in self.batches)


def _identity_of(asset: MediaAsset) -> str:
    """Primary key of an asset object, also when it is detached from its session."""
    identity = sa_inspect(asset).identity
    return str(identity[0]) if identity else asset.id


def queued_asset_ids(queue: JobQueue) -> set[str]:
    """Pictures that already wait in a pending or running description job."""
    found: set[str] = set()
    for status in ("pending", "running"):
        for job in queue.list_jobs(status=status, job_type=CAPTION_JOB, limit=100_000):
            found.update(str(item) for item in job.payload.get("asset_ids", []))
    return found


def plan_caption_batches(
    services: Services, *, days: int, runtime: LlmRuntime | None = None
) -> CaptionPlan:
    """Queue the undescribed pictures of the last ``days`` days as one-time batches.

    Nothing runs until ``twin jobs approve <batch>``; a batch that would cost more than
    ``budget.one_time_usd`` is split into several.
    """
    llm = runtime or build_llm_runtime(services)
    now = services.clock.now_utc()
    cutoff = now - timedelta(days=days)
    queue = JobQueue(services.db, services.clock)
    queued = queued_asset_ids(queue)
    with services.db.session() as session:
        rows = session.execute(
            select(MediaAsset.id, MediaAsset.width, MediaAsset.height)
            .join(Message, Message.id == MediaAsset.message_id)
            .where(
                MediaAsset.kind == "image",
                MediaAsset.status == "available",
                MediaAsset.caption_ct.is_(None),
                Message.create_time_utc >= cutoff,
            )
            .order_by(Message.create_time_utc.desc(), MediaAsset.id)
        ).all()
    candidates = [(row.id, row.width, row.height) for row in rows if row.id not in queued]
    skipped = len(rows) - len(candidates)
    if not candidates:
        return CaptionPlan(days=days, images=0, already_queued=skipped)

    model = services.settings.deepseek.vision_model
    prompt_tokens = llm.estimator.estimate_messages(caption_messages()) + CAPTION_PROMPT_TOKENS
    items = [
        BatchItem(
            model,
            prompt_tokens,
            CAPTION_MAX_TOKENS // 2,
            ((width or UNKNOWN_IMAGE_SIDE, height or UNKNOWN_IMAGE_SIDE),),
        )
        for _, width, height in candidates
    ]
    stamp = now.strftime("%Y%m%d%H%M%S")
    planned: list[PlannedBatch] = []
    position = 0
    for number, chunk in enumerate(llm.batches.split(items), start=1):
        group = candidates[position : position + len(chunk)]
        position += len(chunk)
        estimates = [llm.batches.estimate_item(item)[0] for item in chunk]
        payloads: list[dict[str, Any]] = []
        job_estimates: list[float] = []
        batch_id = f"caption-{stamp}-{number}"
        for start in range(0, len(group), CAPTION_JOB_SIZE):
            part = group[start : start + CAPTION_JOB_SIZE]
            payloads.append({"asset_ids": [item[0] for item in part], "batch_id": batch_id})
            job_estimates.append(sum(estimates[start : start + len(part)]))
        llm.batches.enqueue(batch_id, CAPTION_JOB, payloads, job_estimates, offpeak_only=True)
        planned.append(PlannedBatch(batch_id, len(payloads), len(group), sum(job_estimates)))
    return CaptionPlan(days=days, images=len(candidates), already_queued=skipped, batches=planned)


class CaptionService:
    """Describes pictures on demand and in jobs."""

    def __init__(self, services: Services, client: DeepSeekClient | None = None) -> None:
        self._services = services
        self._client = client
        self._owns_client = client is None

    def _deepseek(self) -> DeepSeekClient:
        if self._client is None:
            self._client = build_llm_runtime(self._services).client
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ----------------------------------------------------------------- lookups

    def _load(self, media: MediaAsset | str) -> AssetInfo | None:
        asset_id = media if isinstance(media, str) else _identity_of(media)
        with self._services.db.session() as session:
            row = session.get(MediaAsset, asset_id)
            if row is None:
                return None
            return AssetInfo(row.id, row.sha256, row.status, row.kind, row.caption)

    def _store(self, asset_id: str, caption: str) -> None:
        now = self._services.clock.now_utc()
        with self._services.db.transaction(bump_state=False) as session:
            row = session.get(MediaAsset, asset_id)
            if row is not None:
                row.caption = caption
                row.caption_at = now

    # -------------------------------------------------------------- generation

    async def generate(self, info: AssetInfo, tag: LedgerTag = DAILY) -> str:
        """Ask the vision model for a description of ``info`` and return the cleaned text."""
        if not info.sha256:
            raise ImageError(f"asset {info.id} has no stored file")
        image = ImageInput.from_media(self._services.media, info.sha256, detail="auto")
        result = await self._deepseek().chat(
            caption_messages(),
            purpose=Purpose.CAPTION,
            images=[image],
            tag=tag,
            max_tokens=CAPTION_MAX_TOKENS,
            temperature=0.2,
        )
        caption = clean_caption(result.content)
        if not caption:
            raise ApiError("the model returned an empty description")
        return caption

    async def caption_asset(self, asset_id: str, tag: LedgerTag = DAILY) -> bool:
        """Describe and store one picture; ``False`` if it needs nothing (already described)."""
        info = await asyncio.to_thread(self._load, asset_id)
        if info is None or info.caption or info.status != "available":
            return False
        caption = await self.generate(info, tag)
        await asyncio.to_thread(self._store, asset_id, caption)
        return True

    # ------------------------------------------------------------------ public

    async def get_caption(self, media: MediaAsset | str, *, wait: bool) -> str | None:
        """The description of a picture (R-IMP-012).

        ``wait=True`` is for a picture the person has just sent: it is described now,
        within ``ingest.caption_wait_timeout_s``; on a timeout or an API problem ``None`` is
        returned and a job finishes the work later.  ``wait=False`` is for historic pictures:
        a stored description is returned, otherwise ``None`` and a job is queued.  Neither
        mode ever blocks the reply path for history.
        """
        info = await asyncio.to_thread(self._load, media)
        if info is None:
            return None
        if info.caption:
            return info.caption
        if info.status != "available" or not info.sha256:
            return None
        if not wait:
            await asyncio.to_thread(self.queue_caption, info.id)
            return None
        timeout = self._services.settings.ingest.caption_wait_timeout_s
        try:
            caption = await asyncio.wait_for(self.generate(info), timeout)
        except (TimeoutError, LlmError):
            log.warning("caption_wait_gave_up", asset_id=info.id)
            await asyncio.to_thread(self.queue_caption, info.id)
            return None
        await asyncio.to_thread(self._store, info.id, caption)
        return caption

    async def describe_stored(self, sha256: str) -> str | None:
        """Describe a picture that is only in the media store, within the wait timeout.

        This is the picture the user has just sent to the bot (a channel message, not a row of
        ``media_assets``): it is described now, as ``get_caption(wait=True)`` does for an imported
        one, and ``None`` comes back on a timeout or when the description cannot be made - the
        reply then goes on without it (R-ENG-013).
        """
        info = AssetInfo(sha256, sha256, "available", "image", None)
        timeout = self._services.settings.ingest.caption_wait_timeout_s
        try:
            return await asyncio.wait_for(self.generate(info), timeout)
        except Exception as exc:  # the description is a nicety; the reply must not depend on it
            log.warning("inbound_caption_gave_up", reason=type(exc).__name__)
            return None

    def queue_caption(self, asset_id: str) -> str | None:
        """Queue a description job for one picture unless one is already waiting."""
        queue = JobQueue(self._services.db, self._services.clock)
        if asset_id in queued_asset_ids(queue):
            return None
        return queue.enqueue(
            CAPTION_JOB,
            {"asset_ids": [asset_id], "batch_id": None},
            priority=200,
            offpeak_only=True,
        )


@job_handler(CAPTION_JOB)
async def handle_caption_job(ctx: JobContext) -> None:
    """Describe the pictures of one ``image_caption`` job (a batch job or a single request)."""
    services = ctx.services
    if services is None:
        raise RuntimeError("image_caption needs the services container")
    payload = ctx.job.payload
    batch_id = payload.get("batch_id")
    tag = LedgerTag("one_time", str(batch_id)) if batch_id else DAILY
    service = CaptionService(services)
    failures = 0
    try:
        for asset_id in payload.get("asset_ids", []):
            try:
                await service.caption_asset(str(asset_id), tag)
            except ImageError:
                log.warning("caption_skipped_unreadable_image", asset_id=str(asset_id))
            except BudgetDeniedError as exc:
                raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
            except CircuitOpenError as exc:
                raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
            except ApiError:
                failures += 1
    finally:
        await service.aclose()
    if failures:
        raise ApiError(f"{failures} picture(s) could not be described yet; the job will retry")
