"""One-time batch budgets (R-LLM-014).

Big one-off jobs (full memory replay, first image captioning, sticker tagging, persona card
generation, plan synthesis, evaluation generation) are run as a *batch*: a set of jobs that
share a ``batch_id``.  The life of a batch:

1. **estimate** - :meth:`OneTimeBatches.estimate` prices the work with the token estimator,
   the measured image token counts and the price table (peak prices, the safe upper bound);
2. **enqueue** - :meth:`OneTimeBatches.enqueue` refuses a batch whose estimate exceeds
   ``budget.one_time_usd`` (use :meth:`OneTimeBatches.split`) and queues the jobs as
   ``requires_approval`` so no worker touches them;
3. **approve** - ``twin jobs approve <batch>`` calls :meth:`OneTimeBatches.approve`, which
   records the approval time and amount and sets the overspend cap (estimate plus 20%);
4. **run** - every model call of the batch is recorded on the ``one_time`` account with the
   ``batch_id``; after each call :meth:`OneTimeBatches.after_call` compares the actual spend with
   the cap.  Above it the batch is *paused*: the approval of its unfinished jobs is withdrawn,
   an alert is raised, and the next call of a running handler raises
   :class:`BatchPausedError` (a :class:`~twin.ops.jobs.JobDeferred`, so the job goes back to
   the queue without losing an attempt).  Approving again continues with a new cap.

One-time spending never counts toward the daily and monthly budget or its degradation levels.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from twin.clock import Clock
from twin.config.settings import BudgetConfig
from twin.llm.ledger import LedgerStore
from twin.llm.pricing import Pricing
from twin.llm.tokens import TokenEstimator
from twin.llm.types import ChatMessage
from twin.ops.alerts import AlertSink
from twin.ops.jobs import BatchApproval, BatchTooLargeError, JobDeferred, JobQueue
from twin.ops.logging import get_logger
from twin.storage.db import Database
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.llm.onetime")

OVERRUN_FACTOR = 1.2
STATE_PREFIX = "onetime.batch."


class BatchPausedError(JobDeferred):
    """The batch was paused for overspending; approve it again to continue."""

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id
        super().__init__(
            f"batch {batch_id} is paused (spend above the estimate)", retry_in_s=3600.0
        )


class BatchGuard(Protocol):
    """Checks the DeepSeek client makes around calls that belong to a batch."""

    def before_call(self, batch_id: str) -> None: ...

    def after_call(self, batch_id: str) -> None: ...


@dataclass(frozen=True)
class BatchItem:
    """One planned model call, in tokens."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    image_sizes: tuple[tuple[int, int], ...] = ()  # (width, height) of every image
    cache_hit_ratio: float = 0.0


@dataclass(frozen=True)
class BatchEstimate:
    items: int
    prompt_tokens: int
    completion_tokens: int
    image_tokens: int
    per_item_usd: tuple[float, ...]

    @property
    def total_usd(self) -> float:
        return sum(self.per_item_usd)


@dataclass(frozen=True)
class BatchStatus:
    batch_id: str
    estimated_usd: float
    spent_usd: float
    cap_usd: float | None
    approved_at: str | None
    paused_at: str | None

    @property
    def approved(self) -> bool:
        return self.approved_at is not None

    @property
    def paused(self) -> bool:
        return self.paused_at is not None

    @property
    def overrun_ratio(self) -> float:
        return self.spent_usd / self.estimated_usd if self.estimated_usd > 0 else 0.0


class OneTimeBatches:
    """Estimating, approving, tracking and pausing one-time batches."""

    def __init__(
        self,
        *,
        queue: JobQueue,
        ledger: LedgerStore,
        pricing: Pricing,
        estimator: TokenEstimator,
        budget: BudgetConfig,
        db: Database,
        clock: Clock,
        alerts: AlertSink | None = None,
    ) -> None:
        self._queue = queue
        self._ledger = ledger
        self._pricing = pricing
        self._estimator = estimator
        self._budget = budget
        self._db = db
        self._clock = clock
        self._alerts = alerts
        self._lock = threading.Lock()  # after_call runs on worker threads

    @property
    def limit_usd(self) -> float:
        """Largest estimate a single batch may have (``budget.one_time_usd``)."""
        return self._budget.one_time_usd

    # -------------------------------------------------------------- estimating

    def item_from_messages(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        completion_tokens: int,
        image_sizes: Sequence[tuple[int, int]] = (),
        cache_hit_ratio: float = 0.0,
    ) -> BatchItem:
        """A :class:`BatchItem` whose prompt size comes from the token estimator."""
        text_messages: list[ChatMessage] = [
            {"role": message["role"], "content": _text_of(message)} for message in messages
        ]
        prompt = self._estimator.estimate_messages(text_messages)
        return BatchItem(model, prompt, completion_tokens, tuple(image_sizes), cache_hit_ratio)

    def estimate_item(self, item: BatchItem) -> tuple[float, int]:
        """``(USD, image tokens)`` of one planned call, at peak prices."""
        image_tokens = sum(self._estimator.estimate_image(w, h) for w, h in item.image_sizes)
        usd = self._pricing.estimate(
            item.model,
            prompt_tokens=item.prompt_tokens + image_tokens,
            completion_tokens=item.completion_tokens,
            cache_hit_ratio=item.cache_hit_ratio,
        )
        return usd, image_tokens

    def estimate(self, items: Sequence[BatchItem]) -> BatchEstimate:
        """Price a list of planned calls."""
        per_item: list[float] = []
        image_tokens = 0
        for item in items:
            usd, images = self.estimate_item(item)
            per_item.append(usd)
            image_tokens += images
        return BatchEstimate(
            items=len(items),
            prompt_tokens=sum(item.prompt_tokens for item in items),
            completion_tokens=sum(item.completion_tokens for item in items),
            image_tokens=image_tokens,
            per_item_usd=tuple(per_item),
        )

    def split(
        self, items: Sequence[BatchItem], *, limit_usd: float | None = None
    ) -> list[list[BatchItem]]:
        """Greedy chunks of ``items`` that each estimate to at most the limit."""
        limit = self.limit_usd if limit_usd is None else limit_usd
        chunks: list[list[BatchItem]] = []
        current: list[BatchItem] = []
        running = 0.0
        for item in items:
            usd, _ = self.estimate_item(item)
            if usd > limit:
                raise BatchTooLargeError("single item", usd, limit)
            if current and running + usd > limit:
                chunks.append(current)
                current, running = [], 0.0
            current.append(item)
            running += usd
        if current:
            chunks.append(current)
        return chunks

    # ----------------------------------------------------------------- queueing

    def enqueue(
        self,
        batch_id: str,
        job_type: str,
        payloads: Sequence[object],
        estimates_usd: Sequence[float],
        *,
        priority: int = 100,
        max_attempts: int = 3,
        offpeak_only: bool = True,
        deadline: datetime | None = None,
    ) -> list[str]:
        """Queue the jobs of a batch, waiting for ``twin jobs approve``."""
        if len(payloads) != len(estimates_usd):
            raise ValueError("every payload needs an estimate")
        if not payloads:
            raise ValueError("a batch needs at least one job")
        total = sum(estimates_usd)
        if total > self.limit_usd:
            raise BatchTooLargeError(batch_id, total, self.limit_usd)
        ids = [
            self._queue.enqueue(
                job_type,
                payload,
                priority=priority,
                max_attempts=max_attempts,
                offpeak_only=offpeak_only,
                deadline=deadline,
                batch_id=batch_id,
                estimated_cost_usd=estimate,
                requires_approval=True,
            )
            for payload, estimate in zip(payloads, estimates_usd, strict=True)
        ]
        self._save_state(batch_id, {"estimated_usd": total, "approved_at": None, "paused_at": None})
        log.info("batch_queued", batch_id=batch_id, jobs=len(ids), estimated_usd=round(total, 4))
        return ids

    # ------------------------------------------------------------------- state

    def _state(self, batch_id: str) -> dict[str, object]:
        with self._db.session() as session:
            raw = get_setting(session, STATE_PREFIX + batch_id, None)
        return dict(raw) if isinstance(raw, dict) else {}

    def _save_state(self, batch_id: str, changes: dict[str, object]) -> dict[str, object]:
        state = {**self._state(batch_id), **changes}
        with self._db.transaction(bump_state=False) as session:
            put_setting(
                session, STATE_PREFIX + batch_id, state, clock=self._clock, record_history=False
            )
        return state

    def status(self, batch_id: str) -> BatchStatus:
        state = self._state(batch_id)
        cap = state.get("cap_usd")
        estimated = state.get("estimated_usd")
        return BatchStatus(
            batch_id=batch_id,
            estimated_usd=float(estimated) if isinstance(estimated, int | float) else 0.0,
            spent_usd=self._ledger.batch_spent_usd(batch_id),
            cap_usd=float(cap) if isinstance(cap, int | float) else None,
            approved_at=str(state["approved_at"]) if state.get("approved_at") else None,
            paused_at=str(state["paused_at"]) if state.get("paused_at") else None,
        )

    # ---------------------------------------------------------------- approving

    def approve(self, batch_id: str) -> BatchApproval:
        """Approve the waiting jobs of a batch (``twin jobs approve``).

        The batch as a whole - what it already spent plus the estimate of what is left - must
        stay within ``budget.one_time_usd``.  Records the approval time and amount and sets the
        cap at :data:`OVERRUN_FACTOR` times that projected total.  Approving a paused batch
        resumes it.
        """
        spent = self._ledger.batch_spent_usd(batch_id)
        try:
            approval = self._queue.approve_batch(batch_id, max_usd=max(0.0, self.limit_usd - spent))
        except BatchTooLargeError as exc:
            raise BatchTooLargeError(batch_id, spent + exc.estimate, self.limit_usd) from exc
        projected = spent + approval.total_usd
        self._save_state(
            batch_id,
            {
                "estimated_usd": projected,
                "approved_at": self._clock.now_utc().isoformat(),
                "approved_usd": approval.total_usd,
                "cap_usd": projected * OVERRUN_FACTOR,
                "paused_at": None,
            },
        )
        log.info(
            "batch_approved", batch_id=batch_id, jobs=approval.job_count, usd=approval.total_usd
        )
        return approval

    # -------------------------------------------------------------- enforcement

    def before_call(self, batch_id: str) -> None:
        """Refuse calls of a paused batch (raises :class:`BatchPausedError`)."""
        if self._state(batch_id).get("paused_at"):
            raise BatchPausedError(batch_id)

    def after_call(self, batch_id: str) -> None:
        """Pause the batch if its actual spend passed the cap."""
        self.check(batch_id)

    def check(self, batch_id: str) -> BatchStatus:
        """Compare actual spend with the cap and pause the batch when it is exceeded."""
        with self._lock:
            return self._check(batch_id)

    def _check(self, batch_id: str) -> BatchStatus:
        status = self.status(batch_id)
        if status.cap_usd is None or status.paused or status.spent_usd <= status.cap_usd:
            return status
        withdrawn = self._queue.revoke_approval(batch_id)
        paused_at = self._clock.now_utc().isoformat()
        self._save_state(batch_id, {"paused_at": paused_at})
        log.warning(
            "batch_paused",
            batch_id=batch_id,
            spent_usd=round(status.spent_usd, 4),
            cap_usd=round(status.cap_usd, 4),
            jobs=withdrawn,
        )
        if self._alerts is not None:
            self._alerts.raise_alert(
                "batch_overrun",
                f"one-time batch {batch_id} paused: ${status.spent_usd:.2f} spent, "
                f"estimate ${status.estimated_usd:.2f}",
                severity="warning",
                detail={
                    "batch_id": batch_id,
                    "spent_usd": round(status.spent_usd, 4),
                    "estimated_usd": round(status.estimated_usd, 4),
                    "cap_usd": round(status.cap_usd, 4),
                    "jobs_paused": withdrawn,
                },
                dedup_key=f"batch_overrun:{batch_id}:{paused_at}",
            )
        return self.status(batch_id)


def _text_of(message: ChatMessage) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if part["type"] == "text")
