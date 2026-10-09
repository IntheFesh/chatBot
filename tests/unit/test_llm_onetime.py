"""One-time batch budgets: estimate, approve, overrun pause (R-LLM-014)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from tests.support.policies import AlwaysOffPeak
from twin.config.loader import load_settings
from twin.config.settings import BudgetConfig
from twin.llm.budget import BudgetManager
from twin.llm.ledger import LedgerRecord, LedgerStore
from twin.llm.onetime import (
    OVERRUN_FACTOR,
    BatchItem,
    BatchPausedError,
    OneTimeBatches,
)
from twin.llm.pricing import Pricing
from twin.llm.tokens import ImageTokenTable, TokenEstimator
from twin.llm.types import ChatMessage, CostBreakdown, LedgerTag, Usage
from twin.ops.jobs import (
    BatchTooLargeError,
    HandlerRegistry,
    JobContext,
    JobQueue,
    Worker,
)
from twin.schedule.time_service import ConfiguredTimeService
from twin.storage.db import Database

NOW = datetime(2026, 10, 9, 17, 0, tzinfo=UTC)


class Rig:
    def __init__(self, db: Database, clock: ManualClock, one_time_usd: float = 30.0) -> None:
        clock.set_time(NOW)
        self.db = db
        self.clock = clock
        self.time = ConfiguredTimeService(clock, lambda: "America/Chicago")
        self.ledger = LedgerStore(db, clock, self.time)
        self.queue = JobQueue(db, clock)
        self.alerts = RecordingAlerts()
        settings = load_settings(None, {"budget": {"one_time_usd": one_time_usd}})
        self.estimator = TokenEstimator()
        self.batches = OneTimeBatches(
            queue=self.queue,
            ledger=self.ledger,
            pricing=Pricing.from_settings(settings),
            estimator=self.estimator,
            budget=settings.budget,
            db=db,
            clock=clock,
            alerts=self.alerts,
        )

    def spend(self, batch_id: str, usd: float) -> None:
        self.ledger.record(
            LedgerRecord(
                provider="deepseek",
                model="deepseek-flash",
                purpose="persona",
                usage=Usage(prompt_tokens=1, completion_tokens=1, cache_miss_tokens=1),
                cost=CostBreakdown(usd, 0.0, 0.0, True, 1.0),
                thinking=False,
                latency_ms=1,
                at=self.clock.now_utc(),
                tag=LedgerTag("one_time", batch_id),
            )
        )

    def queue_batch(self, batch_id: str, estimates: list[float]) -> list[str]:
        return self.batches.enqueue(
            batch_id, "replay", [{"n": i} for i in range(len(estimates))], estimates
        )

    def claimable(self) -> int:
        count = 0
        while self.queue.claim_next({"replay"}, offpeak_allowed=True, worker_id="t") is not None:
            count += 1
        return count


@pytest.fixture
def rig(db: Database, clock: ManualClock) -> Rig:
    return Rig(db, clock)


# -------------------------------------------------------------------- estimating


def test_estimates_use_peak_prices_tokens_and_image_costs(rig: Rig) -> None:
    item = BatchItem("deepseek-flash", prompt_tokens=1_000_000, completion_tokens=500_000)
    usd, images = rig.batches.estimate_item(item)
    assert images == 0 and usd == pytest.approx(0.30 + 0.60)
    with_images = BatchItem(
        "deepseek-flash", 0, 0, image_sizes=((1000, 1000), (50, 50)), cache_hit_ratio=0.0
    )
    usd, images = rig.batches.estimate_item(with_images)
    assert images == 2048  # the documented cap per image until the probe measured something
    assert usd == pytest.approx(2048 * 0.30 / 1_000_000)
    rig.estimator.set_image_table(ImageTokenTable([(2500, 40), (1_000_000, 700)]))
    _, images = rig.batches.estimate_item(with_images)
    assert images == 700 + 40


def test_estimate_totals_and_cache_hits(rig: Rig) -> None:
    items = [
        BatchItem("deepseek-flash", 1_000_000, 0),
        BatchItem("deepseek-flash", 1_000_000, 0, cache_hit_ratio=1.0),
        BatchItem("deepseek-v4-pro", 0, 1_000_000),
    ]
    estimate = rig.batches.estimate(items)
    assert estimate.items == 3 and estimate.prompt_tokens == 2_000_000
    assert estimate.completion_tokens == 1_000_000
    assert estimate.per_item_usd == pytest.approx((0.30, 0.006, 3.96))
    assert estimate.total_usd == pytest.approx(4.266)


def test_items_can_be_built_from_messages(rig: Rig) -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "你是助手" * 50},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        },
    ]
    item = rig.batches.item_from_messages(
        "deepseek-flash", messages, completion_tokens=80, image_sizes=[(640, 480)]
    )
    assert item.completion_tokens == 80 and item.image_sizes == ((640, 480),)
    assert item.prompt_tokens < 1024  # the image is priced separately, not counted as text
    assert item.prompt_tokens > 100


def test_split_keeps_every_chunk_within_the_limit(rig: Rig) -> None:
    item = BatchItem("deepseek-flash", 0, 1_000_000)  # $1.20 each
    chunks = rig.batches.split([item] * 10, limit_usd=5.0)
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert rig.batches.split([item] * 3) == [[item] * 3]
    assert rig.batches.split([]) == []
    with pytest.raises(BatchTooLargeError):
        rig.batches.split([item], limit_usd=1.0)


# ------------------------------------------------------------------- enqueueing


def test_enqueued_jobs_wait_for_approval(rig: Rig) -> None:
    ids = rig.queue_batch("replay-1", [1.0, 2.0, 3.0])
    assert len(ids) == 3
    jobs = [rig.queue.get(i) for i in ids]
    assert all(j is not None and j.requires_approval and j.approved_at is None for j in jobs)
    assert [j.estimated_cost_usd for j in jobs if j] == [1.0, 2.0, 3.0]
    assert all(j.offpeak_only and j.batch_id == "replay-1" for j in jobs if j)
    assert rig.claimable() == 0
    status = rig.batches.status("replay-1")
    assert (status.estimated_usd, status.approved, status.paused) == (6.0, False, False)


def test_a_batch_above_the_one_time_limit_is_refused(db: Database, clock: ManualClock) -> None:
    small = Rig(db, clock, one_time_usd=5.0)
    with pytest.raises(BatchTooLargeError, match="split"):
        small.queue_batch("big", [3.0, 3.0])
    with pytest.raises(ValueError, match="estimate"):
        small.batches.enqueue("x", "replay", [{}], [])
    with pytest.raises(ValueError, match="at least one"):
        small.batches.enqueue("x", "replay", [], [])


# ---------------------------------------------------------------------- approval


def test_approval_records_time_and_amount_and_sets_the_cap(rig: Rig) -> None:
    rig.queue_batch("replay-1", [2.0, 3.0])
    approval = rig.batches.approve("replay-1")
    assert (approval.job_count, approval.total_usd) == (2, 5.0)
    status = rig.batches.status("replay-1")
    assert status.approved and status.approved_at == NOW.isoformat()
    assert status.cap_usd == pytest.approx(5.0 * OVERRUN_FACTOR)
    jobs = rig.queue.list_jobs(batch_id="replay-1")
    assert all(j.approved_at == NOW and j.approved_usd in (2.0, 3.0) for j in jobs)
    assert rig.claimable() == 2


def test_approval_is_refused_when_spent_plus_remaining_exceeds_the_limit(
    db: Database, clock: ManualClock
) -> None:
    rig = Rig(db, clock, one_time_usd=10.0)
    rig.queue_batch("b", [4.0, 4.0])
    rig.spend("b", 3.0)  # something already ran earlier
    with pytest.raises(BatchTooLargeError) as info:
        rig.batches.approve("b")
    assert info.value.estimate == pytest.approx(11.0) and info.value.limit == 10.0
    assert rig.claimable() == 0


# ----------------------------------------------------------------- overrun pause


def test_spending_up_to_twenty_percent_above_the_estimate_is_allowed(rig: Rig) -> None:
    rig.queue_batch("b", [5.0, 5.0])
    rig.batches.approve("b")
    rig.spend("b", 11.9)
    rig.batches.after_call("b")
    rig.batches.before_call("b")  # does not raise
    status = rig.batches.status("b")
    assert not status.paused and status.spent_usd == pytest.approx(11.9)
    assert status.overrun_ratio == pytest.approx(1.19)
    assert rig.alerts.alerts == []


def test_overspending_pauses_the_batch_and_alerts_once(rig: Rig) -> None:
    rig.queue_batch("b", [5.0, 5.0])
    rig.batches.approve("b")
    rig.spend("b", 12.5)
    rig.batches.after_call("b")
    status = rig.batches.status("b")
    assert status.paused and status.paused_at == NOW.isoformat()
    assert rig.claimable() == 0  # approvals are withdrawn
    assert [a.category for a in rig.alerts.alerts] == ["batch_overrun"]
    assert rig.alerts.alerts[0].detail is not None
    assert rig.alerts.alerts[0].detail["batch_id"] == "b"
    with pytest.raises(BatchPausedError, match="paused"):
        rig.batches.before_call("b")
    rig.batches.after_call("b")  # already paused: no second alert
    assert len(rig.alerts.alerts) == 1


def test_a_paused_batch_continues_after_approval_with_a_new_cap(rig: Rig) -> None:
    rig.queue_batch("b", [5.0, 5.0])
    rig.batches.approve("b")
    rig.spend("b", 12.5)
    rig.batches.after_call("b")
    rig.clock.tick(60)
    approval = rig.batches.approve("b")
    assert approval.job_count == 2 and approval.total_usd == 10.0
    status = rig.batches.status("b")
    assert not status.paused
    assert status.estimated_usd == pytest.approx(22.5)  # spent so far plus what is left
    assert status.cap_usd == pytest.approx(22.5 * OVERRUN_FACTOR)
    rig.batches.before_call("b")
    assert rig.claimable() == 2


def test_batches_without_an_approval_are_not_enforced(rig: Rig) -> None:
    rig.spend("probe-1", 99.0)  # e.g. the M0 probe: no estimate, no approval
    rig.batches.after_call("probe-1")
    rig.batches.before_call("probe-1")
    status = rig.batches.status("probe-1")
    assert status.cap_usd is None and not status.paused and status.overrun_ratio == 0.0


def test_one_time_spending_does_not_move_the_daily_budget(rig: Rig) -> None:
    manager = BudgetManager(
        BudgetConfig(),
        examples_k=8,
        ledger=rig.ledger,
        time_service=rig.time,
        clock=rig.clock,
        db=rig.db,
        alerts=rig.alerts,
        ttl_s=0.0,
    )
    rig.queue_batch("b", [5.0])
    rig.batches.approve("b")
    rig.spend("b", 5.5)
    manager.note_spend()
    assert manager.current_level() == 0
    assert manager.status().daily_spent == 0.0
    assert all(a.category != "budget" for a in rig.alerts.alerts)


# ------------------------------------------------------------ worker integration


async def test_the_batch_stops_after_the_job_that_crossed_the_cap(
    db: Database, clock: ManualClock
) -> None:
    rig = Rig(db, clock)
    rig.queue_batch("b", [5.0, 5.0, 5.0])
    rig.batches.approve("b")  # estimate 15, cap 18
    registry = HandlerRegistry()
    ran: list[int] = []

    async def handler(ctx: JobContext) -> None:
        rig.batches.before_call("b")
        ran.append(ctx.job.payload["n"])
        rig.spend("b", 18.5)  # one job costs more than the whole cap
        rig.batches.after_call("b")

    registry.register("replay", handler)
    worker = Worker(rig.queue, registry, clock, offpeak=AlwaysOffPeak(), concurrency=1)
    summary = await worker.run_until_idle()
    assert (summary.done, summary.failed, summary.retried) == (1, 0, 0)
    assert len(ran) == 1 and rig.batches.status("b").paused
    pending = [j for j in rig.queue.list_jobs(batch_id="b") if j.status == "pending"]
    assert len(pending) == 2 and all(j.attempts == 0 and j.approved_at is None for j in pending)
    assert rig.claimable() == 0

    rig.batches.approve("b")  # the user looks at the numbers and continues
    assert rig.claimable() == 2


async def test_a_job_that_meets_the_paused_batch_is_handed_back_not_failed(
    db: Database, clock: ManualClock
) -> None:
    rig = Rig(db, clock)
    ids = rig.queue_batch("b", [5.0])
    rig.batches.approve("b")
    registry = HandlerRegistry()

    async def handler(ctx: JobContext) -> None:
        rig.spend("b", 50.0)  # another handler of the same batch overspent meanwhile
        rig.batches.check("b")
        rig.batches.before_call("b")

    registry.register("replay", handler)
    worker = Worker(rig.queue, registry, clock, offpeak=AlwaysOffPeak(), concurrency=1)
    summary = await worker.run_until_idle()
    assert (summary.deferred, summary.failed, summary.retried, summary.done) == (1, 0, 0, 0)
    job = rig.queue.get(ids[0])
    assert job is not None and job.status == "pending" and job.attempts == 0
    assert job.approved_at is None
    assert job.run_after > clock.now_utc()  # it also waits before being claimed again
