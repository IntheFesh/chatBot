"""Persistent job queue and worker (R-ARCH-003)."""

from __future__ import annotations

import asyncio
import random
from datetime import timedelta
from typing import Any

import pytest

from tests.support.clock import ManualClock
from tests.support.synthetic import mobile
from tests.support.waiting import wait_until
from twin.ops.alerts import DbAlertSink
from twin.ops.jobs import (
    BatchNotFoundError,
    BatchTooLargeError,
    DeferredOffPeakPolicy,
    HandlerRegistry,
    JobContext,
    JobDeferred,
    JobQueue,
    Worker,
    backoff_delay,
    default_registry,
    describe_failure,
    get_offpeak_policy,
    job_handler,
    load_handlers,
    set_offpeak_policy,
)
from twin.storage.db import Database
from twin.storage.models import Alert, Job

PHONE = mobile()


class AlwaysOffPeak:
    """Test-only policy: off-peak jobs may always run (production policy: round 01)."""

    def allows(self, now: object) -> bool:
        return True


class NeverOffPeak:
    def allows(self, now: object) -> bool:
        return False


@pytest.fixture
def queue(db: Database, clock: ManualClock) -> JobQueue:
    return JobQueue(db, clock)


@pytest.fixture
def registry() -> HandlerRegistry:
    return HandlerRegistry()


def make_worker(
    queue: JobQueue,
    registry: HandlerRegistry,
    db: Database,
    clock: ManualClock,
    **options: Any,
) -> Worker:
    options.setdefault("offpeak", AlwaysOffPeak())
    options.setdefault("alerts", DbAlertSink(db, clock))
    return Worker(queue, registry, clock, **options)


def record_handler(registry: HandlerRegistry, job_type: str, seen: list[str]) -> None:
    async def handler(ctx: JobContext) -> None:
        seen.append(f"{ctx.job.type}:{ctx.job.payload['n']}")

    registry.register(job_type, handler)


# ----------------------------------------------------------------- queueing


def test_enqueue_stores_encrypted_payload_and_defaults(queue: JobQueue, clock: ManualClock) -> None:
    job_id = queue.enqueue("demo", {"n": 1, "text": "正文"})
    job = queue.get(job_id)
    assert job is not None
    assert (job.type, job.status, job.attempts, job.max_attempts, job.priority) == (
        "demo",
        "pending",
        0,
        3,
        100,
    )
    assert job.payload == {"n": 1, "text": "正文"}
    assert job.run_after == clock.now_utc() == job.created_at
    assert queue.get("nope") is None


def test_enqueue_validation(queue: JobQueue) -> None:
    with pytest.raises(ValueError, match="empty"):
        queue.enqueue("", {})
    with pytest.raises(ValueError, match="max_attempts"):
        queue.enqueue("x", {}, max_attempts=0)
    with pytest.raises(ValueError, match="batch"):
        queue.enqueue("x", {}, requires_approval=True)


def test_listing_filters_and_counts(queue: JobQueue, clock: ManualClock) -> None:
    for index in range(3):
        clock.tick(1)
        queue.enqueue("a", {"n": index})
    queue.enqueue("b", {"n": 9}, batch_id="batch-1")
    assert [j.payload["n"] for j in queue.list_jobs(job_type="a")] == [2, 1, 0]  # newest first
    assert [j.type for j in queue.list_jobs(batch_id="batch-1")] == ["b"]
    assert len(queue.list_jobs(limit=2)) == 2
    assert queue.counts() == {"pending": 4, "running": 0, "done": 0, "failed": 0, "cancelled": 0}
    with pytest.raises(ValueError, match="unknown job status"):
        queue.list_jobs(status="sleeping")
    assert queue.list_jobs(status="done") == []


async def test_priority_then_age_decides_the_order(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    seen: list[str] = []
    record_handler(registry, "t", seen)
    queue.enqueue("t", {"n": 1}, priority=50)
    clock.tick(1)
    queue.enqueue("t", {"n": 2}, priority=10)
    clock.tick(1)
    queue.enqueue("t", {"n": 3}, priority=10)
    queue.enqueue("t", {"n": 4}, priority=200)
    worker = make_worker(queue, registry, db, clock, concurrency=1)
    summary = await worker.run_until_idle()
    assert seen == ["t:2", "t:3", "t:1", "t:4"]
    assert summary.done == 4 and summary.failed == 0


async def test_jobs_without_a_handler_stay_pending_and_are_not_touched(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    orphan = queue.enqueue("later_round_type", {"n": 1})
    seen: list[str] = []
    record_handler(registry, "known", seen)
    queue.enqueue("known", {"n": 2})
    summary = await make_worker(queue, registry, db, clock).run_until_idle()
    assert summary.done == 1 and seen == ["known:2"]
    job = queue.get(orphan)
    assert job is not None and job.status == "pending" and job.attempts == 0
    assert not default_registry.has("later_round_type")  # what `twin jobs list` reports as 无处理器


# --------------------------------------------------------- retry and failure


def test_backoff_delay_is_exponential_and_capped() -> None:
    assert [backoff_delay(n, 30, 3600) for n in (1, 2, 3, 4, 5, 6, 7, 8, 9)] == [
        30,
        60,
        120,
        240,
        480,
        960,
        1920,
        3600,
        3600,
    ]
    assert backoff_delay(0, 30, 3600) == 30


async def test_failed_jobs_retry_with_exponential_backoff_then_fail_with_an_alert(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    calls: list[int] = []

    async def flaky(ctx: JobContext) -> None:
        calls.append(ctx.job.attempts)
        raise RuntimeError(f"upstream refused {PHONE}")

    registry.register("flaky", flaky)
    job_id = queue.enqueue("flaky", {"n": 1}, max_attempts=3)
    worker = make_worker(queue, registry, db, clock, backoff_base_s=30, backoff_cap_s=3600)

    first = await worker.run_until_idle()
    job = queue.get(job_id)
    assert job is not None
    assert (first.retried, first.failed) == (1, 0)
    assert job.status == "pending" and job.attempts == 1
    assert job.run_after == clock.now_utc() + timedelta(seconds=30)
    assert (
        job.last_error is not None and PHONE not in job.last_error and "[手机号]" in job.last_error
    )

    assert (await worker.run_until_idle()).retried == 0  # not due yet
    clock.tick(30)
    await worker.run_until_idle()
    job = queue.get(job_id)
    assert job is not None and job.attempts == 2
    assert job.run_after == clock.now_utc() + timedelta(seconds=60)  # doubled

    clock.tick(60)
    last = await worker.run_until_idle()
    job = queue.get(job_id)
    assert job is not None
    assert (last.failed, job.status, job.attempts) == (1, "failed", 3)
    assert job.finished_at == clock.now_utc()
    assert calls == [1, 2, 3]
    with db.session() as session:
        alerts = session.query(Alert).all()
        assert len(alerts) == 1
        assert alerts[0].category == "job_failed" and alerts[0].severity == "warning"
        assert alerts[0].detail["job_id"] == job_id  # type: ignore[index]
        assert PHONE not in str(alerts[0].detail)
    clock.tick(10_000)
    assert (await worker.run_until_idle()).done == 0  # failed jobs are never picked up again


async def test_retry_delay_jitter_is_bounded_and_seedable(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    worker = make_worker(
        queue, registry, db, clock, backoff_base_s=100, jitter_ratio=0.25, rng=random.Random(1)
    )
    delays = {round(worker.retry_delay(1), 3) for _ in range(20)}
    assert len(delays) > 1
    assert all(75 <= d <= 125 for d in delays)


async def test_handler_exception_does_not_stop_other_jobs(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    seen: list[str] = []
    record_handler(registry, "ok", seen)

    async def boom(ctx: JobContext) -> None:
        raise ValueError("nope")

    registry.register("boom", boom)
    queue.enqueue("boom", {"n": 0}, max_attempts=1, priority=1)
    queue.enqueue("ok", {"n": 1})
    queue.enqueue("ok", {"n": 2})
    summary = await make_worker(queue, registry, db, clock, concurrency=1).run_until_idle()
    assert (summary.done, summary.failed) == (2, 1)
    assert seen == ["ok:1", "ok:2"]


def test_describe_failure_redacts_and_truncates() -> None:
    text = describe_failure(RuntimeError(f"{PHONE} " + "x" * 2000))
    assert text.startswith("RuntimeError: [手机号]") and len(text) <= 500


# ------------------------------------------------------ restart and recovery


async def test_running_jobs_return_to_pending_after_a_restart(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    job_id = queue.enqueue("t", {"n": 1}, max_attempts=3)
    claimed = queue.claim_next({"t"}, offpeak_allowed=True, worker_id="dead-process")
    assert claimed is not None and claimed.status == "running" and claimed.attempts == 1
    seen: list[str] = []
    record_handler(registry, "t", seen)
    worker = make_worker(queue, registry, db, clock)
    assert await worker.recover() == 1
    job = queue.get(job_id)
    assert job is not None and job.status == "pending" and job.attempts == 1
    await worker.run_until_idle()
    assert seen == ["t:1"]


async def test_recovery_fails_jobs_that_already_used_all_attempts(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    """A job that keeps killing the process must not be retried forever."""
    job_id = queue.enqueue("poison", {"n": 1}, max_attempts=1)
    queue.claim_next({"poison"}, offpeak_allowed=True, worker_id="dead")
    worker = make_worker(queue, registry, db, clock)
    assert await worker.recover() == 0
    job = queue.get(job_id)
    assert job is not None and job.status == "failed"
    assert job.last_error is not None and "attempts exhausted" in job.last_error
    with db.session() as session:
        assert session.query(Alert).count() == 1


async def test_cancelled_while_running_discards_the_result(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(ctx: JobContext) -> None:
        started.set()
        await release.wait()

    registry.register("slow", slow)
    job_id = queue.enqueue("slow", {})
    worker = make_worker(queue, registry, db, clock)
    runner = asyncio.create_task(worker.run_until_idle())
    await started.wait()
    assert queue.cancel(job_id) is True
    release.set()
    summary = await runner
    assert summary.discarded == 1 and summary.done == 0
    job = queue.get(job_id)
    assert job is not None and job.status == "cancelled"


async def test_graceful_stop_releases_unfinished_jobs_without_costing_an_attempt(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    started = asyncio.Event()

    async def forever(ctx: JobContext) -> None:
        started.set()
        await asyncio.sleep(3600)

    registry.register("forever", forever)
    job_id = queue.enqueue("forever", {}, max_attempts=2)
    worker = make_worker(queue, registry, db, clock)
    stop = asyncio.Event()
    runner = asyncio.create_task(worker.run_forever(stop))
    await started.wait()
    assert worker.running_count == 1
    stop.set()
    await asyncio.wait_for(worker.drain(timeout_s=0.05), timeout=5)
    await asyncio.wait_for(runner, timeout=5)
    job = queue.get(job_id)
    assert job is not None and job.status == "pending" and job.attempts == 0


# ----------------------------------------------------------------- off-peak


def test_default_policy_defers_offpeak_jobs_until_round_01_registers_one() -> None:
    assert isinstance(get_offpeak_policy(), DeferredOffPeakPolicy)
    from datetime import UTC, datetime

    assert not get_offpeak_policy().allows(datetime(2026, 1, 1, tzinfo=UTC))


def test_set_offpeak_policy_swaps_and_returns_the_previous_one() -> None:
    previous = set_offpeak_policy(AlwaysOffPeak())
    try:
        assert isinstance(get_offpeak_policy(), AlwaysOffPeak)
    finally:
        restored = set_offpeak_policy(previous)
    assert isinstance(restored, AlwaysOffPeak)
    assert get_offpeak_policy() is previous


async def test_offpeak_only_jobs_follow_the_policy_and_the_deadline(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    seen: list[str] = []
    record_handler(registry, "t", seen)
    queue.enqueue("t", {"n": 1}, offpeak_only=True)
    queue.enqueue("t", {"n": 2}, offpeak_only=True, deadline=clock.now_utc() + timedelta(hours=2))
    queue.enqueue("t", {"n": 3})

    blocked = make_worker(queue, registry, db, clock, offpeak=NeverOffPeak())
    await blocked.run_until_idle()
    assert seen == ["t:3"]  # peak hours: only the normal job ran

    clock.tick(3 * 3600)  # job 2's deadline has passed: it is forced to run
    await blocked.run_until_idle()
    assert seen == ["t:3", "t:2"]

    allowed = make_worker(queue, registry, db, clock, offpeak=AlwaysOffPeak())
    await allowed.run_until_idle()
    assert seen == ["t:3", "t:2", "t:1"]  # off-peak window opened


async def test_worker_uses_the_registered_policy_when_none_is_injected(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    seen: list[str] = []
    record_handler(registry, "t", seen)
    queue.enqueue("t", {"n": 1}, offpeak_only=True)
    worker = Worker(queue, registry, clock)  # no policy injected
    await worker.run_until_idle()
    assert seen == []  # DeferredOffPeakPolicy
    previous = set_offpeak_policy(AlwaysOffPeak())
    try:
        await worker.run_until_idle()
    finally:
        set_offpeak_policy(previous)
    assert seen == ["t:1"]


# ------------------------------------------------------ concurrency, loops


async def test_concurrency_is_bounded(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    active = 0
    peak = 0

    async def tracked(ctx: JobContext) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.15)  # long enough that three claims overlap even on a loaded machine
        active -= 1

    registry.register("t", tracked)
    for index in range(8):
        queue.enqueue("t", {"n": index})
    summary = await make_worker(queue, registry, db, clock, concurrency=3).run_until_idle()
    assert summary.done == 8 and peak == 3
    with pytest.raises(ValueError, match="concurrency"):
        make_worker(queue, registry, db, clock, concurrency=0)


async def test_run_forever_picks_up_new_jobs_and_stops_cleanly(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    seen: list[str] = []
    record_handler(registry, "t", seen)
    worker = make_worker(queue, registry, db, clock, poll_interval_s=5)
    stop = asyncio.Event()
    runner = asyncio.create_task(worker.run_forever(stop))
    await clock.settle()
    queue.enqueue("t", {"n": 1})
    await clock.advance(5)  # the poll timer fires
    await wait_until(lambda: seen == ["t:1"])
    queue.enqueue("t", {"n": 2})
    await clock.advance(5)
    await wait_until(lambda: len(seen) == 2)
    stop.set()
    await asyncio.wait_for(runner, timeout=5)
    assert 5 in clock.sleeps


async def test_handlers_receive_services_clock_and_a_job_snapshot(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    captured: list[JobContext] = []

    async def handler(ctx: JobContext) -> None:
        captured.append(ctx)

    registry.register("t", handler)
    queue.enqueue("t", {"n": 1})
    await make_worker(queue, registry, db, clock).run_until_idle()
    assert captured[0].clock is clock and captured[0].services is None
    assert captured[0].job.status == "running" and captured[0].job.attempts == 1


# ------------------------------------------------------------- user actions


def test_retry_and_cancel(queue: JobQueue) -> None:
    job_id = queue.enqueue("t", {}, max_attempts=1)
    assert queue.cancel(job_id) is True
    assert queue.cancel(job_id) is False  # already finished
    assert queue.retry(job_id) is True
    job = queue.get(job_id)
    assert (
        job is not None
        and job.status == "pending"
        and job.attempts == 0
        and job.finished_at is None
    )
    assert queue.retry(job_id) is False  # pending jobs are not retried
    assert queue.retry("missing") is False and queue.cancel("missing") is False


async def test_registry_rules(registry: HandlerRegistry) -> None:
    async def one(ctx: JobContext) -> None: ...

    async def two(ctx: JobContext) -> None: ...

    registry.register("t", one)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("t", two)
    registry.register("t", two, replace=True)
    assert registry.get("t") is two and registry.types() == frozenset({"t"})
    assert registry.get("zzz") is None


def test_decorator_registers_into_the_chosen_registry(registry: HandlerRegistry) -> None:
    @job_handler("decorated", registry=registry)
    async def handler(ctx: JobContext) -> None: ...

    assert registry.get("decorated") is handler
    assert not default_registry.has("decorated")


def test_load_handlers_imports_the_listed_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_handlers([]) is default_registry
    assert load_handlers(["twin.ops.alerts"]) is default_registry
    with pytest.raises(ModuleNotFoundError):
        load_handlers(["twin.no_such_module"])


# --------------------------------------------------- one-time batches (R-LLM-014)


async def test_unapproved_batch_jobs_are_never_claimed(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    seen: list[str] = []
    record_handler(registry, "t", seen)
    for index in range(3):
        queue.enqueue(
            "t", {"n": index}, batch_id="replay-1", estimated_cost_usd=2.5, requires_approval=True
        )
    worker = make_worker(queue, registry, db, clock)
    await worker.run_until_idle()
    assert seen == []
    summary = queue.batch_summary("replay-1")
    assert (summary.jobs, summary.awaiting_approval, summary.estimated_usd) == (3, 3, 7.5)

    approval = queue.approve_batch("replay-1", max_usd=30.0)
    assert (approval.job_count, approval.total_usd) == (3, 7.5)
    await worker.run_until_idle()
    assert sorted(seen) == ["t:0", "t:1", "t:2"]
    done = queue.batch_summary("replay-1")
    assert done.awaiting_approval == 0 and done.approved_usd == 7.5
    first = queue.list_jobs(batch_id="replay-1")[0]
    assert first.approved_at == clock.now_utc() and first.approved_usd == 2.5


def test_batch_approval_enforces_the_one_time_limit_and_existence(queue: JobQueue) -> None:
    queue.enqueue("t", {}, batch_id="big", estimated_cost_usd=40.0, requires_approval=True)
    with pytest.raises(BatchTooLargeError, match="split"):
        queue.approve_batch("big", max_usd=30.0)
    assert queue.batch_summary("big").awaiting_approval == 1  # nothing was approved
    with pytest.raises(BatchNotFoundError):
        queue.approve_batch("unknown", max_usd=30.0)
    with pytest.raises(BatchNotFoundError):
        queue.batch_summary("unknown")
    queue.approve_batch("big", max_usd=100.0)
    with pytest.raises(BatchNotFoundError, match="waiting"):
        queue.approve_batch("big", max_usd=100.0)  # already approved


def test_job_view_reflects_the_table(queue: JobQueue, db: Database) -> None:
    job_id = queue.enqueue("t", {"n": 1}, batch_id="b", estimated_cost_usd=1.5)
    with db.session() as session:
        row = session.get(Job, job_id)
        assert row is not None and row.estimated_cost_usd == 1.5 and row.batch_id == "b"


# ------------------------------------------------------------------ deferral


async def test_a_deferred_job_goes_back_unchanged_and_waits_before_it_is_claimed_again(
    queue: JobQueue, registry: HandlerRegistry, db: Database, clock: ManualClock
) -> None:
    calls: list[int] = []

    async def handler(ctx: JobContext) -> None:
        calls.append(ctx.job.attempts)
        raise JobDeferred("budget level forbids this today", retry_in_s=120)

    registry.register("held", handler)
    job_id = queue.enqueue("held", {"n": 1}, max_attempts=1)
    worker = make_worker(queue, registry, db, clock)
    summary = await worker.run_until_idle()
    assert (summary.deferred, summary.failed, summary.retried, summary.done) == (1, 0, 0, 0)
    job = queue.get(job_id)
    assert job is not None and job.status == "pending" and job.attempts == 0
    assert job.run_after == clock.now_utc() + timedelta(seconds=120)
    assert job.last_error is None
    clock.tick(121)
    await worker.run_until_idle()
    assert calls == [1, 1]  # claimed again after the wait, still on its first attempt


def test_revoking_the_approval_of_a_batch_pauses_only_its_unfinished_jobs(
    queue: JobQueue,
) -> None:
    ids = [
        queue.enqueue("t", {}, batch_id="b", estimated_cost_usd=1.0, requires_approval=True)
        for _ in range(3)
    ]
    other = queue.enqueue("t", {}, batch_id="c", estimated_cost_usd=1.0, requires_approval=True)
    queue.approve_batch("b", max_usd=10)
    queue.approve_batch("c", max_usd=10)
    first = queue.claim_next({"t"}, offpeak_allowed=True, worker_id="w")
    assert first is not None
    queue.complete(first.id, "w")
    assert queue.revoke_approval("b") == 2
    views = {i: queue.get(i) for i in [*ids, other]}
    finished = views[first.id]
    assert finished is not None and finished.status == "done"
    for job_id in ids:
        view = views[job_id]
        assert view is not None
        if job_id != first.id:
            assert view.approved_at is None
    survivor = views[other]
    assert survivor is not None and survivor.approved_at is not None
    assert queue.claim_next({"t"}, offpeak_allowed=True, worker_id="w") is not None  # batch c
    assert queue.claim_next({"t"}, offpeak_allowed=True, worker_id="w") is None
    assert queue.revoke_approval("missing") == 0
