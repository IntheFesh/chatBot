"""Image descriptions (R-IMP-012, R-LLM-014, R-LLM-004, R-LLM-009)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import select

from tests.fixtures.synth_export import SynthExport
from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.ingest import make_export, run_import
from tests.support.policies import AlwaysOffPeak
from tests.support.synthetic import mobile
from twin.ingest.captions import (
    CAPTION_JOB,
    CAPTION_JOB_SIZE,
    CaptionService,
    caption_messages,
    clean_caption,
    handle_caption_job,
    plan_caption_batches,
    queued_asset_ids,
)
from twin.llm.ledger import LedgerRecord
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.llm.types import CostBreakdown, Usage
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.services import Services
from twin.storage.chat_models import MediaAsset, Message
from twin.storage.models import Job

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
HEAVY_IMAGES = {"image": 60.0, "text": 40.0}


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def imported(services: Services, tmp_path, clock: ManualClock) -> SynthExport:  # type: ignore[no-untyped-def]
    """A conversation spanning about 150 days, half of it older than the 90 day window."""
    clock.set_time(NOW)
    export = make_export(
        tmp_path,
        target_messages=240,
        mix=HEAVY_IMAGES,
        start=datetime(2026, 4, 20, 8, tzinfo=UTC),
        average_gap_s=43_200.0,
        missing_image_ratio=0.1,
        other_conversations=0,
        include_group=False,
    )
    run_import(services, export)
    drop_queued_descriptions(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    return export


def drop_queued_descriptions(services: Services) -> None:
    """Remove the jobs the post-import hook queued, so a test can plan them itself."""
    with services.db.transaction() as session:
        for job in session.scalars(select(Job).where(Job.type == CAPTION_JOB)):
            session.delete(job)


def image_assets(services: Services, *, since: datetime | None = None) -> list[str]:
    """Ids of available image assets (optionally only those of messages since ``since``)."""
    with services.db.session() as session:
        stmt = (
            select(MediaAsset.id, Message.create_time_utc)
            .join(Message, Message.id == MediaAsset.message_id)
            .where(MediaAsset.kind == "image", MediaAsset.status == "available")
        )
        rows = session.execute(stmt).all()
    return sorted(i for i, when in rows if since is None or when >= since)


def captions(services: Services) -> dict[str, str | None]:
    with services.db.session() as session:
        return {a.id: a.caption for a in session.scalars(select(MediaAsset))}


def caption_reply(text: str = "一只猫趴在窗台上") -> httpx.Response:
    return ok(content=text, prompt=300, completion_tokens=12)


async def run_queue(services: Services, *, batch: str | None = None) -> Any:
    registry = HandlerRegistry()
    registry.register(CAPTION_JOB, handle_caption_job)
    worker = Worker(
        JobQueue(services.db, services.clock),
        registry,
        services.clock,
        services=services,
        offpeak=AlwaysOffPeak(),
        alerts=services.alerts,
    )
    return await worker.run_until_idle()


# ----------------------------------------------------------------------- planning


def test_only_pictures_of_the_last_90_days_are_queued_and_they_await_approval(
    services: Services, imported: SynthExport
) -> None:
    cutoff = NOW - timedelta(days=90)
    recent = image_assets(services, since=cutoff)
    everything = image_assets(services)
    assert 5 < len(recent) < len(everything)
    plan = plan_caption_batches(services, days=90)
    assert plan.images == len(recent) and plan.days == 90 and plan.estimated_usd > 0
    queue = JobQueue(services.db, services.clock)
    jobs = queue.list_jobs(job_type=CAPTION_JOB, limit=1000)
    queued = [asset for job in jobs for asset in job.payload["asset_ids"]]
    assert sorted(queued) == recent
    assert all(len(job.payload["asset_ids"]) <= CAPTION_JOB_SIZE for job in jobs)
    for job in jobs:
        assert job.requires_approval and job.approved_at is None  # R-LLM-014
        assert job.offpeak_only and job.status == "pending"
        assert job.batch_id in {batch.batch_id for batch in plan.batches}
        assert job.estimated_cost_usd and job.estimated_cost_usd > 0
        assert job.payload["batch_id"] == job.batch_id
    assert sum(job.estimated_cost_usd or 0 for job in jobs) == pytest.approx(plan.estimated_usd)


def test_planning_again_queues_nothing_new(services: Services, imported: SynthExport) -> None:
    first = plan_caption_batches(services, days=90)
    second = plan_caption_batches(services, days=90)
    assert second.images == 0 and second.already_queued == first.images and not second.batches


def test_a_longer_window_adds_the_older_pictures(services: Services, imported: SynthExport) -> None:
    first = plan_caption_batches(services, days=90)
    wider = plan_caption_batches(services, days=3650)
    assert first.images + wider.images == len(image_assets(services))


def test_a_batch_above_the_one_time_limit_is_split(
    services: Services, imported: SynthExport
) -> None:
    services.settings.budget.one_time_usd = 0.002
    plan = plan_caption_batches(services, days=90)
    assert len(plan.batches) > 1
    assert all(batch.estimated_usd <= 0.002 for batch in plan.batches)
    assert sum(batch.images for batch in plan.batches) == plan.images


def test_there_is_nothing_to_queue_without_pictures(services: Services, tmp_path, clock) -> None:  # type: ignore[no-untyped-def]
    export = make_export(tmp_path, target_messages=30, mix={"text": 100.0})
    run_import(services, export)
    assert plan_caption_batches(services, days=90).images == 0


def test_queued_ids_cover_pending_and_running_jobs(
    services: Services, imported: SynthExport
) -> None:
    queue = JobQueue(services.db, services.clock)
    assert queued_asset_ids(queue) == set()
    plan_caption_batches(services, days=90)
    assert len(queued_asset_ids(queue)) == len(
        image_assets(services, since=NOW - timedelta(days=90))
    )


def test_the_prompt_asks_for_one_objective_sentence_without_guessing_who(
    services: Services,
) -> None:
    system, user = caption_messages()
    assert "一句" in system["content"] and "不要猜测" in system["content"]  # type: ignore[operator]
    assert user["role"] == "user"


# ---------------------------------------------------------------------- approval


def test_jobs_wait_for_approval_and_for_the_off_peak_window(
    services: Services, imported: SynthExport
) -> None:
    plan = plan_caption_batches(services, days=90)
    queue = JobQueue(services.db, services.clock)
    types = {CAPTION_JOB}
    assert queue.claim_next(types, offpeak_allowed=True, worker_id="w") is None  # not approved
    batches = build_llm_runtime(services).batches
    approval = batches.approve(plan.batches[0].batch_id)
    assert approval.job_count == plan.batches[0].jobs
    assert queue.claim_next(types, offpeak_allowed=False, worker_id="w") is None  # peak hours
    claimed = queue.claim_next(types, offpeak_allowed=True, worker_id="w")
    assert claimed is not None and claimed.type == CAPTION_JOB


# ------------------------------------------------------------------------- jobs


async def test_an_approved_batch_describes_redacts_and_books_the_cost_as_one_time(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    plan = plan_caption_batches(services, days=90)
    runtime = build_llm_runtime(services)
    for batch in plan.batches:
        runtime.batches.approve(batch.batch_id)
    secret_reply = f"门口的招牌上写着电话 {mobile()}，旁边是一只猫"
    route = api.post(API).mock(return_value=caption_reply(secret_reply))
    summary = await run_queue(services)
    recent = image_assets(services, since=NOW - timedelta(days=90))
    assert summary.done == plan.jobs and summary.failed == 0
    assert route.call_count == len(recent)
    stored = captions(services)
    assert {k for k, v in stored.items() if v} == set(recent)
    for caption in (v for v in stored.values() if v):
        assert mobile() not in caption and "[手机号]" in caption  # redacted before storing
    # what was sent: a user message with the picture, the system prompt, no thinking
    body = request_json(route.calls[0].request)
    assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "disabled"}
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    parts = body["messages"][1]["content"]
    image_part = next(p for p in parts if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/")
    # the ledger: one-time account, caption purpose, the batch id
    ledger = runtime.ledger
    spent = sum(ledger.batch_spent_usd(batch.batch_id) for batch in plan.batches)
    assert spent > 0
    day_start, day_end = NOW - timedelta(days=1), NOW + timedelta(days=1)
    assert ledger.total_usd(day_start, day_end, account="daily") == 0  # R-LLM-014
    assert ledger.total_usd(
        day_start, day_end, account="one_time", purpose="caption"
    ) == pytest.approx(spent)


async def test_the_picture_that_was_sent_is_the_stored_one(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    plan = plan_caption_batches(services, days=90)
    build_llm_runtime(services).batches.approve(plan.batches[0].batch_id)
    seen: list[bytes] = []

    def answer(request: httpx.Request) -> httpx.Response:
        parts = json.loads(request.content)["messages"][1]["content"]
        url = next(p for p in parts if p["type"] == "image_url")["image_url"]["url"]
        seen.append(base64.b64decode(url.split(",", 1)[1]))
        return caption_reply()

    api.post(API).mock(side_effect=answer)
    await run_queue(services)
    with services.db.session() as session:
        stored = {
            services.media.read_bytes(a.sha256)
            for a in session.scalars(select(MediaAsset).where(MediaAsset.caption_ct.is_not(None)))
            if a.sha256
        }
    assert stored and set(seen) == stored


async def test_a_failed_description_is_retried_and_finished_ones_are_kept(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    plan = plan_caption_batches(services, days=90)
    runtime = build_llm_runtime(services)
    runtime.batches.approve(plan.batches[0].batch_id)
    route = api.post(API)
    route.side_effect = [caption_reply("第一张")] + [error(401, "bad key")] * 100
    summary = await run_queue(services)
    assert summary.retried >= 1 and summary.failed == 0
    stored = [c for c in captions(services).values() if c]
    assert stored == ["第一张"]


async def test_when_the_daily_budget_is_exhausted_a_single_request_waits(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    asset_id = image_assets(services)[0]
    CaptionService(services).queue_caption(asset_id)
    services.settings.budget.daily_usd = 0.10
    runtime = build_llm_runtime(services)
    runtime.ledger.record(
        LedgerRecord(
            provider="deepseek",
            model="deepseek-flash",
            purpose="reply",
            usage=Usage(prompt_tokens=1, completion_tokens=1, cache_miss_tokens=1),
            cost=CostBreakdown(5.0, 0.0, 0.0, True, 1.0),
            thinking=False,
            latency_ms=1,
            at=services.clock.now_utc(),
        )
    )
    route = api.post(API).mock(return_value=caption_reply())
    summary = await run_queue(services)
    assert summary.deferred == 1 and summary.failed == 0 and route.call_count == 0
    (job,) = JobQueue(services.db, services.clock).list_jobs(job_type=CAPTION_JOB)
    assert job.status == "pending" and job.attempts == 0  # waits for a better day, no attempt lost


# ---------------------------------------------------------------- get_caption


async def test_a_stored_description_is_returned_without_calling_the_api(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    asset_id = image_assets(services)[0]
    with services.db.transaction() as session:
        asset = session.get(MediaAsset, asset_id)
        assert asset is not None
        asset.caption = "已有的描述"
    route = api.post(API).mock(return_value=caption_reply())
    service = CaptionService(services)
    try:
        assert await service.get_caption(asset_id, wait=False) == "已有的描述"
        assert await service.get_caption(asset_id, wait=True) == "已有的描述"
    finally:
        await service.aclose()
    assert route.call_count == 0


async def test_history_is_never_waited_for_it_returns_none_and_queues_a_job(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    asset_id = image_assets(services)[0]
    route = api.post(API).mock(return_value=caption_reply())
    service = CaptionService(services)
    try:
        assert await service.get_caption(asset_id, wait=False) is None
        assert await service.get_caption(asset_id, wait=False) is None  # not queued twice
    finally:
        await service.aclose()
    assert route.call_count == 0  # not a single request on the reply path
    jobs = JobQueue(services.db, services.clock).list_jobs(job_type=CAPTION_JOB)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.payload == {"asset_ids": [asset_id], "batch_id": None}
    assert not job.requires_approval and job.offpeak_only  # a single picture needs no approval


async def test_a_picture_just_received_is_described_at_once_and_cached(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    asset_id = image_assets(services)[0]
    route = api.post(API).mock(return_value=caption_reply("窗边的一杯咖啡"))
    service = CaptionService(services)
    try:
        assert await service.get_caption(asset_id, wait=True) == "窗边的一杯咖啡"
        assert await service.get_caption(asset_id, wait=True) == "窗边的一杯咖啡"
        assert await service.get_caption(asset_id, wait=False) == "窗边的一杯咖啡"
    finally:
        await service.aclose()
    assert route.call_count == 1
    assert captions(services)[asset_id] == "窗边的一杯咖啡"
    ledger = build_llm_runtime(services).ledger
    day_start, day_end = NOW - timedelta(days=1), NOW + timedelta(days=1)
    assert ledger.total_usd(day_start, day_end, account="daily", purpose="caption") > 0
    assert not JobQueue(services.db, services.clock).list_jobs(job_type=CAPTION_JOB)


async def test_waiting_has_a_timeout_and_leaves_a_job_behind(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    asset_id = image_assets(services)[0]
    services.settings.ingest.caption_wait_timeout_s = 0.05
    never = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        await never.wait()
        return caption_reply()

    api.post(API).mock(side_effect=hang)
    service = CaptionService(services)
    try:
        assert await service.get_caption(asset_id, wait=True) is None
    finally:
        await service.aclose()
    assert captions(services)[asset_id] is None
    assert len(JobQueue(services.db, services.clock).list_jobs(job_type=CAPTION_JOB)) == 1


async def test_an_api_error_on_the_live_path_gives_none_not_an_exception(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    asset_id = image_assets(services)[0]
    api.post(API).mock(return_value=error(401, "bad key"))
    service = CaptionService(services)
    try:
        assert await service.get_caption(asset_id, wait=True) is None
    finally:
        await service.aclose()
    assert len(JobQueue(services.db, services.clock).list_jobs(job_type=CAPTION_JOB)) == 1


async def test_missing_unknown_and_unavailable_pictures_have_no_description(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=caption_reply())
    with services.db.session() as session:
        missing = session.scalars(
            select(MediaAsset.id).where(MediaAsset.status == "missing")
        ).first()
    service = CaptionService(services)
    try:
        assert await service.get_caption("0" * 32, wait=True) is None
        if missing is not None:
            assert await service.get_caption(missing, wait=True) is None
            assert await service.get_caption(missing, wait=False) is None
    finally:
        await service.aclose()
    assert route.call_count == 0


async def test_a_media_asset_object_is_accepted_as_well_as_its_id(
    services: Services, imported: SynthExport, api: respx.MockRouter
) -> None:
    asset_id = image_assets(services)[0]
    api.post(API).mock(return_value=caption_reply("桌上的书"))
    with services.db.session() as session:
        asset = session.get(MediaAsset, asset_id)
        assert asset is not None
    service = CaptionService(services)
    try:
        assert await service.get_caption(asset, wait=True) == "桌上的书"
    finally:
        await service.aclose()


# ----------------------------------------------------------------------- cleaning


def test_captions_are_one_short_line_and_redacted() -> None:
    assert clean_caption("  “一只猫\n趴着”  ") == "一只猫 趴着"
    long = clean_caption("很长" * 200)
    assert len(long) == 200 and long.endswith("…")
    assert "[手机号]" in clean_caption(f"墙上写着 {mobile()}")
    assert clean_caption("   ") == ""


def test_the_import_itself_queues_the_recent_pictures_for_approval(
    services: Services,
    tmp_path,
    clock: ManualClock,  # type: ignore[no-untyped-def]
) -> None:
    clock.set_time(NOW)
    export = make_export(
        tmp_path,
        target_messages=240,
        mix=HEAVY_IMAGES,
        start=datetime(2026, 4, 20, 8, tzinfo=UTC),
        average_gap_s=43_200.0,
        other_conversations=0,
        include_group=False,
    )
    outcome = run_import(services, export)
    hook = outcome.run.hooks["image_caption"]
    recent = image_assets(services, since=NOW - timedelta(days=90))
    assert hook["status"] == "queued" and "twin jobs approve" in hook["detail"]
    assert hook["jobs"] >= 1 and hook["backfill_command"] == "images caption-backfill"
    queue = JobQueue(services.db, services.clock)
    queued = {
        a
        for job in queue.list_jobs(job_type=CAPTION_JOB, limit=1000)
        for a in job.payload["asset_ids"]
    }
    assert queued == set(recent)
    assert all(
        j.requires_approval and j.approved_at is None
        for j in queue.list_jobs(job_type=CAPTION_JOB, limit=1000)
    )
