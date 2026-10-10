"""Planning and running the sticker tagging jobs (R-STK-003, R-LLM-014, R-IMP-011, R-TRN-013)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import httpx
import pytest
import respx

from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.embedding import HashingBackend
from tests.support.persona import Scenario, sticker_scenario
from tests.support.policies import AlwaysOffPeak
from tests.support.synth_chat import append_texts
from twin.ingest.hooks import HookContext
from twin.llm.errors import BudgetDeniedError
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.profile.holdout import holdout_cutoff, resplit_holdout
from twin.services import Services
from twin.stickers import download
from twin.stickers.catalog import StickerCatalog
from twin.stickers.hook import describe_plan, queue_sticker_tagging, stickers_after_resplit
from twin.stickers.tag_jobs import (
    TAG_JOB,
    TagPlan,
    find_work,
    handle_sticker_tag,
    never_tagged,
    plan_tagging,
    queued_stickers,
)
from twin.stickers.tagging import StickerTagger
from twin.stickers.vectors import sticker_table
from twin.storage.chat_models import Sticker


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def scenario(services: Services, embedder: HashingBackend) -> Scenario:
    found = sticker_scenario(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    return found


def queue_of(services: Services) -> JobQueue:
    return JobQueue(services.db, services.clock)


def jobs_of(services: Services) -> list[Any]:
    return queue_of(services).list_jobs(job_type=TAG_JOB, limit=100)


async def run_tag_jobs(services: Services) -> Any:
    registry = HandlerRegistry()
    registry.register(TAG_JOB, handle_sticker_tag)
    worker = Worker(
        queue_of(services),
        registry,
        services.clock,
        services=services,
        offpeak=AlwaysOffPeak(),
        alerts=services.alerts,
    )
    return await worker.run_until_idle()


def answer(request: httpx.Request) -> httpx.Response:
    """A vision reply for a tagging call and a correction for a context call."""
    body = request_json(request)
    system = body["messages"][0]["content"]
    if "用法分析员" in system:
        return ok(
            content=json.dumps({"tags": ["委屈"], "meaning": "有点委屈时用"}, ensure_ascii=False)
        )
    return ok(
        content=json.dumps(
            {"tags": ["开心"], "description": "一只笑着的猫", "use_cases": "回应好消息"},
            ensure_ascii=False,
        ),
        prompt=300,
        completion_tokens=40,
    )


# ------------------------------------------------------------------ the work


def test_the_work_is_every_untagged_sticker_and_every_sticker_due_a_correction(
    scenario: Scenario, services: Services
) -> None:
    a, b, c, d = scenario.md5s
    work = find_work(services)
    assert set(work.vision) == {a, b, c, d} and work.vision[:2] == [a, b]  # hers, most used first
    assert work.context == [a, b]  # four and three uses before the cutoff; the others have fewer
    assert work.stale_vectors == 0 and not work.empty
    assert work.stickers[:2] == [a, b] and set(work.stickers) == {a, b, c, d}
    StickerCatalog(services).save_vision(a, ["开心"], "d", "u", at=services.clock.now_utc())
    assert a not in find_work(services).vision and a in find_work(services).context


def test_stickers_without_a_usable_file_are_left_alone(
    scenario: Scenario, services: Services
) -> None:
    a, b = scenario.md5s[:2]
    with services.db.transaction(bump_state=False) as session:
        first, second = session.get(Sticker, a), session.get(Sticker, b)
        assert first is not None and second is not None
        first.status = "pending"
        second.sha256 = None
    work = find_work(services)
    assert a not in work.vision and a not in work.context
    assert b not in work.vision and b not in work.context


def test_a_description_without_a_vector_counts_as_work(
    scenario: Scenario, services: Services
) -> None:
    catalog = StickerCatalog(services)
    a = scenario.md5s[0]
    catalog.save_vision(a, ["开心"], "d", "u", at=services.clock.now_utc())
    catalog.save_context(
        a, ["委屈"], "m", uses=4, cutoff=holdout_cutoff(services), at=services.clock.now_utc()
    )
    assert find_work(services).stale_vectors == 1


def test_the_first_tagging_is_the_first_until_something_was_tagged(
    scenario: Scenario, services: Services
) -> None:
    assert never_tagged(services)
    StickerCatalog(services).save_vision(
        scenario.md5s[0], ["开心"], "d", "u", at=services.clock.now_utc()
    )
    assert not never_tagged(services)


# ---------------------------------------------------------------- planning


def test_the_first_tagging_is_a_priced_batch_waiting_for_approval(
    scenario: Scenario, services: Services
) -> None:
    services.settings.stickers.tag_job_size = 3
    plan = plan_tagging(services)
    assert plan.mode == "batch" and plan.stickers == 4 and plan.contexts == 2 and plan.jobs == 2
    assert 0 < plan.estimated_usd < 1 and len(plan.batch_ids) == 1
    jobs = jobs_of(services)
    assert len(jobs) == 2
    for job in jobs:
        assert job.requires_approval and job.approved_at is None and job.offpeak_only
        assert job.batch_id == plan.batch_ids[0] and job.payload["batch_id"] == job.batch_id
        assert job.estimated_cost_usd and job.estimated_cost_usd > 0
    assert sorted(m for j in jobs for m in j.payload["vision"]) == sorted(scenario.md5s)
    assert sorted(m for j in jobs for m in j.payload["context"]) == sorted(scenario.md5s[:2])
    assert queue_of(services).claim_next({TAG_JOB}, offpeak_allowed=True, worker_id="w") is None
    assert sum(j.estimated_cost_usd or 0 for j in jobs) == pytest.approx(plan.estimated_usd)


def test_planning_again_queues_nothing_new(scenario: Scenario, services: Services) -> None:
    plan_tagging(services)
    queue = queue_of(services)
    assert queued_stickers(queue) == set(scenario.md5s)
    again = plan_tagging(services)
    assert again.mode == "none" and again.already_queued == 4 and len(jobs_of(services)) == 1


def test_a_large_estimate_is_split_into_several_batches(
    scenario: Scenario, services: Services
) -> None:
    services.settings.stickers.tag_job_size = 1
    services.settings.budget.one_time_usd = 0.003
    plan = plan_tagging(services)
    assert len(plan.batch_ids) > 1 and len(set(plan.batch_ids)) == len(plan.batch_ids)
    runtime = build_llm_runtime(services)
    for batch_id in plan.batch_ids:
        assert runtime.batches.status(batch_id).estimated_usd <= 0.003
    assert sum(1 for _ in jobs_of(services)) == plan.jobs == 4


def test_later_work_goes_on_the_daily_account_without_approval(
    scenario: Scenario, services: Services
) -> None:
    StickerCatalog(services).save_vision(
        scenario.md5s[0], ["开心"], "d", "u", at=services.clock.now_utc()
    )
    plan = plan_tagging(services)
    assert plan.mode == "daily" and plan.batch_ids == [] and plan.estimated_usd == 0
    for job in jobs_of(services):
        assert not job.requires_approval and job.offpeak_only and job.payload["batch_id"] is None
    forced = plan_tagging(services, batch=True)  # `twin stickers tag-all` always prices the work
    assert forced.mode == "none"  # (everything is queued already)


def test_tag_all_prices_the_work_even_after_the_first_tagging(
    scenario: Scenario, services: Services
) -> None:
    StickerCatalog(services).save_vision(
        scenario.md5s[0], ["开心"], "d", "u", at=services.clock.now_utc()
    )
    plan = plan_tagging(services, batch=True)
    assert plan.mode == "batch" and plan.estimated_usd > 0 and plan.stickers == 4


def test_nothing_to_do_is_said_so(services: Services) -> None:
    plan = plan_tagging(services)
    assert plan.mode == "none" and plan.stickers == 0 and plan.jobs == 0


# ----------------------------------------------------------------- running


async def test_an_approved_batch_tags_corrects_and_indexes_the_stickers(
    scenario: Scenario, services: Services, api: respx.MockRouter, embedder: HashingBackend
) -> None:
    route = api.post(API).mock(side_effect=answer)
    plan = plan_tagging(services)
    waiting = await run_tag_jobs(services)
    assert waiting.done == 0 and route.call_count == 0  # not before the approval
    runtime = build_llm_runtime(services)
    for batch_id in plan.batch_ids:
        runtime.batches.approve(batch_id)
    done = await run_tag_jobs(services)
    assert done.done == plan.jobs and done.failed == 0
    assert route.call_count == 4 + 2  # four pictures, two corrections from her use
    catalog = StickerCatalog(services)
    a, b, c, d = (catalog.require(m) for m in scenario.md5s)
    assert a.tags == ("委屈",) and a.tag_source == "context" and a.vision_tags == ("开心",)
    assert c.tags == ("开心",) and c.tag_source == "vision" and d.tags == ("开心",)
    assert a.desc_vector_id and b.desc_vector_id and c.desc_vector_id and d.desc_vector_id
    assert (
        sticker_table(services).count() == 4
    )  # the four stickers are hers; user-only ones are not
    assert a.context_note == "有点委屈时用" and b.context_uses == 3
    # every call of the batch is booked on the one-time account
    spent = sum(runtime.ledger.batch_spent_usd(batch_id) for batch_id in plan.batch_ids)
    assert spent > 0
    now = services.clock.now_utc()
    assert (
        runtime.ledger.total_usd(now - timedelta(days=1), now + timedelta(days=1), account="daily")
        == 0
    )
    assert plan_tagging(services).mode == "none"
    assert never_tagged(services) is False


async def test_a_daily_job_runs_without_approval_on_the_daily_account(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    catalog = StickerCatalog(services)
    for md5 in scenario.md5s[1:]:
        catalog.save_vision(md5, ["开心"], "d", "u", at=services.clock.now_utc())
    route = api.post(API).mock(side_effect=answer)
    plan = plan_tagging(services)
    assert plan.mode == "daily"
    done = await run_tag_jobs(services)
    assert done.done == 1 and route.call_count == 3  # one picture, two corrections
    runtime = build_llm_runtime(services)
    now = services.clock.now_utc()
    assert (
        runtime.ledger.total_usd(now - timedelta(days=1), now + timedelta(days=1), account="daily")
        > 0
    )


async def test_a_sticker_that_fails_is_retried_and_the_others_keep_their_tags(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return error(400) if calls["n"] == 2 else answer(request)

    api.post(API).mock(side_effect=flaky)
    plan = plan_tagging(services)
    runtime = build_llm_runtime(services)
    for batch_id in plan.batch_ids:
        runtime.batches.approve(batch_id)
    first = await run_tag_jobs(services)
    assert first.done == 0 and first.retried == 1
    catalog = StickerCatalog(services)
    failed = catalog.require(scenario.md5s[1])
    assert failed.vision_tags == () and len(catalog.records(tagged=True)) == 4  # b has her use only
    assert (
        len([r for r in catalog.records() if r.vision_tags]) == 3
    )  # the others were saved at once
    services.clock.set_time(services.clock.now_utc() + timedelta(hours=1))
    second = await run_tag_jobs(services)
    assert second.done == 1 and catalog.require(scenario.md5s[1]).vision_tags == ("开心",)


async def test_a_denied_budget_hands_the_job_back(
    scenario: Scenario, services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refuse(self: StickerTagger, md5: str, tag: Any = None) -> bool:
        raise BudgetDeniedError("sticker_tag", 3)

    monkeypatch.setattr(StickerTagger, "tag_sticker", refuse)
    plan = plan_tagging(services)
    for batch_id in plan.batch_ids:
        build_llm_runtime(services).batches.approve(batch_id)
    summary = await run_tag_jobs(services)
    assert summary.deferred == 1 and summary.failed == 0
    assert jobs_of(services)[0].status == "pending"


# ------------------------------------------------------- after a download


def test_the_files_that_just_arrived_are_planned(scenario: Scenario, services: Services) -> None:
    download.queue_tagging_of_new_files(services)
    assert len(jobs_of(services)) == 1


def test_a_failure_to_plan_never_fails_the_download(
    scenario: Scenario, services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    import twin.stickers.tag_jobs as tag_jobs

    def broken(*args: Any, **kwargs: Any) -> TagPlan:
        raise RuntimeError("planning failed")

    monkeypatch.setattr(tag_jobs, "plan_tagging", broken)
    download.queue_tagging_of_new_files(services)
    assert jobs_of(services) == []


# ------------------------------------------------------------- the hook


def context(services: Services) -> HookContext:
    return HookContext(services, "run", "conv", None, 5, 0, True)


def test_the_hook_prices_the_first_tagging_and_waits_for_the_approval(
    scenario: Scenario, services: Services
) -> None:
    result = queue_sticker_tagging(context(services))
    assert result.status == "queued" and result.jobs == 1
    assert "4 sticker(s)" in result.detail and "2 correction(s)" in result.detail
    assert "twin jobs approve <batch>" in result.detail and "stickers-" in result.detail
    again = queue_sticker_tagging(context(services))
    assert again.status == "skipped" and "already queued" in again.detail


def test_the_hook_is_quiet_when_there_is_nothing_to_tag(services: Services) -> None:
    result = queue_sticker_tagging(context(services))
    assert result.status == "skipped" and "no sticker" in result.detail


def test_the_plan_is_described_without_any_sticker_content() -> None:
    batch = TagPlan("batch", 3, 1, 1, 0.0123, ["stickers-1"])
    assert "$0.01" in describe_plan(batch) and "stickers-1" in describe_plan(batch)
    daily = TagPlan("daily", 2, 0, 1)
    assert "queued off-peak" in describe_plan(daily) and "correction" not in describe_plan(daily)


# ------------------------------------------------------------ after a re-split


def test_after_a_resplit_the_corrections_of_the_old_cutoff_are_dropped_and_made_again(
    scenario: Scenario, services: Services
) -> None:
    catalog = StickerCatalog(services)
    cutoff = holdout_cutoff(services)
    a, b = scenario.md5s[:2]
    for md5 in (a, b):
        catalog.save_vision(md5, ["开心"], "d", "u", at=services.clock.now_utc())
        catalog.save_context(md5, ["委屈"], "m", uses=3, cutoff=cutoff, at=services.clock.now_utc())
    # newer messages move the split point
    append_texts(
        services,
        [
            (scenario.episode_time(39) + timedelta(days=1, minutes=i), True, "后来")
            for i in range(80)
        ],
    )
    outcome = resplit_holdout(services)
    assert outcome.current.cutoff > cutoff
    notes = " | ".join(outcome.notes)
    assert "stickers: 2 sticker correction(s) made for the old cutoff dropped" in notes
    assert catalog.require(a).context_tags == () and catalog.require(a).tags == ("开心",)
    work = find_work(services)
    assert set(work.context) >= {a, b}
    jobs = jobs_of(services)
    assert jobs and all(not j.requires_approval for j in jobs)
    again = stickers_after_resplit(services, None, outcome.current)
    assert "nothing to queue" in again


async def test_a_description_without_a_vector_gets_a_job_that_only_indexes(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    catalog = StickerCatalog(services)
    cutoff = holdout_cutoff(services)
    for md5 in scenario.md5s:
        catalog.save_vision(md5, ["开心"], "一只猫", "场合", at=services.clock.now_utc())
    for md5 in scenario.md5s[:2]:
        catalog.save_context(md5, ["委屈"], "m", uses=3, cutoff=cutoff, at=services.clock.now_utc())
    plan = plan_tagging(services)
    assert plan.mode == "daily" and plan.jobs == 1 and plan.stickers == 0
    job = jobs_of(services)[0]
    assert job.payload == {"vision": [], "context": [], "batch_id": None}
    assert not job.requires_approval
    assert plan_tagging(services).mode == "none"  # the waiting job will do it
    route = api.post(API).mock(side_effect=answer)
    done = await run_tag_jobs(services)
    assert done.done == 1 and route.call_count == 0  # no model call, only the embedding
    assert sticker_table(services).count() == 4
    assert plan_tagging(services).mode == "none"
