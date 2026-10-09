"""Replaying the real history: known_at, resuming and the one-time batch (R-MEM-010)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import respx
from sqlalchemy import select

from tests.support.deepseek import API, TEST_KEY
from tests.support.embedding import HashingBackend
from tests.support.memory import (
    CloseRule,
    FactRule,
    FollowupRule,
    ScriptedMemoryModel,
    make_memory,
)
from tests.support.policies import AlwaysOffPeak
from tests.support.synth_chat import MessageWriter
from twin.llm.errors import ApiError
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY, LedgerTag
from twin.memory.api import AsOfView, Memory, MemoryQuery, memory_view
from twin.memory.jobs import handle_memory_replay
from twin.memory.replay import (
    REPLAY_JOB,
    MemoryReplayer,
    ReplayRequest,
    estimate_replay,
    group_jobs,
    plan_replay,
    replay_status,
    scan_days,
)
from twin.ops.jobs import BatchTooLargeError, HandlerRegistry, JobQueue, Worker
from twin.profile.holdout import holdout_cutoff
from twin.services import Services
from twin.storage.models import CostLedger

CHICAGO = ZoneInfo("America/Chicago")
SECRET = "我偷偷报了潜水课"


def local(day: int, hour: int, minute: int = 0, month: int = 3) -> datetime:
    """A wall-clock time in Chicago in 2026 as an instant."""
    return datetime(2026, month, day, hour, minute, tzinfo=CHICAGO).astimezone(UTC)


T = local(10, 13)  # the sample: her reply block that first says the secret

CONVERSATION: list[tuple[datetime, bool, str]] = [
    (local(2, 14), False, "你家的猫叫什么"),
    (local(2, 14, 1), True, "我家的猫叫豆包"),
    (local(4, 20, 30), False, "明天下午三点我要考试"),
    (local(4, 20, 31), True, "加油呀"),
    (local(5, 17), False, "考试考完了"),
    (local(5, 17, 1), True, "考得怎么样"),
    (local(9, 12), False, "后天下午四点我要去看牙医"),
    (local(9, 12, 1), True, "记得带医保卡"),
    (T - timedelta(seconds=30), False, "最近有什么新鲜事"),
    (T, True, SECRET),
    (T + timedelta(minutes=5), True, "别告诉别人"),
    (local(11, 9), False, "潜水课好玩吗"),
    (local(11, 9, 1), True, "超好玩"),
    (local(11, 17), False, "牙医看完了"),
    (local(11, 17, 1), True, "太好了"),
    (local(12, 10), False, "周末一起去爬山吧"),
    (local(12, 10, 1), True, "好呀我想去爬山"),  # the last reply block: the hold-out starts here
]


def script() -> ScriptedMemoryModel:
    return ScriptedMemoryModel(
        facts=[
            FactRule(
                "我家的猫叫豆包", {"subject": "her", "category": "life", "text": "她的猫叫豆包"}
            ),
            FactRule(
                "偷偷报了潜水课",
                {"subject": "her", "category": "plan", "text": "她报了潜水课", "importance": 4},
            ),
            FactRule(
                "超好玩", {"subject": "her", "category": "life", "text": "她觉得潜水课很好玩"}
            ),
            FactRule("我想去爬山", {"subject": "her", "category": "plan", "text": "她想去爬山"}),
        ],
        followups=[
            FollowupRule(
                "明天下午三点我要考试",
                {"text": "对方明天下午三点考试", "due": "明天下午三点", "window_minutes": 180},
            ),
            FollowupRule(
                "后天下午四点我要去看牙医",
                {"text": "对方后天下午四点看牙医", "due": "后天下午四点", "window_minutes": 120},
            ),
        ],
        closes=[
            CloseRule("考试考完了", "对方明天下午三点考试"),
            CloseRule("牙医看完了", "对方后天下午四点看牙医"),
        ],
        summaries={"2026-03-10": f"周二她说{SECRET}"},
    )


@dataclass
class Env:
    services: Services
    memory: Memory
    runtime: LlmRuntime
    model: ScriptedMemoryModel
    replayer: MemoryReplayer


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def env(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> AsyncIterator[Env]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    services.settings.memory.recall_min_similarity = 0.3
    writer = MessageWriter(services)
    for at, her, text in CONVERSATION:
        writer.add(at, her, "text", text)
    writer.store()
    runtime = build_llm_runtime(services)
    model = script()
    api.post(API).mock(side_effect=model)
    memory = make_memory(services)
    yield Env(services, memory, runtime, model, MemoryReplayer(memory, runtime.client))
    await runtime.client.aclose()


ALL_DAYS = [date(2026, 3, d) for d in (2, 4, 5, 9, 10, 11, 12)]


# ------------------------------------------------------------------------ the days


def test_the_days_of_the_history_are_found_with_their_size_and_fingerprint(env: Env) -> None:
    stats = scan_days(env.services, env.memory.clock, ReplayRequest(), env.runtime.estimator)
    assert [s.day for s in stats] == ALL_DAYS
    assert [s.lines for s in stats] == [2, 2, 2, 2, 3, 4, 2]
    assert all(s.state == "new" and s.tokens > 0 and len(s.input_hash) == 40 for s in stats)
    ranged = scan_days(
        env.services,
        env.memory.clock,
        ReplayRequest(date(2026, 3, 5), date(2026, 3, 10)),
        env.runtime.estimator,
    )
    assert [s.day for s in ranged] == [date(2026, 3, d) for d in (5, 9, 10)]


def test_a_day_is_cut_where_her_midnight_is(env: Env) -> None:
    """22:00 on the 9th in Chicago is already the 10th in UTC."""
    writer = MessageWriter(env.services)
    writer.add(datetime(2026, 3, 10, 3, 0, tzinfo=UTC), False, "text", "晚安")  # 22:00 on the 9th
    writer.store(append=True)
    stats = scan_days(env.services, env.memory.clock, ReplayRequest(), env.runtime.estimator)
    assert next(s for s in stats if s.day == date(2026, 3, 9)).lines == 3


# ----------------------------------------------------------------------- replaying


async def test_replaying_every_day_writes_summaries_facts_and_follow_ups_with_their_known_times(
    env: Env,
) -> None:
    run = await env.replayer.replay_days(ALL_DAYS, DAILY)
    assert run.replayed == 7 and run.skipped == 0 and run.failures == []
    env.memory.refresh()
    facts = {f.text: f for f in env.memory.corpus.facts.values()}
    assert set(facts) == {"她的猫叫豆包", "她报了潜水课", "她觉得潜水课很好玩", "她想去爬山"}
    cat, secret, fun = facts["她的猫叫豆包"], facts["她报了潜水课"], facts["她觉得潜水课很好玩"]
    assert cat.known_at == local(2, 14, 1)  # the time of the message that proves it
    assert secret.known_at == T  # first said in the target reply block
    assert fun.known_at == local(11, 9, 1)
    assert secret.source == "real_record" and secret.evidence == {
        "kind": "messages",
        "ids": [secret.evidence["ids"][0]],  # type: ignore[index]
    }
    assert env.model.calls["summary"] == 7 and env.model.calls["extract"] == 7
    assert env.memory.store.counts()["memory_replay_days"] == 7
    summaries = {s.local_date: s for s in env.memory.corpus.summaries.values()}
    assert set(summaries) == set(ALL_DAYS)
    day = summaries[date(2026, 3, 10)]
    assert (day.scope, day.timezone, day.version) == ("real", "America/Chicago", 1)
    assert (day.utc_start, day.utc_end) == (local(10, 0), local(11, 0))
    assert day.text == f"周二她说{SECRET}"


async def test_follow_ups_close_at_the_message_that_ends_them_and_the_rest_expire(env: Env) -> None:
    await env.replayer.replay_days(ALL_DAYS, DAILY)
    follows = {f.text: f for f in env.memory.store.followups()}
    exam, dentist = follows["对方明天下午三点考试"], follows["对方后天下午四点看牙医"]
    assert exam.created_at == local(4, 20, 30) and exam.due_at == local(5, 15)
    assert (exam.status, exam.closed_at, exam.close_reason) == ("done", local(5, 17), "mentioned")
    assert dentist.due_at == local(11, 16) and dentist.window_minutes == 120
    assert (dentist.status, dentist.closed_at) == ("done", local(11, 17))
    assert exam.origin == dentist.origin == "real_record"


async def test_the_history_covers_the_held_out_period_too(env: Env) -> None:
    cutoff = holdout_cutoff(env.services)
    assert cutoff == local(12, 10, 1)  # the start of her last reply block
    await env.replayer.replay_days(ALL_DAYS, DAILY)
    env.memory.refresh()
    held_out = [f for f in env.memory.corpus.facts.values() if f.known_at >= cutoff]
    assert [f.text for f in held_out] == ["她想去爬山"]
    assert any(
        s.local_date == cutoff.astimezone(CHICAGO).date()
        for s in env.memory.corpus.summaries.values()
    )


async def test_the_memory_at_the_sample_time_has_nothing_that_was_said_in_the_target_block(
    env: Env,
) -> None:
    """The replay's end-to-end leak check (R-TRN-013): real extraction, then the as-of views."""
    await env.replayer.replay_days(ALL_DAYS, DAILY)
    services = env.services
    for query in ("潜水课", "牙医", "猫"):
        shown = memory_view(services, T, memory=env.memory).render(MemoryQuery(query), 2000)
        via_asof = AsOfView(services, T).memory_block(MemoryQuery(query), 2000)
        assert shown.text == via_asof.text
        for hidden in ("潜水课", "很好玩", "周二她说", "看完了"):
            assert hidden not in shown.text, (query, hidden)
    block = AsOfView(services, T).memory_block(MemoryQuery("猫"), 2000)
    assert "她的猫叫豆包" in block.text and "对方后天下午四点看牙医" in block.text
    (follow,) = [f for f in AsOfView(services, T).memory.followups() if "牙医" in f.text]
    assert follow.status == "open" and follow.closed_at is None  # closed on the 11th, open at T
    after = AsOfView(services, T + timedelta(seconds=1)).memory_block(MemoryQuery("潜水课"), 2000)
    assert "她报了潜水课" in after.text


# --------------------------------------------------------------------- resuming


async def test_a_replay_run_again_does_nothing_for_days_that_are_done(env: Env) -> None:
    await env.replayer.replay_days(ALL_DAYS, DAILY)
    calls = len(env.model.requests)
    again = await env.replayer.replay_days(ALL_DAYS, DAILY)
    assert again.skipped == 7 and again.replayed == 0 and len(env.model.requests) == calls
    forced = await env.replayer.replay_days([date(2026, 3, 10)], DAILY, force=True)
    assert forced.replayed == 1 and len(env.model.requests) > calls
    versions = env.memory.store.summary_versions("real", date(2026, 3, 10))
    assert [v.version for v in versions] == [1, 2] and [v.is_current for v in versions] == [
        False,
        True,
    ]
    env.memory.refresh()
    assert len([f for f in env.memory.corpus.facts.values() if f.text == "她报了潜水课"]) == 1


async def test_a_replay_that_stops_half_way_resumes_without_repeating_or_duplicating(
    env: Env,
) -> None:
    env.model.fail_from_call = 8  # the model refuses from the eighth call: day 4 is under way
    with pytest.raises(ApiError):
        await env.replayer.replay_days(ALL_DAYS, DAILY)
    done_before = set(env.memory.store.replay_days())
    assert 0 < len(done_before) < 7
    env.model.fail_from_call = None
    env.model.calls.clear()
    resumed = await env.replayer.replay_days(ALL_DAYS, DAILY)
    assert resumed.skipped == len(done_before) and resumed.replayed == 7 - len(done_before)
    env.memory.refresh()
    texts = sorted(f.text for f in env.memory.corpus.facts.values() if f.current)
    assert texts == sorted(["她的猫叫豆包", "她报了潜水课", "她觉得潜水课很好玩", "她想去爬山"])
    assert len(env.memory.store.followups()) == 2
    assert len(env.memory.store.current_summaries()) == 7


async def test_a_day_whose_messages_changed_is_replayed_again(env: Env) -> None:
    await env.replayer.replay_days(ALL_DAYS, DAILY)
    writer = MessageWriter(env.services)
    writer.add(local(2, 15), True, "text", "对了它三岁了")
    writer.store(append=True)
    stats = scan_days(env.services, env.memory.clock, ReplayRequest(), env.runtime.estimator)
    states = {s.day: s.state for s in stats}
    assert states[date(2026, 3, 2)] == "changed" and states[date(2026, 3, 4)] == "done"
    plan = estimate_replay(env.services)
    assert [d.day for d in plan.days] == [date(2026, 3, 2)] and plan.already_done == 6
    run = await env.replayer.replay_days([date(2026, 3, 2)], DAILY)
    assert run.replayed == 1
    assert [v.version for v in env.memory.store.summary_versions("real", date(2026, 3, 2))] == [
        1,
        2,
    ]


async def test_a_failing_day_is_noted_and_the_others_go_on(env: Env, api: respx.MockRouter) -> None:
    from tests.support.deepseek import ok

    def answer(request: Any) -> Any:
        body = request.content.decode()
        if "2026-03-05" in body and "摘要员" in body:
            return ok(content="not json at all")
        return env.model(request)

    api.post(API).mock(side_effect=answer)
    run = await env.replayer.replay_days(ALL_DAYS, DAILY)
    assert run.failures == [date(2026, 3, 5)] and run.replayed == 6
    assert date(2026, 3, 5) not in env.memory.store.replay_days()


# ---------------------------------------------------- the one-time batch (R-LLM-014)


async def test_the_estimate_is_an_upper_bound_and_nothing_is_queued_by_asking(env: Env) -> None:
    estimate = estimate_replay(env.services, runtime=env.runtime)
    assert [d.day for d in estimate.days] == ALL_DAYS and estimate.first_replay
    assert estimate.lines == 17 and estimate.tokens > 0 and 0 < estimate.estimated_usd < 1.0
    assert len(estimate.day_prices) == 7 and all(price > 0 for price in estimate.day_prices)
    assert JobQueue(env.services.db, env.services.clock).list_jobs(job_type=REPLAY_JOB) == []


async def test_planning_queues_jobs_that_wait_for_approval_and_cost_nothing_yet(env: Env) -> None:
    env.services.settings.memory.replay_job_days = 4
    plan = plan_replay(env.services, runtime=env.runtime)
    assert not plan.approved and len(plan.batches) == 1 and plan.jobs == 2
    assert plan.batches[0].days == 7 and plan.batches[0].batch_id.startswith("memory-")
    jobs = JobQueue(env.services.db, env.services.clock).list_jobs(job_type=REPLAY_JOB, limit=10)
    assert len(jobs) == 2
    for job in jobs:
        assert job.requires_approval and job.approved_at is None and job.offpeak_only
        assert job.estimated_cost_usd and job.batch_id == plan.batches[0].batch_id
    sizes = sorted(len(job.payload["dates"]) for job in jobs)
    assert sizes == [3, 4]
    assert env.model.requests == []
    status = env.runtime.batches.status(plan.batches[0].batch_id)
    assert not status.approved and status.estimated_usd == pytest.approx(plan.estimated_usd)
    again = plan_replay(env.services, runtime=env.runtime)  # nothing is queued twice
    assert again.batches == [] and again.already_queued == 7


async def test_the_approved_batch_runs_on_the_one_time_account_and_never_touches_the_daily_budget(
    env: Env,
) -> None:
    plan = plan_replay(env.services, runtime=env.runtime)
    batch_id = plan.batches[0].batch_id
    registry = HandlerRegistry()
    registry.register(REPLAY_JOB, handle_memory_replay)
    queue = JobQueue(env.services.db, env.services.clock)
    worker = Worker(
        queue,
        registry,
        env.services.clock,
        services=env.services,
        offpeak=AlwaysOffPeak(),
        alerts=env.services.alerts,
    )
    assert (await worker.run_until_idle()).done == 0  # not approved: nothing runs
    env.runtime.batches.approve(batch_id)
    summary = await worker.run_until_idle()
    assert summary.done == plan.jobs and summary.failed == 0
    with env.services.db.session() as session:
        spent_on = {
            (account, batch)
            for account, batch in session.execute(select(CostLedger.account, CostLedger.batch_id))
        }
    assert spent_on == {("one_time", batch_id)}
    status = env.runtime.batches.status(batch_id)
    assert 0 < status.spent_usd <= status.estimated_usd and not status.paused
    assert env.runtime.ledger.batch_spent_usd(batch_id) == pytest.approx(status.spent_usd)
    assert env.memory.store.counts()["memory_replay_days"] == 7
    report = replay_status(env.services, runtime=env.runtime)
    assert (report.days_with_messages, report.days_replayed, report.days_waiting) == (7, 7, 0)
    assert report.jobs["done"] == plan.jobs and report.summaries == 7
    assert report.facts_by_source == {"real_record": 4} and [
        b.batch_id for b in report.batches
    ] == [batch_id]


async def test_spending_above_the_estimate_pauses_the_batch_and_approving_again_continues(
    env: Env, api: respx.MockRouter
) -> None:
    env.services.settings.memory.replay_job_days = 1
    plan = plan_replay(env.services, runtime=env.runtime)
    batch_id = plan.batches[0].batch_id
    expensive = script()
    expensive.prompt_tokens = 2_000_000  # one call costs more than the whole estimate
    api.post(API).mock(side_effect=expensive)
    registry = HandlerRegistry()
    registry.register(REPLAY_JOB, handle_memory_replay)
    queue = JobQueue(env.services.db, env.services.clock)
    worker = Worker(
        queue,
        registry,
        env.services.clock,
        services=env.services,
        offpeak=AlwaysOffPeak(),
        alerts=env.services.alerts,
    )
    env.runtime.batches.approve(batch_id)
    summary = await worker.run_until_idle()
    paused = env.runtime.batches.status(batch_id)
    assert paused.paused and paused.spent_usd > paused.estimated_usd * 1.2
    assert summary.deferred >= 1 and summary.failed == 0  # deferred: no attempt was used up
    done_when_paused = len(env.memory.store.replay_days())
    assert done_when_paused < 7
    pending = [j for j in queue.list_jobs(job_type=REPLAY_JOB, limit=50) if j.status == "pending"]
    assert pending and all(j.approved_at is None for j in pending)  # their approval was withdrawn
    api.post(API).mock(side_effect=script())  # the model is cheap again
    env.runtime.batches.approve(batch_id)
    env.services.clock.tick(
        3700
    )  # the deferred jobs come back an hour later  # type: ignore[attr-defined]
    await worker.run_until_idle()
    assert not env.runtime.batches.status(batch_id).paused
    assert len(env.memory.store.replay_days()) == 7


async def test_a_large_history_is_split_into_batches_that_each_stay_within_the_limit(
    env: Env,
) -> None:
    estimate = estimate_replay(env.services, runtime=env.runtime)
    smallest = min(estimate.day_prices)
    env.services.settings.budget.one_time_usd = smallest / 2
    with pytest.raises(BatchTooLargeError):  # one day alone costs more than a batch may
        plan_replay(env.services, runtime=env.runtime)
    assert JobQueue(env.services.db, env.services.clock).list_jobs(job_type=REPLAY_JOB) == []
    env.services.settings.budget.one_time_usd = max(estimate.day_prices) * 2.5
    plan = plan_replay(env.services, runtime=env.runtime)
    limit = env.services.settings.budget.one_time_usd
    assert len(plan.batches) > 1 and all(batch.estimated_usd <= limit for batch in plan.batches)
    assert sum(batch.days for batch in plan.batches) == 7


def test_days_are_grouped_into_jobs_by_count_and_by_money() -> None:
    from twin.memory.replay import DayStat

    days = [DayStat(date(2026, 3, d), 1, 1, "h", "new") for d in range(1, 8)]
    assert group_jobs(days, [1.0] * 7, per_job=3, limit_usd=100.0) == [[0, 1, 2], [3, 4, 5], [6]]
    assert group_jobs(days, [4.0] * 7, per_job=5, limit_usd=9.0) == [[0, 1], [2, 3], [4, 5], [6]]
    with pytest.raises(BatchTooLargeError):
        group_jobs(days, [1.0, 20.0, 1.0, 1, 1, 1, 1], per_job=5, limit_usd=9.0)


async def test_with_summaries_and_extraction_off_a_replay_asks_nothing(env: Env) -> None:
    env.services.settings.memory.fact_extraction = False
    env.services.settings.memory.daily_summary = False
    run = await env.replayer.replay_days([date(2026, 3, 2)], DAILY)
    assert run.replayed == 1 and env.model.requests == []
    assert env.memory.store.replay_days()[date(2026, 3, 2)].facts_added == 0
    assert LedgerTag().account == "daily"
