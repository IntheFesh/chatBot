"""The daily life line: drawn when she wakes, checked by code and by the model (R-MEM-005).

DeepSeek is scripted (``respx``): the tests say what the model draws and what the checker finds.
Covered: the input of the draw (plan, recent days, real facts only), the rule failure that is drawn
again with the reasons, a contradiction with a real fact found by the checker, the day corrected
by rule after two failures with an alert, the job and ``lifeline_at``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
import respx
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY
from tests.support.embedding import HashingBackend
from tests.support.lifeline import ScriptedLifelineModel, event, good_workday
from tests.support.memory import add_event, add_fact, make_memory
from tests.support.policies import AlwaysOffPeak
from tests.support.routine import Rig, fixed_model
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY
from twin.memory.api import lifeline_at
from twin.memory.lifeline import LifelineStore, PlannedEvent
from twin.memory.lifeline_gen import LifelineGenerator
from twin.memory.memory import Memory
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.schedule.jobs import LIFELINE_JOB, handle_lifeline_generate, queue_lifeline
from twin.services import Services
from twin.storage.models import Alert

FRIDAY = date(2026, 10, 9)
THURSDAY = date(2026, 10, 8)
SHANGHAI = "Asia/Shanghai"


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def runtime(services: Services, embedder: HashingBackend) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


@pytest.fixture
def rig(services: Services, clock: ManualClock) -> Rig:
    clock.set_time(utc(2026, 10, 9, 12, 30))  # 07:30 in Chicago: she wakes now
    return Rig.build(services, clock, fixed_model())


@pytest.fixture
def memory(services: Services, embedder: HashingBackend) -> Memory:
    return make_memory(services)


def generator(rig: Rig, memory: Memory, runtime: LlmRuntime) -> LifelineGenerator:
    return LifelineGenerator(memory, runtime.client, rig.kit.time)


def stored(memory: Memory, day: date = FRIDAY) -> list[tuple[str, str, str]]:
    store = LifelineStore(memory)
    return [(e.start_local or "", e.end_local or "", e.activity) for e in store.day(day)]


def alerts(services: Services) -> list[tuple[str, str, str | None, dict[str, Any] | None]]:
    """``(category, severity, dedup key, detail)`` of every alert raised so far."""
    with services.db.session() as session:
        return [
            (a.category, a.severity, a.dedup_key, a.detail) for a in session.scalars(select(Alert))
        ]


def real_facts(memory: Memory) -> None:
    known = utc(2026, 9, 1)
    add_fact(
        memory, "她在北京大学读研究生，周五下午在实验室", known, source="real_record",
        subject="her", category="work_study", importance=4, embed=False,
    )  # fmt: skip
    add_fact(
        memory, "她住在学校附近的公寓", known, source="real_record", subject="her",
        category="life", importance=3, embed=False,
    )  # fmt: skip
    add_fact(
        memory, "她昨天去爬山了", known, source="bot_invented", subject="her", category="life",
        embed=False,
    )  # fmt: skip
    add_fact(
        memory, "她好像下个月要出差", known, source="user_said", subject="her", category="plan",
        embed=False,
    )  # fmt: skip
    add_fact(
        memory, "对方在芝加哥读书", known, source="real_record", subject="user", category="life",
        embed=False,
    )  # fmt: skip
    add_fact(
        memory, "她喜欢看悬疑电影", known, source="real_record", subject="her",
        category="preference", importance=2, embed=False,
    )  # fmt: skip


# ----------------------------------------------------------------- the first draw


async def test_a_consistent_day_is_drawn_checked_and_stored(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter, services: Services
) -> None:
    real_facts(memory)
    LifelineStore(memory).replace_plan(
        THURSDAY,
        [PlannedEvent("在图书馆写综述", "09:00", "12:00", "图书馆", "专注")],
    )
    plan = rig.planner.ensure(FRIDAY)
    model = ScriptedLifelineModel()
    api.post(API).mock(side_effect=model)
    result = await generator(rig, memory, runtime).generate(plan, DAILY)
    assert (result.day, result.plan_id, result.events) == (FRIDAY, plan.id, 7)
    assert (result.drafts, result.corrected, result.left, result.calls) == (1, False, (), 2)
    assert result.cost_usd > 0
    assert stored(memory) == [(e["start"], e["end"], e["activity"]) for e in good_workday()]
    entries = LifelineStore(memory).day(FRIDAY)
    assert {e.source for e in entries} == {"plan"} and all(
        e.consistency_checked_at for e in entries
    )
    assert (entries[0].place, entries[0].mood, entries[0].detail) == ("家", "平静", "煎蛋配牛奶")
    assert memory.era().online_at is not None  # a drawn day is made by the bot
    assert model.draws == 1 and model.reviews == 1


async def test_the_draw_is_given_the_plan_the_recent_days_and_only_real_facts(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    real_facts(memory)
    add_fact(
        memory, "她周五下午有组会", utc(2026, 9, 2), source="real_record", subject="her",
        category="other", event_date=FRIDAY, embed=False,
    )  # fmt: skip
    LifelineStore(memory).replace_plan(
        THURSDAY, [PlannedEvent("在图书馆写综述", "09:00", "12:00", "图书馆", "专注")]
    )
    plan = rig.planner.ensure(FRIDAY)
    model = ScriptedLifelineModel()
    api.post(API).mock(side_effect=model)
    await generator(rig, memory, runtime).generate(plan, DAILY)
    (prompt,) = model.prompts("draw")
    assert "日期：2026-10-09（周五，工作日，时区 America/Chicago）" in prompt
    assert "起床：07:30" in prompt and "睡觉：23:30" in prompt
    assert "- 13:00–17:00" in prompt and "饭点：\n- " in prompt and "午饭约" in prompt
    assert "晚饭约" in prompt
    assert "10月8日周四" in prompt and "09:00-12:00 在图书馆写综述（图书馆，专注）" in prompt
    assert "她在北京大学读研究生，周五下午在实验室" in prompt and "她住在学校附近的公寓" in prompt
    assert "她喜欢看悬疑电影" in prompt and "她周五下午有组会" in prompt
    for kept_out in ("她昨天去爬山了", "她好像下个月要出差", "对方在芝加哥读书"):
        assert kept_out not in prompt  # only what the records say about her
    assert "上一版被退回了" not in prompt
    (review,) = model.prompts("check")
    assert "1. 07:45-08:15 吃早饭" in review and "4. 13:00-17:00 上课和做实验（实验室）" in review
    assert "她在北京大学读研究生" in review and "09:00-12:00 在图书馆写综述" in review


async def test_without_any_fact_or_earlier_day_the_prompt_says_so(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    model = ScriptedLifelineModel()
    api.post(API).mock(side_effect=model)
    await generator(rig, memory, runtime).generate(plan, DAILY)
    prompt = model.prompts("draw")[0]
    assert (
        "已知的事（来自真实记录）：\n（没有）" in prompt
        and "最近几天她的安排：\n（没有）" in prompt
    )


# ------------------------------------------------------------ the second draw


def broken_day() -> list[dict[str, Any]]:
    return [
        event("06:00", "08:00", "写论文"),  # she sleeps until 07:30
        event("07:45", "09:00", "吃早饭"),  # overlaps the previous one
        event("13:00", "17:00", "在咖啡馆发呆", busy=False),  # a busy period, not marked
        event("18:00", "19:00", "吃晚饭"),
    ]


async def test_a_day_that_breaks_the_rules_is_drawn_again_with_the_reasons(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    model = ScriptedLifelineModel(days=[broken_day(), good_workday()])
    api.post(API).mock(side_effect=model)
    result = await generator(rig, memory, runtime).generate(plan, DAILY)
    assert (result.drafts, result.corrected, result.events, result.calls) == (2, False, 7, 3)
    assert model.draws == 2 and model.reviews == 1  # the broken draw was not sent to the checker
    first, second = model.prompts("draw")
    assert "上一版被退回了" not in first and "上一版被退回了" in second
    assert "第 1 段（06:00-08:00）落在她睡觉的时间里" in second
    assert "第 2 段与第 1 段的时间重叠" in second
    assert "第 3 段落在忙碌时段内，要写成她在忙的事，busy 必须是 true" in second
    assert stored(memory)[0] == ("07:45", "08:15", "吃早饭")  # the second draw is what was stored


async def test_a_day_that_contradicts_a_real_fact_is_drawn_again(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    real_facts(memory)
    plan = rig.planner.ensure(FRIDAY)
    other = good_workday()
    other[3] = event("13:00", "17:00", "在实验室做实验", place="实验室", busy=True)
    model = ScriptedLifelineModel(
        days=[good_workday(), other],
        checks=[
            [{"event": 4, "against": "fact", "reason": "已知她周五下午在实验室，不是在上课"}],
            [],
        ],
    )
    api.post(API).mock(side_effect=model)
    result = await generator(rig, memory, runtime).generate(plan, DAILY)
    assert (result.drafts, result.corrected, result.calls) == (2, False, 4)
    assert model.draws == 2 and model.reviews == 2
    second = model.prompts("draw")[1]
    assert "第 4 段（上课和做实验）与已知的事矛盾：已知她周五下午在实验室，不是在上课" in second
    assert ("13:00", "17:00", "在实验室做实验") in stored(memory)


async def test_a_contradiction_with_the_recent_days_is_told_as_such(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    model = ScriptedLifelineModel(
        checks=[[{"event": 2, "against": "recent_day", "reason": "昨天说好今天去复查"}], []],
    )
    api.post(API).mock(side_effect=model)
    await generator(rig, memory, runtime).generate(plan, DAILY)
    assert "与最近几天的安排衔接不上：昨天说好今天去复查" in model.prompts("draw")[1]


async def test_a_checker_that_names_a_stretch_that_does_not_exist_is_ignored(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    model = ScriptedLifelineModel(checks=[[{"event": 99, "against": "fact", "reason": "?"}]])
    api.post(API).mock(side_effect=model)
    result = await generator(rig, memory, runtime).generate(plan, DAILY)
    assert result.drafts == 1 and result.events == 7


# ---------------------------------------------------------- corrected by rule


async def test_a_day_that_fails_twice_is_corrected_by_rule_and_reported(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter, services: Services
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    model = ScriptedLifelineModel(days=[broken_day(), broken_day()])
    api.post(API).mock(side_effect=model)
    result = await generator(rig, memory, runtime).generate(plan, DAILY)
    assert result.drafts == 2 and result.corrected and result.calls == 2
    assert set(result.left) == {"during_sleep", "overlap", "busy_mismatch"}
    assert model.reviews == 0  # the checker is only asked about a day that obeys the rules
    assert stored(memory) == [
        ("07:30", "08:00", "写论文"),  # the sleeping part cut off
        ("08:00", "09:00", "吃早饭"),  # moved behind the previous one
        ("13:00", "17:00", "在咖啡馆发呆"),  # now marked busy
        ("18:00", "19:00", "吃晚饭"),
    ]
    store = LifelineStore(memory, time_service=rig.kit.time)
    assert store.check_consistency(FRIDAY).ok
    ((category, severity, key, detail),) = alerts(services)
    assert (category, severity) == ("lifeline_corrected", "warning")
    assert key == "lifeline_corrected:2026-10-09"
    assert detail is not None and detail["drawn"] == 4 and detail["kept"] == 4
    assert "写论文" not in str(detail)  # an alert never carries the day's text


async def test_a_contradiction_that_stays_is_dropped_from_the_corrected_day(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter, services: Services
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    flagged = [{"event": 4, "against": "fact", "reason": "已知她不上课"}]
    model = ScriptedLifelineModel(days=[good_workday(), good_workday()], checks=[flagged, flagged])
    api.post(API).mock(side_effect=model)
    result = await generator(rig, memory, runtime).generate(plan, DAILY)
    assert result.corrected and result.left == ("contradiction",) and result.events == 6
    assert "上课和做实验" not in [a for _, _, a in stored(memory)]
    assert len(alerts(services)) == 1


async def test_what_the_bot_let_slip_stays_and_the_old_plan_goes(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    add_event(memory, FRIDAY, "早上去买了奶茶", start="08:20", end="08:25")  # an earlier plan
    store = LifelineStore(memory)
    store.add_improvised(FRIDAY, PlannedEvent("跟室友吵了一架", "21:00", "21:30"))
    api.post(API).mock(side_effect=ScriptedLifelineModel(days=[[*good_workday()[:3]]]))
    await generator(rig, memory, runtime).generate(plan, DAILY)
    activities = [a for _, _, a in stored(memory)]
    assert "早上去买了奶茶" not in activities  # the earlier plan is replaced
    assert "跟室友吵了一架" in activities  # what she told the user stays


async def test_a_failing_model_stores_nothing(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    api.post(API).mock(side_effect=ScriptedLifelineModel(fail_from_call=1))
    with pytest.raises(Exception):  # noqa: B017 - any API failure surfaces unchanged
        await generator(rig, memory, runtime).generate(plan, DAILY)
    assert stored(memory) == []


# ------------------------------------------------------------- a plan mid-day


async def test_a_day_planned_after_a_time_zone_switch_starts_where_the_plan_starts(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    rig.planner.refresh("startup")
    rig.move_to(rig.at(2026, 10, 9, 10))  # 15:00 UTC = 23:00 in Shanghai
    plan = rig.planner.switch_timezone(SHANGHAI).plan
    assert plan is not None
    early = [event("08:00", "09:00", "吃早饭"), *good_workday()[3:]]
    model = ScriptedLifelineModel(days=[early, [event("23:00", "23:25", "洗漱，准备睡觉")]])
    api.post(API).mock(side_effect=model)
    result = await generator(rig, memory, runtime).generate(plan, DAILY)
    second = model.prompts("draw")[1]
    assert (
        "第 1 段在 23:00 之前开始" in second
        and "（已经醒着，这份安排从 23:00 开始写）" in model.prompts("draw")[0]
    )
    assert result.drafts == 2 and result.corrected is False
    assert LifelineStore(memory).day(FRIDAY)[0].timezone == SHANGHAI


# ------------------------------------------------------------------------ the job


def worker(services: Services) -> Worker:
    registry = HandlerRegistry()
    registry.register(LIFELINE_JOB, handle_lifeline_generate)
    return Worker(
        JobQueue(services.db, services.clock),
        registry,
        services.clock,
        services=services,
        offpeak=AlwaysOffPeak(),
        alerts=services.alerts,
    )


async def test_the_job_draws_the_day_and_marks_the_plan(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter, services: Services
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    job_id = queue_lifeline(services, plan)
    rig.planner.store.mark_lifeline_queued(plan.id, job_id)
    model = ScriptedLifelineModel()
    api.post(API).mock(side_effect=model)
    summary = await worker(services).run_until_idle()
    assert summary.done == 1 and summary.failed == 0
    assert len(stored(memory)) == 7
    done = rig.planner.store.get(plan.id)
    assert done is not None and done.lifeline_done_at == services.clock.now_utc()
    (queued,) = JobQueue(services.db, services.clock).list_jobs(job_type=LIFELINE_JOB)
    assert queued.status == "done" and queued.max_attempts == 4 and not queued.offpeak_only


async def test_a_job_for_a_plan_that_is_gone_or_replaced_has_nothing_to_do(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter, services: Services
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    queue_lifeline(services, plan)
    rig.move_to(rig.at(2026, 10, 9, 10))
    rig.planner.switch_timezone(SHANGHAI)  # the Chicago plan is replaced by the Shanghai one
    model = ScriptedLifelineModel()
    api.post(API).mock(side_effect=model)
    assert (await worker(services).run_until_idle()).done == 1
    assert model.requests == [] and stored(memory) == []
    queue = JobQueue(services.db, services.clock)
    queue.enqueue(LIFELINE_JOB, {"date": "2026-10-09", "plan_id": "does-not-exist"})
    assert (await worker(services).run_until_idle()).done == 1
    assert model.requests == []


@pytest.mark.parametrize("failure", ["circuit", "budget"])
async def test_a_denied_budget_or_an_open_circuit_hands_the_job_back(
    rig: Rig,
    memory: Memory,
    runtime: LlmRuntime,
    services: Services,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    async def refuse(self: DeepSeekClient, *args: Any, **kwargs: Any) -> Any:
        raise CircuitOpenError(60.0) if failure == "circuit" else BudgetDeniedError("plan", 3)

    monkeypatch.setattr(DeepSeekClient, "chat_json", refuse)
    queue_lifeline(services, rig.planner.ensure(FRIDAY))
    summary = await worker(services).run_until_idle()
    assert summary.deferred == 1 and summary.failed == 0 and summary.retried == 0
    (waiting,) = JobQueue(services.db, services.clock).list_jobs(status="pending")
    assert waiting.attempts == 0


# ------------------------------------------------------------------- lifeline_at


async def test_lifeline_at_tells_what_the_drawn_day_says_she_is_doing(
    rig: Rig, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter, services: Services
) -> None:
    plan = rig.planner.ensure(FRIDAY)
    api.post(API).mock(side_effect=ScriptedLifelineModel())
    await generator(rig, memory, runtime).generate(plan, DAILY)
    found = lifeline_at(services, rig.at(2026, 10, 9, 14), memory=memory)
    assert found is not None and found.activity == "上课和做实验" and found.place == "实验室"
    assert lifeline_at(services, rig.at(2026, 10, 9, 12, 55), memory=memory) is None  # a gap
    rig.move_to(rig.at(2026, 10, 9, 9))
    now = lifeline_at(services, memory=memory)
    assert now is not None and now.activity == "在图书馆看文献"
    assert lifeline_at(services, rig.at(2026, 10, 9, 14) + timedelta(days=1), memory=memory) is None
