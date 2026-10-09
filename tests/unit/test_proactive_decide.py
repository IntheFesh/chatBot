"""The two-step decision: what the planner is asked, what it may answer, what is made of it."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from tests.support.proactive_world import World
from twin.config.runtime import THINKING_PROACTIVE
from twin.engine.backend import BackendRequest
from twin.engine.backend_select import BackendChoice
from twin.engine.style_backend import StyleWrite
from twin.engine.style_prompt import PlanFields
from twin.engine.types import Bubble, UsageSummary
from twin.retrieval.examples import Example, ExampleLine
from twin.schedule.proactive.decide import (
    CUE_TEXT,
    Brief,
    Draft,
    ProactiveDecider,
    ProactivePlan,
    repeats_previous,
)
from twin.schedule.proactive.types import Candidate, Reason, TriggerKind


def decider_of(world: World) -> ProactiveDecider:
    return world.scheduler._decider  # type: ignore[return-value,attr-defined]


def candidate(world: World, kind: TriggerKind, **detail: Any) -> Candidate:
    planned = world.at(10, 0)
    return Candidate(kind, f"{kind.value}:t", planned, planned + timedelta(hours=1), detail=detail)


async def decide(
    world: World,
    kind: TriggerKind,
    *,
    unanswered: int = 0,
    bubbles: int = 8,
    previous: tuple[str, ...] = (),
    **d: Any,
) -> Draft:
    world.go_to(world.at(10, 5))
    brief = Brief(
        now=world.clock.now_utc(), unanswered=unanswered, max_bubbles=bubbles, previous=previous
    )
    return await decider_of(world).decide(candidate(world, kind, **d), brief)


def last_request(world: World) -> dict[str, Any]:
    return world.script.requests[-1]


def context_of(world: World) -> str:
    return str(last_request(world)["messages"][-1]["content"])


@pytest.fixture
def started(calm: World) -> World:
    calm.user_writes(at=calm.at(9, 0))
    return calm


# ------------------------------------------------------------------------- the plan JSON


def test_a_plan_repairs_what_a_model_writes_loosely() -> None:
    plan = ProactivePlan.model_validate(
        {
            "send": True,
            "messages": "在吗",
            "reason": None,
            "kind": " meal ",
            "facts_to_use": [" a ", "", None, "b"],
            "new_detail": {"activity": "  "},
            "future_field": 1,
        }
    )
    assert plan.messages == ["在吗"] and plan.reason == "" and plan.kind == "meal"
    assert plan.facts_to_use == ["a", "b"] and plan.new_detail is None
    assert plan.raw_text() == "在吗"


def test_a_plan_with_a_sticker_and_no_words_is_a_plan() -> None:
    plan = ProactivePlan.model_validate({"send": True, "sticker_hint": "笑"})
    assert plan.raw_text() == "[表情包:笑]"
    assert plan.fields().sticker_hint == "笑"


def test_a_plan_that_sends_nothing_is_not_a_plan_to_send() -> None:
    with pytest.raises(ValueError, match="no message"):
        ProactivePlan.model_validate({"send": True, "messages": []})
    declined = ProactivePlan.model_validate({"send": False})
    assert declined.messages == [] and not declined.send


def test_the_plan_fields_carry_what_the_style_model_needs() -> None:
    plan = ProactivePlan.model_validate(
        {
            "send": True,
            "messages": ["a", "b"],
            "intent": "找他聊聊",
            "facts_to_use": ["1", "2", "3", "4", "5", "6"],
            "tone": "随意",
        }
    )
    fields = plan.fields()
    assert isinstance(fields, PlanFields) and fields.intent == "找他聊聊"
    assert len(fields.facts_to_use) == 4 and fields.bubble_hint == "2条"


# ------------------------------------------------------------------- what the planner is told


@pytest.mark.parametrize(
    ("kind", "detail", "expected"),
    [
        (TriggerKind.GREETING, {}, "greeting（起床问候）：她刚起床不久"),
        (TriggerKind.MEAL, {"meal": "lunch"}, "meal（饭点）：快到她午饭的时候了"),
        (TriggerKind.BEDTIME, {}, "bedtime（睡前）：她快要睡觉了"),
        (TriggerKind.FOLLOWUP, {}, "followup（跟进）"),
        (TriggerKind.SILENCE, {}, "silence（沉默）：你们有一阵没聊了"),
        (TriggerKind.SHARE, {}, "share（分享）：想把今天的一件小事讲给对方听"),
        (TriggerKind.EDGE, {"side": "falling"}, "edge（睡不着/刚醒）：她快要睡着，睡不着"),
        (TriggerKind.EDGE, {"side": "waking"}, "edge（睡不着/刚醒）：她刚醒"),
    ],
)
async def test_the_planner_is_told_why_a_message_is_being_considered(
    started: World, kind: TriggerKind, detail: dict[str, Any], expected: str
) -> None:
    draft = await decide(started, kind, **detail)
    assert draft.usable, draft
    assert expected in context_of(started)


async def test_the_planner_reads_the_moment_her_day_and_the_conversation(started: World) -> None:
    from twin.memory.lifeline import PlannedEvent

    started.lifeline.replace_plan(
        started.rig.kit.time.local_date(started.at(10, 0)),
        [PlannedEvent("在图书馆看文献", "09:00", "11:00", "图书馆")],
    )
    await decide(started, TriggerKind.SHARE)
    request = last_request(started)
    text = context_of(started)
    assert "【此刻】" in text and "当地时间：" in text and "她现在的状态：" in text
    assert "L1 " in text and "在图书馆看文献" in text
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0]["role"] == "system"
    assert "$" not in request["messages"][0]["content"]  # every field of the template was filled
    roles = [m["role"] for m in request["messages"]]
    assert roles[-1] == "user" and "在吗" in " ".join(m["content"] for m in request["messages"])


async def test_the_planner_sees_the_followup_it_is_asked_to_follow(started: World) -> None:
    record, _ = started.followups.add(
        "周五的考试", started.at(9, 0), created_at=started.at(20, 0, day=8)
    )
    draft = await decide(started, TriggerKind.FOLLOWUP, followup_id=record.id)
    assert draft.usable
    assert "【要跟进的事】\n周五的考试" in context_of(started)


async def test_a_chase_is_told_what_she_said_last_and_must_say_something_else(
    started: World,
) -> None:
    started.script.messages["silence"] = ("在干嘛呀",)
    draft = await decide(started, TriggerKind.SILENCE, unanswered=1, previous=("在干嘛呀",))
    # the planner said the same thing twice: it is asked again, then the slot is given up
    assert draft.failure is Reason.GENERATION_FAILED and draft.failure_detail == "repeat_previous"
    assert len(started.script.requests) == 2
    again = context_of(started)
    assert "【上一条主动消息没有回" in again and "- 在干嘛呀" in again
    assert "几乎一样" in again  # the second call carries what was wrong


async def test_a_chase_that_says_something_else_goes_out(started: World) -> None:
    started.script.messages["silence"] = ("你怎么不理我", "哼")
    draft = await decide(started, TriggerKind.SILENCE, unanswered=1, previous=("在干嘛呀",))
    assert draft.usable and [b.text for b in draft.bubbles] == ["你怎么不理我", "哼"]


def test_what_counts_as_the_same_message_again() -> None:
    def said(*texts: str) -> list[Bubble]:
        return [Bubble("text", t) for t in texts]

    assert repeats_previous(said("在干嘛呀"), ["在干嘛呀"])
    assert repeats_previous(said("在干嘛呀!!"), ["在干嘛呀"])
    assert repeats_previous(said("在干嘛", "呀"), ["在干嘛 呀"])
    assert repeats_previous(said("你在干嘛呀"), ["你在干嘛呀？"])
    assert not repeats_previous(said("你怎么不理我"), ["在干嘛呀"])
    assert not repeats_previous(said("在干嘛呀"), [])
    assert not repeats_previous([Bubble("sticker", "[表情包:笑]")], ["在干嘛呀"])


# ------------------------------------------------------------------------- thinking


async def test_the_planner_thinks_by_default_and_the_setting_turns_it_off(started: World) -> None:
    await decide(started, TriggerKind.SHARE)
    assert last_request(started)["thinking"] == {"type": "enabled"}
    started.services.runtime.set(THINKING_PROACTIVE, "off", by="test")
    await decide(started, TriggerKind.SHARE)
    assert last_request(started)["thinking"] == {"type": "disabled"}
    assert last_request(started)["temperature"] == pytest.approx(0.8)


async def test_auto_thinks_for_a_followup_and_for_silence_only(started: World) -> None:
    started.services.runtime.set(THINKING_PROACTIVE, "auto", by="test")
    seen: dict[str, str] = {}
    for kind in (TriggerKind.FOLLOWUP, TriggerKind.SILENCE, TriggerKind.SHARE, TriggerKind.MEAL):
        await decide(started, kind)
        seen[kind.value] = last_request(started)["thinking"]["type"]
    assert seen == {
        "followup": "enabled",
        "silence": "enabled",
        "share": "disabled",
        "meal": "disabled",
    }


async def test_a_tight_budget_takes_the_planners_thinking_away(started: World) -> None:
    decider_of(started)._thinking_allowed = lambda: False  # type: ignore[attr-defined]
    await decide(started, TriggerKind.FOLLOWUP)
    assert last_request(started)["thinking"] == {"type": "disabled"}


# ------------------------------------------------------------------------- the answers


async def test_a_plan_that_says_no_ends_the_decision_with_its_reason(started: World) -> None:
    started.script.decline = {"share"}
    draft = await decide(started, TriggerKind.SHARE)
    assert not draft.send and not draft.usable and draft.failure is None
    assert draft.reason == "现在发不合适" and draft.plan is not None
    assert len(started.script.requests) == 1


async def test_an_answer_that_is_not_json_fails_the_plan(started: World) -> None:
    started.script.invalid_first = 9
    draft = await decide(started, TriggerKind.SHARE)
    assert draft.failure is Reason.PLANNER_FAILED and draft.failure_detail == "invalid_json"


@pytest.mark.parametrize(
    ("message", "detail"),
    [("Content Exists Risk", "content_risk"), ("messages are malformed", "invalid_request")],
)
async def test_a_request_the_api_refuses_fails_the_plan_and_says_why(
    started: World, message: str, detail: str
) -> None:
    started.script.refuse_with = message
    draft = await decide(started, TriggerKind.SHARE)
    assert draft.failure is Reason.PLANNER_FAILED and draft.failure_detail == detail


@pytest.mark.parametrize(
    ("write_at", "expected"),
    [
        ((8, 5), "已经有 2 小时 没说话了"),
        ((9, 0), "已经有 1 小时 5 分钟 没说话了"),
        ((9, 55), "已经有 10 分钟 没说话了"),
    ],
)
async def test_how_long_the_silence_has_lasted_is_told_in_hours_and_minutes(
    calm: World, write_at: tuple[int, int], expected: str
) -> None:
    hour, minute = write_at
    calm.user_writes(at=calm.at(hour, minute) - timedelta(seconds=30))
    await decide(calm, TriggerKind.SILENCE)
    assert expected in context_of(calm)


async def test_a_silence_of_days_is_told_in_days(calm: World) -> None:
    calm.user_writes(at=calm.at(9, 0, day=7))
    await decide(calm, TriggerKind.SILENCE)
    assert "已经有 2 天多 没说话了" in context_of(calm)


class Openings:
    """The examples of her openings, as the decider asks for them."""

    def __init__(self, found: list[Example] | None = None, *, fail: bool = False) -> None:
        self.found = found or []
        self.fail = fail
        self.asked: list[dict[str, Any]] = []

    async def query(self, **kwargs: Any) -> list[Example]:
        self.asked.append(kwargs)
        if self.fail:
            raise RuntimeError("the library is being rebuilt")
        return self.found


def wire_openings(world: World, openings: Openings) -> None:
    material = decider_of(world)._material  # type: ignore[attr-defined]
    material._openers = openings
    material._examples_k = lambda: 3


async def test_her_real_openings_around_this_hour_are_shown_as_examples_to_imitate(
    started: World,
) -> None:
    opening = Example(
        window_id="w1",
        reply_at=started.at(9, 40),
        local_slot=38,
        day_type="workday",
        context=(),
        reply=(
            ExampleLine("早啊", "text", True, None),
            ExampleLine("今天好冷", "text", True, None),
        ),
    )
    openings = Openings([opening])
    wire_openings(started, openings)
    draft = await decide(started, TriggerKind.GREETING)
    assert draft.usable
    seen = context_of(started)
    assert "她过去自己先开口时是怎么说的" in seen and "早啊" in seen
    assert openings.asked[0]["k"] == 3 and openings.asked[0]["day_type"] == "workday"
    assert openings.asked[0]["local_minute"] == pytest.approx(605.0, abs=1.0)


async def test_a_library_that_cannot_be_read_costs_the_examples_not_the_message(
    started: World,
) -> None:
    wire_openings(started, Openings(fail=True))
    draft = await decide(started, TriggerKind.GREETING)
    assert draft.usable and "她过去自己先开口时是怎么说的" not in context_of(started)


async def test_a_plan_to_send_without_anything_to_send_fails_the_plan(started: World) -> None:
    started.script.by_hand = lambda kind, body: {"send": True, "messages": []}
    draft = await decide(started, TriggerKind.SHARE)
    assert draft.failure is Reason.PLANNER_FAILED


async def test_an_account_without_balance_fails_the_plan_without_a_message(started: World) -> None:
    started.script.unavailable = True
    draft = await decide(started, TriggerKind.SHARE)
    assert draft.failure is Reason.PLANNER_FAILED and not draft.usable


async def test_a_filtered_answer_fails_the_plan(started: World) -> None:
    started.script.finish = "content_filter"
    draft = await decide(started, TriggerKind.SHARE)
    assert draft.failure is Reason.PLANNER_FAILED and draft.failure_detail == "content_filter"


# --------------------------------------------------------------------- post-processing


async def test_a_promise_in_the_message_is_asked_again_and_the_second_answer_is_used(
    started: World,
) -> None:
    answers = iter(
        [
            {"send": True, "kind": "share", "messages": ["我明天给你打电话"], "reason": "a"},
            {"send": True, "kind": "share", "messages": ["今天好困"], "reason": "b"},
        ]
    )
    started.script.by_hand = lambda kind, body: next(answers)
    draft = await decide(started, TriggerKind.SHARE)
    assert draft.usable and draft.attempts == 2
    assert [b.text for b in draft.bubbles] == ["今天好困"]
    assert "不要答应打电话" in context_of(started)  # the second call says what was wrong


async def test_a_promise_that_is_asked_for_twice_is_cut_out_of_the_last_try(started: World) -> None:
    started.script.by_hand = lambda kind, body: {
        "send": True,
        "kind": kind,
        "messages": ["我明天给你打电话", "先睡啦"],
        "reason": "x",
    }
    draft = await decide(started, TriggerKind.BEDTIME)
    assert draft.usable and draft.attempts == 2
    assert [b.text for b in draft.bubbles] == ["先睡啦"]
    assert any(a.step == "commitment_removed" for a in draft.actions)


async def test_a_message_that_is_nothing_but_a_promise_is_given_up(started: World) -> None:
    started.script.by_hand = lambda kind, body: {
        "send": True,
        "kind": kind,
        "messages": ["我明天给你打电话"],
        "reason": "x",
    }
    draft = await decide(started, TriggerKind.SHARE)
    assert not draft.usable and draft.failure is Reason.GENERATION_FAILED
    assert len(started.script.requests) == 2  # one more try, then nothing: no fallback line
    assert draft.attempts == 2


@pytest.mark.parametrize(
    ("text", "kind"),
    [("作为一个AI助手我不能", "ai_self_reference"), ("[图片]", "event_text_only")],
)
async def test_a_text_that_breaks_a_hard_rule_is_never_sent(
    started: World, text: str, kind: str
) -> None:
    started.script.by_hand = lambda k, body: {"send": True, "kind": k, "messages": [text]}
    draft = await decide(started, TriggerKind.SHARE)
    assert draft.failure is Reason.GENERATION_FAILED
    assert kind in (draft.failure_detail or "")


async def test_the_bubbles_are_cut_to_what_the_platform_has_left(started: World) -> None:
    started.script.messages["share"] = ("一二", "三四", "五六", "七八")
    draft = await decide(started, TriggerKind.SHARE, bubbles=2)
    assert draft.usable and len(draft.bubbles) <= 2
    joined = "".join(b.text for b in draft.bubbles)
    assert joined.startswith("一二")


# ------------------------------------------------------------------------ the style model


class StyleDouble:
    """A style model that writes a fixed text and keeps what it was asked."""

    def __init__(self, text: str, *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[tuple[BackendRequest, PlanFields | None]] = []

    async def write(self, request: BackendRequest, plan: PlanFields | None = None) -> StyleWrite:
        self.calls.append((request, plan))
        if self.fail:
            raise RuntimeError("the style model is down")
        return StyleWrite(self.text, UsageSummary(), 12, {"model": "style-v3"})


class Backends:
    def __init__(self, name: str) -> None:
        self.name = name

    async def choose(self) -> BackendChoice:
        return BackendChoice(self.name, self.name)


@pytest.mark.parametrize("backend", ["style", "hybrid"])
async def test_with_the_style_model_the_plan_goes_into_its_prompt_and_it_writes(
    started: World, backend: str
) -> None:
    decider = decider_of(started)
    double = StyleDouble("早啊\n刚醒")
    decider._style = double  # type: ignore[assignment,attr-defined]
    decider._backends = Backends(backend)  # type: ignore[attr-defined]
    draft = await decide(started, TriggerKind.GREETING, bubbles=5)
    assert draft.usable and draft.backend == backend
    assert [b.text for b in draft.bubbles] == ["早啊", "刚醒"]
    request, plan = double.calls[0]
    assert request.context.inbound[0].text == CUE_TEXT
    assert request.context.woke_up is True and request.context.limits.max_bubbles == 5
    assert plan is not None and plan.intent == "找他聊聊"
    assert draft.meta == {"model": "style-v3"}
    assert len(started.script.requests) == 1  # DeepSeek only planned


async def test_a_style_model_that_fails_costs_nothing_the_planners_messages_are_used(
    started: World,
) -> None:
    decider = decider_of(started)
    decider._style = StyleDouble("", fail=True)  # type: ignore[assignment,attr-defined]
    decider._backends = Backends("hybrid")  # type: ignore[attr-defined]
    draft = await decide(started, TriggerKind.MEAL, meal="lunch")
    assert draft.usable and draft.backend == "deepseek"
    assert [b.text for b in draft.bubbles] == ["吃饭了吗"] and draft.meta == {"style_failed": True}


async def test_the_deepseek_backend_never_calls_the_style_model(started: World) -> None:
    decider = decider_of(started)
    double = StyleDouble("不该出现")
    decider._style = double  # type: ignore[assignment,attr-defined]
    decider._backends = Backends("deepseek")  # type: ignore[attr-defined]
    draft = await decide(started, TriggerKind.MEAL, meal="lunch")
    assert draft.usable and double.calls == []


async def test_a_selector_that_breaks_does_not_break_the_message(started: World) -> None:
    class Broken:
        async def choose(self) -> BackendChoice:
            raise RuntimeError("no selector")

    decider = decider_of(started)
    decider._style = StyleDouble("x")  # type: ignore[assignment,attr-defined]
    decider._backends = Broken()  # type: ignore[attr-defined]
    draft = await decide(started, TriggerKind.MEAL, meal="lunch")
    assert draft.usable and draft.backend == "deepseek"
