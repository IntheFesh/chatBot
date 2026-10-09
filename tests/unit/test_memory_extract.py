"""Facts and follow-ups from a conversation: schema checks, evidence, time zones (R-MEM-006/007)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
import respx

from tests.support.deepseek import API, TEST_KEY, ok
from tests.support.embedding import HashingBackend
from tests.support.memory import (
    CloseRule,
    FactRule,
    FollowupRule,
    ScriptedMemoryModel,
    add_followup,
    make_memory,
    memory_clock,
    utc,
)
from twin.llm.errors import StructuredOutputError
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY
from twin.memory.extract import DialogueLine, FactExtractor
from twin.memory.schemas import (
    ClosedFollowup,
    ExtractedFact,
    ExtractedFollowup,
    ExtractionOut,
)
from twin.services import Services

SAID_AT = datetime(
    2026, 3, 5, 2, 30, tzinfo=UTC
)  # 20:30 on 4 March in Chicago, 10:30 on 5 March in Beijing


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


def lines(*texts: str, start: datetime = SAID_AT) -> list[DialogueLine]:
    """A dialogue of alternating sides, one minute apart (odd lines are hers)."""
    return [
        DialogueLine(
            f"m{number}",
            "her" if number % 2 == 1 else "user",
            text,
            start + timedelta(minutes=number),
        )
        for number, text in enumerate(texts, start=1)
    ]


def fact(**fields: Any) -> dict[str, Any]:
    base = {"subject": "her", "category": "life", "text": "她养了一只猫", "importance": 3}
    base.update(fields)
    return base


def extractor(
    services: Services, runtime: LlmRuntime, zone: str = "America/Chicago"
) -> FactExtractor:
    return FactExtractor(services, runtime.client, memory_clock(services.clock, zone))


# ------------------------------------------------------------------- the schema


def test_a_reply_that_fits_the_schema_is_parsed_and_unknown_fields_are_ignored() -> None:
    out = ExtractionOut.model_validate(
        {
            "facts": [{**fact(), "evidence": ["1", 2], "colour": "x"}],
            "followups": [{"text": "她明天面试", "due": "明天下午三点", "evidence": [1]}],
            "closed_followups": [{"ref": "F1"}],
        }
    )
    assert out.facts[0].evidence == [1, 2] and out.facts[0].confidence == 0.8
    assert out.followups[0].window_minutes is None and out.closed_followups[0].reason == "done"
    assert ExtractionOut.model_validate({}).facts == []


@pytest.mark.parametrize(
    "bad",
    [
        {"facts": [{**fact(subject="dog"), "evidence": [1]}]},
        {"facts": [{**fact(category="mood"), "evidence": [1]}]},
        {"facts": [{**fact(importance=9), "evidence": [1]}]},
        {"facts": [{**fact(), "evidence": []}]},
        {"facts": [{**fact(text="x"), "evidence": [1]}]},
        {"facts": [{**fact(recurrence="daily"), "evidence": [1]}]},
        {"facts": "none"},
        {"followups": [{"text": "x", "evidence": [1]}]},
        {"closed_followups": [{"ref": "F1", "reason": "later"}]},
    ],
)
def test_a_reply_that_does_not_fit_is_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        ExtractionOut.model_validate(bad)


async def test_a_reply_that_does_not_fit_is_sent_back_once_and_two_failures_end_the_task(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    good = {"facts": [{**fact(), "evidence": [1]}], "followups": [], "closed_followups": []}
    route = api.post(API).mock(
        side_effect=[
            ok(content=json.dumps({"facts": [{**fact(subject="dog"), "evidence": [1]}]})),
            ok(content=json.dumps(good, ensure_ascii=False)),
        ]
    )
    result = await extractor(services, runtime).extract(lines("我家的猫"), mode="real", tag=DAILY)
    assert [f.text for f in result.facts] == ["她养了一只猫"] and result.calls == 2
    retry = json.loads(route.calls[1].request.content)["messages"]
    assert "could not be used" in retry[-1]["content"] and "subject" in retry[-1]["content"]

    api.post(API).mock(return_value=ok(content="not json"))
    with pytest.raises(StructuredOutputError):
        await extractor(services, runtime).extract(lines("x"), mode="real", tag=DAILY)


# -------------------------------------------------------------------- evidence


async def test_a_fact_is_known_from_the_latest_line_it_cites(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = ScriptedMemoryModel(
        facts=[FactRule("猫叫豆包", fact(text="她的猫叫豆包"), also=("三岁",))]
    )
    api.post(API).mock(side_effect=model)
    dialogue = lines("早上好", "我家的猫叫豆包", "它三岁了", "吃饭了吗")
    result = await extractor(services, runtime).extract(dialogue, mode="real", tag=DAILY)
    (draft,) = result.facts
    assert draft.evidence_refs == ("m2", "m3") and draft.evidence_kind == "messages"
    assert draft.known_at == dialogue[2].at  # the latest cited line, not the last of the dialogue
    assert (draft.source, draft.subject, draft.confidence) == ("real_record", "her", 0.8)
    assert draft.evidence() == {"kind": "messages", "ids": ["m2", "m3"]}


def test_items_that_cite_missing_lines_or_have_unusable_parts_are_dropped_and_counted(
    services: Services, runtime: LlmRuntime
) -> None:
    ex = extractor(services, runtime)
    dialogue = lines("甲", "乙", "丙")
    out = ExtractionOut(
        facts=[
            ExtractedFact(subject="her", text="她喜欢吃火锅", evidence=[1]),
            ExtractedFact(subject="her", text="引用了不存在的行", evidence=[1, 9]),
            ExtractedFact(subject="her", text="零号行也不存在", evidence=[0]),
        ],
        followups=[ExtractedFollowup(text="她要考试", due="某个时候", evidence=[1])],
        closed_followups=[ClosedFollowup(ref="F7", evidence=[1]), ClosedFollowup(ref="x")],
    )
    result = ex.interpret(out, dialogue, "real")
    assert [f.text for f in result.facts] == ["她喜欢吃火锅"]
    assert result.followups == [] and result.closed == []
    assert result.dropped == 2 + 1 + 2


def test_text_is_tidied_and_the_event_date_is_read_against_the_evidence_time(
    services: Services, runtime: LlmRuntime
) -> None:
    ex = extractor(services, runtime)  # Chicago
    dialogue = lines("下周三是她的考试")
    out = ExtractionOut(
        facts=[
            ExtractedFact(
                subject="her",
                category="anniversary",
                text="  她下周三   考试  ",
                evidence=[1],
                event_date="2031-01-01",  # the model's arithmetic is wrong
                event_phrase="下周三",
                recurrence="none",
            ),
            ExtractedFact(
                subject="both",
                category="anniversary",
                text="她的生日是三月五日",
                evidence=[1],
                event_date="2000-03-05",
                recurrence="yearly",
                valid_from="2026-03-01",
                valid_to="2026-03-31",
            ),
            ExtractedFact(
                subject="her",
                text="她喜欢蓝色",
                evidence=[1],
                recurrence="yearly",
                event_date="soon",
            ),
        ]
    )
    exam, birthday, plain = ex.interpret(out, dialogue, "real").facts
    assert exam.text == "她下周三 考试"
    assert exam.event_date == date(2026, 3, 11)  # the phrase, from Wednesday 4 March in Chicago
    assert (birthday.event_date, birthday.recurrence) == (date(2000, 3, 5), "yearly")
    assert birthday.valid_from == datetime(2026, 3, 1, 6, 0, tzinfo=UTC)  # local midnight, UTC-6
    assert birthday.valid_to is not None and birthday.valid_to < datetime(
        2026, 4, 1, 5, 0, tzinfo=UTC
    )
    assert plain.event_date is None and plain.recurrence == "none"  # a repeat needs a date


# ------------------------------------------------------------------- the source


def test_what_each_side_said_in_the_bot_conversation_decides_the_source(
    services: Services, runtime: LlmRuntime
) -> None:
    ex = extractor(services, runtime)
    dialogue = [
        DialogueLine("t1", "user", "我下周要考试", utc(2026, 3, 5)),
        DialogueLine("t2", "bot", "我今天去了图书馆", utc(2026, 3, 5, 12, 1)),
        DialogueLine("t3", "bot", "你上次说你在读研", utc(2026, 3, 5, 12, 2)),
    ]
    out = ExtractionOut(
        facts=[
            ExtractedFact(speaker="user", subject="user", text="对方下周要考试", evidence=[1]),
            ExtractedFact(
                speaker="bot",
                subject="her",
                category="life",
                text="她今天去了图书馆",
                evidence=[2],
                lifeline={
                    "activity": "去图书馆",
                    "place": "学校图书馆",
                    "start": "14:00",
                    "end": "9:99",
                },  # type: ignore[arg-type]
            ),
            ExtractedFact(speaker="bot", subject="user", text="对方在读研", evidence=[3]),
            ExtractedFact(speaker="bot", subject="other", text="她的室友很吵", evidence=[2]),
            ExtractedFact(speaker=None, subject="her", text="没写是谁说的", evidence=[1]),
            ExtractedFact(speaker="her", subject="her", text="她说了她自己", evidence=[1]),
        ]
    )
    result = ex.interpret(out, dialogue, "bot")
    assert [(f.source, f.evidence_kind) for f in result.facts] == [
        ("user_said", "bot_turns"),
        ("bot_invented", "bot_turns"),
    ]
    hint = result.facts[1].lifeline
    assert hint is not None and (hint.activity, hint.place, hint.start, hint.end) == (
        "去图书馆",
        "学校图书馆",
        "14:00",
        None,  # an unreadable time is left out; the fact itself stays
    )
    assert result.dropped == 4  # the echo about the user, the third party, no speaker, "her"
    assert result.facts[0].lifeline is None


def test_the_user_command_is_the_most_certain_source(
    services: Services, runtime: LlmRuntime
) -> None:
    ex = extractor(services, runtime)
    command = [DialogueLine("cmd-1", "user", "她的生日是三月五日", utc(2026, 3, 1))]
    out = ExtractionOut(
        facts=[
            ExtractedFact(
                subject="her",
                category="anniversary",
                text="她的生日是三月五日",
                confidence=0.4,
                evidence=[1],
                event_date="2000-03-05",
                recurrence="yearly",
            )
        ]
    )
    (draft,) = ex.interpret(out, command, "command").facts
    assert (draft.source, draft.confidence, draft.evidence_kind) == ("user_command", 1.0, "command")


# ------------------------------------------------------------------ follow-ups


async def test_the_model_is_told_the_local_time_and_tomorrow_differs_between_chicago_and_beijing(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    """The same instant, the same words: the zone of the conversation decides (R-MEM-007)."""
    model = ScriptedMemoryModel(
        followups=[
            FollowupRule(
                "考试",
                {
                    "text": "她明天下午考试",
                    "due": "明天下午三点",
                    "due_local": "2099-01-01 00:00",  # the model's own arithmetic is not trusted
                    "window_minutes": 90,
                },
            )
        ]
    )
    api.post(API).mock(side_effect=model)
    dialogue = [DialogueLine("m1", "user", "明天下午三点考试", SAID_AT)]
    chicago = await extractor(services, runtime, "America/Chicago").extract(
        dialogue, mode="real", tag=DAILY
    )
    beijing = await extractor(services, runtime, "Asia/Shanghai").extract(
        dialogue, mode="real", tag=DAILY
    )
    prompts = [req["messages"][1]["content"] for req in model.requests]
    assert "对话发生的时区：America/Chicago" in prompts[0]
    assert "对话开始的当地时间：2026-03-04 20:30（周三）" in prompts[0]
    assert "对话发生的时区：Asia/Shanghai" in prompts[1]
    assert "对话开始的当地时间：2026-03-05 10:30（周四）" in prompts[1]
    assert "1. [03-04 20:30] 对方：明天下午三点考试" in prompts[0]
    assert "1. [03-05 10:30] 对方：明天下午三点考试" in prompts[1]
    (due_chicago,) = chicago.followups
    (due_beijing,) = beijing.followups
    assert due_chicago.due_at == datetime(2026, 3, 5, 21, 0, tzinfo=UTC)  # 15:00 CST
    assert due_beijing.due_at == datetime(2026, 3, 6, 7, 0, tzinfo=UTC)  # 15:00 in Beijing
    assert due_beijing.due_at - due_chicago.due_at == timedelta(hours=10)
    assert (due_chicago.window_minutes, due_chicago.created_at) == (90, SAID_AT)
    assert due_chicago.origin == "real_record" and due_chicago.evidence_refs == ("m1",)


def test_follow_up_times_fall_back_to_the_models_local_time_and_bad_ones_are_dropped(
    services: Services, runtime: LlmRuntime
) -> None:
    ex = extractor(services, runtime)
    dialogue = lines("甲", "乙")

    def follow(**fields: Any) -> ExtractionOut:
        return ExtractionOut(
            followups=[ExtractedFollowup(text="她要去面试", evidence=[1], **fields)]
        )

    exact = ex.interpret(follow(due="考完试以后", due_local="2026-03-05 15:00"), dialogue, "real")
    assert exact.followups[0].due_at == datetime(2026, 3, 5, 21, 0, tzinfo=UTC)
    dated = ex.interpret(
        follow(due="", due_local="2026-03-06", window_minutes=60), dialogue, "real"
    )
    only_date = dated.followups[0]
    assert only_date.due_at == datetime(2026, 3, 6, 15, 0, tzinfo=UTC)  # 09:00 on that day
    assert only_date.window_minutes == 720  # a whole day: it was said with no time
    clamped = ex.interpret(follow(due="明天上午9点", window_minutes=99999), dialogue, "real")
    assert clamped.followups[0].window_minutes == 2880
    short = ex.interpret(follow(due="明天上午9点", window_minutes=1), dialogue, "real")
    assert short.followups[0].window_minutes == 30
    default = ex.interpret(follow(due="明天上午9点"), dialogue, "real")
    assert default.followups[0].window_minutes == 240
    unreadable = ex.interpret(follow(due="改天", due_local="whenever"), dialogue, "real")
    assert unreadable.followups == [] and unreadable.dropped == 1
    past = ex.interpret(follow(due="昨天下午三点"), dialogue, "real")
    assert past.followups == [] and past.dropped == 1  # already over when it was said


def test_a_bot_follow_up_remembers_the_turn_it_came_from(
    services: Services, runtime: LlmRuntime
) -> None:
    ex = extractor(services, runtime)
    dialogue = [
        DialogueLine("turn-1", "user", "明天上午九点我要面试", utc(2026, 3, 5, 14)),
        DialogueLine("turn-2", "bot", "加油", utc(2026, 3, 5, 14, 1)),
    ]
    out = ExtractionOut(followups=[ExtractedFollowup(text="面试", due="明天上午9点", evidence=[1])])
    (follow,) = ex.interpret(out, dialogue, "bot").followups
    assert (follow.origin, follow.source_turn_id) == ("bot_session", "turn-1")


async def test_open_follow_ups_are_shown_and_the_ones_the_conversation_closes_are_returned(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    open_one = add_followup(memory, "她周五面试", utc(2026, 3, 6, 21), utc(2026, 3, 4))
    other = add_followup(memory, "她周日看牙医", utc(2026, 3, 8, 21), utc(2026, 3, 4))
    model = ScriptedMemoryModel(closes=[CloseRule("面试怎么样", "她周五面试")])
    api.post(API).mock(side_effect=model)
    dialogue = lines("面试怎么样", "还不错")
    result = await extractor(services, runtime).extract(
        dialogue, mode="real", tag=DAILY, open_followups=[open_one, other]
    )
    prompt = model.requests[0]["messages"][1]["content"]
    assert "F1：她周五面试（预定 2026-03-06 15:00）" in prompt and "F2：她周日看牙医" in prompt
    (closed,) = result.closed
    assert (closed.followup_id, closed.reason, closed.at) == (open_one.id, "done", dialogue[0].at)


# ------------------------------------------------------------- size and switches


async def test_a_long_conversation_is_cut_into_chunks_that_keep_their_own_line_numbers(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    services.settings.memory.replay_chunk_lines = 20
    model = ScriptedMemoryModel(facts=[FactRule("第四十五句", fact(text="她在第四十五句说了养猫"))])
    api.post(API).mock(side_effect=model)
    texts = [f"第{n}句" if n != 45 else "第四十五句" for n in range(1, 46)]
    dialogue = lines(*texts)
    result = await extractor(services, runtime).extract(dialogue, mode="real", tag=DAILY)
    assert model.calls["extract"] == 3  # 20 + 20 + 5 lines
    (draft,) = result.facts
    assert draft.evidence_refs == ("m45",)  # line 5 of the last chunk is message 45
    assert "对话（共 5 行）" in model.requests[2]["messages"][1]["content"]


async def test_with_fact_extraction_off_nothing_is_asked(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    services.settings.memory.fact_extraction = False
    route = api.post(API).mock(side_effect=ScriptedMemoryModel())
    ex = extractor(services, runtime)
    assert not ex.enabled
    result = await ex.extract(lines("甲"), mode="real", tag=DAILY)
    assert result.facts == [] and route.call_count == 0
    services.settings.memory.fact_extraction = True
    assert (await ex.extract([], mode="real", tag=DAILY)).facts == []
    assert route.call_count == 0  # no lines: nothing to ask either
