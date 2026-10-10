"""The memory's jobs and the day loaders: summaries, extraction from the bot's conversation
(R-MEM-002, R-MEM-007, R-MEM-008) and the post-import replay hook (R-IMP-011, R-LLM-014)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import respx
from sqlalchemy import select

from tests.support.bot_turns import ListBotTurnReader, conversation
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
from twin.ingest.hooks import HookContext, load_hooks
from twin.ingest.transcript import message_text
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.memory.dayload import (
    LineHasher,
    bot_day_lines,
    day_lines,
    lines_hash,
    real_day_lines,
)
from twin.memory.extract import DialogueLine
from twin.memory.hook import describe_plan, queue_memory_replay
from twin.memory.jobs import (
    EXTRACT_JOB,
    SUMMARY_JOB,
    handle_memory_extract,
    handle_memory_replay,
    handle_memory_summary,
    is_quiet,
    queue_bot_extraction,
    queue_daily_summary,
)
from twin.memory.memory import Memory
from twin.memory.recent import BotMessage, register_bot_turn_reader, use_bot_turn_reader
from twin.memory.replay import REPLAY_JOB, plan_replay
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.services import Services
from twin.storage.chat_models import Message

CHICAGO = ZoneInfo("America/Chicago")
DAY = date(2026, 3, 5)
FIRST = datetime(2026, 3, 5, 15, 0, tzinfo=UTC)  # 09:00 in Chicago


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def memory(services: Services, embedder: HashingBackend) -> Memory:
    services.settings.memory.recall_min_similarity = 0.3
    return make_memory(services)


@pytest.fixture
async def runtime(services: Services, embedder: HashingBackend) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


@pytest.fixture(autouse=True)
def no_bot_reader() -> Iterator[None]:
    """No conversation of the bot while a test starts; the application's reader comes back after."""
    with use_bot_turn_reader(None):
        yield


def worker_for(services: Services, **handlers: Any) -> Worker:
    registry = HandlerRegistry()
    for name, handler in handlers.items():
        registry.register(name, handler)
    return Worker(
        JobQueue(services.db, services.clock),
        registry,
        services.clock,
        services=services,
        offpeak=AlwaysOffPeak(),
        alerts=services.alerts,
    )


def jobs_of(services: Services, job_type: str, status: str | None = None) -> list[Any]:
    return JobQueue(services.db, services.clock).list_jobs(
        status=status, job_type=job_type, limit=100
    )


def write_messages(
    services: Services, rows: list[tuple[datetime, bool, str]], *, append: bool = False
) -> None:
    writer = MessageWriter(services)
    for at, her, text in rows:
        writer.add(at, her, "text", text)
    writer.store(append=append)


# --------------------------------------------------------------------- queueing


def test_summaries_are_queued_off_peak_once_per_day_and_scope(services: Services) -> None:
    ids = queue_daily_summary(services, DAY)
    assert len(ids) == 2
    jobs = jobs_of(services, SUMMARY_JOB)
    assert {(j.payload["date"], j.payload["scope"]) for j in jobs} == {
        ("2026-03-05", "real"),
        ("2026-03-05", "bot"),
    }
    assert all(j.offpeak_only and j.payload["force"] is False for j in jobs)
    assert queue_daily_summary(services, DAY) == []  # waiting already
    assert len(queue_daily_summary(services, DAY, ("real",), force=True)) == 0
    assert len(queue_daily_summary(services, date(2026, 3, 6), ("real",), force=True)) == 1
    forced = [
        j for j in jobs_of(services, SUMMARY_JOB, "pending") if j.payload["date"] == "2026-03-06"
    ]
    assert [j.payload["force"] for j in forced] == [True]
    deadline = FIRST + timedelta(days=1)
    (late,) = queue_daily_summary(services, date(2026, 3, 7), ("bot",), deadline=deadline)
    assert late
    with pytest.raises(ValueError, match="scope"):
        queue_daily_summary(services, DAY, ("both",))


def test_bot_extraction_is_queued_with_the_turns_inside_the_job(services: Services) -> None:
    turns = list(conversation(6, start=FIRST).messages_since(None))
    job_id = queue_bot_extraction(services, turns)
    assert job_id is not None
    (job,) = jobs_of(services, EXTRACT_JOB)
    assert job.id == job_id and len(job.payload["turns"]) == 6
    assert job.payload["turns"][0] == {
        "id": "b1",
        "role": "user",
        "text": "第1轮第1句",
        "at": FIRST.isoformat(),
    }
    assert queue_bot_extraction(services, []) is None
    services.settings.memory.fact_extraction = False
    assert queue_bot_extraction(services, turns) is None
    assert len(jobs_of(services, EXTRACT_JOB)) == 1


def test_a_conversation_is_quiet_after_the_configured_minutes() -> None:
    last = FIRST
    assert not is_quiet(last, last + timedelta(minutes=29, seconds=59), 30)
    assert is_quiet(last, last + timedelta(minutes=30), 30)
    assert is_quiet(last, last + timedelta(hours=5), 30)


# ----------------------------------------------------------------- summary job


async def test_the_summary_job_writes_the_summary_of_a_real_day(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    write_messages(
        services,
        [(FIRST, False, "今天去考试"), (FIRST + timedelta(minutes=1), True, "加油")],
    )
    model = ScriptedMemoryModel(summaries={"2026-03-05": "对方去考试，她鼓励了对方。"})
    api.post(API).mock(side_effect=model)
    (job_id,) = queue_daily_summary(services, DAY, ("real",))
    summary = await worker_for(services, **{SUMMARY_JOB: handle_memory_summary}).run_until_idle()
    assert summary.done == 1 and summary.failed == 0
    stored = memory.store.current_summary("real", DAY)
    assert stored is not None and stored.text == "对方去考试，她鼓励了对方。"
    assert job_id and model.calls["summary"] == 1
    again = queue_daily_summary(services, DAY, ("real",))  # an unchanged day costs nothing
    await worker_for(services, **{SUMMARY_JOB: handle_memory_summary}).run_until_idle()
    assert again and model.calls["summary"] == 1
    assert len(memory.store.summary_versions("real", DAY)) == 1


async def test_the_bot_summary_waits_for_round_09_and_then_reads_its_conversation(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = ScriptedMemoryModel()
    api.post(API).mock(side_effect=model)
    queue_daily_summary(services, DAY, ("bot",))
    worker = worker_for(services, **{SUMMARY_JOB: handle_memory_summary})
    assert (await worker.run_until_idle()).done == 1  # no reader: nothing to summarise
    assert model.calls["summary"] == 0 and memory.store.current_summary("bot", DAY) is None

    reader = conversation(4, start=FIRST)
    register_bot_turn_reader(lambda _services: reader)
    queue_daily_summary(services, DAY, ("bot",))
    assert (await worker.run_until_idle()).done == 1
    stored = memory.store.current_summary("bot", DAY)
    assert stored is not None and model.calls["summary"] == 1
    assert memory.store.bot_online_at() == FIRST


async def test_a_day_without_messages_writes_nothing(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = ScriptedMemoryModel()
    api.post(API).mock(side_effect=model)
    queue_daily_summary(services, DAY, ("real",))
    summary = await worker_for(services, **{SUMMARY_JOB: handle_memory_summary}).run_until_idle()
    assert summary.done == 1 and model.requests == []
    assert memory.store.current_summary("real", DAY) is None


@pytest.mark.parametrize("failure", ["circuit", "budget"])
@pytest.mark.parametrize("job", ["summary", "extract", "replay"])
async def test_an_open_circuit_or_a_denied_budget_hands_the_job_back(
    services: Services,
    memory: Memory,
    runtime: LlmRuntime,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    job: str,
) -> None:
    async def refuse(self: DeepSeekClient, *args: Any, **kwargs: Any) -> Any:
        raise CircuitOpenError(60.0) if failure == "circuit" else BudgetDeniedError("memory", 3)

    monkeypatch.setattr(DeepSeekClient, "chat_json", refuse)
    write_messages(
        services, [(FIRST, False, "我下周要考试"), (FIRST + timedelta(minutes=1), True, "加油")]
    )
    handlers: dict[str, Any]
    if job == "summary":
        queue_daily_summary(services, DAY, ("real",))
        handlers, kind = {SUMMARY_JOB: handle_memory_summary}, SUMMARY_JOB
    elif job == "extract":
        queue_bot_extraction(services, list(conversation(2, start=FIRST).messages_since(None)))
        handlers, kind = {EXTRACT_JOB: handle_memory_extract}, EXTRACT_JOB
    else:
        plan = plan_replay(services, runtime=runtime)
        runtime.batches.approve(plan.batches[0].batch_id)
        handlers, kind = {REPLAY_JOB: handle_memory_replay}, REPLAY_JOB
    summary = await worker_for(services, **handlers).run_until_idle()
    assert summary.deferred == 1 and summary.failed == 0 and summary.retried == 0
    (waiting,) = jobs_of(services, kind, "pending")
    assert waiting.attempts == 0  # a hand-back costs no attempt


# --------------------------------------------------------------- extraction job


def bot_script() -> ScriptedMemoryModel:
    return ScriptedMemoryModel(
        facts=[
            FactRule(
                "我养了一只狗",
                {
                    "speaker": "user",
                    "subject": "user",
                    "category": "life",
                    "text": "对方养了一只狗",
                },
            ),
            FactRule(
                "我今天去爬山了",
                {
                    "speaker": "bot",
                    "subject": "her",
                    "category": "life",
                    "text": "她今天去爬山了",
                    "lifeline": {
                        "activity": "爬山",
                        "place": "后山",
                        "mood": "开心",
                        "start": "09:00",
                        "end": "12:00",
                    },
                },
            ),
            FactRule(
                "你太瘦了",
                {"speaker": "bot", "subject": "user", "category": "life", "text": "对方很瘦"},
            ),
        ],
        followups=[
            FollowupRule(
                "后天下午三点我要考试",
                {"text": "对方后天下午三点考试", "due": "后天下午三点", "window_minutes": 180},
            )
        ],
    )


def bot_turns() -> list[BotMessage]:
    return [
        BotMessage("t1", "user", "我养了一只狗", FIRST),
        BotMessage("t2", "bot", "我今天去爬山了", FIRST + timedelta(seconds=30)),
        BotMessage("t3", "user", "后天下午三点我要考试", FIRST + timedelta(minutes=2)),
        BotMessage("t4", "bot", "你太瘦了", FIRST + timedelta(minutes=3)),
    ]


async def test_the_extraction_job_keeps_what_the_user_said_and_what_the_bot_made_up(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = bot_script()
    api.post(API).mock(side_effect=model)
    assert queue_bot_extraction(services, bot_turns())
    summary = await worker_for(services, **{EXTRACT_JOB: handle_memory_extract}).run_until_idle()
    assert summary.done == 1 and summary.failed == 0
    facts = {f.text: f for f in memory.store.all_facts()}
    assert set(facts) == {"对方养了一只狗", "她今天去爬山了"}  # the echo about the user is dropped
    assert facts["对方养了一只狗"].source == "user_said"
    assert facts["她今天去爬山了"].source == "bot_invented"
    assert memory.store.bot_online_at() == FIRST  # the bot's era begins with the conversation
    events = memory.store.events(day=memory.clock.bot_date(FIRST))
    assert [(e.activity, e.source, e.fact_id) for e in events] == [
        ("爬山", "improvised", facts["她今天去爬山了"].id)
    ]
    (follow,) = memory.store.followups()
    assert follow.origin == "bot_session" and follow.is_open and follow.window_minutes == 180
    assert follow.due_at > FIRST


async def test_the_extraction_job_does_nothing_when_it_is_switched_off(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = bot_script()
    api.post(API).mock(side_effect=model)
    queue_bot_extraction(services, bot_turns())
    services.settings.memory.fact_extraction = False  # switched off after the job was queued
    summary = await worker_for(services, **{EXTRACT_JOB: handle_memory_extract}).run_until_idle()
    assert summary.done == 1 and model.requests == [] and memory.store.all_facts() == []


async def test_an_open_follow_up_is_shown_to_the_extractor_and_closed_when_the_user_raises_it(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = bot_script()
    api.post(API).mock(side_effect=model)
    queue_bot_extraction(services, bot_turns())
    worker = worker_for(services, **{EXTRACT_JOB: handle_memory_extract})
    await worker.run_until_idle()
    model.closes.append(CloseRule("考试考完了", "对方后天下午三点考试"))
    later = FIRST + timedelta(days=3)
    queue_bot_extraction(
        services,
        [
            BotMessage("t5", "user", "考试考完了", later),
            BotMessage("t6", "bot", "考得怎么样", later),
        ],
    )
    await worker.run_until_idle()
    (follow,) = memory.store.followups()
    assert follow.status == "done" and follow.closed_at == later
    last_prompt = model.requests[-1]["messages"][1]["content"]
    assert "对方后天下午三点考试" in last_prompt  # the open one was offered for closing


# ------------------------------------------------------------------ day loaders


def test_real_day_lines_are_her_local_day_with_who_said_it(
    services: Services, memory: Memory
) -> None:
    write_messages(
        services,
        [
            (datetime(2026, 3, 5, 5, 59, tzinfo=UTC), True, "前一天深夜"),  # 23:59 on the 4th
            (FIRST, True, "早上好"),
            (FIRST + timedelta(minutes=1), False, "早"),
            (datetime(2026, 3, 6, 5, 59, tzinfo=UTC), True, "这天的最后一句"),  # 23:59 on the 5th
            (datetime(2026, 3, 6, 6, 0, tzinfo=UTC), True, "第二天"),
        ],
    )
    lines = real_day_lines(services, memory.clock, DAY)
    assert [(line.speaker, line.text) for line in lines] == [
        ("her", "早上好"),
        ("user", "早"),
        ("her", "这天的最后一句"),
    ]
    assert lines[0].ref.startswith("synth-") and lines[0].at == FIRST
    assert real_day_lines(services, memory.clock, date(2026, 3, 9)) == []
    assert day_lines(services, memory.clock, "real", DAY, None) == lines
    assert day_lines(services, memory.clock, "bot", DAY, None) == []


def test_stickers_and_empty_messages_become_event_text_or_nothing(
    services: Services, memory: Memory
) -> None:
    writer = MessageWriter(services)
    writer.add(FIRST, False, "text", "你好")
    writer.add(FIRST + timedelta(seconds=5), False, "text", "")
    writer.add(FIRST + timedelta(seconds=10), True, "sticker", None, "a" * 32)
    writer.store()
    lines = real_day_lines(services, memory.clock, DAY)
    assert lines[0].text == "你好"
    assert all(line.text.strip() for line in lines)
    with services.db.session() as session:
        rows = list(session.scalars(select(Message).order_by(Message.sort_seq)))
        expected = [text for row in rows if (text := message_text(row))]
    assert [line.text for line in lines] == expected


def test_bot_day_lines_are_cut_at_the_bots_midnight_and_tidy(memory: Memory) -> None:
    reader = ListBotTurnReader()
    reader.add("user", "  今天\n很累  ", FIRST)
    reader.add("bot", "  ", FIRST + timedelta(seconds=5))
    reader.add("bot", "早点休息" * 200, FIRST + timedelta(seconds=10))
    reader.add("user", "第二天的话", FIRST + timedelta(days=1))
    lines = bot_day_lines(reader, memory.clock, DAY)
    assert [line.speaker for line in lines] == ["user", "bot"]
    assert lines[0].text == "今天 很累" and len(lines[1].text) == 400
    assert day_lines(memory.services, memory.clock, "bot", DAY, reader) == lines


def test_a_fingerprint_changes_with_any_line() -> None:
    lines = [DialogueLine("m1", "her", "你好", FIRST), DialogueLine("m2", "user", "嗨", FIRST)]
    same = [DialogueLine("m1", "her", "你好", FIRST), DialogueLine("m2", "user", "嗨", FIRST)]
    assert lines_hash(lines) == lines_hash(same) and len(lines_hash(lines)) == 40
    assert lines_hash(lines) != lines_hash(lines[:1]) != lines_hash([])
    other = [lines[0], DialogueLine("m2", "user", "嗨呀", FIRST)]
    assert lines_hash(other) != lines_hash(lines)
    hasher = LineHasher()
    for line in lines:
        hasher.update(line)
    assert hasher.hexdigest() == lines_hash(lines)


# ------------------------------------------------------------------ the import hook


def context_for(services: Services, *, first: bool = False) -> HookContext:
    return HookContext(
        services=services,
        run_id="run",
        conversation_id="c",
        export_id="e",
        inserted=10,
        changed=0,
        first_import=first,
    )


def local(day: int, hour: int) -> datetime:
    """A wall-clock time in Chicago (standard time until the 8th of March 2026)."""
    return datetime(2026, 3, day, hour, 0, tzinfo=CHICAGO).astimezone(UTC)


HISTORY = [
    (local(2, 14), False, "你家的猫叫什么"),
    (local(2, 14) + timedelta(minutes=1), True, "我家的猫叫豆包"),
    (local(4, 20), False, "明天下午三点我要考试"),
    (local(4, 20) + timedelta(minutes=1), True, "加油呀"),
]


def history_script() -> ScriptedMemoryModel:
    return ScriptedMemoryModel(
        facts=[
            FactRule(
                "我家的猫叫豆包", {"subject": "her", "category": "life", "text": "她的猫叫豆包"}
            )
        ]
    )


def test_the_hook_is_registered_with_its_backfill_command() -> None:
    hook = next(h for h in load_hooks().hooks() if h.name == "memory_replay")
    assert hook.backfill_command == "memory replay start"
    assert hook.run is queue_memory_replay


def test_the_first_replay_waits_for_the_approval_of_the_person(
    services: Services, memory: Memory, runtime: LlmRuntime
) -> None:
    write_messages(services, HISTORY)
    result = queue_memory_replay(context_for(services, first=True))
    assert result.status == "queued" and result.jobs == 1
    assert "waiting for `twin jobs approve" in result.detail and "$" in result.detail
    (job,) = jobs_of(services, REPLAY_JOB)
    assert job.requires_approval and job.approved_at is None
    again = queue_memory_replay(context_for(services))  # the same days are not queued twice
    assert again.status == "skipped" and "already queued" in again.detail


async def test_a_small_increment_is_approved_at_once_and_a_replayed_history_is_left_alone(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(side_effect=history_script())
    write_messages(services, HISTORY)
    queue_memory_replay(context_for(services, first=True))
    batch = jobs_of(services, REPLAY_JOB)[0].batch_id
    runtime.batches.approve(batch)
    worker = worker_for(services, **{REPLAY_JOB: handle_memory_replay})
    assert (await worker.run_until_idle()).done == 1
    assert len(memory.store.replay_days()) == 2
    done = queue_memory_replay(context_for(services))
    assert done.status == "skipped" and "has been replayed" in done.detail

    write_messages(services, [(local(9, 12), False, "今天天气真好")], append=True)
    increment = queue_memory_replay(context_for(services))
    assert increment.status == "queued" and "approved automatically" in increment.detail
    (new_job,) = jobs_of(services, REPLAY_JOB, "pending")
    assert new_job.approved_at is not None and new_job.payload["dates"] == ["2026-03-09"]
    assert (await worker.run_until_idle()).done == 1
    assert len(memory.store.replay_days()) == 3


async def test_a_changed_day_is_replayed_again_and_a_large_increment_waits(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(side_effect=history_script())
    write_messages(services, HISTORY)
    runtime_plan = plan_replay(services, runtime=runtime)
    runtime.batches.approve(runtime_plan.batches[0].batch_id)
    await worker_for(services, **{REPLAY_JOB: handle_memory_replay}).run_until_idle()

    write_messages(
        services, [(local(2, 15), False, "又想起来一件事")], append=True
    )  # a replayed day
    services.settings.memory.replay_auto_approve_ratio = 0.0  # nothing counts as small
    result = queue_memory_replay(context_for(services))
    assert result.status == "queued" and "waiting for `twin jobs approve" in result.detail
    (job,) = jobs_of(services, REPLAY_JOB, "pending")
    assert job.approved_at is None and job.payload["dates"] == ["2026-03-02"]


def test_the_hook_does_nothing_when_both_memory_features_are_off(services: Services) -> None:
    write_messages(services, HISTORY)
    services.settings.memory.fact_extraction = False
    services.settings.memory.daily_summary = False
    result = queue_memory_replay(context_for(services, first=True))
    assert result.status == "skipped" and "off" in result.detail
    assert jobs_of(services, REPLAY_JOB) == []


def test_the_hook_without_any_record_has_nothing_to_do(services: Services) -> None:
    result = queue_memory_replay(context_for(services, first=True))
    assert result.status == "skipped" and jobs_of(services, REPLAY_JOB) == []


def test_the_description_of_a_plan_names_the_batches_and_the_money(
    services: Services, memory: Memory, runtime: LlmRuntime
) -> None:
    write_messages(services, HISTORY)
    plan = plan_replay(services, runtime=runtime)
    line = describe_plan(plan)
    assert plan.batches[0].batch_id in line and "2 day(s)" in line and "approve" in line
    plan.approved = True
    assert "approved automatically" in describe_plan(plan)
