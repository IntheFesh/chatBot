"""Daily summaries: scopes, versions, the 300-character limit (R-MEM-002)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, date, datetime, timedelta

import pytest
import respx

from tests.support.deepseek import API, TEST_KEY, ok
from tests.support.embedding import HashingBackend
from tests.support.memory import ScriptedMemoryModel, make_memory
from twin.llm.errors import StructuredOutputError
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY
from twin.memory.extract import DialogueLine
from twin.memory.memory import Memory
from twin.memory.summarize import DailySummarizer
from twin.services import Services
from twin.storage.vector_schema import VectorKind

DAY = date(2026, 3, 5)
FIRST = datetime(2026, 3, 5, 15, 0, tzinfo=UTC)  # 09:00 in Chicago


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
def memory(services: Services, embedder: HashingBackend) -> Memory:
    return make_memory(services)


def lines(count: int = 4) -> list[DialogueLine]:
    return [
        DialogueLine(
            f"m{n}", "her" if n % 2 else "user", f"第{n}句话", FIRST + timedelta(minutes=n)
        )
        for n in range(1, count + 1)
    ]


async def test_a_day_is_summarised_stored_as_a_version_and_encoded(
    memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = ScriptedMemoryModel(summaries={"2026-03-05": "这天他们聊了考试和晚饭。"})
    api.post(API).mock(side_effect=model)
    summarizer = DailySummarizer(memory, runtime.client)
    result = await summarizer.summarize("real", DAY, lines(), DAILY)
    record = result.record
    assert record is not None and result.skipped is None and result.calls == 1
    assert (record.scope, record.local_date, record.version, record.is_current) == (
        "real",
        DAY,
        1,
        True,
    )
    assert record.text == "这天他们聊了考试和晚饭。" and record.timezone == "America/Chicago"
    assert (record.utc_start, record.utc_end) == (
        datetime(2026, 3, 5, 6, 0, tzinfo=UTC),
        datetime(2026, 3, 6, 6, 0, tzinfo=UTC),
    )
    prompt = model.requests[0]["messages"][1]["content"]
    assert (
        "日期：2026-03-05（周四，时区 America/Chicago）" in prompt
        and "[09:01] 她：第1句话" in prompt
    )
    assert "真实聊天记录" in model.requests[0]["messages"][0]["content"]
    assert "不超过 300 个字" in model.requests[0]["messages"][0]["content"]
    stored = memory.store.current_summary("real", DAY)
    assert stored is not None and stored.embedding_id == stored.id and stored.embed_version
    table = memory.vectors.table(VectorKind.SUMMARY)
    assert table.count() == 1 and set(table.columns()) == {"id", "vector", "at", "kind"}


async def test_the_bot_scope_is_told_who_is_who_and_starts_the_bot_era(
    memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = ScriptedMemoryModel()
    api.post(API).mock(side_effect=model)
    bot_lines = [
        DialogueLine("t1", "user", "今天好累", FIRST),
        DialogueLine("t2", "bot", "早点休息", FIRST + timedelta(seconds=30)),
    ]
    result = await DailySummarizer(memory, runtime.client).summarize("bot", DAY, bot_lines, DAILY)
    assert result.record is not None and result.record.scope == "bot"
    assert "机器人" in model.requests[0]["messages"][0]["content"]
    assert memory.store.bot_online_at() == FIRST
    assert memory.store.current_summary("real", DAY) is None  # the two scopes are separate rows


async def test_an_unchanged_day_is_not_summarised_again_but_a_changed_or_forced_one_is(
    memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = ScriptedMemoryModel()
    api.post(API).mock(side_effect=model)
    summarizer = DailySummarizer(memory, runtime.client)
    first = await summarizer.summarize("real", DAY, lines(), DAILY)
    again = await summarizer.summarize("real", DAY, lines(), DAILY)
    assert again.skipped == "unchanged" and model.calls["summary"] == 1
    assert first.record is not None and again.record is not None
    assert again.record.id == first.record.id
    more = await summarizer.summarize("real", DAY, lines(5), DAILY)
    assert more.record is not None and more.record.version == 2
    forced = await summarizer.summarize("real", DAY, lines(5), DAILY, force=True)
    assert forced.record is not None and forced.record.version == 3
    assert [v.is_current for v in memory.store.summary_versions("real", DAY)] == [
        False,
        False,
        True,
    ]
    assert model.calls["summary"] == 3


async def test_nothing_is_asked_without_lines_or_with_summaries_switched_off(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(side_effect=ScriptedMemoryModel())
    summarizer = DailySummarizer(memory, runtime.client)
    assert (await summarizer.summarize("real", DAY, [], DAILY)).skipped == "no_lines"
    services.settings.memory.daily_summary = False
    assert not summarizer.enabled
    assert (await summarizer.summarize("real", DAY, lines(), DAILY)).skipped == "disabled"
    assert route.call_count == 0
    with pytest.raises(ValueError, match="scope"):
        await summarizer.summarize("both", DAY, lines(), DAILY)


async def test_a_summary_over_the_limit_is_sent_back_once_and_two_failures_end_the_task(
    memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    long_text = "字" * 301
    route = api.post(API).mock(
        side_effect=[
            ok(content=json.dumps({"summary": long_text}, ensure_ascii=False)),
            ok(content=json.dumps({"summary": "短一点的摘要。"}, ensure_ascii=False)),
        ]
    )
    summarizer = DailySummarizer(memory, runtime.client)
    result = await summarizer.summarize("real", DAY, lines(), DAILY)
    assert (
        result.record is not None and result.record.text == "短一点的摘要。" and result.calls == 2
    )
    retry = json.loads(route.calls[1].request.content)["messages"][-1]["content"]
    assert "could not be used" in retry and "300" in retry
    api.post(API).mock(
        return_value=ok(content=json.dumps({"summary": long_text}, ensure_ascii=False))
    )
    with pytest.raises(StructuredOutputError):
        await summarizer.summarize("real", date(2026, 3, 6), lines(), DAILY)
    assert memory.store.current_summary("real", date(2026, 3, 6)) is None


async def test_a_day_too_long_for_one_call_is_summarised_in_pieces_and_merged(
    services: Services, memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    services.settings.memory.replay_chunk_lines = 20
    model = ScriptedMemoryModel()
    api.post(API).mock(side_effect=model)
    result = await DailySummarizer(memory, runtime.client).summarize("real", DAY, lines(45), DAILY)
    assert (
        model.calls["summary"] == 3 and model.calls["summary_merge"] == 1
    )  # 20 + 20 + 5, then merge
    assert (
        result.record is not None
        and result.record.text == "合并后的一天摘要。"
        and result.calls == 4
    )
    merge_prompt = model.requests[-1]["messages"][1]["content"]
    assert (
        "第1段：" in merge_prompt
        and "第3段：" in merge_prompt
        and "分段摘要（共 3 段）" in merge_prompt
    )
