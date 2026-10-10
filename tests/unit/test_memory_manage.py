"""remember, forget and list: the commands behind /记住, /忘掉, /记忆 (R-MEM-009)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from datetime import date
from pathlib import Path

import pytest
import respx

from tests.support.deepseek import API, TEST_KEY
from tests.support.embedding import HashingBackend
from tests.support.memory import (
    FactRule,
    FollowupRule,
    ScriptedMemoryModel,
    add_fact,
    add_followup,
    make_memory,
    utc,
)
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.memory.manage import PAGE_SIZE, MemoryManager
from twin.memory.memory import Memory
from twin.memory.store import NewEvent, NewFollowup
from twin.ops.logging import configure_logging
from twin.services import Services
from twin.storage.vector_schema import VectorKind

NOW = utc(2026, 3, 10, 18)
AUDIT_FIELDS = {
    "ts", "level", "logger", "event", "pid", "action", "fact_ids", "followup_ids",
    "event_ids", "restored", "replaced", "followups", "enriched", "matched", "deleted", "kind",
}  # fmt: skip


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
    services.settings.memory.conflict_min_similarity = 0.1
    return make_memory(services)


# ---------------------------------------------------------------------- remember


async def test_remember_stores_what_the_model_understood_with_the_highest_certainty(
    memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = ScriptedMemoryModel(
        facts=[
            FactRule(
                "生日",
                {
                    "subject": "her",
                    "category": "anniversary",
                    "text": "她的生日是三月五日",
                    "importance": 5,
                    "event_date": "2000-03-05",
                    "recurrence": "yearly",
                    "confidence": 0.3,
                },
            )
        ],
        followups=[
            FollowupRule(
                "生日",
                {"text": "她三月二十日过生日，记得祝福", "due": "3月20日", "window_minutes": 600},
            )
        ],
    )
    api.post(API).mock(side_effect=model)
    result = await MemoryManager(memory, runtime.client).remember("她的生日是三月五日", now=NOW)
    (item,) = result.facts
    assert result.enriched and result.followups == 1
    assert (item.source, item.subject, item.category, item.importance) == (
        "user_command",
        "her",
        "anniversary",
        5,
    )
    fact = memory.store.fact(item.id)
    assert fact is not None and fact.confidence == 1.0
    assert (fact.event_date, fact.recurrence, fact.known_at) == (date(2000, 3, 5), "yearly", NOW)
    assert fact.evidence is not None and fact.evidence["kind"] == "command"
    (follow,) = memory.store.followups()
    assert follow.fact_id == fact.id and follow.origin == "user_command"
    assert memory.store.bot_online_at() is not None  # the command exists only in the bot's chat
    assert "她的生日是三月五日" in model.requests[0]["messages"][1]["content"]


async def test_remember_never_loses_the_text_when_nothing_is_understood(
    memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(side_effect=ScriptedMemoryModel())  # finds nothing in it
    result = await MemoryManager(memory, runtime.client).remember("  她不吃  香菜 ", now=NOW)
    (item,) = result.facts
    assert not result.enriched and item.text == "她不吃 香菜"
    assert (item.source, item.subject, item.category, item.importance) == (
        "user_command",
        "both",
        "other",
        4,
    )
    without_model = await MemoryManager(memory).remember("她喜欢蓝色", now=NOW)
    assert not without_model.enriched and without_model.facts[0].text == "她喜欢蓝色"
    with pytest.raises(ValueError, match="nothing to remember"):
        await MemoryManager(memory).remember("   ")


async def test_remember_survives_a_model_that_is_down(
    memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    model = ScriptedMemoryModel(fail_from_call=1)
    api.post(API).mock(side_effect=model)
    result = await MemoryManager(memory, runtime.client).remember("她喜欢蓝色", now=NOW)
    assert not result.enriched and result.facts[0].text == "她喜欢蓝色"


async def test_a_remembered_correction_replaces_the_real_record_it_contradicts(
    memory: Memory, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    old = add_fact(memory, "她的生日是三月五日", utc(2026, 1, 1))
    model = ScriptedMemoryModel(
        facts=[
            FactRule(
                "生日", {"subject": "her", "category": "anniversary", "text": "她的生日是三月六日"}
            )
        ]
    )
    model.relate("三月六日", "三月五日", "conflict")
    api.post(API).mock(side_effect=model)
    result = await MemoryManager(memory, runtime.client).remember("她的生日其实是三月六日", now=NOW)
    assert result.replaced == 1
    assert not memory.store.fact(old.id).current  # type: ignore[union-attr]


# ------------------------------------------------------------------------ forget


def test_forget_by_number_deletes_the_fact_and_only_what_came_from_it(memory: Memory) -> None:
    keep = add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1))
    gone = add_fact(memory, "她周五要面试", utc(2026, 3, 2), source="user_said")
    derived = memory.store.add_followup(
        NewFollowup(
            text="问问面试",
            due_at=utc(2026, 3, 6, 21),
            window_minutes=240,
            created_at=utc(2026, 3, 2),
            fact_id=gone.id,
        )
    )
    unrelated = add_followup(memory, "她周日看牙医", utc(2026, 3, 8, 21), utc(2026, 3, 2))
    event = memory.store.add_event(
        NewEvent(
            local_date=date(2026, 3, 9),
            timezone="America/Chicago",
            activity="准备面试",
            source="improvised",
            fact_id=gone.id,
        )
    )
    memory.refresh()
    result = MemoryManager(memory).forget(str(gone.number))
    kinds = sorted((d.kind, d.id) for d in result.deleted)
    assert kinds == sorted([("fact", gone.id), ("followup", derived.id), ("lifeline", event.id)])
    assert memory.store.fact(gone.id) is None and memory.store.followup(derived.id) is None
    assert memory.store.event(event.id) is None
    assert (
        memory.store.fact(keep.id) is not None and memory.store.followup(unrelated.id) is not None
    )
    assert memory.vectors.table(VectorKind.FACT).count() == 1  # the vector is gone too
    assert result.ambiguous == () and result.found


def test_forgetting_a_fact_that_replaced_another_makes_the_old_one_current_again(
    memory: Memory,
) -> None:
    old = add_fact(memory, "她住在北京", utc(2026, 3, 1))
    new = add_fact(memory, "她搬到了上海", utc(2026, 3, 10))
    memory.store.supersede_fact(old.id, new.id, at=new.known_at)
    memory.refresh()
    result = MemoryManager(memory).forget(new.id)
    assert result.restored == (old.number,)
    restored = memory.store.fact(old.id)
    assert restored is not None and restored.current


def test_forget_by_words_needs_exactly_one_match_unless_told_otherwise(memory: Memory) -> None:
    first = add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1))
    second = add_fact(memory, "她最爱吃海底捞火锅", utc(2026, 3, 2))
    other = add_fact(memory, "她每天跑步", utc(2026, 3, 3))
    manager = MemoryManager(memory)
    ambiguous = manager.forget("火锅")
    assert ambiguous.deleted == () and {i.id for i in ambiguous.ambiguous} == {first.id, second.id}
    assert memory.store.fact(first.id) is not None  # nothing was deleted
    precise = manager.forget("海底捞")
    assert [d.id for d in precise.deleted] == [second.id]
    everything = manager.forget("火锅", all_matches=True)
    assert [d.id for d in everything.deleted] == [first.id]
    assert memory.store.fact(other.id) is not None
    nothing = manager.forget("完全不存在的内容")
    assert not nothing.found and nothing.deleted == ()
    with pytest.raises(ValueError, match="which memory"):
        manager.forget("  ")


def test_forget_matches_by_keyword_when_the_words_are_not_contiguous(memory: Memory) -> None:
    fact = add_fact(memory, "她下周三要去看牙医", utc(2026, 3, 1))
    result = MemoryManager(memory).forget("看牙医下周三")
    assert [d.id for d in result.deleted] == [fact.id]


def test_a_follow_up_can_be_forgotten_by_its_words(memory: Memory) -> None:
    follow = add_followup(memory, "她周日看牙医", utc(2026, 3, 8, 21), utc(2026, 3, 2))
    add_followup(memory, "她周日看牙医复查", utc(2026, 3, 8, 22), utc(2026, 3, 2))
    manager = MemoryManager(memory)
    several = manager.forget("看牙医")
    assert several.deleted == () and len(several.followup_matches) == 2
    one = manager.forget("看牙医复查")
    assert one.deleted[0].kind == "followup" and one.deleted[0].number is None
    assert memory.store.followup(follow.id) is not None
    rest = manager.forget("看牙医", all_matches=True)
    assert [d.id for d in rest.deleted] == [follow.id]


async def test_forget_can_run_off_the_event_loop(memory: Memory) -> None:
    fact = add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1))
    result = await MemoryManager(memory).aforget(str(fact.number))
    assert [d.id for d in result.deleted] == [fact.id]


# -------------------------------------------------------------------------- list


def test_the_list_pages_newest_first_and_can_narrow_by_a_word(memory: Memory) -> None:
    for number in range(25):
        add_fact(
            memory,
            f"第{number}件事她喜欢吃火锅" if number % 5 == 0 else f"第{number}件事",
            utc(2026, 3, 1),
            embed=False,
        )
    replaced_old = add_fact(memory, "旧的事", utc(2026, 3, 1), embed=False)
    new = add_fact(memory, "新的事", utc(2026, 3, 2), embed=False)
    memory.store.supersede_fact(replaced_old.id, new.id, at=new.known_at)
    add_fact(memory, "被否决的事", utc(2026, 3, 1), status="rejected", embed=False)
    add_followup(memory, "她周日看牙医", utc(2026, 3, 8, 21), utc(2026, 3, 2))
    manager = MemoryManager(memory)
    first = manager.list_items()
    assert (first.page, first.pages, first.total) == (
        1,
        3,
        26,
    )  # 25 + the new one; not the replaced or vetoed
    assert [i.number for i in first.items] == sorted((i.number for i in first.items), reverse=True)
    assert len(first.items) == PAGE_SIZE and first.items[0].text == "新的事"
    assert first.followups == ("她周日看牙医",)
    last = manager.list_items(3)
    assert len(last.items) == 6 and last.followups == ()
    assert manager.list_items(99).page == 3 and manager.list_items(0).page == 1
    narrowed = manager.list_items(keyword="火锅")
    assert narrowed.total == 5 and narrowed.keyword == "火锅" and narrowed.pages == 1
    assert all("火锅" in i.text for i in narrowed.items) and narrowed.followups == ()
    assert manager.list_items(keyword="没有这个").items == ()
    assert manager.list_items(page_size=4).pages == 7


# ------------------------------------------------------------------------ audit


async def test_every_command_writes_an_audit_record_without_the_text(
    memory: Memory, tmp_path: Path
) -> None:
    secret = "她最隐秘的一句话"
    path = configure_logging(tmp_path / "logs", level="DEBUG", console=False)
    manager = MemoryManager(memory)
    remembered = await manager.remember(secret, now=NOW)
    manager.list_items(keyword="隐秘")
    manager.forget(str(remembered.facts[0].number))
    manager.forget("没有的东西")
    for handler in logging.getLogger("twin").handlers:
        handler.flush()
    raw = path.read_text(encoding="utf-8")
    assert secret not in raw and "隐秘" not in raw
    records = [json.loads(line) for line in raw.splitlines() if "memory_audit" in line]
    assert [r["action"] for r in records] == ["remember", "forget", "forget"]
    assert remembered.facts[0].id in records[0]["fact_ids"]
    assert remembered.facts[0].id in records[1]["fact_ids"]
    assert records[2]["matched"] == 0
    assert all(set(r) <= AUDIT_FIELDS for r in records)
