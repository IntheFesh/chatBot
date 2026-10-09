"""Storing facts: conflicts, priorities, vectors, the life line, follow-ups (R-MEM-004/005/011)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import date, timedelta

import pytest
import respx

from tests.support.deepseek import API, TEST_KEY
from tests.support.embedding import HashingBackend
from tests.support.memory import (
    ScriptedMemoryModel,
    add_event,
    add_fact,
    fact_draft,
    make_memory,
    utc,
)
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY
from twin.memory.conflict import JUDGE_BATCH
from twin.memory.extract import Extraction, FollowupDraft, LifelineDraft
from twin.memory.memory import Memory
from twin.memory.writer import MemoryWriter, merge_drafts
from twin.services import Services
from twin.storage.vector_schema import VectorKind


@dataclass
class Env:
    services: Services
    memory: Memory
    writer: MemoryWriter
    model: ScriptedMemoryModel
    runtime: LlmRuntime


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def env(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> AsyncIterator[Env]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    services.settings.memory.conflict_min_similarity = 0.1  # the toy embedding is a coarse judge
    runtime = build_llm_runtime(services)
    model = ScriptedMemoryModel()
    api.post(API).mock(side_effect=model)
    memory = make_memory(services)
    yield Env(services, memory, MemoryWriter(memory, runtime.client), model, runtime)
    await runtime.client.aclose()


def current_texts(memory: Memory) -> list[str]:
    memory.refresh()
    return sorted(f.text for f in memory.corpus.facts.values() if f.current)


# ------------------------------------------------------------------------ storing


async def test_new_facts_are_stored_with_a_vector_and_cost_no_judge_call(env: Env) -> None:
    report = await env.writer.write_facts(
        [fact_draft("她养了一只猫"), fact_draft("对方在读研", subject="user")], DAILY
    )
    assert report.facts_added == 2 and report.judge_calls == 0
    assert env.model.calls["conflict"] == 0
    facts = env.memory.store.facts(report.inserted)
    assert all(f.embedding_id == f.id and f.embed_version for f in facts)
    table = env.memory.vectors.table(VectorKind.FACT)
    assert table.count() == 2
    assert set(table.columns()) == {"id", "vector", "at", "kind"}  # no text in the index
    assert {row["id"] for row in table.rows()} == {f.id for f in facts}
    assert current_texts(env.memory) == sorted(["对方在读研", "她养了一只猫"])


async def test_an_exact_repeat_is_merged_into_the_fact_that_says_it_without_asking_the_model(
    env: Env,
) -> None:
    first = add_fact(
        env.memory, "她养了一只猫", utc(2026, 3, 5), evidence={"kind": "messages", "ids": ["m5"]}
    )
    report = await env.writer.write_facts(
        [
            fact_draft(
                "她 养了一只猫！", known_at=utc(2026, 3, 1), evidence_refs=("m1",), importance=5
            )
        ],
        DAILY,
    )
    assert (report.merged, report.facts_added, report.judge_calls) == (1, 0, 0)
    merged = env.memory.store.fact(first.id)
    assert merged is not None
    assert merged.known_at == utc(2026, 3, 1)  # the earliest time it could be known
    assert merged.importance == 5 and merged.evidence == {"kind": "messages", "ids": ["m5", "m1"]}
    assert len(current_texts(env.memory)) == 1


async def test_facts_that_say_the_same_in_one_extraction_become_one() -> None:
    merged = merge_drafts(
        [
            fact_draft("她养了一只猫", known_at=utc(2026, 3, 3), evidence_refs=("m3",)),
            fact_draft(
                "她养了 一只猫。", known_at=utc(2026, 3, 1), evidence_refs=("m1",), importance=4
            ),
            fact_draft("对方养了一只猫", subject="user"),
            fact_draft("她养了一只猫", source="user_said"),
        ]
    )
    assert len(merged) == 3
    cat = merged[0]
    assert (cat.known_at, cat.evidence_refs, cat.importance) == (utc(2026, 3, 1), ("m3", "m1"), 4)


# ----------------------------------------------------------- the priority rules


async def test_a_newer_fact_of_the_same_source_replaces_the_older_one(env: Env) -> None:
    old = add_fact(env.memory, "她现在住在北京", utc(2026, 3, 1))
    env.model.relate("上海", "北京", "update")
    report = await env.writer.write_facts(
        [fact_draft("她现在住在上海", known_at=utc(2026, 3, 10))], DAILY
    )
    assert (report.facts_added, report.superseded, report.judge_calls) == (1, 1, 1)
    replaced = env.memory.store.fact(old.id)
    assert replaced is not None and not replaced.current
    assert (replaced.superseded_at, replaced.valid_to) == (utc(2026, 3, 10), utc(2026, 3, 10))
    (new_id,) = report.inserted
    assert replaced.superseded_by == new_id
    assert current_texts(env.memory) == ["她现在住在上海"]


async def test_a_replay_that_meets_an_older_day_last_keeps_the_newer_fact(env: Env) -> None:
    newer = add_fact(env.memory, "她现在住在上海", utc(2026, 3, 10))
    env.model.relate("北京", "上海", "conflict")
    report = await env.writer.write_facts(
        [fact_draft("她现在住在北京", known_at=utc(2026, 3, 1))], DAILY
    )
    (stored_id,) = report.inserted
    stored = env.memory.store.fact(stored_id)
    assert stored is not None and stored.superseded_by == newer.id  # already history
    assert (stored.superseded_at, stored.valid_to) == (utc(2026, 3, 10), utc(2026, 3, 10))
    assert env.memory.store.fact(newer.id).current  # type: ignore[union-attr]
    assert current_texts(env.memory) == ["她现在住在上海"]


@pytest.mark.parametrize(
    ("old_source", "new_source"),
    [("real_record", "user_said"), ("real_record", "bot_invented"), ("user_said", "bot_invented")],
)
async def test_a_lower_source_is_kept_as_a_rejected_candidate_and_never_shown(
    env: Env, old_source: str, new_source: str
) -> None:
    held = add_fact(env.memory, "她现在住在北京", utc(2026, 3, 1), source=old_source)
    env.model.relate("上海", "北京", "conflict")
    report = await env.writer.write_facts(
        [fact_draft("她现在住在上海", source=new_source, known_at=utc(2026, 3, 10))], DAILY
    )
    assert (report.rejected, report.facts_added, report.superseded) == (1, 0, 0)
    assert env.memory.store.fact(held.id).current  # type: ignore[union-attr]
    rejected = [f for f in env.memory.store.all_facts() if f.status == "rejected"]
    assert len(rejected) == 1 and rejected[0].rejected_by == held.id
    assert rejected[0].text == "她现在住在上海" and rejected[0].embedding_id is None
    assert current_texts(env.memory) == ["她现在住在北京"]  # and it is not in the snapshot either


async def test_what_the_user_commands_outranks_a_real_record(env: Env) -> None:
    real = add_fact(env.memory, "她的生日是三月五日", utc(2026, 3, 1))
    env.model.relate("三月六日", "三月五日", "conflict")
    await env.writer.write_facts(
        [
            fact_draft(
                "她的生日是三月六日",
                source="user_command",
                confidence=1.0,
                known_at=utc(2026, 3, 9),
            )
        ],
        DAILY,
    )
    assert not env.memory.store.fact(real.id).current  # type: ignore[union-attr]
    assert current_texts(env.memory) == ["她的生日是三月六日"]


async def test_a_real_record_that_arrives_later_ends_what_the_bot_made_up(env: Env) -> None:
    """R-MEM-011: the bot invented it; the records say otherwise."""
    invented = add_fact(env.memory, "她周末去了爬山", utc(2026, 10, 1), source="bot_invented")
    plan = add_event(env.memory, date(2026, 3, 7), "去爬山", start="09:00", end="12:00")
    other = add_event(env.memory, date(2026, 3, 7), "在家看书", start="14:00", end="16:00")
    env.model.relate("在医院", "爬山", "conflict")
    env.model.relate("在医院", "周末去了爬山", "conflict")
    report = await env.writer.write_facts(
        [
            fact_draft(
                "她周末在医院陪外婆", known_at=utc(2026, 3, 7, 20), event_date=date(2026, 3, 7)
            )
        ],
        DAILY,
    )
    assert (report.facts_added, report.superseded, report.events_invalidated) == (1, 1, 1)
    fallen = env.memory.store.fact(invented.id)
    assert fallen is not None and not fallen.current and fallen.valid_to == utc(2026, 10, 1)
    # its interval is empty: it was never valid, whenever one looks
    events = {e.id: e for e in env.memory.store.events(include_invalid=True)}
    assert events[plan.id].status == "invalidated" and events[other.id].status == "active"
    assert events[plan.id].invalidated_at is not None
    assert [e.id for e in env.memory.store.events(day=date(2026, 3, 7))] == [other.id]


async def test_a_bot_invention_cannot_end_the_life_line_of_the_bot(env: Env) -> None:
    add_event(env.memory, date(2026, 3, 7), "去爬山", start="09:00", end="12:00")
    env.model.relate("在医院", "爬山", "conflict")
    report = await env.writer.write_facts(
        [
            fact_draft(
                "她在医院",
                source="bot_invented",
                known_at=utc(2026, 10, 1),
                event_date=date(2026, 3, 7),
            )
        ],
        DAILY,
    )
    assert report.events_invalidated == 0  # events are only a candidate for higher sources


async def test_without_a_model_the_facts_are_stored_as_unrelated(
    services: Services, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    add_fact(memory, "她住在北京", utc(2026, 3, 1))
    writer = MemoryWriter(memory, None)
    report = await writer.write_facts(
        [fact_draft("她搬到了上海", known_at=utc(2026, 3, 10))], DAILY
    )
    assert report.facts_added == 1 and report.superseded == 0
    assert current_texts(memory) == ["她住在北京", "她搬到了上海"]


async def test_many_new_facts_are_judged_in_batches(env: Env) -> None:
    add_fact(env.memory, "她喜欢吃火锅", utc(2026, 3, 1))
    drafts = [
        fact_draft(f"她喜欢吃火锅和菜{number}", known_at=utc(2026, 3, 2))
        for number in range(JUDGE_BATCH + 1)
    ]
    report = await env.writer.write_facts(drafts, DAILY)
    assert report.judge_calls == 2 and env.model.calls["conflict"] == 2
    assert report.facts_added == JUDGE_BATCH + 1


# -------------------------------------------------------------------- the life line


async def test_what_the_bot_says_about_herself_goes_on_her_life_line_too(env: Env) -> None:
    report = await env.writer.write_facts(
        [
            fact_draft(
                "她今天去了图书馆自习",
                source="bot_invented",
                known_at=utc(2026, 3, 5, 18),
                evidence_kind="bot_turns",
                lifeline=LifelineDraft("去图书馆自习", "学校图书馆", "专心", "14:00", "16:00"),
            ),
            fact_draft(
                "她中午吃了牛肉面",
                source="bot_invented",
                known_at=utc(2026, 3, 5, 18),
                category="life",
            ),
            fact_draft(
                "她喜欢蓝色",
                source="bot_invented",
                known_at=utc(2026, 3, 5, 18),
                category="preference",
            ),
            fact_draft(
                "对方说他在读研",
                source="user_said",
                subject="user",
                known_at=utc(2026, 3, 5, 18),
                category="life",
            ),
        ],
        DAILY,
    )
    assert report.improvised_events == 2
    events = env.memory.store.events()
    assert {e.activity for e in events} == {"去图书馆自习", "她中午吃了牛肉面"}
    library = next(e for e in events if e.place)
    assert (library.source, library.start_local, library.end_local, library.mood) == (
        "improvised",
        "14:00",
        "16:00",
        "专心",
    )
    assert library.local_date == date(2026, 3, 5) and library.fact_id in report.inserted
    assert env.memory.era().reached(utc(2026, 3, 6))  # the bot's conversation exists from here on


# ----------------------------------------------------------------------- follow-ups


async def test_follow_ups_are_stored_linked_to_their_fact_and_not_stored_twice(env: Env) -> None:
    due = utc(2026, 3, 6, 21)
    extraction = Extraction(
        facts=[fact_draft("她周五下午有面试", evidence_refs=("m1",), known_at=utc(2026, 3, 4))],
        followups=[
            FollowupDraft(
                "她周五下午面试", due, 240, utc(2026, 3, 4), "real_record", "messages", ("m1",)
            ),
            FollowupDraft(
                "她周五下午面试了",
                due + timedelta(minutes=5),
                240,
                utc(2026, 3, 4),
                "real_record",
                "messages",
                ("m1",),
            ),
            FollowupDraft(
                "她周日看牙医",
                due + timedelta(days=2),
                120,
                utc(2026, 3, 4),
                "real_record",
                "messages",
                ("m9",),
            ),
        ],
    )
    report = await env.writer.write(extraction, DAILY)
    assert (report.facts_added, report.followups_added, report.followups_repeated) == (1, 2, 1)
    follows = env.memory.store.followups()
    by_text = {f.text: f for f in follows}
    assert by_text["她周五下午面试"].fact_id == report.inserted[0]  # shares its evidence line
    assert by_text["她周日看牙医"].fact_id is None
    assert by_text["她周五下午面试"].created_at == utc(
        2026, 3, 4
    )  # known from the message, not now
    again = await env.writer.write(extraction, DAILY)
    assert again.followups_added == 0 and again.followups_repeated == 3


async def test_a_follow_up_the_conversation_shows_to_be_over_is_closed_at_that_message(
    env: Env,
) -> None:
    from twin.memory.extract import ClosedDraft
    from twin.memory.followups import FollowupStore

    store = FollowupStore(env.memory)
    follow, _ = store.add(
        "她周五面试", utc(2026, 3, 6, 21), created_at=utc(2026, 3, 4), origin="real_record"
    )
    report = await env.writer.write(
        Extraction(closed=[ClosedDraft(follow.id, "done", utc(2026, 3, 6, 23))]), DAILY
    )
    assert report.followups_closed == 1
    closed = env.memory.store.followup(follow.id)
    assert closed is not None and (closed.status, closed.closed_at, closed.close_reason) == (
        "done",
        utc(2026, 3, 6, 23),
        "mentioned",
    )
    again = await env.writer.write(
        Extraction(closed=[ClosedDraft(follow.id, "done", utc(2026, 3, 7))]), DAILY
    )
    assert again.followups_closed == 0  # it was not open any more
