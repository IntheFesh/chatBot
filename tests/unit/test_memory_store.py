"""The memory tables and their snapshot in memory (R-STO-002, R-STO-006, R-MEM-003/002/006)."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from tests.support.embedding import HashingBackend
from tests.support.memory import (
    add_event,
    add_fact,
    add_followup,
    add_summary,
    make_memory,
    utc,
)
from twin.memory.store import NewFact
from twin.services import Services


@pytest.fixture
def memory(services: Services, embedder: HashingBackend):  # type: ignore[no-untyped-def]
    return make_memory(services)


def raw_values(services: Services, sql: str) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(services.paths.db_path)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def test_text_is_sealed_and_numbers_count_up_from_one(services: Services, memory) -> None:  # type: ignore[no-untyped-def]
    first = add_fact(memory, "她的生日是三月五日", utc(2026, 3, 1), embed=False)
    second = add_fact(memory, "对方在读研", utc(2026, 3, 2), subject="user", embed=False)
    assert (first.number, second.number) == (1, 2)
    assert first.text == "她的生日是三月五日" and first.evidence == {
        "kind": "messages",
        "ids": ["m1"],
    }
    stored = raw_values(services, "SELECT text, evidence, subject FROM facts ORDER BY number")
    assert [row[2] for row in stored] == ["her", "user"]
    for text_blob, evidence_blob, _ in stored:
        assert isinstance(text_blob, bytes) and isinstance(evidence_blob, bytes)
        assert "生日".encode() not in text_blob and b"messages" not in evidence_blob
    # a number is never reused, even after the highest one is deleted... until it is the highest
    memory.store.delete_facts([second.id])
    third = add_fact(memory, "她养了一只猫", utc(2026, 3, 3), embed=False)
    assert third.number == 2


def test_the_database_refuses_values_outside_the_closed_vocabularies(memory) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(Exception, match=r"subject|constraint|CHECK"):
        memory.store.add_fact(
            NewFact(
                subject="dog",
                category="life",
                text="x",
                source="real_record",
                known_at=utc(2026, 3, 1),
            )
        )


def test_superseding_marks_the_old_fact_and_deleting_the_new_one_restores_it(memory) -> None:  # type: ignore[no-untyped-def]
    old = add_fact(memory, "她住在北京", utc(2026, 3, 1), embed=False)
    new = add_fact(memory, "她搬到了上海", utc(2026, 3, 10), embed=False)
    replaced = memory.store.supersede_fact(old.id, new.id, at=new.known_at)
    assert (replaced.superseded_by, replaced.superseded_at, replaced.valid_to) == (
        new.id,
        new.known_at,
        new.known_at,
    )
    assert not replaced.current
    # a bound the fact already had, earlier than the replacement, is kept
    bounded = add_fact(memory, "她在实习", utc(2026, 3, 1), valid_to=utc(2026, 3, 5), embed=False)
    assert memory.store.supersede_fact(bounded.id, new.id, at=new.known_at).valid_to == utc(
        2026, 3, 5
    )
    restored = memory.store.delete_facts([new.id])
    assert set(restored) == {old.id, bounded.id}
    back = memory.store.fact(old.id)
    assert (
        back is not None and back.current and back.valid_to is None and back.superseded_at is None
    )
    assert memory.store.fact(new.id) is None
    assert memory.store.delete_facts([]) == []


def test_rejected_candidates_lose_the_link_when_their_veto_is_deleted(memory) -> None:  # type: ignore[no-untyped-def]
    veto = add_fact(memory, "她不吃香菜", utc(2026, 3, 1), embed=False)
    rejected = memory.store.add_fact(
        NewFact(
            subject="her",
            category="preference",
            text="她爱吃香菜",
            source="bot_invented",
            known_at=utc(2026, 3, 9),
            status="rejected",
            rejected_by=veto.id,
        )
    )
    memory.store.delete_facts([veto.id])
    again = memory.store.fact(rejected.id)
    assert again is not None and again.status == "rejected" and again.rejected_by is None


def test_update_fact_reseals_text_and_refuses_unknown_columns(memory) -> None:  # type: ignore[no-untyped-def]
    fact = add_fact(memory, "她喜欢咖啡", utc(2026, 3, 1), embed=False)
    changed = memory.store.update_fact(fact.id, text="她喜欢拿铁", importance=5)
    assert (changed.text, changed.importance) == ("她喜欢拿铁", 5)
    with pytest.raises(AttributeError):
        memory.store.update_fact(fact.id, colour="red")
    with pytest.raises(KeyError):
        memory.store.update_fact("missing", importance=1)
    with pytest.raises(KeyError):
        memory.store.supersede_fact("missing", fact.id, at=utc(2026, 3, 2))


def test_a_recomputed_summary_is_a_new_version_and_only_the_latest_is_current(memory) -> None:  # type: ignore[no-untyped-def]
    day = date(2026, 3, 5)
    first = add_summary(memory, "real", day, "聊了考试", embed=False)
    second = add_summary(memory, "real", day, "聊了考试和晚饭", embed=False)
    other = add_summary(memory, "bot", day, "机器人的一天", embed=False)
    assert (first.version, second.version, other.version) == (1, 2, 1)
    current = memory.store.current_summary("real", day)
    assert current is not None and current.id == second.id and current.text == "聊了考试和晚饭"
    assert [s.version for s in memory.store.summary_versions("real", day)] == [1, 2]
    assert [s.is_current for s in memory.store.summary_versions("real", day)] == [False, True]
    assert {s.id for s in memory.store.current_summaries()} == {second.id, other.id}
    assert memory.store.current_summary("real", date(2026, 3, 6)) is None


def test_follow_ups_are_closed_once_and_remember_when(memory) -> None:  # type: ignore[no-untyped-def]
    follow = add_followup(memory, "她明天面试", utc(2026, 3, 6, 15), utc(2026, 3, 5))
    assert follow.is_open and follow.window_end == follow.due_at + timedelta(minutes=240)
    closed = memory.store.close_followup(
        follow.id, status="done", at=utc(2026, 3, 6, 18), reason="mentioned"
    )
    assert closed is not None and (closed.status, closed.closed_at) == ("done", utc(2026, 3, 6, 18))
    assert (
        memory.store.close_followup(follow.id, status="expired", at=utc(2026, 3, 7), reason=None)
        is None
    )
    with pytest.raises(ValueError, match="closed with"):
        memory.store.close_followup(follow.id, status="open", at=utc(2026, 3, 7), reason=None)
    assert memory.store.followups(only_open=True) == []
    assert len(memory.store.followups()) == 1
    assert memory.store.delete_followups([follow.id]) == 1


def test_life_line_entries_can_be_invalidated_stamped_and_deleted(memory) -> None:  # type: ignore[no-untyped-def]
    day = date(2026, 3, 9)
    one = add_event(memory, day, "去图书馆", start="09:00", end="11:00")
    two = add_event(memory, day, "和朋友吃饭", start="12:00", end="13:00", source="improvised")
    assert [e.activity for e in memory.store.events(day=day)] == ["去图书馆", "和朋友吃饭"]
    assert memory.store.invalidate_event(one.id, by="F1", at=utc(2026, 3, 10))
    assert not memory.store.invalidate_event(one.id, by="F1", at=utc(2026, 3, 10))
    assert [e.id for e in memory.store.events(day=day)] == [two.id]
    assert len(memory.store.events(day=day, include_invalid=True)) == 2
    memory.store.stamp_events_checked([two.id], utc(2026, 3, 10))
    stamped = memory.store.event(two.id)
    assert stamped is not None and stamped.consistency_checked_at == utc(2026, 3, 10)
    assert memory.store.delete_events([one.id, two.id]) == 2


def test_the_start_of_the_bot_conversation_keeps_the_earliest_moment(memory) -> None:  # type: ignore[no-untyped-def]
    assert memory.store.bot_online_at() is None
    assert memory.store.mark_bot_online(utc(2026, 3, 8))
    assert not memory.store.mark_bot_online(utc(2026, 3, 9))  # later: nothing changes
    assert memory.store.bot_online_at() == utc(2026, 3, 8)
    assert memory.store.mark_bot_online(utc(2026, 3, 1))  # an earlier one is taken
    assert memory.store.bot_online_at() == utc(2026, 3, 1)


def test_replay_days_record_what_was_replayed_and_from_which_messages(memory) -> None:  # type: ignore[no-untyped-def]
    day = date(2026, 3, 5)
    memory.store.mark_replayed(
        day,
        input_hash="aa",
        lines=10,
        facts_added=2,
        followups_added=1,
        summary_version=1,
        batch_id="b1",
    )
    memory.store.mark_replayed(
        day,
        input_hash="bb",
        lines=12,
        facts_added=3,
        followups_added=1,
        summary_version=2,
        batch_id=None,
    )
    days = memory.store.replay_days()
    assert list(days) == [day] and days[day].input_hash == "bb" and days[day].lines == 12
    assert memory.store.counts()["memory_replay_days"] == 1


def test_counts_by_source_leave_out_replaced_and_rejected_facts(memory) -> None:  # type: ignore[no-untyped-def]
    old = add_fact(memory, "她住在北京", utc(2026, 3, 1), embed=False)
    new = add_fact(memory, "她搬到了上海", utc(2026, 3, 10), embed=False)
    add_fact(memory, "她喜欢猫", utc(2026, 3, 2), source="bot_invented", embed=False)
    add_fact(memory, "她讨厌狗", utc(2026, 3, 2), status="rejected", embed=False)
    memory.store.supersede_fact(old.id, new.id, at=new.known_at)
    assert memory.store.fact_counts_by_source() == {"real_record": 1, "bot_invented": 1}


# ----------------------------------------------------------------------- corpus


def test_the_snapshot_follows_changes_made_elsewhere(
    services: Services, embedder: HashingBackend
) -> None:
    ours = make_memory(services)
    theirs = make_memory(services)  # another process looking at the same tables
    ours.refresh()
    assert not ours.corpus.facts
    fact = add_fact(theirs, "她养了一只猫", utc(2026, 3, 1), embed=False)
    assert ours.refresh() and fact.id in ours.corpus.facts
    assert not ours.refresh()  # nothing changed since: nothing is reloaded
    assert [h.doc_id for h in ours.corpus.fact_index.search("一只猫", 3)] == [fact.id]
    # the clock is frozen here: a change is seen through the row's revision, not its time
    theirs.store.update_fact(fact.id, text="她养了一条狗")
    assert ours.refresh()
    assert ours.corpus.facts[fact.id].text == "她养了一条狗"
    assert ours.corpus.fact_index.search("一只猫", 3) == []
    theirs.store.delete_facts([fact.id])
    assert ours.refresh() and fact.id not in ours.corpus.facts
    assert ours.corpus.fact_index.search("一条狗", 3) == []


def test_the_snapshot_holds_summaries_follow_ups_and_events_and_the_special_facts(
    memory,  # type: ignore[no-untyped-def]
) -> None:
    summary = add_summary(memory, "real", date(2026, 3, 5), "聊了考试", embed=False)
    follow = add_followup(memory, "她明天面试", utc(2026, 3, 6, 15), utc(2026, 3, 5))
    event = add_event(memory, date(2026, 3, 9), "去图书馆")
    birthday = add_fact(
        memory,
        "她的生日",
        utc(2026, 3, 1),
        event_date=date(2000, 7, 1),
        recurrence="yearly",
        importance=5,
        embed=False,
    )
    add_fact(memory, "她喜欢咖啡", utc(2026, 3, 1), embed=False)
    rejected = add_fact(memory, "她讨厌猫", utc(2026, 3, 1), status="rejected", embed=False)
    corpus = memory.corpus
    assert corpus.summary_of("real", date(2026, 3, 5)) == summary
    assert corpus.summary_of("bot", date(2026, 3, 5)) is None
    assert follow.id in corpus.followups and event.id in corpus.events
    assert [f.id for f in corpus.dated_facts()] == [birthday.id]
    assert [f.id for f in corpus.core_facts()] == [birthday.id]
    assert rejected.id not in corpus.facts  # a vetoed candidate is not in the snapshot
    assert [h.doc_id for h in corpus.summary_index.search("考试", 3)] == [summary.id]
    newer = add_summary(memory, "real", date(2026, 3, 5), "聊了面试", embed=False)
    assert (
        summary.id not in corpus.summaries and corpus.summary_of("real", date(2026, 3, 5)) == newer
    )
    assert memory.era().online_at is None
    memory.note_bot_activity(utc(2026, 3, 8))
    assert memory.era().reached(utc(2026, 3, 9))
