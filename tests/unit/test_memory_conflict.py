"""The rules that decide what a new fact does to the old ones (R-MEM-004, R-MEM-011)."""

from __future__ import annotations

from datetime import date

import pytest

from tests.support.memory import fact_draft, fact_record, utc
from twin.memory.conflict import (
    EventCandidate,
    FactCandidate,
    Recall,
    decide,
    normalised,
)
from twin.memory.records import LifelineRecord

OLD = utc(2026, 3, 1)
NEW = utc(2026, 3, 10)


def candidate(key: str, **fields: object) -> FactCandidate:
    return FactCandidate(key, fact_record(id=f"F-{key}", **fields))


def event(key: str = "l1") -> EventCandidate:
    return EventCandidate(
        key,
        LifelineRecord(
            id=f"E-{key}",
            rev=1,
            local_date=date(2026, 3, 10),
            timezone="America/Chicago",
            start_local="09:00",
            end_local="10:00",
            activity="去图书馆",
            place=None,
            mood=None,
            detail=None,
            source="plan",
            status="active",
            consistency_checked_at=None,
            invalidated_at=None,
            invalidated_by=None,
            fact_id=None,
            created_at=OLD,
            updated_at=OLD,
        ),
    )


def test_a_fact_nothing_clashes_with_is_simply_stored() -> None:
    decision = decide(fact_draft(known_at=NEW), Recall(), {})
    assert (decision.action, decision.supersede, decision.superseded_by) == ("insert", (), None)
    unrelated = Recall((candidate("c1"),))
    assert decide(fact_draft(known_at=NEW), unrelated, {"c1": "unrelated"}).action == "insert"
    assert decide(fact_draft(known_at=NEW), unrelated, {}).supersede == ()  # no verdict: unrelated


@pytest.mark.parametrize(
    ("old_source", "new_source"),
    [
        ("real_record", "real_record"),
        ("real_record", "user_said"),
        ("real_record", "bot_invented"),
        ("user_said", "bot_invented"),
        ("bot_invented", "bot_invented"),
        ("user_command", "real_record"),
    ],
)
def test_the_same_thing_from_an_equal_or_lower_source_adds_nothing(
    old_source: str, new_source: str
) -> None:
    recall = Recall((candidate("c1", source=old_source, known_at=OLD),))
    decision = decide(fact_draft(source=new_source, known_at=NEW), recall, {"c1": "same"})
    assert (decision.action, decision.target_id) == ("merge", "F-c1")


def test_the_same_thing_from_a_better_source_replaces_the_old_one() -> None:
    """The bot invented it, then a real record says so too: the real one is what counts."""
    recall = Recall((candidate("c1", source="bot_invented", known_at=OLD),))
    decision = decide(fact_draft(source="real_record", known_at=NEW), recall, {"c1": "same"})
    assert (decision.action, decision.supersede) == ("insert", ("F-c1",))


def test_the_best_of_several_equal_facts_is_the_one_merged_into() -> None:
    recall = Recall(
        (
            candidate("c1", source="bot_invented", known_at=NEW),
            candidate("c2", source="user_said", known_at=OLD),
        )
    )
    decision = decide(
        fact_draft(source="user_said", known_at=NEW), recall, {"c1": "same", "c2": "same"}
    )
    assert (decision.action, decision.target_id) == ("merge", "F-c2")


@pytest.mark.parametrize("relation", ["update", "conflict"])
@pytest.mark.parametrize(
    ("old_source", "new_source"),
    [("real_record", "user_said"), ("real_record", "bot_invented"), ("user_said", "bot_invented")],
)
def test_a_lower_source_never_overrides_a_higher_one(
    relation: str, old_source: str, new_source: str
) -> None:
    recall = Recall((candidate("c1", source=old_source, known_at=OLD),))
    decision = decide(fact_draft(source=new_source, known_at=NEW), recall, {"c1": relation})
    assert (decision.action, decision.target_id) == ("reject", "F-c1")
    assert decision.supersede == () and decision.invalidate_events == ()


def test_one_vetoing_fact_among_several_rejects_the_new_fact() -> None:
    recall = Recall(
        (
            candidate("c1", source="bot_invented", known_at=OLD),
            candidate("c2", source="real_record", known_at=OLD),
        )
    )
    decision = decide(
        fact_draft(source="user_said", known_at=NEW), recall, {"c1": "conflict", "c2": "update"}
    )
    assert (decision.action, decision.target_id) == ("reject", "F-c2")


@pytest.mark.parametrize("relation", ["update", "conflict"])
@pytest.mark.parametrize(
    ("old_source", "new_source"),
    [
        ("bot_invented", "real_record"),
        ("bot_invented", "user_said"),
        ("user_said", "real_record"),
        ("real_record", "user_command"),
    ],
)
def test_a_higher_source_replaces_a_lower_one(
    relation: str, old_source: str, new_source: str
) -> None:
    recall = Recall((candidate("c1", source=old_source, known_at=OLD),))
    decision = decide(fact_draft(source=new_source, known_at=NEW), recall, {"c1": relation})
    assert (decision.action, decision.supersede, decision.superseded_by) == (
        "insert",
        ("F-c1",),
        None,
    )


def test_with_equal_sources_the_newer_fact_wins() -> None:
    recall = Recall((candidate("c1", source="real_record", known_at=OLD),))
    decision = decide(fact_draft(known_at=NEW), recall, {"c1": "update"})
    assert (decision.action, decision.supersede) == ("insert", ("F-c1",))


def test_a_replay_that_meets_an_older_day_last_stores_it_as_already_replaced() -> None:
    """Known time decides, not the order of processing."""
    recall = Recall((candidate("c1", source="real_record", known_at=NEW),))
    decision = decide(fact_draft(known_at=OLD), recall, {"c1": "conflict"})
    assert (decision.action, decision.supersede, decision.superseded_by) == (
        "insert",
        (),
        "F-c1",
    )


def test_replacing_some_and_being_replaced_by_another_can_happen_at_once() -> None:
    recall = Recall(
        (
            candidate("c1", source="real_record", known_at=utc(2026, 3, 1)),
            candidate("c2", source="real_record", known_at=utc(2026, 3, 20)),
        )
    )
    decision = decide(
        fact_draft(known_at=utc(2026, 3, 10)), recall, {"c1": "update", "c2": "conflict"}
    )
    assert (decision.supersede, decision.superseded_by) == (("F-c1",), "F-c2")


def test_a_real_fact_invalidates_the_life_line_entries_it_contradicts() -> None:
    recall = Recall((), (event("l1"), event("l2")))
    decision = decide(fact_draft(known_at=NEW), recall, {"l1": "conflict", "l2": "unrelated"})
    assert decision.invalidate_events == ("E-l1",)
    user = decide(fact_draft(source="user_said", known_at=NEW), recall, {"l2": "conflict"})
    assert user.invalidate_events == ("E-l2",)


def test_the_bot_cannot_invalidate_its_own_life_line_and_a_vetoed_fact_invalidates_nothing() -> (
    None
):
    recall = Recall((), (event(),))
    bot = decide(fact_draft(source="bot_invented", known_at=NEW), recall, {"l1": "conflict"})
    assert bot.invalidate_events == ()
    vetoed = Recall((candidate("c1", source="real_record", known_at=OLD),), (event(),))
    decision = decide(
        fact_draft(source="user_said", known_at=NEW), vetoed, {"c1": "conflict", "l1": "conflict"}
    )
    assert decision.action == "reject" and decision.invalidate_events == ()


def test_the_relations_are_counted_for_the_report() -> None:
    recall = Recall((candidate("c1"), candidate("c2")), (event(),))
    decision = decide(fact_draft(), recall, {"c1": "same", "l1": "conflict"})
    assert decision.relation_counts == {"same": 1, "unrelated": 1, "conflict": 1}


def test_normalising_ignores_spaces_punctuation_and_case() -> None:
    assert normalised("她 喜欢，吃 Hot Pot！") == normalised("她喜欢吃hotpot")
    assert normalised("她喜欢吃火锅") != normalised("她喜欢吃烧烤")
