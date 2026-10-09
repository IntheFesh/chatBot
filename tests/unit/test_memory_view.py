"""``memory_view(as_of=t)``: only what was known by then (R-MEM-010, R-TRN-013)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

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
from twin.memory.api import Memory, MemoryQuery, memory_view
from twin.memory.view import MemoryView
from twin.services import Services

T = utc(2026, 3, 10, 18)  # 13:00 on 10 March in Chicago (daylight time since the 8th)
DAY = date(2026, 3, 10)


@pytest.fixture
def memory(services: Services, embedder: HashingBackend) -> Memory:
    return make_memory(services)


def view_at(memory: Memory, moment: datetime) -> MemoryView:
    return memory_view(memory.services, moment, memory=memory)


# ------------------------------------------------------------------------- facts


def test_a_fact_shows_only_from_the_moment_it_was_known(memory: Memory) -> None:
    early = add_fact(memory, "她养了一只猫", utc(2026, 3, 1))
    on_the_dot = add_fact(memory, "她报了潜水课", T)
    later = add_fact(memory, "她在学法语", T + timedelta(hours=1))
    ids = {f.id for f in view_at(memory, T).facts()}
    assert ids == {early.id}  # `known_at < t`, strictly
    assert {f.id for f in view_at(memory, T + timedelta(seconds=1)).facts()} == {
        early.id,
        on_the_dot.id,
    }
    assert {f.id for f in view_at(memory, T + timedelta(days=1)).facts()} == {
        early.id,
        on_the_dot.id,
        later.id,
    }
    assert view_at(memory, utc(2026, 2, 1)).facts() == ()


def test_a_replacement_that_came_later_is_not_shown_and_the_fact_stays_as_it_was(
    memory: Memory,
) -> None:
    old = add_fact(memory, "她住在北京", utc(2026, 3, 1))
    new = add_fact(memory, "她搬到了上海", utc(2026, 3, 20))
    memory.store.supersede_fact(old.id, new.id, at=new.known_at)
    memory.refresh()
    before = view_at(memory, T).facts()
    assert [f.text for f in before] == ["她住在北京"]
    assert (before[0].superseded_by, before[0].superseded_at, before[0].valid_to) == (
        None,
        None,
        None,
    )
    after = view_at(memory, utc(2026, 3, 25)).facts()
    assert [f.text for f in after] == ["她搬到了上海"]


def test_rejected_candidates_never_appear(memory: Memory) -> None:
    add_fact(memory, "她讨厌猫", utc(2026, 3, 1), status="rejected")
    assert view_at(memory, utc(2030, 1, 1)).facts() == ()


def test_before_the_bot_went_online_what_comes_from_it_is_empty(memory: Memory) -> None:
    real = add_fact(memory, "她在读研", utc(2026, 3, 1))
    invented = add_fact(memory, "她昨天去爬山了", utc(2026, 3, 9), source="bot_invented")
    said = add_fact(memory, "对方周五考试", utc(2026, 3, 9), source="user_said", subject="user")
    add_summary(memory, "bot", date(2026, 3, 8), "和机器人聊了周末")
    add_event(memory, date(2026, 3, 9), "去爬山", created_at=utc(2026, 3, 9))
    memory.store.mark_bot_online(utc(2026, 3, 8))
    memory.refresh()
    early = view_at(memory, utc(2026, 3, 7, 12))  # before the bot existed
    assert [f.id for f in early.facts()] == [real.id]
    assert early.lifeline() == () and early.summaries() == ()
    late = view_at(memory, T)  # after it
    assert {f.id for f in late.facts()} == {real.id, invented.id, said.id}
    assert [e.activity for e in late.lifeline()] == ["去爬山"]
    assert [s.scope for s in late.summaries()] == ["bot"]
    assert early.bot_era is False and late.bot_era is True


# ---------------------------------------------------------------------- summaries


def test_summaries_of_earlier_local_days_only(memory: Memory) -> None:
    for day, text in (
        (date(2026, 3, 8), "周日聊了天气"),
        (date(2026, 3, 9), "周一聊了考试"),
        (DAY, "周二这天聊了潜水课"),
        (date(2026, 3, 11), "周三聊了旅行"),
    ):
        add_summary(memory, "real", day, text)
    texts = [s.text for s in view_at(memory, T).summaries()]
    assert texts == ["周日聊了天气", "周一聊了考试"]
    # just after her midnight the 10th has ended: 05:00 UTC on the 11th is 00:00 in Chicago
    after_midnight = view_at(memory, utc(2026, 3, 11, 5, 0))
    assert [s.local_date for s in after_midnight.summaries()][-1] == DAY
    before_midnight = view_at(memory, utc(2026, 3, 11, 4, 59))
    assert DAY not in [s.local_date for s in before_midnight.summaries()]


def test_the_latest_version_of_a_day_is_the_one_shown(memory: Memory) -> None:
    add_summary(memory, "real", date(2026, 3, 9), "第一版")
    add_summary(memory, "real", date(2026, 3, 9), "第二版")
    assert [s.text for s in view_at(memory, T).summaries()] == ["第二版"]


# --------------------------------------------------------------------- follow-ups


def test_a_follow_up_closed_after_the_moment_is_shown_open(memory: Memory) -> None:
    created, closed = utc(2026, 3, 8), utc(2026, 3, 12)
    follow = add_followup(
        memory, "她周三面试", utc(2026, 3, 11, 20), created, close_at=closed, status="done"
    )
    at_t = view_at(memory, T).followups()
    assert [(f.id, f.status, f.closed_at, f.close_reason) for f in at_t] == [
        (follow.id, "open", None, None)
    ]
    assert view_at(memory, closed + timedelta(minutes=1)).followups() == ()  # closed by then
    assert view_at(memory, created).followups() == ()  # not created before the moment
    assert memory.store.followup(follow.id).status == "done"  # type: ignore[union-attr]


# ------------------------------------------------------------------------ search


def test_search_finds_visible_facts_by_meaning_and_by_keyword_and_never_a_future_one(
    memory: Memory,
) -> None:
    known = add_fact(memory, "她喜欢吃火锅和烧烤", utc(2026, 3, 1))
    other = add_fact(memory, "她每天早上跑步", utc(2026, 3, 2))
    future = add_fact(memory, "她最爱吃火锅", T + timedelta(minutes=5))  # a closer match, but later
    view = view_at(memory, T)
    hits = view.search_facts("想吃火锅", 5)
    assert hits[0].fact.id == known.id
    assert future.id not in {h.fact.id for h in hits} and other.id not in {
        h.fact.id for h in hits[:1]
    }
    assert hits[0].similarity > 0 and hits[0].keyword > 0
    assert hits[0].relevance == max(hits[0].similarity, hits[0].keyword)
    assert view.search_facts("", 5) == [] and view.search_facts("火锅", 0) == []
    later = view_at(memory, T + timedelta(hours=1)).search_facts("想吃火锅", 5)
    assert future.id in {h.fact.id for h in later}


def test_the_limit_counts_visible_facts_only(memory: Memory) -> None:
    """Hidden facts of the future must not use up the places of the search."""
    visible = add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1))
    for number in range(6):
        add_fact(memory, f"她喜欢吃火锅{number}", T + timedelta(hours=number + 1))
    hits = view_at(memory, T).search_facts("火锅", 2)
    assert [h.fact.id for h in hits] == [visible.id]


def test_summaries_are_searched_the_same_way(memory: Memory) -> None:
    past = add_summary(memory, "real", date(2026, 3, 5), "那天他们聊了潜水课的事")
    add_summary(memory, "real", date(2026, 3, 20), "那天他们也聊了潜水课的事")  # the future
    hits = view_at(memory, T).search_summaries("潜水课", 3)
    assert [h.summary.id for h in hits] == [past.id]
    assert view_at(memory, T).search_summaries("", 3) == []


def test_the_special_facts_and_the_dates_the_view_works_in(memory: Memory) -> None:
    birthday = add_fact(
        memory,
        "她的生日",
        utc(2026, 3, 1),
        event_date=date(2000, 7, 1),
        recurrence="yearly",
        importance=5,
    )
    add_fact(memory, "她喜欢咖啡", utc(2026, 3, 1))
    view = view_at(memory, T)
    assert [f.id for f in view.dated_facts()] == [birthday.id]
    assert [f.id for f in view.core_facts()] == [birthday.id]
    assert view.today() == DAY and view.zone().key == "America/Chicago"
    assert view.vector_bound == int(T.timestamp())
    # 01:00 UTC on the 11th is still the evening of the 10th where she is
    assert view_at(memory, utc(2026, 3, 11, 1)).today() == DAY
    assert view.fact(birthday.id) is not None and view.fact("missing") is None
    assert view.summary("missing") is None
    assert view.summary_for("real", date(2026, 3, 1)) is None


def test_a_view_without_a_renderer_cannot_render(memory: Memory) -> None:
    bare = MemoryView(memory, T)
    with pytest.raises(RuntimeError, match="renderer"):
        bare.render(MemoryQuery("火锅"))
