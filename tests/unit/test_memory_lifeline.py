"""The life line and the follow-ups: the data layer of rounds 08 and 10 (R-MEM-005, R-MEM-006)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.support.embedding import HashingBackend
from tests.support.memory import add_fact, add_followup, make_memory, utc
from tests.support.synth_chat import ChatSpec, build_chat
from twin.memory.followups import FollowupStore, same_followup
from twin.memory.lifeline import LifelineStore, PlannedEvent, minutes_of
from twin.memory.memory import Memory
from twin.profile.builder import rebuild
from twin.services import Services

DAY = date(2026, 3, 10)  # a Tuesday


@pytest.fixture
def memory(services: Services, embedder: HashingBackend) -> Memory:
    return make_memory(services)


def plan(*entries: tuple[str, str | None, str | None]) -> list[PlannedEvent]:
    return [
        PlannedEvent(activity, start, end, place="学校", mood="平静")
        for activity, start, end in entries
    ]


# ----------------------------------------------------------------------- life line


def test_a_day_plan_is_stored_replaced_and_read_in_time_order(memory: Memory) -> None:
    store = LifelineStore(memory)
    stored = store.replace_plan(
        DAY, plan(("吃午饭", "12:00", "13:00"), ("去图书馆", "09:00", "11:00"))
    )
    assert [e.source for e in stored] == ["plan", "plan"] and stored[0].place == "学校"
    assert [e.activity for e in store.day(DAY)] == ["去图书馆", "吃午饭"]
    assert memory.era().online_at is not None  # a plan is made by the bot
    improvised = store.add_improvised(
        DAY, PlannedEvent("临时去买奶茶", "15:00", "15:30"), fact_id="F1"
    )
    assert (improvised.source, improvised.fact_id) == ("improvised", "F1")
    store.replace_plan(DAY, plan(("上课", "08:00", "10:00")))  # a new plan: the old plan goes
    activities = [e.activity for e in store.day(DAY)]
    assert activities == ["上课", "临时去买奶茶"]  # what she let slip stays
    assert store.day(date(2026, 3, 11)) == []


def test_entries_without_a_start_come_last_and_recent_days_are_listed_oldest_first(
    memory: Memory,
) -> None:
    store = LifelineStore(memory)
    store.replace_plan(DAY, plan(("晚上追剧", None, None), ("早餐", "08:00", "08:30")))
    store.replace_plan(DAY - timedelta(days=1), plan(("上班", "09:00", "17:00")))
    store.replace_plan(DAY - timedelta(days=5), plan(("很久以前的事", "09:00", "10:00")))
    assert [e.activity for e in store.day(DAY)] == ["早餐", "晚上追剧"]
    recent = store.recent(DAY, days=3)
    assert (
        list(recent) == [DAY - timedelta(days=1)]
        and recent[DAY - timedelta(days=1)][0].activity == "上班"
    )
    assert list(store.recent(DAY, days=7)) == [DAY - timedelta(days=5), DAY - timedelta(days=1)]


def test_what_she_is_doing_at_a_moment(memory: Memory) -> None:
    store = LifelineStore(memory)
    store.replace_plan(DAY, plan(("去图书馆", "09:00", "11:00"), ("吃午饭", "12:00", "13:00")))

    def at(hour: int, minute: int = 0) -> datetime:
        local = datetime(2026, 3, 10, hour, minute, tzinfo=memory.clock.bot_zone())
        return local.astimezone(UTC)

    found = store.at(at(10, 30))
    assert found is not None and found.activity == "去图书馆"
    assert store.at(at(11)) is None  # the end is not part of the span
    assert store.at(at(11, 30)) is None
    assert store.at(at(12)) is not None


def test_an_invalidated_entry_is_gone_from_the_day(memory: Memory) -> None:
    store = LifelineStore(memory)
    (entry,) = store.replace_plan(DAY, plan(("去爬山", "09:00", "12:00")))
    assert store.invalidate(entry.id, by_fact="F9") and not store.invalidate(entry.id)
    assert store.day(DAY) == []
    stored = memory.store.event(entry.id)
    assert stored is not None and (stored.status, stored.invalidated_by) == ("invalidated", "F9")


# ------------------------------------------------------------------ consistency


def test_a_consistent_day_passes_and_its_entries_are_stamped(memory: Memory) -> None:
    store = LifelineStore(memory)
    store.replace_plan(
        DAY,
        plan(
            ("去图书馆", "09:00", "11:00"), ("吃午饭", "11:00", "12:00"), ("睡前刷手机", None, None)
        ),
    )
    report = store.check_consistency(DAY)
    assert report.ok and report.checked == 3 and report.day == DAY
    assert all(e.consistency_checked_at is not None for e in store.day(DAY))


def test_overlaps_unreadable_times_and_backwards_spans_are_found(memory: Memory) -> None:
    store = LifelineStore(memory)
    entries = store.replace_plan(
        DAY,
        plan(
            ("上课", "09:00", "11:00"),
            ("开会", "10:30", "11:30"),
            ("发呆", "9:7x", "12:00"),
            ("倒着的一段", "15:00", "14:00"),
            ("没有结束", "16:00", None),
        ),
    )
    ids = {e.activity: e.id for e in entries}
    report = store.check_consistency(DAY)
    issues = {(i.kind, i.event_id, i.other_id) for i in report.issues}
    assert ("overlap", ids["开会"], ids["上课"]) in issues
    assert ("bad_time", ids["发呆"], None) in issues
    assert ("outside_day", ids["倒着的一段"], None) in issues
    assert not report.ok and len(report.issues) == 3


def test_an_entry_in_the_middle_of_her_sleep_is_found_unless_it_is_sleep(
    services: Services, memory: Memory
) -> None:
    build_chat(services, ChatSpec())  # she sleeps from 01:00 to 08:30 on every day
    rebuild(services, "all", reason="test")
    store = LifelineStore(memory)
    entries = store.replace_plan(
        date(2026, 8, 18),  # a Tuesday
        plan(
            ("凌晨三点逛街", "02:30", "04:00"),
            ("补觉", "02:30", "04:00"),
            ("吃午饭", "12:00", "13:00"),
        ),
    )
    report = store.check_consistency(date(2026, 8, 18))
    flagged = {(i.kind, i.event_id) for i in report.issues if i.kind == "during_sleep"}
    assert flagged == {("during_sleep", entries[0].id)}
    assert {i.kind for i in report.issues} == {
        "during_sleep",
        "overlap",
    }  # the two night entries overlap


def test_the_generator_is_given_the_last_days_the_real_facts_of_the_day_and_her_sleep(
    services: Services, memory: Memory
) -> None:
    build_chat(services, ChatSpec())
    rebuild(services, "all", reason="test")
    store = LifelineStore(memory)
    store.replace_plan(date(2026, 8, 17), plan(("上班", "09:00", "17:00")))
    birthday = add_fact(
        memory, "对方的生日", utc(2026, 1, 1), subject="user", category="anniversary",
        event_date=date(1999, 8, 18), recurrence="yearly", importance=5,
    )  # fmt: skip
    add_fact(memory, "她喜欢咖啡", utc(2026, 1, 1))
    add_fact(
        memory, "她那天去爬山", utc(2026, 8, 1), source="bot_invented", event_date=date(2026, 8, 18)
    )
    context = store.context_for(date(2026, 8, 18))
    assert context.previous_days == ((date(2026, 8, 17), ("09:00-17:00 上班（学校，平静）",)),)
    assert [f.id for f in context.facts_for_the_day] == [birthday.id]  # real facts only
    assert (
        context.sleep is not None
        and context.sleep[0] < "04:00"
        and context.sleep[1].startswith("08")
    )


def test_without_a_routine_nothing_is_said_about_sleep(memory: Memory) -> None:
    store = LifelineStore(memory)
    store.replace_plan(DAY, plan(("凌晨三点逛街", "02:30", "04:00")))
    assert store.check_consistency(DAY).ok
    assert store.context_for(DAY).sleep is None


def test_clock_times_are_read_strictly() -> None:
    assert [minutes_of(t) for t in ("00:00", "9:05", "23:59", "24:00", "9:7", "ab", "", None)] == [
        0,
        545,
        1439,
        None,
        None,
        None,
        None,
        None,
    ]


# ---------------------------------------------------------------------- follow-ups


def test_a_follow_up_is_stored_once_and_found_when_it_is_due(memory: Memory) -> None:
    store = FollowupStore(memory)
    due = utc(2026, 3, 11, 21)
    follow, created = store.add(
        "她周三晚上面试", due, window_minutes=120, created_at=utc(2026, 3, 9)
    )
    assert created and follow.is_open and follow.window_end == due + timedelta(minutes=120)
    same, again = store.add(
        "她周三晚上面试了", due + timedelta(minutes=5), created_at=utc(2026, 3, 10)
    )
    assert not again and same.id == follow.id  # the same commitment, said twice
    other, fresh = store.add("她周四去看牙医", due + timedelta(days=1), created_at=utc(2026, 3, 9))
    assert fresh and other.id != follow.id
    assert [f.id for f in store.open()] == [follow.id, other.id]
    assert store.due(due - timedelta(minutes=1)) == []
    assert [f.id for f in store.due(due + timedelta(minutes=30))] == [follow.id]
    assert [f.id for f in store.due(due - timedelta(hours=1), lookahead=timedelta(hours=2))] == [
        follow.id
    ]
    assert store.due(due + timedelta(hours=3)) == []  # the window has passed
    assert store.get(follow.id) == store.open()[0]
    assert store.get("missing") is None


def test_closing_remembers_when_and_why_and_works_once(memory: Memory) -> None:
    store = FollowupStore(memory)
    follow, _ = store.add("她周三面试", utc(2026, 3, 11, 21), created_at=utc(2026, 3, 9))
    closed = store.close(follow.id, status="done", reason="mentioned", at=utc(2026, 3, 11, 23))
    assert closed is not None and (closed.status, closed.closed_at, closed.close_reason) == (
        "done",
        utc(2026, 3, 11, 23),
        "mentioned",
    )
    assert store.close(follow.id) is None and store.open() == []
    cancelled, _ = store.add("她周五聚餐", utc(2026, 3, 13, 23), created_at=utc(2026, 3, 9))
    assert store.close(cancelled.id, status="cancelled") is not None


def test_follow_ups_nobody_raised_expire_at_the_end_of_their_window(memory: Memory) -> None:
    store = FollowupStore(memory)
    old, _ = store.add(
        "她周一交论文", utc(2026, 3, 9, 20), window_minutes=120, created_at=utc(2026, 3, 8)
    )
    live, _ = store.add("她周四看牙医", utc(2026, 3, 12, 20), created_at=utc(2026, 3, 8))
    assert store.expire_overdue(utc(2026, 3, 10)) == 1
    expired = memory.store.followup(old.id)
    assert expired is not None and (expired.status, expired.closed_at) == (
        "expired",
        utc(2026, 3, 9, 22),
    )
    assert [f.id for f in store.open()] == [live.id]
    assert store.expire_overdue(utc(2026, 3, 10)) == 0


def test_the_same_commitment_needs_about_the_same_time_and_the_same_words(memory: Memory) -> None:
    follow = add_followup(memory, "她周三面试", utc(2026, 3, 11, 21), utc(2026, 3, 9))
    assert same_followup("她周三面试", utc(2026, 3, 11, 21, 8), follow)
    assert not same_followup("她周三面试", utc(2026, 3, 11, 22), follow)  # an hour apart
    assert not same_followup("对方周末去旅行", utc(2026, 3, 11, 21), follow)
    assert same_followup(
        "a", utc(2026, 3, 11, 21), add_followup(memory, "a", utc(2026, 3, 11, 21), utc(2026, 3, 9))
    )
