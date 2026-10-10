"""The scheduler's days: who is considered when, what the planner may say, what is kept.

Real scheduler, real engine parts, a manual clock; the planner is a respx fake of DeepSeek.
``calm`` never draws (only the candidates a test puts), ``eager`` draws at the first chance.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from tests.support.proactive_world import World
from twin.memory.lifeline import PlannedEvent
from twin.schedule.events import CandidatesExpired
from twin.schedule.proactive.types import REASON_LABELS, Reason, TriggerKind

FRIDAY = date(2026, 10, 9)


def kinds_of(world: World, outcome: str = "sent") -> list[str]:
    return [row.kind for row in world.log.entries(outcomes=[outcome])]


# ------------------------------------------------------------------ while the user talks


async def test_nothing_is_considered_while_the_engine_is_in_the_middle_of_a_round(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    calm.talk.busy()
    report = await calm.tick_at(calm.at(10, 5))
    assert report.skipped == "user_active" and report.candidate is None
    assert calm.log.entries(outcomes=["sent", "rejected", "declined"]) == []
    assert calm.script.requests == []
    calm.talk.idle()
    assert (await calm.tick_at(calm.at(10, 10))).outcome == "sent"


async def test_nothing_is_considered_within_ten_minutes_of_an_exchange(calm: World) -> None:
    calm.user_writes(at=calm.at(9, 55))  # his message and her answer half a minute later
    calm.put(TriggerKind.SHARE, calm.at(9, 50), minutes=90)
    for minute in (0, 2, 5):
        report = await calm.tick_at(calm.at(10, minute))
        assert report.skipped == "user_active", minute
    assert calm.channel.out == []
    report = await calm.tick_at(calm.at(10, 7))  # more than ten minutes after her answer
    assert report.outcome == "sent", report


async def test_nothing_is_drawn_while_the_user_is_chatting(eager: World) -> None:
    eager.user_writes(at=eager.at(9, 55))
    for minute in range(0, 10, 5):
        report = await eager.tick_at(eager.at(10, minute))
        assert report.skipped == "user_active" and not report.drawn
    assert eager.script.requests == [] and eager.channel.out == []


# --------------------------------------------------------------------- the window rule


async def test_after_a_day_of_silence_the_window_suppresses_and_a_message_of_his_reopens_it(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(11, 0, day=8))  # 23 hours before the candidate
    row = calm.put(TriggerKind.SHARE, calm.at(10, 0), minutes=120)
    assert (await calm.tick_at(calm.at(10, 5))).reason == "window_closed"
    assert calm.channel.out == [] and calm.script.requests == []
    assert (await calm.tick_at(calm.at(10, 30))).outcome == "rejected"
    assert [r.reason for r in calm.rejected()] == ["window_closed"]  # once, not at every tick
    calm.user_writes(at=calm.at(10, 40))
    assert (await calm.tick_at(calm.at(10, 45))).skipped == "user_active"
    report = await calm.tick_at(calm.at(10, 55))
    assert report.outcome == "sent", report
    assert calm.candidates.get(row.id).status == "sent"  # type: ignore[union-attr]
    assert calm.channel.texts[-2:] == ["刚刚在图书馆看文献", "有点困"]


# ----------------------------------------------------------- restart, wake-up, late start


@pytest.mark.parametrize("reason", ["startup", "wake"])
async def test_candidates_that_passed_while_the_program_was_off_are_void_and_never_sent(
    calm: World, reason: str
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    await calm.tick_at(calm.at(0, 10))  # the day is laid out: the greeting waits
    greeting = [c for c in calm.candidates.pending() if c.kind is TriggerKind.GREETING]
    assert len(greeting) == 1
    before = calm.scheduler.epoch
    calm.clock.set_time(calm.at(10, 0))  # the machine was off (or asleep) until ten
    await calm.publish(
        (CandidatesExpired(at=calm.clock.now_utc(), cutoff=calm.clock.now_utc(), reason=reason),)  # type: ignore[arg-type]
    )
    assert calm.scheduler.epoch == before + 1
    assert calm.candidates.get(greeting[0].id).status == "expired"  # type: ignore[union-attr]
    voided = calm.log.entries(outcomes=["expired"])
    assert [(r.kind, r.reason) for r in voided] == [("greeting", "interrupted")]
    assert REASON_LABELS[Reason.INTERRUPTED] == "中断期间过期"
    moment = calm.at(10, 0)
    while moment < calm.at(20, 0):
        await calm.tick_at(moment)
        moment += timedelta(minutes=5)
    assert calm.channel.out == [] and calm.sent_rows() == []  # nothing is made up, ever
    assert calm.script.requests == []


async def test_a_future_candidate_survives_a_restart(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    keep = calm.put(TriggerKind.SHARE, calm.at(15, 0))
    lose = calm.put(TriggerKind.SHARE, calm.at(9, 0), minutes=300)
    calm.clock.set_time(calm.at(10, 0))
    await calm.publish(
        (CandidatesExpired(at=calm.at(10, 0), cutoff=calm.at(10, 0), reason="startup"),)
    )
    assert calm.candidates.get(lose.id).status == "expired"  # type: ignore[union-attr]
    assert calm.candidates.get(keep.id).status == "pending"  # type: ignore[union-attr]
    assert (await calm.tick_at(calm.at(15, 5))).outcome == "sent"


async def test_moments_that_passed_before_the_first_tick_of_the_day_are_not_triggered(
    full: World,
) -> None:
    """The program starts at two in the afternoon: breakfast and lunch are over, dinner is not."""
    full.user_writes(at=full.at(20, 0, day=8))
    report = await full.tick_at(full.at(14, 0))
    assert report.outcome is None
    rows = {r.key: r for r in full.candidates.for_day(FRIDAY, "America/Chicago")}
    # she was awake long before two: the plan allows no greeting for the day at all
    assert set(rows) == {"meal:breakfast", "meal:lunch", "meal:dinner", "bedtime"}
    assert [rows[k].status for k in ("meal:breakfast", "meal:lunch")] == ["expired", "expired"]
    assert [rows[k].status for k in ("meal:dinner", "bedtime")] == ["pending", "pending"]
    expired = full.log.entries(outcomes=["expired"])
    assert sorted(r.reason or "" for r in expired) == ["window_over", "window_over"]
    assert full.channel.out == [] and full.script.requests == []
    await full.tick_at(full.at(14, 5))
    assert full.sent_rows() == []


async def test_the_greeting_goes_out_when_the_program_runs_through_its_window(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    await calm.tick_at(calm.at(0, 10))
    greeting = next(c for c in calm.candidates.pending() if c.kind is TriggerKind.GREETING)
    assert (await calm.tick_at(greeting.planned_at + timedelta(minutes=1))).outcome == "sent"
    assert kinds_of(calm) == ["greeting"]
    assert calm.rig.planner.greeting_decision(
        calm.clock.now_utc() + timedelta(hours=1)
    ).allowed is (False)  # the eighteen hours begin: no second greeting soon


async def test_a_second_greeting_inside_eighteen_hours_is_refused(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.rig.planner.record_wake_greeting(calm.at(7, 0))
    calm.put(TriggerKind.GREETING, calm.at(9, 0))
    report = await calm.tick_at(calm.at(9, 5))
    assert report.reason == "greeting_gap", report
    assert calm.channel.out == []


# ------------------------------------------------------------------------- follow-ups


def add_followup(world: World, due_hour: int = 11, window: int = 240) -> Any:
    record, created = world.followups.add(
        "周五的考试",
        world.at(due_hour, 0),
        window_minutes=window,
        created_at=world.at(20, 0, day=8),
    )
    assert created
    return record


def waiting_followup(world: World) -> Any:
    found = [c for c in world.candidates.pending() if c.kind is TriggerKind.FOLLOWUP]
    assert len(found) == 1
    return found[0]


async def test_a_followup_that_came_due_is_asked_a_little_later_and_then_closed(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    record = add_followup(calm)
    await calm.tick_at(calm.at(10, 30))
    assert not [c for c in calm.candidates.pending() if c.kind is TriggerKind.FOLLOWUP]
    await calm.tick_at(calm.at(11, 1))
    slot = waiting_followup(calm)
    assert calm.at(11, 20) <= slot.planned_at <= calm.at(12, 40)
    assert calm.channel.out == []
    report = await calm.tick_at(slot.planned_at + timedelta(minutes=1))
    assert report.outcome == "sent" and report.candidate is not None, report
    assert report.candidate.kind is TriggerKind.FOLLOWUP
    assert calm.channel.texts == ["考试怎么样了"]
    assert "周五的考试" in str(calm.script.requests[-1]["messages"][-1]["content"])
    closed = calm.followups.get(record.id)
    assert closed is not None and closed.status == "done" and closed.close_reason == "asked"
    row = calm.sent_rows()[0]
    assert row.kind == "followup" and row.followup_id == record.id
    await calm.tick_at(slot.planned_at + timedelta(hours=3))  # never asked twice
    assert len(calm.sent_rows()) == 1


async def test_a_followup_the_user_raised_himself_is_closed_without_a_message(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    record = add_followup(calm)

    def planner(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "send": False,
            "kind": kind,
            "messages": [],
            "reason": "他刚才已经说过考完了",
            "followup_done": True,
        }

    calm.script.by_hand = planner
    await calm.tick_at(calm.at(11, 1))
    slot = waiting_followup(calm)
    report = await calm.tick_at(slot.planned_at + timedelta(minutes=1))
    assert report.outcome == "declined", report
    closed = calm.followups.get(record.id)
    assert closed is not None and closed.status == "done"
    assert closed.close_reason == "mentioned_before"
    assert calm.candidates.get(slot.id).status == "dropped"  # type: ignore[union-attr]
    assert calm.candidates.get(slot.id).last_reason == "followup_closed"  # type: ignore[union-attr]
    assert calm.channel.out == []
    n = len(calm.script.requests)
    await calm.tick_at(slot.planned_at + timedelta(hours=1))
    assert len(calm.script.requests) == n  # not asked again


async def test_a_followup_whose_window_passed_is_dropped_not_asked(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    record = add_followup(calm, due_hour=9, window=60)
    await calm.tick_at(calm.at(12, 0))  # the program came back after the window
    assert calm.channel.out == [] and calm.script.requests == []
    closed = calm.followups.get(record.id)
    assert closed is not None and closed.status == "expired"


async def test_a_followup_is_considered_before_a_routine_message_and_before_sharing(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    add_followup(calm, due_hour=10)
    calm.put(TriggerKind.SHARE, calm.at(11, 0), minutes=600)
    calm.put(TriggerKind.MEAL, calm.at(11, 0), minutes=600, detail={"meal": "lunch"})
    await calm.tick_at(calm.at(10, 1))
    slot = waiting_followup(calm)
    order: list[str] = []
    moment = max(slot.planned_at, calm.at(11, 5))
    for _ in range(3):
        report = await calm.tick_at(moment)
        order.append(report.candidate.kind.value if report.candidate else "-")
        moment += timedelta(minutes=65)
    assert order == ["followup", "meal", "share"]


# ---------------------------------------------------- what the planner decides: send=false


async def test_a_routine_message_the_planner_declines_is_planned_once_more_then_given_up(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.script.decline = {"meal"}
    row = calm.put(TriggerKind.MEAL, calm.at(10, 0), minutes=180, detail={"meal": "lunch"})
    first = await calm.tick_at(calm.at(10, 5))
    assert first.outcome == "declined" and first.reason == "planner_declined", first
    again = calm.candidates.get(row.id)
    assert again is not None and again.status == "pending" and again.attempts == 1
    assert again.planned_at > calm.at(10, 5)
    assert (await calm.tick_at(calm.at(10, 10))).candidate is None  # it waits for its new time
    second = await calm.tick_at(again.planned_at + timedelta(minutes=1))
    assert second.outcome == "declined"
    final = calm.candidates.get(row.id)
    assert final is not None and final.status == "declined"
    assert len(calm.script.requests) == 2
    assert calm.reasons("declined") == ["planner_declined", "planner_declined"]
    await calm.tick_at(calm.at(12, 0))
    assert len(calm.script.requests) == 2 and calm.channel.out == []


async def test_a_drawn_message_the_planner_declines_is_given_up_and_the_draw_rests(
    eager: World,
) -> None:
    eager.user_writes(at=eager.at(20, 0, day=8))
    eager.script.decline = {"silence"}
    report = await eager.tick_at(eager.at(10, 0))
    assert report.drawn and report.outcome == "declined", report
    assert eager.candidates.pending() == []  # a drawn candidate leaves nothing behind
    for minute in range(5, 30, 5):
        quiet = await eager.tick_at(eager.at(10, minute))
        assert quiet.candidate is None
    assert len(eager.script.requests) == 1
    later = await eager.tick_at(eager.at(10, 35))  # the rest is over: a new draw
    assert later.drawn and len(eager.script.requests) == 2


async def test_a_planner_that_cannot_be_reached_sends_nothing_and_says_why(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.script.unavailable = True
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "failed" and report.reason == "planner_failed", report
    assert calm.channel.out == []
    assert calm.reasons("failed") == ["planner_failed"]


async def test_a_planner_that_answers_with_something_else_than_json_sends_nothing(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.script.invalid_first = 5
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "failed" and report.reason == "planner_failed", report
    assert calm.channel.out == [] and calm.sent_rows() == []
    entry = calm.log.entries(outcomes=["failed"])[0]
    assert entry.result == {"detail": "invalid_json"}


# ---------------------------------------------------------------- the random side of a day


async def test_after_a_long_silence_she_breaks_it_and_after_a_short_one_she_shares(
    eager: World,
) -> None:
    eager.user_writes(at=eager.at(9, 0))  # an hour before the first draw
    first = await eager.tick_at(eager.at(10, 5))
    assert first.drawn and first.candidate is not None
    assert first.candidate.kind is TriggerKind.SHARE
    later = await eager.tick_at(eager.at(14, 30))  # four and a half hours after her message
    assert later.drawn and later.candidate is not None
    assert later.candidate.kind is TriggerKind.SILENCE
    assert kinds_of(eager) == ["share", "silence"]


async def test_the_days_quota_is_spent_exactly_and_never_in_deep_sleep(eager: World) -> None:
    """Two a day, drawn at the first chances: the greeting counts, nothing goes out at night."""
    eager.user_writes(at=eager.at(20, 0, day=8))
    moment = eager.at(0, 10)
    while moment < eager.at(18, 0):
        await eager.tick_at(moment)
        moment += timedelta(minutes=5)
    sent = eager.sent_rows()
    assert [row.kind for row in sent] == ["greeting", "share"]
    assert {row.her_state for row in sent} <= {"free", "busy"}  # awake, never asleep
    assert sent[1].at - sent[0].at >= timedelta(minutes=60)
    assert sent[1].chase_seq == 1  # the greeting had no answer, so the second one is a chase


# ------------------------------------------------------------------- the edge of her sleep


def edge_start(world: World) -> Any:
    plan = world.rig.planner.ensure(FRIDAY)
    assert plan.night is not None
    return next(a for kind, a, _ in plan.night.intervals() if kind == "sleep_edge")


async def test_at_the_edge_of_her_sleep_she_may_write_once_and_then_the_draw_rests(
    eager_night: World,
) -> None:
    world = eager_night
    world.user_writes(at=world.at(20, 0))
    start = edge_start(world)
    moment = start + timedelta(minutes=2)
    for _ in range(5):
        await world.tick_at(moment)
        moment += timedelta(minutes=5)
    assert kinds_of(world) == ["edge"] and world.channel.texts == ["睡不着"]
    assert world.sent_rows()[0].her_state == "sleep_edge"
    assert world.reasons() == ["spacing"]  # asked again once, refused, and then it rests
    assert world.log.entries(outcomes=["rejected"])[0].kind == "edge"


async def test_the_weekly_allowance_of_edge_messages_stops_the_draw(eager_night: World) -> None:
    world = eager_night
    world.user_writes(at=world.at(20, 0))
    start = edge_start(world)
    world.add_log(start - timedelta(days=3), kind="edge", state="sleep_edge")
    world.add_log(start - timedelta(days=1), kind="edge", state="sleep_edge")
    moment = start + timedelta(minutes=2)
    for _ in range(5):
        await world.tick_at(moment)
        moment += timedelta(minutes=5)
    assert world.channel.out == [] and world.script.requests == []
    assert world.reasons() == ["edge_weekly"]


async def test_the_edge_of_sleep_never_brings_the_kinds_of_the_awake_day(
    eager_night: World,
) -> None:
    world = eager_night
    world.user_writes(at=world.at(20, 0))
    world.add_log(edge_start(world) - timedelta(days=1), kind="edge", state="sleep_edge")
    start = edge_start(world)
    moment = start
    plan = world.rig.planner.ensure(FRIDAY)
    assert plan.night is not None
    while moment < plan.night.wake:
        await world.tick_at(moment)
        moment += timedelta(minutes=5)
    assert {row.kind for row in world.sent_rows()} <= {"edge"}
    assert all(row.her_state == "sleep_edge" for row in world.sent_rows())


# ------------------------------------------------------------------------- the books


async def test_a_sent_message_is_written_to_the_log_the_conversation_and_the_candidate(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    row = calm.put(TriggerKind.MEAL, calm.at(11, 40), detail={"meal": "lunch"})
    assert (await calm.tick_at(calm.at(11, 45))).outcome == "sent"
    entry = calm.sent_rows()[0]
    assert (entry.kind, entry.outcome, entry.her_state) == ("meal", "sent", "free")
    assert entry.local_at == "2026-10-09 11:45" or entry.local_at.startswith("2026-10-09 11:")
    assert entry.timezone == "America/Chicago" and entry.chase_seq == 0
    assert entry.bubbles_sent == 1 and entry.backend == "deepseek"
    assert entry.plan_reason == "meal 的理由"
    assert entry.content is not None and entry.content["bubbles"] == ["吃饭了吗"]
    assert entry.candidate_id == row.id and entry.cost_usd and entry.cost_usd > 0
    assert (entry.range_min, entry.range_max, entry.quota_total) == (1, 6, 6)
    turns = calm.turns.reply(entry.reply_id)
    assert [t.text for t in turns] == ["吃饭了吗"] and turns[0].direction == "out"
    assert any(a.get("step") == "proactive" for a in turns[0].actions)
    assert calm.candidates.get(row.id).status == "sent"  # type: ignore[union-attr]
    assert calm.channel.window.outbound_since_inbound >= 2  # the platform's count moved


async def test_the_log_is_open_for_the_day_once(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    for minute in (10, 15, 20):
        await calm.tick_at(calm.at(0, minute))
    opened = calm.log.entries(outcomes=["opened"])
    assert len(opened) == 1 and opened[0].kind == "day"
    assert (opened[0].range_min, opened[0].range_max) == (1, 6)
    assert opened[0].enabled is True and opened[0].quota_total == 6


async def test_what_she_told_of_her_day_is_marked_and_a_made_up_detail_joins_the_life_line(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.lifeline.replace_plan(
        FRIDAY,
        [
            PlannedEvent("在图书馆看文献", "09:00", "11:00", "图书馆"),
            PlannedEvent("去食堂吃饭", "11:30", "12:00", "食堂"),
        ],
    )

    def planner(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "send": True,
            "kind": kind,
            "messages": ["刚从图书馆出来", "买了杯奶茶"],
            "reason": "分享",
            "shares": ["L1"],
            "new_detail": {"activity": "路上买了杯奶茶", "start": "11:10", "end": "11:20"},
        }

    calm.script.by_hand = planner
    calm.put(TriggerKind.SHARE, calm.at(11, 30))
    assert (await calm.tick_at(calm.at(11, 35))).outcome == "sent"
    entries = calm.lifeline.day(FRIDAY)
    told = [e for e in entries if e.shared]
    assert sorted(e.activity for e in told) == ["在图书馆看文献", "路上买了杯奶茶"]
    reply_id = calm.sent_rows()[0].reply_id
    assert {e.shared_reply_id for e in told} == {reply_id}
    assert [e.activity for e in calm.lifeline.unshared(FRIDAY)] == ["去食堂吃饭"]
    # the next planner call sees what was told already
    calm.put(TriggerKind.SHARE, calm.at(13, 0))
    calm.channel.user_writes(calm.at(12, 0))
    calm.turns.add_inbound(at=calm.at(12, 0), kind="text", text="在吗", external_id="u9")
    await calm.tick_at(calm.at(13, 40))
    last = str(calm.script.requests[-1]["messages"][-1]["content"])
    assert "（已经对他说过）" in last
