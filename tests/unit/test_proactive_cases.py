"""Cases that cross several parts: chasing, the platform count, interruptions, a new time zone."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from tests.support.proactive_world import World
from twin.schedule.proactive.types import REASON_LABELS, Reason, TriggerKind

FRIDAY = date(2026, 10, 9)
SHANGHAI = "Asia/Shanghai"


def deep_sleep_moment(world: World, *, night: bool) -> Any:
    plan = world.rig.planner.ensure(FRIDAY)
    episode = plan.night if night else plan.morning
    assert episode is not None
    start, end = next((a, b) for kind, a, b in episode.intervals() if kind == "deep_sleep")
    return start + (end - start) / 2


# ------------------------------------------------------------------------------ chasing


async def test_a_chase_is_told_what_was_said_before_and_says_something_else(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.put(TriggerKind.SHARE, calm.at(9, 0))
    assert (await calm.tick_at(calm.at(9, 5))).outcome == "sent"
    calm.script.messages["silence"] = ("你怎么不理我", "哼")
    calm.put(TriggerKind.SILENCE, calm.at(11, 0))
    report = await calm.tick_at(calm.at(11, 5))
    assert report.outcome == "sent", report
    seen = str(calm.script.requests[-1]["messages"][-1]["content"])
    assert "【上一条主动消息没有回" in seen
    assert "- 刚刚在图书馆看文献" in seen and "- 有点困" in seen
    second = calm.sent_rows()[-1]
    assert second.kind == "silence" and second.chase_seq == 1
    assert calm.channel.texts[-2:] == ["你怎么不理我", "哼"]


async def test_a_chase_that_repeats_the_unanswered_message_is_not_sent(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.put(TriggerKind.SHARE, calm.at(9, 0))
    await calm.tick_at(calm.at(9, 5))
    calm.script.messages["silence"] = ("刚刚在图书馆看文献", "有点困")
    calm.put(TriggerKind.SILENCE, calm.at(11, 0))
    report = await calm.tick_at(calm.at(11, 5))
    assert report.outcome in ("failed", "declined"), report
    assert report.reason == "generation_failed"
    assert len(calm.sent_rows()) == 1
    entry = calm.log.entries(outcomes=["failed"])[0]
    assert entry.result == {"detail": "repeat_previous"}


async def test_once_the_user_answers_the_next_message_is_not_a_chase(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.put(TriggerKind.SHARE, calm.at(9, 0))
    await calm.tick_at(calm.at(9, 5))
    calm.user_writes(at=calm.at(9, 40))
    calm.put(TriggerKind.SHARE, calm.at(11, 0))
    await calm.tick_at(calm.at(11, 5))
    assert [row.chase_seq for row in calm.sent_rows()] == [0, 0]
    seen = str(calm.script.requests[-1]["messages"][-1]["content"])
    assert "【上一条主动消息没有回" not in seen


# ------------------------------------------------------------------ the platform's count


async def test_a_message_longer_than_the_count_allows_is_fitted_into_it(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    window = calm.channel.window
    window.on_outbound(window.remaining_quota() - 3)  # three left, one of them kept for a chase
    calm.script.messages["share"] = ("一二", "三四", "五六", "七八")
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "sent" and report.sent <= 2, report
    assert window.remaining_quota() >= 1  # the chase to come still has a bubble
    entry = calm.sent_rows()[0]
    assert entry.bubbles_sent == report.sent
    assert "".join(calm.channel.texts[-report.sent :]).replace(" ", "").startswith("一二")


async def test_with_one_bubble_left_the_whole_message_is_one_bubble(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    window = calm.channel.window
    window.on_outbound(window.remaining_quota() - 1)
    calm.script.messages["share"] = ("一二", "三四", "五六")
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "sent" and report.sent == 1, report
    assert calm.channel.texts[-1].replace(" ", "") == "一二三四五六"


async def test_with_no_bubble_left_nothing_is_sent_but_it_is_logged(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.channel.window.on_outbound(calm.channel.window.remaining_quota())
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "rejected" and report.reason == "quota_exhausted"
    assert calm.channel.out == [] and calm.script.requests == []
    assert calm.reasons() == ["quota_exhausted"]


async def test_a_count_that_ran_out_during_the_planning_is_found_by_the_second_check(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    window = calm.channel.window
    calm.script.on_request = lambda _kind: window.on_outbound(window.remaining_quota())
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "rejected" and report.reason == "quota_exhausted", report
    assert calm.channel.out == [] and calm.sent_rows() == []


# ----------------------------------------------------------------------- the user comes


async def test_a_message_of_the_user_during_the_planning_drops_the_proactive_one(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    row = calm.put(TriggerKind.SHARE, calm.at(10, 0))
    calm.script.on_request = lambda _kind: calm.talk.user_wrote()
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "dropped" and report.reason == "user_active", report
    assert calm.channel.out == []
    assert calm.log.entries(outcomes=["dropped"])[0].reason == "user_active"
    assert calm.candidates.get(row.id).status == "dropped"  # type: ignore[union-attr]


async def test_the_engine_starting_a_round_during_the_planning_drops_it_too(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    calm.script.on_request = lambda _kind: calm.talk.busy()
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "dropped" and report.reason == "user_active", report
    assert calm.channel.out == []


async def test_a_user_who_writes_between_two_bubbles_is_answered_by_the_engine_not_by_us(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    seen: list[str] = []

    def after_first(item: Any) -> None:
        if item.kind == "text":
            seen.append(item.text)
            calm.talk.user_wrote()

    calm.channel.on_send = after_first
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "sent" and report.sent == 1, report
    assert calm.channel.texts == ["刚刚在图书馆看文献"]  # the rest was dropped
    entry = calm.sent_rows()[0]
    assert entry.bubbles_sent == 1 and entry.result is not None
    assert entry.result["interrupted_by"] == "user" and entry.result["planned"] == 2
    assert entry.content is not None and entry.content["bubbles"] == ["刚刚在图书馆看文献"]


async def test_she_falls_asleep_between_two_bubbles_and_the_rest_is_dropped(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0))
    planned = calm.at(23, 0)
    calm.put(TriggerKind.BEDTIME, planned, minutes=20)
    deep = deep_sleep_moment(calm, night=True)
    calm.channel.on_send = lambda item: calm.clock.set_time(deep) if item.kind == "text" else None
    report = await calm.tick_at(planned + timedelta(minutes=5))
    assert report.outcome == "sent" and report.sent == 1, report
    assert calm.channel.texts == ["我先睡啦"]
    entry = calm.sent_rows()[0]
    assert entry.result is not None and entry.result["interrupted_by"] == "sleep"


async def test_the_channel_refusing_the_first_bubble_is_logged_as_not_sent(calm: World) -> None:
    from twin.channel.base import OutboundKind, OutboundResult

    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.channel.fail_with = OutboundResult.failure(OutboundKind.WINDOW_REJECTED, "window_elapsed")
    row = calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "rejected" and report.reason in ("window_closed", "send_failed")
    assert calm.sent_rows() == []
    assert calm.candidates.get(row.id).status == "dropped"  # type: ignore[union-attr]


# ------------------------------------------------------------------ other time zones


async def test_a_switch_to_shanghai_voids_the_candidates_and_never_greets_twice(
    full: World,
) -> None:
    """Chicago to Shanghai at noon: the old day's candidates are void, no second greeting comes
    in the Shanghai night and morning, and nothing is sent while she sleeps there."""
    full.user_writes(at=full.at(20, 0, day=8))
    await full.tick_at(full.at(0, 10))
    greeting = next(c for c in full.candidates.pending() if c.kind is TriggerKind.GREETING)
    assert (await full.tick_at(greeting.planned_at + timedelta(minutes=1))).outcome == "sent"
    full.user_writes(at=full.at(12, 5))
    full.go_to(full.at(12, 10))
    after, last = full.at(12, 15), full.at(23, 59)  # read on Chicago's clock, before the switch
    waiting = [c.key for c in full.candidates.pending()]
    assert waiting, "the Chicago day should still hold candidates"
    switched = full.rig.planner.switch_timezone(SHANGHAI)
    assert switched.changed
    await full.publish(switched.events)
    assert full.candidates.pending() == [] or all(
        c.timezone == SHANGHAI for c in full.candidates.pending()
    )
    voided = [r for r in full.log.entries(outcomes=["expired"]) if r.reason == "timezone_switch"]
    assert sorted(r.kind for r in voided) == sorted(
        k.split(":")[0] if k != "bedtime" else "bedtime" for k in waiting
    )
    assert REASON_LABELS[Reason.TIMEZONE_SWITCH] == "切换时区后作废"
    moment = after
    while moment < last:  # until Saturday noon in Shanghai
        await full.tick_at(moment)
        moment += timedelta(minutes=5)
    time = full.rig.kit.time
    sent = full.sent_rows()
    assert [row.kind for row in sent].count("greeting") == 1
    for row in sent:
        assert str(time.her_state(row.at).kind) != "deep_sleep", row
        assert row.her_state != "deep_sleep"
    shanghai = full.candidates.for_day(date(2026, 10, 10), SHANGHAI)
    assert shanghai and "greeting" not in {c.kind.value for c in shanghai}
    assert len(full.log.entries(outcomes=["opened"])) == 2  # one per local day and zone


async def test_a_switch_back_and_forth_does_not_send_a_moment_twice(full: World) -> None:
    full.user_writes(at=full.at(20, 0, day=8))
    await full.tick_at(full.at(0, 10))
    full.user_writes(at=full.at(9, 0))
    full.go_to(full.at(9, 30))
    after, last = full.at(9, 40), full.at(23, 0)
    for name in (SHANGHAI, "America/Chicago"):
        outcome = full.rig.planner.switch_timezone(name)
        await full.publish(outcome.events)
    keys = [
        (c.local_date, c.timezone, c.key)
        for c in full.candidates.for_day(FRIDAY, "America/Chicago")
    ]
    assert len(keys) == len(set(keys))
    moment = after
    while moment < last:
        await full.tick_at(moment)
        moment += timedelta(minutes=5)
    meals = [row.kind for row in full.sent_rows() if row.kind == "meal"]
    assert len(meals) <= 3
