"""Every hard constraint, one test each, through the real scheduler (R-PRO-003).

A candidate is put into the books by hand, the clock is set, the scheduler ticks.  Each refusal
must be logged with its own reason, must leave nothing in the channel, and must not even ask the
planner (no money is spent on a message that cannot go out).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from tests.support.proactive_world import World
from twin.config.runtime import (
    ENGINE_PAUSED_UNTIL,
    PAUSED,
    PROACTIVE_DAILY_MAX,
    PROACTIVE_ENABLED,
)
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.proactive.scheduler import TickReport
from twin.schedule.proactive.types import REASON_LABELS, Reason, TriggerKind

FRIDAY = date(2026, 10, 9)


def interval(world: World, state: str, *, night: bool, nth: int = 0) -> tuple[datetime, datetime]:
    """The ``nth`` piece of a given state in last night (the morning) or tonight."""
    plan = world.rig.planner.ensure(FRIDAY)
    episode = plan.night if night else plan.morning
    assert episode is not None
    found = [(a, b) for kind, a, b in episode.intervals() if kind == state]
    return found[nth]


def middle(window: tuple[datetime, datetime]) -> datetime:
    return window[0] + (window[1] - window[0]) / 2


async def refused(
    world: World,
    kind: TriggerKind,
    planned: datetime,
    reason: Reason,
    *,
    ahead: int = 5,
    detail: dict[str, object] | None = None,
) -> TickReport:
    """Put ``kind`` at ``planned``, tick ``ahead`` minutes later and check the refusal."""
    world.put(kind, planned, detail=detail)  # type: ignore[arg-type]
    report = await world.tick_at(planned + timedelta(minutes=ahead))
    assert report.outcome == "rejected" and report.reason == reason.value, report
    assert world.channel.out == [], "something went out although a constraint refused it"
    assert world.script.requests == [], "the planner was asked about a message that cannot go out"
    mine = [row for row in world.rejected() if row.kind == kind.value]
    assert [row.reason for row in mine] == [reason.value]
    assert mine[0].candidate_id is not None
    return report


# ---------------------------------------------------------------------------- the control


async def test_with_nothing_in_the_way_the_message_goes_out(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    planned = calm.at(10, 0)
    calm.put(TriggerKind.SHARE, planned)
    report = await calm.tick_at(planned + timedelta(minutes=5))
    assert report.outcome == "sent", report
    assert calm.channel.texts == ["刚刚在图书馆看文献", "有点困"]
    assert calm.rejected() == []
    assert len(calm.script.requests) == 1


# ------------------------------------------------------------------------------ deep sleep


@pytest.mark.parametrize("kind", [k for k in TriggerKind if k is not TriggerKind.EDGE])
async def test_nothing_goes_out_while_she_is_in_deep_sleep(calm: World, kind: TriggerKind) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    deep = middle(interval(calm, "deep_sleep", night=False))
    await refused(calm, kind, deep - timedelta(minutes=5), Reason.DEEP_SLEEP)


async def test_an_edge_message_is_refused_in_deep_sleep_too(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    deep = middle(interval(calm, "deep_sleep", night=False))
    await refused(calm, TriggerKind.EDGE, deep - timedelta(minutes=5), Reason.DEEP_SLEEP)


async def test_no_setting_and_no_condition_reaches_deep_sleep(calm: World) -> None:
    """Every other switch in its most generous position: the refusal is the same."""
    calm.services.runtime.set(PROACTIVE_DAILY_MAX, 99, by="test")
    calm.services.runtime.set(PROACTIVE_ENABLED, True, by="test")
    calm.rig.set_quota(QuotaRange(0, 99))
    calm.user_writes(at=calm.at(20, 0, day=8))
    deep = middle(interval(calm, "deep_sleep", night=False))
    await refused(calm, TriggerKind.FOLLOWUP, deep - timedelta(minutes=5), Reason.DEEP_SLEEP)
    # and again, a long time later in the same sleep: the candidate is still not sent
    again = await calm.tick_at(deep + timedelta(minutes=20))
    assert again.outcome in (None, "rejected")
    assert calm.channel.out == [] and calm.script.requests == []


async def test_the_draw_never_picks_a_moment_of_deep_sleep(calm: World) -> None:
    """Over a whole night of ticks nothing is drawn and nothing is logged as sent."""
    calm.user_writes(at=calm.at(20, 0, day=8))
    start, end = interval(calm, "deep_sleep", night=False)
    moment = start
    while moment < end:
        report = await calm.tick_at(moment)
        assert report.candidate is None, report
        moment += timedelta(minutes=5)
    assert calm.channel.out == [] and calm.sent_rows() == []


# --------------------------------------------------------------------------- sleep's edge


async def test_at_the_edge_of_her_sleep_only_the_edge_message_may_go(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0))
    edge = middle(interval(calm, "sleep_edge", night=True))
    await refused(calm, TriggerKind.MEAL, edge - timedelta(minutes=5), Reason.SLEEP_EDGE)


async def test_the_edge_message_is_limited_to_a_few_a_week(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0))
    edge = middle(interval(calm, "sleep_edge", night=True))
    for days_ago in (5, 2):
        calm.add_log(edge - timedelta(days=days_ago), kind="edge", state="sleep_edge")
    await refused(calm, TriggerKind.EDGE, edge - timedelta(minutes=5), Reason.EDGE_WEEKLY)


async def test_an_edge_message_inside_the_weekly_allowance_goes_out(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0))
    edge = middle(interval(calm, "sleep_edge", night=True))
    calm.add_log(edge - timedelta(days=2), kind="edge", state="sleep_edge")
    calm.put(TriggerKind.EDGE, edge - timedelta(minutes=5), detail={"side": "falling"})
    report = await calm.tick_at(edge)
    assert report.outcome == "sent", report
    assert calm.channel.texts == ["睡不着"]
    sent = calm.sent_rows()[-1]
    assert sent.her_state == "sleep_edge" and sent.kind == "edge"


# ----------------------------------------------------------------------------- the others


async def test_a_pause_stops_everything(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.services.runtime.set(ENGINE_PAUSED_UNTIL, calm.at(18, 0), by="test")
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.PAUSED)


async def test_the_pause_switch_stops_everything_too(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.services.runtime.set(PAUSED, True, by="test")
    await refused(calm, TriggerKind.FOLLOWUP, calm.at(10, 0), Reason.PAUSED)


async def test_switched_off_means_nothing_goes_out_and_nothing_is_drawn(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.services.runtime.set(PROACTIVE_ENABLED, False, by="test")
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.DISABLED)
    for step in range(1, 40):  # a whole afternoon of ticks: no candidate is drawn
        report = await calm.tick_at(calm.at(11, 0) + timedelta(minutes=5 * step))
        assert not report.drawn


async def test_the_window_of_the_platform_suppresses_the_message(calm: World) -> None:
    calm.user_writes(at=calm.at(11, 0, day=8))  # more than 22 hours before
    report = await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.WINDOW_CLOSED)
    assert report.candidate is not None
    entry = calm.rejected()[0]
    assert entry.result == {"detail": "window_elapsed"}
    assert REASON_LABELS[Reason.WINDOW_CLOSED] == "被窗口抑制"


async def test_without_any_message_of_the_user_the_window_never_opened(calm: World) -> None:
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.WINDOW_CLOSED)
    assert calm.rejected()[0].result == {"detail": "no_inbound_yet"}


async def test_the_platform_count_runs_out(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.channel.window.on_outbound(calm.channel.window.remaining_quota())
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.QUOTA_EXHAUSTED)


async def test_a_channel_that_is_not_bound_cannot_send(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.channel.bound = False
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.CHANNEL_UNAVAILABLE)


async def test_the_budget_at_level_three_pauses_the_proactive_messages(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.go_to(calm.at(9, 55))
    calm.exhaust_budget()
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.BUDGET)


async def test_two_messages_keep_their_distance(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.add_log(calm.at(9, 30))
    await refused(calm, TriggerKind.FOLLOWUP, calm.at(10, 0), Reason.SPACING)


async def test_after_the_spacing_has_passed_the_message_goes(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.add_log(calm.at(8, 55))
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "sent", report


async def test_nobody_is_chased_more_than_the_limit(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.add_log(calm.at(8, 0))
    calm.add_log(calm.at(9, 0), chase=1)
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.CHASE_LIMIT)


async def test_one_chase_is_allowed_after_an_unanswered_message(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.add_log(calm.at(8, 0))
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "sent", report
    assert calm.sent_rows()[-1].chase_seq == 1


async def test_a_message_the_user_answered_is_not_a_chase(calm: World) -> None:
    calm.add_log(calm.at(8, 0))
    calm.add_log(calm.at(9, 0))
    calm.user_writes(at=calm.at(9, 20))  # he wrote after both: nothing is unanswered
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "sent", report
    assert calm.sent_rows()[-1].chase_seq == 0


async def test_the_daily_maximum_holds(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.services.runtime.set(PROACTIVE_DAILY_MAX, 1, by="test")
    calm.add_log(calm.at(8, 0))
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.DAILY_MAX)


async def test_the_daily_maximum_holds_for_a_followup_too(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.services.runtime.set(PROACTIVE_DAILY_MAX, 1, by="test")
    calm.add_log(calm.at(8, 0))
    await refused(calm, TriggerKind.FOLLOWUP, calm.at(10, 0), Reason.DAILY_MAX)


async def test_when_the_days_draw_is_used_only_a_followup_may_still_go(calm: World) -> None:
    calm.rig.set_quota(QuotaRange(1, 1))
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.add_log(calm.at(8, 0))
    await refused(calm, TriggerKind.SHARE, calm.at(10, 0), Reason.QUOTA_SPENT)
    calm.put(TriggerKind.FOLLOWUP, calm.at(10, 10))
    report = await calm.tick_at(calm.at(10, 15))
    assert report.outcome == "sent" and report.candidate is not None
    assert report.candidate.kind is TriggerKind.FOLLOWUP


# --------------------------------------------------------------------- how a refusal is kept


async def test_a_refusal_is_written_once_and_the_candidate_waits_for_the_next_tick(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(11, 0, day=8))  # the window is closed
    planned = calm.at(10, 0)
    row = calm.put(TriggerKind.SHARE, planned, minutes=90)
    for step in range(1, 6):
        await calm.tick_at(planned + timedelta(minutes=5 * step))
    assert [r.reason for r in calm.rejected()] == ["window_closed"]
    assert calm.candidates.get(row.id).status == "pending"  # type: ignore[union-attr]
    assert calm.candidates.get(row.id).last_reason == "window_closed"  # type: ignore[union-attr]
    # the constraint changes: the new reason is written, the old one is not repeated
    calm.channel.user_writes(planned + timedelta(minutes=30))
    calm.channel.window.on_outbound(calm.channel.window.remaining_quota())
    await calm.tick_at(planned + timedelta(minutes=35))
    assert [r.reason for r in calm.rejected()] == ["window_closed", "quota_exhausted"]


async def test_the_window_closes_while_the_planner_works_and_the_message_is_not_sent(
    calm: World,
) -> None:
    """The constraints are checked again right before sending, with fresh facts."""
    calm.user_writes(at=calm.at(20, 0, day=8))
    planned = calm.at(10, 0)
    calm.put(TriggerKind.SHARE, planned)
    calm.script.on_request = lambda _kind: calm.channel.window.mark_expired(calm.clock.now_utc())
    report = await calm.tick_at(planned + timedelta(minutes=5))
    assert report.outcome == "rejected" and report.reason == "window_closed", report
    assert calm.channel.out == [] and calm.sent_rows() == []
    entry = calm.rejected()[0]
    assert entry.plan_reason is None  # the sealed note is only read with the text switch ...
    assert calm.log.entries(outcomes=["rejected"], with_text=True)[0].plan_reason  # ... but exists
    assert entry.cost_usd and entry.cost_usd > 0  # and the planner call is accounted for


async def test_she_falls_asleep_while_the_planner_works_and_nothing_is_sent(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0))
    planned = calm.at(23, 0)
    calm.put(TriggerKind.BEDTIME, planned, minutes=20)
    deep = middle(interval(calm, "deep_sleep", night=True))
    calm.script.on_request = lambda _kind: calm.clock.set_time(deep)
    report = await calm.tick_at(planned + timedelta(minutes=5))
    assert report.outcome == "rejected" and report.reason == "deep_sleep", report
    assert calm.channel.out == [] and calm.sent_rows() == []


async def test_it_is_paused_while_the_planner_works_and_nothing_is_sent(calm: World) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    planned = calm.at(10, 0)
    calm.put(TriggerKind.SHARE, planned)
    calm.script.on_request = lambda _kind: calm.services.runtime.set(PAUSED, True, by="test")
    report = await calm.tick_at(planned + timedelta(minutes=5))
    assert report.outcome == "rejected" and report.reason == "paused", report
    assert calm.channel.out == []
