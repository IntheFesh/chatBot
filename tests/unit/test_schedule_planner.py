"""The daily planner: making, keeping and replacing plans, and answering "what is she doing".

R-SCH-001 (``her_state``), R-SCH-002 (switching the time zone), R-SCH-003 (the days of the clock
change), R-SCH-004 (a plan per day, made once, reproducible) and R-SCH-005 (a restart or a wake-up
checks the plan).  Everything runs on the injected clock; the real tables are used.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.routine import Rig, busy_window, fixed_model, make_model, sleep_window
from twin.config.runtime import BOT_TIMEZONE
from twin.profile.overrides import OverrideView
from twin.schedule.events import CandidatesExpired, PlanRebuilt, TimezoneSwitched
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.planner import TimezoneError
from twin.schedule.time_service import PlanUnavailableError
from twin.services import Services
from twin.storage.db import WritePolicy, use_write_policy
from twin.storage.schedule_models import DailyPlanRow

FRIDAY = date(2026, 10, 9)
SHANGHAI = "Asia/Shanghai"
CHICAGO = "America/Chicago"


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


@pytest.fixture
def rig(services: Services, clock: ManualClock) -> Rig:
    clock.set_time(utc(2026, 10, 9, 12, 0))  # 07:00 in Chicago, half an hour before she wakes
    return Rig.build(services, clock, fixed_model())


def local(moment: datetime, rig: Rig) -> str:
    return moment.astimezone(rig.zone()).strftime("%m-%d %H:%M")


# ------------------------------------------------------------------ making plans


def test_the_plan_of_a_day_is_made_once_and_stored_sealed(rig: Rig, services: Services) -> None:
    first = rig.planner.ensure(FRIDAY, reason="daily")
    again = rig.planner.ensure(FRIDAY, reason="demand")
    assert again.id == first.id and first.reason == "daily"
    assert first.timezone == CHICAGO and first.day_type == "workday"
    with services.db.session() as session:
        rows = list(session.scalars(select(DailyPlanRow)))
        assert len(rows) == 1
        row = rows[0]
        assert row.local_date == FRIDAY and row.seed == first.seed and row.wake_at == first.wake
        assert (row.effective_from, row.ends_at) == (first.effective_from, first.ends_at)
        sealed = bytes(row.plan_ct)
        assert b"America/Chicago" not in sealed and b"deep_sleep" not in sealed
        assert row.plan["timezone"] == CHICAGO  # readable through the key
    assert rig.planner.store.get(first.id) == first


def test_the_seed_is_the_date_and_the_installation_salt(rig: Rig) -> None:
    plan = rig.planner.ensure(FRIDAY)
    salt = rig.planner.salt.get()
    assert len(salt) == 32 and rig.planner.salt.get() == salt  # made once, then kept
    from twin.schedule.plan_builder import plan_seed

    assert plan.seed == plan_seed(FRIDAY, salt)
    other = rig.planner.ensure(FRIDAY + timedelta(days=1))
    assert other.seed == plan_seed(FRIDAY + timedelta(days=1), salt) != plan.seed


def test_the_same_day_is_planned_identically_after_the_plan_is_lost(
    rig: Rig, services: Services
) -> None:
    """Debugging and restarts: the seed alone reproduces the plan."""
    from sqlalchemy import delete

    first = rig.planner.ensure(FRIDAY)
    with services.db.transaction() as session:
        session.execute(delete(DailyPlanRow))
    rig.planner.invalidate()
    second = rig.planner.ensure(FRIDAY)
    assert second.id != first.id
    assert second.morning == first.morning and second.night == first.night
    assert second.quota == first.quota and second.meals == first.meals and second.busy == first.busy


def test_a_morning_is_the_night_of_the_day_before(rig: Rig) -> None:
    thursday = rig.planner.ensure(FRIDAY - timedelta(days=1))
    friday = rig.planner.ensure(FRIDAY)
    assert thursday.night is not None and friday.morning is not None
    assert (
        friday.morning.onset == thursday.night.onset and friday.morning.wake == thursday.night.wake
    )
    assert friday.morning.adopted_from == thursday.id
    assert friday.night is not None and friday.night.adopted_from is None
    # the two plans agree wherever they overlap
    overlap_start = friday.effective_from
    for plan in (thursday, friday):
        assert plan.covers(overlap_start + timedelta(hours=2))
    one = rig.planner.state_at(overlap_start + timedelta(hours=2))
    assert one.kind == "deep_sleep" and one.since == thursday.night.onset + timedelta(minutes=30)


def test_a_day_without_a_neighbour_draws_its_own_morning(rig: Rig) -> None:
    friday = rig.planner.ensure(FRIDAY)
    assert friday.morning is not None and friday.morning.adopted_from is None
    assert local(friday.morning.onset, rig) == "10-08 23:30"


def test_a_changed_routine_does_not_take_over_the_old_night(rig: Rig) -> None:
    thursday = rig.planner.ensure(FRIDAY - timedelta(days=1))
    rig.use(make_model({"all": sleep_window(1.0, 9.0)}))
    friday = rig.planner.ensure(FRIDAY)
    assert thursday.night is not None and friday.morning is not None
    assert friday.morning.adopted_from is None
    assert local(friday.morning.wake, rig) == "10-09 09:00"


# ------------------------------------------------------------------- her state


def test_her_state_through_a_day_and_at_its_boundaries(rig: Rig) -> None:
    rig.planner.ensure(FRIDAY)
    probes = {
        "06:59:59": ("deep_sleep", "10-09 00:00", "10-09 07:00"),
        "07:00:00": ("sleep_edge", "10-09 07:00", "10-09 07:30"),
        "07:29:59": ("sleep_edge", "10-09 07:00", "10-09 07:30"),
        "07:30:00": ("free", "10-09 07:30", "10-09 13:00"),
        "12:59:59": ("free", "10-09 07:30", "10-09 13:00"),
        "13:00:00": ("busy", "10-09 13:00", "10-09 17:00"),
        "16:59:59": ("busy", "10-09 13:00", "10-09 17:00"),
        "17:00:00": ("free", "10-09 17:00", "10-09 23:30"),
        "23:29:59": ("free", "10-09 17:00", "10-09 23:30"),
        "23:30:00": ("sleep_edge", "10-09 23:30", "10-10 00:00"),
    }
    for clock_text, (kind, since, until) in probes.items():
        hour, minute, second = (int(part) for part in clock_text.split(":"))
        moment = rig.at(2026, 10, 9, hour, minute) + timedelta(seconds=second)
        state = rig.kit.time.her_state(moment)
        assert (state.kind, local(state.since, rig), local(state.until, rig)) == (
            kind,
            since,
            until,
        ), clock_text
        assert state.since <= moment < state.until
    assert rig.kit.time.her_state(rig.at(2026, 10, 9, 14)).busy is not None
    assert rig.kit.time.her_state(rig.at(2026, 10, 9, 10)).busy is None


def test_her_state_now_comes_from_the_injected_clock(rig: Rig) -> None:
    assert rig.kit.time.her_state().kind == "sleep_edge"  # 07:00: about to wake
    rig.advance(minutes=31)
    assert rig.kit.time.her_state().kind == "free"
    rig.advance(hours=6)  # 13:31
    state = rig.kit.time.her_state()
    assert state.kind == "busy" and local(state.until, rig) == "10-09 17:00"


def test_a_night_across_midnight_is_one_state_even_across_two_plans(rig: Rig) -> None:
    """Deep sleep from 22:30 to 07:00 is one stretch although two plans describe it."""
    rig.use(make_model({"all": sleep_window(22.0, 7.0)}))
    rig.planner.ensure(FRIDAY)
    rig.planner.ensure(FRIDAY + timedelta(days=1))
    assert (
        len(rig.planner.store.for_date(FRIDAY))
        == len(rig.planner.store.for_date(date(2026, 10, 10)))
        == 1
    )
    state = rig.kit.time.her_state(rig.at(2026, 10, 10, 3))
    assert state.kind == "deep_sleep"
    assert (local(state.since, rig), local(state.until, rig)) == ("10-09 22:30", "10-10 06:30")
    assert rig.kit.time.her_state(rig.at(2026, 10, 9, 23, 59)).since == state.since
    late = rig.kit.time.her_state(rig.at(2026, 10, 10, 8))  # free since the wake-up
    assert late.kind == "free" and local(late.since, rig) == "10-10 07:00"


def test_a_plan_is_made_when_the_state_of_an_unplanned_day_is_asked(rig: Rig) -> None:
    assert rig.planner.store.recent() == []
    state = rig.kit.time.her_state()
    assert state.plan_id is not None
    (plan,) = rig.planner.store.recent()
    assert plan.id == state.plan_id and plan.local_date == FRIDAY and plan.reason == "demand"
    tomorrow = rig.kit.time.her_state(rig.at(2026, 10, 10, 12))  # one day ahead is allowed
    assert tomorrow.kind == "free"
    with pytest.raises(PlanUnavailableError, match="only made for"):
        rig.kit.time.her_state(rig.at(2026, 10, 20, 12))
    with pytest.raises(PlanUnavailableError, match="only made for"):
        rig.kit.time.her_state(rig.at(2026, 9, 1, 12))


def test_without_a_routine_model_she_is_simply_free(rig: Rig) -> None:
    rig.use(None)
    state = rig.kit.time.her_state()
    assert state.kind == "free"
    (plan,) = rig.planner.store.recent()
    assert plan.warnings == ("no_routine_model",)


def test_the_states_of_the_days_of_the_clock_change_have_no_hole_and_no_overlap(
    services: Services, clock: ManualClock
) -> None:
    """R-SCH-003: every instant of the 23-hour and the 25-hour day has exactly one state."""
    model = make_model(
        {"all": sleep_window(23.5, 7.5)},
        {"all": (busy_window(1.5, 4.0), busy_window(13, 17))},  # a window inside the changing hours
        initiations=3.0,
    )
    for day, hours in ((date(2026, 3, 8), 23), (date(2026, 11, 1), 25)):
        clock.set_time(utc(2026, 3, 1) if hours == 23 else utc(2026, 10, 25))
        rig = Rig.build(services, clock, model, zone=CHICAGO)
        start, end = rig.kit.time.day_bounds_utc(day)
        assert end - start == timedelta(hours=hours)
        clock.set_time(start + timedelta(hours=1))
        rig.planner.invalidate()
        rig.planner.ensure(day)
        moment = start
        while moment < end:
            state = rig.planner.state_at(moment)
            assert state.since <= moment < state.until
            following = rig.planner.state_at(state.until) if state.until < end else None
            if following is not None:
                assert (following.kind, following.busy) != (state.kind, state.busy)
                assert following.since == state.until  # the next stretch starts where this ends
            moment += timedelta(minutes=5)
        plan = rig.planner.store.current_for(day, CHICAGO)
        assert plan is not None
        inside = [s for s in plan.segments if s.start < end]
        assert inside[0].start == start
        assert all(a.end == b.start for a, b in pairwise(plan.segments))
        with services.db.transaction() as session:
            from sqlalchemy import delete

            session.execute(delete(DailyPlanRow))
        rig.planner.invalidate()


# -------------------------------------------------------------------- refresh


def test_refresh_keeps_a_plan_whose_inputs_did_not_change(rig: Rig) -> None:
    first = rig.planner.refresh("startup")
    assert (
        first.changed
        and first.plan.reason == "startup"
        and first.plan.effective_from == rig.at(2026, 10, 9)
    )
    clock_moved = rig.clock.now_utc() + timedelta(hours=3)
    rig.move_to(clock_moved)
    again = rig.planner.refresh("wake")
    assert not again.changed and again.plan.id == first.plan.id and again.events == ()
    assert len(rig.planner.store.recent()) == 1


def test_a_changed_routine_replaces_the_plan_from_now_on(rig: Rig) -> None:
    first = rig.planner.refresh("startup").plan
    rig.move_to(rig.at(2026, 10, 9, 10))
    rig.use(fixed_model(initiations=3.6, manual_busy=(busy_window(15, 18, weekdays=(4,)),)))
    changed = rig.planner.refresh("settings_changed")
    assert changed.changed and changed.superseded == (first.id,)
    plan = changed.plan
    assert plan.reason == "settings_changed" and plan.effective_from == rig.at(2026, 10, 9, 10)
    assert [local(b.start, rig)[6:] for b in plan.busy] == ["15:00"]  # the new weekly correction
    assert plan.morning == first.morning  # the morning is over: it is history
    old = rig.planner.store.get(first.id)
    assert (
        old is not None
        and old.superseded_by == plan.id
        and old.superseded_at == plan.effective_from
    )
    # the past still reads from the plan that was in force then; the future from the new one
    rig.planner.invalidate()
    assert rig.planner.store.covering(rig.at(2026, 10, 9, 9)).id == first.id  # type: ignore[union-attr]
    assert rig.planner.store.covering(rig.at(2026, 10, 9, 14)).id == plan.id  # type: ignore[union-attr]
    assert rig.kit.time.her_state(rig.at(2026, 10, 9, 14)).kind == "free"  # 13-17 is no longer busy
    assert rig.kit.time.her_state(rig.at(2026, 10, 9, 16)).kind == "busy"
    assert rig.kit.time.her_state(rig.at(2026, 10, 9, 9)).kind == "free"
    (event,) = changed.events
    assert isinstance(event, PlanRebuilt) and event.superseded == (first.id,)


def test_a_new_proactive_range_or_a_switch_off_replaces_the_plan(rig: Rig) -> None:
    first = rig.planner.refresh("startup").plan
    rig.set_quota(QuotaRange(2, 4))
    second = rig.planner.refresh("settings_changed")
    assert second.changed and (second.plan.quota.minimum, second.plan.quota.maximum) == (2, 4)
    rig.set_quota(QuotaRange(2, 4, enabled=False))
    third = rig.planner.refresh("settings_changed")
    assert third.changed and third.plan.quota.total == 0 and not third.plan.quota.enabled
    assert first.quota.enabled


def test_a_manual_override_or_holiday_in_the_routine_reaches_the_plan(
    rig: Rig, services: Services
) -> None:
    override = OverrideView(
        "o1",
        "sleep",
        {"start": "01:00", "end": "09:00", "day_types": None},
        True,
        None,
        rig.clock.now_utc(),
    )
    rig.use(fixed_model().with_overrides([override]))
    plan = rig.planner.refresh("settings_changed").plan
    assert plan.morning is not None and local(plan.morning.wake, rig) == "10-09 09:00"
    assert plan.morning.source == "override"


def test_a_forced_refresh_redraws_the_same_plan_from_the_same_seed(rig: Rig) -> None:
    first = rig.planner.refresh("startup").plan
    rig.move_to(rig.at(2026, 10, 9, 10))
    forced = rig.planner.refresh("manual", force=True)
    assert forced.changed and forced.plan.id != first.id
    assert forced.plan.seed == first.seed
    assert forced.plan.night == first.night and forced.plan.quota.total == first.quota.total


def test_a_rebuild_does_not_wake_her_up(rig: Rig) -> None:
    """A routine changed while she sleeps: the night that began is not redrawn."""
    rig.use(make_model({"all": sleep_window(23.5, 7.5)}))
    first = rig.planner.refresh("startup").plan
    rig.move_to(rig.at(2026, 10, 9, 23, 45))  # asleep already
    rig.use(make_model({"all": sleep_window(23.5, 7.5)}, initiations=2.0))
    again = rig.planner.refresh("settings_changed", force=True)
    assert again.plan.night == first.night
    assert rig.kit.time.her_state().kind == "sleep_edge"


def test_the_bookkeeping_of_daily_jobs_survives_a_rebuild_of_the_same_day(rig: Rig) -> None:
    first = rig.planner.refresh("startup").plan
    rig.planner.store.mark_lifeline_queued(first.id, "J1")
    rig.planner.store.mark_summary_queued(first.id, rig.clock.now_utc())
    rig.move_to(rig.at(2026, 10, 9, 10))
    rebuilt = rig.planner.refresh("manual", force=True).plan
    assert rebuilt.lifeline_job_id == "J1" and rebuilt.summary_queued_at is not None


# ---------------------------------------------------------------- time zone switch


def test_an_unknown_time_zone_changes_nothing(rig: Rig, services: Services) -> None:
    rig.planner.refresh("startup")
    for bad in ("Mars/Olympus", "", "../etc/passwd", "/UTC", " "):
        with pytest.raises(TimezoneError):
            rig.planner.switch_timezone(bad)
    assert services.runtime.get(BOT_TIMEZONE) == CHICAGO
    assert rig.planner.history.recent() == []
    same = rig.planner.switch_timezone(CHICAGO)
    assert not same.changed and same.record is None and same.events == ()


def test_switching_by_day_keeps_the_sleep_of_the_new_zone_and_skips_nothing(rig: Rig) -> None:
    """Chicago 10:00 -> Shanghai 23:00 the same evening: she goes to bed at 23:30 there."""
    rig.planner.refresh("startup")
    rig.move_to(rig.at(2026, 10, 9, 10))  # 15:00 UTC
    assert rig.kit.time.her_state().kind == "free"
    outcome = rig.planner.switch_timezone(SHANGHAI, source="cli")
    assert outcome.changed and outcome.old_timezone == CHICAGO and outcome.new_timezone == SHANGHAI
    assert rig.kit.time.bot_timezone().key == SHANGHAI
    plan = outcome.plan
    assert plan is not None and plan.timezone == SHANGHAI and plan.reason == "timezone_switch"
    assert plan.local_date == date(2026, 10, 9) and plan.effective_from == utc(2026, 10, 9, 15, 0)
    assert plan.night is not None and plan.night.onset == utc(2026, 10, 9, 15, 30)  # 23:30 CST
    state = rig.kit.time.her_state()
    assert state.kind == "free" and state.until == plan.night.onset
    assert rig.kit.time.her_state(plan.night.onset).kind == "sleep_edge"
    assert rig.kit.time.her_state(plan.night.onset + timedelta(hours=2)).kind == "deep_sleep"
    wake = plan.night.wake  # 07:30 on the 10th in Shanghai
    assert wake == utc(2026, 10, 9, 23, 30) and rig.kit.time.her_state(wake).kind == "free"
    # the time service now speaks for China: Saturday 10 October is a make-up working day
    assert rig.kit.time.day_type(date(2026, 10, 10)) == "workday"


def test_the_new_zone_is_night_already_and_she_goes_to_sleep_at_once(rig: Rig) -> None:
    """Chicago 14:00 -> Shanghai 03:00: no night is skipped, she falls asleep now."""
    rig.planner.refresh("startup")
    rig.move_to(rig.at(2026, 10, 9, 14))  # 19:00 UTC = 03:00 on the 10th in Shanghai
    assert rig.kit.time.her_state().kind == "busy"  # 13:00-17:00 in Chicago
    outcome = rig.planner.switch_timezone(SHANGHAI)
    plan = outcome.plan
    assert plan is not None and plan.local_date == date(2026, 10, 10)
    assert plan.morning is not None and plan.morning.clipped
    assert plan.morning.onset == rig.clock.now_utc()  # she goes to bed at the moment of the switch
    assert plan.morning.wake == utc(2026, 10, 9, 23, 30)  # 07:30 in Shanghai
    state = rig.kit.time.her_state()
    assert state.kind == "sleep_edge" and state.since == rig.clock.now_utc()
    assert state.until == rig.clock.now_utc() + timedelta(minutes=30)
    assert rig.kit.time.her_state(rig.clock.now_utc() + timedelta(hours=1)).kind == "deep_sleep"
    assert rig.kit.time.her_state(utc(2026, 10, 9, 23, 30)).kind == "free"
    # and the Chicago afternoon that already happened is not rewritten
    before = rig.kit.time.her_state(utc(2026, 10, 9, 17, 0))  # noon in Chicago
    assert before.kind == "free" and before.plan_id != plan.id


def test_the_new_zone_is_a_deep_night_for_the_old_one_and_the_day_for_the_new_one(rig: Rig) -> None:
    """Chicago 02:00, asleep -> Shanghai 15:00: awake, and the old night stays history."""
    rig.planner.refresh("startup")
    rig.move_to(rig.at(2026, 10, 10, 2))  # 07:00 UTC
    assert rig.kit.time.her_state().kind == "deep_sleep"
    old_plan_id = rig.kit.time.her_state().plan_id
    outcome = rig.planner.switch_timezone(SHANGHAI)
    plan = outcome.plan
    assert plan is not None and plan.local_date == date(2026, 10, 10)
    now = rig.kit.time.her_state()
    # 15:00 on the make-up working day of 2026-10-10 in China: the busy window of workdays
    assert now.kind == "busy" and now.until == utc(2026, 10, 10, 9, 0)
    assert rig.kit.time.her_state(utc(2026, 10, 10, 6, 0)).kind == "deep_sleep"  # an hour ago
    assert rig.kit.time.her_state(utc(2026, 10, 10, 6, 0)).plan_id == old_plan_id
    assert plan.night is not None and plan.night.onset == utc(2026, 10, 10, 15, 30)
    assert rig.kit.time.her_state(utc(2026, 10, 10, 15, 45)).kind == "sleep_edge"  # 23:45 Shanghai


def test_the_greeting_just_sent_is_not_sent_again_within_eighteen_hours(rig: Rig) -> None:
    """The greeting went out at 08:20 in Chicago; in Shanghai the next wake-up is 10 hours on."""
    rig.planner.refresh("startup")
    sent = rig.at(2026, 10, 9, 8, 20)
    rig.move_to(sent)
    rig.planner.record_wake_greeting(sent)
    rig.move_to(sent + timedelta(minutes=5))
    outcome = rig.planner.switch_timezone(SHANGHAI)
    assert outcome.plan is not None
    assert outcome.plan.greeting.allowed is False  # her wake-up in the new zone is already behind
    tomorrow = rig.planner.ensure(date(2026, 10, 10))
    assert tomorrow.timezone == SHANGHAI and tomorrow.morning is not None
    assert tomorrow.morning.wake == utc(2026, 10, 9, 23, 30)  # 10 hours after the greeting
    assert not tomorrow.greeting.allowed
    assert tomorrow.greeting.reason == "within_min_gap_of_the_last_greeting"
    decision = rig.planner.greeting_decision(utc(2026, 10, 9, 23, 40))
    assert not decision.allowed and decision.reason == "within_min_gap_of_the_last_greeting"
    assert rig.planner.history.recent()[0].last_greeting_at == sent


def test_a_greeting_long_enough_ago_is_allowed_after_the_switch(rig: Rig) -> None:
    rig.planner.refresh("startup")
    yesterday = rig.at(2026, 10, 8, 8, 20)
    rig.planner.record_wake_greeting(yesterday)
    rig.move_to(rig.at(2026, 10, 9, 10))
    rig.planner.switch_timezone(SHANGHAI)
    tomorrow = rig.planner.ensure(date(2026, 10, 10))  # 07:30 Shanghai = 31 hours later
    assert tomorrow.greeting.allowed and tomorrow.greeting.reason == "ok"
    assert rig.planner.greeting_decision(utc(2026, 10, 9, 23, 20)).reason == "too_early"
    assert rig.planner.greeting_decision(utc(2026, 10, 9, 23, 36)).allowed
    assert rig.planner.greeting_decision(utc(2026, 10, 10, 0, 30)).reason == "window_over"
    rig.planner.record_wake_greeting(utc(2026, 10, 9, 23, 40))
    assert rig.planner.greeting_decision(utc(2026, 10, 9, 23, 45)).reason == (
        "already_sent_after_this_wake_up"
    )


def test_a_greeting_time_never_moves_backwards(rig: Rig) -> None:
    rig.planner.record_wake_greeting(utc(2026, 10, 9, 13, 0))
    rig.planner.record_wake_greeting(utc(2026, 10, 8, 13, 0))
    assert rig.planner.greetings.last() == utc(2026, 10, 9, 13, 0)


def test_a_switch_is_recorded_and_announced(rig: Rig) -> None:
    rig.planner.refresh("startup")
    rig.move_to(rig.at(2026, 10, 9, 10))
    outcome = rig.planner.switch_timezone(SHANGHAI, source="command")
    record = outcome.record
    assert record is not None and record.source == "command"
    assert (record.from_timezone, record.to_timezone) == (CHICAGO, SHANGHAI)
    assert record.changed_at == rig.clock.now_utc() and record.plan_id == outcome.plan.id  # type: ignore[union-attr]
    assert rig.planner.history.latest() == record
    expired, switched, rebuilt = outcome.events
    assert isinstance(expired, CandidatesExpired) and expired.reason == "timezone_switch"
    assert expired.covers(rig.clock.now_utc() + timedelta(days=3))  # later candidates go too
    assert isinstance(switched, TimezoneSwitched) and switched.new_timezone == SHANGHAI
    assert isinstance(rebuilt, PlanRebuilt) and len(rebuilt.superseded) == 1
    back = rig.planner.switch_timezone(CHICAGO)
    assert back.changed and [r.to_timezone for r in rig.planner.history.recent()] == [
        CHICAGO,
        SHANGHAI,
    ]
    assert [r.id for r in rig.planner.history.after(record.id)] == [back.record.id]  # type: ignore[union-attr]
    assert [r.id for r in rig.planner.history.after(None)] == [record.id, back.record.id]  # type: ignore[union-attr]


def test_the_plan_after_a_switch_is_not_redrawn_by_the_settings_check(rig: Rig) -> None:
    rig.planner.refresh("startup")
    rig.move_to(rig.at(2026, 10, 9, 10))
    outcome = rig.planner.switch_timezone(SHANGHAI)
    rig.planner.invalidate()
    check = rig.planner.refresh("settings_changed")
    assert not check.changed and check.plan.id == outcome.plan.id  # type: ignore[union-attr]


def test_the_quota_after_a_switch_is_the_share_of_the_day_that_is_left(rig: Rig) -> None:
    rig.set_quota(QuotaRange(6, 6))
    rig.planner.refresh("startup")
    rig.move_to(rig.at(2026, 10, 9, 10))
    outcome = rig.planner.switch_timezone(SHANGHAI)  # 23:00 in Shanghai: half an hour awake left
    assert outcome.plan is not None
    assert outcome.plan.quota.total == 6 and outcome.plan.quota.for_plan < 6


def test_the_old_zones_plans_stop_at_the_switch(rig: Rig) -> None:
    first = rig.planner.refresh("startup").plan
    rig.move_to(rig.at(2026, 10, 9, 10))
    rig.planner.switch_timezone(SHANGHAI)
    old = rig.planner.store.get(first.id)
    assert old is not None and old.superseded_at == rig.clock.now_utc()
    assert not old.covers(rig.clock.now_utc()) and old.covers(
        rig.clock.now_utc() - timedelta(minutes=1)
    )
    assert rig.planner.store.current_for(FRIDAY, CHICAGO) is None
    assert rig.planner.store.current_for(FRIDAY, SHANGHAI) is not None


# ---------------------------------------------------------------- reading only


def test_previews_and_plan_lookups_never_write(rig: Rig, services: Services) -> None:
    with use_write_policy(WritePolicy(read_only=True)):
        plan = rig.planner.preview(FRIDAY)
        assert plan.id == "" and plan.reason == "preview"
        assert rig.planner.store.current_for(FRIDAY, CHICAGO) is None
        assert rig.planner.store.covering(rig.clock.now_utc()) is None
        assert rig.planner.greetings.last() is None and rig.planner.salt.peek() is None
    stored = rig.planner.ensure(FRIDAY)
    with use_write_policy(WritePolicy(read_only=True)):
        again = rig.planner.preview(FRIDAY)
        assert again.seed == stored.seed  # with the salt in place the preview is the plan
        assert again.morning == stored.morning and again.night == stored.night
