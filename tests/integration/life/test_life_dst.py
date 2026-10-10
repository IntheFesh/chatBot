"""The day the clocks go back: Sunday 2026-11-01 in Chicago has 25 hours (R-SCH-003, R-NFR-005).

At 02:00 daylight time (UTC-5) the clock goes back to 01:00 standard time (UTC-6).  The application
of ``twin run`` lives through the whole day, from the Saturday night before to the Monday after:

* the day has 25 real hours, and so has everything that is counted per day: the plan, the grid of
  five-minute looks of the proactive scheduler (300 steps, not 288) and the cost of the day;
* her wake-up and her bedtime are clock times of the zone - about nine in the morning, a quarter to
  midnight - at the *standard* time offset, whatever the hour that was repeated in between;
* the repeated hour is night: what he writes at 01:20 and at the second 01:20 waits for her waking,
  and is answered once, in the morning;
* the audit of the proactive messages of the day holds: the quota, the spacing, no deep sleep.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_proactive_rules,
    assert_screen_matches_records,
    assert_within_quota,
)
from tests.support.proactive_world import opening_curve, proactive_model
from twin.schedule.proactive.curve import build_grid
from twin.storage.models import CostLedger

pytestmark = pytest.mark.integration

SUNDAY = date(2026, 11, 1)
HOUR = timedelta(hours=1)


async def test_the_25_hour_day(make_world: WorldFactory) -> None:
    world = await make_world(
        datetime(2026, 11, 1, 4, 30, tzinfo=UTC),  # Saturday 23:30 in Chicago, daylight time
        model=proactive_model(opening_curve(base=0.02)),
    )
    await world.say("晚安啦")
    assert world.now.astimezone(world.zone).utcoffset() == timedelta(hours=-5)

    # ---- the day is 25 hours long, and the plan of it too -----------------------------------
    await world.run_until(world.local(0, 10, on=SUNDAY))
    start, end = world.kit.time.day_bounds_utc(SUNDAY)
    assert (start, end - start) == (datetime(2026, 11, 1, 5, 0, tzinfo=UTC), 25 * HOUR)
    plan = world.plan(SUNDAY)
    assert plan.day_type == "weekend" and plan.effective_from == start
    wake = plan.morning.wake.astimezone(world.zone)
    onset = plan.night.onset.astimezone(world.zone)
    assert wake.utcoffset() == onset.utcoffset() == timedelta(hours=-6)  # standard time
    assert 8 <= wake.hour <= 10  # about nine in the morning on a weekend, by the clock
    assert onset.hour in (23, 0) and onset.date() == SUNDAY  # the bedtime of the evening
    grid = build_grid(
        plan,
        step=timedelta(minutes=world.services.settings.proactive.tick_minutes),
        model=None,
        day_type=plan.day_type,
    )
    assert grid.size == 25 * 12  # a look every five minutes: 300 of them, not 288

    # ---- the repeated hour is night: both of his messages wait for her waking ---------------
    await world.run_until(world.local(1, 20, on=SUNDAY))
    assert world.now.astimezone(world.zone).utcoffset() == timedelta(hours=-5)  # the first 01:20
    await world.say("睡不着")
    await world.run_for(hours=1)
    again = world.now.astimezone(world.zone)
    assert (again.hour, again.minute, again.utcoffset()) == (1, 20, timedelta(hours=-6))
    await world.say("还是睡不着")
    spoken = len(world.persona_said)
    await world.run_until(plan.morning.wake + timedelta(hours=2))
    answers = [s for s in world.persona_said[spoken:] if s.kind == "text" and s.at > again]
    assert answers and all(s.at >= plan.morning.wake for s in answers)
    assert world.deepseek.of_kind("reply")[-1].last_user.count("睡不着") >= 2  # one round, both

    # ---- the last hour of the day belongs to the day --------------------------------------------
    await world.run_until(world.local(23, 30, on=SUNDAY))
    assert world.now == end - timedelta(minutes=30)  # half past eleven: 24.5 hours after midnight
    await world.say("今天好累")
    await world.run_until_idle()
    await world.run_until(end + timedelta(minutes=30))
    ledger = world.assembly.llm.ledger
    with world.services.db.session() as session:
        late = [
            row.cost_usd
            for row in session.scalars(select(CostLedger))
            if start + 24 * HOUR <= row.at < end and row.purpose == "reply"
        ]
    assert late, "no reply was paid for in the 25th hour"
    assert ledger.spent_on_day(SUNDAY) >= sum(late) > 0  # the 25th hour is Sunday's, not Monday's

    # ---- what holds for any story ---------------------------------------------------------------
    audit = assert_proactive_rules(world, SUNDAY, SUNDAY)
    assert audit.days[0].observed and audit.days[0].sent <= plan.quota.total
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_within_quota(world)
    assert world.deepseek.unexpected == []
