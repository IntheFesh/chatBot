"""A simulated month of the production scheduler, judged by the production audit (R-PRO-001..005).

The scheduler ticks every five minutes over whole local days on a manual clock; the simulated user
writes in sessions (see ``tests.support.proactive_sim``).  The statistics are the ones of the SPEC:
no message in deep sleep, every day inside its range, the daily mean close to the target, nothing
closer than the spacing, no chase beyond the limit, at most two edge messages in any seven days.

Under ``SLOWDB_MS`` every database access is slow on purpose, so the month is shortened and only
the invariants are checked (the mean of a few days says nothing).
"""

from __future__ import annotations

import os
from datetime import date, timedelta

from tests.support.proactive_sim import ALL_DAY, UserBehavior, render, simulate
from tests.support.proactive_world import World
from twin.engine.turns import BotTurnMessages
from twin.eval.proactive_audit import audit_days

SLOW = float(os.environ.get("SLOWDB_MS", "0") or 0) > 0
FIRST = date(2026, 10, 12)  # a Monday
MONTH = 4 if SLOW else 30
WEEK = 3 if SLOW else 10


async def test_a_month_of_ticks_keeps_every_rule_and_hits_the_target(pro: World) -> None:
    result = await simulate(
        pro, first_day=FIRST, days=MONTH, behavior=UserBehavior(sessions=ALL_DAY), seed=2
    )
    audit = audit_days(
        pro.log,
        pro.ratings,
        pro.services.settings.proactive,
        first_day=FIRST,
        last_day=FIRST + timedelta(days=MONTH - 1),
        now=pro.clock.now_utc(),
    )
    sent = result.sent
    time = pro.rig.kit.time
    # the core of the night: nothing, whatever the log says and whatever the plan says
    assert audit.deep_sleep == 0
    assert all(row.her_state in ("free", "busy", "sleep_edge") for row in sent)
    for row in sent:
        assert str(time.her_state(row.at).kind) != "deep_sleep", row.local_at
    # every day is inside its range and watched
    assert audit.complete and len(audit.days) == MONTH
    assert all(day.count_ok for day in audit.days), [
        (str(day.day), day.sent, day.low, day.high) for day in audit.days if not day.count_ok
    ]
    assert all(1 <= day.sent <= 6 for day in audit.days)
    # spacing, chase, the edge of sleep
    assert audit.spacing_violations == 0 and audit.chase_violations == 0
    assert audit.edge_max_week <= pro.services.settings.proactive.edge_of_sleep_weekly_max <= 2
    assert audit.compliant, [day.problems for day in audit.days if not day.compliant]
    assert len(result.sent) == audit.sent
    if not SLOW:  # the mean of the month is the target of the plans, within 15 %
        target = result.mean_quota
        assert abs(result.mean_per_day - target) <= 0.15 * target, render(result)
        assert 1 < result.mean_per_day < 6
    # the mix of what she writes: fixed moments and the random ones
    kinds = result.kinds()
    assert kinds["greeting"] > 0 and kinds["meal"] > 0
    assert set(kinds) <= {"greeting", "meal", "bedtime", "share", "silence", "edge", "followup"}


async def test_a_user_who_writes_in_the_evening_only_leaves_the_rest_to_the_window(
    pro: World,
) -> None:
    """The window closes for the hours before his evening message: those are suppressed."""
    result = await simulate(pro, first_day=FIRST, days=WEEK, behavior=UserBehavior(), seed=5)
    audit = audit_days(
        pro.log,
        pro.ratings,
        pro.services.settings.proactive,
        first_day=FIRST,
        last_day=FIRST + timedelta(days=WEEK - 1),
        now=pro.clock.now_utc(),
    )
    assert audit.deep_sleep == 0 and audit.spacing_violations == 0 and audit.chase_violations == 0
    assert audit.compliant, [day.problems for day in audit.days if not day.compliant]
    assert result.refused()["window_closed"] > 0  # something was held back, and it says so
    assert audit.suppressed_window == result.refused()["window_closed"] + result.refused().get(
        "quota_exhausted", 0
    )


async def test_days_without_a_message_of_the_user_are_quiet_and_excused(pro: World) -> None:
    # days 2-4 are silent (2-3 and a shorter week when the database is slow); then he writes again
    days, silent, back = (8, frozenset({2, 3, 4}), 5) if not SLOW else (6, frozenset({2, 3}), 4)
    result = await simulate(
        pro,
        first_day=FIRST,
        days=days,
        behavior=UserBehavior(sessions=ALL_DAY, silent_days=silent),
        seed=4,
    )
    by_day = {summary.day: summary for summary in result.days}
    third = by_day[FIRST + timedelta(days=3)]  # a whole local day with the window closed
    assert third.sent == 0 and third.refused["window_closed"] > 0
    audit = audit_days(
        pro.log,
        pro.ratings,
        pro.services.settings.proactive,
        first_day=FIRST,
        last_day=FIRST + timedelta(days=days - 1),
        now=pro.clock.now_utc(),
    )
    quiet = next(day for day in audit.days if day.day == FIRST + timedelta(days=3))
    assert quiet.excused and quiet.compliant  # below the minimum, but the window did it
    assert audit.compliant and audit.suppressed_window > 0
    resumed = [s for s in result.days if s.day >= FIRST + timedelta(days=back)]
    assert sum(s.sent for s in resumed) > 0  # after he writes again she writes too


async def test_no_message_starts_within_ten_minutes_of_the_talk(pro: World) -> None:
    result = await simulate(
        pro, first_day=FIRST, days=WEEK, behavior=UserBehavior(sessions=ALL_DAY), seed=9
    )
    reader = BotTurnMessages(pro.services.db)
    quiet = timedelta(minutes=pro.services.settings.proactive.user_active_min)
    assert result.sent
    for row in result.sent:
        assert list(reader.messages_between(row.at - quiet, row.at)) == [], row.local_at
