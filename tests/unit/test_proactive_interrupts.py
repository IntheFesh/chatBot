"""A count that shrinks, and an interruption before the first bubble (R-PRO-003, R-PRO-007)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from tests.support.proactive_world import World
from twin.schedule.proactive.types import TriggerKind

FRIDAY = date(2026, 10, 9)


def deep_sleep_tonight(world: World) -> Any:
    plan = world.rig.planner.ensure(FRIDAY)
    assert plan.night is not None
    start, end = next((a, b) for kind, a, b in plan.night.intervals() if kind == "deep_sleep")
    return start + (end - start) / 2


async def test_a_count_that_shrank_while_the_planner_worked_is_fitted_at_the_last_moment(
    calm: World,
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    window = calm.channel.window
    calm.script.messages["share"] = ("一二", "三四", "五六")
    calm.script.on_request = lambda _kind: window.on_outbound(window.remaining_quota() - 2)
    calm.put(TriggerKind.SHARE, calm.at(10, 0))
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "sent" and report.sent == 1, report
    entry = calm.sent_rows()[0]
    assert entry.result is not None and entry.result["planned"] == 1
    assert entry.result["fitted"] == [{"step": "quota_merge", "count": 2}]
    assert calm.channel.texts[-1].replace(" ", "") == "一二三四五六"


@pytest.mark.parametrize(
    ("what", "reason"),
    [("user", "user_active"), ("sleep", "deep_sleep"), ("resume", "interrupted")],
)
async def test_an_interruption_before_the_first_bubble_leaves_nothing_but_the_log(
    calm: World, what: str, reason: str
) -> None:
    calm.user_writes(at=calm.at(20, 0) if what == "sleep" else calm.at(20, 0, day=8))
    planned = calm.at(23, 0) if what == "sleep" else calm.at(10, 0)
    row = calm.put(TriggerKind.BEDTIME if what == "sleep" else TriggerKind.SHARE, planned)
    deep = deep_sleep_tonight(calm)

    def interrupt(item: Any) -> None:
        if item.kind != "typing":
            return
        if what == "user":
            calm.talk.user_wrote()
        elif what == "sleep":
            calm.clock.set_time(deep)
        else:
            calm.scheduler._epoch += 1  # type: ignore[attr-defined]

    calm.channel.on_send = interrupt
    report = await calm.tick_at(planned + timedelta(minutes=5))
    assert report.outcome == "dropped" and report.reason == reason, report
    assert calm.channel.texts == [] and calm.sent_rows() == []
    assert [r.reason for r in calm.log.entries(outcomes=["dropped"])] == [reason]
    assert calm.candidates.get(row.id).status == "dropped"  # type: ignore[union-attr]
