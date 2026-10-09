"""The scheduler end to end on the synthetic world: the first messages (R-PRO-001, R-PRO-004)."""

from __future__ import annotations

from datetime import timedelta

from tests.support.proactive_world import World


async def test_the_wake_up_greeting_goes_out_in_its_window(pro: World) -> None:
    pro.user_writes(at=pro.at(21, 0, day=8))
    await pro.tick_at(pro.at(0, 10))  # the day opens and its fixed moments are laid out
    plan = pro.rig.planner.ensure(pro.rig.kit.time.local_date())
    assert plan.greeting.allowed and plan.greeting.earliest is not None
    greeting = [row for row in pro.candidates.pending() if row.kind.value == "greeting"]
    assert len(greeting) == 1
    when = greeting[0].planned_at + timedelta(minutes=1)
    report = await pro.tick_at(when)
    assert report.outcome == "sent", report
    assert pro.channel.texts == ["早啊", "刚醒"]
    rows = pro.sent_rows()
    assert len(rows) == 1 and rows[0].kind == "greeting" and rows[0].bubbles_sent == 2
