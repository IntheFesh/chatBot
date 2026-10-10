"""The platform window closes and opens again (R-CH-008, R-PRO-003, R-ARCH-004).

WeChat lets a bot talk only for a while after the user has written
(``channel.proactive_window_safe_h``, 22 hours).  The application of ``twin run`` has the terminal
channel simulate exactly that (``LocalConsoleChannel`` keeps the same window and count as the
WeChat channel):

* while the window is open her messages of her own go out, none of them later than the window;
* when it closes, a message that is due is not sent and not queued: the log says the window
  suppressed it, and she stays quiet through the evening, the night and the next morning - the good
  night, the greeting - without an error that anyone can see;
* when he writes again the window opens: she answers, and on her own again after a while.

(The chase limit - she does not write again and again to someone who does not answer - would hide
the window in a story where he never writes, so this one raises it; it has its own scenarios.)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_proactive_rules,
    assert_screen_matches_records,
)
from tests.support.proactive_world import opening_curve, proactive_model
from twin.services import Services

pytestmark = pytest.mark.integration

FRIDAY, SATURDAY = date(2026, 10, 9), date(2026, 10, 10)
PEAKS = {31: 0.5, 32: 0.5, 48: 0.6, 49: 0.6, 73: 0.6, 74: 0.6, 90: 0.5, 91: 0.5, 92: 0.5}
PEAKS |= {93: 0.5, 94: 0.5, 95: 0.5}


def let_her_chase(services: Services) -> None:
    services.settings.proactive.max_chase = 6


async def test_the_window_closes_suppresses_her_and_opens_again(make_world: WorldFactory) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 3, 0, tzinfo=UTC),  # Thursday 22:00 in Chicago
        model=proactive_model(opening_curve(base=0.02, peaks=PEAKS)),
        configure=let_her_chase,
    )
    window = timedelta(hours=world.services.settings.channel.proactive_window_safe_h)
    assert window == timedelta(hours=22)
    await world.say("我先睡啦，晚安")
    opened = world.now
    closes = opened + window
    assert closes == world.local(20, 0, on=FRIDAY)

    # ---- the window is open: she writes on her own, none of it later than the window -----------
    await world.run_until(closes - timedelta(minutes=1))
    sent = world.proactive_rows(outcomes=["sent"])
    assert [r.kind for r in sent][:2] == ["bedtime", "greeting"]
    assert {"meal"} & {r.kind for r in sent}
    assert all(r.at < closes for r in sent)
    assert world.channel.window().can_send_proactive(world.now)

    # ---- it closes: the evening, the night and the morning go by in silence ---------------------
    await world.run_until(world.local(12, 0, on=SATURDAY))
    assert not world.channel.window().can_send_proactive(world.now)
    assert world.channel.session_state().window_remaining is not None
    assert world.channel.session_state().window_remaining < timedelta(0)
    assert [r for r in world.proactive_rows(outcomes=["sent"]) if r.at >= closes] == []
    suppressed = [
        r for r in world.proactive_rows(outcomes=["rejected"]) if r.reason == "window_closed"
    ]
    assert suppressed and all(r.at >= closes for r in suppressed)
    assert {"bedtime", "greeting"} <= {r.kind for r in suppressed}  # the good night, the greeting
    assert not [s for s in world.persona_said if s.at >= closes]  # nothing on the screen either
    assert_clean_screen(world)  # and no error text

    # ---- he writes: the window opens, she answers, and writes on her own again -----------------
    reopened = world.now
    await world.say("早呀 我睡过头了")
    await world.run_until_idle()
    assert world.persona_said[-1].at > reopened
    await world.say("/状态")
    line = next(x for x in world.system_said[-1].text.splitlines() if x.startswith("平台窗口"))
    assert "剩余 22 小时" in line and "还能发 8 条" in line  # the window runs again, from now
    assert world.channel.window().can_send_proactive(world.now)
    await world.run_until(reopened + timedelta(hours=8))
    again = [r for r in world.proactive_rows(outcomes=["sent"]) if r.at > reopened]
    assert again, "she did not write on her own after the window opened"
    assert all(r.at - reopened < window for r in again)

    assert_proactive_rules(world, FRIDAY, FRIDAY)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert world.deepseek.unexpected == []
