"""A message that sounds like a crisis: she steps out of the role, at once (R-SAFE-001).

At ten past three in the morning, with her in deep sleep, the user writes that he does not want to
live any more.  The application of ``twin run`` screens the words locally, asks DeepSeek for a
second judgement, and - confirmed - answers *before* the night is over, out of the role, with a
fixed text and the help lines of the country the bot lives in.  What the rest of the story shows:

* the answer is not queued behind her sleep, carries no persona, and costs one judgement;
* an alert for the user himself is written; no mail goes anywhere (the emergency contact is off);
* a word from the list in a harmless sentence is judged harmless and answered as always;
* the help line follows the time zone of the bot: 988 in Chicago, 12356 after ``/时区 北京``;
* ``/重来`` refuses to redo an answer given out of the role;
* the next ordinary message is answered in the role again.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_bot_text_not_in_her_data,
    assert_clean_screen,
    assert_screen_matches_records,
    her_kind_at,
)
from tests.support.proactive_world import opening_curve, proactive_model
from twin.commands import texts

pytestmark = pytest.mark.integration

FRIDAY = date(2026, 10, 9)
CRISIS = "我真的不想活了"
QUIET_WINDOW = timedelta(seconds=15)


async def test_a_crisis_is_answered_at_once_and_out_of_the_role(make_world: WorldFactory) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 8, 10, tzinfo=UTC),  # ten past three in the night, Chicago
        model=proactive_model(opening_curve(base=0.02)),
    )

    # ---- deep in her sleep: the answer does not wait for her to wake -----------------------------
    await world.say(CRISIS)
    written = world.now
    assert world.kit.time.her_state(written).kind == "deep_sleep"
    await world.run_for(seconds=40)
    answer = world.persona_said
    assert answer and answer[0].at - written <= QUIET_WINDOW + timedelta(seconds=10)
    assert her_kind_at(world, answer[0]) == "deep_sleep"  # she is asleep and still answers
    spoken = "\n".join(s.text for s in answer)
    assert "988" in spoken and "12356" not in spoken  # the line of the country she lives in
    assert "我想先停一下" in spoken  # out of the role: fixed words of care
    assert world.deepseek.calls["crisis"] == 1 and world.deepseek.calls["reply"] == 0
    rows = world.out_rows()
    assert {r.backend for r in rows if r.backend} == {"safety"}
    assert any(a["step"] == "crisis_answer" for a in rows[0].actions)
    await world.run_until_idle()
    categories = [c for c, _ in world.alerts()]
    assert ("crisis_detected", "critical") in world.alerts()  # told to him, in the alerts
    assert "emergency_contact" not in categories  # nobody else is told (R-SAFE-001): it is off

    # ---- an answer out of the role cannot be thrown away and written again -----------------------
    await world.say("/重来")
    assert world.system_said[-1].text == texts.PREFIX + texts.REDO_SAFETY
    assert world.deepseek.calls["reply"] == 0

    # ---- he is better; she is in the role again, after her sleep ---------------------------------
    await world.say("谢谢你 我好多了")
    await world.run_until_idle()
    wake = world.plan(FRIDAY).morning.wake
    assert world.persona_said[-1].text == "嗯嗯" and world.persona_said[-1].at >= wake
    assert world.deepseek.calls["reply"] == 1

    # ---- a word of the list in a harmless sentence: judged, and answered as always ---------------
    await world.run_until(world.local(14, 0, on=FRIDAY))
    alerts = len(world.alerts())
    await world.say("刚才好尴尬 我想消失")
    await world.run_until_idle()
    assert world.deepseek.calls["crisis"] == 2  # the screen caught the word, the judge dismissed it
    assert world.persona_said[-1].text == "嗯嗯" and "988" not in world.persona_said[-1].text
    assert (
        len([a for a in world.alerts() if a[0] == "crisis_detected"]) == 1
        and len(world.alerts()) >= alerts
    )

    # ---- the help line follows the zone of the bot -----------------------------------------------
    await world.say("/时区 北京")
    assert "Asia/Shanghai" in world.system_said[-1].text
    await world.run_for(minutes=1)
    before = len(world.persona_said)
    await world.say(CRISIS)
    await world.run_for(seconds=40)
    again = "\n".join(s.text for s in world.persona_said[before:])
    assert "12356" in again and "988" not in again
    assert len([a for a in world.alerts() if a[0] == "crisis_detected"]) == 2

    assert_clean_screen(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []
