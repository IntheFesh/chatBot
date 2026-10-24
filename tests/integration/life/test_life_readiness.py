"""The first start, before anything was imported (R18-B8, D-870 to D-872).

``twin run`` starts on a database with no history: no profile, no routine model, no persona card.
The user writes on Friday evening.  The story:

* she says nothing - one system message tells him what is missing (the import first) and how to
  start it; his second message an hour later gets no second one;
* she writes nothing of her own either: the scheduler opens the day as one she may not write;
* the import ends and its jobs run (the profile and the routine model are computed, the examples
  are indexed): the engine is woken by the state version and tells him the one thing still
  missing - the persona card, whose description waits for approval;
* at one in the morning, in her deep sleep, the description is approved and written: the bot is
  ready, and the two messages are one round that waits for her waking - then she answers, as if
  just woken.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_bot_text_not_in_her_data,
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_proactive_rules,
    assert_screen_matches_records,
    snapshot_isolation,
)
from tests.support.proactive_world import opening_curve, proactive_model

pytestmark = pytest.mark.integration

FRIDAY, SATURDAY = date(2026, 10, 9), date(2026, 10, 10)
PEAKS = {31: 0.5, 32: 0.5, 48: 0.6, 49: 0.6, 73: 0.6, 74: 0.6, 90: 0.5, 91: 0.5, 92: 0.5}


async def test_before_the_import_she_is_silent_and_answers_the_backlog_when_she_wakes(
    make_world: WorldFactory,
) -> None:
    """R18-B8: not ready -> stored, told what is missing, no word of hers; ready -> her plan."""
    world = await make_world(
        datetime(2026, 10, 10, 1, 0, tzinfo=UTC),  # Friday 20:00 in Chicago
        model=proactive_model(opening_curve(base=0.05, peaks=PEAKS)),
        ready=False,
    )
    world.deepseek.book.when("在吗", "刚醒 怎么啦")
    await world.say("在吗")
    await world.run_for(minutes=30)
    assert world.persona_said == []
    (first,) = world.system_said
    assert "缺 画像、作息模型、人设卡" in first.text and "`twin import <导出目录>`" in first.text

    await world.run_until(world.local(21, 0))
    await world.say("在不在")  # an hour later: the same things are missing, no second notice
    await world.run_for(minutes=20)
    assert len(world.system_said) == 1 and world.persona_said == []

    await world.finish_import()  # the import and the jobs after it are done; the card waits
    before = snapshot_isolation(world)  # her history as the import left it
    await world.run_for(minutes=10)  # the state watcher wakes the engine: the missing set changed
    assert len(world.system_said) == 2, [item.text for item in world.system_said]
    second = world.system_said[1].text
    assert "缺 人设卡。" in second and "画像与作息" not in second and "导入：已完成" in second
    assert world.persona_said == []

    approved_at = world.local(1, 0, on=SATURDAY)  # in her deep sleep
    await world.run_until(approved_at)
    assert world.persona_said == []
    await world.describe_her()
    await world.run_until_idle()

    (answer,) = [item for item in world.persona_said if item.text == "刚醒 怎么啦"]
    wake = world.plan(SATURDAY).morning
    assert wake is not None and answer.at > wake.wake  # not at one in the morning: after waking
    assert answer.at - wake.wake < timedelta(hours=1)
    reply = [row for row in world.out_rows() if row.text == "刚醒 怎么啦"]
    assert len(reply) == 1
    assert len(world.in_rows()) == 2 and len(world.system_said) == 2  # one round, no more notices
    assert world.decision() is None  # the round is over

    # nothing of hers went out before the bot was ready: Friday was a day she may not write
    sent = world.proactive_rows(outcomes=["sent"])
    assert all(row.at > approved_at for row in sent), [(row.kind, row.at) for row in sent]
    friday = [row for row in world.proactive_rows() if row.local_date == FRIDAY]
    opened = [row for row in friday if row.outcome == "opened"]
    assert opened and all(row.enabled is False for row in opened)
    audit = assert_proactive_rules(world, FRIDAY, FRIDAY)
    assert audit.days[0].sent == 0 and not audit.days[0].active  # no observation day (D-872)

    assert_never_in_deep_sleep(world)
    assert_clean_screen(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world, before)
