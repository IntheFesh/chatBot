"""The platform's count of messages runs short (R-ENG-009, R-CH-008, R-PRO-003).

After each message of the user the platform lets the bot send only a few (here four, the "safe"
number of ``channel.outbound_quota_safe``), and ``channel.proactive_reserve`` of them are kept for
messages she starts herself.  The application of ``twin run`` lives with that:

* a reply with more bubbles than the count leaves it (the count minus the reserve) is not cut: the
  neighbouring bubbles are put together, joined by a space, and the record says how often;
* a message of her own with more bubbles than are left is shortened the same way, and sent;
* with nothing left she does not write first at all: the log says that the count was used up;
* when he writes again the count starts afresh and she is back to normal.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_bot_text_not_in_her_data,
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
    assert_within_quota,
)
from tests.support.proactive_world import opening_curve, proactive_model
from twin.services import Services

pytestmark = pytest.mark.integration

FRIDAY = date(2026, 10, 9)
QUOTA, RESERVE = 4, 2
PEAKS = {48: 0.6, 49: 0.6, 73: 0.6, 74: 0.6}  # lunch and dinner


def let_her_chase(services: Services) -> None:
    services.settings.proactive.max_chase = 6  # the chase limit has its own scenarios


def steps_of(row: object) -> dict[str, int]:
    return {a["step"]: a["count"] for a in getattr(row, "actions", None) or []}


async def test_a_short_count_merges_bubbles_and_closes_the_door_on_her_own_messages(
    make_world: WorldFactory,
) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 15, 0, tzinfo=UTC),  # 10:00 in Chicago
        model=proactive_model(opening_curve(base=0.02, peaks=PEAKS)),
        quota=QUOTA,
        configure=let_her_chase,
    )
    assert world.services.settings.channel.proactive_reserve == RESERVE
    world.deepseek.book.when("数数", "一", "二", "三", "四", "五")
    world.deepseek.on_her_own["meal"] = (("吃了吗", "吃的什么", "我也饿了"),)
    state = world.channel.session_state

    # ---- a reply of five bubbles into the two that the reserve leaves ----------------------------
    await world.say("数数")
    assert state().remaining_quota == QUOTA
    await world.run_until_idle()
    bubbles = world.persona_said
    assert len(bubbles) == QUOTA - RESERVE  # two
    assert "".join("".join(b.text.split()) for b in bubbles) == "一二三四五"  # nothing is lost
    assert all(" " in b.text for b in bubbles if len(b.text.split()) > 1)  # put together by a space
    first = world.out_rows()[0]
    assert steps_of(first).get("quota_merge") == 3  # five bubbles into two: three merges
    assert state().remaining_quota == RESERVE

    # ---- lunch: she asks in three bubbles; one is kept for a chase, so one is left ---------------
    await world.run_until(world.local(13, 0))
    lunch = [r for r in world.proactive_rows(outcomes=["sent"]) if r.kind == "meal"]
    assert len(lunch) == 1 and lunch[0].bubbles_sent == 1
    sent = [s for s in world.persona_said if s.at >= lunch[0].at]
    assert [s.text for s in sent] == ["吃了吗 吃的什么 我也饿了"]  # put together, not cut
    row = next(r for r in world.out_rows() if r.at >= lunch[0].at)
    assert steps_of(row).get("quota_merge") == 2
    assert state().remaining_quota == 1

    # ---- dinner is the chase: the last message that fits; then the count is used up --------------
    await world.run_until(world.local(19, 0))
    dinner = [r for r in world.proactive_rows(outcomes=["sent"]) if r.at > lunch[0].at]
    assert lunch[0].chase_seq == 0
    assert len(dinner) == 1 and dinner[0].chase_seq == 1  # the one that the reserve kept back
    assert dinner[0].bubbles_sent == 1
    assert state().remaining_quota == 0
    await world.run_until(world.local(23, 0))
    assert [r for r in world.proactive_rows(outcomes=["sent"]) if r.at > dinner[0].at] == []
    refused = [
        r for r in world.proactive_rows(outcomes=["rejected"]) if r.reason == "quota_exhausted"
    ]
    assert refused and all(r.at > dinner[0].at for r in refused)  # the log says: the count is used
    assert not [s for s in world.persona_said if s.at > dinner[0].at + timedelta(minutes=5)]

    # ---- he writes: the count starts afresh ------------------------------------------------------
    await world.say("数数")
    assert state().remaining_quota == QUOTA
    await world.run_until_idle()
    assert len([s for s in world.persona_said if s.at > refused[-1].at]) == QUOTA - RESERVE
    assert_within_quota(world)
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []
