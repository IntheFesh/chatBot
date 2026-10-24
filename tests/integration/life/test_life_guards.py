"""The guard that closes every scenario for rule 7 is not blind (CLAUDE.md rule 7, R-STO-007).

Every end-to-end scenario ends with ``assert_bot_text_not_in_her_data``: the conversation with the
bot is not among her real messages and not in the library of examples.  A guard that cannot fail
proves nothing, so here a story is made to leak on purpose - a window the library did not have, a
line of the bot among her real messages - and both the cheap guard and the heavy one must say so.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.embedding import HashingBackend
from tests.support.life_checks import (
    IsolationSnapshot,
    assert_bot_text_not_in_her_data,
    assert_bot_text_stays_out,
    snapshot_isolation,
)
from tests.support.synth_chat import MessageWriter

pytestmark = pytest.mark.integration


async def test_the_guard_of_rule_7_notices_a_message_or_a_window_that_was_not_there(
    make_world: WorldFactory, embedder: HashingBackend
) -> None:
    world = await make_world(datetime(2026, 10, 9, 15, 0, tzinfo=UTC))  # Friday 10:00 in Chicago
    await world.say("今天好冷呀")
    await world.run_until_idle()
    assert world.persona_said, "she did not answer"
    assert_bot_text_not_in_her_data(world)  # the story as it is: clean
    start = world.real_data
    assert start is not None and start.message_ids and start.window_ids

    # a window that the library did not have when the story began
    fewer = IsolationSnapshot(start.message_ids, frozenset(sorted(start.window_ids)[1:]))
    with pytest.raises(AssertionError, match="entered the library"):
        assert_bot_text_not_in_her_data(world, fewer)

    # a line of the conversation, stored as one more of her real messages
    writer = MessageWriter(world.services)
    writer.add(world.now - timedelta(days=1), True, "text", world.persona_said[0].text)
    writer.store(append=True)
    with pytest.raises(AssertionError, match="wrote real messages"):
        assert_bot_text_not_in_her_data(world)

    # the heavy guard looks at the text itself, even where the number of rows is accepted
    accepted = snapshot_isolation(world)
    with pytest.raises(AssertionError):
        await assert_bot_text_stays_out(world, accepted, embedder)
