"""Sending a proactive message at her pace, and stopping the way a reply stops (R-PRO-007)."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

import pytest

from tests.support.proactive_world import World
from twin.engine.pacing import PacingModel
from twin.engine.sender import BubbleSender, SentBubble, StopReason
from twin.engine.turns import ReplyMeta
from twin.engine.types import Bubble
from twin.schedule.proactive.send import ProactiveSender, SendOutcome, _written

Wait = Callable[[int, float], Awaitable[bool]]


def bubbles(*texts: str) -> list[Bubble]:
    return [Bubble("text", text) for text in texts]


class Run:
    """One send through the sender of a world, with the pauses under the test's control."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.waits: list[float] = []
        self.firsts: list[tuple[str, str]] = []
        self.user_after: int | None = None  # the user writes during this pause (1-based)
        self.sleep_after: int | None = None
        self.epoch_after: int | None = None
        self.epoch = 0
        self.asleep = False
        parts = world.engine.kit
        assert parts is not None
        self.sender = ProactiveSender(
            sender=BubbleSender(
                parts.channel,
                parts.stickers,
                parts.lookup,
                world.clock,
                world.services.alerts,
                random.Random(3),
            ),
            store=parts.store,
            clock=world.clock,
            arrivals=lambda: world.talk.count,
            wait_for_arrival=self._wait,
            asleep=lambda _moment: self.asleep,
            epoch=lambda: self.epoch,
        )
        self.pacing = PacingModel.from_view(parts.data.view(world.clock.now_utc()))

    async def _wait(self, since: int, seconds: float) -> bool:
        self.waits.append(seconds)
        self.world.clock.tick(seconds)
        number = len(self.waits)
        if number == self.user_after:
            self.world.talk.user_wrote()
        if number == self.sleep_after:
            self.asleep = True
        if number == self.epoch_after:
            self.epoch += 1
        return self.world.talk.count != since

    async def on_first(self, sent: SentBubble, reply_id: str) -> None:
        self.firsts.append((sent.bubble.text, reply_id))

    async def send(self, *texts: str) -> SendOutcome:
        return await self.sender.send(
            bubbles(*texts),
            meta=ReplyMeta("deepseek", actions=({"step": "proactive", "count": 1},)),
            pacing=self.pacing,
            since_arrivals=self.world.talk.count,
            on_first=self.on_first,
        )


@pytest.fixture
def run(calm: World) -> Run:
    calm.user_writes(at=calm.at(9, 0))
    calm.go_to(calm.at(10, 0))
    return Run(calm)


async def test_every_bubble_is_stored_as_it_goes_out_under_one_reply(calm: World, run: Run) -> None:
    outcome = await run.send("在吗", "刚看到一个好玩的", "发你")
    assert outcome.sent == 3 and outcome.stop is None and outcome.interrupted_by is None
    assert outcome.texts == ["在吗", "刚看到一个好玩的", "发你"]
    assert calm.channel.texts == outcome.texts
    assert len(run.firsts) == 1 and run.firsts[0] == ("在吗", outcome.reply_id)
    stored = calm.turns.reply(outcome.reply_id or "")
    assert [t.text for t in stored] == outcome.texts and {t.direction for t in stored} == {"out"}
    assert any(a.get("step") == "proactive" for a in stored[0].actions)
    assert stored[1].actions == () and stored[1].backend is None  # the meta sits on the first one
    assert outcome.first_at is not None and outcome.first_at < calm.clock.now_utc()


async def test_she_types_before_each_bubble_like_in_a_reply(calm: World, run: Run) -> None:
    await run.send("嗯", "哦")
    typing = [item for item in calm.channel.out if item.kind == "typing"]
    assert typing, "no typing indicator although the channel can show one"
    assert len(run.waits) == 2 and all(seconds > 0 for seconds in run.waits)


async def test_a_message_of_the_user_during_a_pause_drops_the_rest(calm: World, run: Run) -> None:
    run.user_after = 2
    outcome = await run.send("在吗", "我跟你说", "刚才那个")
    assert outcome.sent == 1 and outcome.stop is StopReason.INTERRUPTED
    assert outcome.interrupted_by == "user"
    assert calm.channel.texts == ["在吗"]  # what is out stays, the rest is not sent


async def test_falling_asleep_in_the_middle_of_a_message_drops_the_rest(
    calm: World, run: Run
) -> None:
    run.sleep_after = 2
    outcome = await run.send("在吗", "我跟你说", "刚才那个")
    assert outcome.sent == 1 and outcome.interrupted_by == "sleep"
    assert calm.channel.texts == ["在吗"]


async def test_a_restart_or_wake_up_in_the_middle_of_a_message_drops_the_rest(
    calm: World, run: Run
) -> None:
    run.epoch_after = 2
    outcome = await run.send("在吗", "我跟你说", "刚才那个")
    assert outcome.sent == 1 and outcome.interrupted_by == "resume"


async def test_a_pause_that_nothing_interrupts_is_not_an_interruption(
    calm: World, run: Run
) -> None:
    outcome = await run.send("嗯", "哦")
    assert outcome.interrupted_by is None and outcome.sent == 2


async def test_a_refused_first_bubble_leaves_nothing_behind(calm: World, run: Run) -> None:
    stored = calm.turns.count(direction="out")
    calm.channel.window.mark_expired(calm.clock.now_utc())
    outcome = await run.send("在吗", "我跟你说")
    assert outcome.sent == 0 and not outcome.started and outcome.stop is StopReason.EXPIRED
    assert run.firsts == [] and calm.turns.count(direction="out") == stored


async def test_the_platform_count_running_out_stops_the_message_midway(calm: World) -> None:
    calm.user_writes(at=calm.at(9, 0))
    left = calm.channel.window.remaining_quota()
    calm.channel.window.on_outbound(left - 2)
    calm.go_to(calm.at(10, 0))
    run = Run(calm)
    outcome = await run.send("一", "二", "三")
    assert outcome.sent == 2 and outcome.stop is StopReason.QUOTA
    assert calm.channel.texts[-2:] == ["一", "二"]


async def test_a_bubble_that_is_out_is_written_down_even_if_the_task_is_cancelled() -> None:
    done: list[str] = []

    async def writing() -> None:
        for step in ("a", "b", "c"):
            await asyncio.sleep(0)
            done.append(step)

    task = asyncio.ensure_future(_written(writing()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert done == ["a", "b", "c"]


def test_the_outcome_knows_whether_anything_went_out() -> None:
    assert not SendOutcome().started
    assert SendOutcome(sent=1).started
