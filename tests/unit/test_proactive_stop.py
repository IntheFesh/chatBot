"""A stop of the application while a proactive bubble is being handed over (round 18, B5, D-756).

Before this item ``ProactiveComponent.stop`` cancelled its task at once: a bubble the channel
already had - on the user's phone a moment later - was never written to ``bot_turns`` nor to the
proactive log, so after the restart the spacing, the chase and the day's count did not know of
it.  Now the stop does what the engine's stop does (D-721): no new bubble is begun, the one that
is being handed to the channel is let finish and is written down, for at most a few real seconds;
a channel that never answers is given up after that grace and its bubble is written down as sent
"in doubt" (the rule of D-552: a bubble said twice hurts more than one that is missing).
"""

from __future__ import annotations

import asyncio
from typing import Any

from tests.support.proactive_world import World
from tests.support.waiting import wait_until
from twin.channel.base import OutboundResult, QuoteTarget, SendBypass
from twin.schedule.proactive.component import ProactiveComponent
from twin.schedule.proactive.types import TriggerKind

FIRST, SECOND = "刚刚在图书馆看文献", "有点困"  # the planner's message for a share


class Gate:
    """Holds the channel's text sends: ``reached`` once one is waiting, ``open`` lets it go."""

    def __init__(self, world: World) -> None:
        self.reached = asyncio.Event()
        self.open = asyncio.Event()
        self.calls = 0
        original = world.channel.send_text

        async def held(
            text: str,
            quote: QuoteTarget | None = None,
            *,
            recipient: str | None = None,
            bypass: SendBypass | None = None,
        ) -> OutboundResult:
            self.calls += 1
            self.reached.set()
            await self.open.wait()
            return await original(text, quote, recipient=recipient, bypass=bypass)

        world.channel.send_text = held  # type: ignore[method-assign]


def share_is_due(world: World) -> None:
    world.user_writes(at=world.at(20, 0, day=8))
    world.put(TriggerKind.SHARE, world.at(10, 0))
    world.go_to(world.at(10, 5))


def out_rows(world: World) -> list[Any]:
    reply = world.turns.latest_reply()
    return list(reply) if reply else []


async def test_r18_b5_a_stop_lets_the_bubble_being_sent_finish_and_books_it(calm: World) -> None:
    """R18-B5: ``stop()`` waits for the bubble the channel has, then it is in both books."""
    share_is_due(calm)
    gate = Gate(calm)
    component = ProactiveComponent(calm.scheduler, calm.services, calm.rig.kit, stop_grace_s=30.0)
    await component.start()  # the first tick is made at once: the share goes out
    await wait_until(gate.reached.is_set)
    stopping = asyncio.create_task(component.stop())
    for _ in range(25):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    assert not stopping.done()  # it waits for the bubble that is on its way
    gate.open.set()
    await asyncio.wait_for(stopping, 10)
    assert calm.channel.texts == [FIRST] and gate.calls == 1  # nothing new after the stop
    rows = out_rows(calm)
    assert [row.text for row in rows] == [FIRST]
    assert any(a.get("step") == "proactive" for a in rows[0].actions)
    (logged,) = calm.sent_rows()
    assert logged.kind == "share" and logged.reply_id == rows[0].reply_id


async def test_r18_b5_a_channel_that_never_answers_is_given_up_after_the_grace(
    calm: World,
) -> None:
    """R18-B5: the wait is bounded; the bubble is booked as sent, in doubt (D-552)."""
    share_is_due(calm)
    gate = Gate(calm)  # never opened
    component = ProactiveComponent(calm.scheduler, calm.services, calm.rig.kit, stop_grace_s=0.2)
    await component.start()
    await wait_until(gate.reached.is_set)
    loop = asyncio.get_running_loop()
    began = loop.time()
    await asyncio.wait_for(component.stop(), 10)
    assert 0.15 <= loop.time() - began < 9  # the grace was given, then the task was cancelled
    rows = out_rows(calm)
    assert [row.text for row in rows] == [FIRST]  # it may be on his phone: it counts
    assert rows[0].external_id is None
    steps = [a.get("step") for a in rows[0].actions]
    assert "proactive" in steps and "bubble_in_doubt" in steps
    (logged,) = calm.sent_rows()
    assert logged.reply_id == rows[0].reply_id


async def test_r18_b5_a_stop_between_two_bubbles_does_not_wait_and_starts_nothing(
    calm: World,
) -> None:
    """R18-B5: nothing is being handed over: the stop is at once, the next bubble never starts."""
    share_is_due(calm)
    sent: list[str] = []
    stopped = asyncio.Event()
    component = ProactiveComponent(calm.scheduler, calm.services, calm.rig.kit, stop_grace_s=30.0)

    def first_out(item: Any) -> None:
        if item.kind == "text":
            sent.append(item.text)

    calm.channel.on_send = first_out
    stop_task: list[asyncio.Task[None]] = []

    async def stop_after_first(since: int, seconds: float) -> bool:
        if sent and not stop_task:
            stop_task.append(asyncio.create_task(component.stop()))
            await stopped.wait()
        calm.clock.tick(seconds)
        return False

    calm.scheduler._sender._wait_for_arrival = stop_after_first  # type: ignore[attr-defined]
    await component.start()
    await wait_until(lambda: bool(stop_task))
    loop = asyncio.get_running_loop()
    began = loop.time()
    await asyncio.wait_for(stop_task[0], 10)
    assert loop.time() - began < 9
    stopped.set()
    assert calm.channel.texts == [FIRST]


async def test_r18_b5_after_a_stop_the_scheduler_is_not_left_unable_to_send(calm: World) -> None:
    """R18-B5: the hold on new bubbles lasts as long as the stop, not beyond it."""
    component = ProactiveComponent(calm.scheduler, calm.services, calm.rig.kit, first_tick=False)
    await component.start()
    await component.stop()
    share_is_due(calm)
    report = await calm.scheduler.tick()
    assert report.outcome == "sent" and calm.channel.texts == [FIRST, SECOND]
