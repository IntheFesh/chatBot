"""Sending a proactive message the way she sends a reply (R-PRO-007, R-ENG-009).

:class:`ProactiveSender` puts the bubbles of a checked draft through the engine's own
:class:`~twin.engine.sender.BubbleSender`: before each bubble the pause between two messages of a
burst and the time to type it, "typing" where the channel shows it, the same gate for the login,
the window and the count, a sticker only through the sticker sender.  What differs from a reply:

* **nobody is waiting**, so the first bubble comes after its typing time alone;
* **the user may start to talk meanwhile**: if a message of his arrives, the bubbles that are out
  stay and the rest is dropped - what he wrote is answered by the engine, not by this message;
* **she may fall asleep meanwhile**: the rest is dropped as soon as the plan says deep sleep (a
  long message does not run on into the night), and so is the rest after a restart or a wake-up
  from sleep (:attr:`epoch` moved: the candidates are void, R-SCH-005);
* every bubble is written to ``bot_turns`` the moment it is out, as the first bubble of a reply
  that is flagged as proactive in its actions, so the conversation the engine and the memory
  read is complete even if the program stops in the middle.

The scheduler is told about the first bubble at once (``on_first``): that is when the log row is
written, before the rest of the message is sent.

*A stop of the application* (:meth:`ProactiveSender.hold`, D-756, as the engine's D-721): no new
bubble is begun, and a bubble that is being handed to the channel is let finish and is written
down - for at most a few real seconds.  A task that is cancelled while the channel still has a
bubble (the grace ran out: the channel never answered) writes it down as sent "in doubt"
(``bubble_in_doubt`` in the actions; the rule of D-552: a bubble said twice hurts more than one
that is missing), so the log, the spacing and the day's count know of it after the restart.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from twin.clock import Clock
from twin.engine.pacing import PacingModel
from twin.engine.sender import BubbleSender, OutBubble, SendReport, SentBubble, StopReason
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.engine.types import Bubble
from twin.ops.logging import get_logger

log = get_logger("twin.schedule.proactive.send")

FirstBubble = Callable[[SentBubble, str], Awaitable[None]]


@dataclass
class SendOutcome:
    """What was sent and how it ended."""

    sent: int = 0
    skipped: int = 0
    reply_id: str | None = None
    first_at: datetime | None = None
    stop: StopReason | None = None
    interrupted_by: str | None = None  # user | sleep | resume | stop (stop INTERRUPTED)
    texts: list[str] = field(default_factory=list)

    @property
    def started(self) -> bool:
        return self.sent > 0


class ProactiveSender:
    """Sends the bubbles of a proactive message (see the module description)."""

    def __init__(
        self,
        *,
        sender: BubbleSender,
        store: BotTurnStore,
        clock: Clock,
        arrivals: Callable[[], int],
        wait_for_arrival: Callable[[int, float], Awaitable[bool]],
        asleep: Callable[[datetime], bool],
        epoch: Callable[[], int],
    ) -> None:
        self._sender = sender
        self._store = store
        self._clock = clock
        self._arrivals = arrivals
        self._wait_for_arrival = wait_for_arrival
        self._asleep = asleep
        self._epoch = epoch
        self._holding = False  # the application is stopping: no new bubble is begun
        # clear while a bubble is being handed to the channel and written down (D-756)
        self._safe_to_cancel = asyncio.Event()
        self._safe_to_cancel.set()

    async def hold(self, grace_s: float) -> None:
        """Begin no new bubble; wait up to ``grace_s`` real seconds for the one being handed over.

        The stop of the application calls this before it cancels the task (D-756); the grace is
        real time because the channel's answer takes real (network) time.
        """
        self._holding = True
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._safe_to_cancel.wait(), grace_s)

    def release(self) -> None:
        """The stop is over: a later call of :meth:`send` sends again."""
        self._holding = False

    async def send(
        self,
        bubbles: Sequence[Bubble],
        *,
        meta: ReplyMeta,
        pacing: PacingModel,
        since_arrivals: int,
        on_first: FirstBubble,
    ) -> SendOutcome:
        """Send ``bubbles`` in order (see the module description)."""
        outcome = SendOutcome()
        out = [
            OutBubble("sticker", bubble.text, bubble.sticker_md5)
            if bubble.is_sticker and bubble.sticker_md5
            else OutBubble("text", bubble.text)
            for bubble in bubbles
        ]
        epoch = self._epoch()
        reply: list[str | None] = [None]
        in_flight: list[tuple[OutBubble, datetime] | None] = [None]

        async def wait(seconds: float) -> bool:
            # a pause: the channel has nothing of this message in hand now (a sticker that was
            # skipped, a send that never left the machine and is about to be repeated)
            in_flight[0] = None
            self._safe_to_cancel.set()
            if self._holding:
                outcome.interrupted_by = "stop"
                return True
            if await self._wait_for_arrival(since_arrivals, seconds):
                outcome.interrupted_by = "user"
                return True
            if self._epoch() != epoch:
                outcome.interrupted_by = "resume"
                return True
            if self._asleep(self._clock.now_utc()):
                outcome.interrupted_by = "sleep"
                return True
            return False

        async def about_to_send(bubble: OutBubble) -> None:
            self._safe_to_cancel.clear()  # from here until ``keep``: a stop lets the send finish
            in_flight[0] = (bubble, self._clock.now_utc())

        async def keep(sent: SentBubble) -> None:
            # the bubble is out: write it down even if the program is stopped meanwhile
            try:
                await _written(self._record(sent, meta, reply, outcome, on_first))
            finally:
                in_flight[0] = None
                self._safe_to_cancel.set()

        try:
            report: SendReport = await self._sender.send(
                out, pacing=pacing, wait=wait, on_sent=keep, on_sending=about_to_send
            )
        except asyncio.CancelledError:
            doubt = in_flight[0]
            if doubt is not None:  # cancelled while the channel had it: it counts as sent
                in_flight[0] = None
                await _written(self._in_doubt(doubt, meta, reply, outcome, on_first))
            raise
        finally:
            self._safe_to_cancel.set()
        outcome.skipped = len(report.skipped)
        outcome.stop = report.stop
        if report.stop is not StopReason.INTERRUPTED:
            outcome.interrupted_by = None
        return outcome

    async def _in_doubt(
        self,
        doubt: tuple[OutBubble, datetime],
        meta: ReplyMeta,
        reply: list[str | None],
        outcome: SendOutcome,
        on_first: FirstBubble,
    ) -> None:
        """A bubble the channel had when the task was cancelled: written down as sent (D-756)."""
        bubble, began = doubt
        log.warning("proactive_bubble_in_doubt", kind=bubble.kind)
        noted = replace(meta, actions=(*meta.actions, {"step": "bubble_in_doubt", "count": 1}))
        first = reply[0] is None
        await self._record(
            SentBubble(bubble, began, None, ambiguous=True), noted, reply, outcome, on_first
        )
        if not first and reply[0] is not None:
            await asyncio.to_thread(self._store.update_meta, reply[0], noted)

    async def _record(
        self,
        sent: SentBubble,
        meta: ReplyMeta,
        reply: list[str | None],
        outcome: SendOutcome,
        on_first: FirstBubble,
    ) -> None:
        bubble = OutboundBubble(
            sent.bubble.text,
            sent.at,
            kind=sent.bubble.kind,
            sticker_md5=sent.bubble.sticker_md5,
            external_id=sent.message_id,
        )
        first = reply[0] is None
        if first:
            row = await asyncio.to_thread(self._store.add_bubble, bubble, meta=meta)
            reply[0] = row.reply_id
        else:
            await asyncio.to_thread(self._store.add_bubble, bubble, reply_id=reply[0])
        outcome.sent += 1
        outcome.reply_id = reply[0]
        outcome.texts.append(sent.bubble.text)
        if first and reply[0] is not None:
            outcome.first_at = sent.at
            await on_first(sent, reply[0])


async def _written(work: Awaitable[None]) -> None:
    """Finish writing down a bubble that is out, even if the task is cancelled meanwhile."""
    task = asyncio.ensure_future(work)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise
