"""SENDING: the bubbles of a reply, one by one, the way she sends them (R-ENG-009, R-STK-007).

:class:`BubbleSender` sends the bubbles of one reply through the channel:

* before each bubble it waits as long as she would - her pause between two messages of a burst plus
  the time to type the bubble (:meth:`~twin.engine.pacing.PacingModel.bubble_interval`) - showing
  "typing" while the channel can (``capabilities().supports_typing``);
* the wait ends early when the user writes again: the bubbles that are out stay, the rest is not
  sent (the engine writes the rest again with "what she already said", R-ENG-009);
* before each bubble it asks the channel whether the user is still bound, the login and the window
  are alive and a message is left in the quota (``session_state()``); a session that has expired
  stops the reply and raises an alert instead of trying again (R-CH-008);
* a sticker goes through :class:`~twin.engine.sticker_sender.StickerSender` and nothing else;
* what the channel answers is handled by its kind: a send that never left the machine is repeated
  twice at a short distance, a send whose outcome is unknown is not repeated (it would be said
  twice), a refusal of the window or the login ends the reply, a sticker that cannot be sent is
  skipped.

Every bubble that goes out is reported to ``on_sent`` at once, so the engine can store it
(``bot_turns``, ``conversation_state``) before the next one starts: a process that dies between
two bubbles resumes after the last one that was stored.  A process that dies *inside* a send - the
bubble may be on the user's phone or not - would otherwise say it twice after the restart, so the
engine is told before each send (``on_sending``) and, finding that note unanswered, takes the
bubble as sent (an unknown outcome is never repeated, as above).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal

from twin.channel.base import (
    AuthState,
    CapabilityNotSupported,
    Channel,
    ChannelError,
    MediaNotAllowed,
    OutboundKind,
    OutboundResult,
    QuoteTarget,
    RecipientNotAllowed,
)
from twin.clock import Clock
from twin.engine.pacing import PacingModel
from twin.engine.sticker_sender import StickerSender
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger
from twin.stickers.catalog import StickerRecord

log = get_logger("twin.engine.sender")

NETWORK_RETRIES = 2  # a send that never left the machine is repeated this often
NETWORK_RETRY_S = (5.0, 15.0)
ALERT_SESSION = "channel_session_expired"
ALERT_QUOTA = "channel_quota_exhausted"
ALERT_UNBOUND = "channel_unbound"
ALERT_SEND = "channel_send_failed"


class StopReason(StrEnum):
    """Why a reply stopped before its last bubble."""

    INTERRUPTED = "interrupted"  # the user wrote again
    EXPIRED = "expired"  # the platform session (window, login) is over
    QUOTA = "quota"  # no message is left in the platform's count
    UNBOUND = "unbound"  # nobody is bound as the user
    FAILED = "failed"  # the channel could not send it


@dataclass(frozen=True)
class OutBubble:
    """One bubble to send: a line of text, or a sticker of the library by its MD5."""

    kind: Literal["text", "sticker"]
    text: str
    sticker_md5: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "sticker" and not self.sticker_md5:
            raise ValueError("a sticker bubble needs the MD5 of its sticker")

    @property
    def chars(self) -> int:
        return len(self.text) if self.kind == "text" else 0


@dataclass(frozen=True)
class SentBubble:
    """A bubble that is out."""

    bubble: OutBubble
    at: datetime
    message_id: str | None
    ambiguous: bool = False  # sent, but the channel could not say whether it arrived


@dataclass
class SendReport:
    """What :meth:`BubbleSender.send` did."""

    sent: list[SentBubble] = field(default_factory=list)
    skipped: list[OutBubble] = field(default_factory=list)
    stop: StopReason | None = None
    rest: list[OutBubble] = field(default_factory=list)  # bubbles that were not sent


Wait = Callable[[float], Awaitable[bool]]  # sleeps; True when the user wrote in the meantime
OnSent = Callable[[SentBubble], Awaitable[None]]
OnSending = Callable[[OutBubble], Awaitable[None]]
StickerLookup = Callable[[str], StickerRecord | None]


class BubbleSender:
    """Sends a reply bubble by bubble (see the module description)."""

    def __init__(
        self,
        channel: Channel,
        stickers: StickerSender,
        lookup: StickerLookup,
        clock: Clock,
        alerts: AlertSink,
        rng: random.Random,
    ) -> None:
        self._channel = channel
        self._stickers = stickers
        self._lookup = lookup
        self._clock = clock
        self._alerts = alerts
        self._rng = rng

    # ------------------------------------------------------------------------ the loop

    async def send(
        self,
        bubbles: Sequence[OutBubble],
        *,
        pacing: PacingModel,
        wait: Wait,
        on_sent: OnSent,
        on_sending: OnSending | None = None,
        first_of_reply: bool = True,
        quote: QuoteTarget | None = None,
        paced: bool = True,
    ) -> SendReport:
        """Send ``bubbles`` in order.

        ``on_sending`` is told which bubble is about to be handed to the channel, before it is;
        ``first_of_reply`` is false when the first of them continues a reply that is already
        under way; ``quote`` goes with the first text bubble; ``paced=False`` (the fixed
        out-of-role answer to a crisis, R-SAFE-001) sends without waiting.
        """
        report = SendReport()
        typing = paced and self._channel.capabilities().supports_typing
        quote_pending = quote
        for index, bubble in enumerate(bubbles):
            rest = list(bubbles[index:])
            if paced:
                delay = pacing.bubble_interval(
                    self._rng, bubble.chars, first=first_of_reply and index == 0
                )
                if typing:
                    await self._typing(True)
                if await wait(delay):
                    await self._typing(False)
                    report.stop, report.rest = StopReason.INTERRUPTED, rest
                    return report
            blocked = await self._gate()
            if blocked is not None:
                await self._typing(False)
                report.stop, report.rest = blocked, rest
                return report
            quote_for_bubble = quote_pending if bubble.kind == "text" else None
            if on_sending is not None:
                await on_sending(bubble)
            outcome = await self._deliver(bubble, quote_for_bubble, wait)
            if isinstance(outcome, StopReason):
                await self._typing(False)
                report.stop, report.rest = outcome, rest
                return report
            if outcome is None:
                report.skipped.append(bubble)
                continue
            if quote_for_bubble is not None:
                quote_pending = None
            sent = SentBubble(bubble, self._clock.now_utc(), outcome.message_id, outcome.ambiguous)
            report.sent.append(sent)
            await on_sent(sent)
        return report

    # -------------------------------------------------------------------- the checks

    async def _gate(self) -> StopReason | None:
        """May the next bubble go out?  A "no" is alerted once and ends the reply."""
        try:
            state = await asyncio.to_thread(self._channel.session_state)
        except ChannelError:
            return self._block(StopReason.FAILED, ALERT_SEND, "the channel state is not readable")
        if not state.bound:
            return self._block(StopReason.UNBOUND, ALERT_UNBOUND, "nobody is bound as the user")
        if state.auth is not AuthState.OK or state.expired:
            return self._block(
                StopReason.EXPIRED,
                ALERT_SESSION,
                "the WeChat session is over: send the bot a message from the phone",
            )
        if state.remaining_quota <= 0:
            return self._block(
                StopReason.QUOTA, ALERT_QUOTA, "the platform's message count is used up"
            )
        return None

    def _block(self, reason: StopReason, category: str, title: str) -> StopReason:
        log.warning("reply_blocked", reason=reason.value)
        self._alerts.raise_alert(category, title, severity="warning", dedup_key=category)
        return reason

    async def _typing(self, active: bool) -> None:
        try:
            await self._channel.send_typing(active)
        except (ChannelError, OSError) as exc:  # the indicator is a nicety, never a reason to stop
            log.info("typing_indicator_failed", reason=type(exc).__name__)

    # ----------------------------------------------------------------------- sending

    async def _deliver(
        self, bubble: OutBubble, quote: QuoteTarget | None, wait: Wait
    ) -> _Delivered | StopReason | None:
        """Send one bubble; the result, a reason to stop, or ``None`` for a skipped sticker."""
        for attempt in range(NETWORK_RETRIES + 1):
            try:
                result = await self._once(bubble, quote)
            except RecipientNotAllowed:
                return self._block(StopReason.UNBOUND, ALERT_UNBOUND, "nobody is bound as the user")
            except (MediaNotAllowed, CapabilityNotSupported) as exc:
                log.warning("bubble_not_sendable", reason=type(exc).__name__, kind=bubble.kind)
                return None
            except ChannelError as exc:
                return self._block(StopReason.FAILED, ALERT_SEND, type(exc).__name__)
            if result is None:
                return None  # a sticker the library no longer has
            if result.ok:
                return _Delivered(result.message_id, False)
            kind = result.kind
            if kind is OutboundKind.AMBIGUOUS:
                return _Delivered(result.message_id, True)  # not repeated: it may have arrived
            if kind in (OutboundKind.WINDOW_REJECTED, OutboundKind.AUTH_EXPIRED) or (
                result.session_expired
            ):
                return self._block(
                    StopReason.EXPIRED,
                    ALERT_SESSION,
                    "the WeChat session is over: send the bot a message from the phone",
                )
            if kind is OutboundKind.NETWORK and attempt < NETWORK_RETRIES:
                low, high = NETWORK_RETRY_S
                if await wait(self._rng.uniform(low, high)):
                    return StopReason.INTERRUPTED
                continue
            if bubble.kind == "sticker":
                log.warning("sticker_not_sent", kind=kind.value, reason=result.reason)
                return None
            return self._block(StopReason.FAILED, ALERT_SEND, f"{kind.value}: {result.reason}")
        return self._block(StopReason.FAILED, ALERT_SEND, "the send was repeated and failed")

    async def _once(self, bubble: OutBubble, quote: QuoteTarget | None) -> OutboundResult | None:
        if bubble.kind == "text":
            return await self._channel.send_text(bubble.text, quote)
        sticker = await asyncio.to_thread(self._lookup, bubble.sticker_md5 or "")
        if sticker is None:
            return None
        return await self._stickers.send_sticker(sticker)


@dataclass(frozen=True)
class _Delivered:
    message_id: str | None
    ambiguous: bool
