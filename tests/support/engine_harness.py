"""Test doubles and helpers for the conversation engine (round 09 step 3).

The engine's collaborators are narrow interfaces, so a test can put a scripted one in each place
and keep everything else real: the stores (``bot_turns``, ``conversation_state``) run on the
migrated test database, the clock is the manual clock, and the doubles below stand for the world
outside - the channel (:class:`ScriptedChannel`), the reply pipeline (:class:`ScriptedWriter`), the
day plan (:class:`FixedDay`), the crisis check (:class:`ScriptedCrisis`).  Nothing here is imported
by ``src``.

Time is driven by events, never by sleeping: :func:`run_to_idle` and :func:`advance_to_next_wait`
move the manual clock to the moment the engine says it is waiting for (``engine.waiting_until``).
"""

from __future__ import annotations

import asyncio
import itertools
import random
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from tests.support.clock import DEFAULT_START, ManualClock
from tests.support.reply_view import StaticDataView, make_profile
from tests.support.waiting import wait_until
from twin.channel.base import (
    AuthState,
    Channel,
    ChannelCapabilities,
    InboundMessage,
    MessageKind,
    OutboundKind,
    OutboundResult,
    QuoteTarget,
    SendBypass,
    SessionState,
)
from twin.engine.decision import Decision
from twin.engine.fallback import ShortAnswers
from twin.engine.history import HistoryLoader
from twin.engine.inbound import InboundRenderer
from twin.engine.machine import ConversationEngine
from twin.engine.pacing import PacingModel
from twin.engine.rounds import RoundStore
from twin.engine.safety.crisis import CrisisAssessment, CrisisOutcome
from twin.engine.state_store import ConversationStateStore
from twin.engine.sticker_sender import StickerSender
from twin.engine.turns import BotTurnMessages, BotTurnStore
from twin.engine.types import (
    Bubble,
    PostAction,
    ReplyContext,
    ReplyDraft,
    UsageSummary,
    Violation,
)
from twin.memory.recent import BotMessage, HistoryWindow
from twin.profile.distribution import EmpiricalDistribution
from twin.schedule.plan_model import HerState, StateKind
from twin.schedule.time_service import PlanUnavailableError
from twin.services import Services
from twin.stickers.catalog import StickerCatalog

_MESSAGE_NUMBERS = itertools.count(1)  # channel message ids are unique, whichever channel
START = DEFAULT_START  # 12:00 UTC on a Friday: 07:00 in Chicago


# ---------------------------------------------------------------------------- channel


@dataclass(frozen=True)
class Out:
    """One thing the engine sent through the channel."""

    kind: Literal["text", "image", "typing"]
    text: str | None
    at: datetime
    quote: QuoteTarget | None = None
    data: bytes | None = None
    active: bool | None = None


class ScriptedChannel(Channel):
    """A channel the test controls: it records what is sent and fails when told to."""

    def __init__(
        self,
        clock: ManualClock,
        *,
        quota: int = 8,
        supports_quote: bool = False,
        supports_typing: bool = True,
    ) -> None:
        self.clock = clock
        self.quota = quota
        self.remaining = quota
        self.expired = False
        self.bound = True
        self.auth = AuthState.OK
        self.supports_quote = supports_quote
        self.supports_typing = supports_typing
        self.out: list[Out] = []
        self.results: deque[OutboundResult] = deque()  # answers for the next sends, in order
        self.image_error: Exception | None = None
        self._inbox: asyncio.Queue[InboundMessage | None] = asyncio.Queue()
        self.on_send: Callable[[Out], None] | None = None

    # ------------------------------------------------------------------ the user

    def push(
        self,
        text: str,
        *,
        kind: MessageKind = MessageKind.TEXT,
        at: datetime | None = None,
        message_id: str | None = None,
    ) -> InboundMessage:
        """The user writes ``text`` (the window and the count start afresh, like WeChat)."""
        message = InboundMessage(
            message_id or f"m{next(_MESSAGE_NUMBERS)}", at or self.clock.now_utc(), kind, text=text
        )
        self.remaining = self.quota
        self._inbox.put_nowait(message)
        return message

    def end_input(self) -> None:
        self._inbox.put_nowait(None)

    # ------------------------------------------------------------------ Channel

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def incoming(self) -> AsyncIterator[InboundMessage]:
        while True:
            item = await self._inbox.get()
            if item is None:
                return
            yield item

    def _result(self) -> OutboundResult:
        if self.results:
            return self.results.popleft()
        self.remaining -= 1
        return OutboundResult.success(message_id=f"out{next(_MESSAGE_NUMBERS)}")

    async def send_text(
        self,
        text: str,
        quote: QuoteTarget | None = None,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        result = self._result()
        if result.ok or result.kind is OutboundKind.AMBIGUOUS:
            self._record(Out("text", text, self.clock.now_utc(), quote))
        return result

    async def send_image(
        self,
        data: bytes | Path,
        mime: str,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        if self.image_error is not None:
            raise self.image_error
        result = self._result()
        if result.ok:
            payload = (
                data if isinstance(data, bytes) else await asyncio.to_thread(Path(data).read_bytes)
            )
            self._record(Out("image", None, self.clock.now_utc(), data=payload))
        return result

    async def send_typing(self, active: bool, *, recipient: str | None = None) -> None:
        self._record(Out("typing", None, self.clock.now_utc(), active=active))

    def _record(self, item: Out) -> None:
        self.out.append(item)
        if self.on_send is not None:
            self.on_send(item)

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_quote=self.supports_quote,
            supports_typing=self.supports_typing,
            outbound_quota=self.quota,
            max_text_chars=4000,
        )

    def session_state(self) -> SessionState:
        return SessionState(
            auth=self.auth,
            bound=self.bound,
            last_inbound_at=None,
            outbound_since_inbound=self.quota - self.remaining,
            expired=self.expired,
            remaining_quota=self.remaining,
            window_remaining=None,
            has_context_token=True,
        )

    # ------------------------------------------------------------------ the record

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self.out if item.kind == "text" and item.text is not None]

    @property
    def images(self) -> list[bytes]:
        return [item.data for item in self.out if item.kind == "image" and item.data is not None]


# ---------------------------------------------------------------------------- pipeline


def make_draft(
    *lines: str,
    backend: str = "deepseek",
    cost: float = 0.001,
    no_reply: bool = False,
    fallback: Literal["refused", "violations", "backend_error", "empty"] | None = None,
    reasoning: str | None = None,
    quote: str | None = None,
    stickers: dict[str, str] | None = None,
    thinking: bool = False,
    actions: Sequence[PostAction] = (),
) -> ReplyDraft:
    """A pipeline answer: ``lines`` become bubbles; a line in ``stickers`` is a sticker (by MD5)."""
    table = stickers or {}
    bubbles = tuple(
        Bubble("sticker", line, sticker_md5=table[line], sticker_tag=line)
        if line in table
        else Bubble("text", line)
        for line in lines
    )
    return ReplyDraft(
        bubbles=bubbles,
        quote=quote,
        no_reply=no_reply,
        needs_fallback=fallback is not None,
        fallback_reason=fallback,
        backend=backend,
        thinking=thinking,
        reasoning=reasoning,
        plan=None,
        cost_usd=cost,
        usage=UsageSummary(1, 100, 10, 80, 20),
        timings_ms={"gather": 5, "generate": 20, "post": 1, "total": 26},
        actions=tuple(actions),
        violations=(Violation("ai_self_reference"),) if fallback == "violations" else (),
        attempts=1,
    )


class ScriptedWriter:
    """The reply pipeline of a test: answers from a script, optionally held back by a gate."""

    def __init__(self, *answers: ReplyDraft | BaseException) -> None:
        self.answers: deque[ReplyDraft | BaseException] = deque(answers)
        self.contexts: list[ReplyContext] = []
        self.started = asyncio.Event()
        self.gate: asyncio.Event | None = None
        self.cancelled = 0
        self.calls = 0

    def add(self, *answers: ReplyDraft | BaseException) -> None:
        self.answers.extend(answers)

    async def run(self, context: ReplyContext, data: Any) -> ReplyDraft:
        self.calls += 1
        self.contexts.append(context)
        self.started.set()
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        answer = self.answers.popleft() if self.answers else make_draft("好的")
        if isinstance(answer, BaseException):
            raise answer
        return answer


# --------------------------------------------------------------------------- the world


class FixedDay:
    """Her day as a list of ``(from, kind)`` changes: the state at a moment is the last change."""

    def __init__(self, *changes: tuple[datetime, StateKind], busy: Any = None) -> None:
        self.changes = sorted(changes, key=lambda item: item[0])
        self.busy = busy
        self.unavailable = False

    def her_state(self, at: datetime | None = None) -> HerState:
        if self.unavailable or at is None:
            raise PlanUnavailableError("no plan")
        index = max((i for i, (start, _) in enumerate(self.changes) if start <= at), default=None)
        if index is None:
            raise PlanUnavailableError("before the plan")
        start, kind = self.changes[index]
        end = (
            self.changes[index + 1][0]
            if index + 1 < len(self.changes)
            else start + timedelta(days=1)
        )
        return HerState(kind, start, end, "plan1", self.busy if kind == "busy" else None)

    def state_at(self, moment: datetime) -> HerState:
        """The :class:`~twin.schedule.time_service.StateSource` form of :meth:`her_state`."""
        return self.her_state(moment)


class ClockData:
    """A data source whose view follows the clock (the static view of the pipeline tests)."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock

    def view(self, at: datetime | None = None) -> StaticDataView:
        return StaticDataView(at=at or self.clock.now_utc(), profile=make_profile())


class ScriptedCrisis:
    """The crisis check: words that count as a hit, and whether the judgement confirms it."""

    def __init__(self, *words: str, confirm: bool = True) -> None:
        self.words = words
        self.confirm = confirm
        self.handled: list[list[str]] = []

    def screen(self, texts: Sequence[str]) -> int:
        return sum(1 for text in texts for word in self.words if word in text)

    async def handle(
        self, texts: Sequence[str], context: Sequence[str] = ()
    ) -> CrisisOutcome | None:
        self.handled.append(list(texts))
        if not self.confirm:
            return None
        return CrisisOutcome(
            CrisisAssessment(True, "high", 1, True),
            ("我想先停一下。", "你现在还好吗？", "也可以联系 988。"),
            ("988",),
            False,
        )


class Alerts:
    """An alert sink that keeps what it is given."""

    def __init__(self) -> None:
        self.raised: list[tuple[str, str, dict[str, Any] | None]] = []

    def raise_alert(
        self,
        category: str,
        title: str,
        *,
        severity: str = "warning",
        detail: dict[str, Any] | None = None,
        dedup_key: str | None = None,
    ) -> None:
        self.raised.append((category, title, detail))

    @property
    def categories(self) -> list[str]:
        return [category for category, _, _ in self.raised]


def reference_pacing(
    *,
    latency_s: float = 20.0,
    gap_s: float = 4.0,
    user_gap_s: float = 10.0,
    seconds_per_char: float | None = 0.5,
) -> PacingModel:
    """A pacing without randomness: every latency, pause and user pause is a single number."""

    def only(value: float) -> EmpiricalDistribution:
        return EmpiricalDistribution.from_samples([value] * 5, discrete=False)

    return PacingModel(
        latency_all=only(latency_s),
        burst_gap=only(gap_s),
        user_burst_gap=only(user_gap_s),
        seconds_per_char=seconds_per_char,
    )


@dataclass
class Harness:
    """An engine on the test database with its doubles."""

    engine: ConversationEngine
    channel: ScriptedChannel
    writer: ScriptedWriter
    day: FixedDay
    clock: ManualClock
    services: Services
    crisis: ScriptedCrisis
    alerts: Alerts
    store: BotTurnStore
    state: ConversationStateStore
    queued: list[list[BotMessage]] = field(default_factory=list)
    commands: Any = None

    async def message(self, text: str, **kwargs: Any) -> None:
        """The user writes: the message reaches the engine the way the channel hands it over."""
        await self.engine.handle_message(self.channel.push(text, **kwargs))

    async def settle(self) -> None:
        await self.engine.drain()


def build_harness(
    services: Services,
    clock: ManualClock,
    *,
    writer: ScriptedWriter | None = None,
    channel: ScriptedChannel | None = None,
    day: FixedDay | None = None,
    pacing: PacingModel | None = None,
    crisis: ScriptedCrisis | None = None,
    commands: Any = None,
    answers: Sequence[str] = ("嗯嗯", "好的"),
    seed: int = 7,
    short_answers: ShortAnswers | None = None,
    renderer: InboundRenderer | None = None,
) -> Harness:
    """Wire an engine the way ``build_engine`` does, with the doubles of this module."""
    chosen_channel = channel or ScriptedChannel(clock)
    chosen_writer = writer or ScriptedWriter()
    chosen_day = day or FixedDay((START - timedelta(days=1), "free"))
    chosen_crisis = crisis or ScriptedCrisis()
    alerts = Alerts()
    state = ConversationStateStore(services.db, services.clock)
    store = BotTurnStore(services.db, services.clock)
    reader = BotTurnMessages(services.db)
    queued: list[list[BotMessage]] = []

    def queue(turns: Sequence[BotMessage]) -> str | None:
        queued.append(list(turns))
        return f"job{len(queued)}"

    catalog = StickerCatalog(services)
    known = (
        short_answers
        if short_answers is not None
        else ShortAnswers(tuple((text, 10) for text in answers))
    )
    chosen_pacing = pacing or reference_pacing()
    engine = ConversationEngine(
        channel=chosen_channel,
        store=store,
        rounds=RoundStore(services.db),
        state=state,
        history=HistoryLoader(reader, HistoryWindow(30, 40), state),
        reader=reader,
        writer=chosen_writer,
        data=ClockData(clock),
        time=chosen_day,  # type: ignore[arg-type]
        renderer=renderer or InboundRenderer(services),
        crisis=chosen_crisis,
        stickers=StickerSender(chosen_channel, services.media),
        lookup=catalog.get,
        short_answers=lambda: known,
        runtime=services.runtime,
        alerts=alerts,
        clock=clock,
        settings=services.settings,
        rng=random.Random(seed),
        commands=commands,
        pacing=lambda view: chosen_pacing,
        queue_extraction=queue,
    )
    return Harness(
        engine,
        chosen_channel,
        chosen_writer,
        chosen_day,
        clock,
        services,
        chosen_crisis,
        alerts,
        store,
        state,
        queued,
        commands,
    )


# ------------------------------------------------------------------------ driving time


def seconds_until(clock: ManualClock, moment: datetime) -> float:
    return max(0.0, (moment - clock.now_utc()).total_seconds())


async def advance_to_next_wait(engine: ConversationEngine, clock: ManualClock) -> datetime:
    """Wait until the engine is waiting for a moment, then move the clock exactly there."""
    await wait_until(
        lambda: engine.waiting_until is not None and not engine.pending_wake,
        limit_s=10.0,
        interval=0.002,
    )
    target = engine.waiting_until
    assert target is not None
    await clock.advance(seconds_until(clock, target))
    return target


async def run_to_idle(
    engine: ConversationEngine,
    clock: ManualClock,
    *,
    limit_s: float = 20.0,
    until: Callable[[], bool] | None = None,
) -> int:
    """Let the engine finish the round: advance to every wait until idle (or ``until`` holds).

    Returns how many waits were skipped.  Only real time is bounded (``limit_s``): the engine's
    own waiting is skipped by moving the manual clock, so a night's sleep costs nothing.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_s
    steps = 0
    while loop.time() < deadline:
        snap = await asyncio.to_thread(engine.snapshot)
        if until is not None and until():
            return steps
        if until is None and snap.state == "IDLE" and not snap.pending:
            return steps
        waiting = engine.waiting_until
        if waiting is not None and not engine.pending_wake:  # not a wait that is about to end
            await clock.advance(seconds_until(clock, waiting))
            steps += 1
        else:
            await asyncio.sleep(0.003)
    raise AssertionError("the engine did not settle")


async def wait_for_state(engine: ConversationEngine, state: str, limit_s: float = 10.0) -> None:
    """Wait (really) until the stored state is ``state``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_s
    while (await asyncio.to_thread(engine.snapshot)).state != state:
        if loop.time() > deadline:
            raise AssertionError(f"the engine never reached {state}")
        await asyncio.sleep(0.003)


def decision_of(engine: ConversationEngine) -> Decision | None:
    from twin.engine.roundstate import RoundData

    return RoundData.of(engine.snapshot()).decision
