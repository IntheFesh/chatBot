"""The conversation state machine (R-ENG-001 to R-ENG-004, R-ENG-009 to R-ENG-011, R-SCH-005).

::

    IDLE --message--> COLLECTING --quiet--> DECIDING --due--> GENERATING --draft--> SENDING --> IDLE
                          ^                    ^                  |                    |
                          |                    +-- retry / pause / new message ------+
                          +----- a new message cancels the generation ---------------+

``ConversationEngine`` serves the one conversation with the bound user.  Every transition is
written to ``conversation_state`` (state, the ids waiting for an answer, the bubbles already out,
the planned send time, and the typed notes of :mod:`twin.engine.roundstate`), so a process that
stops at any point continues from the stored state: :meth:`ConversationEngine.start` first makes
the stored state fit the moment ("recovery", below) and the driver resumes from there.

States
------
``IDLE``        nothing waits for an answer.
``COLLECTING``  the user's messages are gathered until he has been quiet for
                ``engine.quiet_window_s``
                (R-ENG-002; the user's own p75 pause when ``engine.quiet_window_adaptive`` is on) or
                ``engine.max_wait_s`` have passed since the first one.  The messages are screened
                for a crisis (R-SAFE-001) the moment collecting ends - never later than a night's
                sleep - and a confirmed crisis is answered at once, out of the role.
``DECIDING``    when does she answer?  :class:`~twin.engine.decision.Decider` draws it from her
                state
                (R-ENG-003, R-ENG-004, R-SCOPE-006) and the state waits for it, ``planned_send_at``
                on disk.  A message that comes meanwhile joins the round; after 80 % of the wait a
                short extra pause is added, before that the drawn time stands.  The pause setting
                (``engine.paused_until``) holds the round back; after it she has "just seen" the
                messages.
``GENERATING``  :class:`~twin.engine.pipeline.ReplyPipeline` writes the reply as an
                ``asyncio`` task.  A message of the user cancels it (what was paid for stays in the
                cost ledger, the cancellation is noted in the reply's actions) and the round goes
                back to COLLECTING with the new message in.  A failed reply is tried again 2 to 10
                minutes later, three times; then a short natural answer of hers goes out and an
                alert is written (R-ENG-010).  Nothing but her words is ever sent.
``SENDING``     :class:`~twin.engine.sender.BubbleSender` sends the bubbles at her pace.  A message
                of the user stops it: what is out stays, the rest is dropped and the round goes on
                at DECIDING with "what she already said" (R-ENG-009).

Recovery after a restart (and on ``Resumed``, R-SCH-005): ``COLLECTING`` starts its quiet window
afresh (the messages that piled up in the channel while the process was away join the same round);
``DECIDING`` keeps a planned time that lies ahead (the "after she wakes" queue survives) and draws a
new one when it has passed - never a reply the moment the machine wakes; ``GENERATING`` goes to
``DECIDING`` for a fresh draw; ``SENDING`` goes on with the bubbles that were not out, or, if the
user wrote meanwhile, continues like an interruption.

Commands (:class:`~twin.engine.command_port.CommandPort`) are answered the moment they arrive,
without a delay, outside the round, and are marked ``is_command`` (never memory or training data).
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Protocol

from twin.channel.base import (
    Channel,
    ChannelError,
    InboundMessage,
    MessageKind,
    OutboundKind,
    QuoteTarget,
)
from twin.clock import Clock
from twin.config.runtime import (
    BACKEND_ACTIVE,
    ENGINE_PAUSED_UNTIL,
    SHOW_THINKING,
    THINKING_CHAT,
    RuntimeSettings,
)
from twin.config.settings import Settings
from twin.engine.command_port import CommandContext, CommandOutcome, CommandPort
from twin.engine.dataview import ReplyDataView
from twin.engine.decision import MAX_RETRIES, Decider, Decision, Situation
from twin.engine.fallback import ShortAnswers
from twin.engine.history import HistoryLoader
from twin.engine.pacing import PacingModel
from twin.engine.rounds import RoundStore
from twin.engine.roundstate import Outgoing, RoundData
from twin.engine.safety.crisis import CrisisOutcome
from twin.engine.sender import (
    BubbleSender,
    OutBubble,
    SendReport,
    SentBubble,
    StickerLookup,
    StopReason,
)
from twin.engine.state_store import ConversationSnapshot, ConversationStateStore
from twin.engine.sticker_sender import StickerSender
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.engine.types import InboundItem, ReplyContext, ReplyDraft, SendLimits
from twin.llm.redaction import redact_text
from twin.memory.recent import BotMessage, BotTurnReader
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger
from twin.schedule.events import Resumed
from twin.schedule.time_service import TimeService
from twin.storage.ids import new_id

log = get_logger("twin.engine.machine")

MERGE_SHARE = 0.8  # past this share of the wait, a message that joins adds a short pause
STEP_FAILURE_PAUSE_S = 5.0
STEP_FAILURES_BEFORE_RESET = 5
COMMAND_FAILED_REPLY = "⚙️ 这条指令没有处理成功，请稍后再试。"
THINKING_PREFIX = "⚙️ 思考："
THINKING_MAX_CHARS = 500
SLASH_PREFIXES = ("/", "／")
SCREENED_KINDS = ("text", "voice")
CRISIS_CONTEXT_TURNS = 4
STATE_IDLE, STATE_COLLECTING, STATE_DECIDING = "IDLE", "COLLECTING", "DECIDING"
STATE_GENERATING, STATE_SENDING = "GENERATING", "SENDING"


class DraftWriter(Protocol):
    """Writes the reply to one round: :class:`~twin.engine.pipeline.ReplyPipeline`."""

    async def run(self, context: ReplyContext, data: ReplyDataView) -> ReplyDraft: ...


class DataSource(Protocol):
    """Makes the data view of a moment: :class:`~twin.engine.dataview.LiveDataSource`."""

    def view(self, at: datetime | None = None) -> ReplyDataView: ...


class CrisisPort(Protocol):
    """The crisis check: :class:`~twin.engine.safety.crisis.CrisisHandler`."""

    def screen(self, texts: Sequence[str]) -> int: ...

    async def handle(
        self, texts: Sequence[str], context: Sequence[str] = ()
    ) -> CrisisOutcome | None: ...


class MessageRenderer(Protocol):
    """Turns a message of the channel into the stable line of the conversation."""

    async def render(self, message: InboundMessage) -> InboundItem: ...


PacingSource = Callable[[ReplyDataView], PacingModel]
ExtractionQueue = Callable[[Sequence[BotMessage]], str | None]


@dataclass(frozen=True)
class QuietWindow:
    """The quiet window in effect and the one the profile suggests (``/状态``, R-ENG-002)."""

    configured_s: float
    suggested_s: float
    adaptive: bool

    @property
    def effective_s(self) -> float:
        return self.suggested_s if self.adaptive else self.configured_s


@dataclass(frozen=True)
class Change:
    """A write to ``conversation_state``: the new state (or ``None``) and the fields to set."""

    state: str | None
    fields: dict[str, Any]


class ConversationEngine:
    """The state machine of the conversation (see the module description)."""

    def __init__(
        self,
        *,
        channel: Channel,
        store: BotTurnStore,
        rounds: RoundStore,
        state: ConversationStateStore,
        history: HistoryLoader,
        reader: BotTurnReader,
        writer: DraftWriter,
        data: DataSource,
        time: TimeService,
        renderer: MessageRenderer,
        crisis: CrisisPort,
        stickers: StickerSender,
        lookup: StickerLookup,
        short_answers: Callable[[], ShortAnswers],
        runtime: RuntimeSettings,
        alerts: AlertSink,
        clock: Clock,
        settings: Settings,
        rng: random.Random,
        commands: CommandPort | None = None,
        pacing: PacingSource = PacingModel.from_view,
        queue_extraction: ExtractionQueue | None = None,
    ) -> None:
        self._channel = channel
        self._store = store
        self._rounds = rounds
        self._state = state
        self._history = history
        self._reader = reader
        self._writer = writer
        self._data = data
        self._time = time
        self._renderer = renderer
        self._crisis = crisis
        self._short_answers = short_answers
        self._runtime = runtime
        self._alerts = alerts
        self._clock = clock
        self._settings = settings
        self._rng = rng
        self._commands = commands
        self._pacing_source = pacing
        self._queue_extraction = queue_extraction
        low, high = settings.schedule.greeting_window_min
        self._decider = Decider(time, rng, wake_window_min=(float(low), float(high)))
        self._sender = BubbleSender(channel, stickers, lookup, clock, alerts, rng)
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._progress = asyncio.Event()
        self._activity = asyncio.Event()
        self._arrival_seq = 0
        self._inflight = 0
        self._resume_pending = False
        self._pacing: PacingModel | None = None
        self._reasoning: str | None = None
        self._failures = 0
        self._waiting_until: datetime | None = None
        self._driver: asyncio.Task[None] | None = None
        self._extractor: asyncio.Task[None] | None = None

    # ====================================================================== lifecycle

    async def start(self) -> None:
        """Make the stored state fit the moment and start the driver (see "Recovery")."""
        if self._driver is not None:
            return
        await self._recover(await self._load())
        self._driver = asyncio.create_task(self._drive(), name="engine-driver")
        self._extractor = asyncio.create_task(self._extraction_loop(), name="engine-extraction")

    def attach_commands(self, port: CommandPort) -> None:
        """Hand the engine the command router (the application wires it after building)."""
        self._commands = port

    async def stop(self) -> None:
        """Stop the driver; whatever state is stored stays and is resumed by the next start."""
        tasks = [t for t in (self._driver, self._extractor) if t is not None]
        self._driver = self._extractor = None
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def drain(self) -> None:
        """Wait until no message is waiting and no reply is under way (tests, ``chat --local``)."""
        while True:
            snap = await self._load()
            if self._inflight == 0 and snap.state == STATE_IDLE and not snap.pending:
                return
            self._progress.clear()
            await self._progress.wait()

    def snapshot(self) -> ConversationSnapshot:
        """The stored state (a blocking read)."""
        return self._state.load()

    @property
    def pending_wake(self) -> bool:
        """Something changed (message, setting, wake-up) that the driver has yet to look at."""
        return self._wake.is_set()

    @property
    def waiting_until(self) -> datetime | None:
        """The moment the driver is waiting for (the quiet window, her delay, a pause between
        bubbles), or ``None`` when it waits for a message or is busy."""
        return self._waiting_until

    async def on_state_change(self, _old: int, _new: int) -> None:
        """A setting changed (the state watcher): pause, backend and profile are read again."""
        self._pacing = None
        self._wake.set()

    async def on_resumed(self, event: Resumed) -> None:
        """The application started or the machine woke up: fit the state to the moment."""
        log.info("engine_resumed", kind=event.kind, gap_s=round(event.gap_s))
        self._pacing = None
        self._resume_pending = True
        self._wake.set()
        self._activity.set()

    async def quiet_window(self) -> QuietWindow:
        """The configured quiet window and the one the profile suggests (R-ENG-002)."""
        engine = self._settings.engine
        pacing = await self._pacing_model()
        suggested = pacing.suggested_quiet_window(
            float(engine.quiet_window_s), float(engine.quiet_window_max_s)
        )
        return QuietWindow(float(engine.quiet_window_s), suggested, engine.quiet_window_adaptive)

    # ============================================================== the user's messages

    async def handle_message(self, message: InboundMessage) -> None:
        """A message of the user arrived (the channel's ``incoming()`` hands them over one by one).

        The message is stored first (sealed, once per channel id), then either answered as a
        command at once or queued for the round.  Returns quickly: the round runs in the driver.
        """
        self._inflight += 1
        try:
            await self._ingest(message)
        finally:
            self._inflight -= 1
            self._progress.set()

    async def _ingest(self, message: InboundMessage) -> None:
        item = await self._renderer.render(message)
        slash = (
            self._commands is not None
            and message.kind is MessageKind.TEXT
            and item.text.lstrip()[:1] in SLASH_PREFIXES
        )
        added = await asyncio.to_thread(
            self._store.add_inbound,
            at=message.at,
            kind=item.kind,
            text=item.text,
            external_id=message.id,
            media=dict(item.media) if item.media is not None else None,
            is_command=slash,
        )
        record = added.record
        if not added.created and not await self._lost(record.id, record.is_command):
            return  # delivered again after a restart: it is queued or answered already
        if slash and self._commands is not None:
            try:
                outcome = await self._commands.handle(
                    item.text, CommandContext(at=message.at, inbound_id=record.id)
                )
            except Exception as exc:  # a command that breaks is answered, never read as chat
                log.warning("command_failed", error=type(exc).__name__)
                outcome = CommandOutcome(COMMAND_FAILED_REPLY)
            if outcome is not None:
                await self._answer_command(outcome)
                return
            await asyncio.to_thread(self._rounds.set_command, record.id, False)
        await self._register((record.id,))

    async def _lost(self, turn_id: str, is_command: bool) -> bool:
        """Is a message that was stored before neither queued nor answered (a crash in between)?"""
        if is_command:
            return False
        snap = await self._load()
        known = {*snap.pending, *RoundData.of(snap).answering}
        if turn_id in known:
            return False
        return not await asyncio.to_thread(self._rounds.answered, turn_id)

    async def _register(self, turn_ids: Sequence[str], *, redo: bool = False) -> None:
        """Queue messages for the round and nudge the driver."""
        now = self._clock.now_utc()

        def build(snap: ConversationSnapshot) -> Change:
            data = RoundData.of(snap)
            fresh = not snap.pending
            pending = (*snap.pending, *(t for t in turn_ids if t not in snap.pending))
            data = replace(
                data,
                quiet_from=now,
                collect_from=now if fresh else (data.collect_from or now),
                skip_quiet=redo if fresh else (data.skip_quiet and redo),
            )
            return Change(
                None, {"pending": pending, "data": data.to_json(), "last_inbound_at": now}
            )

        await self._apply(build)
        self._arrival_seq += 1
        self._wake.set()
        self._activity.set()

    async def _answer_command(self, outcome: CommandOutcome) -> None:
        """A command is answered the moment it arrives, in the system voice (R-CMD-001)."""
        await self._say_system(outcome.reply)
        if outcome.redo:
            await self._redo()

    async def _say_system(self, text: str) -> None:
        """One system message, at once, stored as a command row (never memory, never a prompt)."""
        try:
            result = await self._channel.send_text(text)
        except ChannelError as exc:
            log.warning("system_message_not_sent", reason=type(exc).__name__)
            return
        if not result.ok:
            log.warning("system_message_refused", kind=result.kind.value, reason=result.reason)
            if result.kind in (OutboundKind.WINDOW_REJECTED, OutboundKind.AUTH_EXPIRED):
                return
        at = self._clock.now_utc()
        await asyncio.to_thread(
            self._store.add_bubble,
            OutboundBubble(text, at, external_id=result.message_id),
            meta=ReplyMeta("command"),
            is_command=True,
        )

    async def _redo(self) -> None:
        """``/重来``: write the reply to the previous round again (the old one is thrown away)."""
        latest = await asyncio.to_thread(self._store.latest_reply, include_rejected=True)
        if not latest or latest[0].reply_id is None:
            log.info("redo_without_a_reply")
            return
        ids = await asyncio.to_thread(self._rounds.inbound_of_reply, latest[0].reply_id)
        snap = await self._load()
        if not ids or snap.state != STATE_IDLE or snap.pending:
            log.info("redo_skipped", messages=len(ids), state=snap.state)
            return
        await self._register(ids, redo=True)

    # ================================================================ the driver

    async def _drive(self) -> None:
        """The loop of the state machine: one step per state, forever (R-ARCH-004)."""
        while True:
            try:
                if self._resume_pending:
                    self._resume_pending = False
                    await self._recover(await self._load())
                self._wake.clear()
                snap = await self._load()
                await self._step(snap)
                self._failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # one failing step must never end the conversation
                await self._step_failed(exc)
            finally:
                self._progress.set()

    async def _step(self, snap: ConversationSnapshot) -> None:
        if snap.state == STATE_IDLE:
            await self._idle(snap)
        elif snap.state == STATE_COLLECTING:
            await self._collecting(snap)
        elif snap.state == STATE_DECIDING:
            await self._deciding(snap)
        elif snap.state == STATE_GENERATING:
            await self._generating(snap)
        else:
            await self._sending(snap)

    async def _step_failed(self, exc: Exception) -> None:
        self._failures += 1
        log.exception("engine_step_failed", error=type(exc).__name__, failures=self._failures)
        self._alerts.raise_alert(
            "engine_error",
            "A step of the reply engine failed; it tries again",
            severity="warning",
            detail={"error": type(exc).__name__},
            dedup_key="engine_error",
        )
        if self._failures >= STEP_FAILURES_BEFORE_RESET:
            self._failures = 0
            self._alerts.raise_alert(
                "engine_error",
                "The reply engine gave up on a round after repeated failures",
                severity="critical",
                detail={"error": type(exc).__name__},
                dedup_key="engine_reset",
            )
            await asyncio.to_thread(self._state.reset)
            return
        await self._wait_wake(STEP_FAILURE_PAUSE_S)

    # ---------------------------------------------------------------------- recovery

    async def _recover(self, snap: ConversationSnapshot) -> None:
        """Make a stored state fit the moment (after a start, a restart or a wake-up)."""
        now = self._clock.now_utc()
        if snap.state == STATE_COLLECTING:
            await self._edit(lambda d: replace(d, quiet_from=now, collect_from=now))
        elif snap.state == STATE_DECIDING:
            planned = snap.planned_send_at
            if planned is None or planned <= now:  # a new draw, counted from this moment
                await self._edit(
                    lambda d: replace(d, decision=None, quiet_from=now), planned_send_at=None
                )
        elif snap.state == STATE_GENERATING:
            await self._edit(
                lambda d: replace(d, decision=None, answering=(), quiet_from=now),
                state=STATE_DECIDING,
                planned_send_at=None,
            )
        log.info("engine_recovered", state=snap.state, waiting=len(snap.pending))

    # --------------------------------------------------------------------------- IDLE

    async def _idle(self, snap: ConversationSnapshot) -> None:
        if snap.pending:
            await self._edit(state=STATE_COLLECTING, round_id=new_id())
            return
        await self._wait_wake(None)

    # --------------------------------------------------------------------- COLLECTING

    async def _collecting(self, snap: ConversationSnapshot) -> None:
        data = RoundData.of(snap)
        now = self._clock.now_utc()
        quiet_from = data.quiet_from or snap.last_inbound_at or now
        collect_from = data.collect_from or quiet_from
        window_s = await self._quiet_window_s()
        deadline = min(
            quiet_from + timedelta(seconds=window_s),
            collect_from + timedelta(seconds=float(self._settings.engine.max_wait_s)),
        )
        if data.skip_quiet:
            deadline = now
        remaining = (deadline - now).total_seconds()
        if remaining > 0:
            await self._wait_wake(remaining)
            return
        if await self._screen(snap, data):
            return
        await self._decide(snap, continuation=data.skip_quiet)

    async def _quiet_window_s(self) -> float:
        engine = self._settings.engine
        if not engine.quiet_window_adaptive:
            return float(engine.quiet_window_s)
        pacing = await self._pacing_model()
        return pacing.suggested_quiet_window(
            float(engine.quiet_window_s), float(engine.quiet_window_max_s)
        )

    # ---------------------------------------------------------------------- DECIDING

    async def _decide(self, snap: ConversationSnapshot, *, continuation: bool = False) -> None:
        """Draw when she answers and move to (or stay in) DECIDING with that time stored."""
        now = self._clock.now_utc()
        items = await asyncio.to_thread(self._rounds.inbound_items, snap.pending)
        data = RoundData.of(snap)
        self._pacing = None  # her profile may have been rebuilt since the last round
        pause = await self._active_pause()
        anchor = min(data.quiet_from or snap.last_inbound_at or now, now)
        view = self._data.view(now)
        local = await asyncio.to_thread(lambda: view.local)
        pacing = await self._pacing_model(view)
        previous = data.decision
        situation = Situation(
            now=now,
            anchor=anchor,
            chars=sum(len(item.text) for item in items),
            hour=local.local.hour,
            slot=local.slot,
            day_type=local.day_type,
            paused_until=pause,
            continuation=continuation
            or data.continuation
            or (previous is not None and previous.mode == "paused" and pause is None),
        )
        decision = await asyncio.to_thread(self._decider.decide, pacing, situation)
        log.info(
            "reply_decided",
            mode=decision.mode,
            delay_s=round(decision.delay_s),
            her_state=decision.state,
            messages=len(snap.pending),
        )

        def build(current: ConversationSnapshot) -> Change:
            fresh = RoundData.of(current)
            return Change(
                STATE_DECIDING,
                {
                    "data": replace(
                        fresh, decision=decision, decided_for=len(current.pending), skip_quiet=False
                    ).to_json(),
                    "planned_send_at": decision.send_at,
                },
            )

        await self._apply(build)

    async def _deciding(self, snap: ConversationSnapshot) -> None:
        data = RoundData.of(snap)
        if await self._screen(snap, data):
            return
        decision = data.decision
        planned = snap.planned_send_at
        if decision is None or planned is None:
            await self._decide(snap)
            return
        now = self._clock.now_utc()
        pause = await self._active_pause()
        if decision.paused_until != pause:
            await self._decide(snap)  # the pause was set, changed or lifted
            return
        if len(snap.pending) > data.decided_for:
            await self._merge(snap, data, decision, planned)
            return
        if now >= planned:
            await self._start_generating(snap)
            return
        await self._wait_wake((planned - now).total_seconds())

    async def _merge(
        self,
        snap: ConversationSnapshot,
        data: RoundData,
        decision: Decision,
        planned: datetime,
    ) -> None:
        """A message joined while she waits: past 80 % of the wait a short pause is added."""
        arrived = data.quiet_from or snap.last_inbound_at or self._clock.now_utc()
        elapsed = (arrived - decision.anchor_at).total_seconds()
        extended = planned
        if decision.delay_s > 0 and elapsed >= MERGE_SHARE * decision.delay_s:
            new_ids = snap.pending[data.decided_for :]
            items = await asyncio.to_thread(self._rounds.inbound_items, new_ids)
            pacing = await self._pacing_model()
            pause = pacing.short_pause(self._rng, sum(len(item.text) for item in items))
            extended = max(planned, arrived + timedelta(seconds=pause))
        log.info(
            "reply_merged",
            joined=len(snap.pending) - data.decided_for,
            extended_s=round((extended - planned).total_seconds()),
        )

        def build(current: ConversationSnapshot) -> Change:
            fresh = RoundData.of(current)
            return Change(
                None,
                {
                    "data": replace(
                        fresh,
                        decision=replace(decision, send_at=extended),
                        decided_for=len(current.pending),
                    ).to_json(),
                    "planned_send_at": extended,
                },
            )

        await self._apply(build)

    async def _start_generating(self, snap: ConversationSnapshot) -> None:
        pause = await self._active_pause()
        if pause is not None:  # paused after the time was drawn
            await self._edit(lambda d: replace(d, decision=None), planned_send_at=None)
            return

        def build(current: ConversationSnapshot) -> Change:
            fresh = RoundData.of(current)
            return Change(
                STATE_GENERATING,
                {
                    "data": replace(fresh, answering=current.pending).to_json(),
                    "planned_send_at": None,
                },
            )

        await self._apply(build)

    # -------------------------------------------------------------------- GENERATING

    async def _generating(self, snap: ConversationSnapshot) -> None:
        data = RoundData.of(snap)
        items = await asyncio.to_thread(self._rounds.inbound_items, snap.pending)
        if not items:  # nothing the bot may answer (only commands were queued)
            await self._complete(snap.pending)
            return
        if await self._screen(snap, data):
            return
        started = self._clock.monotonic()
        context, view = await self._context(items, snap, data)
        task: asyncio.Task[ReplyDraft] = asyncio.create_task(
            self._writer.run(context, view), name="engine-generation"
        )
        interrupted: str | None = None
        try:
            while not task.done():
                self._wake.clear()
                current = await self._load()  # what arrived before the nudge was cleared
                if self._resume_pending:
                    interrupted = "resume"
                    break
                if any(p not in data.answering for p in current.pending):
                    interrupted = "arrival"
                    break
                await self._race(task)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if interrupted == "resume":
            return
        if interrupted == "arrival":
            await self._cancelled()
            return
        if task.cancelled():  # something cancelled the pipeline from inside: a failure, not a stop
            log.warning("generation_cancelled_from_inside")
            await self._failed("backend_error", None)
            return
        try:
            draft = task.result()
        except Exception as exc:  # the pipeline reports its own failures as a draft; this is a bug
            log.warning("generation_raised", error=type(exc).__name__)
            await self._failed("backend_error", None)
            return
        current = await self._load()
        if any(p not in data.answering for p in current.pending):
            await self._cancelled()  # the user wrote just as the reply was ready
            return
        draft_ms = round((self._clock.monotonic() - started) * 1000)
        await self._after_draft(draft, items, data, draft_ms)

    async def _race(self, task: asyncio.Task[Any]) -> None:
        """Wait until ``task`` is done or the driver is nudged."""
        waker = asyncio.ensure_future(self._wake.wait())
        try:
            await asyncio.wait({task, waker}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not waker.done():
                waker.cancel()
            await asyncio.gather(waker, return_exceptions=True)

    async def _cancelled(self) -> None:
        """The user wrote while the reply was being made: it is thrown away, the round collects."""
        log.info("generation_cancelled")

        def build(current: ConversationSnapshot) -> Change:
            fresh = RoundData.of(current)
            began = fresh.quiet_from or self._clock.now_utc()
            return Change(
                STATE_COLLECTING,
                {
                    "data": replace(
                        fresh,
                        cancelled=fresh.cancelled + 1,
                        decision=None,
                        decided_for=0,
                        answering=(),
                        collect_from=began,
                        skip_quiet=False,
                    ).to_json(),
                    "planned_send_at": None,
                },
            )

        await self._apply(build)

    async def _context(
        self, items: Sequence[InboundItem], snap: ConversationSnapshot, data: RoundData
    ) -> tuple[ReplyContext, ReplyDataView]:
        """What the pipeline needs for this round (see :class:`~twin.engine.types.ReplyContext`)."""
        now = self._clock.now_utc()
        view = self._data.view(now)
        turn_ids = [item.turn_id for item in items if item.turn_id]
        window = await asyncio.to_thread(self._history.load, exclude_ids=turn_ids)
        thinking = await asyncio.to_thread(self._runtime.get, THINKING_CHAT)
        backend = await asyncio.to_thread(self._runtime.get, BACKEND_ACTIVE)
        recent = await asyncio.to_thread(
            self._store.recent_stickers, self._settings.stickers.no_repeat_window
        )
        session = await asyncio.to_thread(self._channel.session_state)
        reserve = self._settings.channel.proactive_reserve
        limits = SendLimits(
            supports_quote=self._channel.capabilities().supports_quote,
            max_bubbles=max(1, session.remaining_quota - reserve),
        )
        context = ReplyContext(
            inbound=tuple(items),
            history=tuple(window.turns),
            thinking_mode=thinking,
            backend=backend,
            limits=limits,
            woke_up=bool(data.decision and data.decision.woke_up),
            already_said=tuple(str(sent.get("text", "")) for sent in snap.sent),
            recent_stickers=tuple(recent),
        )
        return context, view

    def _meta(self, draft: ReplyDraft, data: RoundData, draft_ms: int) -> ReplyMeta:
        """The numbers stored with the reply: the pipeline's plus how long she made it wait."""
        timings = dict(draft.timings_ms)
        if data.decision is not None:
            timings["delay"] = round(data.decision.delay_s * 1000)
        timings["round"] = draft_ms
        base = draft.to_meta()
        return ReplyMeta(
            backend=base.backend,
            thinking=base.thinking,
            plan=base.plan,
            cost_usd=base.cost_usd,
            timings_ms=timings,
            actions=base.actions,
        )

    def _extras(self, data: RoundData) -> tuple[dict[str, Any], ...]:
        extras: list[dict[str, Any]] = []
        if data.cancelled:
            extras.append({"step": "generation_cancelled", "count": data.cancelled})
        if data.retries:
            extras.append({"step": "retried_later", "count": data.retries})
        if data.continuation:
            extras.append({"step": "continued_after_interruption", "count": 1})
        if data.decision is not None:
            extras.append({"step": f"decided_{data.decision.mode}", "count": 1})
        return tuple(extras)

    async def _after_draft(
        self,
        draft: ReplyDraft,
        items: Sequence[InboundItem],
        data: RoundData,
        draft_ms: int,
    ) -> None:
        if draft.needs_fallback:
            await self._failed(draft.fallback_reason or "empty", draft)
            return
        meta = self._meta(draft, data, draft_ms)
        extras = self._extras(data)
        if draft.no_reply:
            silent = ReplyMeta(
                meta.backend,
                meta.thinking,
                meta.plan,
                meta.cost_usd,
                meta.timings_ms,
                (*meta.actions, *extras),
            )
            await asyncio.to_thread(self._store.add_no_reply, self._clock.now_utc(), silent)
            log.info("reply_silent")
            await self._complete(data.answering)
            return
        bubbles = tuple(
            OutBubble("sticker" if bubble.is_sticker else "text", bubble.text, bubble.sticker_md5)
            for bubble in draft.bubbles
        )
        if not bubbles:
            await self._failed("empty", draft)
            return
        quote = self._quote_target(draft.quote, items)
        outgoing = Outgoing(
            meta,
            bubbles,
            quote_id=quote.message_id if quote else None,
            quote_text=quote.text if quote else None,
            extra_actions=extras,
        )
        self._reasoning = draft.reasoning
        log.info("reply_ready", bubbles=len(bubbles), backend=meta.backend, cost_usd=meta.cost_usd)
        await self._edit(
            lambda d: replace(d, outgoing=outgoing), state=STATE_SENDING, planned_send_at=None
        )

    @staticmethod
    def _quote_target(fragment: str | None, items: Sequence[InboundItem]) -> QuoteTarget | None:
        """The message the quote fragment comes from (the newest one that contains it)."""
        if not fragment:
            return None
        for item in reversed(items):
            if fragment in item.text:
                return QuoteTarget(item.id, fragment)
        return QuoteTarget(items[-1].id, fragment)

    # --------------------------------------------------------------------- failure

    async def _failed(self, reason: str, draft: ReplyDraft | None) -> None:
        """No reply could be made: retry in 2 to 10 minutes (three times), then a short answer."""
        snap = await self._load()
        data = RoundData.of(snap)
        if data.retries < MAX_RETRIES:
            decision = self._decider.retry(self._clock.now_utc())
            log.warning(
                "reply_failed_will_retry",
                reason=reason,
                attempt=data.retries + 1,
                in_s=round(decision.delay_s),
            )

            def build(current: ConversationSnapshot) -> Change:
                fresh = RoundData.of(current)
                return Change(
                    STATE_DECIDING,
                    {
                        "data": replace(
                            fresh,
                            retries=fresh.retries + 1,
                            decision=decision,
                            decided_for=len(current.pending),
                            answering=(),
                        ).to_json(),
                        "planned_send_at": decision.send_at,
                    },
                )

            await self._apply(build)
            return
        await self._short_answer(reason, draft, data)

    async def _short_answer(self, reason: str, draft: ReplyDraft | None, data: RoundData) -> None:
        """The last resort: one of her short answers (never an error text), and an alert."""
        answers = await asyncio.to_thread(self._short_answers)
        phrase = answers.pick(self._rng)
        self._alerts.raise_alert(
            "reply_failed",
            "A reply could not be made after repeated tries; a short answer was sent instead"
            if phrase
            else "A reply could not be made after repeated tries and no short answer is known",
            severity="warning",
            detail={"reason": reason, "retries": data.retries, "answered": phrase is not None},
        )
        if phrase is None:
            await self._complete(data.answering or await self._pending_ids())
            return
        meta = ReplyMeta(
            "fallback",
            cost_usd=draft.cost_usd if draft else None,
            timings_ms=dict(draft.timings_ms) if draft else None,
            actions=tuple(action.to_json() for action in draft.actions) if draft else (),
        )
        outgoing = Outgoing(
            meta,
            (OutBubble("text", phrase),),
            extra_actions=({"step": "fallback_answer", "detail": reason},),
        )
        self._reasoning = None

        def build(current: ConversationSnapshot) -> Change:
            fresh = RoundData.of(current)
            return Change(
                STATE_SENDING,
                {
                    "data": replace(fresh, outgoing=outgoing, answering=current.pending).to_json(),
                    "planned_send_at": None,
                },
            )

        await self._apply(build)

    async def _pending_ids(self) -> tuple[str, ...]:
        return (await self._load()).pending

    # ------------------------------------------------------------------------ crisis

    async def _screen(self, snap: ConversationSnapshot, data: RoundData) -> bool:
        """Screen the messages not yet screened; ``True`` if a crisis was answered (round over)."""
        fresh_ids = [p for p in snap.pending if p not in data.screened]
        if not fresh_ids:
            return False
        items = await asyncio.to_thread(self._rounds.inbound_items, fresh_ids)
        texts = [item.text for item in items if item.kind in SCREENED_KINDS]
        outcome: CrisisOutcome | None = None
        if texts and self._crisis.screen(texts) > 0:
            context = await self._crisis_context(snap)
            outcome = await self._crisis.handle(texts, context)
        marked = (*data.screened, *fresh_ids)
        await self._edit(lambda d: replace(d, screened=marked))
        if outcome is None:
            return False
        await self._answer_crisis(outcome, snap)
        return True

    async def _crisis_context(self, snap: ConversationSnapshot) -> list[str]:
        """The last lines before the user's messages: what the model reads the hit against."""
        turn_ids = list(snap.pending)
        window = await asyncio.to_thread(self._history.load, exclude_ids=turn_ids)
        return [turn.text for turn in window.turns[-CRISIS_CONTEXT_TURNS:]]

    async def _answer_crisis(self, outcome: CrisisOutcome, snap: ConversationSnapshot) -> None:
        """Step out of the role at once: the fixed answer, no pause, no persona (R-SAFE-001)."""
        log.warning("crisis_answered", severity=outcome.assessment.severity)
        meta = ReplyMeta("safety", actions=({"step": "crisis_answer", "count": 1},))
        bubbles = tuple(OutBubble("text", text) for text in outcome.bubbles)
        recorder = _Recorder(self._store, meta)

        async def keep(sent: SentBubble) -> None:
            await recorder.record(sent)
            await self._edit(last_outbound_at=sent.at)

        await self._sender.send(
            bubbles,
            pacing=PacingModel(),
            wait=_never_interrupted,
            on_sent=keep,
            paced=False,
        )
        await self._complete(snap.pending)

    # ----------------------------------------------------------------------- SENDING

    async def _sending(self, _stale: ConversationSnapshot) -> None:
        seq = self._arrival_seq  # a message after this point is seen by the waits below ...
        snap = await self._load()  # ... and one before it is in the stored state
        data = RoundData.of(snap)
        outgoing = data.outgoing
        if outgoing is None or not outgoing.unsent:
            await self._finish_reply(data)
            return
        if any(p not in data.answering for p in snap.pending):
            await self._interrupted(snap, data, outgoing)
            return
        if outgoing.reply_id is None and await self._active_pause() is not None:
            await self._edit(
                lambda d: replace(d, outgoing=None, decision=None),
                state=STATE_DECIDING,
                planned_send_at=None,
            )
            return
        pacing = await self._pacing_model()
        items = await asyncio.to_thread(self._rounds.inbound_items, data.answering)
        quote = (
            QuoteTarget(outgoing.quote_id, outgoing.quote_text or "")
            if outgoing.quote_id and items
            else None
        )
        recorder = _Recorder(self._store, outgoing.final_meta(), outgoing.reply_id)

        async def wait(seconds: float) -> bool:
            return await self._interruptible(seconds, seq)

        async def keep(sent: SentBubble) -> None:
            await recorder.record(sent)
            await self._after_bubble(sent, recorder.reply_id, data.answering)

        report = await self._sender.send(
            outgoing.unsent,
            pacing=pacing,
            wait=wait,
            on_sent=keep,
            first_of_reply=outgoing.reply_id is None,
            quote=quote if outgoing.reply_id is None else None,
        )
        await self._after_send(report, snap)

    async def _after_bubble(
        self, sent: SentBubble, reply_id: str | None, answering: tuple[str, ...]
    ) -> None:
        """A bubble is out: it leaves the list of what is left to send, the row notes it."""

        def build(current: ConversationSnapshot) -> Change:
            fresh = RoundData.of(current)
            out = fresh.outgoing
            if out is None:
                return Change(None, {})
            rest = list(out.unsent)
            with contextlib.suppress(ValueError):
                rest = rest[rest.index(sent.bubble) + 1 :]
            kept = replace(out, unsent=tuple(rest), reply_id=reply_id or out.reply_id)
            noted = {
                "text": sent.bubble.text,
                "kind": sent.bubble.kind,
                "at": sent.at.isoformat(),
                "id": sent.message_id,
            }
            return Change(
                None,
                {
                    "data": replace(fresh, outgoing=kept).to_json(),
                    "pending": tuple(p for p in current.pending if p not in answering),
                    "sent": (*current.sent, noted),
                    "last_outbound_at": sent.at,
                },
            )

        await self._apply(build)
        self._activity.set()

    async def _after_send(self, report: SendReport, snap: ConversationSnapshot) -> None:
        if report.skipped:
            log.info("bubbles_skipped", count=len(report.skipped))
            await self._edit(
                lambda d: _with_extra(d, {"step": "bubble_skipped", "count": len(report.skipped)})
            )
        if report.stop is None:
            await self._finish_reply(RoundData.of(await self._load()))
        elif report.stop is StopReason.INTERRUPTED:
            if self._resume_pending:
                return  # the machine woke up: the driver fits the state to the moment first
            current = await self._load()
            data = RoundData.of(current)
            if data.outgoing is not None:
                await self._interrupted(current, data, data.outgoing)
        else:  # the session, the login, the count or the channel ended the reply
            stopped = {"step": "send_stopped", "detail": report.stop.value}
            await self._edit(lambda d: _with_extra(d, stopped))
            await self._finish_reply(RoundData.of(await self._load()), aborted=True)

    async def _interrupted(
        self, snap: ConversationSnapshot, data: RoundData, outgoing: Outgoing
    ) -> None:
        """The user wrote while she was sending: the rest is dropped, the round goes on."""
        log.info("sending_interrupted", unsent=len(outgoing.unsent), out=len(snap.sent))
        if outgoing.reply_id is not None:
            noted = replace(
                outgoing,
                unsent=(),
                extra_actions=(
                    *outgoing.extra_actions,
                    {"step": "send_interrupted", "count": len(outgoing.unsent)},
                ),
            )
            await asyncio.to_thread(self._store.update_meta, outgoing.reply_id, noted.final_meta())

        def build(current: ConversationSnapshot) -> Change:
            fresh = RoundData.of(current)
            return Change(
                STATE_DECIDING,
                {
                    "data": replace(
                        fresh,
                        outgoing=None,
                        decision=None,
                        decided_for=0,
                        answering=(),
                        continuation=True,
                        cancelled=0,
                        retries=0,
                    ).to_json(),
                    "planned_send_at": None,
                },
            )

        await self._apply(build)

    async def _finish_reply(self, data: RoundData, *, aborted: bool = False) -> None:
        """The reply is out (or ended): its numbers are completed, the round ends."""
        outgoing = data.outgoing
        if outgoing is not None and outgoing.reply_id is not None:
            await asyncio.to_thread(
                self._store.update_meta, outgoing.reply_id, outgoing.final_meta()
            )
        if not aborted and self._reasoning:
            await self._show_thinking(self._reasoning)
        self._reasoning = None
        await asyncio.to_thread(self._history.load)  # the window of the recent turns moves on
        await self._complete(data.answering)

    async def _show_thinking(self, reasoning: str) -> None:
        """``/显示思考 开``: the thinking as a system message, redacted and cut (R-LLM-002)."""
        if not await asyncio.to_thread(self._runtime.get, SHOW_THINKING):
            return
        shown = redact_text(reasoning).strip()
        if shown:
            await self._say_system(THINKING_PREFIX + shown[:THINKING_MAX_CHARS])

    async def _interruptible(self, seconds: float, seq: int) -> bool:
        """Sleep ``seconds``; ``True`` if the user wrote or the machine woke up meanwhile."""
        end = self._clock.monotonic() + seconds
        while True:
            if self._arrival_seq != seq or self._resume_pending:
                return True
            remaining = end - self._clock.monotonic()
            if remaining <= 0:
                return False
            self._wake.clear()
            if self._arrival_seq != seq or self._resume_pending:
                return True
            await self._wait_wake(remaining)

    # ----------------------------------------------------------------- the round ends

    async def _complete(self, answered: Sequence[str]) -> None:
        """The round is over: what was answered leaves, anything that came meanwhile starts anew."""
        done = set(answered)

        def build(current: ConversationSnapshot) -> Change:
            pending = tuple(p for p in current.pending if p not in done)
            if pending:
                fresh = RoundData.of(current)
                began = fresh.quiet_from or self._clock.now_utc()
                data = RoundData(quiet_from=began, collect_from=began)
                return Change(
                    STATE_COLLECTING,
                    {
                        "pending": pending,
                        "sent": (),
                        "planned_send_at": None,
                        "round_id": new_id(),
                        "data": data.to_json(),
                    },
                )
            return Change(
                STATE_IDLE,
                {
                    "pending": (),
                    "sent": (),
                    "planned_send_at": None,
                    "round_id": None,
                    "data": {},
                },
            )

        await self._apply(build)
        self._activity.set()

    # ============================================================== memory (R-MEM-007)

    async def _extraction_loop(self) -> None:
        """After ``memory.quiet_minutes`` of silence the conversation goes to the extractor."""
        while True:
            self._activity.clear()
            wait, latest, through = await self._extraction_plan()
            if wait is None or latest is None:
                await self._wait_activity(None)
            elif wait > 0:
                await self._wait_activity(wait)
            else:
                await self._extract(latest, through)

    async def _extraction_plan(self) -> tuple[float | None, datetime | None, datetime | None]:
        """Seconds until the conversation has been quiet long enough (``None``: nothing new)."""
        snap = await self._load()
        latest = max(
            (t for t in (snap.last_inbound_at, snap.last_outbound_at) if t is not None),
            default=None,
        )
        through = snap.extracted_through
        if latest is None or (through is not None and latest <= through):
            return None, latest, through
        quiet = timedelta(minutes=self._settings.memory.quiet_minutes)
        remaining = (latest + quiet - self._clock.now_utc()).total_seconds()
        return max(0.0, remaining), latest, through

    async def extract_if_quiet(self) -> bool:
        """Queue the conversation for the extractor now if it has been quiet long enough."""
        wait, latest, through = await self._extraction_plan()
        if wait != 0.0 or latest is None:
            return False
        await self._extract(latest, through)
        return True

    async def _extract(self, latest: datetime, through: datetime | None) -> None:
        """Queue the messages after ``through`` for the extractor and remember how far it got."""
        messages = await asyncio.to_thread(self._reader.messages_since, through)
        turns = [m for m in messages if through is None or m.at > through]
        queued: str | None = None
        if turns and self._queue_extraction is not None:
            queued = await asyncio.to_thread(self._queue_extraction, turns)
        log.info("extraction_queued", messages=len(turns), job=queued)
        newest = max((turn.at for turn in turns), default=latest)
        await self._edit(extracted_through=max(latest, newest))  # nothing is handed over twice

    async def _wait_activity(self, seconds: float | None) -> None:
        waker = asyncio.ensure_future(self._activity.wait())
        tasks: set[asyncio.Future[Any]] = {waker}
        if seconds is not None:
            tasks.add(asyncio.ensure_future(self._clock.sleep(max(0.0, seconds))))
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ================================================================== helpers

    async def _load(self) -> ConversationSnapshot:
        return await asyncio.to_thread(self._state.load)

    def _commit(self, change: Change) -> ConversationSnapshot:
        if change.state is None:
            return self._state.update(**change.fields)
        return self._state.transition(change.state, **change.fields)

    async def _apply(
        self, build: Callable[[ConversationSnapshot], Change | None]
    ) -> ConversationSnapshot:
        """Read the stored state and write what ``build`` makes of it, in one step."""
        async with self._lock:
            snap = await asyncio.to_thread(self._state.load)
            change = build(snap)
            if change is None or (change.state is None and not change.fields):
                return snap
            return await asyncio.to_thread(self._commit, change)

    async def _edit(
        self,
        edit: Callable[[RoundData], RoundData] | None = None,
        *,
        state: str | None = None,
        **fields: Any,
    ) -> ConversationSnapshot:
        """Change the typed notes (and fields) of the stored state, keeping concurrent writes."""

        def build(snap: ConversationSnapshot) -> Change:
            data = RoundData.of(snap)
            if edit is not None:
                data = edit(data)
            return Change(state, {**fields, "data": data.to_json()})

        return await self._apply(build)

    async def _wait_wake(self, seconds: float | None) -> None:
        """Return after ``seconds`` (``None``: until nudged) or as soon as the driver is nudged."""
        if self._wake.is_set():
            return
        waker = asyncio.ensure_future(self._wake.wait())
        tasks: set[asyncio.Future[Any]] = {waker}
        if seconds is not None:
            self._waiting_until = self._clock.now_utc() + timedelta(seconds=max(0.0, seconds))
            tasks.add(asyncio.ensure_future(self._clock.sleep(max(0.0, seconds))))
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self._waiting_until = None
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _active_pause(self) -> datetime | None:
        """The end of the pause (``/暂停``) if one is in force now."""
        until = await asyncio.to_thread(self._runtime.get, ENGINE_PAUSED_UNTIL)
        if until is not None and until > self._clock.now_utc():
            return until
        return None

    async def _pacing_model(self, view: ReplyDataView | None = None) -> PacingModel:
        """Her pacing, read from the profile once and again after a setting changed."""
        if self._pacing is None:
            source = view if view is not None else self._data.view(self._clock.now_utc())
            self._pacing = await asyncio.to_thread(self._pacing_source, source)
            if self._pacing.reference:
                log.info("pacing_reference_numbers")
        return self._pacing


def _with_extra(data: RoundData, action: dict[str, Any]) -> RoundData:
    """The notes with one more engine action on the reply that is being sent."""
    out = data.outgoing
    if out is None:
        return data
    return replace(data, outgoing=replace(out, extra_actions=(*out.extra_actions, action)))


async def _never_interrupted(_seconds: float) -> bool:
    return False


class _Recorder:
    """Writes the bubbles of one reply to ``bot_turns``; the first opens it with the numbers."""

    def __init__(self, store: BotTurnStore, meta: ReplyMeta, reply_id: str | None = None) -> None:
        self._store = store
        self._meta = meta
        self.reply_id = reply_id

    async def record(self, sent: SentBubble) -> None:
        bubble = OutboundBubble(
            sent.bubble.text,
            sent.at,
            kind=sent.bubble.kind,
            sticker_md5=sent.bubble.sticker_md5,
            external_id=sent.message_id,
        )
        if self.reply_id is None:
            row = await asyncio.to_thread(self._store.add_bubble, bubble, meta=self._meta)
            self.reply_id = row.reply_id
        else:
            await asyncio.to_thread(self._store.add_bubble, bubble, reply_id=self.reply_id)
