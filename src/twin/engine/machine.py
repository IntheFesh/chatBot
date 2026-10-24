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
                sleep - and a confirmed crisis is answered at once, out of the role.  The answer
                is noted in the round before its first bubble and every pass of the driver
                finishes it before anything else: a kill, a stop or a failed write halfway
                through never hands those messages to the persona (D-750).
``DECIDING``    when does she answer?  :class:`~twin.engine.decision.Decider` draws it from her
                state
                (R-ENG-003, R-ENG-004, R-SCOPE-006) and the state waits for it, ``planned_send_at``
                on disk.  A message that comes meanwhile joins the round; after 80 % of the wait a
                short extra pause is added, before that the drawn time stands.  The pause setting
                (``engine.paused_until``) holds the round back; after it she has "just seen" the
                messages.  Her state is looked at again on every pass and before the generation
                starts: a plan that changed under the queued reply (a time-zone switch) and put the
                reply into her deep sleep makes it wait for her waking (D-722).  The reply is begun
                ``generation_lead_s`` before the moment drawn - her measured latency contains her
                typing - and its first bubble is awaited silently until typing it ends at that
                moment (D-723); a message that comes before the moment still joins the round: the
                reply made early is dropped and the decision kept.
``GENERATING``  :class:`~twin.engine.pipeline.ReplyPipeline` writes the reply as an
                ``asyncio`` task.  A message of the user cancels it (what was paid for stays in the
                cost ledger, the cancellation is noted in the reply's actions) and the round goes
                back to COLLECTING with the new message in.  A failed reply is tried again 2 to 10
                minutes later, three times; then a short natural answer of hers goes out and an
                alert is written (R-ENG-010).  Nothing but her words is ever sent.
``SENDING``     :class:`~twin.engine.sender.BubbleSender` sends the bubbles at her pace.  A message
                of the user stops it: what is out stays, the rest is dropped and the round goes on
                at DECIDING with "what she already said" (R-ENG-009).  A first bubble that never
                left the machine (the network) makes the round try again 2 to 10 minutes later,
                like a failed generation (D-751); a probe that begins before the first bubble is
                out sends the round back to wait for it (D-752).

Recovery after a restart (and on ``Resumed``, R-SCH-005): ``COLLECTING`` starts its quiet window
afresh (the messages that piled up in the channel while the process was away join the same round);
``DECIDING`` keeps a planned time that lies ahead (the "after she wakes" queue survives) and draws a
new one when it has passed - never a reply the moment the machine wakes; ``GENERATING`` goes to
``DECIDING`` for a fresh draw; ``SENDING`` goes on with the bubbles that were not out, or, if the
user wrote meanwhile, continues like an interruption - but the rest of a reply that is partly out
is dropped when she is in deep sleep now or its last bubble is more than ``STALE_REST_S`` ago,
and a reply none of whose bubbles is out yet is decided again (queued to her waking) when she is
in deep sleep now (D-753).

A step of the driver that fails is tried again after a few seconds (alert ``engine_error``); after
five failures in a row the round is given up: one short answer of hers goes out first, once per
round (D-720), then the state is reset.

A stop (:meth:`ConversationEngine.stop`) first lets the bubble that is being handed to the channel
finish and be written down - for at most ``stop_grace_s`` real seconds - and starts nothing new;
what is unsent stays in the state for the next start (D-721).

Commands (:class:`~twin.engine.command_port.CommandPort`) are answered the moment they arrive,
without a delay, outside the round, and are marked ``is_command`` (never memory or training data).
``/重来`` (``redo``) writes the reply to the thrown-away round again: from IDLE at once (no quiet
window, no first delay, her pace), and while that reply is still being sent the rest of it is
dropped first.  A command that comes while a *new* round is under way finds no reply to redo (the
router refuses it) and does nothing to the round.

A correction in plain words ("她不会这么说") is no command.  The engine hands the answered
messages to the :class:`~twin.engine.correction_port.CorrectionPort` once the reply to them is out
(round 11) and sends the question it returns as one more system message; a message of the user
that is the "yes" to an open question is taken by the same port, answered at once like a command
and marked ``is_command`` (it is not conversation, memory or learning).  A system message is
written down only when the channel delivered it (or may have, ``AMBIGUOUS``); a question whose
message was not delivered is withdrawn, so the user's next "是" is chat (D-755).

While a channel probe runs (``probe_active``) the engine says nothing (D-684): the user's messages
are stored and screened for a crisis, the round waits in COLLECTING, commands are not run (what
the router would not take for a command - "/::)", "/ 吃饭" - is chat and waits too, D-754), and when
the probe ends the round is decided afresh from that moment with the short "just seen" pause.  The
answer to a crisis is the one exception (it goes out as an urgent send, D-685).

Before the bot is ready - her live profile, routine model and described persona card all exist
(:mod:`twin.engine.readiness`, D-870) - the round is held the same way (D-871): the user's messages
are stored and screened for a crisis (a crisis is answered at once, as always), commands work, and
nothing of hers is said.  One system message says what is missing and what stands in the way, at
most once in six hours for the same missing things (the book is in the database: a restart does not
repeat it).  The driver looks again when a setting changes (a new profile, routine or card moves
``state_version``) and every ``READY_RECHECK_S``; once ready, the messages that piled up are one
round, timed from that moment like the end of a probe ("just seen"), and her plan decides when she
answers (asleep: when she wakes).

Backends: with a :class:`BackendChooser` (``BackendSelector`` of the style model) every reply asks
it which backend to use and tells it how the reply went, which is what makes the fallback to
DeepSeek and the way back work (R-SRV-004); without one the setting ``backend.active`` is used
as it is.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

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
from twin.engine.backend_select import BackendChoice
from twin.engine.command_port import CommandContext, CommandOutcome, CommandPort
from twin.engine.correction_port import CorrectionPort
from twin.engine.dataview import ReplyDataView
from twin.engine.decision import MAX_RETRIES, Decider, Decision, Situation
from twin.engine.fallback import ShortAnswers
from twin.engine.history import HistoryLoader
from twin.engine.kit import EngineKit
from twin.engine.pacing import PacingModel, first_bubble_lead
from twin.engine.rounds import RoundStore
from twin.engine.roundstate import CrisisAnswer, Outgoing, RoundData
from twin.engine.safety.crisis import CrisisOutcome
from twin.engine.sender import (
    BubbleSender,
    OutBubble,
    SendReport,
    SentBubble,
    StickerLookup,
    StopReason,
    Wait,
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

if TYPE_CHECKING:
    from twin.engine.readiness import Readiness

log = get_logger("twin.engine.machine")

MERGE_SHARE = 0.8  # past this share of the wait, a message that joins adds a short pause
STEP_FAILURE_PAUSE_S = 5.0
PROBE_RECHECK_S = 60.0  # while a probe holds the round: look again at least this often
# before the bot is ready: look again at least this often (a new profile, routine or persona card
# moves ``state_version`` and wakes the driver at once; this is the safety net, D-871)
READY_RECHECK_S = 300.0
HOLD_PROBE, HOLD_NOT_READY = "probe", "not_ready"  # why a round is held (``_hold``)
STEP_FAILURES_BEFORE_RESET = 5
STOP_GRACE_S = 10.0  # real seconds a stop waits for a bubble that is being handed over (D-721)
LEAD_MAX_S = 30.0  # the reply is begun at most this long before her delay ends (D-723)
LEAD_SMOOTHING = 0.3  # weight of the newest reply in the running estimate of its lead
# the rest of a reply that a stop cut in two is not sent after this long a silence (D-753): the
# conversation has gone quiet by then (the default ``memory.quiet_minutes`` of R-MEM-007)
STALE_REST_S = 30 * 60.0
SYSTEM_PREFIX = "⚙️ "
COMMAND_FAILED_REPLY = "⚙️ 这条指令没有处理成功，请稍后再试。"
CORRECTION_TIMEOUT_S = 30.0  # the question after a correction in plain words may not hold her up
THINKING_PREFIX = "⚙️ 思考："
THINKING_MAX_CHARS = 500
SLASH_PREFIXES = ("/", "／")
SCREENED_KINDS = ("text", "voice")
CRISIS_CONTEXT_TURNS = 4
STATE_IDLE, STATE_COLLECTING, STATE_DECIDING = "IDLE", "COLLECTING", "DECIDING"
STATE_GENERATING, STATE_SENDING = "GENERATING", "SENDING"
DEFAULT_BACKEND = "deepseek"
FALLBACK_BACKEND = "fallback"  # her short answer of the last resort (never a style sample)


class DraftWriter(Protocol):
    """Writes the reply to one round: :class:`~twin.engine.pipeline.ReplyPipeline`."""

    async def run(self, context: ReplyContext, data: ReplyDataView) -> ReplyDraft: ...


class BackendChooser(Protocol):
    """Chooses the backend of each reply and learns how it went: ``BackendSelector`` (R-SRV-004)."""

    async def choose(self) -> BackendChoice: ...

    def record(self, backend: str, draft: ReplyDraft) -> None: ...


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


class ReadinessPort(Protocol):
    """Is the bot ready, and what to say until it is: :class:`~twin.engine.readiness.ReadinessGate`.

    Every method is blocking (the engine calls it in a worker thread).
    """

    def check(self) -> Readiness: ...

    def notice(
        self, readiness: Readiness, now: datetime, last_inbound: datetime | None
    ) -> str | None: ...

    def noted(self, readiness: Readiness, at: datetime) -> None: ...


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
        readiness: ReadinessPort,
        commands: CommandPort | None = None,
        backends: BackendChooser | None = None,
        pacing: PacingSource = PacingModel.from_view,
        queue_extraction: ExtractionQueue | None = None,
        corrections: CorrectionPort | None = None,
        probe_active: Callable[[], bool] | None = None,
        stop_grace_s: float = STOP_GRACE_S,
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
        self._corrections = corrections
        self._backends = backends
        self._pacing_source = pacing
        self._queue_extraction = queue_extraction
        self._probe_active = probe_active
        self._readiness = readiness
        self._held: set[str] = set()  # why the round is held (probe, not ready): re-timed after
        self._stop_grace_s = stop_grace_s
        self._stopping = False  # ``stop`` was called: nothing new is started, a send is let finish
        # clear while a bubble is being handed to the channel (and while the crisis answer goes out)
        self._safe_to_cancel = asyncio.Event()
        self._safe_to_cancel.set()
        low, high = settings.schedule.greeting_window_min
        self._decider = Decider(time, rng, wake_window_min=(float(low), float(high)))
        self._sender = BubbleSender(channel, stickers, lookup, clock, alerts, rng)
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._progress = asyncio.Event()
        self._activity = asyncio.Event()
        self._arrival_seq = 0
        self._arrival_signal = asyncio.Event()
        self._inflight = 0
        self._resume_pending = False
        self._pacing: PacingModel | None = None
        self._reasoning: str | None = None
        self._thrown_away: str | None = None  # the reply a ``/重来`` stopped while it was sent
        self._open_reply: str | None = (
            None  # the reply whose bubbles are being sent, once it has a row
        )
        self._failures = 0
        self._gave_up_round: str | None = None  # the round that got the word of ``_give_up_round``
        self._rejudged_for: datetime | None = None  # the planned time ``_rejudge`` last moved
        self._lead_s: float | None = None  # running estimate of the lead of a reply (D-723)
        self._waiting_until: datetime | None = None
        self._driver: asyncio.Task[None] | None = None
        self._extractor: asyncio.Task[None] | None = None
        self.kit: EngineKit | None = None
        """The parts the engine was built from, for the proactive scheduler (``build_engine``)."""

    # ====================================================================== lifecycle

    async def start(self) -> None:
        """Make the stored state fit the moment and start the driver (see "Recovery")."""
        if self._driver is not None:
            return
        self._stopping = False
        await self._recover(await self._load())
        self._driver = asyncio.create_task(self._drive(), name="engine-driver")
        self._extractor = asyncio.create_task(self._extraction_loop(), name="engine-extraction")

    def attach_commands(self, port: CommandPort) -> None:
        """Hand the engine the command router (the application wires it after building)."""
        self._commands = port

    @property
    def commands(self) -> CommandPort | None:
        """The command port in use (the router of ``twin.commands`` in the running application)."""
        return self._commands

    def attach_corrections(self, port: CorrectionPort) -> None:
        """Hand the engine the service for corrections in plain words (round 11)."""
        self._corrections = port

    @property
    def corrections(self) -> CorrectionPort | None:
        """The correction service in use (``None``: such corrections are not noticed)."""
        return self._corrections

    async def notify(self, text: str) -> None:
        """Say ``text`` as a system message now (the report of work done in the background)."""
        await self._say_system(text if text.startswith(SYSTEM_PREFIX) else SYSTEM_PREFIX + text)

    async def stop(self) -> None:
        """Stop the driver; whatever state is stored stays and is resumed by the next start.

        A bubble that is being handed to the channel is let finish (and is noted) before the
        driver is cancelled, for at most ``stop_grace_s`` real seconds: it is out on the user's
        phone as soon as the channel has it, and a cancel in that moment would leave it
        "in doubt" for the next start (D-721).  Everything else is cancelled at once; the sender
        starts nothing new once the stop is asked for.
        """
        tasks = [t for t in (self._driver, self._extractor) if t is not None]
        self._driver = self._extractor = None
        self._stopping = True
        self._wake.set()  # a driver that waits looks again at once
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._safe_to_cancel.wait(), self._stop_grace_s)
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
    def arrivals(self) -> int:
        """How many messages of the user (commands excluded) have been queued so far.

        The proactive scheduler (round 10) reads it before it writes to the user and again before
        it sends: a message that arrived in between means the user is talking, not waiting.
        """
        return self._arrival_seq

    async def wait_for_arrival(self, since: int, seconds: float) -> bool:
        """Sleep up to ``seconds``; ``True`` as soon as :attr:`arrivals` is no longer ``since``."""
        end = self._clock.monotonic() + max(0.0, seconds)
        while True:
            if self._arrival_seq != since:
                return True
            remaining = end - self._clock.monotonic()
            if remaining <= 0:
                return False
            self._arrival_signal.clear()
            if self._arrival_seq != since:
                return True
            waker = asyncio.ensure_future(self._arrival_signal.wait())
            sleeper = asyncio.ensure_future(self._clock.sleep(remaining))
            tasks: set[asyncio.Future[Any]] = {waker, sleeper}
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def pending_wake(self) -> bool:
        """Something changed (message, setting, wake-up) that the driver has yet to look at."""
        return self._wake.is_set()

    @property
    def waiting_until(self) -> datetime | None:
        """The moment the driver is waiting for (the quiet window, her delay, a pause between
        bubbles), or ``None`` when it waits for a message or is busy."""
        return self._waiting_until

    @property
    def generation_lead_s(self) -> float:
        """How long before her delay ends a reply is begun (D-723).

        The running estimate (this process only) of how long a reply needs from the start of its
        generation until its first bubble could appear - the generation, the silent second and
        the typing of the bubble (:func:`~twin.engine.pacing.first_bubble_lead`) - at most
        :data:`LEAD_MAX_S`; 0 until the first reply of the process is made.
        """
        return min(self._lead_s or 0.0, LEAD_MAX_S)

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
        confirming = (
            not slash
            and self._corrections is not None
            and message.kind is MessageKind.TEXT
            and await self._corrections.is_confirmation(item.text, message.at)
        )
        added = await asyncio.to_thread(
            self._store.add_inbound,
            at=message.at,
            kind=item.kind,
            text=item.text,
            external_id=message.id,
            media=dict(item.media) if item.media is not None else None,
            is_command=slash or confirming,
        )
        record = added.record
        if not added.created and not await self._lost(record.id, record.is_command):
            return  # delivered again after a restart: it is queued or answered already
        if (slash or confirming) and await self._probe_running():
            if slash and not self._recognised(item.text):  # "/::)", "/ 吃饭": chat (D-754)
                await asyncio.to_thread(self._rounds.set_command, record.id, False)
                await self._register((record.id,))
                return
            log.info("command_held_for_probe")  # stored as a command row; not run, not answered
            return
        if confirming and self._corrections is not None:
            outcome = await self._confirmation(item.text, message.at, record.id)
            if outcome is not None:
                await self._answer_command(outcome)
                return
            await asyncio.to_thread(self._rounds.set_command, record.id, False)
        elif slash and self._commands is not None:
            began = self._clock.now_utc()  # a question the command asks is asked from here on
            try:
                outcome = await self._commands.handle(
                    item.text, CommandContext(at=message.at, inbound_id=record.id)
                )
            except Exception as exc:  # a command that breaks is answered, never read as chat
                log.warning("command_failed", error=type(exc).__name__)
                outcome = CommandOutcome(COMMAND_FAILED_REPLY)
            if outcome is not None:
                await self._answer_command(outcome, asked_since=began)
                return
            await asyncio.to_thread(self._rounds.set_command, record.id, False)
        await self._register((record.id,))

    def _recognised(self, text: str) -> bool:
        """Would the router take ``text`` for a command?  (Its own reading, nothing is run.)"""
        port = self._commands
        return port is not None and port.recognises(text)

    async def _confirmation(self, text: str, at: datetime, turn_id: str) -> CommandOutcome | None:
        """The "yes" to an open question about a correction (``None``: no answer after all)."""
        port = self._corrections
        if port is None:
            return None
        try:
            return await port.confirm(CommandContext(at=at, inbound_id=turn_id))
        except Exception as exc:  # a failing service must never swallow a message of the user
            log.warning("correction_confirm_failed", error=type(exc).__name__)
            return None

    async def _propose_correction(self, answered: Sequence[str]) -> None:
        """After a reply: ask whether the user's remark was a correction to record (round 11)."""
        port = self._corrections
        if port is None or not answered:
            return
        began = self._clock.now_utc()
        try:
            question = await asyncio.wait_for(port.propose(answered), CORRECTION_TIMEOUT_S)
        except Exception as exc:  # the question is optional: the conversation goes on without it
            log.warning("correction_proposal_failed", error=type(exc).__name__)
            return
        if question and not await self._say_system(question):
            await self._withdraw_question(began)

    async def _withdraw_question(self, asked_since: datetime) -> None:
        """A question asked from ``asked_since`` on never reached the user: it is closed, so that
        his next "是" is not taken for its answer (D-755)."""
        port = self._corrections
        if port is None:
            return
        try:
            await port.withdraw(asked_since)
        except Exception as exc:  # at worst the question lapses with its window
            log.warning("correction_withdraw_failed", error=type(exc).__name__)

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
        self._arrival_signal.set()
        self._wake.set()
        self._activity.set()

    async def _answer_command(
        self, outcome: CommandOutcome, *, asked_since: datetime | None = None
    ) -> None:
        """A command is answered the moment it arrives, in the system voice (R-CMD-001).

        ``asked_since``: when the command began; a question it asked (the fact question of
        ``/不像 <说法>``, D-674) is withdrawn if the answer that carries it is not delivered.
        """
        delivered = await self._say_system(outcome.reply)
        if not delivered and asked_since is not None:
            await self._withdraw_question(asked_since)
        if outcome.redo:
            await self._redo()

    async def _say_system(self, text: str) -> bool:
        """One system message, at once, stored as a command row (never memory, never a prompt).

        ``True`` if it was delivered - or may have been (``AMBIGUOUS``: it may be on his phone);
        only then is it written down (D-755): a message the channel refused or could not send
        is not "said".
        """
        try:
            result = await self._channel.send_text(text)
        except ChannelError as exc:
            log.warning("system_message_not_sent", reason=type(exc).__name__)
            return False
        if not result.ok and result.kind is not OutboundKind.AMBIGUOUS:
            log.warning("system_message_refused", kind=result.kind.value, reason=result.reason)
            return False
        at = self._clock.now_utc()
        await asyncio.to_thread(
            self._store.add_bubble,
            OutboundBubble(text, at, external_id=result.message_id),
            meta=ReplyMeta("command"),
            is_command=True,
        )
        return True

    async def _redo(self) -> None:
        """``/重来``: write the reply to the previous round again (the old one is thrown away).

        The router has already marked the old reply ``rejected`` and written the ``redo`` feedback.
        From IDLE the messages it answered are queued again, without the quiet window and the
        first delay (the user has just asked, not written).  While that reply is still being sent
        its unsent bubbles are dropped first.  Any other state is a *new* round in the making (the
        router refuses a ``/重来`` then, but a message may land in between): it is left alone.
        """
        latest = await asyncio.to_thread(self._store.latest_reply, include_rejected=True)
        if not latest or latest[0].reply_id is None:
            log.info("redo_without_a_reply")
            return
        reply_id = latest[0].reply_id
        ids = await asyncio.to_thread(self._rounds.inbound_of_reply, reply_id)
        if not ids:
            log.info("redo_without_messages")
            return
        if not await self._drop_unsent(reply_id):
            snap = await self._load()
            if snap.state != STATE_IDLE or snap.pending:
                log.info("redo_skipped", messages=len(ids), state=snap.state)
                return
        await self._register(ids, redo=True)

    async def _drop_unsent(self, reply_id: str) -> bool:
        """Stop sending ``reply_id`` (the user threw it away); ``True`` if it was being sent.

        The round ends as if the reply had been sent in full - without the thinking that would
        have followed it - and the sender, which may be waiting for its next bubble, is woken so
        that it stops.  A bubble that was already on its way when this ran is marked ``rejected``
        by the sender's loop (:attr:`_thrown_away`).
        """
        snap = await self._load()
        data = RoundData.of(snap)
        outgoing = data.outgoing
        if snap.state != STATE_SENDING or outgoing is None:
            return False
        # the state notes the reply once its first bubble is written down; the row exists a moment
        # earlier, and ``_open_reply`` is the sender's own knowledge of it for that moment
        if (outgoing.reply_id or self._open_reply) != reply_id:
            return False
        if any(p not in data.answering for p in snap.pending):
            return False  # a new message is queued: the interruption rules deal with the rest
        self._thrown_away = reply_id
        self._reasoning = None
        noted = replace(
            outgoing,
            unsent=(),
            extra_actions=(
                *outgoing.extra_actions,
                {"step": "send_stopped_by_redo", "count": len(outgoing.unsent)},
            ),
        )
        await asyncio.to_thread(self._store.update_meta, reply_id, noted.final_meta())
        await self._complete(data.answering)
        self._arrival_seq += 1  # the sender waits on this: it stops before the next bubble
        self._wake.set()
        return True

    # ================================================================ the driver

    async def _drive(self) -> None:
        """The loop of the state machine: one step per state, until stopped (R-ARCH-004)."""
        while not self._stopping:
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
        crisis = RoundData.of(snap).crisis
        if crisis is not None:  # an answer to a crisis is finished first (D-750)
            await self._answer_crisis(crisis)
            return
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
            await self._give_up_round()
            return
        await self._wait_wake(STEP_FAILURE_PAUSE_S)

    async def _give_up_round(self) -> None:
        """Five steps in a row failed: one short word of hers, once, and then the round is dropped.

        The user must not be left without a word (D-550, revised in round 17): the round goes to
        SENDING with one of her short answers (:meth:`_short_answer`, reason ``step_failed``; the
        alert ``reply_failed``, the bubble marked ``fallback`` so that it never becomes a style
        sample) and the normal sender says it.  The attempt is made once per round
        (``_gave_up_round``): when the sender is what keeps failing, the next fifth failure resets
        the state silently, as before.  Before the bot is ready (D-871) no word of hers is said:
        the state is reset silently.  Any error of the attempt itself is the same as no
        attempt: the state is reset and the loop lives on.
        """
        try:
            if await self._answer_step_failure():
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the way out of a failing engine may not fail it again
            log.warning("step_failure_answer_failed", error=type(exc).__name__)
        try:
            await asyncio.to_thread(self._state.reset)
        except Exception as exc:  # ... and neither may the reset end the loop
            log.warning("engine_reset_failed", error=type(exc).__name__)

    async def _answer_step_failure(self) -> bool:
        """Hand the round's messages the short answer; ``False`` if the round gets none."""
        snap = await self._load()
        if not snap.pending or snap.round_id is None or snap.round_id == self._gave_up_round:
            return False  # nothing waits, or the answer was tried for this round and failed
        data = RoundData.of(snap)
        if data.crisis is not None:
            return False  # the round answers a crisis: never a word of the persona (D-750)
        if data.outgoing is not None and data.outgoing.reply_id is not None:
            return False  # a part of her reply is out: no second word on top of it
        if not (await asyncio.to_thread(self._readiness.check)).ready:
            return False  # before the bot is ready nothing of hers is said (D-871)
        self._gave_up_round = snap.round_id
        await self._short_answer("step_failed", None, data)
        return True

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
        if await self._probe_hold(snap) or await self._ready_hold(snap):
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
        if await self._probe_hold(snap) or await self._ready_hold(snap):
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
        if await self._rejudge(snap, decision, planned):
            return
        if len(snap.pending) > data.decided_for:
            await self._merge(snap, data, decision, planned)
            return
        start_at = planned - timedelta(seconds=self.generation_lead_s)
        if decision.paused_until is not None:  # nothing is made while the pause lasts
            start_at = max(start_at, decision.paused_until)
        if now >= start_at:
            await self._start_generating(snap)
            return
        await self._wait_wake((start_at - now).total_seconds())

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

    async def _rejudge(
        self, snap: ConversationSnapshot, decision: Decision, planned: datetime
    ) -> bool:
        """A reply that is due while she sleeps, by today's plan, waits for her waking (D-722).

        The decision was made for the plan of that moment; the plan may have changed since (the
        time zone was switched, ``/作息``, a new plan) and now put the reply into her deep sleep.
        Then it is decided anew - queued to her waking plus the jitter (D-233 ①, R-ENG-003).
        Only a decision that is not such a queue is looked at, and only deep sleep counts: the
        dice of the edge of her sleep were thrown when the time was drawn and are not thrown
        again.  Asked on every pass of DECIDING (the switch wakes the driver) and again before
        the generation starts.  ``True``: the round was decided anew, the driver looks again.
        """
        if decision.woke_up or planned == self._rejudged_for:
            return False  # a queue to her waking already, or this time was moved once already
        moment = max(planned, self._clock.now_utc())
        state = await asyncio.to_thread(self._decider.her_state, moment)
        if state is None or state.kind != "deep_sleep":
            return False
        self._rejudged_for = planned
        log.info("reply_decided_again_asleep", mode=decision.mode, her_state=state.kind)
        await self._decide(snap)
        return True

    async def _start_generating(self, snap: ConversationSnapshot) -> None:
        decision = RoundData.of(snap).decision
        if (
            decision is not None
            and snap.planned_send_at is not None
            and await self._rejudge(snap, decision, snap.planned_send_at)
        ):
            return  # the plan changed since the pass of DECIDING: she sleeps when the reply is due
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
        self._record_backend(context.backend, draft)
        current = await self._load()
        if any(p not in data.answering for p in current.pending):
            await self._cancelled()  # the user wrote just as the reply was ready
            return
        draft_ms = round((self._clock.monotonic() - started) * 1000)
        await self._note_generation(draft, draft_ms / 1000.0)
        await self._after_draft(draft, items, data, draft_ms)

    async def _note_generation(self, draft: ReplyDraft, generation_s: float) -> None:
        """Remember how long a reply needed: the next one is begun that long before it is due.

        Only a reply that was made says anything of it (a failure or silence does not).
        """
        if draft.needs_fallback or not draft.bubbles:
            return
        first = draft.bubbles[0]
        pacing = await self._pacing_model()
        typing_s = pacing.bubble_interval(
            self._rng, 0 if first.is_sticker else len(first.text), first=True
        )
        lead = min(first_bubble_lead(max(generation_s, 0.0), typing_s), LEAD_MAX_S)
        previous = self._lead_s
        self._lead_s = (
            lead if previous is None else (1.0 - LEAD_SMOOTHING) * previous + LEAD_SMOOTHING * lead
        )

    def _in_her_delay(self, data: RoundData) -> bool:
        """Is it still before the moment her delay ends, with nothing of the reply out? (D-723)

        A reply that is made or waited for early is still inside the delay she drew: a message
        that comes then joins the round like any that comes while she waits (R-ENG-004), it does
        not start the round again.
        """
        decision = data.decision
        out = data.outgoing
        return (
            decision is not None
            and decision.send_at > self._clock.now_utc()
            and (out is None or (out.reply_id is None and out.in_flight is None))
        )

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
            if fresh.decision is not None and self._in_her_delay(fresh):
                return self._back_to_her_delay(current, fresh.decision)
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

    @staticmethod
    def _back_to_her_delay(current: ConversationSnapshot, decision: Decision) -> Change:
        """The reply made early is thrown away; the round waits for the time drawn (D-723).

        ``decided_for`` still counts the messages the decision knew, so DECIDING sees the new one
        and merges it (``_merge``: a short pause after 80 % of the wait, otherwise the drawn time
        stands).  The notes are read from ``current``: a message that was queued meanwhile has
        written its clocks there.
        """
        fresh = RoundData.of(current)
        return Change(
            STATE_DECIDING,
            {
                "data": replace(
                    fresh, cancelled=fresh.cancelled + 1, answering=(), outgoing=None
                ).to_json(),
                "planned_send_at": decision.send_at,
            },
        )

    async def _context(
        self, items: Sequence[InboundItem], snap: ConversationSnapshot, data: RoundData
    ) -> tuple[ReplyContext, ReplyDataView]:
        """What the pipeline needs for this round (see :class:`~twin.engine.types.ReplyContext`)."""
        now = self._clock.now_utc()
        view = self._data.view(now)
        turn_ids = [item.turn_id for item in items if item.turn_id]
        window = await asyncio.to_thread(self._history.load, exclude_ids=turn_ids)
        thinking = await asyncio.to_thread(self._runtime.get, THINKING_CHAT)
        backend = await self._backend()
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

    async def _backend(self) -> str:
        """The backend of this reply: the selector's choice, or the user's setting without one.

        A selector that breaks must not keep her from answering: DeepSeek is always there.
        """
        if self._backends is None:
            return str(await asyncio.to_thread(self._runtime.get, BACKEND_ACTIVE))
        try:
            choice = await self._backends.choose()
        except Exception as exc:
            log.warning("backend_choice_failed", error=type(exc).__name__)
            return DEFAULT_BACKEND
        if choice.fell_back:
            log.info("backend_fell_back", backend=choice.name, asked=choice.requested)
        return choice.name

    def _record_backend(self, backend: str, draft: ReplyDraft) -> None:
        """Tell the selector how the reply of ``backend`` went (failures and hard violations)."""
        if self._backends is None:
            return
        try:
            self._backends.record(backend, draft)
        except Exception as exc:  # the books of the selector are not worth a lost reply
            log.warning("backend_record_failed", error=type(exc).__name__)

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
            due_at=data.decision.send_at if data.decision is not None else None,
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
            await self._retry_later(reason, data)
            return
        await self._short_answer(reason, draft, data)

    async def _retry_later(self, reason: str, data: RoundData) -> None:
        """The round is tried again in 2 to 10 minutes, a new draft (R-ENG-010; also D-751).

        A draft that was made and could not be sent is dropped with it.
        """
        decision = self._decider.retry(self._clock.now_utc())
        log.warning(
            "reply_failed_will_retry",
            reason=reason,
            attempt=data.retries + 1,
            in_s=round(decision.delay_s),
        )
        self._reasoning = None

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
                        outgoing=None,
                    ).to_json(),
                    "planned_send_at": decision.send_at,
                },
            )

        await self._apply(build)

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
            FALLBACK_BACKEND,
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
        if outcome is None:
            await self._edit(lambda d: replace(d, screened=marked))
            return False
        log.warning("crisis_answered", severity=outcome.assessment.severity)
        # the answer is noted with the mark, before its first bubble: whatever stops it halfway,
        # the next pass finishes it - the persona never gets these messages (D-750)
        answer = CrisisAnswer(tuple(outcome.bubbles), tuple(snap.pending))
        await self._edit(lambda d: replace(d, screened=marked, crisis=answer))
        await self._answer_crisis(answer)
        return True

    async def _crisis_context(self, snap: ConversationSnapshot) -> list[str]:
        """The last lines before the user's messages: what the model reads the hit against."""
        turn_ids = list(snap.pending)
        window = await asyncio.to_thread(self._history.load, exclude_ids=turn_ids)
        return [turn.text for turn in window.turns[-CRISIS_CONTEXT_TURNS:]]

    async def _answer_crisis(self, answer: CrisisAnswer) -> None:
        """Step out of the role at once: the fixed answer, no pause, no persona (R-SAFE-001).

        ``answer`` is the round's note :class:`~twin.engine.roundstate.CrisisAnswer` as stored
        (written by :meth:`_screen` before the first bubble); each bubble that is out leaves it.
        Writing a bubble down is best-effort here: a write that fails is logged and the answer
        goes on (a
        bubble not written down is sent again after a restart - the hotline twice is better than
        never).  A send that the channel could not make (``FAILED``: the network) is tried again
        2 to 10 minutes later, at most ``MAX_RETRIES`` times; any other end of the send, or the
        last try, ends the round as before.  Never the persona (D-750).
        """
        now = self._clock.now_utc()
        if answer.retry_at is not None and answer.retry_at > now:
            await self._wait_wake((answer.retry_at - now).total_seconds())
            return
        if answer.reply_id is not None or answer.retries:
            log.info("crisis_answer_resumed", unsent=len(answer.unsent), retries=answer.retries)
        meta = ReplyMeta("safety", actions=({"step": "crisis_answer", "count": 1},))
        recorder = _Recorder(self._store, meta, answer.reply_id)

        async def note(sent: SentBubble) -> None:
            try:
                await recorder.record(sent)
            except Exception as exc:  # the answer goes on: the row is not worth the hotline
                log.warning("crisis_bubble_not_written", error=type(exc).__name__)
            try:
                await self._edit(
                    lambda d: _crisis_sent(d, sent.bubble.text, recorder.reply_id),
                    last_outbound_at=sent.at,
                )
            except Exception as exc:  # ... nor is the note: at worst a bubble is said again
                log.warning("crisis_note_not_written", error=type(exc).__name__)

        async def keep(sent: SentBubble) -> None:
            await _written(note(sent))

        # a stop lets the whole answer go out and the round end: nothing else may be cut short
        self._safe_to_cancel.clear()
        try:
            report = await self._sender.send(
                tuple(OutBubble("text", text) for text in answer.unsent),
                pacing=PacingModel(),
                wait=self._crisis_pause,
                on_sent=keep,
                paced=False,
                urgent=True,  # safety wins over a running probe (D-685); no other send sets this
            )
            if report.stop is StopReason.FAILED and answer.retries < MAX_RETRIES:
                retry_at = self._decider.retry(self._clock.now_utc()).send_at
                log.warning("crisis_answer_will_retry", attempt=answer.retries + 1)
                await self._edit(lambda d: _crisis_retry(d, retry_at))
                return
            if report.stop is not None:
                log.warning("crisis_answer_not_sent", reason=report.stop.value)
            await self._complete(answer.answering)
        finally:
            self._safe_to_cancel.set()

    async def _crisis_pause(self, seconds: float) -> bool:
        """The pause before a send of the crisis answer is repeated (:func:`uninterruptible_pause`).

        Nothing cuts it short; :attr:`waiting_until` says what the driver waits for, as for any
        other wait (the pause is a whole number of microseconds, so that the moment said is the
        moment it ends).
        """
        pause = timedelta(seconds=max(0.0, seconds))
        self._waiting_until = self._clock.now_utc() + pause
        try:
            return await uninterruptible_pause(self._clock)(pause.total_seconds())
        finally:
            self._waiting_until = None

    # ----------------------------------------------------------------------- SENDING

    async def _sending(self, _stale: ConversationSnapshot) -> None:
        seq = self._arrival_seq  # a message after this point is seen by the waits below ...
        snap = await self._load()  # ... and one before it is in the stored state
        data = RoundData.of(snap)
        outgoing = data.outgoing
        if outgoing is not None and outgoing.in_flight is not None:
            await self._settle_in_doubt(data, outgoing)  # a send was under way when we stopped
            snap = await self._load()
            data = RoundData.of(snap)
            outgoing = data.outgoing
        if outgoing is None or not outgoing.unsent:
            await self._finish_reply(data)
            return
        if outgoing.reply_id is not None:
            stale = await self._stale_rest(snap)
            if stale is not None:
                await self._drop_stale_rest(outgoing, stale)
                return
        if any(p not in data.answering for p in snap.pending):
            await self._interrupted(snap, data, outgoing)
            return
        if outgoing.reply_id is None and (
            await self._active_pause() is not None
            or await self._probe_running()
            or await self._asleep_before_first(data)
        ):
            await self._draft_back_to_deciding()
            return
        pacing = await self._pacing_model()
        items = await asyncio.to_thread(self._rounds.inbound_items, data.answering)
        quote = (
            QuoteTarget(outgoing.quote_id, outgoing.quote_text or "")
            if outgoing.quote_id and items
            else None
        )
        recorder = _Recorder(self._store, outgoing.final_meta(), outgoing.reply_id)
        self._open_reply = outgoing.reply_id

        async def wait(seconds: float) -> bool:
            return await self._interruptible(seconds, seq)

        async def note(sent: SentBubble) -> None:
            await recorder.record(sent)
            self._open_reply = recorder.reply_id
            await self._after_bubble(sent, recorder.reply_id, data.answering)

        async def keep(sent: SentBubble) -> None:
            try:
                await _written(note(sent))
            finally:
                self._safe_to_cancel.set()  # the bubble is out and written down: a stop may go on

        async def about_to_send(bubble: OutBubble) -> None:
            self._safe_to_cancel.clear()  # from here until ``keep``: a stop lets the send finish
            try:
                await self._edit(lambda d: _with_in_flight(d, bubble, self._clock.now_utc()))
            except BaseException:
                self._safe_to_cancel.set()  # nothing was handed over
                raise

        try:
            report = await self._sender.send(
                outgoing.unsent,
                pacing=pacing,
                wait=wait,
                on_sent=keep,
                on_sending=about_to_send,
                first_of_reply=outgoing.reply_id is None,
                quote=quote if outgoing.reply_id is None else None,
                first_due_at=outgoing.due_at,
            )
        finally:
            self._safe_to_cancel.set()  # a send that ended without a bubble: nothing to wait for
        thrown = self._thrown_away
        if thrown is not None and recorder.reply_id == thrown:
            self._thrown_away = None  # a ``/重来`` ended this reply (see ``_drop_unsent``)
            await asyncio.to_thread(self._store.reject_reply, thrown)  # a bubble that was in flight
            return
        await self._after_send(report, snap)

    async def _draft_back_to_deciding(self) -> None:
        """Nothing of the reply is out and it may not go out now (``/暂停``, a probe): the draft
        is dropped and the round decided again when it may (D-686; D-752)."""
        self._reasoning = None
        await self._edit(
            lambda d: replace(d, outgoing=None, decision=None),
            state=STATE_DECIDING,
            planned_send_at=None,
        )

    async def _asleep_before_first(self, data: RoundData) -> bool:
        """Is she in deep sleep now, with nothing of a reply out that was not queued to her
        waking?  (The re-judgement of D-722 for a reply that is made already, D-753.)

        After a long stop the reply that was ready may meet her deep sleep: it is decided again
        from now (queued to her waking) instead of being sent at night.  As in :meth:`_rejudge`
        (whose memory of the moved time it shares), a reply that is queued to her waking, or
        whose time was moved once already, is not moved again: no loop.
        """
        decision = data.decision
        if decision is not None and (decision.woke_up or decision.send_at == self._rejudged_for):
            return False
        state = await asyncio.to_thread(self._decider.her_state, self._clock.now_utc())
        if state is None or state.kind != "deep_sleep":
            return False
        if decision is not None:
            self._rejudged_for = decision.send_at
        log.info("reply_decided_again_asleep", her_state=state.kind, made=True)
        return True

    async def _stale_rest(self, snap: ConversationSnapshot) -> str | None:
        """Is the rest of a reply that is partly out too late to send? (D-753)

        ``"deep_sleep"``: by her plan she is asleep now; ``"gap"``: the last bubble is longer ago
        than :data:`STALE_REST_S` (the process was stopped, the machine slept).  ``None``:
        the reply goes on.  Asked when the sending of a reply begins or begins again - after a
        restart, a wake-up of the machine or a failed step - never between two of its bubbles.
        """
        now = self._clock.now_utc()
        state = await asyncio.to_thread(self._decider.her_state, now)
        if state is not None and state.kind == "deep_sleep":
            return "deep_sleep"
        last = snap.last_outbound_at
        if last is not None and (now - last).total_seconds() > STALE_REST_S:
            return "gap"
        return None

    async def _drop_stale_rest(self, outgoing: Outgoing, why: str) -> None:
        """The rest of the reply is dropped and noted on it; the round ends (D-753)."""
        log.info("send_dropped_stale", unsent=len(outgoing.unsent), reason=why)
        noted = {"step": "send_dropped_stale", "count": len(outgoing.unsent), "detail": why}
        await self._edit(lambda d: _with_extra(d, noted))
        await self._finish_reply(RoundData.of(await self._load()), aborted=True)

    async def _settle_in_doubt(self, data: RoundData, outgoing: Outgoing) -> None:
        """The process stopped while a bubble was being handed to the channel: it counts as sent.

        It may be on the user's phone or not; there is no way to know, and a bubble said twice
        hurts more than one that is missing (the channel's rule for an unknown outcome).  So it
        is written down like any other - with the moment the send began and a note in the reply's
        actions - and the reply goes on with the bubbles after it.
        """
        bubble = outgoing.in_flight
        if bubble is None:
            return
        log.warning("bubble_in_doubt", kind=bubble.kind)
        sent = SentBubble(
            bubble, outgoing.in_flight_at or self._clock.now_utc(), message_id=None, ambiguous=True
        )
        recorder = _Recorder(self._store, outgoing.final_meta(), outgoing.reply_id)
        await _written(recorder.record(sent))
        if recorder.reply_id is not None:
            noted = {"step": "bubble_in_doubt", "count": 1}
            await self._edit(lambda d: _with_extra(d, noted))
        await self._after_bubble(sent, recorder.reply_id, data.answering)

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
            kept = replace(
                out,
                unsent=tuple(rest),
                reply_id=reply_id or out.reply_id,
                in_flight=None,
                in_flight_at=None,
            )
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
        left = RoundData.of(await self._load()).outgoing
        if left is not None and left.in_flight is not None:
            await self._edit(_without_in_flight)  # a send that ended without a bubble out
        if report.skipped:
            log.info("bubbles_skipped", count=len(report.skipped))
            await self._edit(
                lambda d: _with_extra(d, {"step": "bubble_skipped", "count": len(report.skipped)})
            )
        if report.stop is None:
            await self._finish_reply(RoundData.of(await self._load()))
        elif report.stop is StopReason.INTERRUPTED:
            if self._resume_pending or self._stopping:
                return  # the machine woke up (the driver fits the state to the moment first), or
                # the engine is being stopped: what is unsent stays for the next start
            current = await self._load()
            data = RoundData.of(current)
            if data.outgoing is not None:
                await self._interrupted(current, data, data.outgoing)
        else:  # the session, the login, the count, the channel or a probe ended the reply
            data = RoundData.of(await self._load())
            out = data.outgoing
            if out is not None and out.reply_id is None:  # nothing of it is out
                if report.stop is StopReason.PROBE:  # the probe began meanwhile: wait for it
                    log.info("reply_held_for_probe_before_first_bubble")
                    await self._draft_back_to_deciding()
                    return
                if self._send_retryable(report, data, out):
                    await self._retry_later("send_failed", data)
                    return
            stopped = {"step": "send_stopped", "detail": report.stop.value}
            await self._edit(lambda d: _with_extra(d, stopped))
            await self._finish_reply(RoundData.of(await self._load()), aborted=True)

    @staticmethod
    def _send_retryable(report: SendReport, data: RoundData, outgoing: Outgoing) -> bool:
        """Is a reply none of whose bubbles left the machine tried again later? (D-751)

        Only when the channel could not reach the platform (``NETWORK``, also after the sender
        repeated it), while the round has retries left, and never her short answer of the last
        resort (it ends the round as before: no loop).
        """
        return (
            report.stop is StopReason.FAILED
            and report.failed_kind is OutboundKind.NETWORK
            and data.retries < MAX_RETRIES
            and outgoing.meta.backend != FALLBACK_BACKEND
        )

    async def _interrupted(
        self, snap: ConversationSnapshot, data: RoundData, outgoing: Outgoing
    ) -> None:
        """The user wrote while she was sending: the rest is dropped, the round goes on."""
        log.info("sending_interrupted", unsent=len(outgoing.unsent), out=len(snap.sent))
        if outgoing.reply_id is None and data.decision is not None and self._in_her_delay(data):
            decision = data.decision  # the draft was early: the user wrote inside her delay
            self._reasoning = None
            await self._apply(lambda current: self._back_to_her_delay(current, decision))
            return
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
        if not aborted:
            await self._propose_correction(data.answering)
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
        """Sleep ``seconds``; ``True`` if the user wrote, the machine woke or a stop was asked."""
        end = self._clock.monotonic() + seconds
        while True:
            if self._arrival_seq != seq or self._resume_pending or self._stopping:
                return True
            remaining = end - self._clock.monotonic()
            if remaining <= 0:
                return False
            self._wake.clear()
            if self._arrival_seq != seq or self._resume_pending or self._stopping:
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
                await _written(self._extract(latest, through))

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
        await _written(self._extract(latest, through))
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

    async def _probe_running(self) -> bool:
        """Is a channel probe running?  The probe has the channel to itself (D-684)."""
        check = self._probe_active
        return check is not None and await asyncio.to_thread(check)

    async def _probe_hold(self, snap: ConversationSnapshot) -> bool:
        """Hold the round while a probe runs; ``True`` if the driver has nothing more to do now.

        Nothing is decided, written or sent meanwhile: what the user writes is stored (and
        screened for a crisis before this is asked) and waits in COLLECTING.  When the probe has
        ended the round is timed afresh from that moment, as a continuation: she has "just seen"
        the messages, and answers after a short pause instead of at once as if they were old
        news.  The pause of ``/暂停`` is a separate thing (``_active_pause``).
        """
        if await self._probe_running():
            await self._hold(snap, HOLD_PROBE)
            await self._wait_wake(PROBE_RECHECK_S)
            return True
        return await self._release(snap, HOLD_PROBE)

    async def _ready_hold(self, snap: ConversationSnapshot) -> bool:
        """Hold the round until the bot is ready (D-871); ``True`` if the driver is done for now.

        The same hold as for a probe (:meth:`_hold`): asked after the crisis screening, so a
        crisis is answered before anything else.  While held, the system message that says what
        is missing goes out when its book says it is due (:meth:`_tell_not_ready`).  Once ready
        the round is timed afresh from that moment, as after a probe; the decision then reads her
        plan, so a backlog that becomes answerable in her deep sleep waits for her waking.
        """
        readiness = await asyncio.to_thread(self._readiness.check)
        if readiness.ready:
            return await self._release(snap, HOLD_NOT_READY)
        await self._hold(snap, HOLD_NOT_READY, missing=readiness.key)
        await self._tell_not_ready(snap, readiness)
        await self._wait_wake(READY_RECHECK_S)
        return True

    async def _tell_not_ready(self, snap: ConversationSnapshot, readiness: Readiness) -> None:
        """Say what is missing, if the book says it is due; noted only once it is delivered."""
        now = self._clock.now_utc()
        text = await asyncio.to_thread(self._readiness.notice, readiness, now, snap.last_inbound_at)
        if text is None:
            return
        if await self._say_system(SYSTEM_PREFIX + text):
            log.info("readiness_notice_sent", missing=readiness.key)
            await asyncio.to_thread(self._readiness.noted, readiness, now)

    async def _hold(self, snap: ConversationSnapshot, why: str, **detail: Any) -> None:
        """Hold the round for ``why``: nothing is decided, written or sent; it waits in COLLECTING.

        A decision or a planned time made before the hold is taken back.  The one mechanism of
        the probe (D-684) and of readiness (D-871).
        """
        if why not in self._held:
            log.info("round_held", why=why, waiting=len(snap.pending), **detail)
        self._held.add(why)
        if snap.state != STATE_COLLECTING:  # a decision or a planned time made before the hold
            await self._edit(
                lambda d: replace(d, decision=None, decided_for=0),
                state=STATE_COLLECTING,
                planned_send_at=None,
            )

    async def _release(self, snap: ConversationSnapshot, why: str) -> bool:
        """The hold ``why`` is over: the round is timed afresh from now, as a continuation (she
        has "just seen" the messages).  ``True`` if it was held: the driver looks again."""
        if why not in self._held:
            return False
        self._held.discard(why)
        log.info("round_released", why=why, waiting=len(snap.pending))
        now = self._clock.now_utc()
        await self._edit(lambda d: replace(d, quiet_from=now, skip_quiet=True))
        return True

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


def _crisis_sent(data: RoundData, text: str, reply_id: str | None) -> RoundData:
    """The notes with one more bubble of the answer to a crisis out (and its reply row)."""
    answer = data.crisis
    if answer is None:
        return data
    rest = list(answer.unsent)
    with contextlib.suppress(ValueError):
        rest = rest[rest.index(text) + 1 :]
    return replace(
        data,
        crisis=replace(answer, unsent=tuple(rest), reply_id=reply_id or answer.reply_id),
    )


def _crisis_retry(data: RoundData, at: datetime) -> RoundData:
    """The notes with the answer to a crisis to be tried again at ``at``."""
    answer = data.crisis
    if answer is None:
        return data
    return replace(data, crisis=replace(answer, retries=answer.retries + 1, retry_at=at))


def uninterruptible_pause(clock: Clock) -> Wait:
    """A pause that really lasts and that nothing cuts short.

    The out-of-role answer to a crisis goes out whatever the user writes next, so its sender never
    hears of a new message; but when the network fails, the sender still waits before it repeats a
    send, and that wait must take time (it does not return at once).
    """

    async def pause(seconds: float) -> bool:
        await clock.sleep(max(0.0, seconds))
        return False

    return pause


def _with_in_flight(data: RoundData, bubble: OutBubble, at: datetime) -> RoundData:
    """The notes with the bubble that is being handed to the channel."""
    out = data.outgoing
    if out is None:
        return data
    return replace(data, outgoing=replace(out, in_flight=bubble, in_flight_at=at))


def _without_in_flight(data: RoundData) -> RoundData:
    out = data.outgoing
    if out is None or out.in_flight is None:
        return data
    return replace(data, outgoing=replace(out, in_flight=None, in_flight_at=None))


async def _written(work: Coroutine[Any, Any, None]) -> None:
    """Finish writing down a bubble that is out, even if the driver is stopped meanwhile.

    The channel has the bubble before the engine notes it (its row, then ``sent`` in the stored
    state); the same goes for a conversation that is handed to the extractor and the mark of how
    far it got.  A stop that lands in between - ``Ctrl+C``, the end of ``twin chat --local`` - would
    leave the state without it and the next start would send it again.  The write is therefore
    not cancelled with the driver: the driver waits for it, and then stops.  (A kill of the whole
    process in that instant, or a stop that lands while the channel is still sending, is covered
    by the note that is made before each send: :meth:`ConversationEngine._settle_in_doubt`.)
    """
    task = asyncio.ensure_future(work)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


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
