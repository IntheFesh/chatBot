"""Corrections in plain words: "她不会这么说" (R-LRN-002), and facts in a ``/不像`` wording.

When the user answers the bot with a remark about the way it talks - "她不会这么说", "你说话不像
她", "别这么说话" - the bot first answers like anyone would, and then asks, as a system message,
whether this should be recorded as "不像"; the user's "是" records it.  Nothing is recorded on a
guess: the question needs the user's yes, within ``commands.confirm_window_min`` minutes, and the
yes only counts as the answer to the question while the persona has not said anything since (a
"是" in the middle of the conversation is a "是" to whatever she asked).  A "不用" / "算了" /
"不是" in the same place is the answer "no": the question is closed and nothing is recorded.

*Noticing.*  A local rule (:func:`looks_like_correction`) reads the message first - most chat never
reaches the model, and the rule favours recall (D-673) - and only a message that matches is judged
by DeepSeek (``correction_check`` template: is this about the way the reply was said?).  A message
the model does not confirm, a model that is not reachable, a budget that does not allow it: no
question.

*The question* is stored (``learning.correction.pending``: the reply it is about, when it was
asked and what kind it is), survives a restart, and is replaced by a newer one; a question whose
system message the channel did not deliver is withdrawn (:meth:`CorrectionService.withdraw`,
D-755) - the user never saw it, so his next "是" is chat.  The reply it is
about is the newest reply of the persona before the user's remark.  *The yes* writes ``feedback``
(``not_like``) through :class:`~twin.learning.dislike.NotLikeRecorder` - no wording, so no
preference pair; the reply becomes one of the negative examples of the weekly consolidation
(R-LRN-003).

*Facts.*  A ``/不像 <说法>`` whose wording states a fact ("我们是在上海认识的") is a correction of
what the reply *said*, not of how she said it, and a rule of the persona card cannot hold it
(R-LRN-003 refuses facts).  :class:`FactProposer` notices such a wording (the local rule of
:mod:`twin.learning.rulecheck`, else one small DeepSeek call, ``correction_fact_check``) and
stores the same kind of question with ``kind="fact"`` and the wording; the command's answer ends
with the question, and the yes stores the wording with :meth:`MemoryManager.remember` (source
``user_command``).  Without a yes it is not kept (D-674).

The engine reaches this service through :class:`~twin.engine.correction_port.CorrectionPort`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict

from twin.clock import ensure_aware
from twin.commands import texts
from twin.commands.memory_commands import render_remembered
from twin.commands.parse import fold
from twin.config.secrets import SecretStoreError
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.machine import CORRECTION_TIMEOUT_S
from twin.engine.rounds import RoundStore
from twin.engine.turns import BotTurnStore
from twin.learning.dislike import SAFETY_BACKEND, NotLikeRecorder
from twin.learning.pairs import reply_text
from twin.learning.rulecheck import check_rule
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import LlmError
from twin.llm.types import DAILY, Purpose
from twin.memory.manage import MemoryManager
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import CORRECTION_CHECK, CORRECTION_FACT_CHECK, TemplateStore
from twin.services import Services
from twin.storage.settings_store import delete_setting, get_setting, put_setting

log = get_logger("twin.learning.corrections")

PENDING_KEY = "learning.correction.pending"
KIND_NOT_LIKE = "not_like"  # the question is "record the last reply as 不像?"
KIND_FACT = "fact"  # the question is "remember what the wording of a /不像 said?"
MAX_REMARK_CHARS = 200  # a longer message is a conversation, not a remark about the wording
SLASHES = ("/", "／")
MAX_REPLY_CHARS = 400

# The local rule is a pre-filter that favours recall (D-673): it only decides what is worth one
# small model call, and DeepSeek's ``correction_check`` is the judge.  The messages are read
# normalised (white space and punctuation removed, width folded, case folded), so the parts below
# are plain strings of that text.
_ADVERB = r"(?:才|从来|根本|一般|肯定|绝对|压根|真的|其实|本来|就)?"
_NOT = r"(?:不会|不爱|不用|不说|不是|不像|从不|从来不|才不|没有|没)"
_THIS = r"(?:这么|这样|那么|那样|这种|那种|这个|那个|这类|这句)"
_SAY = r"(?:说|讲|回|用|写|发|叫|喊)"
_WORDS = r"(?:话|词|语气|口气|说法|用词|措辞|称呼|口吻|腔调|语调)"
_MANNER = (
    r"(?:正式|书面|客气|官方|生硬|啰嗦|机械|刻板|死板|礼貌|严肃|文艺|肉麻|油腻|做作|刻意|僵硬|"
    r"公式化)"
)
_SPEECH = r"(?:说话|讲话|语气|口气|措辞|用词|说法|口吻|腔调|称呼|风格)"
_ASK_NOT_TO = r"(?:别|不要|不许|不准|少)"

_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        # 她 + not + "this way" + a way of speaking / a manner: 她不会这么说, 她没这么正式
        rf"她{_ADVERB}{_NOT}{_THIS}(?:{_SAY}|{_WORDS}|{_MANNER})",
        # 她 + not + say/use + (this) + words: 她不会说这种话, 她从来不用这个词
        rf"她{_ADVERB}{_NOT}{_SAY}{_THIS}?{_WORDS}",
        # her way of speaking + not + "this way": 她说话没这么正式, 她的语气不是这样的
        rf"她(?:的)?{_SPEECH}{_ADVERB}{_NOT}{_THIS}",
        # not like her / not like a person: 不像她, 不像真人, 不是她会说的
        r"不像(?:她|真人|个人|人说|人话)",
        r"不是她(?:的)?(?:说话|讲话|语气|口气|风格|方式|口吻|腔调|说的|会说|会讲|能说)",
        # how it sounds: 听起来好假, 听起来像机器人, 像客服, ai味, 书面腔
        r"听起来(?:很|好|太|有点|特别|挺|都)?(?:假|不像|像|机器|书面|生硬|刻意|官方|客服|奇怪|怪)",
        r"像(?:个|是)?(?:机器人|客服|ai|助手|机器)",
        r"(?:ai|书面|客服|机器|翻译|官方|公文)(?:腔|味)",
        # too formal / too polite: 太正式了, 这么客气, 好生硬, 太假
        rf"(?:太|这么|好|很|有点|特别|过于|过分){_MANNER}",
        r"(?:太|好|很|有点|特别)假",
        # her way of speaking is off: 语气不对, 说话不像, 措辞太生硬, 说法有点怪
        rf"{_SPEECH}(?:方式)?(?:不对|不像|不太对|不太像|不自然|不对劲|有点怪|很怪|怪怪的|太|"
        r"没这么|不够|有问题)",
        # asking for another way: 别这么说话, 别用这种语气, 换个语气
        rf"{_ASK_NOT_TO}(?:这么|这样|那样|那么)(?:说话|讲话|说|讲|回|客气|正式|书面|严肃|啰嗦)",
        rf"{_ASK_NOT_TO}{_SAY}{_THIS}{_WORDS}",
        rf"(?:换|改)(?:个|种|一种|一下|下)(?:{_WORDS}|说话方式|说话的方式)",
        # saying what she would say: 她一般会说哈哈, 她平时都回好呀
        r"她(?:一般|通常|平时|平常|应该|肯定|大概|多半|总是|一定|都){1,2}(?:会)?(?:说|回|讲)",
    )
)
_FILLER = re.compile(r"[\s,.!?;:~，。！？；：、…]+")


def normalise(text: str) -> str:
    """``text`` as the rules read it: width folded, white space and punctuation removed."""
    return _FILLER.sub("", fold(text))


def looks_like_correction(text: str) -> bool:
    """Does the message look like a remark that the reply did not sound like her?"""
    if len(text) > MAX_REMARK_CHARS or text.lstrip()[:1] in SLASHES:
        return False  # a command is answered by the router; a long message is a conversation
    plain = normalise(text)
    return any(pattern.search(plain) for pattern in _PATTERNS)


def is_confirmation_word(text: str) -> bool:
    """Is the message just the "yes" that answers the question?"""
    return normalise(text) in {normalise(word) for word in texts.CORRECTION_CONFIRM_WORDS}


def is_rejection_word(text: str) -> bool:
    """Is the message just the "no" (不用, 算了, 不是 ...) that answers the question?"""
    return normalise(text) in {normalise(word) for word in texts.CORRECTION_REJECT_WORDS}


class CorrectionVerdict(BaseModel):
    """The answer of the ``correction_check`` prompt."""

    model_config = ConfigDict(extra="ignore")

    is_correction: bool


class FactVerdict(BaseModel):
    """The answer of the ``correction_fact_check`` prompt."""

    model_config = ConfigDict(extra="ignore")

    is_fact: bool


@dataclass(frozen=True)
class PendingCorrection:
    """The open question: which reply it is about, when it was asked, its kind and its wording.

    ``text`` is the wording of a ``fact`` question (what the yes remembers); a ``not_like``
    question has none.
    """

    reply_id: str
    asked_at: datetime
    kind: str = KIND_NOT_LIKE
    text: str | None = None


class PendingStore:
    """The open question in the settings table (sealed): one at a time, a newer one replaces it.

    Shared by :class:`CorrectionService` (natural-language corrections, the yes) and
    :class:`FactProposer` (the question after a ``/不像`` whose wording is a fact).  A value
    written before facts were asked about (no ``kind``) reads as a ``not_like`` question.
    """

    def __init__(self, services: Services) -> None:
        self._services = services

    def get(self) -> PendingCorrection | None:
        with self._services.db.session() as session:
            raw = get_setting(session, PENDING_KEY)
        if not isinstance(raw, dict):
            return None
        try:
            kind = str(raw.get("kind", KIND_NOT_LIKE))
            text = raw.get("text")
            if kind not in (KIND_NOT_LIKE, KIND_FACT) or (kind == KIND_FACT and not text):
                return None
            return PendingCorrection(
                str(raw["reply_id"]),
                datetime.fromisoformat(raw["asked_at"]),
                kind,
                str(text) if text else None,
            )
        except (KeyError, ValueError, TypeError):
            return None

    def put(self, pending: PendingCorrection | None) -> None:
        """Store ``pending`` (``None`` closes the question)."""
        with self._services.db.transaction(bump_state=False) as session:
            if pending is None:
                delete_setting(session, PENDING_KEY)
                return
            value: dict[str, str] = {
                "reply_id": pending.reply_id,
                "asked_at": pending.asked_at.isoformat(),
                "kind": pending.kind,
            }
            if pending.text:
                value["text"] = pending.text
            put_setting(
                session,
                PENDING_KEY,
                value,
                clock=self._services.clock,
                by="learning",
                record_history=False,
            )


class CorrectionService:
    """Notices corrections in plain words and records the confirmed ones (see the module text)."""

    def __init__(
        self,
        services: Services,
        client: DeepSeekClient | None,
        recorder: NotLikeRecorder,
        turns: BotTurnStore,
        rounds: RoundStore,
        *,
        memory: MemoryManager,
        templates: TemplateStore | None = None,
    ) -> None:
        self._services = services
        self._client = client
        self._recorder = recorder
        self._turns = turns
        self._rounds = rounds
        self._memory = memory
        self._pending = PendingStore(services)
        self._templates = templates or TemplateStore(services.db, services.clock)
        self._window = timedelta(minutes=services.settings.commands.confirm_window_min)

    # --------------------------------------------------------------------- the open question

    def pending(self) -> PendingCorrection | None:
        """The question that is open now (it may have lapsed; see :meth:`_valid`)."""
        return self._pending.get()

    def _store(self, pending: PendingCorrection | None) -> None:
        self._pending.put(pending)

    def _valid(self, pending: PendingCorrection | None, at: datetime) -> bool:
        """Is the question still open at ``at``: in the window, and she has not spoken since?"""
        if pending is None:
            return False
        asked = ensure_aware(pending.asked_at)
        moment = ensure_aware(at)
        if moment < asked or moment - asked > self._window:
            return False
        return not self._turns.spoke_since(asked)

    # ---------------------------------------------------------------------- the engine's port

    async def is_confirmation(self, text: str, at: datetime) -> bool:
        """Is ``text`` the answer, yes or no, to the question that is open at ``at``?"""
        if not (is_confirmation_word(text) or is_rejection_word(text)):
            return False
        return await asyncio.to_thread(lambda: self._valid(self.pending(), at))

    async def confirm(self, context: CommandContext) -> CommandOutcome | None:
        def take() -> tuple[PendingCorrection | None, bool]:
            pending = self.pending()
            if not self._valid(pending, context.at):
                return None, False
            self._store(None)
            return pending, self._said_no(context.inbound_id)

        pending, said_no = await asyncio.to_thread(take)
        if pending is None:
            return None
        if said_no:
            return CommandOutcome(texts.PREFIX + texts.CORRECTION_REJECTED)
        if pending.kind == KIND_FACT:
            return await self._remember(pending)
        result = await self._recorder.arecord(None, reply_id=pending.reply_id)
        if result.status != "recorded":
            return CommandOutcome(texts.PREFIX + texts.NOT_LIKE_NOTHING)
        return CommandOutcome(texts.PREFIX + texts.CORRECTION_DONE)

    def _said_no(self, turn_id: str) -> bool:
        """Is the message of the row ``turn_id`` a "no"?  (A row that is not there is a yes.)"""
        row = self._turns.get(turn_id)
        return row is not None and is_rejection_word(row.text)

    async def _remember(self, pending: PendingCorrection) -> CommandOutcome:
        """The yes to a fact question: the wording becomes a fact of the memory (once)."""
        try:
            result = await self._memory.remember(pending.text or "")
        except Exception:
            await asyncio.to_thread(self._store, pending)  # not lost: another "是" tries again
            raise
        log.info("correction_fact_remembered")
        return CommandOutcome(texts.PREFIX + render_remembered(result))

    async def withdraw(self, asked_since: datetime) -> None:
        """The question asked at or after ``asked_since`` never reached the user: close it.

        Its system message was refused or could not be sent (D-755).  An older question - one
        the user did see - is left as it is.
        """

        def take() -> bool:
            pending = self.pending()
            if pending is None or ensure_aware(pending.asked_at) < ensure_aware(asked_since):
                return False
            self._store(None)
            return True

        if await asyncio.to_thread(take):
            log.info("correction_question_withdrawn")

    async def propose(self, answered: Sequence[str]) -> str | None:
        settings = self._services.settings.learning
        if not settings.detect_corrections or self._client is None or not answered:
            return None
        found = await asyncio.to_thread(self._candidate, answered)
        if found is None:
            return None
        reply_id, remark, reply = found
        if not await self._judged(remark, reply):
            return None
        asked = self._services.clock.now_utc()
        await asyncio.to_thread(self._store, PendingCorrection(reply_id, asked))
        log.info("correction_proposed")
        return texts.PREFIX + texts.CORRECTION_ASK

    def _candidate(self, answered: Sequence[str]) -> tuple[str, str, str] | None:
        """``(reply id, the remark, the reply)`` when a message of the round looks like a remark."""
        items = self._rounds.inbound_items(answered)
        remarks = [
            item for item in items if item.kind == "text" and looks_like_correction(item.text)
        ]
        if not remarks:
            return None
        first = min(items, key=lambda item: item.at)
        rows = self._turns.reply_before(first.at)
        if not rows or rows[0].reply_id is None or rows[0].backend == SAFETY_BACKEND:
            return None
        for feedback in self._recorder.feedback_of(rows[0].reply_id):
            if feedback.type == "not_like":
                return None  # already recorded: nothing to ask
        reply = reply_text(rows)[:MAX_REPLY_CHARS]
        return rows[0].reply_id, remarks[-1].text, reply

    async def _judged(self, remark: str, reply: str) -> bool:
        client = self._client
        if client is None:
            return False
        try:
            template = await asyncio.to_thread(self._templates.active, CORRECTION_CHECK)
            messages = template.render(reply=reply, message=remark)
            verdict = await client.chat_json(
                messages, CorrectionVerdict, purpose=Purpose.EXTRACT, tag=DAILY
            )
        except (LlmError, SecretStoreError) as exc:
            log.info("correction_not_judged", error=type(exc).__name__)
            return False
        return verdict.value.is_correction


class FactProposer:
    """After ``/不像 <说法>``: does the wording state a fact worth remembering? (D-674)

    The question comes from what the user's wording says, so it is only asked when
    ``learning.detect_corrections`` is on.  The local rule of the persona card says "fact" for a
    number, a date or a time, an amount of money or an event; otherwise one small DeepSeek call
    decides (``correction_fact_check``, bounded by the engine's correction timeout).  A model that
    is not reachable, a budget that does not allow it, a wrong answer: no question - the command
    has done what it was asked to do either way.
    """

    def __init__(
        self,
        services: Services,
        client: DeepSeekClient | None,
        turns: BotTurnStore,
        pending: PendingStore,
        *,
        templates: TemplateStore | None = None,
    ) -> None:
        self._services = services
        self._client = client
        self._turns = turns
        self._pending = pending
        self._templates = templates or TemplateStore(services.db, services.clock)

    async def propose(self, wording: str, reply_id: str | None) -> str | None:
        """The question to append to the command's answer, or ``None``; the question is stored."""
        settings = self._services.settings.learning
        text = " ".join(wording.split())
        if not settings.detect_corrections or not text:
            return None
        local = check_rule(text, max_chars=settings.rule_max_chars).kind == "fact"
        if not local and not await self._judged(text, reply_id):
            return None
        asked = self._services.clock.now_utc()
        await asyncio.to_thread(
            self._pending.put, PendingCorrection(reply_id or "", asked, KIND_FACT, text)
        )
        log.info("fact_proposed", local=local)
        return texts.FACT_ASK

    async def _judged(self, text: str, reply_id: str | None) -> bool:
        client = self._client
        if client is None:
            return False
        reply = await asyncio.to_thread(self._reply_of, reply_id)
        try:
            template = await asyncio.to_thread(self._templates.active, CORRECTION_FACT_CHECK)
            messages = template.render(reply=reply, message=text)
            verdict = await asyncio.wait_for(
                client.chat_json(messages, FactVerdict, purpose=Purpose.EXTRACT, tag=DAILY),
                CORRECTION_TIMEOUT_S,
            )
        except (LlmError, SecretStoreError, TimeoutError) as exc:
            log.info("fact_not_judged", error=type(exc).__name__)
            return False
        return verdict.value.is_fact

    def _reply_of(self, reply_id: str | None) -> str:
        rows = self._turns.reply(reply_id) if reply_id else []
        return reply_text(rows)[:MAX_REPLY_CHARS]
