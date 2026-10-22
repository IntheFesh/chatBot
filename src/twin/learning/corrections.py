"""Corrections in plain words: "她不会这么说" (R-LRN-002).

When the user answers the bot with a remark about the way it talks - "她不会这么说", "你说话不像
她", "别这么说话" - the bot first answers like anyone would, and then asks, as a system message,
whether this should be recorded as "不像"; the user's "是" records it.  Nothing is recorded on a
guess: the question needs the user's yes, within ``commands.confirm_window_min`` minutes, and the
yes only counts as the answer to the question while the persona has not said anything since (a
"是" in the middle of the conversation is a "是" to whatever she asked).

*Noticing.*  A local rule (:func:`looks_like_correction`) reads the message first - most chat never
reaches the model, and the rule favours recall (D-673) - and only a message that matches is judged
by DeepSeek
(``correction_check`` template: is this about the way the reply was said?).  A message the model
does not confirm, a model that is not reachable, a budget that does not allow it: no question.

*The question* is stored (``learning.correction.pending``: the reply it is about and when it was
asked), survives a restart, and is replaced by a newer one.  The reply it is about is the newest
reply of the persona before the user's remark.  *The yes* writes ``feedback`` (``not_like``) through
:class:`~twin.learning.dislike.NotLikeRecorder` - no wording, so no preference pair; the reply
becomes one of the negative examples of the weekly consolidation (R-LRN-003).

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
from twin.commands.parse import fold
from twin.config.secrets import SecretStoreError
from twin.engine.command_port import CommandContext, CommandOutcome
from twin.engine.rounds import RoundStore
from twin.engine.turns import BotTurnStore
from twin.learning.dislike import SAFETY_BACKEND, NotLikeRecorder
from twin.learning.pairs import reply_text
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import LlmError
from twin.llm.types import DAILY, Purpose
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import CORRECTION_CHECK, TemplateStore
from twin.services import Services
from twin.storage.settings_store import delete_setting, get_setting, put_setting

log = get_logger("twin.learning.corrections")

PENDING_KEY = "learning.correction.pending"
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


class CorrectionVerdict(BaseModel):
    """The answer of the ``correction_check`` prompt."""

    model_config = ConfigDict(extra="ignore")

    is_correction: bool


@dataclass(frozen=True)
class PendingCorrection:
    """The open question: which reply it is about and when it was asked."""

    reply_id: str
    asked_at: datetime


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
        templates: TemplateStore | None = None,
    ) -> None:
        self._services = services
        self._client = client
        self._recorder = recorder
        self._turns = turns
        self._rounds = rounds
        self._templates = templates or TemplateStore(services.db, services.clock)
        self._window = timedelta(minutes=services.settings.commands.confirm_window_min)

    # --------------------------------------------------------------------- the open question

    def pending(self) -> PendingCorrection | None:
        """The question that is open now (it may have lapsed; see :meth:`_valid`)."""
        with self._services.db.session() as session:
            raw = get_setting(session, PENDING_KEY)
        if not isinstance(raw, dict):
            return None
        try:
            return PendingCorrection(str(raw["reply_id"]), datetime.fromisoformat(raw["asked_at"]))
        except (KeyError, ValueError, TypeError):
            return None

    def _store(self, pending: PendingCorrection | None) -> None:
        with self._services.db.transaction(bump_state=False) as session:
            if pending is None:
                delete_setting(session, PENDING_KEY)
                return
            put_setting(
                session,
                PENDING_KEY,
                {"reply_id": pending.reply_id, "asked_at": pending.asked_at.isoformat()},
                clock=self._services.clock,
                by="learning",
                record_history=False,
            )

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
        if not is_confirmation_word(text):
            return False
        return await asyncio.to_thread(lambda: self._valid(self.pending(), at))

    async def confirm(self, context: CommandContext) -> CommandOutcome | None:
        def take() -> PendingCorrection | None:
            pending = self.pending()
            if not self._valid(pending, context.at):
                return None
            self._store(None)
            return pending

        pending = await asyncio.to_thread(take)
        if pending is None:
            return None
        result = await self._recorder.arecord(None, reply_id=pending.reply_id)
        if result.status != "recorded":
            return CommandOutcome(texts.PREFIX + texts.NOT_LIKE_NOTHING)
        return CommandOutcome(texts.PREFIX + texts.CORRECTION_DONE)

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
