"""Corrections in plain words: "她不会这么说" (R-LRN-002).

When the user answers the bot with a remark about the way it talks - "她不会这么说", "你说话不像
她", "别这么说话" - the bot first answers like anyone would, and then asks, as a system message,
whether this should be recorded as "不像"; the user's "是" records it.  Nothing is recorded on a
guess: the question needs the user's yes, within ``commands.confirm_window_min`` minutes, and the
yes only counts as the answer to the question while the persona has not said anything since (a
"是" in the middle of the conversation is a "是" to whatever she asked).

*Noticing.*  A local rule (:func:`looks_like_correction`) reads the message first - most chat never
reaches the model - and only a message that matches is judged by DeepSeek
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

_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"她(?:才|从来|根本|一般|肯定|绝对)?不会(?:这么|这样|那么|那样|用这种)(?:说|讲|回|讲话|说话|语气)",
        r"她不(?:是)?(?:这么|这样|那么|那样)(?:说话|讲话|说|讲)",
        r"不像她",
        r"不是她(?:的)?(?:说话|讲话|语气|口气|风格|方式)",
        r"(?:别|不要|不许|不准|少)(?:这么|这样|那样|那么)(?:说话|讲话|说|讲|回)",
        r"(?:说话|语气|口气|风格)(?:方式)?(?:不对|不像)",
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
