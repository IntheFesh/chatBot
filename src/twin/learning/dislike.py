"""``/不像``: the last reply did not sound like her (R-LRN-002, R-CMD-002).

:class:`NotLikeRecorder` is what the command and the "yes" of a natural-language correction share:
it marks the reply in ``feedback`` (type ``not_like``, with the user's wording when there is one)
and, when there is a wording, stores the preference pair (``chosen`` is the user's wording,
``rejected`` the reply, the situation as a structured sample).  The reply itself stays in the
conversation - it was said, and the user went on from it; only ``/重来`` throws a reply away.

The reply the user is talking about is the newest one of the bot, or the one the caller names (a
natural-language correction points at the reply it was made about).  Replies that were not
the persona's - the answer out of the role to a crisis, a system message, a reply that was
silence - are not "unlike her" and are refused.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from twin.engine.feedback import FeedbackRecord, FeedbackStore
from twin.engine.turns import BotTurnStore, TurnRecord
from twin.learning.pairs import (
    PairError,
    PairRecord,
    PreferencePairStore,
    clean_wording,
    reply_text,
)
from twin.learning.sample import SampleBuilder
from twin.ops.logging import get_logger

log = get_logger("twin.learning.dislike")

SAFETY_BACKEND = "safety"

NotLikeStatus = Literal["recorded", "no_reply", "not_hers", "unknown_reply"]


@dataclass(frozen=True)
class NotLikeResult:
    """What a ``/不像`` did."""

    status: NotLikeStatus
    reply_id: str | None = None
    feedback_id: str | None = None
    pair: PairRecord | None = None
    pair_created: bool = False
    pair_skipped: str | None = None  # why a given wording did not become a pair
    pairs_total: int = 0


class NotLikeRecorder:
    """Marks a reply as "not like her" and stores the pair (see the module description)."""

    def __init__(
        self,
        turns: BotTurnStore,
        feedback: FeedbackStore,
        pairs: PreferencePairStore,
        samples: SampleBuilder,
    ) -> None:
        self._turns = turns
        self._feedback = feedback
        self._pairs = pairs
        self._samples = samples

    def record(self, correction: str | None, *, reply_id: str | None = None) -> NotLikeResult:
        """Record the verdict on ``reply_id`` (the newest reply if not given)."""
        rows = self._turns.reply(reply_id) if reply_id else self._turns.latest_reply()
        if not rows:
            return NotLikeResult("unknown_reply" if reply_id else "no_reply")
        first = rows[0]
        if first.reply_id is None or first.is_command or first.direction != "out":
            return NotLikeResult("not_hers")
        if first.backend == SAFETY_BACKEND:
            return NotLikeResult("not_hers", first.reply_id)
        text = reply_text([row for row in rows if row.kind != "no_reply"])
        if not text:
            return NotLikeResult("no_reply", first.reply_id)
        wording = clean_wording(correction) if correction else ""
        feedback_id = self._feedback_for(first, first.reply_id, wording or None)
        if not wording:
            return NotLikeResult(
                "recorded", first.reply_id, feedback_id, pairs_total=self._pairs.count()
            )
        try:
            built = self._samples.build(rows)
            pair, created = self._pairs.add(
                reply_id=first.reply_id,
                sample=built.sample,
                chosen=wording,
                rejected=text,
                template_version=built.template_version,
                persona_version=built.persona_version,
                feedback_id=feedback_id,
            )
        except PairError as exc:
            log.info("pair_not_made", reason=str(exc)[:120])
            return NotLikeResult(
                "recorded",
                first.reply_id,
                feedback_id,
                pair_skipped=str(exc),
                pairs_total=self._pairs.count(),
            )
        return NotLikeResult(
            "recorded", first.reply_id, feedback_id, pair, created, pairs_total=self._pairs.count()
        )

    def feedback_of(self, reply_id: str) -> list[FeedbackRecord]:
        """What the user already said about this reply."""
        return self._feedback.for_reply(reply_id)

    def _feedback_for(self, first: TurnRecord, reply_id: str, wording: str | None) -> str:
        """The ``not_like`` row of this reply and wording (a repeated verdict adds none)."""
        for existing in self._feedback.for_reply(reply_id):
            if existing.type == "not_like" and existing.correction == wording:
                return existing.id
        return self._feedback.add("not_like", reply_id, bot_turn_id=first.id, correction=wording).id

    async def arecord(
        self, correction: str | None, *, reply_id: str | None = None
    ) -> NotLikeResult:
        return await asyncio.to_thread(self.record, correction, reply_id=reply_id)
