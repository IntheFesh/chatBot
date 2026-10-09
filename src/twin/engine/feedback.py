"""What the user said about a reply: the ``feedback`` table (R-CMD-002, R-LRN-002).

``/重来`` throws the last reply away and asks for another (type ``redo``); ``/不像`` says the reply
did not sound like her (type ``not_like``), optionally with the way she would have said it.  The
commands (rounds 09 step 2 and 11) write here; the learning of round 11 reads the rows that are not
processed yet and turns them into preference pairs.  The rejected reply itself stays in
``bot_turns`` (marked ``rejected_at``); the correction is text the user typed, so it is sealed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, update

from twin.clock import Clock
from twin.storage.db import Database
from twin.storage.engine_models import FEEDBACK_TYPES, Feedback


@dataclass(frozen=True)
class FeedbackRecord:
    id: str
    type: str
    reply_id: str
    bot_turn_id: str | None
    correction: str | None
    processed_at: datetime | None
    created_at: datetime


def _record(row: Feedback) -> FeedbackRecord:
    return FeedbackRecord(
        row.id,
        row.type,
        row.reply_id,
        row.bot_turn_id,
        row.correction,
        row.processed_at,
        row.created_at,
    )


class FeedbackStore:
    """Writes and reads ``feedback`` rows."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(
        self,
        feedback_type: str,
        reply_id: str,
        *,
        bot_turn_id: str | None = None,
        correction: str | None = None,
    ) -> FeedbackRecord:
        """Record that the user rejected (``redo``) or disliked (``not_like``) a reply."""
        if feedback_type not in FEEDBACK_TYPES:
            raise ValueError(f"unknown feedback type {feedback_type!r}")
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = Feedback(
                type=feedback_type,
                reply_id=reply_id,
                bot_turn_id=bot_turn_id,
                correction=correction.strip() if correction and correction.strip() else None,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return _record(row)

    def for_reply(self, reply_id: str) -> list[FeedbackRecord]:
        with self._db.session() as session:
            rows = session.scalars(
                select(Feedback).where(Feedback.reply_id == reply_id).order_by(Feedback.id)
            )
            return [_record(row) for row in rows]

    def unprocessed(self, feedback_type: str | None = None) -> list[FeedbackRecord]:
        """Feedback that learning has not used yet, oldest first."""
        stmt = select(Feedback).where(Feedback.processed_at.is_(None)).order_by(Feedback.id)
        if feedback_type is not None:
            stmt = stmt.where(Feedback.type == feedback_type)
        with self._db.session() as session:
            return [_record(row) for row in session.scalars(stmt)]

    def mark_processed(self, ids: Sequence[str]) -> int:
        """Note that learning used these rows; returns how many were open."""
        if not ids:
            return 0
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            result = session.execute(
                update(Feedback)
                .where(Feedback.id.in_(list(ids)), Feedback.processed_at.is_(None))
                .values(processed_at=now, updated_at=now)
            )
            return int(getattr(result, "rowcount", 0) or 0)
