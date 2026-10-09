"""What the engine reads from ``bot_turns`` besides the conversation itself (R-ENG-001, R-CMD-001).

:class:`~twin.engine.turns.BotTurnStore` writes the conversation and answers the questions of the
history; this module has the few lookups the state machine needs and that belong to no one else:

* the **messages of a round** by id (``conversation_state`` keeps ids, not text: the text stays
  sealed in ``bot_turns``), as :class:`~twin.engine.types.InboundItem` objects;
* whether a message has been **answered** already (a message the channel delivers a second time
  after a restart must not be answered twice, nor lost if it was stored but never queued);
* the messages a reply **answered** (``/重来`` asks for them again);
* the **command** flag of an inbound row (a message that starts with ``/`` is stored as a command
  until the router says it is none, so it can never be read as conversation in between).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, select, update

from twin.clock import ensure_aware
from twin.engine.types import InboundItem
from twin.storage.db import Database
from twin.storage.engine_models import BotTurn


def _item(row: BotTurn) -> InboundItem:
    return InboundItem(
        id=row.external_id or row.id,
        at=ensure_aware(row.at),
        kind=row.kind,
        text=row.text,
        media=row.media,
        turn_id=row.id,
    )


class RoundStore:
    """Lookups on ``bot_turns`` for the state machine."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def inbound_items(self, turn_ids: Sequence[str]) -> list[InboundItem]:
        """The messages with these row ids, oldest first (commands and unknown ids left out)."""
        if not turn_ids:
            return []
        stmt = (
            select(BotTurn)
            .where(
                BotTurn.id.in_(list(turn_ids)),
                BotTurn.direction == "in",
                BotTurn.is_command.is_(False),
            )
            .order_by(BotTurn.at, BotTurn.id)
        )
        with self._db.session() as session:
            return [_item(row) for row in session.scalars(stmt)]

    def set_command(self, turn_id: str, flag: bool) -> None:
        """Mark (or unmark) an inbound row as a command: not conversation, not memory."""
        with self._db.transaction(bump_state=False) as session:
            session.execute(update(BotTurn).where(BotTurn.id == turn_id).values(is_command=flag))

    def last_answer_at(self) -> datetime | None:
        """When the bot last said something (or chose silence) in the conversation."""
        stmt = select(func.max(BotTurn.at)).where(
            BotTurn.direction == "out", BotTurn.is_command.is_(False)
        )
        with self._db.session() as session:
            found = session.scalar(stmt)
        return ensure_aware(found) if found is not None else None

    def answered(self, turn_id: str) -> bool:
        """Has the bot spoken after this message of the user?"""
        with self._db.session() as session:
            row = session.get(BotTurn, turn_id)
            if row is None:
                return True  # nothing to answer
            arrived = ensure_aware(row.at)
            newest = session.scalar(
                select(func.max(BotTurn.at)).where(
                    BotTurn.direction == "out", BotTurn.is_command.is_(False)
                )
            )
        return newest is not None and ensure_aware(newest) >= arrived

    def inbound_of_reply(self, reply_id: str) -> list[str]:
        """The row ids of the user's messages the reply answered (``/重来`` answers them again).

        These are the messages after the bot's previous answer up to the first bubble of this one.
        """
        with self._db.session() as session:
            first = session.scalar(
                select(func.min(BotTurn.at)).where(
                    BotTurn.reply_id == reply_id, BotTurn.direction == "out"
                )
            )
            if first is None:
                return []
            previous = session.scalar(
                select(func.max(BotTurn.at)).where(
                    BotTurn.direction == "out",
                    BotTurn.is_command.is_(False),
                    BotTurn.reply_id != reply_id,
                    BotTurn.at < first,
                )
            )
            stmt = select(BotTurn.id).where(
                BotTurn.direction == "in",
                BotTurn.is_command.is_(False),
                BotTurn.at <= first,
            )
            if previous is not None:
                stmt = stmt.where(BotTurn.at > previous)
            return list(session.scalars(stmt.order_by(BotTurn.at, BotTurn.id)))
