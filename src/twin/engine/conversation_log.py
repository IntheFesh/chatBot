"""The bot's conversation as plain records, for the code that must not see its table (R-EVAL-002).

The style metrics of the evaluation measure what the bot really sent in the last days.  Only the
engine and the storage may name the table of the bot's conversation (R-STO-007); this module is the
engine's answer to a reader outside that circle: the messages of the conversation since a moment,
as values, in time order.  Commands, the decision to say nothing and replies the user threw away
are not messages of the conversation and are left out, as they are for the memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from twin.clock import ensure_aware
from twin.engine.turns import conversation_rows
from twin.storage.engine_models import BotTurn

if TYPE_CHECKING:
    from twin.services import Services


@dataclass(frozen=True)
class LoggedTurn:
    """One message of the bot's conversation: what the user said, or one bubble of the bot."""

    id: str
    at: datetime
    outbound: bool
    kind: str
    text: str
    sticker_md5: str | None


def conversation_since(services: Services, since: datetime) -> list[LoggedTurn]:
    """The messages of the conversation at or after ``since``, oldest first."""
    stmt = (
        conversation_rows()
        .where(BotTurn.at >= ensure_aware(since))
        .order_by(BotTurn.at, BotTurn.id)
    )
    with services.db.session() as session:
        return [
            LoggedTurn(row.id, row.at, row.direction == "out", row.kind, row.text, row.sticker_md5)
            for row in session.scalars(stmt)
        ]
