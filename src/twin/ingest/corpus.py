"""The only way style samples, retrieval examples and training data read ``messages``.

R-STO-007 keeps the bot's own words out of everything that teaches the bot how *she* writes:
``messages`` (real messages from exports, every row naming its export) and ``bot_turns`` (the
bot's conversation, round 09) are different tables, and any style / retrieval / training query
starts from the statements below, which read her real messages (``is_sent`` false) only.
Later rounds add their filters (time range, hold-out, kinds) on top of these statements instead
of writing their own ``SELECT`` on ``messages``.
"""

from __future__ import annotations

from sqlalchemy import Select, select

from twin.ingest.events import REPRODUCIBLE_KINDS
from twin.storage.chat_models import Message


def her_messages(conversation_id: str | None = None) -> Select[Message]:
    """Her real messages, oldest first."""
    stmt = select(Message).where(Message.is_sent.is_(False))
    if conversation_id is not None:
        stmt = stmt.where(Message.conversation_id == conversation_id)
    return stmt.order_by(Message.create_time_utc, Message.sort_seq, Message.id)


def her_reproducible_messages(conversation_id: str | None = None) -> Select[Message]:
    """Her real messages of the kinds the bot could send itself (text, sticker, quote)."""
    return her_messages(conversation_id).where(Message.kind.in_(sorted(REPRODUCIBLE_KINDS)))
