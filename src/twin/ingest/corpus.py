"""The only way style samples, retrieval examples and training data read ``messages``.

R-STO-007 keeps the bot's own words out of everything that teaches the bot how *she* writes:
``messages`` (real messages from exports, every row naming its export) and ``bot_turns`` (the
bot's conversation, round 09) are different tables, and any style / retrieval / training query
starts from the statements below, which read her real messages (``is_sent`` false) only.
Later rounds add their filters (time range, hold-out, kinds) on top of these statements instead
of writing their own ``SELECT`` on ``messages``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Select, select
from sqlalchemy.orm import load_only

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


def conversation_messages(conversation_id: str | None = None) -> Select[Message]:
    """Both sides of the conversation, oldest first, without the bulky raw JSON.

    For statistics that compare her with the user (reply latency needs both sides, a burst
    ends where the other side speaks).  Style samples, retrieval examples and training targets
    never start here: they start from :func:`her_messages`.  ``system`` notices are included;
    the caller drops them.
    """
    stmt = select(Message).options(
        load_only(
            Message.id,
            Message.conversation_id,
            Message.create_time_utc,
            Message.is_sent,
            Message.kind,
            Message.text_ct,
            Message.sticker_md5,
        )
    )
    if conversation_id is not None:
        stmt = stmt.where(Message.conversation_id == conversation_id)
    return stmt.order_by(Message.create_time_utc, Message.sort_seq, Message.id)


def conversation_timeline(
    conversation_id: str | None = None,
) -> Select[datetime, bool, str]:
    """``(time, is_sent, kind)`` of every message of both sides, oldest first (no text)."""
    stmt = select(Message.create_time_utc, Message.is_sent, Message.kind)
    if conversation_id is not None:
        stmt = stmt.where(Message.conversation_id == conversation_id)
    return stmt.order_by(Message.create_time_utc, Message.sort_seq, Message.id)
