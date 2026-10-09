"""The only way style samples, retrieval examples and training data read ``messages``.

R-STO-007 keeps the bot's own words out of everything that teaches the bot how *she* writes:
``messages`` (real messages from exports, every row naming its export) and ``bot_turns`` (the
bot's conversation, round 09) are different tables, and any style / retrieval / training query
starts from the statements below, which read her real messages (``is_sent`` false) only.
Later rounds add their filters (time range, hold-out, kinds) on top of these statements instead
of writing their own ``SELECT`` on ``messages``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import Select, func, select
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


def conversation_skeleton(conversation_id: str | None = None) -> Select[Message]:
    """Both sides in time order with only the columns that place a message in the conversation.

    ``id``, ``conversation_id``, ``create_time_utc``, ``is_sent`` and ``kind``: no text, no raw
    JSON.  The example windows of the retrieval library (round 05) are laid out from it; the
    words are fetched by id when a window is encoded or rendered.  ``system`` notices are
    included; the caller drops them.
    """
    stmt = select(Message).options(
        load_only(
            Message.id,
            Message.conversation_id,
            Message.create_time_utc,
            Message.is_sent,
            Message.kind,
        )
    )
    if conversation_id is not None:
        stmt = stmt.where(Message.conversation_id == conversation_id)
    return stmt.order_by(Message.create_time_utc, Message.sort_seq, Message.id)


def messages_by_ids(ids: Sequence[str]) -> Select[Message]:
    """The ``messages`` rows with these ids (both sides, any order).

    Ids of the bot's own turns are not rows of this table, so they simply never match: the
    retrieval library renders its windows from ids and cannot be handed the bot's words.
    """
    return select(Message).where(Message.id.in_(list(ids)))


def conversation_timeline(
    conversation_id: str | None = None,
) -> Select[datetime, bool, str]:
    """``(time, is_sent, kind)`` of every message of both sides, oldest first (no text)."""
    stmt = select(Message.create_time_utc, Message.is_sent, Message.kind)
    if conversation_id is not None:
        stmt = stmt.where(Message.conversation_id == conversation_id)
    return stmt.order_by(Message.create_time_utc, Message.sort_seq, Message.id)


def her_message_count(before: datetime | None = None) -> Select[int]:
    """How many messages she wrote (``system`` notices excluded), optionally before a moment.

    The persona card remembers this number when its description is written, and asks for a new
    description when it has grown by ``persona.regen_ratio`` (R-IMP-011).
    """
    stmt = (
        select(func.count())
        .select_from(Message)
        .where(Message.is_sent.is_(False), Message.kind != "system")
    )
    if before is not None:
        stmt = stmt.where(Message.create_time_utc < before)
    return stmt


def her_bubble_skeleton(before: datetime | None = None) -> Select[Message]:
    """Her messages in time order with only id, time, kind and sticker MD5 (no text).

    The sticker selector measures from it how often she repeats a sticker within a few bubbles
    (R-STK-004); ``before`` limits the data to one scope (the hold-out cutoff for the
    pre-holdout view).  ``system`` notices are not bubbles and are left out.
    """
    stmt = (
        select(Message)
        .options(
            load_only(
                Message.id,
                Message.create_time_utc,
                Message.kind,
                Message.sticker_md5,
            )
        )
        .where(Message.is_sent.is_(False), Message.kind != "system")
    )
    if before is not None:
        stmt = stmt.where(Message.create_time_utc < before)
    return stmt.order_by(Message.create_time_utc, Message.sort_seq, Message.id)


def messages_between(
    start: datetime, end: datetime, *, before: datetime | None = None
) -> Select[Message]:
    """Both sides of the conversation from ``start`` to ``end`` (both included), oldest first.

    For a stretch of conversation that is shown to a model: a sampled segment of the persona
    card (R-PERS-001) or the surroundings of a sticker use (R-STK-003).  ``before`` is a hard
    upper bound (exclusive): nothing at or after it is returned, which is how the pre-holdout
    scope keeps the held-out period out (R-TRN-013).  ``system`` notices are included; the
    caller drops them.
    """
    stmt = select(Message).where(Message.create_time_utc >= start, Message.create_time_utc <= end)
    if before is not None:
        stmt = stmt.where(Message.create_time_utc < before)
    return stmt.order_by(Message.create_time_utc, Message.sort_seq, Message.id)
