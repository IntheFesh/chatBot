"""The bot's conversation on disk: ``bot_turns`` (R-ENG-011, R-STO-007, R-MEM-001).

:class:`BotTurnStore` is the write side and the lookups of the engine; :class:`BotTurnMessages`
implements :class:`~twin.memory.recent.BotTurnReader` on the table, so the memory (the bot's
daily summary, the recent-conversation window) reads the conversation through the interface the
memory defined and never sees the table.  Importing this module registers that reader
(:func:`~twin.memory.recent.register_bot_turn_reader`); the module is listed among the job
handler modules so every process that runs memory jobs has it.

What counts as conversation.  Commands (``is_command``), the bot's decision to say nothing
(``no_reply``) and replies the user threw away (``rejected_at``) are rows of the table - the
audit trail - but not messages of the conversation: the readers below leave them out, so a command
or a rejected reply can never reach the prompt, the memory or the sticker share.

Nothing here is read by the style samples, the retrieval library or the training set: those start
from :mod:`twin.ingest.corpus` and ``messages`` (R-STO-007; tests scan the packages for this table).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, func, select, update
from sqlalchemy.orm import Session

from twin.clock import Clock, ensure_aware
from twin.memory.recent import BotMessage, register_bot_turn_reader
from twin.storage.db import Database
from twin.storage.engine_models import BACKENDS, BotTurn

if TYPE_CHECKING:
    from twin.services import Services

CONVERSATION_KINDS = ("text", "image", "voice", "video", "file", "sticker", "unknown")
BUBBLE_KINDS = ("text", "sticker")


@dataclass(frozen=True)
class ReplyMeta:
    """What belongs to a whole reply; stored on its first bubble (R-ENG-011)."""

    backend: str
    thinking: bool | None = None
    plan: dict[str, Any] | None = None
    cost_usd: float | None = None
    timings_ms: dict[str, int] | None = None
    actions: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}")


@dataclass(frozen=True)
class TurnRecord:
    """One row of the table, decrypted."""

    id: str
    at: datetime
    direction: str
    kind: str
    text: str
    media: dict[str, Any] | None
    sticker_md5: str | None
    is_command: bool
    external_id: str | None
    reply_id: str | None
    bubble_index: int | None
    backend: str | None
    thinking: bool | None
    plan: dict[str, Any] | None
    cost_usd: float | None
    timings_ms: dict[str, int] | None
    actions: tuple[dict[str, Any], ...] = field(default=())
    rejected_at: datetime | None = None

    @property
    def inbound(self) -> bool:
        return self.direction == "in"

    @property
    def counts_as_conversation(self) -> bool:
        return not (self.is_command or self.kind == "no_reply" or self.rejected_at is not None)


@dataclass(frozen=True)
class AddedTurn:
    """The row of an inbound message, and whether this call created it (a replay does not)."""

    record: TurnRecord
    created: bool


@dataclass(frozen=True)
class OutboundBubble:
    """One bubble of a reply as it is stored."""

    text: str
    at: datetime
    kind: str = "text"
    sticker_md5: str | None = None
    external_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in BUBBLE_KINDS:
            raise ValueError(f"a bubble is text or a sticker, not {self.kind!r}")
        if self.kind == "sticker" and not self.sticker_md5:
            raise ValueError("a sticker bubble needs the MD5 of its sticker")


def _record(row: BotTurn) -> TurnRecord:
    return TurnRecord(
        id=row.id,
        at=row.at,
        direction=row.direction,
        kind=row.kind,
        text=row.text,
        media=row.media,
        sticker_md5=row.sticker_md5,
        is_command=row.is_command,
        external_id=row.external_id,
        reply_id=row.reply_id,
        bubble_index=row.bubble_index,
        backend=row.backend,
        thinking=row.thinking,
        plan=row.plan,
        cost_usd=row.cost_usd,
        timings_ms=row.timings,
        actions=tuple(row.actions or ()),
        rejected_at=row.rejected_at,
    )


def _apply_meta(row: BotTurn, meta: ReplyMeta) -> None:
    row.backend = meta.backend
    row.thinking = meta.thinking
    row.plan = meta.plan
    row.cost_usd = meta.cost_usd
    row.timings = dict(meta.timings_ms) if meta.timings_ms is not None else None
    row.actions = [dict(action) for action in meta.actions] if meta.actions else None


def conversation_rows(*, outbound_bubbles_only: bool = False) -> Select[BotTurn]:
    """The rows that are messages of the conversation (no commands, no silence, no rejects)."""
    stmt = select(BotTurn).where(
        BotTurn.is_command.is_(False),
        BotTurn.rejected_at.is_(None),
        BotTurn.kind.in_(CONVERSATION_KINDS),
    )
    if outbound_bubbles_only:
        stmt = stmt.where(BotTurn.direction == "out", BotTurn.kind.in_(BUBBLE_KINDS))
    return stmt


class BotTurnStore:
    """Writes and looks up the bot's conversation."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    # --------------------------------------------------------------------- inbound

    def add_inbound(
        self,
        *,
        at: datetime,
        kind: str,
        text: str,
        external_id: str | None = None,
        media: dict[str, Any] | None = None,
        is_command: bool = False,
    ) -> AddedTurn:
        """Store a message of the user.  The same channel message id is stored once.

        The check and the insert share one ``BEGIN IMMEDIATE`` transaction, so two writers cannot
        both store it.

        ``text`` is the message as it stands in the conversation (the user's words; for a
        picture, voice message or sticker its stable description), so it reads the same in
        every later prompt (R-LLM-010).
        """
        if kind not in CONVERSATION_KINDS:
            raise ValueError(f"an inbound message cannot be of kind {kind!r}")
        moment = ensure_aware(at)
        with self._db.transaction(bump_state=False) as session:
            if external_id is not None:
                found = self._by_external(session, "in", external_id)
                if found is not None:
                    return AddedTurn(_record(found), False)
            row = BotTurn(
                at=moment,
                direction="in",
                kind=kind,
                text=text,
                media=media,
                is_command=is_command,
                external_id=external_id,
            )
            session.add(row)
            session.flush()
            return AddedTurn(_record(row), True)

    @staticmethod
    def _by_external(session: Session, direction: str, external_id: str) -> BotTurn | None:
        return session.scalars(
            select(BotTurn).where(
                BotTurn.direction == direction, BotTurn.external_id == external_id
            )
        ).first()

    # -------------------------------------------------------------------- outbound

    def add_bubble(
        self,
        bubble: OutboundBubble,
        *,
        reply_id: str | None = None,
        meta: ReplyMeta | None = None,
        is_command: bool = False,
    ) -> TurnRecord:
        """Store one bubble.  Without ``reply_id`` it opens a new reply (and carries ``meta``)."""
        with self._db.transaction(bump_state=False) as session:
            return _record(
                self._add_bubble(session, bubble, reply_id=reply_id, meta=meta, command=is_command)
            )

    def add_reply(
        self,
        bubbles: Sequence[OutboundBubble],
        meta: ReplyMeta,
        *,
        is_command: bool = False,
    ) -> list[TurnRecord]:
        """Store all bubbles of one reply in one transaction (the first carries ``meta``)."""
        if not bubbles:
            raise ValueError("a reply has at least one bubble; use add_no_reply for silence")
        records: list[TurnRecord] = []
        with self._db.transaction(bump_state=False) as session:
            reply_id: str | None = None
            for index, bubble in enumerate(bubbles):
                row = self._add_bubble(
                    session,
                    bubble,
                    reply_id=reply_id,
                    meta=meta if index == 0 else None,
                    command=is_command,
                )
                reply_id = row.reply_id
                records.append(_record(row))
        return records

    def add_no_reply(self, at: datetime, meta: ReplyMeta) -> TurnRecord:
        """Note that the bot chose to say nothing (R-ENG-007): one row, no text."""
        with self._db.transaction(bump_state=False) as session:
            row = BotTurn(
                at=ensure_aware(at),
                direction="out",
                kind="no_reply",
                text="",
                is_command=False,
                bubble_index=0,
            )
            row.reply_id = row.id
            _apply_meta(row, meta)
            session.add(row)
            session.flush()
            return _record(row)

    def _add_bubble(
        self,
        session: Session,
        bubble: OutboundBubble,
        *,
        reply_id: str | None,
        meta: ReplyMeta | None,
        command: bool,
    ) -> BotTurn:
        row = BotTurn(
            at=ensure_aware(bubble.at),
            direction="out",
            kind=bubble.kind,
            text=bubble.text,
            sticker_md5=bubble.sticker_md5,
            is_command=command,
            external_id=bubble.external_id,
        )
        if reply_id is None:
            row.reply_id = row.id
            row.bubble_index = 0
        else:
            row.reply_id = reply_id
            last = session.scalar(
                select(func.max(BotTurn.bubble_index)).where(BotTurn.reply_id == reply_id)
            )
            row.bubble_index = 0 if last is None else int(last) + 1
        if meta is not None:
            _apply_meta(row, meta)
        session.add(row)
        session.flush()
        return row

    def update_meta(self, reply_id: str, meta: ReplyMeta) -> bool:
        """Set the reply-level numbers on the first bubble of a reply (after the fact)."""
        with self._db.transaction(bump_state=False) as session:
            row = session.scalars(
                select(BotTurn).where(BotTurn.reply_id == reply_id).order_by(BotTurn.bubble_index)
            ).first()
            if row is None:
                return False
            _apply_meta(row, meta)
            return True

    def reject_reply(self, reply_id: str, at: datetime | None = None) -> int:
        """Mark a reply as thrown away (``/重来``); returns how many rows it touched."""
        moment = ensure_aware(at) if at is not None else self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            result = session.execute(
                update(BotTurn)
                .where(
                    BotTurn.reply_id == reply_id,
                    BotTurn.direction == "out",
                    BotTurn.rejected_at.is_(None),
                )
                .values(rejected_at=moment)
            )
            return int(getattr(result, "rowcount", 0) or 0)

    # --------------------------------------------------------------------- lookups

    def get(self, turn_id: str) -> TurnRecord | None:
        with self._db.session() as session:
            row = session.get(BotTurn, turn_id)
            return _record(row) if row is not None else None

    def by_external(self, direction: str, external_id: str) -> TurnRecord | None:
        with self._db.session() as session:
            row = self._by_external(session, direction, external_id)
            return _record(row) if row is not None else None

    def reply(self, reply_id: str) -> list[TurnRecord]:
        """The rows of one reply, in order."""
        with self._db.session() as session:
            rows = session.scalars(
                select(BotTurn).where(BotTurn.reply_id == reply_id).order_by(BotTurn.bubble_index)
            )
            return [_record(row) for row in rows]

    def latest_reply(self, *, include_rejected: bool = False) -> list[TurnRecord]:
        """The newest reply of the conversation (commands excluded), empty if there is none."""
        with self._db.session() as session:
            stmt = select(BotTurn).where(
                BotTurn.direction == "out",
                BotTurn.is_command.is_(False),
                BotTurn.reply_id.is_not(None),
            )
            if not include_rejected:
                stmt = stmt.where(BotTurn.rejected_at.is_(None))
            newest = session.scalars(stmt.order_by(BotTurn.at.desc(), BotTurn.id.desc())).first()
            if newest is None or newest.reply_id is None:
                return []
            rows = session.scalars(
                select(BotTurn)
                .where(BotTurn.reply_id == newest.reply_id)
                .order_by(BotTurn.bubble_index)
            )
            return [_record(row) for row in rows]

    def recent_stickers(self, limit: int) -> list[str | None]:
        """For each of the last ``limit`` bubbles: the sticker MD5, or ``None`` for text.

        Oldest first - what the sticker selector needs to avoid repeating itself (R-STK-004).
        """
        stmt = (
            conversation_rows(outbound_bubbles_only=True)
            .order_by(BotTurn.at.desc(), BotTurn.id.desc())
            .limit(limit)
        )
        with self._db.session() as session:
            rows = list(session.scalars(stmt))
            return [row.sticker_md5 if row.kind == "sticker" else None for row in reversed(rows)]

    def last_message_at(self, direction: str | None = None) -> datetime | None:
        """When the newest message of the conversation (or of one side) was written."""
        stmt = select(func.max(BotTurn.at)).where(
            BotTurn.is_command.is_(False),
            BotTurn.rejected_at.is_(None),
            BotTurn.kind.in_(CONVERSATION_KINDS),
        )
        if direction is not None:
            stmt = stmt.where(BotTurn.direction == direction)
        with self._db.session() as session:
            found = session.scalar(stmt)
        return ensure_aware(found) if found is not None else None

    def count(self, *, direction: str | None = None) -> int:
        stmt = select(func.count()).select_from(BotTurn)
        if direction is not None:
            stmt = stmt.where(BotTurn.direction == direction)
        with self._db.session() as session:
            return int(session.scalar(stmt) or 0)


class BotTurnMessages:
    """:class:`~twin.memory.recent.BotTurnReader` over ``bot_turns`` (R-MEM-001, R-MEM-002)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @staticmethod
    def _message(row: BotTurn) -> BotMessage:
        return BotMessage(row.id, "user" if row.direction == "in" else "bot", row.text, row.at)

    def messages_between(self, start: datetime, end: datetime) -> Sequence[BotMessage]:
        stmt = (
            conversation_rows()
            .where(BotTurn.at >= ensure_aware(start), BotTurn.at < ensure_aware(end))
            .order_by(BotTurn.at, BotTurn.id)
        )
        with self._db.session() as session:
            return [self._message(row) for row in session.scalars(stmt)]

    def messages_since(
        self, since: datetime | None, limit: int | None = None
    ) -> Sequence[BotMessage]:
        with self._db.session() as session:
            if since is not None:
                stmt = (
                    conversation_rows()
                    .where(BotTurn.at >= ensure_aware(since))
                    .order_by(BotTurn.at, BotTurn.id)
                )
                return [self._message(row) for row in session.scalars(stmt)]
            newest = conversation_rows().order_by(BotTurn.at.desc(), BotTurn.id.desc())
            if limit:
                newest = newest.limit(limit)
            return [self._message(row) for row in reversed(list(session.scalars(newest)))]


def _reader_for(services: Services) -> BotTurnMessages:
    return BotTurnMessages(services.db)


def install_bot_turn_reader() -> None:
    """Make ``bot_turns`` the conversation the memory reads (done once, when this module loads)."""
    register_bot_turn_reader(_reader_for)


install_bot_turn_reader()
