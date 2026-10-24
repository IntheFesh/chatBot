"""Plain, decrypted copies of the memory rows (round 07).

The ORM rows (:mod:`twin.storage.memory_models`) decrypt their text on every attribute read and
are tied to a session.  Everything above the store - the in-memory corpus, the as-of view, the
assembler, the replay - works on these frozen records instead, so a record can be kept, filtered
and projected (see :mod:`twin.memory.visible`) without touching the database again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from twin.storage.memory_models import (
    BOT_CONVERSATION_SOURCES,
    SOURCE_PRIORITY,
    DailySummary,
    Fact,
    Followup,
    LifelineEvent,
)


@dataclass(frozen=True)
class FactRecord:
    id: str
    rev: int  # how often the row was changed: the corpus reloads a row whose revision moved
    number: int
    subject: str
    category: str
    text: str
    source: str
    status: str
    confidence: float
    importance: int
    known_at: datetime
    valid_from: datetime | None
    valid_to: datetime | None
    event_date: date | None
    recurrence: str
    superseded_by: str | None
    superseded_at: datetime | None
    rejected_by: str | None
    evidence: dict[str, Any] | None
    embedding_id: str | None
    embed_version: str | None
    created_at: datetime
    updated_at: datetime

    @property
    def priority(self) -> int:
        return SOURCE_PRIORITY[self.source]

    @property
    def bot_conversation(self) -> bool:
        """True for what exists only because the bot conversation exists (R-MEM-010)."""
        return self.source in BOT_CONVERSATION_SOURCES

    @property
    def current(self) -> bool:
        """Neither vetoed nor replaced: what the bot holds to be true now."""
        return self.status == "active" and self.superseded_by is None

    @classmethod
    def from_row(cls, row: Fact) -> FactRecord:
        return cls(
            id=row.id,
            rev=row.rev,
            number=row.number,
            subject=row.subject,
            category=row.category,
            text=row.text,
            source=row.source,
            status=row.status,
            confidence=row.confidence,
            importance=row.importance,
            known_at=row.known_at,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            event_date=row.event_date,
            recurrence=row.recurrence,
            superseded_by=row.superseded_by,
            superseded_at=row.superseded_at,
            rejected_by=row.rejected_by,
            evidence=row.evidence,
            embedding_id=row.embedding_id,
            embed_version=row.embed_version,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


@dataclass(frozen=True)
class SummaryRecord:
    id: str
    rev: int
    scope: str
    local_date: date
    timezone: str
    utc_start: datetime
    utc_end: datetime
    text: str
    version: int
    is_current: bool
    embedding_id: str | None
    embed_version: str | None
    input_hash: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: DailySummary) -> SummaryRecord:
        return cls(
            id=row.id,
            rev=row.rev,
            scope=row.scope,
            local_date=row.local_date,
            timezone=row.timezone,
            utc_start=row.utc_start,
            utc_end=row.utc_end,
            text=row.text,
            version=row.version,
            is_current=row.is_current,
            embedding_id=row.embedding_id,
            embed_version=row.embed_version,
            input_hash=row.input_hash,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


@dataclass(frozen=True)
class LifelineRecord:
    id: str
    rev: int
    local_date: date
    timezone: str
    start_local: str | None
    end_local: str | None
    activity: str
    place: str | None
    mood: str | None
    detail: str | None
    source: str
    status: str
    consistency_checked_at: datetime | None
    invalidated_at: datetime | None
    invalidated_by: str | None
    fact_id: str | None
    created_at: datetime
    updated_at: datetime
    shared_at: datetime | None = None
    shared_reply_id: str | None = None

    @property
    def active(self) -> bool:
        return self.status == "active"

    @property
    def shared(self) -> bool:
        """Whether a proactive message already told the user about this (round 10)."""
        return self.shared_at is not None

    def line(self) -> str:
        """One line for the prompt: the time span, what she does, where, in which mood."""
        span = ""
        if self.start_local and self.end_local:
            span = f"{self.start_local}-{self.end_local} "
        elif self.start_local:
            span = f"{self.start_local} "
        extras = "，".join(part for part in (self.place, self.mood) if part)
        text = f"{span}{self.activity}"
        if extras:
            text += f"（{extras}）"
        return text

    @classmethod
    def from_row(cls, row: LifelineEvent) -> LifelineRecord:
        return cls(
            id=row.id,
            rev=row.rev,
            local_date=row.local_date,
            timezone=row.timezone,
            start_local=row.start_local,
            end_local=row.end_local,
            activity=row.activity,
            place=row.place,
            mood=row.mood,
            detail=row.detail,
            source=row.source,
            status=row.status,
            consistency_checked_at=row.consistency_checked_at,
            invalidated_at=row.invalidated_at,
            invalidated_by=row.invalidated_by,
            fact_id=row.fact_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            shared_at=row.shared_at,
            shared_reply_id=row.shared_reply_id,
        )


@dataclass(frozen=True)
class FollowupRecord:
    id: str
    text: str
    due_at: datetime
    window_minutes: int
    source_turn_id: str | None
    status: str
    created_at: datetime
    closed_at: datetime | None
    close_reason: str | None
    origin: str
    fact_id: str | None
    evidence: dict[str, Any] | None = field(default=None, compare=False)
    rev: int = field(default=0, compare=False)

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def window_end(self) -> datetime:
        return self.due_at + timedelta(minutes=self.window_minutes)

    @classmethod
    def from_row(cls, row: Followup) -> FollowupRecord:
        return cls(
            id=row.id,
            text=row.text,
            due_at=row.due_at_utc,
            window_minutes=row.window_minutes,
            source_turn_id=row.source_turn_id,
            status=row.status,
            created_at=row.created_at,
            closed_at=row.closed_at,
            close_reason=row.close_reason,
            origin=row.origin,
            fact_id=row.fact_id,
            evidence=row.evidence,
            rev=row.rev,
        )
