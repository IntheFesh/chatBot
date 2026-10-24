"""The persisted state of the conversation: ``conversation_state`` (R-ENG-001, R-MEM-001).

One row, written at every transition of the engine's state machine, so that a restart goes on
from where the process stopped: the user messages waiting for an answer, the bubbles of the
current reply that are already out, the instant the next one is due.  The state machine itself is
the engine's (round 09 step 3); this module is the storage it uses and the only code that touches
the table.

Three fields are not state-machine business:

``window_start_at``
    where the window of the recent conversation starts - the time of its first turn.  It moves
    in one step when the window outgrows its limit (:class:`~twin.memory.recent.HistoryWindow`),
    so consecutive prompts share their beginning and the provider's cache keeps hitting
    (R-MEM-001, R-LLM-010).  :class:`twin.engine.history.HistoryLoader` keeps it.
``last_inbound_at`` / ``last_outbound_at``
    the newest message of each side, for "the conversation has been quiet for 30 minutes"
    (R-MEM-007).
``extracted_through``
    the time up to which the conversation has been handed to the fact extractor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from twin.clock import Clock, ensure_aware
from twin.storage.db import Database
from twin.storage.engine_models import MAIN_CONVERSATION, STATES, ConversationState

FIELDS = frozenset(
    {
        "round_id",
        "pending",
        "sent",
        "data",
        "planned_send_at",
        "window_start_at",
        "last_inbound_at",
        "last_outbound_at",
        "extracted_through",
    }
)
DATETIME_FIELDS = frozenset(
    {
        "planned_send_at",
        "window_start_at",
        "last_inbound_at",
        "last_outbound_at",
        "extracted_through",
    }
)


@dataclass(frozen=True)
class ConversationSnapshot:
    """The state of the conversation as it is on disk."""

    state: str
    state_since: datetime
    round_id: str | None = None
    pending: tuple[str, ...] = ()  # ids of the user messages waiting for an answer
    sent: tuple[dict[str, Any], ...] = ()  # bubbles of the current reply that are already out
    data: dict[str, Any] = field(default_factory=dict)  # engine notes (delay drawn, woke up, ...)
    planned_send_at: datetime | None = None
    window_start_at: datetime | None = None
    last_inbound_at: datetime | None = None
    last_outbound_at: datetime | None = None
    extracted_through: datetime | None = None

    @property
    def idle(self) -> bool:
        return self.state == "IDLE"


class ConversationStateStore:
    """Reads and writes the one ``conversation_state`` row."""

    def __init__(self, db: Database, clock: Clock, conversation: str = MAIN_CONVERSATION) -> None:
        self._db = db
        self._clock = clock
        self._id = conversation

    def _snapshot(self, row: ConversationState | None) -> ConversationSnapshot:
        if row is None:
            return ConversationSnapshot("IDLE", self._clock.now_utc())
        return ConversationSnapshot(
            state=row.state,
            state_since=row.state_since,
            round_id=row.round_id,
            pending=tuple(str(item) for item in (row.pending or ())),
            sent=tuple(dict(item) for item in (row.sent or ())),
            data=dict(row.data or {}),
            planned_send_at=row.planned_send_at,
            window_start_at=row.window_start_at,
            last_inbound_at=row.last_inbound_at,
            last_outbound_at=row.last_outbound_at,
            extracted_through=row.extracted_through,
        )

    def load(self) -> ConversationSnapshot:
        """The stored state; a conversation that never had one is idle since now."""
        with self._db.session() as session:
            return self._snapshot(session.get(ConversationState, self._id))

    def update(self, **fields: Any) -> ConversationSnapshot:
        """Change the named fields and keep the rest (``None`` clears a field)."""
        return self._write(None, fields)

    def transition(self, state: str, **fields: Any) -> ConversationSnapshot:
        """Move to ``state`` (stamping the time) and change the named fields in the same write."""
        if state not in STATES:
            raise ValueError(f"unknown conversation state {state!r}")
        return self._write(state, fields)

    def set_window_start(self, start_at: datetime | None) -> None:
        """Remember where the window of the recent conversation starts (R-MEM-001)."""
        self._write(None, {"window_start_at": start_at})

    def _write(self, state: str | None, fields: dict[str, Any]) -> ConversationSnapshot:
        unknown = set(fields) - FIELDS
        if unknown:
            raise ValueError(f"not fields of the conversation state: {', '.join(sorted(unknown))}")
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ConversationState, self._id)
            if row is None:
                row = ConversationState(id=self._id, state="IDLE", state_since=now)
                session.add(row)
            if state is not None:
                row.state = state
                row.state_since = now
            for name, value in fields.items():
                if name in DATETIME_FIELDS:
                    value = ensure_aware(value) if value is not None else None
                    setattr(row, name, value)
                elif name == "pending":
                    row.pending = [str(item) for item in value] if value is not None else None
                elif name == "sent":
                    row.sent = [dict(item) for item in value] if value is not None else None
                elif name == "data":
                    row.data = dict(value) if value is not None else None
                else:
                    setattr(row, name, value)
            session.flush()
            return self._snapshot(row)

    def reset(self) -> ConversationSnapshot:
        """Back to idle with nothing waiting (the window start and the clocks are kept)."""
        return self.transition(
            "IDLE", round_id=None, pending=(), sent=(), data={}, planned_send_at=None
        )
