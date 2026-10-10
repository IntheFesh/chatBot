"""Follow-ups: things worth asking about later (R-MEM-006).

The extractor finds them in what the user says (an exam, an interview, a doctor's visit); this
module keeps them: ``due_at`` is when the thing happens, ``window_minutes`` how long afterwards
asking is still natural.  A follow-up is **open** until it is closed - by the user mentioning it
(``done``), because it was cancelled, because the proactive message asked about it (round 10), or
because its window passed without anyone raising it (``expired``).  The moment of closing is
kept, so the state at any past moment can be read (the as-of view uses ``closed_at``).

Round 10 builds the proactive "follow-up" messages on :meth:`FollowupStore.due`; round 11 and the
engine close follow-ups the user brings up with :meth:`FollowupStore.close`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from twin.memory.keywords import tokens_of
from twin.memory.memory import Memory
from twin.memory.records import FollowupRecord
from twin.memory.store import NewFollowup
from twin.ops.logging import get_logger

log = get_logger("twin.memory.followups")

SAME_TIME_TOLERANCE = timedelta(minutes=10)
SAME_TEXT_OVERLAP = 0.5
DEFAULT_WINDOW_MINUTES = 240


def same_followup(first_text: str, first_due: datetime, other: FollowupRecord) -> bool:
    """True if ``other`` is the same commitment: about the same time and mostly the same words."""
    if abs(other.due_at - first_due) > SAME_TIME_TOLERANCE:
        return False
    mine, theirs = tokens_of(first_text), tokens_of(other.text)
    if not mine or not theirs:
        return first_text.strip() == other.text.strip()
    return len(mine & theirs) / len(mine | theirs) >= SAME_TEXT_OVERLAP


class FollowupStore:
    """Writing, reading and closing follow-ups (see the module description)."""

    def __init__(self, memory: Memory) -> None:
        self._memory = memory
        self._store = memory.store

    # ------------------------------------------------------------------ writing

    def add(
        self,
        text: str,
        due_at: datetime,
        *,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        origin: str = "bot_session",
        source_turn_id: str | None = None,
        created_at: datetime | None = None,
        fact_id: str | None = None,
        evidence: dict[str, object] | None = None,
    ) -> tuple[FollowupRecord, bool]:
        """Store a follow-up; returns ``(record, created)``.

        A follow-up that repeats an open one (about the same time, mostly the same words) is not
        stored twice: the open one is returned with ``created`` false.
        """
        self._memory.refresh()
        for known in self._memory.corpus.followups.values():
            if known.is_open and same_followup(text, due_at, known):
                return known, False
        now = self._memory.services.clock.now_utc()
        record = self._store.add_followup(
            NewFollowup(
                text=text,
                due_at=due_at,
                window_minutes=window_minutes,
                created_at=created_at or now,
                origin=origin,
                source_turn_id=source_turn_id,
                evidence=evidence,
                fact_id=fact_id,
            )
        )
        if origin != "real_record":
            self._store.mark_bot_online(record.created_at)
        self._memory.refresh()
        return record, True

    def close(
        self,
        followup_id: str,
        *,
        status: str = "done",
        reason: str | None = None,
        at: datetime | None = None,
    ) -> FollowupRecord | None:
        """Close an open follow-up as of ``at`` (now by default); ``None`` if it was not open."""
        moment = at or self._memory.services.clock.now_utc()
        closed = self._store.close_followup(followup_id, status=status, at=moment, reason=reason)
        self._memory.refresh()
        return closed

    def expire_overdue(self, now: datetime | None = None) -> int:
        """Close the follow-ups whose window has passed unraised; returns how many."""
        moment = now or self._memory.services.clock.now_utc()
        expired = 0
        for followup in self.open():
            if followup.window_end < moment:
                done = self._store.close_followup(
                    followup.id, status="expired", at=followup.window_end, reason="window_passed"
                )
                expired += int(done is not None)
        if expired:
            self._memory.refresh()
        return expired

    # ------------------------------------------------------------------ reading

    def get(self, followup_id: str) -> FollowupRecord | None:
        self._memory.refresh()
        return self._memory.corpus.followups.get(followup_id)

    def open(self) -> list[FollowupRecord]:
        """Every open follow-up, the earliest due first."""
        self._memory.refresh()
        found = [f for f in self._memory.corpus.followups.values() if f.is_open]
        found.sort(key=lambda f: (f.due_at, f.created_at))
        return found

    def due(self, now: datetime, *, lookahead: timedelta = timedelta(0)) -> list[FollowupRecord]:
        """Open follow-ups whose time has come (or comes within ``lookahead``) and whose window
        has not passed: the ones a proactive message may ask about."""
        return [f for f in self.open() if f.due_at <= now + lookahead and now <= f.window_end]
