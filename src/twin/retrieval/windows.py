"""Example windows: each of her reply blocks with the conversation just before it (R-RET-001).

Definitions (the units are the ones of the style profile, :mod:`twin.profile.units`; nothing is
defined twice):

* a **reply block** is one of her bursts (``profile.burst_gap_s``): consecutive messages of hers
  that follow each other within the burst gap, whatever their kind (``system`` notices are not
  messages of either side and are skipped, as everywhere);
* its **context** is the merged turns - bursts of either side - that precede it in the same
  conversation segment (``profile.segment_gap_min``), at most ``retrieval.context_turns`` (6) of
  them, oldest first.  A block that opens a segment has no context.

Every one of her bursts is a window, also one that only holds a photo or a call (its
``reply_reproducible`` is 0; the query ranks such windows lower).  The hold-out set of
:func:`~twin.profile.holdout.holdout_cutoff` counts only the bursts the bot could have written
itself; a window is held out when its reply time is at or after that cutoff, so the held-out
windows are exactly the windows of the evaluation and the test split of the training set.

**Her real replies only (R-RET-004, R-STO-007).**  :func:`build_window` accepts rows of the
``messages`` table and nothing else: a row of any other type (the bot's own turns, round 09) is
a :class:`TypeError`, and a reply member the user sent (``is_sent``) is refused.  The ids stored
in ``example_windows`` are looked up in ``messages`` again whenever a window is rendered.

Streaming.  :class:`WindowAssembler` walks the messages once in time order and keeps only the
last few turns, so a conversation of millions of messages needs a few kilobytes.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from twin.ingest.corpus import conversation_skeleton, messages_by_ids
from twin.ingest.events import is_reproducible
from twin.ingest.times import SourceTime
from twin.profile.holdout import holdout_cutoff
from twin.profile.localtime import LocalClock, LocalStamp
from twin.profile.overrides import RoutineOverrides
from twin.profile.units import BurstSegmenter
from twin.retrieval.records import NotHerMessageError, require_message
from twin.schedule.daytype import DayTypeCalendar
from twin.storage.chat_models import Message
from twin.storage.retrieval_models import WINDOW_ID_LENGTH, ExampleWindow

if TYPE_CHECKING:
    from twin.services import Services

SYSTEM_KIND = "system"
SYNC_CHUNK = 2000


def window_id_of(first_reply_id: str) -> str:
    """The stable id of the window whose reply block starts with this message."""
    digest = hashlib.sha1(first_reply_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return ("w" + digest)[:WINDOW_ID_LENGTH]


def signature_of(
    reply_ids: Sequence[str], context_ids: Sequence[Sequence[str]], local_slot: int, day_type: str
) -> str:
    """Hash of everything a window's vector depends on besides the model."""
    payload = json.dumps([list(reply_ids), [list(t) for t in context_ids], local_slot, day_type])
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass(frozen=True)
class WindowDraft:
    """A window as the assembler found it (before the local time is known)."""

    id: str
    conversation_id: str
    reply_ids: tuple[str, ...]
    context_ids: tuple[tuple[str, ...], ...]
    reply_at: datetime
    reproducible: int


@dataclass(frozen=True, slots=True)
class MessageRef:
    """The columns of a ``messages`` row that lay a window out (copied while the row is live)."""

    id: str
    conversation_id: str
    at: datetime
    is_sent: bool
    kind: str

    @classmethod
    def from_row(cls, row: object) -> MessageRef:
        message = require_message(row)
        return cls(
            message.id,
            message.conversation_id,
            message.create_time_utc.astimezone(UTC),
            message.is_sent,
            message.kind,
        )


def _window_of(reply: Sequence[MessageRef], context: Sequence[Sequence[MessageRef]]) -> WindowDraft:
    if not reply:
        raise ValueError("a window needs at least one reply message")
    if any(ref.is_sent for ref in reply):
        raise NotHerMessageError("a message sent by the user cannot be the reply of a window")
    first = reply[0]
    return WindowDraft(
        id=window_id_of(first.id),
        conversation_id=first.conversation_id,
        reply_ids=tuple(ref.id for ref in reply),
        context_ids=tuple(tuple(ref.id for ref in turn) for turn in context if turn),
        reply_at=first.at,
        reproducible=sum(1 for ref in reply if is_reproducible(ref.kind)),
    )


def build_window(reply: Sequence[Message], context: Sequence[Sequence[Message]]) -> WindowDraft:
    """The window of ``reply`` (her messages) with the turns ``context`` before it.

    Both arguments are rows of the ``messages`` table; ``reply`` must be hers.  This is the one
    function that makes windows (R-RET-004): an object of any other type - the bot's own turns -
    is a :class:`TypeError`, a reply member sent by the user is a :class:`NotHerMessageError`.
    """
    return _window_of(
        [MessageRef.from_row(message) for message in reply],
        [[MessageRef.from_row(message) for message in turn] for turn in context],
    )


def window_from_ids(
    session: Session, reply_ids: Sequence[str], context_ids: Sequence[Sequence[str]]
) -> WindowDraft:
    """:func:`build_window` for message ids: they are looked up in ``messages``.

    An id that is not a row of ``messages`` (a bot turn, an unknown id) raises
    :class:`LookupError`; a user message among ``reply_ids`` raises :class:`NotHerMessageError`.
    """
    wanted = {*reply_ids, *(i for turn in context_ids for i in turn)}
    found = {row.id: row for row in session.scalars(messages_by_ids(sorted(wanted)))}
    missing = wanted - set(found)
    if missing:
        raise LookupError(f"{len(missing)} message id(s) are not rows of the messages table")
    return build_window(
        [found[i] for i in reply_ids], [[found[i] for i in turn] for turn in context_ids]
    )


@dataclass
class _Turn:
    her: bool
    messages: list[MessageRef]


class WindowAssembler:
    """Walks the conversation in time order and emits a window for each of her bursts."""

    def __init__(self, burst_gap_s: float, segment_gap_s: float, context_turns: int) -> None:
        self._segmenter = BurstSegmenter(burst_gap_s, segment_gap_s)
        self._context: deque[_Turn] = deque(maxlen=context_turns)
        self._open: _Turn | None = None

    def _close(self) -> WindowDraft | None:
        turn, self._open = self._open, None
        if turn is None:
            return None
        draft = None
        if turn.her:
            draft = _window_of(turn.messages, [t.messages for t in self._context])
        self._context.append(turn)
        return draft

    def add(self, message: Message) -> WindowDraft | None:
        """Feed the next message (oldest first); returns the window this message completed."""
        row = MessageRef.from_row(message)
        if row.kind == SYSTEM_KIND:
            return None
        her = not row.is_sent
        boundary = self._segmenter.feed(row.at.timestamp(), her)
        if not boundary.new_block and self._open is not None:
            self._open.messages.append(row)
            return None
        draft = self._close()
        if boundary.new_segment:
            self._context.clear()
        self._open = _Turn(her, [row])
        return draft

    def finish(self) -> WindowDraft | None:
        """The window of the burst still open at the end of the data."""
        return self._close()


def assemble(
    messages: Iterable[Message], burst_gap_s: float, segment_gap_s: float, context_turns: int
) -> Iterable[WindowDraft]:
    """All windows of a stream of messages in time order."""
    assembler = WindowAssembler(burst_gap_s, segment_gap_s, context_turns)
    for message in messages:
        draft = assembler.add(message)
        if draft is not None:
            yield draft
    last = assembler.finish()
    if last is not None:
        yield last


@dataclass(frozen=True)
class WindowRecord:
    """A stored window copied out of its database row."""

    id: str
    conversation_id: str
    reply_block_ids: tuple[str, ...]
    context_block_ids: tuple[tuple[str, ...], ...]
    reply_at_utc: datetime
    local_slot: int
    day_type: str
    reply_reproducible: int
    holdout: bool
    context_turns: int = 0
    signature: str = ""
    embed_version: str | None = None

    @classmethod
    def from_row(cls, row: ExampleWindow) -> WindowRecord:
        return cls(
            id=row.id,
            conversation_id=row.conversation_id,
            reply_block_ids=tuple(row.reply_block_ids),
            context_block_ids=tuple(tuple(turn) for turn in row.context_block_ids),
            reply_at_utc=row.reply_at_utc,
            local_slot=row.local_slot,
            day_type=row.day_type,
            reply_reproducible=row.reply_reproducible,
            holdout=row.holdout,
            context_turns=row.context_turns,
            signature=row.signature,
            embed_version=row.embed_version,
        )


# ------------------------------------------------------------------ local time


class LocalPlace:
    """Slot and day type of a moment on the clock of the place she was in (R-ACT-001)."""

    def __init__(self, clock: LocalClock, calendar: DayTypeCalendar) -> None:
        self._clock = clock
        self._calendar = calendar

    @classmethod
    def create(cls, services: Services) -> LocalPlace:
        settings = services.settings
        holidays = RoutineOverrides(services.db, services.clock).holiday_ranges()
        return cls(
            LocalClock(SourceTime.from_config(settings.time)),
            DayTypeCalendar(
                zone_countries=settings.safety.timezone_country, holiday_ranges=holidays
            ),
        )

    def of(self, moment: datetime) -> tuple[int, str]:
        stamp = self._clock.stamp(moment)
        return stamp.slot, str(self._calendar.day_type(stamp.day, stamp.zone))

    def stamp(self, moment: datetime) -> LocalStamp:
        """Where ``moment`` falls on the clock of the place she was in (day, minute, zone)."""
        return self._clock.stamp(moment)

    def day_type(self, day: date, zone: str) -> str:
        """``workday``, ``weekend`` or ``holiday`` for a local date in a zone (R-ACT-002)."""
        return str(self._calendar.day_type(day, zone))


# ------------------------------------------------------------------- syncing


@dataclass(frozen=True)
class SyncReport:
    """What :func:`sync_windows` changed in ``example_windows``."""

    total: int
    added: int
    changed: int
    removed: int
    holdout: int
    cutoff: datetime
    removed_ids: tuple[str, ...] = ()
    newly_held_ids: tuple[str, ...] = ()  # windows that just entered the hold-out

    @property
    def unchanged(self) -> int:
        return self.total - self.added - self.changed


@dataclass(frozen=True)
class _Known:
    signature: str
    holdout: bool
    embed_version: str | None


def _row_values(
    draft: WindowDraft, place: tuple[int, str], holdout: bool, now: datetime
) -> dict[str, object]:
    slot, day_type = place
    return {
        "conversation_id": draft.conversation_id,
        "reply_block_ids": list(draft.reply_ids),
        "context_block_ids": [list(turn) for turn in draft.context_ids],
        "context_turns": len(draft.context_ids),
        "reply_reproducible": draft.reproducible,
        "reply_at_utc": draft.reply_at,
        "local_slot": slot,
        "day_type": day_type,
        "holdout": holdout,
        "signature": signature_of(draft.reply_ids, draft.context_ids, slot, day_type),
        "updated_at": now,
    }


def _flush(
    services: Services, inserts: list[dict[str, object]], updates: list[dict[str, object]]
) -> None:
    if not inserts and not updates:
        return
    with services.db.transaction(bump_state=False) as session:
        if inserts:
            session.execute(insert(ExampleWindow), inserts)
        if updates:
            session.execute(update(ExampleWindow), updates)


def sync_windows(services: Services, *, cutoff: datetime | None = None) -> SyncReport:
    """Bring ``example_windows`` in line with the stored messages (R-RET-001, R-RET-006).

    One pass over the conversation.  New windows are added; a window whose messages (or local
    time) changed gets its vector invalidated (``embed_version`` NULL) so only it is encoded
    again; windows that no longer exist are removed; the ``holdout`` flag follows
    :func:`~twin.profile.holdout.holdout_cutoff`.  Vectors are not touched here.
    """
    settings = services.settings
    split = cutoff if cutoff is not None else holdout_cutoff(services)
    place = LocalPlace.create(services)
    now = services.clock.now_utc()
    with services.db.session() as session:
        known = {
            row.id: _Known(row.signature, row.holdout, row.embed_version)
            for row in session.execute(
                select(
                    ExampleWindow.id,
                    ExampleWindow.signature,
                    ExampleWindow.holdout,
                    ExampleWindow.embed_version,
                )
            )
        }
    seen: set[str] = set()
    inserts: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []
    added = changed = held = 0
    newly_held: list[str] = []

    stream = assemble(
        _skeleton_rows(services),
        settings.profile.burst_gap_s,
        settings.profile.segment_gap_min * 60.0,
        settings.retrieval.context_turns,
    )
    for draft in stream:
        seen.add(draft.id)
        holdout = draft.reply_at >= split
        held += int(holdout)
        values = _row_values(draft, place.of(draft.reply_at), holdout, now)
        old = known.get(draft.id)
        if old is None:
            inserts.append({"id": draft.id, "embed_version": None, "created_at": now, **values})
            added += 1
        elif old.signature != values["signature"] or old.holdout != holdout:
            if old.signature != values["signature"]:
                changed += 1
            if holdout and not old.holdout:
                newly_held.append(draft.id)
            # a changed or newly held-out window loses its vector; a window that left the
            # hold-out has none and is encoded by the next run
            updates.append({"id": draft.id, "embed_version": None, **values})
        if len(inserts) + len(updates) >= SYNC_CHUNK:
            _flush(services, inserts, updates)
            inserts, updates = [], []
    _flush(services, inserts, updates)

    gone = sorted(set(known) - seen)
    for start in range(0, len(gone), SYNC_CHUNK):
        with services.db.transaction(bump_state=False) as session:
            session.execute(
                delete(ExampleWindow).where(ExampleWindow.id.in_(gone[start : start + SYNC_CHUNK]))
            )
    return SyncReport(
        total=len(seen),
        added=added,
        changed=changed,
        removed=len(gone),
        holdout=held,
        cutoff=split,
        removed_ids=tuple(gone),
        newly_held_ids=tuple(newly_held),
    )


def _skeleton_rows(services: Services) -> Iterable[Message]:
    """The conversation's messages in time order (ids, times, senders, kinds)."""
    with services.db.session() as session:
        yield from session.scalars(conversation_skeleton().execution_options(yield_per=5000))


def apply_holdout(services: Services, cutoff: datetime) -> tuple[tuple[str, ...], int]:
    """Move the ``holdout`` flags to a new cutoff without reading any message.

    Returns ``(ids of the windows that are now held out, number of windows released)``.  A
    newly held-out window loses its vector marker (the caller removes it from the index); a
    released window waits for encoding.
    """
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=False) as session:
        entering = tuple(
            session.scalars(
                select(ExampleWindow.id).where(
                    ExampleWindow.holdout.is_(False), ExampleWindow.reply_at_utc >= cutoff
                )
            )
        )
        for start in range(0, len(entering), SYNC_CHUNK):
            session.execute(
                update(ExampleWindow)
                .where(ExampleWindow.id.in_(entering[start : start + SYNC_CHUNK]))
                .values(holdout=True, embed_version=None, updated_at=now)
            )
        released = cast(
            "CursorResult[Any]",
            session.execute(
                update(ExampleWindow)
                .where(ExampleWindow.holdout.is_(True), ExampleWindow.reply_at_utc < cutoff)
                .values(holdout=False, embed_version=None, updated_at=now)
            ),
        )
        return entering, int(released.rowcount)
