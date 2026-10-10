"""The conversation of one local day, as lines for the extractor and the summarizer.

Real records are read through :mod:`twin.ingest.corpus` (the only door to ``messages``, R-STO-007)
and turned into lines by :mod:`twin.ingest.transcript` (the one place a stored message becomes a
line of text for a model): text as written, stickers and calls as their event text, system
notices dropped.  The bot's own conversation comes from a
:class:`~twin.memory.recent.BotTurnReader`.  A day is the local day of the scope
(:class:`~twin.memory.localdate.MemoryClock`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, timedelta
from typing import TYPE_CHECKING

from twin.ingest.corpus import messages_between
from twin.ingest.transcript import message_text
from twin.memory.extract import DialogueLine
from twin.memory.localdate import BOT, REAL, MemoryClock
from twin.memory.recent import BotTurnReader

if TYPE_CHECKING:
    from twin.services import Services

BOT_LINE_CHARS = 400
ONE_MICROSECOND = timedelta(microseconds=1)


def real_day_lines(services: Services, clock: MemoryClock, day: date) -> list[DialogueLine]:
    """Her messages and the user's of ``day`` (her local day), oldest first."""
    start, end = clock.real_bounds(day)
    lines: list[DialogueLine] = []
    with services.db.session() as session:
        for row in session.scalars(messages_between(start, end - ONE_MICROSECOND)):
            text = message_text(row)
            if text:
                lines.append(
                    DialogueLine(
                        row.id, "user" if row.is_sent else "her", text, row.create_time_utc
                    )
                )
    return lines


def bot_day_lines(reader: BotTurnReader, clock: MemoryClock, day: date) -> list[DialogueLine]:
    """The bot's conversation on ``day`` (the bot's local day), oldest first."""
    start, end = clock.bounds_of(BOT, day)
    return [
        DialogueLine(m.id, m.role, " ".join(m.text.split())[:BOT_LINE_CHARS], m.at)
        for m in reader.messages_between(start, end)
        if m.text.strip()
    ]


def day_lines(
    services: Services, clock: MemoryClock, scope: str, day: date, reader: BotTurnReader | None
) -> list[DialogueLine]:
    if scope == REAL:
        return real_day_lines(services, clock, day)
    return bot_day_lines(reader, clock, day) if reader is not None else []


class LineHasher:
    """Fingerprint of dialogue lines (ids, times and texts), fed one line at a time."""

    def __init__(self) -> None:
        self._digest = hashlib.sha1(usedforsecurity=False)

    def update(self, line: DialogueLine) -> None:
        self._digest.update(f"{line.ref}\x1f{line.at.isoformat()}\x1f{line.text}\x1e".encode())

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def lines_hash(lines: Sequence[DialogueLine]) -> str:
    """A fingerprint of the lines: same hash, same input."""
    hasher = LineHasher()
    for line in lines:
        hasher.update(line)
    return hasher.hexdigest()
