"""The text a window is encoded as (R-RET-002).

A context is encoded as one short text: the turns oldest first, each prefixed with who spoke
("你：" for the user, "我：" for her, as she would see it).  The **last two turns carry more
weight**: they are written once at the *front* and then again in their place in the full
context, so the model sees them twice.

Why repetition and not a weighted average of two vectors.  One encoding per window costs one
model run (the average costs two, on a CPU the dominant cost of a rebuild); the repetition puts
the recent turns first, where a long context is never cut off by the model's input limit; and it
needs no weight to tune against vectors that live in different regions of the space.  The same
function builds the text of a query, so windows and queries are always encoded alike.

Non-text messages are written as their event text (:func:`~twin.ingest.events.render_event_text`,
without picture descriptions, so a vector does not change when a description arrives later);
a sticker is ``[表情包]``.  Each turn is clipped so that a long message cannot push the rest out.

``SCHEME`` names this construction; it is part of the encoding identity of an index, so changing
the construction (or its constants) forces a rebuild instead of mixing two kinds of vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from twin.ingest.events import Kind, render_event_text
from twin.retrieval.records import MessageData

SCHEME = "c1"
LABEL_USER = "你："
LABEL_HER = "我："
RECENT_TURNS = 2
RECENT_TURN_CHARS = 120
EARLIER_TURN_CHARS = 60
STICKER_TEXT = "[表情包]"


@dataclass(frozen=True)
class TurnText:
    """One merged turn of a context: who spoke and what (one line)."""

    her: bool
    text: str


def one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def message_text(message: MessageData) -> str:
    """One message as it reads inside an encoded context (no picture description)."""
    kind = message.kind
    if kind in (Kind.TEXT.value, Kind.QUOTE.value):
        return one_line(message.text)
    if kind == Kind.STICKER.value:
        return STICKER_TEXT
    return render_event_text(message) or ""


def turn_text(messages: Sequence[MessageData]) -> TurnText:
    """The turn made of ``messages`` (same speaker, in order)."""
    if not messages:
        raise ValueError("a turn needs at least one message")
    parts = [part for part in (message_text(m) for m in messages) if part]
    return TurnText(messages[0].her, " ".join(parts))


def encoding_text(turns: Sequence[TurnText]) -> str:
    """The text that stands for ``turns`` (oldest first); empty if there is nothing to encode."""
    kept = [turn for turn in turns if turn.text]
    if not kept:
        return ""

    def line(turn: TurnText, limit: int) -> str:
        return (LABEL_HER if turn.her else LABEL_USER) + clip(turn.text, limit)

    full = [line(turn, EARLIER_TURN_CHARS) for turn in kept[:-RECENT_TURNS]]
    recent = [line(turn, RECENT_TURN_CHARS) for turn in kept[-RECENT_TURNS:]]
    if len(kept) <= RECENT_TURNS:
        return "\n".join(recent)
    return "\n".join([*recent, *full, *recent])
