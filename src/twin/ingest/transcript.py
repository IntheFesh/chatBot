"""Chat messages as the lines of a transcript shown to a model (R-IMP-007, R-LLM-009).

The persona card generation (round 06) and the sticker context correction show stretches of the
real conversation to DeepSeek.  This module is the one place that turns a stored message into
one line of such a stretch: text and quote replies as written (one line, shortened), stickers as
``[表情包:<标签>]`` when the library knows a tag and ``[表情包]`` otherwise, everything else as
its event text (:func:`~twin.ingest.events.render_event_text`).  System notices have no line.
The caller redacts the finished text (:mod:`twin.llm.redaction`) before it leaves the machine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from twin.ingest.events import Kind, render_event_text
from twin.storage.chat_models import Message

MESSAGE_CHAR_CAP = 120
STICKER_TEXT = "[表情包]"
HER_LABEL = "她"
USER_LABEL = "对方"

StickerTagOf = Callable[[str], str | None]


@dataclass(frozen=True)
class TranscriptLine:
    """One message of a transcript: who wrote it, what it says, and which message it is."""

    message_id: str
    her: bool
    text: str

    def render(self, *, her_label: str = HER_LABEL, user_label: str = USER_LABEL) -> str:
        return f"{her_label if self.her else user_label}：{self.text}"


def _one_line(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def message_text(
    message: Message, *, sticker_tag_of: StickerTagOf | None = None, limit: int = MESSAGE_CHAR_CAP
) -> str | None:
    """The text of one message as it appears in a transcript; ``None`` for a system notice."""
    kind = message.kind
    if kind == Kind.SYSTEM.value:
        return None
    if kind in (Kind.TEXT.value, Kind.QUOTE.value):
        text = _one_line(message.text or "", limit)
        return text or None
    if kind == Kind.STICKER.value:
        tag = (
            sticker_tag_of(message.sticker_md5) if sticker_tag_of and message.sticker_md5 else None
        )
        return f"[表情包:{tag}]" if tag else STICKER_TEXT
    return render_event_text(message)


def transcript_lines(
    messages: Iterable[Message],
    *,
    sticker_tag_of: StickerTagOf | None = None,
    limit: int = MESSAGE_CHAR_CAP,
) -> list[TranscriptLine]:
    """The lines of ``messages`` (in the order given), without system notices."""
    lines: list[TranscriptLine] = []
    for message in messages:
        text = message_text(message, sticker_tag_of=sticker_tag_of, limit=limit)
        if text:
            lines.append(TranscriptLine(message.id, not message.is_sent, text))
    return lines
