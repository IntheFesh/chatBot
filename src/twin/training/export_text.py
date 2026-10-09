"""How her messages are written into a training sample (R-TRN-003, R-SAFE-006).

Two different renderings of the same messages:

**The target** (her reply block - the only part the model learns):

* her messages follow each other as lines (a burst is one line per message);
* a sticker is ``[表情包:<标签>]`` with the tag of the sticker library, ``[表情包]`` when it has
  none;
* an emoji code such as ``[拥抱]`` stays as she typed it;
* a reply that quotes starts with the line ``[引用:<quoted text, 30 characters>]`` - only when the
  first message of the block is the quote, because the output convention allows the quote line
  as the first line only (``engine.parsing``);
* everything the bot could not send itself - pictures, voice, video, calls, transfers, red
  packets, locations, files, links, chat histories, system events - is **deleted** from the
  target, and so is a text line that merely looks like one of those (a typed ``[链接]``).  A
  block with nothing left is not a sample.

**The context** (the turns before it, hers and the user's): the same text, sticker and quote
lines, but a message the bot could not send stays visible as its event text
(:func:`~twin.ingest.events.render_event_text`), because that is what the user sends the bot.

Newlines inside one message become spaces: a line break in a reply means "another bubble".
Control tokens of the chat template and the slot names of LLaMA-Factory are removed
(:func:`~twin.engine.style_prompt.scrub`).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from twin.engine.style_prompt import scrub
from twin.ingest.events import EventTextDetector, Kind, default_detector, render_event_text
from twin.profile.textstats import EMOJI_CODE_RE
from twin.retrieval.records import MessageData

QUOTE_CHARS = 30
STICKER_TEXT = "[表情包]"
QUOTE_PREFIX = "[引用:"

# drop reasons that are not a message kind
EVENT_TEXT_LITERAL = "event_text_literal"
EMPTY_TEXT = "empty_text"
QUOTE_NOT_FIRST = "quote_not_first"

StickerLabeler = Callable[[str], str | None]

_LINE_BREAKS = re.compile(r"\s*[\r\n]+\s*")


def collapse(text: str | None) -> str:
    """One line: line breaks become single spaces, control text is removed, ends are trimmed."""
    if not text:
        return ""
    return scrub(_LINE_BREAKS.sub(" ", text)).strip()


def sticker_line(message: MessageData, labeler: StickerLabeler | None) -> str:
    """``[表情包:<标签>]``, or ``[表情包]`` for a sticker without a tag."""
    tag = labeler(message.sticker_md5) if labeler and message.sticker_md5 else None
    label = collapse(tag).replace("]", "")
    return f"[表情包:{label}]" if label else STICKER_TEXT


def quote_line(message: MessageData, *, limit: int = QUOTE_CHARS) -> str | None:
    """``[引用:<quoted text cut to 30 characters>]`` of a quote reply, or ``None``."""
    if message.kind != Kind.QUOTE.value:
        return None
    quote = message.quote or {}
    source = quote.get("quoteContent") or quote.get("quoteTitle")
    if not isinstance(source, str):
        return None
    fragment = collapse(source)[:limit].rstrip().replace("]", ")")
    return f"{QUOTE_PREFIX}{fragment}]" if fragment else None


def _own_lines(message: MessageData, labeler: StickerLabeler | None) -> list[str]:
    """The lines of a message the bot could have sent (text, quote text, sticker)."""
    if message.kind == Kind.STICKER.value:
        return [sticker_line(message, labeler)]
    text = collapse(message.text)
    return [text] if text else []


def context_text(messages: Sequence[MessageData], labeler: StickerLabeler | None) -> str:
    """One turn of the context: its messages as lines (event text where the bot cannot copy)."""
    lines: list[str] = []
    for message in messages:
        if message.kind in (Kind.TEXT.value, Kind.STICKER.value, Kind.QUOTE.value):
            header = quote_line(message)
            if header:
                lines.append(header)
            lines.extend(_own_lines(message, labeler))
        elif message.kind != Kind.SYSTEM.value:
            event = render_event_text(message)
            if event:
                lines.append(collapse(event))
    return "\n".join(line for line in lines if line)


@dataclass
class Target:
    """The reply block as the training target, and what was left out of it."""

    lines: list[str] = field(default_factory=list)
    stickers: int = 0
    text_lines: int = 0
    code_lines: int = 0
    quoted: bool = False
    dropped: Counter[str] = field(default_factory=Counter)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def empty(self) -> bool:
        return not self.has_content

    @property
    def has_content(self) -> bool:
        """True when at least one line is not just the quote header."""
        return self.stickers > 0 or self.text_lines > 0


def render_target(
    messages: Sequence[MessageData],
    labeler: StickerLabeler | None = None,
    detector: EventTextDetector = default_detector,
) -> Target:
    """Her reply block as a target (see the module description)."""
    target = Target()
    for message in messages:
        kind = message.kind
        if kind not in (Kind.TEXT.value, Kind.STICKER.value, Kind.QUOTE.value):
            target.dropped[kind] += 1
            continue
        if kind == Kind.QUOTE.value:
            header = quote_line(message)
            if header:
                if target.lines or target.quoted:
                    target.dropped[QUOTE_NOT_FIRST] += 1
                else:
                    target.lines.append(header)
                    target.quoted = True
        for line in _own_lines(message, labeler):
            if kind != Kind.STICKER.value and detector.is_event_text(line):
                target.dropped[EVENT_TEXT_LITERAL] += 1
            elif kind == Kind.STICKER.value:
                target.lines.append(line)
                target.stickers += 1
            else:
                target.lines.append(line)
                target.text_lines += 1
                target.code_lines += int(EMOJI_CODE_RE.search(line) is not None)
        if kind != Kind.STICKER.value and not collapse(message.text):
            target.dropped[EMPTY_TEXT] += 1
    return target


def event_lines(lines: Sequence[str], detector: EventTextDetector = default_detector) -> list[str]:
    """The lines among ``lines`` that are event text (an export must contain none in a target)."""
    return [line for line in lines if detector.is_event_text(line)]
