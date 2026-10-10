"""Reading the model's output: one bubble per line (R-ENG-007).

The output convention the fixed rules teach the model:

``普通文字``                 a text bubble
``[表情包:<标签>]``          a sticker bubble, on a line of its own
``[引用:<被引用片段>]``       the first line only: quote that fragment of the user's message
``[不回]``                   the choice to say nothing (alone on its line)

:func:`parse_reply` turns the raw text into a :class:`ParsedReply` and is forgiving about what
models do in practice: full-width colons and brackets, a sticker marker in the middle of a text
line (it becomes a line of its own), a speaker label such as ``她：`` in front of a line.  What it
cannot read as one of the four forms is text - the post-processing decides what to do with it.
Nothing here drops a word of text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

LineKind = Literal["text", "sticker", "no_reply"]
MAX_QUOTE_CHARS = 60

_OPEN = r"[\[【]"
_CLOSE = r"[\]】]"
_COLON = r"[:：]"
_QUOTE_LINE = re.compile(rf"^{_OPEN}\s*引用\s*{_COLON}\s*(?P<body>.*?)\s*{_CLOSE}$")
_STICKER_LINE = re.compile(rf"^{_OPEN}\s*表情包\s*{_COLON}\s*(?P<body>[^\]】]*?)\s*{_CLOSE}$")
_STICKER_INLINE = re.compile(rf"{_OPEN}\s*表情包\s*{_COLON}\s*(?P<body>[^\]】]*?)\s*{_CLOSE}")
_NO_REPLY_LINE = re.compile(rf"^{_OPEN}\s*不回\s*{_CLOSE}$")
_SPEAKER = re.compile(r"^(?:她的回复|回复|她)\s*[:：]\s*")
_INVISIBLE = dict.fromkeys(map(ord, "﻿​‌‍⁠"))


@dataclass(frozen=True)
class ParsedLine:
    """One line of the output: its form and its text (a sticker line holds the tag)."""

    kind: LineKind
    text: str = ""


@dataclass(frozen=True)
class ParsedReply:
    """The output as lines, plus the quote of its first line and what had to be tidied."""

    quote: str | None
    lines: tuple[ParsedLine, ...]
    stray_quotes: int = 0  # quote lines that were not the first line (dropped)
    labels_removed: int = 0  # speaker labels taken off the front of lines

    @property
    def empty(self) -> bool:
        return self.quote is None and not self.lines


def parse_reply(raw: str) -> ParsedReply:
    """Split ``raw`` into lines and recognise the four forms of the output convention."""
    text = raw.translate(_INVISIBLE).replace("\r\n", "\n").replace("\r", "\n")
    quote: str | None = None
    lines: list[ParsedLine] = []
    stray = 0
    labels = 0
    first = True
    for original in text.split("\n"):
        line = original.strip()
        if not line:
            continue
        match = _QUOTE_LINE.match(line)
        if match:
            if first and match["body"] and quote is None:
                quote = match["body"][:MAX_QUOTE_CHARS]
            else:
                stray += 1
            first = False
            continue
        first = False
        if _NO_REPLY_LINE.match(line):
            lines.append(ParsedLine("no_reply"))
            continue
        sticker = _STICKER_LINE.match(line)
        if sticker:
            lines.append(ParsedLine("sticker", sticker["body"]))
            continue
        labelled = _SPEAKER.sub("", line, count=1)
        if labelled != line:
            labels += 1
            line = labelled.strip()
            if not line:
                continue
        lines.extend(_split_inline_stickers(line))
    return ParsedReply(quote, tuple(lines), stray, labels)


def _split_inline_stickers(line: str) -> list[ParsedLine]:
    """A text line with sticker markers inside it becomes text, sticker, text, ... lines."""
    parts: list[ParsedLine] = []
    position = 0
    for match in _STICKER_INLINE.finditer(line):
        before = line[position : match.start()].strip()
        if before:
            parts.append(ParsedLine("text", before))
        parts.append(ParsedLine("sticker", match["body"]))
        position = match.end()
    rest = line[position:].strip()
    if rest:
        parts.append(ParsedLine("text", rest))
    return parts
