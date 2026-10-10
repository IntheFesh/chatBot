"""One way to show a reply, hers or the bot's (R-EVAL-001).

A blind test only measures style when nothing but style can give a candidate away.  Both the
real reply and the bot's reply are therefore first turned into the same structure - a
:class:`Candidate`: the bubbles (a line of text, with the emoji codes as she typed them, or a
sticker) and the fragment that is quoted - and only then shown, by one function,
:func:`render_candidate`:

* every bubble starts a new line; a line break inside a bubble is another line, so a message
  with two lines and two bubbles with one line each look the same;
* a sticker is always ``[表情包：<描述>]``, the description of the sticker library for its MD5
  (the bot's sticker is looked up the same way as hers);
* a quote is always one first line ``[引用：<被引用的话>]`` of at most 30 characters;
* emoji codes (``[拥抱]``) stay as they are.

The messages of the conversation before the reply are shown with the same rules
(:func:`render_turn`), the ones the bot cannot send as their event text (``[图片：…]``).  The
function takes the sticker description as a parameter, so a test can show that equal content gives
equal output without a library.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from twin.ingest.events import Kind, default_detector, render_event_text
from twin.retrieval.records import MessageData

QUOTE_CHARS = 30
STICKER_UNLABELLED = "未标注"
STICKER_LABEL_CHARS = 40
HER_LABEL = "她"
USER_LABEL = "我"
_BREAKS = re.compile(r"\s*[\r\n]+\s*")
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")

StickerLabel = Callable[[str], str | None]
LineKind = Literal["text", "sticker"]


@dataclass(frozen=True)
class CandidateLine:
    """One bubble: a line of text, or a sticker (by MD5)."""

    kind: LineKind
    text: str = ""
    sticker_md5: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "sticker" and not self.sticker_md5:
            raise ValueError("a sticker line needs the MD5 of its sticker")

    def to_json(self) -> dict[str, str]:
        if self.kind == "sticker":
            return {"k": "sticker", "md5": self.sticker_md5 or ""}
        return {"k": "text", "t": self.text}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CandidateLine:
        if data.get("k") == "sticker":
            return cls("sticker", sticker_md5=str(data["md5"]))
        return cls("text", text=str(data["t"]))


@dataclass(frozen=True)
class Candidate:
    """A reply in the one structure both sides are shown in."""

    lines: tuple[CandidateLine, ...]
    quote: str | None = None

    @property
    def empty(self) -> bool:
        return not self.lines

    @property
    def stickers(self) -> tuple[str, ...]:
        return tuple(line.sticker_md5 for line in self.lines if line.sticker_md5)

    def to_json(self) -> dict[str, Any]:
        return {"quote": self.quote, "lines": [line.to_json() for line in self.lines]}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Candidate:
        quote = data.get("quote")
        return cls(
            tuple(CandidateLine.from_json(line) for line in data.get("lines", [])),
            str(quote) if quote is not None else None,
        )


# ------------------------------------------------------------------- the pieces


def clean(text: str | None) -> str:
    """One line of text: control characters out, line breaks and runs of blanks to one space."""
    if not text:
        return ""
    return _BREAKS.sub(" ", _CONTROL.sub("", text)).strip()


def text_lines(text: str | None) -> list[str]:
    """The lines a text is shown as: one per line break, blanks dropped."""
    if not text:
        return []
    return [
        line for line in (clean(part) for part in _BREAKS.split(_CONTROL.sub("", text))) if line
    ]


def quote_fragment(source: str | None) -> str | None:
    """The quoted words as shown: one line, at most 30 characters, ``]`` written as ``)``."""
    fragment = clean(source)[:QUOTE_CHARS].rstrip().replace("]", ")")
    return fragment or None


def quote_of(message: MessageData) -> str | None:
    """What a quoting message quotes, as shown; ``None`` for any other message."""
    if message.kind != Kind.QUOTE.value:
        return None
    quote = message.quote or {}
    source = quote.get("quoteContent") or quote.get("quoteTitle")
    return quote_fragment(source) if isinstance(source, str) else None


def sticker_text(md5: str, label: StickerLabel) -> str:
    """``[表情包：<描述>]`` for the sticker with this MD5."""
    described = clean(label(md5))[:STICKER_LABEL_CHARS].replace("]", ")")
    return f"[表情包：{described or STICKER_UNLABELLED}]"


# ----------------------------------------------------------------------- candidates


def candidate_from_messages(messages: Sequence[MessageData]) -> Candidate:
    """Her real reply (messages of the ``messages`` table) as a candidate.

    Only what the bot could have sent is kept: text, stickers and the quote of a quote reply
    (``twin.ingest.events.is_reproducible``); the pipeline output has nothing else.
    """
    quote: str | None = None
    lines: list[CandidateLine] = []
    for message in messages:
        if message.kind == Kind.STICKER.value:
            if message.sticker_md5:
                lines.append(CandidateLine("sticker", sticker_md5=message.sticker_md5))
            continue
        if message.kind not in (Kind.TEXT.value, Kind.QUOTE.value):
            continue
        header = quote_of(message)
        if header and quote is None and not lines:
            quote = header  # the output convention allows the quote on the first bubble only
        lines.extend(CandidateLine("text", text=line) for line in text_lines(message.text))
    return Candidate(tuple(lines), quote)


def candidate_from_bubbles(
    bubbles: Sequence[tuple[str, str | None, str]], quote: str | None
) -> Candidate:
    """The bot's reply as a candidate; bubbles are ``(kind, sticker MD5, text)``."""
    lines: list[CandidateLine] = []
    for kind, md5, text in bubbles:
        if kind == "sticker":
            if md5:
                lines.append(CandidateLine("sticker", sticker_md5=md5))
            continue
        lines.extend(CandidateLine("text", text=line) for line in text_lines(text))
    return Candidate(tuple(lines), quote_fragment(quote))


def has_event_text(candidate: Candidate) -> bool:
    """True when a text line of the candidate is event text or a media marker (R-SAFE-006)."""
    return any(
        line.kind == "text" and default_detector.is_event_text(line.text)
        for line in candidate.lines
    )


# ----------------------------------------------------------------------- rendering


def render_candidate(candidate: Candidate, label: StickerLabel) -> str:
    """The text a candidate is shown as (see the module description)."""
    shown: list[str] = []
    if candidate.quote:
        fragment = quote_fragment(candidate.quote)
        if fragment:
            shown.append(f"[引用：{fragment}]")
    for line in candidate.lines:
        if line.kind == "sticker":
            shown.append(sticker_text(line.sticker_md5 or "", label))
        else:
            shown.extend(text_lines(line.text))
    return "\n".join(shown)


def message_lines(message: MessageData, label: StickerLabel) -> list[str]:
    """One message of the context as lines: its words, or the event text of what it carries."""
    if message.kind == Kind.STICKER.value:
        return [sticker_text(message.sticker_md5 or "", label)] if message.sticker_md5 else []
    if message.kind in (Kind.TEXT.value, Kind.QUOTE.value):
        lines: list[str] = []
        header = quote_of(message)
        if header:
            lines.append(f"[引用：{header}]")
        lines.extend(text_lines(message.text))
        return lines
    if message.kind == Kind.SYSTEM.value:
        return []
    event = render_event_text(message)
    return [clean(event)] if event else []


def render_turn(who: str, lines: Sequence[str]) -> str:
    """A turn of the context for the screen: the first line carries who spoke."""
    if not lines:
        return ""
    return "\n".join([f"{who}：{lines[0]}", *(f"　　{line}" for line in lines[1:])])
