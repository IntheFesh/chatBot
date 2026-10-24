"""Text helpers of the post-processing: where a line may be cut, and how it is cut.

Nothing is ever truncated.  A line that is too long is split at the places a person would break
it - the end of a sentence first, then a comma or a space, then after an emoji code - and the
pieces are packed into bubbles of at most the limit.  A piece that has no place to cut stays whole
even when it is longer than the limit: a long bubble is better than a broken word.
"""

from __future__ import annotations

import re

STRONG_ENDS = "。！？!?；;…\n"
WEAK_ENDS = "，,、 　"
_CODE_END = re.compile(r"\[[一-龥A-Za-z]{1,6}\]")
_REPEATED_MARKS = re.compile(r"([！？!?])\1{3,}")
_SPACES = re.compile(r"[ \t　]{2,}")
_LEADING_LIST = re.compile(r"^\s*(?:[-*•·]\s+|\d{1,2}[、)）]\s*|\d{1,2}\.\s+)")
_TRAILING_STOPS = "，,、"


def split_after(text: str, delimiters: str) -> list[str]:
    """``text`` cut after every delimiter character, the delimiter staying with its piece."""
    pieces: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in delimiters:
            pieces.append(text[start : index + 1])
            start = index + 1
    if start < len(text):
        pieces.append(text[start:])
    return [piece for piece in pieces if piece.strip()]


def split_after_codes(text: str) -> list[str]:
    """``text`` cut after every bracket emoji code."""
    pieces: list[str] = []
    start = 0
    for match in _CODE_END.finditer(text):
        if match.end() < len(text):
            pieces.append(text[start : match.end()])
            start = match.end()
    pieces.append(text[start:])
    return [piece for piece in pieces if piece.strip()]


def _refine(pieces: list[str], limit: int, delimiters: str | None) -> list[str]:
    refined: list[str] = []
    for piece in pieces:
        if len(piece.strip()) <= limit:
            refined.append(piece)
        elif delimiters is None:
            refined.extend(split_after_codes(piece))
        else:
            refined.extend(split_after(piece, delimiters))
    return refined


def split_semantic(text: str, limit: int) -> list[str]:
    """Split ``text`` into bubbles of at most ``limit`` characters where it can be done."""
    if limit < 1:
        raise ValueError("the limit must be at least one character")
    if len(text.strip()) <= limit:
        return [text.strip()]
    pieces = split_after(text, STRONG_ENDS)
    pieces = _refine(pieces, limit, WEAK_ENDS)
    pieces = _refine(pieces, limit, None)
    bubbles: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > limit:
            bubbles.append(current.strip())
            current = piece
        else:
            current += piece
    if current.strip():
        bubbles.append(current.strip())
    return bubbles or [text.strip()]


def strip_trailing_stops(text: str) -> str:
    """``text`` without commas and pauses at its end."""
    return text.rstrip(_TRAILING_STOPS + " 　")


def clamp_marks(text: str, longest: int = 3) -> str:
    """Runs of the same ``!`` / ``?`` longer than ``longest`` are cut to ``longest``."""
    return _REPEATED_MARKS.sub(lambda m: m.group(1) * longest, text)


def squeeze_spaces(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def strip_list_marker(text: str) -> str:
    """Take a bullet or a number ("- ", "1. ") off the front of a line."""
    return _LEADING_LIST.sub("", text, count=1)
