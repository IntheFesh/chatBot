"""Recognising a command in a message (R-CMD-001, R-CMD-003).

A message is a command when, after the white space around it is removed, it starts with a slash
(``/`` or the full-width ``／``) that is **immediately** followed by a name, and the name ends the
message or is followed by white space or a colon (``:`` or ``：``).  So

* ``/思考 开``, ``／思考：开``, ``/思考:开``, ``  /思考　开 `` (ideographic space) all name the
  command ``思考`` with the argument ``开``;
* ``/Status`` and ``／ＳＴＡＴＵＳ`` name the same command - names are compared after Unicode
  compatibility normalisation (full-width letters become ASCII) and case folding;
* ``/::)`` and ``/:8-)`` (WeChat's old text emoticons), ``/home/user`` (a path), ``/`` alone and
  ``/ 思考`` (a slash and then a space) are **not** commands: they are chat.

The argument is returned as it was written (only trimmed): free text such as a memory to keep must
not be altered; choice arguments are matched after the same normalisation as names
(:func:`fold`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SLASHES = ("/", "／")
SEPARATORS = " \t　:："
_NAME = re.compile(r"\w+")


def fold(text: str) -> str:
    """``text`` as names and choices are compared: compatibility-normalised, case-folded."""
    return unicodedata.normalize("NFKC", text).casefold().strip()


@dataclass(frozen=True)
class ParsedCommand:
    """A message that is a command: its folded name, its argument text and the trimmed message."""

    name: str
    args: str
    raw: str


def parse_command(text: str) -> ParsedCommand | None:
    """The command in ``text``, or ``None`` when the message is chat."""
    stripped = text.strip()
    if not stripped or stripped[0] not in SLASHES:
        return None
    rest = stripped[1:]
    named = _NAME.match(rest)
    if named is None:
        return None
    after = rest[named.end() :]
    if after and after[0] not in SEPARATORS:
        return None
    return ParsedCommand(fold(named.group()), after.lstrip(SEPARATORS).strip(), stripped)


def is_command_text(text: str) -> bool:
    """Is this message a command?  The engine marks the row of such a message ``is_command``."""
    return parse_command(text) is not None
