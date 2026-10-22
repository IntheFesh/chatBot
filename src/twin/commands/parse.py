"""Recognising a command in a message (R-CMD-001, R-CMD-003).

Two readings of a message share one function, :func:`parse_command`.

**Strict** (no command table at hand, ``known=None``; this is what :func:`is_command_text` uses).
A message is a command when, after the white space around it is removed, it starts with a slash
(``/`` or the full-width ``／``) that is **immediately** followed by a name, and the name ends the
message or is followed by white space or a colon (``:`` or ``：``).  So

* ``/思考 开``, ``／思考：开``, ``/思考:开``, ``  /思考　开 `` (ideographic space) all name the
  command ``思考`` with the argument ``开``;
* ``/Status`` and ``／ＳＴＡＴＵＳ`` name the same command - names are compared after Unicode
  compatibility normalisation (full-width letters become ASCII) and case folding;
* ``/::)`` and ``/:8-)`` (WeChat's old text emoticons), ``/home/user`` (a path), ``/`` alone and
  ``/ 思考`` (a slash and then a space) are **not** commands: they are chat.

**With the table** (``known`` says whether a folded name is a command or an alias, the router
passes ``registry.find``).  R-CMD-003 asks for tolerance ("多余空格、中英文冒号都能识别"), and the
table is what tells ``/ 状态`` from ``/ 吃饭``, so the same message is read once more, leniently,
when the strict reading did not find a registered name:

* a slash followed by white space may be followed by a name: ``/ 状态``, ``／ 状态``;
* the name may be followed by any sentence punctuation instead of a space or a colon (``Po``
  characters such as ``，,、。；！？…``; not ``/ \\ . # * & % @`` and quotes, which belong to paths
  and to the argument): ``/思考，开``;
* a name written together with its argument is split by the **longest registered name** that
  is a prefix: ``/思考开`` is the command ``思考`` with the argument ``开``.  Only a name that
  ends with a non-ASCII character is split that way (``/Statusbar`` stays one unknown name, so
  a mistyped ASCII word is answered with the help summary), and not after ``/ `` (``/ 思考开``
  is chat).

Everything that is not recognised is as it was: a slash and an unknown name is still a command
that is answered with the help summary (R-CMD-001); chat, paths and emoticons stay chat.

The argument is returned as it was written (only trimmed): free text such as a memory to keep must
not be altered; choice arguments are matched after the same normalisation as names
(:func:`fold`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

SLASHES = ("/", "／")
SEPARATORS = " \t　:："
_NAME = re.compile(r"\w+")
# Po characters that are not a separator: they belong to paths, URLs, hash tags and quotes
_NOT_SEPARATORS = frozenset("/\\.#*&%@'\"`")

Known = Callable[[str], object | None]
"""Says whether a folded name is in the command table: the spec (or anything but ``None``)."""


def fold(text: str) -> str:
    """``text`` as names and choices are compared: compatibility-normalised, case-folded."""
    return unicodedata.normalize("NFKC", text).casefold().strip()


@dataclass(frozen=True)
class ParsedCommand:
    """A message that is a command: its folded name, its argument text and the trimmed message."""

    name: str
    args: str
    raw: str


def _is_separator(char: str) -> bool:
    """White space, a colon or sentence punctuation (``，,、。；！？…``) after a name."""
    if char in SEPARATORS or char.isspace():
        return True
    if unicodedata.category(char) != "Po":
        return False
    return unicodedata.normalize("NFKC", char) not in _NOT_SEPARATORS


def _strict(rest: str, stripped: str) -> ParsedCommand | None:
    named = _NAME.match(rest)
    if named is None:
        return None
    after = rest[named.end() :]
    if after and after[0] not in SEPARATORS:
        return None
    return ParsedCommand(fold(named.group()), after.lstrip(SEPARATORS).strip(), stripped)


def _lenient(rest: str, stripped: str, known: Known) -> ParsedCommand | None:
    """The command of ``rest`` (the message after its slash) found with the table, else ``None``."""
    body = rest.lstrip()
    spaced = len(body) != len(rest)
    run = _NAME.match(body)
    if run is None:
        return None
    word = run.group()
    for end in range(len(word), 0, -1):
        name = fold(word[:end])
        if known(name) is None:
            continue
        remainder = body[end:]
        if not remainder:
            return ParsedCommand(name, "", stripped)
        first = remainder[0]
        if first in SEPARATORS or first.isspace():
            return ParsedCommand(name, remainder.lstrip(SEPARATORS).strip(), stripped)
        if _is_separator(first):  # "，": one punctuation mark stands for the space
            return ParsedCommand(name, remainder[1:].lstrip(SEPARATORS).strip(), stripped)
        # Nothing separates the name from what follows: "思考开" is split, but only after a name
        # that ends in a CJK character and not after "/ " ("/ 思考开" is chat); what follows a
        # name inside a word of an ASCII command, or a path ("/status/x"), is not an argument.
        if end == len(word) or spaced or word[end - 1].isascii():
            return None
        return ParsedCommand(name, remainder.strip(), stripped)
    return None


def parse_command(text: str, known: Known | None = None) -> ParsedCommand | None:
    """The command in ``text``, or ``None`` when the message is chat.

    Without ``known`` this is the strict reading; with it (``registry.find``) the lenient one is
    tried when the strict one did not find a registered name (see the module description).  A
    slash and a name that is not registered is a command either way (``/zzz``): the router
    answers it with the help summary.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in SLASHES:
        return None
    rest = stripped[1:]
    strict = _strict(rest, stripped)
    if known is None or (strict is not None and known(strict.name) is not None):
        return strict
    return _lenient(rest, stripped, known) or strict


def is_command_text(text: str) -> bool:
    """Is this message a command (strict reading, no table)?  ``/ 吃饭``, ``/思考，开`` are not."""
    return parse_command(text) is not None
