"""Loading of the editable word lists in ``config/lists/`` (R-CFG-005).

File format: one entry per line, UTF-8; blank lines and lines starting with ``#``
are ignored; a trailing `` # comment`` is not supported (so ``#`` can occur in an
entry).  ``ai_phrases.txt`` and ``crisis_keywords.txt`` hold literal substrings;
``commitment_patterns.txt`` holds Python regular expressions.
"""

from __future__ import annotations

import re
from pathlib import Path

_CHECKOUT = Path(__file__).resolve().parents[3]


class WordListError(ValueError):
    """A list file is unreadable or contains an invalid entry."""


def locate_list_file(root: Path, relative: str) -> Path:
    """The file named by a ``*_file`` setting: below ``root``, else in the source checkout.

    ``root`` is the project root of the running installation.  The lists ship with the
    repository; when ``root`` has no copy (tests with a temporary home, a data-only root) the
    copy next to the source tree is used so a missing user copy never changes behaviour.
    """
    for base in (root, _CHECKOUT):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    raise WordListError(f"list file {relative!r} was found neither in {root} nor in {_CHECKOUT}")


def load_word_list(path: Path) -> list[str]:
    """Read literal entries, de-duplicated, in file order."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WordListError(f"cannot read word list {path}: {exc}") from exc
    seen: dict[str, None] = {}
    for line in text.splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            seen.setdefault(entry, None)
    return list(seen)


def load_regex_list(path: Path) -> list[re.Pattern[str]]:
    """Read regular expressions and compile them (invalid entries are errors)."""
    patterns: list[re.Pattern[str]] = []
    for entry in load_word_list(path):
        try:
            patterns.append(re.compile(entry))
        except re.error as exc:
            raise WordListError(f"{path}: invalid regular expression {entry!r}: {exc}") from exc
    return patterns
