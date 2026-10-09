"""The AI-flavoured phrases that must not appear in her replies (R-ENG-008, R-SAFE-003).

``config/lists/ai_phrases.txt`` (``engine.ai_phrases_file``) is one list in groups, each group
opened by a comment line ``# --- <title>``.  The groups matter:

``self``
    the bot calling itself an AI ("作为AI", "我是一个语言模型", "人工智能").  Normally a hard
    violation: the reply is asked for again.  The one exception is the honest answer to a user who
    sincerely asks whether she is an AI (R-SAFE-003): then these words are not held against the
    reply.
``register``
    customer-service wording and essay connectors ("希望对你有帮助", "总之", "首先，").  The bubble
    that contains one is removed.
``markup``
    Markdown leaking into a chat ("**", "##", "```").  The marks are taken out of the line.

A phrase matches as a case-insensitive substring.  A list without the group titles still works:
every phrase is then a ``register`` phrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from twin.config.lists import WordListError

GROUP_SELF = "self"
GROUP_REGISTER = "register"
GROUP_MARKUP = "markup"
_HEADER = re.compile(r"^#\s*---\s*(?P<title>.+?)\s*$")
_TITLE_WORDS = {
    "self-identification": GROUP_SELF,
    "markdown": GROUP_MARKUP,
    "formatting": GROUP_MARKUP,
}


@dataclass(frozen=True)
class AiPhrases:
    """The phrases of the three groups, lower-cased."""

    self_reference: tuple[str, ...]
    register: tuple[str, ...]
    markup: tuple[str, ...]

    def find_self_reference(self, text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in self.self_reference)

    def self_reference_hits(self, text: str) -> tuple[str, ...]:
        """The self-identification phrases ``text`` contains (lower-cased, list order)."""
        lowered = text.lower()
        return tuple(phrase for phrase in self.self_reference if phrase in lowered)

    def find_register(self, text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in self.register)

    def strip_markup(self, text: str) -> str:
        """``text`` without the Markdown marks (longest mark first)."""
        for mark in sorted(self.markup, key=len, reverse=True):
            text = text.replace(mark, "")
        return text

    def has_markup(self, text: str) -> bool:
        return any(mark in text for mark in self.markup)


def parse_ai_phrases(text: str) -> AiPhrases:
    """The groups of a phrase list (see the module description)."""
    groups: dict[str, dict[str, None]] = {GROUP_SELF: {}, GROUP_REGISTER: {}, GROUP_MARKUP: {}}
    current = GROUP_REGISTER
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        header = _HEADER.match(line)
        if header:
            title = header["title"].lower()
            current = next(
                (group for word, group in _TITLE_WORDS.items() if word in title), GROUP_REGISTER
            )
            continue
        if line.startswith("#"):
            continue
        groups[current].setdefault(line.lower(), None)
    return AiPhrases(
        tuple(groups[GROUP_SELF]), tuple(groups[GROUP_REGISTER]), tuple(groups[GROUP_MARKUP])
    )


def load_ai_phrases(path: Path) -> AiPhrases:
    try:
        return parse_ai_phrases(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WordListError(f"cannot read word list {path}: {exc}") from exc
