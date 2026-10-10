"""The short natural answer for a reply that could not be made (R-ENG-010).

When the model failed three times, the user still gets a word - never an error text, a stack or a
piece of the system prompt - and the word is hers: one of the short answers she says most often,
drawn from the frequent sentences of her profile in proportion to how often she says them.  A
sentence is a short answer when it is at most :data:`MAX_CHARS` characters long and consists of
words only - no brackets (a media marker), no digits or Latin letters (a name, a number, a
link), no punctuation of a question.  A profile without such a sentence gives no answer: the engine
then stays silent and the alert says so.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MAX_CHARS = 6
_WORDS_ONLY = re.compile(r"^[\u4e00-\u9fff\u3040-\u30ff]+$")


@dataclass(frozen=True)
class ShortAnswers:
    """Her frequent short answers with how often she says each."""

    answers: tuple[tuple[str, int], ...]

    @classmethod
    def from_phrases(cls, phrases: dict[str, Any] | None) -> ShortAnswers:
        """The short answers among the frequent sentences of a profile (``None``: no profile).

        ``phrases`` is the payload a profile version stores: the frequent sentences of each
        party, ``{"her": {"sentences": [[text, count], ...], ...}, "user": {...}}``.  The words
        are hers - what the user says is never an answer of hers.
        """
        found: list[tuple[str, int]] = []
        her = (phrases or {}).get("her")
        for entry in her.get("sentences", ()) if isinstance(her, dict) else ():
            try:
                text, count = str(entry[0]).strip(), int(entry[1])
            except (IndexError, TypeError, ValueError):
                continue
            if 0 < len(text) <= MAX_CHARS and count > 0 and _WORDS_ONLY.match(text):
                found.append((text, count))
        return cls(tuple(found))

    def __bool__(self) -> bool:
        return bool(self.answers)

    def texts(self) -> Sequence[str]:
        return [text for text, _ in self.answers]

    def pick(self, rng: random.Random) -> str | None:
        """One answer, drawn in proportion to how often she says it; ``None`` if there are none."""
        if not self.answers:
            return None
        texts = [text for text, _ in self.answers]
        weights = [count for _, count in self.answers]
        return rng.choices(texts, weights=weights, k=1)[0]
