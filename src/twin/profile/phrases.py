"""High-frequency sentences, n-grams and forms of address (R-PROF-002).

These are pieces of the real messages.  They are kept apart from the numeric metrics: the
profile builder stores them in the separately sealed ``phrases`` column of ``profile_versions``
and never in ``metrics``.  They are used locally only (by the persona card generator, after
redaction, and by tooling that shows candidates to the user).

Nothing here knows a single word.  A *form of address* candidate is an n-gram that occurs
mostly at the start or at the end of messages: for each message the prefix and the suffix of
2 and 3 characters (after dropping trailing punctuation and emoji codes) are counted, and an
n-gram becomes a candidate when it stands at an edge often enough and at least
``MIN_EDGE_SHARE`` of all its occurrences are at an edge.  The list is a set of
*candidates with frequencies*; deciding which are real forms of address is left to the user
and the persona card.

Counting is exact up to :data:`MAX_TRACKED` distinct entries per counter; beyond that the
rarest entries (count 1, then 2, ...) are dropped, which can only lose entries that would not
have reached the reporting thresholds.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from twin.profile.textstats import EMOJI_CODE_RE, EMOJI_RE, strip_trailing

MAX_TRACKED = 400_000
SENTENCE_MIN_CHARS = 2
SENTENCE_MAX_CHARS = 40
NGRAM_SIZES = (2, 3, 4)
MIN_SENTENCE_COUNT = 5
MIN_NGRAM_COUNT = 8
MIN_EDGE_COUNT = 5
MIN_EDGE_SHARE = 0.4
TOP_SENTENCES = 100
TOP_NGRAMS = 100
TOP_ADDRESS = 30

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n\r]+")
_FRAGMENT_SPLIT_RE = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’～~…\[\]()（）]+")


def _prune(counter: Counter[str], limit: int) -> None:
    """Drop the rarest entries until at most ``limit`` remain."""
    threshold = 1
    while len(counter) > limit:
        for key in [k for k, v in counter.items() if v <= threshold]:
            del counter[key]
        threshold += 1


class PhraseCollector:
    """Counts sentences, n-grams and edge n-grams of the messages fed to it."""

    def __init__(self, *, limit: int = MAX_TRACKED) -> None:
        self._limit = limit
        self.sentences: Counter[str] = Counter()
        self.ngrams: dict[int, Counter[str]] = {n: Counter() for n in NGRAM_SIZES}
        self.prefixes: dict[int, Counter[str]] = {2: Counter(), 3: Counter()}
        self.suffixes: dict[int, Counter[str]] = {2: Counter(), 3: Counter()}
        self.messages = 0

    def feed(self, text: str) -> None:
        self.messages += 1
        plain = EMOJI_RE.sub(" ", EMOJI_CODE_RE.sub(" ", text))
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            sentence = sentence.strip()
            if SENTENCE_MIN_CHARS <= len(sentence) <= SENTENCE_MAX_CHARS:
                self.sentences[sentence] += 1
        for fragment in _FRAGMENT_SPLIT_RE.split(plain):
            length = len(fragment)
            for n in NGRAM_SIZES:
                if length < n:
                    continue
                counter = self.ngrams[n]
                for i in range(length - n + 1):
                    counter[fragment[i : i + n]] += 1
        body = strip_trailing(text).strip()
        for n in (2, 3):
            if len(body) >= n:
                head = body[:n]
                tail = body[-n:]
                if _FRAGMENT_SPLIT_RE.search(head) is None:
                    self.prefixes[n][head] += 1
                if _FRAGMENT_SPLIT_RE.search(tail) is None:
                    self.suffixes[n][tail] += 1
        if len(self.sentences) > 2 * self._limit:
            _prune(self.sentences, self._limit)
        for counter in self.ngrams.values():
            if len(counter) > 2 * self._limit:
                _prune(counter, self._limit)

    # ------------------------------------------------------------------ result

    def result(self) -> dict[str, Any]:
        """The sealed payload: frequent sentences, n-grams and address candidates."""
        sentences = [
            [text, count]
            for text, count in self.sentences.most_common(TOP_SENTENCES * 3)
            if count >= MIN_SENTENCE_COUNT
        ][:TOP_SENTENCES]
        ngrams = {
            str(n): [
                [text, count]
                for text, count in counter.most_common(TOP_NGRAMS * 3)
                if count >= MIN_NGRAM_COUNT
            ][:TOP_NGRAMS]
            for n, counter in self.ngrams.items()
        }
        return {
            "messages": self.messages,
            "sentences": sentences,
            "ngrams": ngrams,
            "address_candidates": self.address_candidates(),
        }

    def address_candidates(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for n in (2, 3):
            for term in set(self.prefixes[n]) | set(self.suffixes[n]):
                start = self.prefixes[n].get(term, 0)
                end = self.suffixes[n].get(term, 0)
                edge = start + end
                total = self.ngrams[n].get(term, 0)
                if edge < MIN_EDGE_COUNT or total == 0:
                    continue
                share = min(1.0, edge / total)
                if share < MIN_EDGE_SHARE:
                    continue
                found.append(
                    {
                        "term": term,
                        "start": start,
                        "end": end,
                        "total": total,
                        "edge_share": round(share, 3),
                    }
                )
        found.sort(key=lambda item: (-(item["start"] + item["end"]), item["term"]))
        return found[:TOP_ADDRESS]


def phrase_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    """Sizes of the lists in a phrase payload (what may safely be logged or shown)."""
    return {
        "sentences": len(payload.get("sentences", [])),
        "address_candidates": len(payload.get("address_candidates", [])),
        **{f"ngrams_{n}": len(rows) for n, rows in payload.get("ngrams", {}).items()},
    }
