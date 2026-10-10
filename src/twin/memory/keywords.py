"""The keyword side of the memory search: an inverted index kept in memory (R-MEM-008).

Facts and summaries are few (thousands, not millions), so their words are indexed in memory,
built from the decrypted rows when the process starts and kept up to date as rows change.
There is deliberately no full-text table in SQLite: it would hold the plain text next to the
sealed columns (R-STO-002).

Chinese has no spaces between words and the project carries no word segmenter, so the index
works on **character bigrams** of every run of Chinese text (a run of one character counts as
itself unless it is a particle), plus lower-cased Latin words and numbers.  A query is
tokenised the same way; a document's score is the share of the query's weight - each token
weighs its inverse document frequency - that the document contains, a number in ``[0, 1]``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

_CJK_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
_LATIN = re.compile(r"[A-Za-z]{2,}|[0-9]{2,}")
# single Chinese characters that carry no topic when they stand alone
PARTICLES = frozenset("的了是我你他她它们吗呢吧啊呀哦嗯哈就都也还和与在有不这那个之么")


def tokens_of(text: str) -> frozenset[str]:
    """The index tokens of ``text``: bigrams of Chinese runs, Latin words and numbers."""
    found: set[str] = set()
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            if run not in PARTICLES:
                found.add(run)
            continue
        found.update(run[i : i + 2] for i in range(len(run) - 1))
    found.update(word.lower() for word in _LATIN.findall(text))
    return frozenset(found)


@dataclass(frozen=True)
class KeywordHit:
    doc_id: str
    score: float  # share of the query weight the document contains, in [0, 1]


class KeywordIndex:
    """Inverted index of short texts, with incremental add and remove."""

    def __init__(self) -> None:
        self._postings: dict[str, set[str]] = {}
        self._documents: dict[str, frozenset[str]] = {}

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._documents

    def add(self, doc_id: str, text: str) -> None:
        """Index ``text`` under ``doc_id`` (a changed text replaces the old one)."""
        self.remove(doc_id)
        found = tokens_of(text)
        self._documents[doc_id] = found
        for token in found:
            self._postings.setdefault(token, set()).add(doc_id)

    def remove(self, doc_id: str) -> None:
        old = self._documents.pop(doc_id, None)
        if old is None:
            return
        for token in old:
            posting = self._postings.get(token)
            if posting is None:
                continue
            posting.discard(doc_id)
            if not posting:
                del self._postings[token]

    def clear(self) -> None:
        self._postings.clear()
        self._documents.clear()

    def _weight(self, token: str) -> float:
        """Inverse document frequency (always positive)."""
        frequency = len(self._postings.get(token, ()))
        return math.log(1.0 + len(self._documents) / (1.0 + frequency))

    def search(
        self, query: str, limit: int, *, allowed: Callable[[str], bool] | None = None
    ) -> list[KeywordHit]:
        """The best ``limit`` documents for ``query``; ``allowed`` drops documents up front.

        ``allowed`` is how the as-of view keeps documents of the future out *before* the limit
        is applied, so a limit never leaves the view with fewer hits than there are.
        """
        query_tokens = tokens_of(query)
        wanted = [t for t in query_tokens if t in self._postings]
        if limit <= 0 or not wanted:
            return []
        # a query token that no document has still counts against the share, as it would for a
        # document that lacks it
        total = sum(self._weight(t) for t in query_tokens)
        scores: dict[str, float] = {}
        for token in wanted:
            weight = self._weight(token)
            for doc_id in self._postings[token]:
                scores[doc_id] = scores.get(doc_id, 0.0) + weight
        hits = [
            KeywordHit(doc_id, min(1.0, score / total))
            for doc_id, score in scores.items()
            if allowed is None or allowed(doc_id)
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return hits[:limit]
