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

A search with a filter (``allowed``, how the as-of view hides what was not yet known) counts the
document frequencies among the documents the filter lets through only (D-783): otherwise facts of
the future would change the weights, and with them the hits and the block of a past moment
(iron rule 13).  The filter is asked once per document and search: a search costs one pass over
the documents plus the postings of the query's tokens, never documents times tokens.
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

    def search(
        self, query: str, limit: int, *, allowed: Callable[[str], bool] | None = None
    ) -> list[KeywordHit]:
        """The best ``limit`` documents for ``query``; ``allowed`` drops documents up front.

        ``allowed`` is how the as-of view keeps documents of the future out *before* the limit
        is applied, so a limit never leaves the view with fewer hits than there are - and before
        the weights are counted, so they never see them either (D-783).
        """
        query_tokens = tokens_of(query)
        if limit <= 0 or not any(t in self._postings for t in query_tokens):
            return []
        population, postings = self._population(query_tokens, allowed)
        if not postings:
            return []
        weights = {t: _idf(population, len(postings.get(t, ()))) for t in query_tokens}
        # a query token that no document has still counts against the share, as it would for a
        # document that lacks it
        total = sum(weights.values())
        scores: dict[str, float] = {}
        for token, documents in postings.items():
            for doc_id in documents:
                scores[doc_id] = scores.get(doc_id, 0.0) + weights[token]
        hits = [KeywordHit(doc_id, min(1.0, score / total)) for doc_id, score in scores.items()]
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return hits[:limit]

    def _population(
        self, tokens: frozenset[str], allowed: Callable[[str], bool] | None
    ) -> tuple[int, dict[str, set[str]]]:
        """How many documents count, and the postings of ``tokens`` among them."""
        if allowed is None:
            return len(self._documents), {
                t: self._postings[t] for t in tokens if t in self._postings
            }
        visible = {doc_id for doc_id in self._documents if allowed(doc_id)}
        found: dict[str, set[str]] = {}
        for token in tokens:
            documents = self._postings.get(token, set()) & visible
            if documents:
                found[token] = documents
        return len(visible), found


def _idf(population: int, frequency: int) -> float:
    """Inverse document frequency (always positive)."""
    return math.log(1.0 + population / (1.0 + frequency))
