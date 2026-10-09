"""A tiny offline embedding model for the retrieval tests, and a builder of spoken dialogues.

``HashingBackend`` implements :class:`twin.retrieval.embedder.EmbeddingBackend` with hashed
character unigrams and bigrams: texts that share words get similar vectors, which is all the
tests need to check ranking behaviour, and it runs anywhere in milliseconds.  It lives here, in
``tests/``, and nowhere in ``src/``: the real model is exercised by the ``live`` integration test.

``write_dialogue`` stores hand-written conversations (with chosen clock times) in the database so
that the windows, the hold-out and the ranking can be asserted exactly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np

from tests.support.synth_chat import MessageWriter
from twin.retrieval.embedder import EmbedderInfo, Vectors
from twin.services import Services

DIMENSION = 512
STOP = frozenset(
    "你我他她了的吗呢吧啊：:，。！？,.!?"
)  # speaker labels and particles carry no topic


class HashingBackend:
    """Hashed character n-grams, length-1 vectors, deterministic."""

    def __init__(
        self,
        dimension: int = DIMENSION,
        *,
        model: str = "test/hashing-bigram",
        weights: str = "tiny-weights-1",
    ) -> None:
        self._info = EmbedderInfo(
            model=model,
            dimension=dimension,
            revision="test-revision",
            weights_sha256=hashlib.sha256(weights.encode()).hexdigest(),
            device="cpu",
        )
        self.seen: list[str] = []  # every text the model was asked to encode (already redacted)
        self.runs = 0

    @property
    def info(self) -> EmbedderInfo:
        return self._info

    def _features(self, text: str) -> list[str]:
        chars = [c for c in text if not c.isspace() and c not in STOP]
        return [*chars, *(a + b for a, b in pairwise(chars))]

    def encode(self, texts: Sequence[str], *, batch_size: int) -> Vectors:
        self.runs += 1
        self.seen.extend(texts)
        out = np.zeros((len(texts), self._info.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.md5(feature.encode("utf-8"), usedforsecurity=False).digest()
                out[row, int.from_bytes(digest[:4], "big") % self._info.dimension] += 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm:
                out[row] /= norm
        return out


@dataclass(frozen=True)
class Msg:
    """One message of a hand-written dialogue."""

    who: str  # "u" the user, "h" her
    text: str | None = None
    kind: str = "text"
    md5: str | None = None
    gap: float | None = None  # seconds after the previous message (default by speaker change)


def U(text: str | None = None, kind: str = "text", gap: float | None = None) -> Msg:
    return Msg("u", text, kind, None, gap)


def H(text: str | None = None, kind: str = "text", md5: str | None = None) -> Msg:
    return Msg("h", text, kind, md5)


SPEAKER_CHANGE_S = 40.0
SAME_SPEAKER_S = 3.0


def write_dialogue(
    services: Services, episodes: Sequence[tuple[datetime, Sequence[Msg]]], *, append: bool = False
) -> list[datetime]:
    """Store episodes ``(start time, messages)``; returns the time of each episode's first message.

    Messages of one episode follow each other by 3 s (same speaker) or 40 s (speaker change),
    so each run of one speaker is one burst; give episodes at least two hours apart to keep them
    separate segments.
    """
    writer = MessageWriter(services)
    starts = []
    for start, messages in episodes:
        starts.append(start)
        moment = start
        previous: Msg | None = None
        for message in messages:
            if previous is not None:
                step = message.gap
                if step is None:
                    step = SAME_SPEAKER_S if message.who == previous.who else SPEAKER_CHANGE_S
                moment += timedelta(seconds=step)
            writer.add(moment, message.who == "h", message.kind, message.text, message.md5)
            previous = message
    writer.store(append=append)
    return starts


def day(n: int, hour: int = 12, minute: int = 0) -> datetime:
    """The ``n``-th day after 2026-03-01 at ``hour``:``minute`` UTC (a fixed, tidy calendar)."""
    return datetime(2026, 3, 1, hour, minute, tzinfo=UTC) + timedelta(days=n)
