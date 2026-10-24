"""Prompt layout for the context cache (R-LLM-010).

DeepSeek caches by prefix, so every request is built as ``stable_prefix`` followed by
``variable_tail``:

* ``stable_prefix`` - what repeats verbatim from call to call: the rules, the persona card
  and the recent conversation window.  The window only grows at its end and moves forward in
  batches (R-MEM-001) so the prefix of request *n+1* starts with the prefix of request *n*;
* ``variable_tail`` - what changes every time: retrieved examples, memory snippets, the
  current time and the newest user message (R-ENG-005).

:class:`PromptLayout` keeps the two parts apart, hashes the stable prefix and can say whether
a new layout extends an earlier one.  :class:`CacheMonitor` compares the cache hits that the
API reports with the length of the stable prefix, so a cache that stops working (a stray
timestamp in the prefix, say) shows up in the statistics.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import orjson

from twin.llm.tokens import TokenEstimator
from twin.llm.types import ChatMessage, Usage

MAX_OBSERVATIONS = 200


def _canonical(messages: Sequence[ChatMessage]) -> bytes:
    return orjson.dumps(list(messages), option=orjson.OPT_SORT_KEYS)


@dataclass(frozen=True)
class PromptLayout:
    """A request split into its cacheable prefix and its changing tail."""

    stable_prefix: tuple[ChatMessage, ...]
    variable_tail: tuple[ChatMessage, ...]

    @classmethod
    def of(
        cls, stable_prefix: Sequence[ChatMessage], variable_tail: Sequence[ChatMessage]
    ) -> PromptLayout:
        return cls(tuple(stable_prefix), tuple(variable_tail))

    def messages(self) -> list[ChatMessage]:
        """The full message list in send order."""
        return [*self.stable_prefix, *self.variable_tail]

    @property
    def prefix_hash(self) -> str:
        """SHA-256 of the stable prefix; equal hashes mean an identical cacheable prefix."""
        return hashlib.sha256(_canonical(self.stable_prefix)).hexdigest()

    def prefix_tokens(self, estimator: TokenEstimator) -> int:
        return estimator.estimate_messages(self.stable_prefix)

    def extends(self, previous: PromptLayout) -> bool:
        """Whether this layout's stable prefix begins with ``previous``'s stable prefix.

        True for the normal case where the conversation window grew at its end; False when
        the window moved forward in a batch or something in the prefix changed.
        """
        count = len(previous.stable_prefix)
        return (
            len(self.stable_prefix) >= count
            and self.stable_prefix[:count] == previous.stable_prefix
        )


@dataclass(frozen=True)
class CacheObservation:
    """What one call tells about the cache."""

    purpose: str
    prefix_hash: str
    prefix_tokens: int
    hit_tokens: int
    miss_tokens: int
    prefix_changed: bool

    @property
    def hit_ratio(self) -> float:
        """Cache hits as a share of the whole prompt."""
        prompt = self.hit_tokens + self.miss_tokens
        return self.hit_tokens / prompt if prompt else 0.0

    @property
    def prefix_hit_ratio(self) -> float:
        """Share of the stable prefix that was served from the cache (capped at 1)."""
        if self.prefix_tokens <= 0:
            return 0.0
        return min(1.0, self.hit_tokens / self.prefix_tokens)


@dataclass(frozen=True)
class CacheSummary:
    calls: int
    hit_tokens: int
    miss_tokens: int
    prefix_changes: int
    mean_prefix_hit_ratio: float

    @property
    def hit_ratio(self) -> float:
        prompt = self.hit_tokens + self.miss_tokens
        return self.hit_tokens / prompt if prompt else 0.0


class CacheMonitor:
    """Rolling cache statistics per purpose, fed after every call."""

    def __init__(self, estimator: TokenEstimator, *, window: int = MAX_OBSERVATIONS) -> None:
        self._estimator = estimator
        self._window = window
        self._observations: dict[str, deque[CacheObservation]] = {}
        self._last_hash: dict[str, str] = {}

    def observe(self, layout: PromptLayout, usage: Usage, *, purpose: str) -> CacheObservation:
        """Record one call.  ``prefix_changed`` is true when the prefix differs from the last
        call of the same purpose (expected when the conversation window moves forward)."""
        prefix_hash = layout.prefix_hash
        previous = self._last_hash.get(purpose)
        observation = CacheObservation(
            purpose=purpose,
            prefix_hash=prefix_hash,
            prefix_tokens=layout.prefix_tokens(self._estimator),
            hit_tokens=usage.cache_hit_tokens,
            miss_tokens=usage.cache_miss_tokens,
            prefix_changed=previous is not None and previous != prefix_hash,
        )
        self._last_hash[purpose] = prefix_hash
        self._observations.setdefault(purpose, deque(maxlen=self._window)).append(observation)
        return observation

    def summary(self, purpose: str | None = None) -> CacheSummary:
        """Statistics over the retained observations of one purpose (or all of them)."""
        if purpose is not None:
            observed = list(self._observations.get(purpose, ()))
        else:
            observed = [o for series in self._observations.values() for o in series]
        if not observed:
            return CacheSummary(0, 0, 0, 0, 0.0)
        return CacheSummary(
            calls=len(observed),
            hit_tokens=sum(o.hit_tokens for o in observed),
            miss_tokens=sum(o.miss_tokens for o in observed),
            prefix_changes=sum(1 for o in observed if o.prefix_changed),
            mean_prefix_hit_ratio=sum(o.prefix_hit_ratio for o in observed) / len(observed),
        )
