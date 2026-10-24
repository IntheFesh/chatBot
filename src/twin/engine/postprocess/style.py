"""The numbers of her style that the post-processing holds a reply to (R-ENG-008).

All of it comes from her profile (:mod:`twin.profile.api`):

========================  ================================================================
``max_chars``             a bubble may be at most 1.5 times her 95th percentile of characters
``max_bubbles``           at most 1.5 times her 90th percentile of bubbles in a burst, and
                          never more than ``engine.max_bubbles``
``comma_rate``            how many of her messages have a comma - below 0.25 she writes
                          without them and the reply is split into bubbles instead
``period_end_rate``       how many end with a full stop - below 0.25 the stop is taken off
``median_chars``          her median length (a comma sentence shorter than twice that stays)
``max_code_run``          the longest run of emoji codes she writes (her 90th percentile)
========================  ================================================================

Without a profile there is nothing to hold the reply to: the limits fall back to a generous
default bubble length and the configured number of bubbles, and no punctuation is changed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from twin.profile.api import ProfileSnapshot
from twin.profile.values import Rates

LENGTH_FACTOR = 1.5
BUBBLE_FACTOR = 1.5
DEFAULT_MAX_CHARS = 60
PERIOD_STRIP_BELOW = 0.25
COMMA_SPLIT_BELOW = 0.25
COMMA_ALWAYS_SPLIT_BELOW = 0.10
MIN_CHARS = 8


def _rates(profile: ProfileSnapshot, name: str) -> Mapping[str, float]:
    """The rates of her metric ``name``; none while there is no data behind them."""
    leaf = profile.metrics.leaf("her", name)
    return leaf.values if isinstance(leaf, Rates) and leaf.n > 0 else {}


@dataclass(frozen=True)
class StyleLimits:
    """The numbers one reply is held to."""

    max_chars: int = DEFAULT_MAX_CHARS
    max_bubbles: int = 8
    comma_rate: float | None = None
    period_end_rate: float | None = None
    median_chars: float | None = None
    max_code_run: int | None = None

    @property
    def strips_periods(self) -> bool:
        return self.period_end_rate is not None and self.period_end_rate < PERIOD_STRIP_BELOW

    @property
    def splits_commas(self) -> bool:
        return self.comma_rate is not None and self.comma_rate < COMMA_SPLIT_BELOW

    def comma_split_threshold(self) -> int:
        """Lines up to this length keep their commas: 0 means every comma sentence is split."""
        if self.comma_rate is None or self.comma_rate < COMMA_ALWAYS_SPLIT_BELOW:
            return 0
        return math.ceil(2 * (self.median_chars or DEFAULT_MAX_CHARS / 6))

    @classmethod
    def from_profile(cls, profile: ProfileSnapshot | None, *, bubble_cap: int) -> StyleLimits:
        """The limits of her profile (``bubble_cap`` is ``engine.max_bubbles``)."""
        if profile is None:
            return cls(max_bubbles=max(1, bubble_cap))
        metrics = profile.metrics
        lengths = metrics.distribution("her", "text_length")
        bursts = metrics.distribution("her", "burst_size")
        runs = metrics.distribution("her", "emoji_code_run_length")
        punct = _rates(profile, "punct_rate")
        ends = _rates(profile, "end_rate")
        max_chars = DEFAULT_MAX_CHARS
        median = None
        if lengths is not None:
            max_chars = max(MIN_CHARS, math.ceil(LENGTH_FACTOR * lengths.quantile(0.95)))
            median = lengths.median()
        max_bubbles = bubble_cap
        if bursts is not None:
            max_bubbles = min(bubble_cap, math.ceil(BUBBLE_FACTOR * bursts.quantile(0.9)))
        return cls(
            max_chars=max_chars,
            max_bubbles=max(1, max_bubbles),
            comma_rate=punct.get("comma"),
            period_end_rate=ends.get("period"),
            median_chars=median,
            max_code_run=max(1, round(runs.quantile(0.9))) if runs is not None else None,
        )
