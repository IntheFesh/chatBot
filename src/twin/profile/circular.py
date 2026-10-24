"""Statistics of clock times, which live on a circle (R-ACT-003).

A time of day is a number of minutes in ``[0, 1440)``; 23:50 and 00:10 are 20 minutes apart,
not 1420.  The mean is the direction of the average of the unit vectors; the median is taken
on the deviations from that mean, which is robust against a few odd nights.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from twin.profile.distribution import EmpiricalDistribution

DAY_MINUTES = 1440.0
HALF_DAY = DAY_MINUTES / 2


def wrap(minute: float) -> float:
    """``minute`` brought into ``[0, 1440)``."""
    return minute % DAY_MINUTES


def signed_diff(a: float, b: float) -> float:
    """The shortest signed way from ``b`` to ``a`` on the clock, in ``[-720, 720)``."""
    return (a - b + HALF_DAY) % DAY_MINUTES - HALF_DAY


def circular_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no values")
    sin = sum(math.sin(v / DAY_MINUTES * math.tau) for v in values)
    cos = sum(math.cos(v / DAY_MINUTES * math.tau) for v in values)
    if abs(sin) < 1e-12 and abs(cos) < 1e-12:
        return wrap(values[0])
    return wrap(math.atan2(sin, cos) / math.tau * DAY_MINUTES)


def _median(sorted_values: Sequence[float]) -> float:
    n = len(sorted_values)
    middle = n // 2
    if n % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def circular_median(values: Sequence[float]) -> float:
    mean = circular_mean(values)
    deviations = sorted(signed_diff(v, mean) for v in values)
    return wrap(mean + _median(deviations))


@dataclass(frozen=True)
class CircularStats:
    """Summary of clock times: median, mean, robust spread and the distribution of deviations."""

    median: float
    mean: float
    std: float
    n: int
    offsets: EmpiricalDistribution  # minutes from the median; empty for a fixed time

    @classmethod
    def from_values(cls, values: Sequence[float]) -> CircularStats:
        if not values:
            raise ValueError("no values")
        median = circular_median(values)
        deviations = [signed_diff(v, median) for v in values]
        # a robust spread (1.4826 x the median absolute deviation, the standard deviation of a
        # normal distribution): one odd night must not swell the "fluctuation" shown to the user
        std = 1.4826 * _median(sorted(abs(d) for d in deviations))
        counts: dict[float, int] = {}
        for deviation in deviations:
            key = float(round(deviation))
            counts[key] = counts.get(key, 0) + 1
        return cls(
            median,
            circular_mean(values),
            std,
            len(values),
            EmpiricalDistribution.from_counter(counts, discrete=True),
        )

    @classmethod
    def fixed(cls, minute: float) -> CircularStats:
        """A time that is known exactly (a manual correction, a valley edge)."""
        return cls(wrap(minute), wrap(minute), 0.0, 0, EmpiricalDistribution.empty(discrete=True))

    def quantile(self, q: float) -> float:
        """The clock time at quantile ``q`` of the deviations around the median."""
        if self.offsets.is_empty:
            return self.median
        return wrap(self.median + self.offsets.quantile(q))

    def sample(self, rng: random.Random) -> float:
        """A random clock time distributed like the observed ones."""
        if self.offsets.is_empty:
            return self.median
        return wrap(self.median + self.offsets.sample(rng))

    def to_json(self) -> dict[str, Any]:
        return {
            "median": round(self.median, 3),
            "mean": round(self.mean, 3),
            "std": round(self.std, 3),
            "n": self.n,
            "offsets": self.offsets.to_json(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CircularStats:
        return cls(
            float(data["median"]),
            float(data["mean"]),
            float(data["std"]),
            int(data["n"]),
            EmpiricalDistribution.from_json(data["offsets"]),
        )
