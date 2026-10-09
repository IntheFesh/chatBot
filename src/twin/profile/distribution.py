"""Empirical distributions that can be stored as JSON and sampled (R-PROF-003, R-ACT-002).

:class:`EmpiricalDistribution` keeps the quantile function of a sample at a fixed grid of
probabilities (:data:`GRID`: every percent plus 99.5 % and 99.9 %), the sample size and the
exact mean.  Sampling is the inverse CDF with linear interpolation between grid points, with
optional truncation to ``[lo, hi]`` (the draw is restricted to the probability mass inside the
interval).  A ``discrete`` distribution (lengths, counts) returns whole numbers.

:class:`BucketedDistribution` holds one distribution per bucket (for example per hour of the
day) and falls back to a coarser level when a bucket has too few samples: exact bucket, then
groups of neighbouring buckets, then the distribution of everything ("全天").
"""

from __future__ import annotations

import random
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

GRID: tuple[float, ...] = tuple(sorted({i / 100 for i in range(101)} | {0.995, 0.999}))
_GRID_LAST = len(GRID) - 1
_MIX_ITERATIONS = 40


class EmptyDistributionError(ValueError):
    """A distribution without samples cannot be sampled."""


def _clamp(value: float, lo: float | None, hi: float | None) -> float:
    if lo is not None and value < lo:
        return lo
    if hi is not None and value > hi:
        return hi
    return value


@dataclass(frozen=True)
class EmpiricalDistribution:
    """Quantile summary of a sample (see the module docstring)."""

    n: int
    points: tuple[float, ...]
    mean_value: float
    discrete: bool = False

    # ------------------------------------------------------------ construction

    @classmethod
    def empty(cls, *, discrete: bool = False) -> EmpiricalDistribution:
        return cls(0, (0.0,) * len(GRID), 0.0, discrete)

    @classmethod
    def from_counter(
        cls, counts: Mapping[float, int], *, discrete: bool = False
    ) -> EmpiricalDistribution:
        """Build from ``{value: how many times}`` (weights must be positive integers)."""
        items = sorted((float(v), int(c)) for v, c in counts.items() if c > 0)
        total = sum(c for _, c in items)
        if total == 0:
            return cls.empty(discrete=discrete)
        values = [v for v, _ in items]
        cumulative: list[int] = []
        running = 0
        for _, count in items:
            running += count
            cumulative.append(running)

        def order_statistic(index: int) -> float:
            return values[bisect_right(cumulative, index)]

        points = []
        for probability in GRID:
            position = (total - 1) * probability
            lower = int(position)
            fraction = position - lower
            low_value = order_statistic(lower)
            if fraction == 0.0 or lower + 1 >= total:
                points.append(low_value)
            else:
                high_value = order_statistic(lower + 1)
                points.append(low_value + fraction * (high_value - low_value))
        mean = sum(v * c for v, c in items) / total
        return cls(total, tuple(points), mean, discrete)

    @classmethod
    def from_samples(
        cls, samples: Sequence[float], *, discrete: bool = False
    ) -> EmpiricalDistribution:
        counts: dict[float, int] = {}
        for value in samples:
            counts[float(value)] = counts.get(float(value), 0) + 1
        return cls.from_counter(counts, discrete=discrete)

    @classmethod
    def mix(cls, parts: Sequence[tuple[EmpiricalDistribution, float]]) -> EmpiricalDistribution:
        """The mixture ``sum(w_i * F_i)``; parts without samples or weight are ignored."""
        live = [(d, w) for d, w in parts if d.n > 0 and w > 0]
        if not live:
            return cls.empty(discrete=all(d.discrete for d, _ in parts) if parts else False)
        if len(live) == 1:
            return live[0][0]
        total_weight = sum(w for _, w in live)
        weights = [w / total_weight for _, w in live]
        dists = [d for d, _ in live]
        discrete = all(d.discrete for d in dists)
        count = round(sum(w * d.n for w, d in zip(weights, dists, strict=True)))
        if all(d.points == dists[0].points for d in dists[1:]):  # the same shape: nothing to mix
            return replace(dists[0], n=count, mean_value=dists[0].mean_value)
        low = min(d.points[0] for d in dists)
        high = max(d.points[-1] for d in dists)

        def cdf(x: float) -> float:
            return sum(w * d.cdf(x) for w, d in zip(weights, dists, strict=True))

        points: list[float] = []
        for probability in GRID:
            if probability <= 0.0:
                points.append(low)
                continue
            if probability >= 1.0:
                points.append(high)
                continue
            left, right = low, high
            for _ in range(_MIX_ITERATIONS):
                middle = (left + right) / 2
                if cdf(middle) >= probability:
                    right = middle
                else:
                    left = middle
            points.append(right)
        for index in range(1, len(points)):  # numerical safety: keep the function monotone
            points[index] = max(points[index], points[index - 1])
        mean = sum(w * d.mean_value for w, d in zip(weights, dists, strict=True))
        if discrete:
            points = [float(round(p)) for p in points]
        return cls(count, tuple(points), mean, discrete)

    # ----------------------------------------------------------------- queries

    @property
    def is_empty(self) -> bool:
        return self.n == 0

    def mean(self) -> float:
        return self.mean_value

    def quantile(self, q: float) -> float:
        """The value below which a fraction ``q`` (0..1) of the sample lies."""
        if self.n == 0:
            raise EmptyDistributionError("the distribution has no samples")
        q = min(max(q, 0.0), 1.0)
        index = bisect_right(GRID, q) - 1
        if index >= _GRID_LAST:
            return self.points[-1]
        span = GRID[index + 1] - GRID[index]
        fraction = (q - GRID[index]) / span
        low, high = self.points[index], self.points[index + 1]
        return low + fraction * (high - low)

    def median(self) -> float:
        return self.quantile(0.5)

    def minimum(self) -> float:
        return self.points[0]

    def maximum(self) -> float:
        return self.points[-1]

    def cdf(self, x: float) -> float:
        """``P(X <= x)`` of the stored quantile function (right continuous)."""
        if self.n == 0:
            return 0.0
        if x < self.points[0]:
            return 0.0
        if x >= self.points[-1]:
            return 1.0
        j = bisect_right(self.points, x) - 1
        low, high = self.points[j], self.points[j + 1]
        return GRID[j] + (x - low) / (high - low) * (GRID[j + 1] - GRID[j])

    def _probability_below(self, x: float) -> float:
        """The smallest probability ``p`` with ``quantile(p) >= x``."""
        if x <= self.points[0]:
            return 0.0
        if x > self.points[-1]:
            return 1.0
        i = bisect_left(self.points, x)
        low, high = self.points[i - 1], self.points[i]
        return GRID[i - 1] + (x - low) / (high - low) * (GRID[i] - GRID[i - 1])

    def sample(
        self, rng: random.Random, *, lo: float | None = None, hi: float | None = None
    ) -> float:
        """One draw (inverse CDF), restricted to ``[lo, hi]`` when given."""
        if self.n == 0:
            raise EmptyDistributionError("the distribution has no samples")
        a = 0.0 if lo is None else self._probability_below(lo)
        b = 1.0 if hi is None else self.cdf(hi)
        if a >= b:  # no mass inside the interval: the nearest end of it
            value = self.quantile(min(max(a, 0.0), 1.0))
        else:
            value = self.quantile(a + (b - a) * rng.random())
        value = _clamp(value, lo, hi)
        if self.discrete:
            value = _clamp(float(round(value)), lo, hi)
        return value

    def samples(
        self, rng: random.Random, count: int, *, lo: float | None = None, hi: float | None = None
    ) -> list[float]:
        return [self.sample(rng, lo=lo, hi=hi) for _ in range(count)]

    # --------------------------------------------------------------- JSON form

    def to_json(self) -> dict[str, Any]:
        points: list[float | int] = [
            int(p) if self.discrete and float(p).is_integer() else round(p, 4) for p in self.points
        ]
        return {
            "n": self.n,
            "mean": round(self.mean_value, 6),
            "discrete": self.discrete,
            "q": points,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> EmpiricalDistribution:
        points = tuple(float(p) for p in data["q"])
        if len(points) != len(GRID):
            raise ValueError("stored distribution does not match the quantile grid")
        return cls(int(data["n"]), points, float(data["mean"]), bool(data.get("discrete", False)))


@dataclass(frozen=True)
class BucketedDistribution:
    """One distribution per bucket with fallback to coarser groups and finally to "all".

    ``sizes`` are the group widths of the fallback levels, finest first: ``(1, 4)`` means
    the bucket itself, then ``bucket // 4`` (for 15-minute slots: the hour), then everything.
    A level is used when its distribution has at least ``min_samples`` samples.
    """

    sizes: tuple[int, ...]
    levels: tuple[Mapping[int, EmpiricalDistribution], ...]
    overall: EmpiricalDistribution
    min_samples: int

    @classmethod
    def from_counters(
        cls,
        per_bucket: Mapping[int, Mapping[float, int]],
        *,
        sizes: Sequence[int] = (1,),
        min_samples: int = 30,
        discrete: bool = False,
    ) -> BucketedDistribution:
        if not sizes or any(size < 1 for size in sizes):
            raise ValueError("sizes must be positive group widths")
        levels: list[Mapping[int, EmpiricalDistribution]] = []
        for size in sizes:
            merged: dict[int, dict[float, int]] = {}
            for key, counter in per_bucket.items():
                target = merged.setdefault(key // size, {})
                for value, count in counter.items():
                    target[value] = target.get(value, 0) + count
            levels.append(
                {
                    key: EmpiricalDistribution.from_counter(counter, discrete=discrete)
                    for key, counter in merged.items()
                }
            )
        everything: dict[float, int] = {}
        for counter in per_bucket.values():
            for value, count in counter.items():
                everything[value] = everything.get(value, 0) + count
        return cls(
            tuple(sizes),
            tuple(levels),
            EmpiricalDistribution.from_counter(everything, discrete=discrete),
            min_samples,
        )

    def level_for(self, key: int) -> int:
        """Index of the level that answers for ``key`` (``len(sizes)`` means "all")."""
        for index, size in enumerate(self.sizes):
            found = self.levels[index].get(key // size)
            if found is not None and found.n >= self.min_samples:
                return index
        return len(self.sizes)

    def get(self, key: int) -> EmpiricalDistribution:
        level = self.level_for(key)
        if level == len(self.sizes):
            return self.overall
        return self.levels[level][key // self.sizes[level]]

    def samples_in(self, key: int) -> int:
        """Samples that fall into the exact bucket (before any fallback)."""
        found = self.levels[0].get(key // self.sizes[0])
        return found.n if found is not None else 0

    @classmethod
    def mix(cls, parts: Sequence[tuple[BucketedDistribution, float]]) -> BucketedDistribution:
        """Bucket-wise mixture of distributions that share the same layout."""
        live = [(b, w) for b, w in parts if b.overall.n > 0 and w > 0]
        if not live:
            return parts[0][0] if parts else cls((1,), ({},), EmpiricalDistribution.empty(), 30)
        first = live[0][0]
        levels: list[Mapping[int, EmpiricalDistribution]] = []
        for index in range(len(first.sizes)):
            keys = {key for b, _ in live for key in b.levels[index]}
            levels.append(
                {
                    key: EmpiricalDistribution.mix(
                        [(b.levels[index][key], w) for b, w in live if key in b.levels[index]]
                    )
                    for key in sorted(keys)
                }
            )
        overall = EmpiricalDistribution.mix([(b.overall, w) for b, w in live])
        return cls(first.sizes, tuple(levels), overall, first.min_samples)

    def to_json(self) -> dict[str, Any]:
        return {
            "sizes": list(self.sizes),
            "min_samples": self.min_samples,
            "levels": [
                {str(key): dist.to_json() for key, dist in sorted(level.items())}
                for level in self.levels
            ],
            "overall": self.overall.to_json(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BucketedDistribution:
        return cls(
            tuple(int(s) for s in data["sizes"]),
            tuple(
                {int(key): EmpiricalDistribution.from_json(value) for key, value in level.items()}
                for level in data["levels"]
            ),
            EmpiricalDistribution.from_json(data["overall"]),
            int(data["min_samples"]),
        )
