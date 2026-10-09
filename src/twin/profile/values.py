"""The kinds of values a profile metric can have, and how two windows are blended (R-PROF-003).

``Scalar``
    one number with the size of the sample it was computed from (a rate, a median ...);
``Rates``
    named fractions over a closed vocabulary (punctuation groups, emoji codes, hours);
``Dist`` / ``Hourly``
    sampleable empirical distributions (:mod:`twin.profile.distribution`);
``Table``
    a keyed table that is not blended (per-sticker use counts and last use).

Every leaf serialises to JSON with a ``t`` tag.  Blending two windows (all data and the most
recent ``profile.recent_days``) with the recency weight ``w``::

    blended = w * recent + (1 - w) * full

for numbers, and the mixture of the two distributions for distributions.  A recent window with
fewer samples than the leaf's ``min_n`` is too thin to trust: the blended value is then the
full-window value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution

DEFAULT_MIN_RECENT = 30
MAX_RATE_ENTRIES = 200


@dataclass(frozen=True)
class Scalar:
    value: float
    n: int
    min_n: int = DEFAULT_MIN_RECENT

    def to_json(self) -> dict[str, Any]:
        return {"t": "s", "v": round(self.value, 6), "n": self.n, "m": self.min_n}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Scalar:
        return cls(float(data["v"]), int(data["n"]), int(data.get("m", DEFAULT_MIN_RECENT)))


@dataclass(frozen=True)
class Rates:
    """Named fractions; ``n`` is the number of events they are fractions of."""

    values: Mapping[str, float]
    n: int
    min_n: int = DEFAULT_MIN_RECENT

    def top(self, count: int) -> list[tuple[str, float]]:
        ranked = sorted(self.values.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:count]

    def to_json(self) -> dict[str, Any]:
        items = dict(self.top(MAX_RATE_ENTRIES))
        return {
            "t": "r",
            "n": self.n,
            "m": self.min_n,
            "items": {k: round(v, 6) for k, v in sorted(items.items())},
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Rates:
        return cls(
            {str(k): float(v) for k, v in data["items"].items()},
            int(data["n"]),
            int(data.get("m", DEFAULT_MIN_RECENT)),
        )


@dataclass(frozen=True)
class Dist:
    dist: EmpiricalDistribution
    min_n: int = DEFAULT_MIN_RECENT

    @property
    def n(self) -> int:
        return self.dist.n

    def to_json(self) -> dict[str, Any]:
        return {"t": "d", "m": self.min_n, **self.dist.to_json()}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Dist:
        return cls(EmpiricalDistribution.from_json(data), int(data.get("m", DEFAULT_MIN_RECENT)))


@dataclass(frozen=True)
class Hourly:
    """Distributions per bucket (the local hour) with fallback to "all day"."""

    dist: BucketedDistribution
    min_n: int = DEFAULT_MIN_RECENT

    @property
    def n(self) -> int:
        return self.dist.overall.n

    def to_json(self) -> dict[str, Any]:
        return {"t": "h", "m": self.min_n, **self.dist.to_json()}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Hourly:
        return cls(BucketedDistribution.from_json(data), int(data.get("m", DEFAULT_MIN_RECENT)))


@dataclass(frozen=True)
class Table:
    """A keyed table (not blended: the full window is used)."""

    entries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    n: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"t": "k", "n": self.n, "entries": {k: dict(v) for k, v in self.entries.items()}}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Table:
        return cls({str(k): dict(v) for k, v in data["entries"].items()}, int(data["n"]))


Leaf = Scalar | Rates | Dist | Hourly | Table


def leaf_to_json(leaf: Leaf) -> dict[str, Any]:
    return leaf.to_json()


def leaf_from_json(data: Mapping[str, Any]) -> Leaf:
    tag = data["t"]
    if tag == "s":
        return Scalar.from_json(data)
    if tag == "r":
        return Rates.from_json(data)
    if tag == "d":
        return Dist.from_json(data)
    if tag == "h":
        return Hourly.from_json(data)
    if tag == "k":
        return Table.from_json(data)
    raise ValueError(f"unknown metric value tag {tag!r}")


def _usable(recent_n: int, min_n: int, weight: float) -> bool:
    return weight > 0 and recent_n >= min_n


def blend_leaf(full: Leaf, recent: Leaf | None, weight: float) -> Leaf:
    """The blended value of one metric (see the module docstring)."""
    if recent is None or type(recent) is not type(full):
        return full
    if isinstance(full, Scalar) and isinstance(recent, Scalar):
        if not _usable(recent.n, full.min_n, weight):
            return full
        return Scalar(weight * recent.value + (1 - weight) * full.value, full.n, full.min_n)
    if isinstance(full, Rates) and isinstance(recent, Rates):
        if not _usable(recent.n, full.min_n, weight):
            return full
        keys = set(full.values) | set(recent.values)
        mixed = {
            k: weight * recent.values.get(k, 0.0) + (1 - weight) * full.values.get(k, 0.0)
            for k in keys
        }
        return Rates(dict(Rates(mixed, full.n).top(MAX_RATE_ENTRIES)), full.n, full.min_n)
    if isinstance(full, Dist) and isinstance(recent, Dist):
        if not _usable(recent.n, full.min_n, weight):
            return full
        return Dist(
            EmpiricalDistribution.mix([(recent.dist, weight), (full.dist, 1 - weight)]),
            full.min_n,
        )
    if isinstance(full, Hourly) and isinstance(recent, Hourly):
        if not _usable(recent.n, full.min_n, weight):
            return full
        return Hourly(
            BucketedDistribution.mix([(recent.dist, weight), (full.dist, 1 - weight)]),
            full.min_n,
        )
    return full
