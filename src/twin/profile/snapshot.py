"""The JSON form of a profile version and typed access to it (R-PROF-003, R-PROF-004).

``metrics`` of a ``profile_versions`` row::

    {
      "schema": 1, "scope": "live" | "pre_holdout",
      "config": {burst_gap_s, segment_gap_min, recent_days, recency_weight, ...},
      "windows": {"full": {...}, "recent": {...}},       # range, days, message counts
      "parties": {"her": {<metric>: {"full": leaf, "recent": leaf, "blended": leaf}},
                  "user": {...}}
    }

Every metric keeps its raw values (the full window and the most recent ``recent_days``) and the
blended value (``recency_weight`` on the recent window).  Leaves are tagged values
(:mod:`twin.profile.values`).  Message text is never part of this document.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.profile.values import (
    Dist,
    Hourly,
    Leaf,
    Rates,
    Scalar,
    Table,
    blend_leaf,
    leaf_from_json,
)

SCHEMA = 1
PARTIES = ("her", "user")
WINDOWS = ("full", "recent", "blended")


def assemble_metrics(
    *,
    scope: str,
    config: Mapping[str, Any],
    weight: float,
    full: Mapping[str, Mapping[str, Leaf]],
    recent: Mapping[str, Mapping[str, Leaf]] | None,
    full_info: Mapping[str, Any],
    recent_info: Mapping[str, Any],
) -> dict[str, Any]:
    """The ``metrics`` document from the leaves of the two windows."""
    parties: dict[str, Any] = {}
    for party in PARTIES:
        metrics: dict[str, Any] = {}
        for name, leaf in full[party].items():
            other = recent[party].get(name) if recent is not None else None
            entry: dict[str, Any] = {"full": leaf.to_json()}
            if not isinstance(leaf, Table):
                if other is not None:
                    entry["recent"] = other.to_json()
                entry["blended"] = blend_leaf(leaf, other, weight).to_json()
            metrics[name] = entry
        parties[party] = metrics
    return {
        "schema": SCHEMA,
        "scope": scope,
        "config": dict(config),
        "windows": {"full": dict(full_info), "recent": dict(recent_info)},
        "parties": parties,
    }


class ProfileMetrics:
    """Typed, read-only access to a stored ``metrics`` document."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported profile schema {data.get('schema')!r}")
        self.data = data

    @property
    def scope(self) -> str:
        return str(self.data["scope"])

    def window_info(self, window: str) -> Mapping[str, Any]:
        key = "recent" if window == "recent" else "full"
        return dict(self.data["windows"][key])

    def names(self, party: str) -> list[str]:
        return sorted(self.data["parties"][party])

    def leaf(self, party: str, name: str, window: str = "blended") -> Leaf | None:
        entry = self.data["parties"][party].get(name)
        if entry is None:
            return None
        raw = entry.get(window) or entry.get("blended") or entry.get("full")
        return leaf_from_json(raw) if raw is not None else None

    def scalar(self, party: str, name: str, window: str = "blended") -> float | None:
        found = self.leaf(party, name, window)
        return found.value if isinstance(found, Scalar) and found.n > 0 else None

    def rates(self, party: str, name: str, window: str = "blended") -> Mapping[str, float]:
        found = self.leaf(party, name, window)
        return found.values if isinstance(found, Rates) else {}

    def distribution(
        self, party: str, name: str, window: str = "blended"
    ) -> EmpiricalDistribution | None:
        """A sampleable distribution; ``None`` if the metric is missing or has no samples."""
        found = self.leaf(party, name, window)
        if isinstance(found, Dist) and not found.dist.is_empty:
            return found.dist
        return None

    def hourly(self, party: str, name: str, window: str = "blended") -> BucketedDistribution | None:
        found = self.leaf(party, name, window)
        return found.dist if isinstance(found, Hourly) else None

    def table(self, party: str, name: str) -> Mapping[str, Mapping[str, Any]]:
        found = self.leaf(party, name, "full")
        return found.entries if isinstance(found, Table) else {}
