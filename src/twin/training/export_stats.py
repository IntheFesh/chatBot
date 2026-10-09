"""The statistics of an exported training set (R-TRN-002, R-TRN-005, R-TRN-006).

Collected while the samples are built and summarised when the splits are known: how many samples
went where and why others were dropped, how long the conversations and the targets are, how often
her stickers and emoji codes appear in the targets next to how often they appear in her profile
(the numbers should be close; if not, the export is not teaching her way of writing), how many
samples carry a plan, how many tokens there are, and about how long each GPU profile would train
on them.

The training times are **planning figures**: the tokens per second of each profile are rough
values for LoRA with gradient checkpointing and 2,048-token samples, not measurements of this
project.  The first real run (``training_runs`` records the time of each step) replaces them.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from twin.training.profiles import PROFILES, epochs_for

# rough training speed in tokens per second of each profile (see the module description)
TOKENS_PER_SECOND: Final = {
    "5090-8b": 5000,
    "5090-14b": 1800,
    "pro6000-14b": 2800,
    "pro6000-32b": 1300,
}
EVAL_OVERHEAD: Final = 1.15  # evaluation and checkpointing next to the training steps
PERCENTILES: Final = (50, 90, 99)


def percentile(values: Sequence[float], p: float) -> float:
    """The ``p``-th percentile (nearest rank) of ``values``; 0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return float(ordered[rank - 1])


def distribution(values: Sequence[float]) -> dict[str, float]:
    """Mean, percentiles and maximum of ``values``."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": round(sum(values) / len(values), 2),
        **{f"p{p}": percentile(values, p) for p in PERCENTILES},
        "max": float(max(values)),
    }


def training_hours(train_tokens: int, train_samples: int) -> dict[str, float]:
    """About how many hours each profile trains on this set (planning figures)."""
    epochs = epochs_for(train_samples)
    return {
        name: round(train_tokens * epochs * EVAL_OVERHEAD / TOKENS_PER_SECOND[name] / 3600, 2)
        for name in PROFILES
    }


@dataclass(frozen=True)
class SampleFacts:
    """What the statistics need to know about one finished sample."""

    pre_holdout: bool  # False: a sample of the test part
    planned: bool
    plan_selected: bool
    turns: int  # messages of the conversation part (her and the user's turns)
    prelude: bool
    trimmed: bool
    target_lines: int
    target_chars: int
    stickers: int
    text_lines: int
    code_lines: int
    quoted: bool
    tokens: int


@dataclass
class StatsCollector:
    """Accumulates the facts of the samples in the order they are made."""

    dropped: Counter[str] = field(default_factory=Counter)
    target_dropped: Counter[str] = field(default_factory=Counter)
    pre: list[SampleFacts] = field(default_factory=list)
    test: list[SampleFacts] = field(default_factory=list)

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1

    def add(self, facts: SampleFacts) -> None:
        (self.pre if facts.pre_holdout else self.test).append(facts)

    def summarise(
        self,
        *,
        val_count: int,
        profile_sticker_share: float | None,
        profile_code_rate: float | None,
    ) -> dict[str, Any]:
        """The report: ``val_count`` samples at the end of the pre-holdout part are validation."""
        train = self.pre[: len(self.pre) - val_count]
        val = self.pre[len(self.pre) - val_count :]
        everything = [*self.pre, *self.test]
        lines = sum(f.target_lines for f in everything)
        text_lines = sum(f.text_lines for f in everything)
        selected = [f for f in self.pre if f.plan_selected]
        planned = sum(1 for f in self.pre if f.planned)
        train_tokens = sum(f.tokens for f in train)
        return {
            "samples": {
                "train": len(train),
                "val": len(val),
                "test": len(self.test),
                "total": len(everything),
            },
            "dropped": dict(sorted(self.dropped.items())),
            "target_lines_dropped": dict(sorted(self.target_dropped.items())),
            "turns_per_sample": distribution([f.turns for f in everything]),
            "target_chars": distribution([f.target_chars for f in everything]),
            "target_lines": distribution([f.target_lines for f in everything]),
            "sticker_share": {
                "export": round(sum(f.stickers for f in everything) / lines, 4) if lines else 0.0,
                "profile": profile_sticker_share,
            },
            "emoji_code_rate": {
                "export": round(sum(f.code_lines for f in everything) / text_lines, 4)
                if text_lines
                else 0.0,
                "profile": profile_code_rate,
            },
            "quoted_samples": sum(1 for f in everything if f.quoted),
            "prelude_samples": sum(1 for f in everything if f.prelude),
            "trimmed_samples": sum(1 for f in everything if f.trimmed),
            "plans": {
                "selected": len(selected),
                "planned": planned,
                "share_of_train_and_val": round(planned / len(self.pre), 4) if self.pre else 0.0,
                "share_of_all": round(planned / len(everything), 4) if everything else 0.0,
            },
            "tokens": {
                "total": sum(f.tokens for f in everything),
                "train": train_tokens,
                "per_sample": distribution([f.tokens for f in everything]),
            },
            "training_hours": training_hours(train_tokens, len(train)),
            "epochs": epochs_for(len(train)),
        }
