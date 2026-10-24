"""The statistics of the blind test (R-EVAL-001, R-SRV-005): rates, intervals and tests.

Everything here is a closed formula with its source, so the numbers can be checked by hand and the
tests compare them with published values.

**Guess rate.**  ``correct / valid``, the point estimate the gates judge (M1 <= 70 %, M4 <= 60 %).
A skipped pair is not a judgement: it is in neither count.

**Wilson score interval** (Wilson, E. B., 1927, "Probable inference, the law of succession, and
statistical inference", JASA 22:209-212; the form below is the one tabulated by Newcombe, 1998,
Statistics in Medicine 17:857-872, method 3).  With ``p = x / n`` and ``z = 1.959964`` for 95 %::

    centre = (p + z^2 / 2n) / (1 + z^2 / n)
    half   = z * sqrt(p (1 - p) / n + z^2 / 4n^2) / (1 + z^2 / n)
    interval = (centre - half, centre + half)

**Two-proportion z-test** (pooled, no continuity correction; any statistics text, e.g. Agresti,
*Categorical Data Analysis*, section 3.2).  For ``x1 / n1`` against ``x2 / n2``::

    p_pool = (x1 + x2) / (n1 + n2)
    z      = (p1 - p2) / sqrt(p_pool (1 - p_pool) (1 / n1 + 1 / n2))
    two-sided p = 2 (1 - Phi(|z|))        one-sided p (p1 < p2) = Phi(z)

R-SRV-005 asks whether the new model's guess rate is *lower* than the DeepSeek backend's, so
``p_less`` (H1: p1 < p2) is the number round 14 uses with ``x1 / n1`` the new model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass

Z_95 = 1.959963984540054  # the 97.5 % point of the standard normal distribution


@dataclass(frozen=True)
class Rate:
    """``successes`` out of ``total`` with its Wilson 95 % interval."""

    successes: int
    total: int

    @property
    def point(self) -> float | None:
        """The point estimate, ``None`` without any observation."""
        return self.successes / self.total if self.total else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson_interval(self.successes, self.total)


@dataclass(frozen=True)
class ProportionTest:
    """The result of :func:`two_proportion_test` (``p1`` is the first sample's rate)."""

    z: float
    p_two_sided: float
    p_less: float  # H1: the first rate is lower than the second
    p_greater: float  # H1: the first rate is higher than the second
    p1: float
    p2: float


def normal_cdf(z: float) -> float:
    """The standard normal distribution function."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float] | None:
    """The Wilson score interval of ``successes / total``; ``None`` for an empty sample."""
    if total <= 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between 0 and total")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    # the interval always contains p; the clamps only remove floating-point noise at 0 and 1
    return max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half))


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> ProportionTest | None:
    """The pooled two-proportion z-test; ``None`` when a sample is empty."""
    if n1 <= 0 or n2 <= 0:
        return None
    if not (0 <= x1 <= n1 and 0 <= x2 <= n2):
        raise ValueError("successes must lie between 0 and total")
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    variance = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    if variance <= 0.0:  # everybody (or nobody) guessed right in both samples: no difference
        return ProportionTest(0.0, 1.0, 1.0, 1.0, p1, p2)
    z = (p1 - p2) / math.sqrt(variance)
    return ProportionTest(
        z=z,
        p_two_sided=min(1.0, 2.0 * (1.0 - normal_cdf(abs(z)))),
        p_less=normal_cdf(z),
        p_greater=1.0 - normal_cdf(z),
        p1=p1,
        p2=p2,
    )


def group_rates[T, K: Hashable](
    items: Iterable[T], key: Callable[[T], K | None], hit: Callable[[T], bool]
) -> dict[K, Rate]:
    """The rate of ``hit`` per group (items whose key is ``None`` belong to no group)."""
    counts: dict[K, list[int]] = {}
    for item in items:
        group = key(item)
        if group is None:
            continue
        pair = counts.setdefault(group, [0, 0])
        pair[0] += int(hit(item))
        pair[1] += 1
    return {group: Rate(pair[0], pair[1]) for group, pair in counts.items()}


def binomial_z(successes: int, total: int, p0: float = 0.5) -> float:
    """The z statistic of ``successes / total`` against the hypothesised rate ``p0``."""
    if total <= 0:
        raise ValueError("an empty sample has no statistic")
    return (successes - total * p0) / math.sqrt(total * p0 * (1.0 - p0))
