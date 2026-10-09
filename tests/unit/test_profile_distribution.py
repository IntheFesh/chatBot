"""Empirical distributions: building, sampling, truncation, mixing, buckets (R-PROF-003)."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from twin.profile.circular import (
    CircularStats,
    circular_mean,
    circular_median,
    signed_diff,
    wrap,
)
from twin.profile.distribution import (
    GRID,
    BucketedDistribution,
    EmpiricalDistribution,
    EmptyDistributionError,
)


def lognormal_sample(count: int, seed: int = 3) -> list[float]:
    rng = random.Random(seed)
    return [round(rng.lognormvariate(math.log(17.0), 1.1)) for _ in range(count)]


def test_quantiles_follow_the_sample() -> None:
    dist = EmpiricalDistribution.from_samples([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert dist.n == 10 and dist.mean() == pytest.approx(5.5)
    assert dist.median() == pytest.approx(5.5)
    assert dist.quantile(0.0) == 1 and dist.quantile(1.0) == 10
    assert dist.quantile(0.9) == pytest.approx(9.1)
    assert list(GRID) == sorted(GRID) and GRID[0] == 0.0 and GRID[-1] == 1.0


def test_large_sample_median_is_within_five_percent() -> None:
    values = lognormal_sample(20_000)
    dist = EmpiricalDistribution.from_samples(values)
    rng = random.Random(9)
    drawn = dist.samples(rng, 20_000)
    assert statistics.median(drawn) == pytest.approx(statistics.median(values), rel=0.05)
    assert dist.quantile(0.9) == pytest.approx(sorted(values)[int(0.9 * len(values))], rel=0.05)


def test_sampling_is_reproducible_for_a_seed() -> None:
    dist = EmpiricalDistribution.from_samples(lognormal_sample(500))
    first = dist.samples(random.Random(4), 20)
    assert first == dist.samples(random.Random(4), 20)
    assert first != dist.samples(random.Random(5), 20)


def test_discrete_distributions_draw_whole_numbers() -> None:
    dist = EmpiricalDistribution.from_counter({2: 50, 3: 30, 6: 20}, discrete=True)
    draws = dist.samples(random.Random(1), 2000)
    assert all(value == int(value) for value in draws)
    assert set(draws) <= {2.0, 3.0, 4.0, 5.0, 6.0}
    assert statistics.median(draws) == pytest.approx(2.0, abs=1.0)


def test_truncation_keeps_draws_inside_the_interval() -> None:
    dist = EmpiricalDistribution.from_samples(lognormal_sample(4000))
    rng = random.Random(2)
    inside = dist.samples(rng, 500, lo=10, hi=60)
    assert all(10 <= value <= 60 for value in inside)
    assert dist.sample(rng, lo=1e9) == 1e9  # no mass up there: the nearest end of the interval
    assert dist.sample(rng, hi=-5) == -5
    only_low = dist.samples(rng, 300, hi=float(dist.quantile(0.5)))
    assert max(only_low) <= dist.quantile(0.5)


def test_an_empty_distribution_cannot_be_sampled() -> None:
    empty = EmpiricalDistribution.empty()
    assert empty.is_empty and empty.cdf(3.0) == 0.0
    with pytest.raises(EmptyDistributionError):
        empty.sample(random.Random(0))
    with pytest.raises(EmptyDistributionError):
        empty.quantile(0.5)
    assert EmpiricalDistribution.from_counter({}).is_empty
    assert EmpiricalDistribution.from_counter({4.0: 0}).is_empty


def test_json_round_trip_keeps_the_distribution() -> None:
    dist = EmpiricalDistribution.from_samples(lognormal_sample(300), discrete=True)
    again = EmpiricalDistribution.from_json(dist.to_json())
    assert again.n == dist.n and again.discrete
    assert again.quantile(0.5) == pytest.approx(dist.quantile(0.5))
    with pytest.raises(ValueError, match="grid"):
        EmpiricalDistribution.from_json({"n": 1, "mean": 1, "q": [1, 2]})


def test_a_mixture_lies_between_its_parts() -> None:
    slow = EmpiricalDistribution.from_samples([100.0] * 50 + [200.0] * 50)
    fast = EmpiricalDistribution.from_samples([1.0] * 50 + [2.0] * 50)
    mixed = EmpiricalDistribution.mix([(fast, 0.7), (slow, 0.3)])
    assert fast.median() < mixed.median() < slow.median()
    assert mixed.mean() == pytest.approx(0.7 * fast.mean() + 0.3 * slow.mean())
    assert mixed.n == round(0.7 * 100 + 0.3 * 100)  # the weighted size of the parts
    assert EmpiricalDistribution.mix([(fast, 1.0)]) is fast
    assert EmpiricalDistribution.mix([(fast, 0.0), (slow, 1.0)]).median() == slow.median()
    assert EmpiricalDistribution.mix([]).is_empty
    share = sum(1 for v in mixed.samples(random.Random(3), 4000) if v > 50) / 4000
    assert share == pytest.approx(0.3, abs=0.05)


def test_buckets_fall_back_to_a_coarser_level_when_thin() -> None:
    per_bucket = {
        0: {5.0: 40},  # slot 0: enough on its own
        1: {5.0: 3},  # thin slot: the hour group {0..3} answers
        2: {9.0: 40},
        9: {1000.0: 2},  # thin hour group: everything answers
    }
    buckets = BucketedDistribution.from_counters(
        per_bucket, sizes=(1, 4), min_samples=30, discrete=True
    )
    assert buckets.level_for(0) == 0 and buckets.get(0).median() == 5.0
    assert buckets.level_for(1) == 1 and buckets.get(1).n == 83
    assert buckets.level_for(9) == 2 and buckets.get(9) is buckets.overall
    assert buckets.get(50) is buckets.overall
    assert buckets.samples_in(1) == 3 and buckets.samples_in(50) == 0
    again = BucketedDistribution.from_json(buckets.to_json())
    assert again.get(0).median() == 5.0 and again.sizes == (1, 4)
    with pytest.raises(ValueError, match="group widths"):
        BucketedDistribution.from_counters({}, sizes=(0,))


def test_bucketed_mixtures_combine_bucket_by_bucket() -> None:
    old = BucketedDistribution.from_counters({3: {10.0: 50}}, sizes=(1,), min_samples=10)
    new = BucketedDistribution.from_counters(
        {3: {30.0: 50}, 4: {7.0: 50}}, sizes=(1,), min_samples=10
    )
    mixed = BucketedDistribution.mix([(old, 0.5), (new, 0.5)])
    assert 10 < mixed.get(3).median() < 30
    assert mixed.get(4).median() == pytest.approx(7.0)
    assert BucketedDistribution.mix([]).overall.is_empty


# ------------------------------------------------------------ clock statistics


def test_circular_mean_and_median_across_midnight() -> None:
    times = [23 * 60 + 50, 10, 23 * 60 + 55, 5]
    assert abs(signed_diff(circular_mean(times), 0)) < 10
    assert signed_diff(10, 1430) == 20 and signed_diff(1430, 10) == -20
    assert wrap(-30) == 1410 and wrap(1500) == 60
    assert abs(signed_diff(circular_median(times), 0)) < 10
    with pytest.raises(ValueError, match="no values"):
        circular_mean([])
    with pytest.raises(ValueError, match="no values"):
        CircularStats.from_values([])


def test_circular_stats_summarise_and_sample_clock_times() -> None:
    values = [60 + d for d in (-20, -10, 0, 0, 5, 10, 25)]
    stats = CircularStats.from_values(values)
    assert stats.n == 7 and stats.median == pytest.approx(60.0, abs=2)
    assert 0 < stats.std < 30
    draws = [stats.sample(random.Random(i)) for i in range(50)]
    assert all(abs(signed_diff(d, 60)) <= 25 for d in draws)
    assert stats.quantile(0.5) == pytest.approx(60.0, abs=2)
    again = CircularStats.from_json(stats.to_json())
    assert again.median == pytest.approx(stats.median) and again.n == 7
    fixed = CircularStats.fixed(1500)
    assert fixed.median == 60 and fixed.sample(random.Random(1)) == 60 and fixed.quantile(0.9) == 60
    across = CircularStats.from_values([23 * 60 + 50, 23 * 60 + 58, 8, 12])
    assert abs(signed_diff(across.median, 0)) < 15
