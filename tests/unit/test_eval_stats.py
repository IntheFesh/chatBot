"""The statistics of the blind test, against published values (R-EVAL-001, R-SRV-005).

Wilson (1927) as tabulated by Newcombe (1998, Statistics in Medicine 17:857-872, Table II, the
score method): 81/263 -> (0.2553, 0.3662), 15/148 -> (0.0624, 0.1605), 0/20 -> (0, 0.1611) and
1/29 -> (0.0061, 0.1718).  The two-proportion z-test: 30/50 against 20/50 pools to p = 0.5, so
SE = sqrt(0.5 * 0.5 * (1/50 + 1/50)) = 0.1 and z = 0.2 / 0.1 = 2, whose normal tail areas are the
textbook 0.0455 (two-sided) and 0.02275 (one-sided).
"""

from __future__ import annotations

import math
import random

import pytest

from twin.eval.stats import (
    Rate,
    binomial_z,
    group_rates,
    normal_cdf,
    two_proportion_test,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("successes", "total", "low", "high"),
    [
        (81, 263, 0.2553, 0.3662),
        (15, 148, 0.0624, 0.1605),
        (0, 20, 0.0, 0.1611),
        (1, 29, 0.0061, 0.1718),
    ],
)
def test_the_wilson_interval_matches_the_published_values(
    successes: int, total: int, low: float, high: float
) -> None:
    found = wilson_interval(successes, total)
    assert found is not None
    assert found[0] == pytest.approx(low, abs=5e-5)
    assert found[1] == pytest.approx(high, abs=5e-5)


def test_the_wilson_interval_stays_inside_zero_and_one_and_contains_the_estimate() -> None:
    for total in (1, 2, 7, 50, 400):
        for successes in range(0, total + 1, max(1, total // 9)):
            low, high = wilson_interval(successes, total) or (-1.0, -1.0)
            assert 0.0 <= low <= successes / total <= high <= 1.0


def test_an_empty_sample_has_no_interval_and_a_bad_count_is_refused() -> None:
    assert (
        wilson_interval(0, 0) is None and Rate(0, 0).point is None and Rate(0, 0).interval is None
    )
    with pytest.raises(ValueError, match="between 0 and total"):
        wilson_interval(5, 4)


def test_the_two_proportion_test_matches_the_textbook_numbers() -> None:
    test = two_proportion_test(30, 50, 20, 50)
    assert test is not None
    assert test.z == pytest.approx(2.0)
    assert test.p_two_sided == pytest.approx(0.0455, abs=5e-5)
    assert test.p_greater == pytest.approx(0.02275, abs=5e-5)  # H1: the first rate is higher
    assert test.p_less == pytest.approx(1 - 0.02275, abs=5e-5)
    assert (test.p1, test.p2) == (0.6, 0.4)


def test_the_one_sided_p_value_for_a_lower_guess_rate_is_the_one_round_14_asks_for() -> None:
    """R-SRV-005: the new model's rate lower than DeepSeek's, one-sided p < 0.1."""
    lower = two_proportion_test(20, 50, 30, 50)  # 40 % against 60 %
    assert lower is not None
    assert lower.p_less == pytest.approx(0.02275, abs=5e-5) and lower.p_less < 0.1
    same = two_proportion_test(25, 50, 25, 50)
    assert same is not None and same.z == 0 and same.p_less == pytest.approx(0.5)
    assert same.p_two_sided == pytest.approx(1.0)


def test_the_two_proportion_test_is_symmetric_and_handles_degenerate_samples() -> None:
    a, b = two_proportion_test(12, 40, 25, 60), two_proportion_test(25, 60, 12, 40)
    assert a is not None and b is not None
    assert a.z == pytest.approx(-b.z) and a.p_two_sided == pytest.approx(b.p_two_sided)
    nobody = two_proportion_test(0, 30, 0, 30)
    assert nobody is not None and nobody.p_two_sided == 1.0 and nobody.z == 0.0
    assert two_proportion_test(0, 0, 3, 5) is None
    with pytest.raises(ValueError, match="between 0 and total"):
        two_proportion_test(9, 8, 1, 2)


def test_the_normal_distribution_function_has_its_known_points() -> None:
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)
    assert normal_cdf(-1.281552) == pytest.approx(0.1, abs=1e-6)


def test_rates_are_grouped_and_items_without_a_group_are_left_out() -> None:
    items = [("a", True), ("a", False), ("a", True), ("b", False), (None, True)]
    grouped = group_rates(items, lambda i: i[0], lambda i: i[1])
    assert grouped == {"a": Rate(2, 3), "b": Rate(0, 1)}
    assert grouped["a"].point == pytest.approx(2 / 3)


def test_a_fair_coin_passes_the_binomial_check() -> None:
    """The check the side randomisation is held to: 4000 flips, |z| well below 3."""
    rng = random.Random(7)
    heads = sum(rng.random() < 0.5 for _ in range(4000))
    assert abs(binomial_z(heads, 4000)) < 3
    assert binomial_z(350, 400) > 10  # a coin that nearly always lands left is rejected
    with pytest.raises(ValueError, match="empty sample"):
        binomial_z(0, 0)
    assert math.isfinite(binomial_z(1, 2))
