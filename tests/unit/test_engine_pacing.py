"""Her pacing from the profile: delays, pauses, typing speed, quiet window (R-ENG-002/004/009)."""

from __future__ import annotations

import random
import statistics
from collections import Counter

import pytest

from tests.support.engine_extras import Routine
from tests.support.reply_view import StaticDataView, dist, make_profile
from tests.support.synth_chat import ChatSpec, build_chat
from twin.engine.pacing import (
    MIN_INTERVAL_S,
    READ_MAX_S,
    READ_MIN_S,
    REFERENCE_BURST_GAP,
    REFERENCE_LATENCY,
    REFERENCE_USER_GAP_P75,
    TYPING_MAX_S,
    Fixed,
    LogNormal,
    PacingModel,
    reading_time,
)
from twin.profile.api import load_profile
from twin.profile.builder import rebuild
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.profile.metrics import TYPING_MIN_GAPS, typing_speed
from twin.profile.snapshot import ProfileMetrics, assemble_metrics
from twin.profile.values import Hourly, Scalar
from twin.services import Services


def profile_with(**leaves: object) -> StaticDataView:
    base = make_profile()
    her = {name: base.metrics.leaf("her", name, "full") for name in base.metrics.names("her")}
    user: dict[str, object] = {}
    for name, leaf in leaves.items():
        (user if name.startswith("user_") else her)[name.removeprefix("user_")] = leaf
    data = assemble_metrics(
        scope="live",
        config={},
        weight=0.6,
        full={"her": her, "user": user},  # type: ignore[arg-type]
        recent=None,
        full_info={},
        recent_info={},
    )
    view = StaticDataView()
    view.profile = type(base)(ProfileMetrics(data))  # type: ignore[assignment]
    return view


# ------------------------------------------------------------------ what she is made of


def test_the_pacing_reads_her_distributions_from_the_profile() -> None:
    hourly = BucketedDistribution.from_counters({12: {7.0: 40}}, sizes=(1,), min_samples=20)
    view = profile_with(
        burst_gap_s=dist({3.0: 10, 5.0: 10}),
        reply_latency_s=dist({11.0: 30}),
        reply_latency_by_hour=Hourly(hourly),
        user_burst_gap_s=dist({40.0: 30}),
        typing_s_per_char=Scalar(0.4, 200),
    )
    pacing = PacingModel.from_view(view)
    assert not pacing.reference
    assert pacing.seconds_per_char == 0.4 and pacing.user_pause_p75() == 40.0
    assert pacing.reply_latency(12, 48).sample(random.Random(1)) == 7.0  # the hour's own bucket
    assert pacing.reply_latency(3, 12).sample(random.Random(1)) == 7.0  # fallback to "all day"


def test_without_a_profile_the_reference_numbers_of_the_spec_stand_in() -> None:
    pacing = PacingModel.from_view(StaticDataView(profile=None))
    assert pacing.reference and pacing.seconds_per_char is None
    assert pacing.reply_latency(12, 48) is REFERENCE_LATENCY
    assert pacing._gap() is REFERENCE_BURST_GAP  # type: ignore[attr-defined]
    assert pacing.user_pause_p75() == REFERENCE_USER_GAP_P75 == 33.0


def test_a_profile_without_the_new_metrics_still_gives_a_model() -> None:
    pacing = PacingModel.from_view(StaticDataView())  # the profile of the pipeline tests
    assert pacing.seconds_per_char is None and pacing.latency_by_hour is None
    assert pacing.bubble_interval(random.Random(1), 10, first=True) == MIN_INTERVAL_S


def test_the_reference_latency_has_her_median_and_ninetieth_percentile() -> None:
    rng = random.Random(8)
    draws = sorted(REFERENCE_LATENCY.sample(rng) for _ in range(6000))
    assert statistics.median(draws) == pytest.approx(17.0, rel=0.1)
    assert draws[int(len(draws) * 0.9)] == pytest.approx(240.0, rel=0.2)
    assert Fixed(6.0).sample(rng, lo=1.0, hi=3.0) == 6.0
    capped = LogNormal(17.0, 240.0).sample(rng, lo=20.0, hi=30.0)
    assert 20.0 <= capped <= 30.0


# --------------------------------------------------------------------------- reading


@pytest.mark.parametrize(
    ("chars", "expected"), [(0, READ_MIN_S), (5, 0.5), (40, 4.0), (900, READ_MAX_S)]
)
def test_reading_time_follows_the_length_between_its_bounds(chars: int, expected: float) -> None:
    assert reading_time(chars) == pytest.approx(expected)


# ------------------------------------------------------------------------ the bubbles


def test_typing_time_is_her_speed_times_the_length_up_to_a_ceiling() -> None:
    pacing = PacingModel(seconds_per_char=0.5)
    assert pacing.typing_time(10) == 5.0 and pacing.typing_time(0) == 0.0
    assert pacing.typing_time(10_000) == TYPING_MAX_S
    assert PacingModel().typing_time(10) == 0.0


def test_the_pause_between_bubbles_is_her_pause_plus_typing_and_never_below_a_second() -> None:
    gap = EmpiricalDistribution.from_samples([0.0] * 5)
    quick = PacingModel(burst_gap=gap, seconds_per_char=0.01)
    assert quick.bubble_interval(random.Random(1), 3, first=False) == MIN_INTERVAL_S
    slow = PacingModel(
        burst_gap=EmpiricalDistribution.from_samples([8.0] * 5), seconds_per_char=0.5
    )
    assert slow.bubble_interval(random.Random(1), 10, first=False) == 13.0
    assert (
        slow.bubble_interval(random.Random(1), 10, first=True) == 5.0
    )  # no pause before the first


# ------------------------------------------------------------------------ quiet window


def test_the_suggested_quiet_window_is_the_users_p75_between_the_configured_bounds() -> None:
    def pacing(p75: float) -> PacingModel:
        return PacingModel(user_burst_gap=EmpiricalDistribution.from_samples([p75] * 5))

    assert pacing(33.0).suggested_quiet_window(15.0, 45.0) == 33.0
    assert pacing(5.0).suggested_quiet_window(15.0, 45.0) == 15.0  # never below the window
    assert pacing(300.0).suggested_quiet_window(15.0, 45.0) == 45.0  # nor above its ceiling
    assert pacing(300.0).suggested_quiet_window(60.0, 45.0) == 60.0  # a ceiling below the base


# ------------------------------------------------------------------------ the busy window


def test_a_busy_reference_the_model_does_not_know_falls_back_to_the_hour() -> None:
    pacing = PacingModel(
        latency_all=EmpiricalDistribution.from_samples([9.0] * 5),
        activity=Routine(),  # type: ignore[arg-type]
    )
    assert pacing.busy_latency(None, 12, 48).sample(random.Random(1)) == 9.0


# ------------------------------------------------------------- the typing speed metric


def test_typing_speed_is_the_slope_of_the_pause_against_the_length() -> None:
    gaps = {size: Counter({2.0 + 0.5 * size: TYPING_MIN_GAPS + 5}) for size in (2, 4, 8, 12, 20)}
    leaf = typing_speed(gaps)
    assert leaf.value == pytest.approx(0.5) and leaf.n == 5 * (TYPING_MIN_GAPS + 5)


def test_typing_speed_ignores_classes_with_too_few_pauses_and_needs_three() -> None:
    thin = {size: Counter({3.0: TYPING_MIN_GAPS - 1}) for size in range(2, 12)}
    assert typing_speed(thin) == Scalar(0.0, 0)
    two = {size: Counter({3.0: TYPING_MIN_GAPS}) for size in (2, 5)}
    assert typing_speed(two).n == 0


def test_typing_speed_is_never_negative_and_uses_the_median_pause() -> None:
    falling = {size: Counter({20.0 - size: TYPING_MIN_GAPS}) for size in (2, 6, 10, 14)}
    assert typing_speed(falling).value == 0.0  # she does not type faster the longer it gets
    noisy = {
        size: Counter({2.0 + 0.5 * size: TYPING_MIN_GAPS, 900.0: 3}) for size in (2, 6, 10, 14)
    }
    assert typing_speed(noisy).value == pytest.approx(0.5)  # a few long absences do not count


def test_the_profile_of_a_conversation_has_her_typing_speed(services: Services) -> None:
    build_chat(services, ChatSpec(days=40))
    rebuild(services, "all")
    profile = load_profile(services, "live")
    assert profile is not None
    assert "typing_s_per_char" in profile.metrics.names("her")
    assert "typing_s_per_char" in profile.metrics.names("user")
    speed = profile.metrics.scalar("her", "typing_s_per_char")
    assert speed is None or speed >= 0.0
    pacing = PacingModel(seconds_per_char=speed)
    assert pacing.typing_time(8) >= 0.0
