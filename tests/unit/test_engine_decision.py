"""DECIDING: when she answers - her state, her distributions, the pause (R-ENG-003/004)."""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass, fields
from datetime import datetime, time, timedelta

import pytest
from pydantic import BaseModel

from tests.support.engine_harness import START, FixedDay, reference_pacing
from twin.config.runtime import registered_settings
from twin.config.settings import Settings
from twin.engine.decision import MAX_RETRIES, Decider, Decision, Situation
from twin.engine.pacing import PacingModel, reading_time
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.schedule.plan_model import BusySpan, busy_ref

NOON = START + timedelta(hours=5)  # 12:00 in Chicago


class Constant(random.Random):
    """A generator whose ``random()`` is a fixed number and whose ``uniform`` takes the midpoint."""

    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.value


def situation(
    now: datetime = NOON, *, chars: int = 7, hour: int = 12, slot: int = 48, **changes: object
) -> Situation:
    anchor = changes.pop("anchor", now)
    assert isinstance(anchor, datetime)
    fields: dict[str, object] = {"day_type": "workday", **changes}
    return Situation(now=now, anchor=anchor, chars=chars, hour=hour, slot=slot, **fields)  # type: ignore[arg-type]


def decider(day: FixedDay, rng: random.Random | None = None) -> Decider:
    return Decider(day, rng or random.Random(3), wake_window_min=(5.0, 40.0))  # type: ignore[arg-type]


def free_day() -> FixedDay:
    return FixedDay((NOON - timedelta(days=1), "free"))


def night() -> FixedDay:
    """She sleeps from 22:00 to 07:00 local with half an hour of edge on each side."""
    return FixedDay(
        (NOON - timedelta(days=1), "free"),
        (NOON + timedelta(hours=9, minutes=30), "sleep_edge"),  # 21:30 local
        (NOON + timedelta(hours=10), "deep_sleep"),
        (NOON + timedelta(hours=18, minutes=30), "sleep_edge"),
        (NOON + timedelta(hours=19), "free"),
    )


# ----------------------------------------------------------------------------- free


def test_free_the_delay_comes_from_her_latency_of_that_hour_plus_reading_time() -> None:
    by_hour = BucketedDistribution.from_counters(
        {12: {10.0: 50}, 23: {500.0: 50}}, sizes=(1,), min_samples=20
    )
    pacing = PacingModel(latency_by_hour=by_hour)
    deciding = decider(free_day())
    noon = deciding.decide(pacing, situation(hour=12))
    assert noon.mode == "free" and not noon.woke_up
    assert noon.delay_s == pytest.approx(10.0 + reading_time(7))
    late = deciding.decide(pacing, situation(hour=23))
    assert late.delay_s == pytest.approx(500.0 + reading_time(7))


def test_the_long_tail_of_her_latency_is_kept() -> None:
    pacing = PacingModel(
        latency_all=EmpiricalDistribution.from_counter({10.0: 80, 900.0: 15, 2400.0: 5})
    )
    deciding = decider(free_day(), random.Random(11))
    delays = [deciding.decide(pacing, situation()).delay_s for _ in range(400)]
    assert min(delays) < 30 and max(delays) > 1500
    assert sum(1 for d in delays if d > 600) > 20  # minutes to tens of minutes happen


def test_a_longer_message_takes_longer_to_read() -> None:
    pacing = reference_pacing()
    short = decider(free_day()).decide(pacing, situation(chars=5))
    long = decider(free_day()).decide(pacing, situation(chars=80))
    assert long.delay_s - short.delay_s == pytest.approx(reading_time(80) - reading_time(5))


def test_the_delay_is_counted_from_the_last_message_and_never_lies_in_the_past() -> None:
    pacing = reference_pacing(latency_s=20.0)
    anchor = NOON - timedelta(seconds=15)  # fifteen seconds of quiet window have passed
    decision = decider(free_day()).decide(pacing, situation(anchor=anchor))
    assert decision.send_at == anchor + timedelta(seconds=20.7)
    quick = decider(free_day()).decide(
        reference_pacing(latency_s=3.0), situation(anchor=NOON - timedelta(seconds=60))
    )
    assert quick.send_at == NOON  # she would have answered already: now


def test_the_same_seed_gives_the_same_decision_and_another_seed_another() -> None:
    pacing = PacingModel(latency_all=EmpiricalDistribution.from_samples(list(range(5, 400, 5))))
    first = decider(free_day(), random.Random(5)).decide(pacing, situation())
    again = decider(free_day(), random.Random(5)).decide(pacing, situation())
    other = decider(free_day(), random.Random(6)).decide(pacing, situation())
    assert first == again and first.send_at != other.send_at


def test_without_a_plan_she_is_free() -> None:
    day = free_day()
    day.unavailable = True
    decision = decider(day).decide(reference_pacing(), situation())
    assert decision.mode == "free" and decision.state is None


# ----------------------------------------------------------------------------- busy


@dataclass
class Window:
    latency: EmpiricalDistribution


class Routine:
    """What the pacing asks of the routine model: busy windows, slot latency, rate, state."""

    def __init__(self, busy: list[Window] | None = None, rates: dict[int, float] | None = None):
        self.busy = busy or []
        self.rates = rates or {}
        self.calls: list[tuple[str, int | None]] = []

    def busy_windows(self, day_type: str, weekday: int | None = None) -> list[Window]:
        self.calls.append((day_type, weekday))
        return self.busy

    def latency_distribution(self, slot: int) -> EmpiricalDistribution:
        return EmpiricalDistribution.empty()

    def rate_at(self, slot: int, day_type: str) -> float:
        return self.rates.get(slot, 1.0)

    def typical_state(self, local_time: time, day_type: str) -> str:
        return "deep_sleep" if local_time.hour < 7 else "free"


def busy_day(span: BusySpan) -> FixedDay:
    return FixedDay(
        (NOON - timedelta(days=1), "free"), (NOON - timedelta(hours=1), "busy"), busy=span
    )


def test_busy_the_delay_comes_from_the_latency_of_that_busy_window() -> None:
    window = Window(EmpiricalDistribution.from_counter({600.0: 50, 1800.0: 50}))
    routine = Routine([Window(EmpiricalDistribution.from_samples([1.0] * 5)), window])
    span = BusySpan(
        NOON - timedelta(hours=1),
        NOON + timedelta(hours=3),
        "11:00–15:00",
        "inferred",
        busy_ref("workday", 4, 1),
    )
    pacing = PacingModel(
        latency_all=EmpiricalDistribution.from_samples([5.0] * 5), activity=routine
    )  # type: ignore[arg-type]
    deciding = decider(busy_day(span), random.Random(2))
    delays = {deciding.decide(pacing, situation()).delay_s for _ in range(40)}
    assert delays <= {600.0, 1800.0} and len(delays) == 2  # not her free latency, and no reading
    assert routine.calls[0] == ("workday", 4)
    decision = deciding.decide(pacing, situation())
    assert decision.mode == "busy" and decision.state == "busy"


def test_a_busy_window_the_model_cannot_find_falls_back_to_her_latency_of_the_hour() -> None:
    span = BusySpan(NOON, NOON + timedelta(hours=1), "x", "inferred", "garbage")
    pacing = PacingModel(latency_all=EmpiricalDistribution.from_samples([40.0] * 5))
    decision = decider(busy_day(span)).decide(pacing, situation())
    assert decision.mode == "busy" and decision.delay_s == pytest.approx(40.0)


# ---------------------------------------------------------------------------- asleep


def test_deep_sleep_queues_the_reply_to_her_waking_plus_a_jitter() -> None:
    pacing = reference_pacing()
    sent = {
        seed: decider(night(), random.Random(seed)).decide(
            pacing, situation(NOON + timedelta(hours=12), hour=0, slot=0)
        )
        for seed in range(30)
    }
    wake = NOON + timedelta(hours=19)  # 07:00 local: the end of the edge
    for decision in sent.values():
        assert decision.mode == "asleep" and decision.woke_up and decision.wake_at == wake
        assert wake + timedelta(minutes=5) <= decision.send_at <= wake + timedelta(minutes=40)
    assert len({d.send_at for d in sent.values()}) > 10  # the jitter varies


def test_a_delay_that_ends_in_deep_sleep_waits_for_the_waking() -> None:
    pacing = reference_pacing(latency_s=3 * 3600.0)
    now = NOON + timedelta(hours=9)  # 21:00, free; three hours later she sleeps
    decision = decider(night()).decide(pacing, situation(now, hour=21, slot=84))
    assert decision.mode == "asleep" and decision.woke_up
    assert decision.send_at >= NOON + timedelta(hours=19, minutes=5)


def test_at_the_edge_of_sleep_she_may_still_be_awake_with_her_real_late_night_chance() -> None:
    routine = Routine(rates={90: 0.25})  # 22:30: a quarter of her activity while awake
    pacing = PacingModel(
        latency_all=EmpiricalDistribution.from_samples([30.0] * 5),
        activity=routine,  # type: ignore[arg-type]
    )
    awake_mean = (67 * 1.0 + 0.25) / 68  # the slots in which she is awake, the 07:00 on
    assert pacing.edge_awake_probability(90, "workday") == pytest.approx(
        0.25 / awake_mean, abs=0.02
    )
    edge_day = FixedDay(
        (NOON - timedelta(days=1), "free"),
        (NOON, "sleep_edge"),
        (NOON + timedelta(hours=9), "free"),
    )
    awake = decider(edge_day, Constant(0.1)).decide(pacing, situation(slot=90))
    asleep = decider(edge_day, Constant(0.9)).decide(pacing, situation(slot=90))
    assert awake.mode == "edge_awake" and not awake.woke_up and awake.state == "sleep_edge"
    assert asleep.mode == "asleep" and asleep.woke_up


def test_without_a_routine_model_she_is_never_awake_at_the_edge() -> None:
    assert PacingModel().edge_awake_probability(90, "workday") == 0.0
    assert (
        PacingModel(activity=Routine(rates={90: 0.0})).edge_awake_probability(90, "workday") == 0.0
    )  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- pause


def test_a_pause_holds_the_reply_back_and_afterwards_she_has_just_seen_it() -> None:
    pacing = reference_pacing()
    until = NOON + timedelta(hours=2)
    decision = decider(free_day()).decide(pacing, situation(paused_until=until))
    assert decision.mode == "paused" and not decision.woke_up and decision.paused_until == until
    assert decision.send_at == until + timedelta(seconds=reading_time(7) + 4.0)


def test_a_pause_that_ends_while_she_sleeps_waits_for_the_waking() -> None:
    pacing = reference_pacing()
    until = NOON + timedelta(hours=13)  # 01:00 local
    decision = decider(night()).decide(pacing, situation(paused_until=until))
    assert decision.mode == "paused" and decision.woke_up
    assert decision.send_at >= NOON + timedelta(hours=19, minutes=5)


def test_a_pause_that_is_over_changes_nothing() -> None:
    pacing = reference_pacing()
    decision = decider(free_day()).decide(
        pacing, situation(paused_until=NOON - timedelta(minutes=1))
    )
    assert decision.mode == "free"


# ------------------------------------------------------------ continuation and retries


def test_a_continuation_needs_only_a_short_pause() -> None:
    decision = decider(free_day()).decide(reference_pacing(), situation(continuation=True))
    assert decision.mode == "continuation"
    assert decision.delay_s == pytest.approx(reading_time(7) + 4.0)


def test_a_retry_comes_two_to_ten_minutes_later() -> None:
    deciding = decider(free_day(), random.Random(9))
    delays = [deciding.retry(NOON).delay_s for _ in range(200)]
    assert min(delays) >= 120.0 and max(delays) <= 600.0 and max(delays) - min(delays) > 200
    assert deciding.retry(NOON).mode == "retry" and MAX_RETRIES == 3


def test_a_decision_survives_the_trip_through_json() -> None:
    decision = decider(night()).decide(
        reference_pacing(), situation(NOON + timedelta(hours=12), paused_until=None)
    )
    assert Decision.from_json(decision.to_json()) == decision
    with pytest.raises(ValueError, match="anchor"):
        Decision.from_json({"mode": "free"})


def test_the_wake_window_must_be_a_real_range() -> None:
    with pytest.raises(ValueError, match="wake-up window"):
        Decider(free_day(), random.Random(), wake_window_min=(40.0, 5.0))  # type: ignore[arg-type]


def test_waking_is_found_through_the_edges_of_the_sleep() -> None:
    deciding = decider(night())
    state = deciding.her_state(NOON + timedelta(hours=12))
    assert state is not None and state.kind == "deep_sleep"
    assert deciding.wake_time(state) == NOON + timedelta(hours=19)


# ------------------------------------------------------------------ no fast-forward switch


def _setting_names() -> list[str]:
    found: list[str] = []

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            found.append(f"{prefix}{name}")
            inner = field.annotation
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                walk(inner, f"{prefix}{name}.")

    walk(Settings, "")
    return [*found, *registered_settings()]


def test_there_is_no_switch_for_an_instant_answer() -> None:
    """Real mode has no fast-forward: no setting, no parameter turns her delay off (R-SCOPE-006)."""
    forbidden = (
        "instant",
        "immediate",
        "no_delay",
        "nodelay",
        "skip_delay",
        "no_wait",
        "fast_reply",
    )
    names = [*_setting_names(), *inspect.signature(Decider.decide).parameters]
    names += [field.name for field in fields(Situation)]
    assert not [n for n in names if any(word in n.lower() for word in forbidden)]
    deciding = decider(free_day(), random.Random(21))
    delays = [deciding.decide(reference_pacing(), situation()).delay_s for _ in range(300)]
    assert min(delays) >= reading_time(7)  # even her quickest answer takes the time to read
