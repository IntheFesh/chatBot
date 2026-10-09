"""How she paces a conversation: waits, pauses between bubbles, typing time (R-ENG-003/004/009).

Everything a delay is drawn from comes out of her profile and her routine, never from a constant
of the engine's own: :class:`PacingModel` is read once from a data view and then answers

* **the first delay** - how long she takes to answer.  Free: her real reply-latency distribution
  of the local hour the message arrives in (the long tail of minutes to tens of minutes stays),
  plus the time it takes to read what was written; busy: the latency distribution of the busy
  window she is in; a short pause after something that happened while she was already writing
  (:meth:`PacingModel.short_pause`);
* **the pause between bubbles** - her pause between two messages of one burst plus the time it
  takes to type the next bubble (her typing speed is the slope fitted by the profile, seconds
  per character), never less than one second (R-ENG-009);
* **the quiet window** - what the profile suggests for "he has finished writing": the 75th
  percentile of the pauses inside the *user's* bursts, between the configured window and its
  ceiling (R-ENG-002; used only when ``engine.quiet_window_adaptive`` is on, shown by ``/状态``
  always);
* **the chance that she is still awake** at the edge of her sleep: her real activity in that
  slot relative to her activity while awake (the same number round 10 uses for the occasional
  "can't sleep" message, R-PRO-005).

Without a profile (nothing imported yet) the reference numbers of SPEC section 0 stand in: her
median reply latency of 17 s with a 90th percentile of 240 s, a median pause of 6 s inside a
burst, and the user's 75th percentile of 33 s.  The model says so (``reference``) and the engine
logs it once; the numbers disappear as soon as a profile exists.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import time
from typing import TYPE_CHECKING, Protocol

from twin.profile.activity_model import ActivityModel
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.schedule.plan_model import BusySpan, parse_busy_ref

if TYPE_CHECKING:
    from twin.engine.dataview import ReplyDataView

READ_CHARS_PER_S = 10.0  # how fast a person reads Chinese text
READ_MIN_S = 0.5
READ_MAX_S = 15.0
TYPING_MAX_S = 30.0  # typing one bubble never takes longer than this
MIN_INTERVAL_S = 1.0  # at least a second between bubbles (R-ENG-009)
SLOTS_PER_DAY = 96
SLOT_MINUTES = 15
Z90 = 1.2815515655446004  # the 90th percentile of the standard normal distribution
EDGE_CHAIN_STATES = ("deep_sleep", "sleep_edge")


class Sampler(Protocol):
    """Something a delay in seconds can be drawn from."""

    def sample(
        self, rng: random.Random, *, lo: float | None = None, hi: float | None = None
    ) -> float: ...


@dataclass(frozen=True)
class LogNormal:
    """A log-normal distribution given by its median and 90th percentile (SPEC section 0)."""

    median: float
    p90: float

    def sample(
        self, rng: random.Random, *, lo: float | None = None, hi: float | None = None
    ) -> float:
        sigma = math.log(self.p90 / self.median) / Z90
        value = self.median * math.exp(rng.gauss(0.0, sigma))
        if lo is not None:
            value = max(value, lo)
        if hi is not None:
            value = min(value, hi)
        return value


@dataclass(frozen=True)
class Fixed:
    """A number that does not vary."""

    seconds: float

    def sample(
        self, rng: random.Random, *, lo: float | None = None, hi: float | None = None
    ) -> float:
        return self.seconds


REFERENCE_LATENCY: Sampler = LogNormal(17.0, 240.0)
REFERENCE_BURST_GAP: Sampler = Fixed(6.0)
REFERENCE_USER_GAP_P75 = 33.0


def reading_time(chars: int) -> float:
    """Seconds it takes to read ``chars`` characters (bounded: a long text is skimmed)."""
    return min(READ_MAX_S, max(READ_MIN_S, chars / READ_CHARS_PER_S))


@dataclass(frozen=True)
class PacingModel:
    """The distributions the engine paces a conversation with (see the module description)."""

    latency_by_hour: BucketedDistribution | None = None
    latency_all: EmpiricalDistribution | None = None
    burst_gap: EmpiricalDistribution | None = None
    user_burst_gap: EmpiricalDistribution | None = None
    seconds_per_char: float | None = None
    activity: ActivityModel | None = None

    # --------------------------------------------------------------------- reading

    @classmethod
    def from_view(cls, view: ReplyDataView) -> PacingModel:
        """The pacing of the data view's profile and routine (live or, for tests, any view)."""
        profile = view.profile
        activity = view.activity
        if profile is None:
            return cls(activity=activity)
        metrics = profile.metrics
        return cls(
            latency_by_hour=metrics.hourly("her", "reply_latency_by_hour"),
            latency_all=metrics.distribution("her", "reply_latency_s"),
            burst_gap=metrics.distribution("her", "burst_gap_s"),
            user_burst_gap=metrics.distribution("user", "burst_gap_s"),
            seconds_per_char=metrics.scalar("her", "typing_s_per_char"),
            activity=activity,
        )

    @property
    def reference(self) -> bool:
        """True when nothing of her was measured and the SPEC reference numbers stand in."""
        return (
            self.latency_by_hour is None
            and self.latency_all is None
            and self.burst_gap is None
            and self.activity is None
        )

    # -------------------------------------------------------------- reply latency

    def reply_latency(self, hour: int, slot: int) -> Sampler:
        """Her reply latency for a message that arrives in this local hour (R-ENG-004)."""
        if self.latency_by_hour is not None:
            found = self.latency_by_hour.get(hour)
            if found.n > 0:
                return found
        if self.activity is not None:
            slotted = self.activity.latency_distribution(slot)
            if slotted.n > 0:
                return slotted
        if self.latency_all is not None and self.latency_all.n > 0:
            return self.latency_all
        return REFERENCE_LATENCY

    def busy_latency(self, span: BusySpan | None, hour: int, slot: int) -> Sampler:
        """The latency distribution of the busy window she is in (the long tail stays)."""
        if span is not None and self.activity is not None:
            try:
                day_type, weekday, index = parse_busy_ref(span.latency_ref)
                windows = self.activity.busy_windows(day_type, weekday)
                if 0 <= index < len(windows) and windows[index].latency.n > 0:
                    return windows[index].latency
            except (ValueError, KeyError):
                pass  # a reference this model cannot resolve: the hour's own distribution
        return self.reply_latency(hour, slot)

    def first_delay(self, rng: random.Random, hour: int, slot: int, chars: int) -> float:
        """Free: her latency of the hour plus the time to read what was written."""
        return self.reply_latency(hour, slot).sample(rng) + reading_time(chars)

    def short_pause(self, rng: random.Random, chars: int) -> float:
        """A short pause: reading the new message plus one of her pauses inside a burst."""
        return reading_time(chars) + self._gap().sample(rng)

    # ----------------------------------------------------------------- bubble pacing

    def _gap(self) -> Sampler:
        if self.burst_gap is not None and self.burst_gap.n > 0:
            return self.burst_gap
        return REFERENCE_BURST_GAP

    def typing_time(self, chars: int) -> float:
        """Seconds it takes her to type ``chars`` characters (her measured typing speed)."""
        if not self.seconds_per_char or chars <= 0:
            return 0.0
        return min(TYPING_MAX_S, self.seconds_per_char * chars)

    def bubble_interval(self, rng: random.Random, chars: int, *, first: bool) -> float:
        """Seconds before a bubble of ``chars`` characters appears (R-ENG-009).

        The first bubble of a reply comes after its typing time alone (the first delay already
        happened); the later ones after one of her pauses between messages plus the typing time.
        At least :data:`MIN_INTERVAL_S`.
        """
        pause = 0.0 if first else self._gap().sample(rng)
        return max(MIN_INTERVAL_S, pause + self.typing_time(chars))

    # ------------------------------------------------------------------ quiet window

    def user_pause_p75(self) -> float:
        """The 75th percentile of the pauses inside the user's bursts (seconds)."""
        if self.user_burst_gap is not None and self.user_burst_gap.n > 0:
            return self.user_burst_gap.quantile(0.75)
        return REFERENCE_USER_GAP_P75

    def suggested_quiet_window(self, base_s: float, ceiling_s: float) -> float:
        """The quiet window the profile suggests: the user's p75, within the configured bounds."""
        return min(max(self.user_pause_p75(), base_s), max(base_s, ceiling_s))

    # ---------------------------------------------------------------------- the edge

    def edge_awake_probability(self, slot: int, day_type: str) -> float:
        """The chance that she is still awake at the edge of her sleep, in this 15-minute slot.

        Her real activity in the slot relative to her mean activity in the slots in which she is
        awake (R-PRO-005: "the probability follows her real late-night frequency").  Without a
        routine model she is never awake at the edge.
        """
        model = self.activity
        if model is None:
            return 0.0
        awake = [
            model.rate_at(index, day_type)
            for index in range(SLOTS_PER_DAY)
            if model.typical_state(_slot_time(index), day_type) not in EDGE_CHAIN_STATES
        ]
        mean = sum(awake) / len(awake) if awake else 0.0
        if mean <= 0.0:
            return 0.0
        return min(1.0, max(0.0, model.rate_at(slot, day_type) / mean))


def _slot_time(slot: int) -> time:
    minutes = slot * SLOT_MINUTES
    return time(minutes // 60, minutes % 60)
