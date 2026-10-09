"""DECIDING: when does she answer? (R-ENG-003, R-ENG-004, R-SCOPE-006, R-SAFE-001's pause).

A :class:`Decision` is the moment the reply starts (``send_at``), counted from ``anchor_at`` - the
arrival of the user's last message - and the reason (``mode``).  :class:`Decider` makes it from her
state in the day plan, her pacing (:mod:`twin.engine.pacing`) and the pause setting:

============  ====================================================================================
``free``      her reply latency of the local hour + the time to read the messages
``busy``      a draw from the latency distribution of the busy window she is in (long tail kept)
``edge_awake``  at the edge of her sleep she is still awake with the probability of her real
              late-night activity: she answers like a free person, drowsily
``asleep``    deep sleep (or the edge, when she is not awake): queued to the moment she wakes plus
              a jitter of ``schedule.greeting_window_min`` minutes (5-40); the prompt says that she
              has just woken up and saw the messages
``paused``    ``/暂停``: nothing before the pause ends, then she has "just seen" them - a short
              pause,
              or the wake-up queue if she sleeps by then
``continuation``  something arrived while she was writing (or the user asked for another try): a
              short pause, no new first delay
``retry``     the model failed: 2 to 10 minutes later (R-ENG-010)
============  ====================================================================================

A moment that falls into her deep sleep is moved to her waking, whatever the mode said ("a reply
due while she is asleep waits until she wakes").  All draws come from the injected generator, so a
seed makes a decision repeatable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from twin.clock import ensure_aware
from twin.engine.pacing import PacingModel
from twin.schedule.plan_model import HerState
from twin.schedule.time_service import PlanUnavailableError, TimeService

DecisionMode = Literal["free", "busy", "edge_awake", "asleep", "paused", "continuation", "retry"]
MAX_WAKE_STEPS = 8  # states that are looked through to find the end of a sleep
RETRY_MIN_S = 120.0  # R-ENG-010: a failed reply is tried again 2 to 10 minutes later
RETRY_MAX_S = 600.0
MAX_RETRIES = 3


@dataclass(frozen=True)
class Decision:
    """When the reply starts, and why."""

    mode: DecisionMode
    anchor_at: datetime
    send_at: datetime
    woke_up: bool = False
    wake_at: datetime | None = None
    paused_until: datetime | None = None
    state: str | None = None  # her state when it was decided

    @property
    def delay_s(self) -> float:
        return max(0.0, (self.send_at - self.anchor_at).total_seconds())

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "anchor_at": self.anchor_at.isoformat(),
            "send_at": self.send_at.isoformat(),
            "woke_up": self.woke_up,
            "wake_at": self.wake_at.isoformat() if self.wake_at else None,
            "paused_until": self.paused_until.isoformat() if self.paused_until else None,
            "state": self.state,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Decision:
        def moment(name: str) -> datetime | None:
            raw = data.get(name)
            return ensure_aware(datetime.fromisoformat(str(raw))) if raw else None

        anchor, send = moment("anchor_at"), moment("send_at")
        if anchor is None or send is None:
            raise ValueError("a stored decision needs its anchor and its send time")
        return cls(
            mode=data["mode"],
            anchor_at=anchor,
            send_at=send,
            woke_up=bool(data.get("woke_up", False)),
            wake_at=moment("wake_at"),
            paused_until=moment("paused_until"),
            state=data.get("state"),
        )


@dataclass(frozen=True)
class Situation:
    """What the decision is about: the clock of the place, the size of the messages, the mood."""

    now: datetime
    anchor: datetime  # the arrival of the user's last message
    chars: int  # characters in the messages of the round (reading time)
    hour: int  # her local hour, from the bot's time zone
    slot: int  # her local 15-minute slot
    day_type: str
    paused_until: datetime | None = None
    continuation: bool = False  # a short pause is enough (something came while she was writing)


class Decider:
    """Makes the :class:`Decision` of a round (see the module description)."""

    def __init__(
        self,
        time: TimeService,
        rng: random.Random,
        *,
        wake_window_min: tuple[float, float] = (5.0, 40.0),
    ) -> None:
        low, high = wake_window_min
        if not 0 <= low <= high:
            raise ValueError("the wake-up window needs 0 <= low <= high")
        self._time = time
        self._rng = rng
        self._wake_window_s = (low * 60.0, high * 60.0)

    # --------------------------------------------------------------------- reading her day

    def her_state(self, at: datetime) -> HerState | None:
        """What she is doing at ``at``; ``None`` when no plan decides it (then she is free)."""
        try:
            return self._time.her_state(at)
        except PlanUnavailableError:
            return None

    def wake_time(self, state: HerState) -> datetime:
        """The moment the sleep that ``state`` belongs to ends."""
        moment = state.until
        for _ in range(MAX_WAKE_STEPS):
            following = self.her_state(moment)
            if following is None or not following.asleep:
                return moment
            moment = following.until
        return moment

    # --------------------------------------------------------------------------- deciding

    def decide(self, pacing: PacingModel, situation: Situation) -> Decision:
        """When she answers a round whose last message arrived at ``situation.anchor``."""
        now = situation.now
        paused = situation.paused_until
        if paused is not None and paused > now:
            return self._after_pause(pacing, situation, paused)
        state = self.her_state(now)
        kind = state.kind if state is not None else "free"
        if kind == "deep_sleep" and state is not None:
            return self._queue_to_wake(situation, state, "asleep")
        if kind == "sleep_edge" and state is not None:
            chance = pacing.edge_awake_probability(situation.slot, situation.day_type)
            if self._rng.random() >= chance:
                return self._queue_to_wake(situation, state, "asleep")
            return self._answer(pacing, situation, state, "edge_awake")
        if kind == "busy" and state is not None and not situation.continuation:
            return self._answer(pacing, situation, state, "busy")
        return self._answer(pacing, situation, state, "free")

    def retry(self, now: datetime) -> Decision:
        """A failed attempt is repeated 2 to 10 minutes later (R-ENG-010)."""
        delay = self._rng.uniform(RETRY_MIN_S, RETRY_MAX_S)
        return Decision("retry", now, now + timedelta(seconds=delay))

    # --------------------------------------------------------------------------- the modes

    def _answer(
        self,
        pacing: PacingModel,
        situation: Situation,
        state: HerState | None,
        mode: DecisionMode,
    ) -> Decision:
        """She is awake: a delay from her distributions, moved out of any sleep it lands in."""
        rng = self._rng
        if situation.continuation:
            delay = pacing.short_pause(rng, situation.chars)
            mode = "continuation"
        elif mode == "busy" and state is not None:
            delay = pacing.busy_latency(state.busy, situation.hour, situation.slot).sample(rng)
        else:
            delay = pacing.first_delay(rng, situation.hour, situation.slot, situation.chars)
        send_at = max(situation.now, situation.anchor + timedelta(seconds=delay))
        later = self.her_state(send_at)
        if later is not None and later.kind == "deep_sleep":
            return self._queue_to_wake(situation, later, "asleep")
        return Decision(
            mode, situation.anchor, send_at, state=state.kind if state is not None else None
        )

    def _queue_to_wake(self, situation: Situation, state: HerState, mode: DecisionMode) -> Decision:
        """She sleeps: the reply waits for her waking plus a jitter (R-ENG-003)."""
        wake = self.wake_time(state)
        low, high = self._wake_window_s
        send_at = wake + timedelta(seconds=self._rng.uniform(low, high))
        return Decision(
            mode,
            situation.anchor,
            send_at,
            woke_up=True,
            wake_at=wake,
            state=state.kind,
            paused_until=situation.paused_until,
        )

    def _after_pause(
        self, pacing: PacingModel, situation: Situation, paused_until: datetime
    ) -> Decision:
        """``/暂停``: after the pause she has just seen the messages (or has just woken up)."""
        state = self.her_state(paused_until)
        if state is not None and state.asleep:
            queued = self._queue_to_wake(situation, state, "asleep")
            return Decision(
                "paused",
                queued.anchor_at,
                queued.send_at,
                woke_up=True,
                wake_at=queued.wake_at,
                paused_until=paused_until,
                state=state.kind,
            )
        pause = pacing.short_pause(self._rng, situation.chars)
        return Decision(
            "paused",
            situation.anchor,
            paused_until + timedelta(seconds=pause),
            paused_until=paused_until,
            state=state.kind if state is not None else None,
        )
