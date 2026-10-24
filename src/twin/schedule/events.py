"""The events the schedule publishes (R-SCH-002, R-SCH-005); rounds 09 and 10 subscribe.

The schedule decides when things stop being valid - a time zone switch, a restart, a wake-up
from sleep - and says so here instead of reaching into the tables of the engine and of the
proactive scheduler, which do not exist yet and belong to other rounds.

``CandidatesExpired``
    Every proactive candidate planned for a time before :attr:`CandidatesExpired.cutoff` is void:
    it is marked expired and never sent late (R-SCH-005).  :meth:`CandidatesExpired.covers` is the
    one definition of "planned before now"; round 10 calls it for the candidates it holds.  A
    switch of the time zone publishes it too, because a candidate was placed on the clock of the
    old zone: the new plan makes new ones.
``TimezoneSwitched``
    The bot's time zone changed (R-SCH-002) and the rest of today was planned again.  Delayed
    replies already queued keep their absolute time - an instant does not depend on the zone, the
    user is waiting and the delay was already drawn - and are judged against the new plan when
    they come due (a reply due while she is asleep waits until she wakes, as always).  Proactive
    candidates are dropped and drawn again from the new plan.
``PlanRebuilt``
    A new plan took over from a moment on (any reason).
``Resumed``
    The application started or the machine woke up: the channel reconnects (round 02 component),
    the engine answers what arrived meanwhile (round 09), the plan is current.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from twin.ops.logging import get_logger

log = get_logger("twin.schedule.events")

ResumeKind = Literal["startup", "wake"]


@dataclass(frozen=True)
class ScheduleEvent:
    """Base class of the events; ``at`` is when it was decided (aware UTC)."""

    at: datetime


@dataclass(frozen=True)
class CandidatesExpired(ScheduleEvent):
    """Proactive candidates planned before ``cutoff`` are void (never sent late)."""

    cutoff: datetime
    reason: Literal["startup", "wake", "timezone_switch"]
    include_future: bool = False  # a time zone switch voids the later candidates as well

    def covers(self, planned_at: datetime) -> bool:
        """Whether a candidate planned for ``planned_at`` is void."""
        return self.include_future or planned_at < self.cutoff


@dataclass(frozen=True)
class TimezoneSwitched(ScheduleEvent):
    old_timezone: str
    new_timezone: str
    switch_id: str
    plan_id: str | None


@dataclass(frozen=True)
class PlanRebuilt(ScheduleEvent):
    plan_id: str
    reason: str
    superseded: tuple[str, ...]


@dataclass(frozen=True)
class Resumed(ScheduleEvent):
    kind: ResumeKind
    gap_s: float  # how long the machine was away (0 for a start)
    plan_id: str | None


Handler = Callable[[Any], Awaitable[None] | None]


class ScheduleEvents:
    """A small in-process bus: handlers by event type, a failing handler never stops the others."""

    def __init__(self) -> None:
        self._handlers: defaultdict[type[ScheduleEvent], list[Handler]] = defaultdict(list)

    def subscribe[E: ScheduleEvent](
        self, kind: type[E], handler: Callable[[E], Awaitable[None] | None]
    ) -> Callable[[], None]:
        """Call ``handler`` for every event of type ``kind``; returns the unsubscribe function."""
        self._handlers[kind].append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers[kind]:
                self._handlers[kind].remove(handler)

        return unsubscribe

    async def publish(self, event: ScheduleEvent) -> int:
        """Deliver ``event`` to its subscribers (in order); returns how many handled it."""
        delivered = 0
        for handler in list(self._handlers.get(type(event), ())):
            try:
                outcome = handler(event)
                if inspect.isawaitable(outcome):
                    await outcome
                delivered += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("schedule_event_handler_failed", event=type(event).__name__)
        return delivered

    def publish_sync(self, event: ScheduleEvent) -> int:
        """Deliver from synchronous code; handlers that are coroutines are not supported here."""
        delivered = 0
        for handler in list(self._handlers.get(type(event), ())):
            try:
                outcome = handler(event)
                if inspect.isawaitable(outcome):
                    raise TypeError("an asynchronous handler needs ScheduleEvents.publish")
                delivered += 1
            except Exception:
                log.exception("schedule_event_handler_failed", event=type(event).__name__)
        return delivered
