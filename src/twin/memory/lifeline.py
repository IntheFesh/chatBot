"""The life line: what the bot's day was like, made up and kept consistent (R-MEM-005, R-MEM-011).

This round owns the data and the rules; who writes it comes later.  Round 08 generates each
day's plan when she "wakes" and stores it with :meth:`LifelineStore.replace_plan`; the engine
(round 09) calls :meth:`LifelineStore.add_improvised` through the extractor when the bot lets a
new detail about herself slip in conversation; round 10 reads the day to find something to share.

Everything here is a bot invention (``source`` is ``plan`` or ``improvised``).  A real fact that
contradicts an entry marks it ``invalidated`` (:mod:`twin.memory.conflict`, R-MEM-011); an
invalidated entry is not shown and not counted, but stays in the table.

:meth:`LifelineStore.check_consistency` is the consistency check of the day: entries must have
readable times that lie inside the day, must not overlap, and must not put her in the middle of
the deep sleep her routine model expects (R-ACT-006 ``typical_state``).  :meth:`context_for`
gives the generator what a consistent day has to respect: the last days' entries, the real facts
that name that day, her usual sleep.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from itertools import pairwise

from twin.memory.localdate import MemoryClock
from twin.memory.memory import Memory
from twin.memory.records import FactRecord, LifelineRecord
from twin.memory.store import NewEvent
from twin.memory.visible import occurrence_offset
from twin.ops.logging import get_logger
from twin.profile.api import load_activity_model
from twin.profile.localtime import format_minute
from twin.schedule.daytype import DayTypeCalendar
from twin.schedule.time_service import PlanUnavailableError, TimeService

log = get_logger("twin.memory.lifeline")

SLEEP_WORDS = ("睡", "休息", "起床", "午觉", "补觉", "小憩", "打盹", "赖床")
MAX_RECENT_DAYS = 14


@dataclass(frozen=True)
class PlannedEvent:
    """One entry of a day plan, as the generator hands it over."""

    activity: str
    start: str | None = None  # HH:MM local
    end: str | None = None
    place: str | None = None
    mood: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ConsistencyIssue:
    kind: str  # bad_time | overlap | outside_day | during_sleep
    event_id: str
    other_id: str | None = None


@dataclass(frozen=True)
class ConsistencyReport:
    day: date
    checked: int
    issues: tuple[ConsistencyIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class LifelineContext:
    """What a new day's plan has to fit."""

    day: date
    previous_days: tuple[tuple[date, tuple[str, ...]], ...] = ()
    facts_for_the_day: tuple[FactRecord, ...] = ()
    sleep: tuple[str, str] | None = None  # usual (fall asleep, wake) clock times of that day type
    notes: tuple[str, ...] = field(default_factory=tuple)


def minutes_of(clock: str | None) -> int | None:
    """Minutes after midnight of ``HH:MM``; ``None`` for a missing or unreadable time."""
    if not clock:
        return None
    hours, _, minutes = clock.partition(":")
    if not (hours.isdigit() and minutes.isdigit() and len(hours) <= 2 and len(minutes) == 2):
        return None
    h, m = int(hours), int(minutes)
    return h * 60 + m if 0 <= h <= 23 and 0 <= m <= 59 else None


class LifelineStore:
    """Writing and reading the life line (see the module description)."""

    def __init__(self, memory: Memory, *, time_service: TimeService | None = None) -> None:
        self._memory = memory
        self._store = memory.store
        self._clock: MemoryClock = memory.clock
        self._time = time_service

    # ------------------------------------------------------------------ writing

    def replace_plan(self, day: date, events: Sequence[PlannedEvent]) -> list[LifelineRecord]:
        """Store the plan of ``day``, replacing an earlier plan; improvised entries stay."""
        self._memory.refresh()
        old = [
            e.id for e in self._store.events(day=day, include_invalid=True) if e.source == "plan"
        ]
        self._store.delete_events(old)
        zone = self._clock.bot_zone().key
        stored = [
            self._store.add_event(
                NewEvent(
                    local_date=day,
                    timezone=zone,
                    activity=event.activity,
                    source="plan",
                    start_local=event.start,
                    end_local=event.end,
                    place=event.place,
                    mood=event.mood,
                    detail=event.detail,
                )
            )
            for event in events
        ]
        self._store.mark_bot_online(self._memory.services.clock.now_utc())
        self._memory.refresh()
        return stored

    def add_improvised(
        self,
        day: date,
        event: PlannedEvent,
        *,
        fact_id: str | None = None,
    ) -> LifelineRecord:
        """A detail the bot made up in conversation, written into the day it belongs to."""
        stored = self._store.add_event(
            NewEvent(
                local_date=day,
                timezone=self._clock.bot_zone().key,
                activity=event.activity,
                source="improvised",
                start_local=event.start,
                end_local=event.end,
                place=event.place,
                mood=event.mood,
                detail=event.detail,
                fact_id=fact_id,
            )
        )
        self._store.mark_bot_online(self._memory.services.clock.now_utc())
        self._memory.refresh()
        return stored

    def invalidate(self, event_id: str, *, by_fact: str | None = None) -> bool:
        """Mark an entry as contradicted by something real (it is no longer shown)."""
        done = self._store.invalidate_event(
            event_id, by=by_fact, at=self._memory.services.clock.now_utc()
        )
        self._memory.refresh()
        return done

    # ------------------------------------------------------------------ reading

    def day(self, day: date) -> list[LifelineRecord]:
        """The entries of ``day`` that still count, in time order."""
        self._memory.refresh()

        def order(event: LifelineRecord) -> tuple[int, datetime]:
            start = minutes_of(event.start_local)  # unreadable or missing: after the timed ones
            return (24 * 60 if start is None else start, event.created_at)

        return sorted(
            (e for e in self._memory.corpus.events.values() if e.active and e.local_date == day),
            key=order,
        )

    def recent(self, today: date, days: int = 3) -> dict[date, list[LifelineRecord]]:
        """The entries of the ``days`` days before ``today`` (oldest day first)."""
        found: dict[date, list[LifelineRecord]] = {}
        for back in range(min(days, MAX_RECENT_DAYS), 0, -1):
            day = today - timedelta(days=back)
            entries = self.day(day)
            if entries:
                found[day] = entries
        return found

    def at(self, moment: datetime) -> LifelineRecord | None:
        """The entry whose time span contains ``moment`` (bot-local time), if any."""
        local = moment.astimezone(self._clock.bot_zone())
        now = local.hour * 60 + local.minute
        for event in self.day(local.date()):
            start, end = minutes_of(event.start_local), minutes_of(event.end_local)
            if start is not None and end is not None and start <= now < end:
                return event
        return None

    # ------------------------------------------------------------- consistency

    def check_consistency(self, day: date) -> ConsistencyReport:
        """Check the entries of ``day`` and stamp them as checked (R-MEM-005)."""
        entries = self.day(day)
        issues: list[ConsistencyIssue] = []
        spans: list[tuple[int, int, LifelineRecord]] = []
        for event in entries:
            start, end = minutes_of(event.start_local), minutes_of(event.end_local)
            if (event.start_local and start is None) or (event.end_local and end is None):
                issues.append(ConsistencyIssue("bad_time", event.id))
                continue
            if start is not None and end is not None:
                if end <= start:
                    issues.append(ConsistencyIssue("outside_day", event.id))
                    continue
                spans.append((start, end, event))
        spans.sort(key=lambda span: span[0])
        for (_, end, first), (start, _, second) in pairwise(spans):
            if start < end:
                issues.append(ConsistencyIssue("overlap", second.id, first.id))
        issues.extend(self._sleep_issues(day, spans))
        self._store.stamp_events_checked(
            [e.id for e in entries], self._memory.services.clock.now_utc()
        )
        self._memory.refresh()
        return ConsistencyReport(day, len(entries), tuple(issues))

    def _sleep_issues(
        self, day: date, spans: Sequence[tuple[int, int, LifelineRecord]]
    ) -> list[ConsistencyIssue]:
        typical: Callable[[int], str] | None = None
        issues: list[ConsistencyIssue] = []
        for start, end, event in spans:
            if any(word in event.activity for word in SLEEP_WORDS):
                continue
            middle = (start + end) // 2
            state = self._planned_state(day, middle)
            if state is None:
                typical = typical or self._typical_state(day)
                state = typical(middle)
            if state == "deep_sleep":
                issues.append(ConsistencyIssue("during_sleep", event.id))
        return issues

    def _planned_state(self, day: date, minute: int) -> str | None:
        """Her state at that clock time by the day plan; ``None`` when no plan decides it."""
        if self._time is None:
            return None
        try:
            return self._time.her_state(self._time.local_to_utc(day, minute)).kind
        except PlanUnavailableError:
            return None

    def _typical_state(self, day: date) -> Callable[[int], str]:
        """Her usual state by the routine model, for a day that has no plan (R-ACT-006)."""
        model = load_activity_model(self._memory.services, "live")
        if model is None:
            return lambda minute: "free"
        kind, following = self._day_types(day)

        def state(minute: int) -> str:
            return model.typical_state(
                time(minute // 60, minute % 60),
                kind,
                next_day_type=following,
                weekday=day.weekday(),
            )

        return state

    def _day_types(self, day: date) -> tuple[str, str]:
        if self._time is not None:
            return self._time.day_type(day), self._time.day_type(day + timedelta(days=1))
        calendar = self._calendar()
        key = self._clock.bot_zone().key
        return calendar.day_type(day, key), calendar.day_type(day + timedelta(days=1), key)

    def _calendar(self) -> DayTypeCalendar:
        return DayTypeCalendar(
            zone_countries=self._memory.services.settings.safety.timezone_country
        )

    def context_for(self, day: date, *, days: int = 3) -> LifelineContext:
        """What the plan of ``day`` must respect: recent days, real facts, her usual sleep."""
        self._memory.refresh()
        previous = tuple(
            (when, tuple(e.line() for e in entries))
            for when, entries in self.recent(day, days).items()
        )
        facts = tuple(
            fact
            for fact in self._memory.corpus.dated_facts()
            if fact.current
            and fact.source != "bot_invented"
            and fact.event_date is not None
            and occurrence_offset(fact.event_date, fact.recurrence, day) == 0
        )
        sleep: tuple[str, str] | None = None
        model = load_activity_model(self._memory.services, "live")
        if model is not None:
            window = model.sleep.window_for(self._day_types(day)[0])
            if window is not None:
                sleep = (format_minute(window.onset_min), format_minute(window.wake_min))
        return LifelineContext(day, previous, facts, sleep)
