"""``/时区 /暂停 /恢复 /主动 /作息``: her day under the user's control (round 11).

Every handler only changes a setting or a stored correction and then asks the schedule to make
the rest of today's plan again - the schedule itself (round 08) decides what a time zone, a
routine correction or a proactive range mean for the day.  What each command writes:

``/时区``      ``switch_timezone`` of the schedule (validated, remembered, announced; the plan of
               the rest of the day is made in the new zone, R-SCH-002); aliases such as 北京 or
               芝加哥 are resolved first (:func:`twin.commands.args.resolve_zone`);
``/暂停``      the runtime setting ``engine.paused_until`` (an absolute UTC moment worked out on
               the bot's clock, never longer than ``commands.pause_max_h``); the engine and the
               proactive scheduler read it and hold back until then;
``/恢复``      clears it;
``/主动``      ``proactive.daily_min`` / ``proactive.daily_max`` (0 <= min <= max <= 12) or
               ``proactive.enabled``; the two bounds are written in the order that keeps
               ``min <= max`` true at every step, because a running application may look in between;
``/作息``      :class:`~twin.profile.overrides.RoutineOverrides` (sleep, busy time, holidays),
               listed with numbers and deleted by number.

The schedule is reached through :class:`ScheduleControl`: inside ``twin run`` it is the schedule
component (which also ticks and announces), elsewhere (``twin chat --local``) the planner of the
schedule kit itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Protocol

from twin.commands import texts
from twin.commands.args import (
    ArgumentError,
    parse_clock_range,
    parse_date_range,
    parse_pause,
    parse_quota,
    resolve_zone,
)
from twin.commands.parse import fold
from twin.commands.registry import CommandCall, UsageError
from twin.commands.status import WEEKDAYS, StatusSources, format_span
from twin.config.runtime import (
    ENGINE_PAUSED_UNTIL,
    PROACTIVE_DAILY_MAX,
    PROACTIVE_DAILY_MIN,
    PROACTIVE_ENABLED,
    SettingSpec,
)
from twin.ops.logging import get_logger
from twin.profile.overrides import OverrideError, OverrideView, RoutineOverrides, parse_weekdays
from twin.schedule.planner import PlanOutcome, SwitchOutcome, TimezoneError
from twin.schedule.service import ScheduleKit
from twin.schedule.time_service import PlanUnavailableError, TimeService

log = get_logger("twin.commands.routine")

QUOTA_CEILING = 12


class ScheduleControl(Protocol):
    """What the commands ask of the schedule."""

    async def switch_timezone(self, name: str) -> SwitchOutcome: ...

    async def rebuild(self, reason: str) -> PlanOutcome: ...


class KitSchedule:
    """:class:`ScheduleControl` over the schedule kit (``twin chat --local`` has no component)."""

    def __init__(self, kit: ScheduleKit) -> None:
        self._kit = kit

    async def switch_timezone(self, name: str) -> SwitchOutcome:
        outcome = await asyncio.to_thread(self._kit.planner.switch_timezone, name, source="command")
        for event in outcome.events:
            await self._kit.events.publish(event)
        return outcome

    async def rebuild(self, reason: str) -> PlanOutcome:
        self._kit.drop_caches()
        outcome = await asyncio.to_thread(self._kit.planner.refresh, reason)
        for event in outcome.events:
            await self._kit.events.publish(event)
        return outcome


class ComponentSchedule:
    """:class:`ScheduleControl` over the running schedule component (``twin run``)."""

    def __init__(self, component: Any, kit: ScheduleKit) -> None:
        self._component = component
        self._kit = kit

    async def switch_timezone(self, name: str) -> SwitchOutcome:
        outcome: SwitchOutcome = await self._component.switch_timezone(name, source="command")
        return outcome

    async def rebuild(self, reason: str) -> PlanOutcome:
        self._kit.drop_caches()
        outcome: PlanOutcome = await self._component.rebuild(reason, force=False)
        return outcome


def moment_text(moment: datetime) -> str:
    """``10-10 周六 08:00``: a moment on the clock it is given in."""
    return f"{moment:%m-%d} {WEEKDAYS[moment.weekday()]} {moment:%H:%M}"


def her_state_text(time: TimeService) -> str | None:
    """What she is doing now and until when, or ``None`` while there is no plan."""
    try:
        state = time.her_state()
    except PlanUnavailableError:
        return None
    until = state.until.astimezone(time.bot_timezone())
    name = texts.STATUS_HER_STATES.get(str(state.kind), str(state.kind))
    return texts.TZ_HER.format(state=name, until=f"{until:%H:%M}")


class RoutineCommands:
    """The handlers of this module (see the module description)."""

    def __init__(
        self,
        sources: StatusSources,
        control: ScheduleControl,
        overrides: RoutineOverrides,
    ) -> None:
        self._s = sources
        self._control = control
        self._overrides = overrides

    # --------------------------------------------------------------------------- /时区

    async def timezone(self, call: CommandCall) -> str:
        if not call.args:
            raise UsageError("")
        time = self._s.time
        if fold(call.args) in {fold(word) for word in texts.TZ_SHOW_WORDS}:
            return self._show_zone(time)
        try:
            name = resolve_zone(call.args)
            outcome = await self._control.switch_timezone(name)
        except (ArgumentError, TimezoneError):
            raise UsageError(texts.TZ_UNKNOWN.format(name=call.args)) from None
        local = self._local_text(time)
        if not outcome.changed:
            return texts.TZ_SAME.format(zone=outcome.new_timezone, local=local)
        lines = [
            texts.TZ_SWITCHED.format(
                old=outcome.old_timezone, new=outcome.new_timezone, local=local
            )
        ]
        state = her_state_text(time)
        if state:
            lines.append(state)
        return "\n".join(lines)

    @staticmethod
    def _local_text(time: TimeService) -> str:
        now = time.now_local()
        return f"{now:%Y-%m-%d} {WEEKDAYS[now.weekday()]} {now:%H:%M}"

    def _show_zone(self, time: TimeService) -> str:
        lines = [texts.TZ_SHOW.format(zone=time.bot_timezone().key, local=self._local_text(time))]
        state = her_state_text(time)
        if state:
            lines.append(state)
        return "\n".join(lines)

    # ------------------------------------------------------------------ /暂停 和 /恢复

    async def pause(self, call: CommandCall) -> str:
        if not call.args:
            raise UsageError("")
        time, config = self._s.time, self._s.settings.commands
        try:
            target = parse_pause(
                call.args,
                time.now_local(),
                morning_hour=config.morning_hour,
                max_hours=config.pause_max_h,
            )
        except ArgumentError as exc:
            reasons = {
                "too_long": texts.PAUSE_TOO_LONG.format(hours=config.pause_max_h),
                "not_future": texts.PAUSE_NOT_FUTURE,
            }
            raise UsageError(
                reasons.get(str(exc), texts.PAUSE_UNREADABLE.format(text=call.args))
            ) from None
        await asyncio.to_thread(
            self._s.runtime.set, ENGINE_PAUSED_UNTIL, target.until, by="command"
        )
        zone = time.bot_timezone()
        return texts.PAUSE_SET.format(
            until=moment_text(target.until.astimezone(zone)),
            zone=zone.key,
            left=format_span(target.until - time.now_utc()),
        )

    async def resume(self, call: CommandCall) -> str:
        until = await asyncio.to_thread(self._s.runtime.get, ENGINE_PAUSED_UNTIL)
        active = until is not None and until > self._s.time.now_utc()
        if until is not None:
            await asyncio.to_thread(self._s.runtime.set, ENGINE_PAUSED_UNTIL, None, by="command")
        return texts.RESUME_DONE if active else texts.RESUME_NOT_PAUSED

    # --------------------------------------------------------------------------- /主动

    async def proactive(self, call: CommandCall) -> str:
        if not call.args:
            raise UsageError("")
        word = fold(call.args)
        runtime = self._s.runtime
        if word in {fold(w) for w in texts.PROACTIVE_ON_WORDS}:
            await asyncio.to_thread(runtime.set, PROACTIVE_ENABLED, True, by="command")
            reply = texts.PROACTIVE_ON
        elif word in {fold(w) for w in texts.PROACTIVE_OFF_WORDS}:
            await asyncio.to_thread(runtime.set, PROACTIVE_ENABLED, False, by="command")
            reply = texts.PROACTIVE_OFF
        else:
            try:
                low, high = parse_quota(call.args, ceiling=QUOTA_CEILING)
            except ArgumentError as exc:
                text = (
                    texts.PROACTIVE_BAD_RANGE
                    if str(exc) == "out_of_range"
                    else texts.PROACTIVE_UNREADABLE
                )
                raise UsageError(text.format(text=call.args)) from None
            await asyncio.to_thread(self._write_quota, low, high)
            reply = texts.PROACTIVE_RANGE.format(low=low, high=high)
            if not await asyncio.to_thread(runtime.get, PROACTIVE_ENABLED):
                reply += texts.PROACTIVE_RANGE_OFF
        await self._rebuild("proactive_changed")
        return reply

    def _current(self, spec: SettingSpec[Any], fallback: int) -> int:
        value = self._s.runtime.get(spec)
        return fallback if value is None else int(value)

    def _write_quota(self, low: int, high: int) -> None:
        """Store the range so that ``min <= max`` holds after every single write."""
        runtime, config = self._s.runtime, self._s.settings.proactive
        current_high = self._current(PROACTIVE_DAILY_MAX, config.daily_max)
        if low > current_high:  # the new minimum is above the old maximum: raise the maximum first
            runtime.set(PROACTIVE_DAILY_MAX, high, by="command")
            runtime.set(PROACTIVE_DAILY_MIN, low, by="command")
        else:
            runtime.set(PROACTIVE_DAILY_MIN, low, by="command")
            runtime.set(PROACTIVE_DAILY_MAX, high, by="command")

    # --------------------------------------------------------------------------- /作息

    async def routine(self, call: CommandCall) -> str:
        if not call.args:
            raise UsageError("")
        verb, _, rest = fold(call.args).partition(" ")
        rest = rest.strip()
        table: tuple[tuple[tuple[str, ...], Callable[[str], str], bool], ...] = (
            (texts.ROUTINE_SLEEP_WORDS, self._add_sleep, True),
            (texts.ROUTINE_BUSY_WORDS, self._add_busy, True),
            (texts.ROUTINE_HOLIDAY_WORDS, self._add_holiday, True),
            (texts.ROUTINE_DELETE_WORDS, self._delete, True),
            (texts.ROUTINE_VIEW_WORDS, self._view, False),
        )
        for words, handler, changes in table:
            if verb in {fold(word) for word in words}:
                reply = await asyncio.to_thread(handler, rest)
                if not changes:
                    return reply
                rebuilt = await self._rebuild("routine_changed")
                note = texts.ROUTINE_REBUILT if rebuilt else texts.ROUTINE_REBUILD_LATER
                return f"{reply}\n{note}"
        raise UsageError("")

    def _position(self, created: OverrideView) -> int:
        for number, item in enumerate(self._overrides.entries(), start=1):
            if item.id == created.id:
                return number
        return 0

    def _add_sleep(self, rest: str) -> str:
        try:
            start, end = parse_clock_range(rest)
            item = self._overrides.add_sleep(start, end, note="chat")
        except (ArgumentError, OverrideError) as exc:
            raise UsageError(texts.ROUTINE_BAD_SLEEP.format(reason=exc)) from None
        zone = self._s.time.bot_timezone().key
        return texts.ROUTINE_SLEEP_SET.format(
            start=start, end=end, zone=zone, number=self._position(item)
        )

    def _add_busy(self, rest: str) -> str:
        parts = rest.split()
        try:
            days_text, range_text = _split_busy(parts)
            days = parse_weekdays(days_text)
            start, end = parse_clock_range(range_text)
            item = self._overrides.add_busy(days, start, end, note="chat")
        except (ArgumentError, OverrideError) as exc:
            raise UsageError(texts.ROUTINE_BAD_BUSY.format(reason=exc)) from None
        names = "、".join(WEEKDAYS[d] for d in days)
        return texts.ROUTINE_BUSY_SET.format(
            days=names, start=start, end=end, number=self._position(item)
        )

    def _add_holiday(self, rest: str) -> str:
        try:
            first, last = parse_date_range(rest, self._s.time.local_date())
            item = self._overrides.add_holiday(first, last, note="chat")
        except (ArgumentError, OverrideError) as exc:
            raise UsageError(texts.ROUTINE_BAD_HOLIDAY.format(reason=exc)) from None
        return texts.ROUTINE_HOLIDAY_SET.format(
            first=first.isoformat(), last=last.isoformat(), number=self._position(item)
        )

    def _view(self, rest: str) -> str:
        entries = self._overrides.entries()
        if not entries:
            return texts.ROUTINE_EMPTY
        lines = [texts.ROUTINE_LIST_HEADER]
        for number, item in enumerate(entries, start=1):
            state = "" if item.enabled else texts.ROUTINE_LIST_DISABLED
            lines.append(
                texts.ROUTINE_LIST_LINE.format(number=number, text=item.describe(), state=state)
            )
        return "\n".join(lines)

    def _delete(self, rest: str) -> str:
        if not rest.isdigit():
            raise UsageError("")
        number = int(rest)
        entries = self._overrides.entries()
        if not 1 <= number <= len(entries):
            raise UsageError(texts.ROUTINE_NO_SUCH.format(number=number))
        item = entries[number - 1]
        self._overrides.remove(item.id)
        return texts.ROUTINE_REMOVED.format(number=number, text=item.describe())

    async def _rebuild(self, reason: str) -> bool:
        """Make the rest of today's plan again; ``False`` when that was not possible now."""
        try:
            await self._control.rebuild(reason)
        except Exception as exc:  # the change is stored; the plan catches up on the next look
            log.warning("plan_rebuild_failed", reason=reason, error=type(exc).__name__)
            return False
        return True


def _split_busy(parts: Sequence[str]) -> tuple[str, str]:
    """``周一至周五 13:00-17:00`` (or with spaces around the dash) as (weekdays, time range)."""
    if len(parts) < 2:
        raise ArgumentError("say the weekdays and the time, like 周一至周五 13:00-17:00")
    if len(parts) >= 4 and parts[-2] in set("-–—~至到"):
        return " ".join(parts[:-3]), "".join(parts[-3:])
    return " ".join(parts[:-1]), parts[-1]
