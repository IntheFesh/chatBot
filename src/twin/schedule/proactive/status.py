"""The proactive line of ``/状态``: range, today's draw, what went out, what is next (round 10).

:class:`ProactiveStatusSource` is the ``proactive=`` source that
:meth:`~twin.commands.router.CommandRouter.from_services` and
:class:`~twin.commands.status.StatusSources` take.  It only reads: the settings (the range, the
switch), the day plan's quota, the log of today, the candidates that wait, and the platform window
as the channel's session state tells it.  Round 11 shows the same through its own commands; it
changes the settings, never the numbers here.
"""

from __future__ import annotations

from collections.abc import Callable

from twin.channel.base import SessionState
from twin.clock import Clock
from twin.commands.status import ProactiveStatus
from twin.config.runtime import RuntimeSettings
from twin.config.settings import ProactiveConfig
from twin.schedule.proactive.rules import window_refusal
from twin.schedule.proactive.settings import daily_range, is_enabled
from twin.schedule.proactive.store import CandidateStore, ProactiveLogStore
from twin.schedule.proactive.types import KIND_LABELS, REASON_LABELS
from twin.schedule.service import ScheduleKit
from twin.schedule.time_service import PlanUnavailableError


class ProactiveStatusSource:
    """Builds the :class:`~twin.commands.status.ProactiveStatus` of the moment."""

    def __init__(
        self,
        *,
        clock: Clock,
        schedule: ScheduleKit,
        runtime: RuntimeSettings,
        config: ProactiveConfig,
        log_store: ProactiveLogStore,
        candidates: CandidateStore,
        channel_state: Callable[[], SessionState],
    ) -> None:
        self._clock = clock
        self._kit = schedule
        self._runtime = runtime
        self._config = config
        self._log = log_store
        self._candidates = candidates
        self._channel_state = channel_state

    def __call__(self) -> ProactiveStatus | None:
        now = self._clock.now_utc()
        low, high = daily_range(self._runtime, self._config)
        today = self._kit.time.local_date(now)
        try:
            quota: int | None = self._kit.planner.plan_at(now).quota.total
        except PlanUnavailableError:
            quota = None
        upcoming = [row for row in self._candidates.pending() if row.window_end >= now]
        upcoming.sort(key=lambda row: row.planned_at)
        nearest = upcoming[0] if upcoming else None
        blocked = None
        try:
            refusal = window_refusal(self._channel_state())
        except Exception:  # a channel that cannot say is not a reason to hide the rest
            refusal = None
        if refusal is not None:
            blocked = REASON_LABELS.get(refusal[0].value, refusal[0].value)
        return ProactiveStatus(
            low=low,
            high=high,
            enabled=is_enabled(self._runtime),
            sent_today=self._log.count_sent_on(today),
            quota=quota,
            next_at=max(nearest.planned_at, now) if nearest is not None else None,
            next_kind=KIND_LABELS.get(nearest.kind.value) if nearest is not None else None,
            blocked=blocked,
        )
