"""The runtime settings the proactive scheduler obeys (R-CFG-003, R-PRO-002, R-PRO-003).

The commands of round 11 (``/主动``, ``/暂停``, ``/恢复``) change these keys and nothing else; the
scheduler reads them on every tick, so a change is felt within one tick:

``proactive.enabled``
    ``/主动 开|关``.  Off means a quota of zero and every message refused (``disabled``).
``proactive.daily_min`` / ``proactive.daily_max``
    ``/主动 <最少>-<最多>``: the range of the day's quota.  A value of ``None`` means "the
    configuration's" (``proactive.daily_min`` / ``daily_max``), so a command that resets the range
    only needs to clear the setting.  The range the day plan was drawn with is
    :class:`~twin.schedule.plan_builder.QuotaRange`.
``engine.paused_until`` and ``paused``
    ``/暂停 <时长>`` and ``/恢复``: nothing is sent while a pause is in force.
"""

from __future__ import annotations

from datetime import datetime

from twin.config.runtime import (
    ENGINE_PAUSED_UNTIL,
    PAUSED,
    PROACTIVE_DAILY_MAX,
    PROACTIVE_DAILY_MIN,
    PROACTIVE_ENABLED,
    RuntimeSettings,
)
from twin.config.settings import ProactiveConfig


def daily_range(runtime: RuntimeSettings, config: ProactiveConfig) -> tuple[int, int]:
    """The range of messages per day in force: the setting, or the configuration's."""
    low: int | None = runtime.get(PROACTIVE_DAILY_MIN)
    high: int | None = runtime.get(PROACTIVE_DAILY_MAX)
    first = config.daily_min if low is None else low
    last = config.daily_max if high is None else high
    return first, max(first, last)


def is_enabled(runtime: RuntimeSettings) -> bool:
    return bool(runtime.get(PROACTIVE_ENABLED))


def is_paused(runtime: RuntimeSettings, now: datetime) -> bool:
    """Whether a pause (``/暂停``) is in force at ``now``."""
    if runtime.get(PAUSED):
        return True
    until = runtime.get(ENGINE_PAUSED_UNTIL)
    return until is not None and until > now
