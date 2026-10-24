"""Time source for the whole project.

Every piece of code obtains the current time through a :class:`Clock`
(CLAUDE.md rule 5 / task H).  This is the only module that may call
``datetime.now``, ``time.monotonic`` or ``asyncio.sleep`` directly; a scan
test (``tests/unit/test_scan_rules.py``) enforces that.

Time handling conventions:

* every ``datetime`` is timezone-aware; storage is UTC;
* ``Clock.now_utc()`` always returns an aware UTC ``datetime``;
* ``Clock.monotonic()`` is for measuring durations, never for wall time.

Components receive a clock by injection.  Places that cannot take one as a
parameter (ORM column defaults, ULID generation) use the *active* clock, which
defaults to :class:`SystemClock` and can be swapped with :func:`use_clock`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Source of wall-clock time, monotonic time and sleeping."""

    def now_utc(self) -> datetime:
        """Current time as an aware UTC datetime."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds, for measuring elapsed time."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the calling task for ``seconds``."""
        ...


class SystemClock:
    """Real clock backed by the operating system."""

    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


_active_clock: Clock = SystemClock()


def get_clock() -> Clock:
    """Return the process-wide active clock."""
    return _active_clock


def set_active_clock(clock: Clock) -> Clock:
    """Install ``clock`` as the active clock and return the previous one."""
    global _active_clock
    previous = _active_clock
    _active_clock = clock
    return previous


@contextmanager
def use_clock(clock: Clock) -> Iterator[Clock]:
    """Temporarily make ``clock`` the active clock."""
    previous = set_active_clock(clock)
    try:
        yield clock
    finally:
        set_active_clock(previous)


def now_utc() -> datetime:
    """Current aware-UTC time from the active clock."""
    return _active_clock.now_utc()


def ensure_aware(value: datetime) -> datetime:
    """Return ``value`` converted to UTC; naive datetimes are rejected."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("naive datetime is not allowed; attach a timezone (CLAUDE.md rule 5)")
    return value.astimezone(UTC)


def to_epoch(value: datetime) -> float:
    """Aware datetime to epoch seconds."""
    return ensure_aware(value).timestamp()


def from_epoch(seconds: float) -> datetime:
    """Epoch seconds to an aware UTC datetime."""
    return datetime.fromtimestamp(seconds, UTC)
