"""A manually advanced clock for deterministic time tests."""

from __future__ import annotations

import asyncio
import heapq
from datetime import UTC, datetime, timedelta

DEFAULT_START = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)


class ManualClock:
    """Implements :class:`twin.clock.Clock`; time moves only when the test says so.

    ``sleep()`` suspends until :meth:`advance` moves time past the wake-up point.
    ``advance`` wakes sleepers in order, giving each a chance to run before the next.
    """

    def __init__(self, start: datetime = DEFAULT_START) -> None:
        if start.tzinfo is None:
            raise ValueError("ManualClock needs an aware start time")
        self._now = start.astimezone(UTC)
        self._mono = 1000.0
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0
        self.sleeps: list[float] = []

    def now_utc(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._waiters, (self._mono + seconds, self._seq, future))
        await future

    def set_time(self, moment: datetime) -> None:
        """Jump the wall clock (monotonic time moves by the same amount if forward)."""
        delta = (moment - self._now).total_seconds()
        self._now = moment.astimezone(UTC)
        if delta > 0:
            self._mono += delta

    def tick(self, seconds: float) -> None:
        """Advance time without waking sleepers (for synchronous tests)."""
        self._now += timedelta(seconds=seconds)
        self._mono += seconds

    @property
    def pending_sleepers(self) -> int:
        return sum(1 for _, _, fut in self._waiters if not fut.done())

    async def settle(self, rounds: int = 25) -> None:
        for _ in range(rounds):
            await asyncio.sleep(0)

    async def advance(self, seconds: float) -> None:
        """Advance time, waking every sleeper whose wake-up point is passed."""
        target = self._mono + seconds
        await self.settle()
        while self._waiters and self._waiters[0][0] <= target:
            wake, _, future = heapq.heappop(self._waiters)
            step = max(0.0, wake - self._mono)
            self._now += timedelta(seconds=step)
            self._mono += step
            if not future.done():
                future.set_result(None)
            await self.settle()
        remaining = target - self._mono
        if remaining > 0:
            self._now += timedelta(seconds=remaining)
            self._mono = target
        await self.settle()
