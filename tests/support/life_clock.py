"""A manual clock that can live through days: it knows who sleeps, and when everything is quiet.

The scenario tests (``tests/integration/life``) run the **whole** application - the state watcher,
the heartbeat, the job worker, the schedule, the engine, the proactive scheduler, the operations
components - on one clock, for simulated days.  :class:`~tests.support.clock.ManualClock` can do
that in principle, but it wakes the sleepers one at a time and waits a few real milliseconds after
each, and a day holds tens of thousands of wake-ups of the two-second state watcher alone.
:class:`LifeClock` keeps the manual clock's semantics and adds what a long run needs:

* **Who sleeps.**  Every sleeper is recorded with the module and function that asked for the sleep
  (``ops/state_watch.py:run``).  The periodic housekeeping of the application (the state watcher,
  the heartbeat, the job worker's poll, the schedule's tick ...) is told apart from the sleeps that
  *mean* something (the engine waiting for the time she answers, a pause between two bubbles, the
  proactive scheduler's next tick, the back-off of a failed request).
* **Stepping.**  :meth:`step` moves the clock on and releases **every** sleeper that is due at
  once, so the housekeeping runs once per step, not once per period.  The scenario helpers
  (:class:`~tests.support.life_world.LifeWorld`) step from one meaningful wake-up to the next, so a
  day costs a few hundred steps, not a hundred thousand.  A task that sleeps again after its wake-up
  sleeps from the new time, as it does for real.  The slow housekeeping (the health snapshot, the
  heartbeat, the daily backup ... whose work is a database write or a disk look) is let go only
  when it is ``slow_every_s`` late: it is what a day of steps would otherwise be made of.
* **Quiet.**  :meth:`settle` waits until nothing is runnable *and* no worker thread is busy: the
  event loop's ``run_in_executor`` is counted (:meth:`track_threads`), so a database commit that
  takes 30 ms (``SLOWDB_MS=30``) cannot be mistaken for "the engine is waiting".  Observing before
  the quiet is reached is the typical way a time-driven test fails on a slow disk
  (``docs/EXECUTION_NOTES.md`` item 7).
"""

from __future__ import annotations

import asyncio
import heapq
import sys
import time
from collections.abc import Coroutine, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.support.clock import DEFAULT_START, ManualClock

# The periodic sleepers of the application that keep the process healthy.  Each entry is the end of
# the path of the module that asks for the sleep.  None of them sets the pace of a scenario.
FAST_HOUSEKEEPING: frozenset[str] = frozenset(
    {
        "ops/state_watch.py",  # the two-second look at ``state_version`` (R-ARCH-006)
        "ops/jobs.py",  # the job worker's poll
        "ops/power_events.py",  # the sleep / wake detector
        "schedule/component.py",  # the schedule's tick: the day plan, the life line, the summaries
        "ilink/poller.py",  # the WeChat channel's poll, once a second
    }
)
SLOW_HOUSEKEEPING: frozenset[str] = frozenset(
    {
        "ops/components.py",  # the heartbeat
        "ops/alert_delivery.py",
        "ops/monitor.py",  # the health snapshot
        "ops/scheduler.py",  # the daily backup and the monthly mail
        "ops/login_recovery.py",
        "engine/backend_select.py",  # the style model's monitor (a scenario about it watches it)
        "learning/component.py",
        "commands/import_report.py",
        "serving/component.py",
        "probe/runner.py",  # the channel probe waits for a plan to be activated
    }
)
HOUSEKEEPING: frozenset[str] = FAST_HOUSEKEEPING | SLOW_HOUSEKEEPING
QUIET_TURNS = 8  # loop turns in a row without a worker thread busy before the world counts as quiet
SLOW_EVERY_S = 1800.0


def _label(frame: Any) -> str:
    """``dir/module.py:function`` of the code that asked for a sleep."""
    path = Path(frame.f_code.co_filename).as_posix().split("/")
    return f"{'/'.join(path[-2:])}:{frame.f_code.co_name}"


def _is_in(caller: str, group: Iterable[str]) -> bool:
    return any(caller.startswith(prefix) for prefix in group)


class LifeClock(ManualClock):
    """A :class:`ManualClock` for runs of simulated days (see the module description)."""

    def __init__(self, start: datetime = DEFAULT_START) -> None:
        super().__init__(start)
        self._callers: dict[asyncio.Future[None], str] = {}
        self._threads = 0
        self._tracking = False
        self.slow_every_s = SLOW_EVERY_S
        self.fast: set[str] = set()  # slow housekeeping that a scenario wants at every step
        self.steps = 0
        self.settling_s = 0.0  # real seconds spent waiting for quiet (what a run costs in waiting)

    # ------------------------------------------------------------------ sleeping

    def sleep(self, seconds: float) -> Coroutine[Any, Any, None]:  # type: ignore[override]
        """Sleep ``seconds`` of this clock; the caller is noted when the call is made.

        Plain ``def`` on purpose: ``asyncio.ensure_future(clock.sleep(x))`` runs the coroutine in a
        new task, so by the time it starts the code that asked for it is no longer on the stack.
        """
        return self._sleep(seconds, _label(sys._getframe(1)))

    async def _sleep(self, seconds: float, caller: str) -> None:
        self.sleeps.append(seconds)
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._waiters, (self._mono + seconds, self._seq, future))
        self._callers[future] = caller
        try:
            await future
        finally:
            self._callers.pop(future, None)

    def sleepers(self) -> list[tuple[float, str]]:
        """``(seconds from now, caller)`` of every pending sleeper, soonest first."""
        found = [
            (wake - self._mono, self._callers.get(future, "?"))
            for wake, _, future in self._waiters
            if not future.done()
        ]
        return sorted(found)

    def next_wake_in(self, ignore: Iterable[str] = HOUSEKEEPING) -> float | None:
        """Seconds until the next sleeper whose caller is not in ``ignore`` (``None``: none)."""
        skipped = tuple(ignore)
        soonest: float | None = None
        for wake, _, future in self._waiters:
            if future.done() or _is_in(self._callers.get(future, "?"), skipped):
                continue
            delta = max(0.0, wake - self._mono)
            if soonest is None or delta < soonest:
                soonest = delta
        return soonest

    def _held_back(self, future: asyncio.Future[None], wake: float) -> bool:
        """Is this sleeper slow housekeeping that is not late enough to be worth waking?"""
        caller = self._callers.get(future, "?")
        if not _is_in(caller, SLOW_HOUSEKEEPING) or _is_in(caller, self.fast):
            return False
        return self._mono - wake < self.slow_every_s

    # ------------------------------------------------------------------- stepping

    async def step(self, seconds: float) -> None:
        """Move on by ``seconds`` and release every sleeper that is due (quiet before and after)."""
        await self.settle()
        self.tick(max(0.0, seconds))
        due: list[asyncio.Future[None]] = []
        held: list[tuple[float, int, asyncio.Future[None]]] = []
        while self._waiters and self._waiters[0][0] <= self._mono:
            entry = heapq.heappop(self._waiters)
            if self._held_back(entry[2], entry[0]):
                held.append(entry)
            else:
                due.append(entry[2])
        for entry in held:
            heapq.heappush(self._waiters, entry)
        for future in due:
            if not future.done():
                future.set_result(None)
        self.steps += 1
        await self.settle()

    async def step_to(self, moment: datetime) -> None:
        await self.step((moment - self.now_utc()).total_seconds())

    # ---------------------------------------------------------------------- quiet

    def track_threads(self) -> None:
        """Count the work handed to worker threads on the running loop (``asyncio.to_thread``)."""
        loop = asyncio.get_running_loop()
        if self._tracking:
            return
        original = loop.run_in_executor

        def counting(executor: Any, func: Any, *args: Any) -> asyncio.Future[Any]:
            future = original(executor, func, *args)
            self._threads += 1
            future.add_done_callback(self._thread_done)
            return future

        loop.run_in_executor = counting  # type: ignore[method-assign,assignment]
        self._tracking = True

    def _thread_done(self, _future: asyncio.Future[Any]) -> None:
        self._threads -= 1

    @property
    def threads_busy(self) -> int:
        return self._threads

    async def settle(self, rounds: int = 25) -> None:
        """Wait until every task is at an await that needs time to pass, and no thread runs."""
        if not self._tracking:
            await super().settle(rounds)
            return
        began = time.perf_counter()
        quiet = 0
        while quiet < QUIET_TURNS:
            await asyncio.sleep(0)
            if self._threads:
                quiet = 0
                await asyncio.sleep(0.0005)  # really wait: a thread is on the disk
            else:
                quiet += 1
        self.settling_s += time.perf_counter() - began
