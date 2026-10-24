"""``StateWatcher``: notices CLI changes made by other processes (R-ARCH-006.2).

LIGHT commands increment ``settings.state_version`` in the same transaction as
their writes.  The watcher reads it every two seconds and, when it changed,
calls every subscriber with ``(old_version, new_version)`` so each component can
drop the caches that depend on database state (persona card, today's plan,
backend selection, ...).  Business components subscribe in later rounds.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from twin.clock import Clock
from twin.ops.logging import get_logger
from twin.storage.db import Database
from twin.storage.state import read_state_version

log = get_logger("twin.state")

POLL_INTERVAL_S = 2.0

StateListener = Callable[[int, int], Awaitable[None] | None]


class StateWatcher:
    """Polls ``state_version`` and broadcasts changes."""

    def __init__(self, db: Database, clock: Clock, *, interval_s: float = POLL_INTERVAL_S) -> None:
        self._db = db
        self._clock = clock
        self._interval_s = interval_s
        self._listeners: list[StateListener] = []
        self._version: int | None = None

    @property
    def version(self) -> int | None:
        """Last version seen (``None`` before the first poll)."""
        return self._version

    def subscribe(self, listener: StateListener) -> Callable[[], None]:
        """Register ``listener``; returns a function that unsubscribes it."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    async def poll_once(self) -> bool:
        """Read the version once; broadcast and return ``True`` if it changed.

        The first call only records the baseline.
        """
        current = await self._db.arun(read_state_version)
        previous = self._version
        self._version = current
        if previous is None or current == previous:
            return False
        log.info("state_changed", old=previous, new=current)
        for listener in list(self._listeners):
            try:
                outcome = listener(previous, current)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                log.exception("state_listener_failed")
        return True

    async def run(self) -> None:
        """Poll forever (run under a TaskSupervisor)."""
        while True:
            await self.poll_once()
            await self._clock.sleep(self._interval_s)
