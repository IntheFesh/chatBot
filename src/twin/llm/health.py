"""What the DeepSeek client has lately been through, for the health check (R-OPS-003).

Every attempt of every client of the process is counted here - the engine, the job worker and
the commands each have a client of their own, but one :class:`LlmHealth` per services container
(:func:`llm_health_of`) - together with the state of the circuit breakers (R-LLM-005).  The
health monitor reads the share of failed attempts over a window and whether a breaker is open;
nothing else about the calls is kept (no text, no purpose), so there is nothing to protect.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from twin.clock import Clock

if TYPE_CHECKING:
    from twin.services import Services

EXTRAS_KEY = "llm_health"
MAX_EVENTS = 2000


@dataclass(frozen=True)
class LlmHealthSnapshot:
    """The attempts of a window: how many, how many failed, and whether a breaker is open."""

    calls: int
    failures: int
    circuit_open: bool

    @property
    def error_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0


class LlmHealth:
    """A thread-safe record of recent DeepSeek attempts."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._events: deque[tuple[float, bool]] = deque(maxlen=MAX_EVENTS)
        self._open_breakers = 0
        self._lock = threading.Lock()

    def record(self, ok: bool) -> None:
        """One attempt that went through (``ok``) or failed."""
        with self._lock:
            self._events.append((self._clock.monotonic(), ok))

    def breaker_opened(self) -> None:
        with self._lock:
            self._open_breakers += 1

    def breaker_closed(self) -> None:
        with self._lock:
            self._open_breakers = max(0, self._open_breakers - 1)

    def snapshot(self, window_s: float) -> LlmHealthSnapshot:
        cutoff = self._clock.monotonic() - window_s
        with self._lock:
            recent = [ok for at, ok in self._events if at >= cutoff]
            return LlmHealthSnapshot(
                calls=len(recent),
                failures=sum(1 for ok in recent if not ok),
                circuit_open=self._open_breakers > 0,
            )


def llm_health_of(services: Services) -> LlmHealth:
    """The one :class:`LlmHealth` of this process (made on first use)."""
    found = services.extras.get(EXTRAS_KEY)
    if isinstance(found, LlmHealth):
        return found
    health = LlmHealth(services.clock)
    services.extras[EXTRAS_KEY] = health
    return health
