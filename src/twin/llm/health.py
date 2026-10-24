"""What the DeepSeek client has lately been through, for the health check (R-OPS-003).

Every attempt of every client of the process is counted here (one :class:`LlmHealth` per
services container, :func:`llm_health_of`), together with the state of the circuit breakers
(R-LLM-005).  The health monitor reads the share of failed attempts over a window and whether a
breaker is open; nothing else about the calls is kept (no text, no purpose), so there is nothing
to protect.

"A breaker is open" is read from the breaker *objects* that clients register
(:meth:`LlmHealth.register_breaker`, held weakly), not counted from open and close events: the
events can repeat (a failed half-open probe) or never come (a client that is dropped while its
breaker is open), and a counter that is off by one stays "open" for ever (round 17, D-701).  A
half-open breaker is letting a probe through and does not count as open.
"""

from __future__ import annotations

import threading
import weakref
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from twin.clock import Clock
from twin.llm.reliability import BreakerState, CircuitBreaker

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
        self._breakers: weakref.WeakSet[CircuitBreaker] = weakref.WeakSet()
        self._lock = threading.Lock()

    def record(self, ok: bool) -> None:
        """One attempt that went through (``ok``) or failed."""
        with self._lock:
            self._events.append((self._clock.monotonic(), ok))

    def register_breaker(self, breaker: CircuitBreaker) -> None:
        """Watch ``breaker``; registering one twice is registering it once."""
        with self._lock:
            self._breakers.add(breaker)

    def unregister_breaker(self, breaker: CircuitBreaker) -> None:
        """Stop watching ``breaker`` (its client was closed); unknown breakers are ignored."""
        with self._lock:
            self._breakers.discard(breaker)

    def snapshot(self, window_s: float) -> LlmHealthSnapshot:
        cutoff = self._clock.monotonic() - window_s
        with self._lock:
            recent = [ok for at, ok in self._events if at >= cutoff]
            breakers = list(self._breakers)
        return LlmHealthSnapshot(
            calls=len(recent),
            failures=sum(1 for ok in recent if not ok),
            circuit_open=any(breaker.state is BreakerState.OPEN for breaker in breakers),
        )


def llm_health_of(services: Services) -> LlmHealth:
    """The one :class:`LlmHealth` of this process (made on first use)."""
    found = services.extras.get(EXTRAS_KEY)
    if isinstance(found, LlmHealth):
        return found
    health = LlmHealth(services.clock)
    services.extras[EXTRAS_KEY] = health
    return health
