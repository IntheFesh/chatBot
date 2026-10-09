"""Retry policy, error classification and the circuit breaker (R-LLM-005).

* Retryable: 429, 5xx (500, 503, ...), connection errors and timeouts.  Not retryable: 400 and
  422 (the request is wrong), 401 and 403 (the key), 402 (no balance) and every other 4xx.
  The error list is the one on the DeepSeek "Error Codes" page (checked 2026-10-09).
* Retries use exponential backoff with jitter: ``base * 2**n`` capped, multiplied by a random
  factor in [0.5, 1.5), and honour a ``Retry-After`` header (capped).  At most
  :attr:`RetryPolicy.max_retries` retries follow the first attempt.
* The circuit breaker counts consecutive *retryable* failures.  After
  :attr:`CircuitBreaker.failure_threshold` (10) it opens for 300 seconds, during which calls
  fail at once; then one probe call is let through (half-open): success closes the breaker,
  failure opens it again.  A reply that proves the service is alive (even a 4xx) counts as
  success.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import openai

from twin.clock import Clock
from twin.llm.errors import CircuitOpenError
from twin.llm.redaction import redact_text

MAX_RETRY_AFTER_S = 60.0
MAX_MESSAGE_CHARS = 300
_KEY_LIKE = re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}")


class FailureKind(StrEnum):
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    AUTH = "auth"
    BALANCE = "balance"
    INVALID = "invalid"
    OTHER = "other"


@dataclass(frozen=True)
class Failure:
    """A classified API failure."""

    kind: FailureKind
    retryable: bool
    status: int | None
    message: str
    retry_after_s: float | None = None
    request_id: str | None = None


def _retry_after(exc: openai.APIStatusError) -> float | None:
    raw = exc.response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return min(MAX_RETRY_AFTER_S, max(0.0, float(raw)))
    except ValueError:
        return None


def safe_message(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Error text that is safe to log or store: secrets and identifiers removed, shortened.

    DeepSeek echoes part of a rejected API key in its 401 message, so the configured secret
    and anything shaped like a key is masked before the text goes anywhere.
    """
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "***")
    cleaned = _mask_key_like(cleaned)
    return redact_text(cleaned)[:MAX_MESSAGE_CHARS]


def _mask_key_like(text: str) -> str:
    return _KEY_LIKE.sub("sk-***", text)


def classify(exc: BaseException, secrets: tuple[str, ...] = ()) -> Failure | None:
    """Classify an exception from the OpenAI SDK; ``None`` if it is not an API failure."""
    if isinstance(exc, openai.APITimeoutError):
        return Failure(FailureKind.TIMEOUT, True, None, "request timed out")
    if isinstance(exc, openai.APIConnectionError):
        return Failure(
            FailureKind.CONNECTION,
            True,
            None,
            f"connection failed: {type(exc.__cause__ or exc).__name__}",
        )
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        message = safe_message(str(exc.message), secrets)
        request_id = exc.request_id
        if status == 429:
            return Failure(
                FailureKind.RATE_LIMIT, True, status, message, _retry_after(exc), request_id
            )
        if status >= 500:
            return Failure(FailureKind.SERVER, True, status, message, _retry_after(exc), request_id)
        if status in (401, 403):
            return Failure(FailureKind.AUTH, False, status, message, None, request_id)
        if status == 402:
            return Failure(FailureKind.BALANCE, False, status, message, None, request_id)
        if status in (400, 422):
            return Failure(FailureKind.INVALID, False, status, message, None, request_id)
        return Failure(FailureKind.OTHER, False, status, message, None, request_id)
    return None


@dataclass(frozen=True)
class RetryPolicy:
    """How often and how long to wait between attempts."""

    max_retries: int = 4
    base_s: float = 1.0
    cap_s: float = 30.0

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def delay(
        self, retry_number: int, rng: random.Random, retry_after_s: float | None = None
    ) -> float:
        """Seconds to wait before retry ``retry_number`` (1 for the first retry)."""
        backoff = float(min(self.cap_s, self.base_s * 2 ** (retry_number - 1)))
        jittered = float(backoff * (0.5 + rng.random()))
        if retry_after_s is not None:
            return max(jittered, retry_after_s)
        return jittered


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerSnapshot:
    state: BreakerState
    consecutive_failures: int
    retry_in_s: float


class CircuitBreaker:
    """Closed -> open after repeated failures -> half-open probe -> closed (or open again)."""

    def __init__(
        self,
        clock: Clock,
        *,
        failure_threshold: int = 10,
        open_seconds: float = 300.0,
        on_open: Callable[[int], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self._clock = clock
        self.failure_threshold = failure_threshold
        self.open_seconds = open_seconds
        self._on_open: list[Callable[[int], None]] = [on_open] if on_open else []
        self._on_close: list[Callable[[], None]] = [on_close] if on_close else []
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    def listen(
        self,
        *,
        on_open: Callable[[int], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Add listeners: ``on_open(failures)`` when the breaker opens, ``on_close()`` on close."""
        if on_open is not None:
            self._on_open.append(on_open)
        if on_close is not None:
            self._on_close.append(on_close)

    def _remaining(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.open_seconds - (self._clock.monotonic() - self._opened_at))

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        return BreakerState.OPEN if self._remaining() > 0 else BreakerState.HALF_OPEN

    def snapshot(self) -> BreakerSnapshot:
        return BreakerSnapshot(self.state, self._failures, self._remaining())

    def before_call(self) -> None:
        """Raise :class:`CircuitOpenError` unless a call may go out now."""
        state = self.state
        if state is BreakerState.CLOSED:
            return
        if state is BreakerState.OPEN:
            raise CircuitOpenError(self._remaining())
        if self._probe_in_flight:  # half-open: a single probe at a time
            raise CircuitOpenError(1.0)
        self._probe_in_flight = True

    def record_success(self) -> None:
        was_open = self._opened_at is not None
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False
        if was_open:
            for listener in self._on_close:
                listener()

    def record_failure(self) -> None:
        self._failures += 1
        was_half_open = self._probe_in_flight
        self._probe_in_flight = False
        if was_half_open or (self._opened_at is None and self._failures >= self.failure_threshold):
            self._opened_at = self._clock.monotonic()
            for listener in self._on_open:
                listener(self._failures)

    def abandon(self) -> None:
        """A call ended without an answer (cancelled): free the half-open probe slot."""
        self._probe_in_flight = False
