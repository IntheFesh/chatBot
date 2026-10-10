"""Retry policy, error classification and the circuit breaker (R-LLM-005)."""

from __future__ import annotations

import random

import httpx
import openai
import pytest

from tests.support.clock import ManualClock
from twin.llm.errors import CircuitOpenError
from twin.llm.reliability import (
    BreakerState,
    CircuitBreaker,
    FailureKind,
    RetryPolicy,
    classify,
    safe_message,
)


def status_error(status: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(
        status,
        request=request,
        headers=headers or {},
        json={"error": {"message": f"status {status}"}},
    )
    return openai.APIStatusError(f"status {status}", response=response, body=None)


def make_status_error(status: int, headers: dict[str, str] | None = None) -> Exception:
    """The exception class the SDK would raise for ``status``."""
    error = status_error(status, headers)
    classes: dict[int, type[openai.APIStatusError]] = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        422: openai.UnprocessableEntityError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
        503: openai.InternalServerError,
    }
    cls = classes.get(status, openai.APIStatusError)
    return cls(error.message, response=error.response, body=None)


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (429, FailureKind.RATE_LIMIT, True),
        (500, FailureKind.SERVER, True),
        (502, FailureKind.SERVER, True),
        (503, FailureKind.SERVER, True),
        (401, FailureKind.AUTH, False),
        (403, FailureKind.AUTH, False),
        (402, FailureKind.BALANCE, False),
        (400, FailureKind.INVALID, False),
        (422, FailureKind.INVALID, False),
        (404, FailureKind.OTHER, False),
    ],
)
def test_status_codes_are_classified(status: int, kind: FailureKind, retryable: bool) -> None:
    failure = classify(make_status_error(status))
    assert failure is not None
    assert (failure.kind, failure.retryable, failure.status) == (kind, retryable, status)


def test_connection_problems_and_timeouts_are_retryable() -> None:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    timeout = classify(openai.APITimeoutError(request=request))
    assert timeout is not None and timeout.kind is FailureKind.TIMEOUT and timeout.retryable
    connection = classify(openai.APIConnectionError(request=request))
    assert connection is not None and connection.kind is FailureKind.CONNECTION
    assert connection.retryable
    assert classify(ValueError("a bug of ours")) is None


def test_retry_after_is_read_and_capped() -> None:
    assert classify(make_status_error(429, {"retry-after": "12"})).retry_after_s == 12.0  # type: ignore[union-attr]
    assert classify(make_status_error(429, {"retry-after": "9999"})).retry_after_s == 60.0  # type: ignore[union-attr]
    assert classify(make_status_error(429, {"retry-after": "soon"})).retry_after_s is None  # type: ignore[union-attr]
    assert classify(make_status_error(429)).retry_after_s is None  # type: ignore[union-attr]


def test_error_text_is_scrubbed_of_secrets_and_identifiers() -> None:
    text = "Your api key: my-secret-value-42 is invalid; also sk-abcdef1234567890 and a@b.org"
    cleaned = safe_message(text, ("my-secret-value-42",))
    assert "my-secret-value-42" not in cleaned and "sk-abcdef1234567890" not in cleaned
    assert "a@b.org" not in cleaned and "***" in cleaned
    assert len(safe_message("x" * 1000)) == 300


def test_backoff_doubles_up_to_the_cap_with_jitter_around_it() -> None:
    policy = RetryPolicy(max_retries=4, base_s=1.0, cap_s=30.0)
    assert policy.max_attempts == 5
    rng = random.Random(1)
    for retry, centre in ((1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0), (9, 30.0)):
        delays = [policy.delay(retry, rng) for _ in range(200)]
        assert min(delays) >= centre * 0.5 and max(delays) < centre * 1.5
        assert max(delays) - min(delays) > centre * 0.5  # really jittered
    assert policy.delay(1, rng, retry_after_s=25.0) >= 25.0


# ----------------------------------------------------------------- circuit breaker


def test_the_breaker_opens_after_ten_straight_failures_and_a_success_resets_the_count(
    clock: ManualClock,
) -> None:
    breaker = CircuitBreaker(clock)
    opened: list[int] = []
    breaker.listen(on_open=opened.append)
    for _ in range(9):
        breaker.before_call()
        breaker.record_failure()
    breaker.before_call()
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED and breaker.snapshot().consecutive_failures == 0
    for _ in range(10):
        breaker.before_call()
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN and opened == [10]
    with pytest.raises(CircuitOpenError) as info:
        breaker.before_call()
    assert info.value.retry_after_s == pytest.approx(300.0)


def test_half_open_lets_one_probe_through_and_the_outcome_decides(clock: ManualClock) -> None:
    closed: list[bool] = []
    breaker = CircuitBreaker(clock, failure_threshold=2, on_close=lambda: closed.append(True))
    for _ in range(2):
        breaker.before_call()
        breaker.record_failure()
    clock.tick(299)
    with pytest.raises(CircuitOpenError):
        breaker.before_call()
    clock.tick(2)
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.before_call()  # the probe
    with pytest.raises(CircuitOpenError):
        breaker.before_call()  # a second concurrent call is held back
    breaker.record_failure()  # the probe failed: open again for another 300 s
    assert breaker.state is BreakerState.OPEN
    assert breaker.snapshot().retry_in_s == pytest.approx(300.0)
    clock.tick(301)
    breaker.before_call()
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED and closed == [True]
    breaker.before_call()  # closed again: calls flow


def test_an_abandoned_probe_frees_the_slot(clock: ManualClock) -> None:
    breaker = CircuitBreaker(clock, failure_threshold=1)
    breaker.before_call()
    breaker.record_failure()
    clock.tick(301)
    breaker.before_call()
    breaker.abandon()
    breaker.before_call()  # not stuck: another probe may go
    assert breaker.state is BreakerState.HALF_OPEN


def test_breaker_settings_are_validated(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(clock, failure_threshold=0)
