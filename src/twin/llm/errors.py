"""Exceptions of the LLM layer."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


class LlmError(Exception):
    """Base class of everything the LLM layer raises on purpose."""


class LlmConfigError(LlmError):
    """The configuration cannot work (unknown model, text-only model asked to read images)."""


class LoopAffinityError(LlmError):
    """The client is being used on two running event loops at once.

    Its connection pool and its concurrency limits belong to one loop.  A command that runs
    several loops one after the other is fine (the client moves to the new one); two loops that
    run at the same time, in two threads, are not supported.
    """


class UnknownModelError(LlmConfigError):
    """No price is configured for the model, so its cost cannot be recorded."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"no price configured for model {model!r}; add it to pricing_usd_per_mtok")


class ImageError(LlmError):
    """Base class for problems with an image input."""


class UnsupportedImageError(ImageError):
    """The bytes are not JPEG, PNG, GIF or WebP, or cannot be decoded."""


class ImagePlacementError(ImageError):
    """An image was placed in a system or assistant message (the API answers 400)."""


class ImageTooLargeError(ImageError):
    """The image still exceeds the API limits after local downscaling."""


class ApiError(LlmError):
    """The API answered with an error (or never answered)."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.request_id = request_id


class RetriesExhaustedError(ApiError):
    """A retryable failure kept happening until the attempt budget was used up."""

    def __init__(self, message: str, *, attempts: int, last_status: int | None = None) -> None:
        super().__init__(message, status=last_status, retryable=True)
        self.attempts = attempts


class AuthenticationFailedError(ApiError):
    """401 or 403: the key is wrong or lacks permission.  Not retried; alerts at once."""


class InsufficientBalanceError(ApiError):
    """402: the account is out of balance.  Not retried; alerts at once."""


class InvalidRequestError(ApiError):
    """400 or 422: the request itself is wrong.  Not retried."""


class EmptyResponseError(ApiError):
    """The API returned no content (it occasionally does in JSON mode)."""


class CircuitOpenError(LlmError):
    """The circuit breaker is open: DeepSeek failed repeatedly, calls are refused for a while."""

    def __init__(self, retry_after_s: float) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(f"DeepSeek circuit breaker is open; probing again in {retry_after_s:.0f}s")


class BudgetDeniedError(LlmError):
    """The budget level forbids this purpose right now (R-LLM-008)."""

    def __init__(self, purpose: str, level: int) -> None:
        self.purpose = purpose
        self.level = level
        super().__init__(f"purpose {purpose!r} is paused at budget level {level}")


class StructuredOutputError(LlmError):
    """The reply is not valid JSON for the requested schema, even after one retry."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class StyleModelError(LlmError):
    """The style model server failed (timeout, connection, error status, malformed reply)."""

    def __init__(self, message: str, *, status: int | None = None, kind: str = "error") -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind  # "timeout" | "unavailable" | "status" | "malformed" | "error"


# R18-D1 (D-795): the errors that say "DeepSeek cannot do this now" rather than "this work is
# wrong", with how long background work waits before it asks again.  A rejected key and an empty
# account are fixed by the user (``twin secrets set``, a top-up) and the budget by the next period,
# so an hour; the breaker probes again after five minutes.  The job worker hands a job that meets
# one of them back to the queue without counting an attempt (``twin.ops.jobs.Worker``); code that
# works through a list of items lets them through instead of counting them as failed items.
UNAVAILABLE_RETRY_S: Mapping[type[LlmError], float] = MappingProxyType(
    {
        BudgetDeniedError: 3600.0,
        CircuitOpenError: 300.0,
        AuthenticationFailedError: 3600.0,
        InsufficientBalanceError: 3600.0,
    }
)
SERVICE_UNAVAILABLE: tuple[type[LlmError], ...] = tuple(UNAVAILABLE_RETRY_S)


def unavailable_retry_s(exc: BaseException) -> float | None:
    """Seconds to wait before asking again if ``exc`` (or what it was raised from) says that
    DeepSeek cannot do the work now; ``None`` for any other failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for kind, wait in UNAVAILABLE_RETRY_S.items():
            if isinstance(current, kind):
                return wait
        current = current.__cause__
    return None
