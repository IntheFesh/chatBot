"""The DeepSeek client (R-LLM-001 to R-LLM-006, R-LLM-009, R-LLM-010, R-LLM-012).

Every call to DeepSeek goes through :class:`DeepSeekClient`.  What happens to a call:

1. the budget may refuse background work (:class:`~twin.llm.budget.BudgetGate`; replies are
   never refused) and may switch thinking off; a paused one-time batch refuses its calls;
2. messages are normalised (only ``role`` and ``content`` are ever sent - a ``reasoning_content``
   of an earlier turn is dropped, R-LLM-002), images are prepared and attached to the user
   message (R-LLM-004), and every piece of text passes :mod:`twin.llm.redaction` (R-PRIV-002);
3. the request is built: thinking is switched explicitly with
   ``{"thinking": {"type": "enabled"|"disabled"}}`` (DeepSeek thinks by default), the effort goes in
   ``reasoning_effort`` and only when thinking, and ``temperature`` and the two penalties are
   left out when thinking because the API ignores them (a warning is logged once);
4. the call is retried on 429, 5xx, connection errors and timeouts with exponential backoff and
   jitter, limited by a semaphore, and guarded by a circuit breaker (R-LLM-005); 400/401/402/
   403/422 are not retried and 401/402/403 raise an alert at once;
5. usage (cache hit and miss tokens, completion tokens) becomes a cost via the price table and
   the peak calendar and is written to ``cost_ledger`` (R-LLM-006); the token estimator learns
   from the real prompt size (R-LLM-012); the cache monitor sees the hit rate (R-LLM-010).

API facts were checked against the official documentation on 2026-10-09 (:mod:`twin.llm.official`).
Messages and replies are never logged above DEBUG level; at DEBUG the logger redacts them.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import random
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import openai
from pydantic import BaseModel, ValidationError

from twin.clock import Clock
from twin.config.settings import DeepSeekConfig
from twin.llm.budget import BudgetGate
from twin.llm.capabilities import DOCUMENTED, LlmCapabilities
from twin.llm.errors import (
    ApiError,
    AuthenticationFailedError,
    BudgetDeniedError,
    InsufficientBalanceError,
    InvalidRequestError,
    LlmConfigError,
    LoopAffinityError,
    RetriesExhaustedError,
    StructuredOutputError,
)
from twin.llm.health import LlmHealth
from twin.llm.images import (
    ImageInput,
    PreparedImage,
    attach_images,
    count_images,
    estimate_encoded_bytes,
    prepare_image,
    validate_image_placement,
)
from twin.llm.layout import CacheMonitor, PromptLayout
from twin.llm.ledger import LedgerRecord, LedgerStore
from twin.llm.official import (
    MANY_IMAGES_THRESHOLD,
    MAX_IMAGES_PER_REQUEST,
    MAX_TOTAL_IMAGE_BYTES,
    REASONING_EFFORTS,
    VISION_MODELS,
)
from twin.llm.onetime import BatchGuard
from twin.llm.pricing import Pricing
from twin.llm.redaction import RedactionResult, redact
from twin.llm.reliability import (
    CircuitBreaker,
    Failure,
    FailureKind,
    RetryPolicy,
    classify,
    safe_message,
)
from twin.llm.scope import current_call_scope
from twin.llm.tokens import TokenEstimator
from twin.llm.types import (
    DAILY,
    LIVE_PURPOSES,
    OFFLINE_PURPOSES,
    ChatMessage,
    ChatResult,
    ContentPart,
    JsonResult,
    LedgerTag,
    Purpose,
    Usage,
)
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger

log = get_logger("twin.llm.deepseek")

JSON_MAX_TOKENS = 4096
JSON_THINKING_MAX_TOKENS = 16384
ALERT_COOLDOWN_S = 600.0
CALIBRATION_SAVE_EVERY = 20
DEADLINE_SLACK_S = 5.0
MAX_ECHOED_REPLY_CHARS = 2000
MAX_VALIDATION_ERRORS = 5

Redactor = Callable[[str], RedactionResult]
Schema = type[BaseModel] | Mapping[str, Any]


# ------------------------------------------------------------------ request building


def normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[ChatMessage]:
    """Copy ``messages`` keeping only ``role`` and ``content``.

    Everything else - above all a ``reasoning_content`` taken from an earlier reply - is
    dropped: the model's chain of thought is never sent back (R-LLM-002).
    """
    if not messages:
        raise ValueError("messages must not be empty")
    cleaned: list[ChatMessage] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"message {index} has an invalid role {role!r}")
        content = message.get("content")
        if isinstance(content, str):
            cleaned.append({"role": role, "content": content})
        elif isinstance(content, Sequence):
            parts: list[ContentPart] = []
            for part in content:
                kind = part.get("type") if isinstance(part, Mapping) else None
                if kind == "text":
                    parts.append({"type": "text", "text": str(part["text"])})
                elif kind == "image_url":
                    parts.append({"type": "image_url", "image_url": dict(part["image_url"])})  # type: ignore[typeddict-item]
                else:
                    raise ValueError(f"message {index} has a content part of unknown type {kind!r}")
            cleaned.append({"role": role, "content": parts})
        else:
            raise ValueError(f"message {index} has no usable content")
    return cleaned


def redact_messages(
    messages: Sequence[ChatMessage], redactor: Redactor = redact
) -> tuple[list[ChatMessage], dict[str, int]]:
    """Messages with every text redacted, and how many replacements were made per kind."""
    counts: dict[str, int] = {}

    def clean(text: str) -> str:
        result = redactor(text)
        for kind, number in result.counts().items():
            counts[kind] = counts.get(kind, 0) + number
        return result.text

    out: list[ChatMessage] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": clean(content)})
            continue
        parts: list[ContentPart] = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": clean(part["text"])})
            else:
                parts.append(part)
        out.append({"role": message["role"], "content": parts})
    return out, counts


def schema_text(schema: Schema) -> str:
    """The JSON schema of ``schema`` as compact JSON text."""
    body = schema.model_json_schema() if isinstance(schema, type) else dict(schema)
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def with_json_instruction(
    messages: Sequence[ChatMessage], schema: Schema | None
) -> list[ChatMessage]:
    """Add the instruction JSON output needs: the word "json" and the expected shape.

    The text goes at the end of the last user message so everything before it keeps its bytes
    (and its cache hits).  A schema, when given, is described there too.
    """
    instruction = "Reply with a single JSON object only, without code fences or commentary."
    if schema is not None:
        instruction = (
            "Reply with a single JSON object only, without code fences or commentary, "
            f"that matches this JSON schema: {schema_text(schema)}"
        )
    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        message = result[index]
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            result[index] = {"role": "user", "content": f"{content}\n\n{instruction}"}
        else:
            parts: list[ContentPart] = [*content, {"type": "text", "text": instruction}]
            result[index] = {"role": "user", "content": parts}
        return result
    result.append({"role": "user", "content": instruction})
    return result


@dataclass(frozen=True)
class RequestBody:
    """Arguments for ``chat.completions.create`` and the parameters left out on purpose."""

    kwargs: dict[str, Any]
    ignored: tuple[str, ...]


def build_request(
    *,
    model: str,
    messages: Sequence[ChatMessage],
    thinking: bool,
    reasoning_effort: str,
    temperature: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> RequestBody:
    """Build the request exactly as DeepSeek expects it (R-LLM-002, R-LLM-003).

    Thinking is on by default at DeepSeek, so both states are sent explicitly.  With thinking
    enabled the API ignores ``temperature`` and the penalties; they are not sent and their names
    are returned in ``ignored``.  ``reasoning_effort`` is only sent when thinking.
    """
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
    kwargs: dict[str, Any] = {"model": model, "messages": list(messages)}
    ignored: list[str] = []
    extra: dict[str, Any]
    if thinking:
        extra = {"thinking": {"type": "enabled"}, "reasoning_effort": reasoning_effort}
        for name, value in (
            ("temperature", temperature),
            ("presence_penalty", presence_penalty),
            ("frequency_penalty", frequency_penalty),
        ):
            if value is not None:
                ignored.append(name)
    else:
        extra = {"thinking": {"type": "disabled"}}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    kwargs["extra_body"] = extra
    return RequestBody(kwargs, tuple(ignored))


def parse_usage(raw: Mapping[str, Any]) -> Usage:
    """Token counts from a response dictionary (cache fields are DeepSeek specific)."""
    usage = raw.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        hit_tokens, miss_tokens = 0, prompt
    else:
        hit_tokens = int(hit or 0)
        miss_tokens = int(miss) if miss is not None else max(0, prompt - hit_tokens)
    details = usage.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    return Usage(prompt, completion, hit_tokens, miss_tokens, reasoning)


# ----------------------------------------------------------------------------- client


def _digest(secret: str) -> bytes:
    """SHA-256 of a key: what is compared and remembered instead of a second copy of the key."""
    return hashlib.sha256(secret.encode("utf-8")).digest()


@dataclass
class _Connection:
    """The SDK client of one API key, and how many requests are using it right now."""

    sdk: openai.AsyncOpenAI
    secret: str  # kept to scrub the key out of error messages of the requests on this connection
    digest: bytes
    leases: int = 0
    retired: bool = False

    async def close(self) -> None:
        with contextlib.suppress(Exception):  # a connection that cannot close is gone anyway
            await self.sdk.close()


class DeepSeekClient:
    """Async client for the DeepSeek chat API with budget, ledger and safety built in."""

    def __init__(
        self,
        *,
        config: DeepSeekConfig,
        pricing: Pricing,
        clock: Clock,
        api_key: Callable[[], str],
        ledger: LedgerStore | None = None,
        alerts: AlertSink | None = None,
        capabilities: Callable[[], LlmCapabilities] | None = None,
        breaker: CircuitBreaker | None = None,
        estimator: TokenEstimator | None = None,
        cache_monitor: CacheMonitor | None = None,
        budget: BudgetGate | None = None,
        batches: BatchGuard | None = None,
        retry: RetryPolicy | None = None,
        rng: random.Random | None = None,
        redactor: Redactor = redact,
        save_calibration: Callable[[TokenEstimator], None] | None = None,
        deadline_slack_s: float = DEADLINE_SLACK_S,
        health: LlmHealth | None = None,
    ) -> None:
        self._config = config
        self._pricing = pricing
        self._clock = clock
        self._api_key = api_key
        self._ledger = ledger
        self._alerts = alerts
        self._capabilities = capabilities or (lambda: DOCUMENTED)
        self._breaker = breaker or CircuitBreaker(clock)
        self._health = health
        self._breaker.listen(on_open=self._on_circuit_open)
        if health is not None:
            health.register_breaker(self._breaker)  # the health check reads its state (D-701)
        self._estimator = estimator or TokenEstimator()
        self._cache = cache_monitor
        self._budget = budget
        self._batches = batches
        self._retry = retry or RetryPolicy()
        self._rng = rng or random.Random()  # noqa: S311 - retry jitter, not security
        self._redactor = redactor
        self._save_calibration = save_calibration
        self._deadline_slack_s = deadline_slack_s
        # offline work may hold all but one of the slots; the last is for the live conversation
        self._offline_slots = max(1, config.max_concurrency - 1)
        # the limits and the connection pool belong to one event loop: see :meth:`_enter`
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._offline_semaphore = asyncio.Semaphore(self._offline_slots)
        self._conn: _Connection | None = None  # what new requests use (see ``_lease``)
        self._closing: set[asyncio.Future[None]] = set()  # retired connections being closed
        self._warned: set[str] = set()
        self._alerted_at: dict[str, float] = {}
        self._unsaved_samples = 0
        if self._pricing.has_model(config.vision_model) and (
            self._pricing.resolve(config.vision_model) not in VISION_MODELS
        ):
            raise LlmConfigError(
                f"deepseek.vision_model {config.vision_model!r} cannot read images; "
                f"use one of: {', '.join(sorted(VISION_MODELS))}"
            )

    # ------------------------------------------------------------------ plumbing

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def estimator(self) -> TokenEstimator:
        return self._estimator

    def model_for(self, purpose: Purpose, *, has_images: bool = False) -> str:
        """The configured model for a purpose (vision model for anything with images)."""
        if has_images or purpose is Purpose.CAPTION:
            return self._config.vision_model
        if purpose in OFFLINE_PURPOSES:
            return self._config.offline_model
        return self._config.chat_model

    # -------------------------------------------------------------- the connection

    def _lease(self) -> _Connection:
        """The connection a request will use, made (and the key read) on first use.

        A request holds its lease from here until it has its answer.  When the key changes, the
        connection that carries the old one is retired: requests already on it finish, new ones
        get a fresh connection, and the retired one is closed when the last lease is returned
        (D-702).
        """
        conn = self._conn
        if conn is None:
            secret = self._api_key()
            conn = _Connection(sdk=self._make_sdk(secret), secret=secret, digest=_digest(secret))
            self._conn = conn
        conn.leases += 1
        return conn

    def _make_sdk(self, secret: str) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(base_url=self._config.base_url, api_key=secret, max_retries=0)

    async def _return(self, conn: _Connection) -> None:
        """Give a lease back; the last one of a retired connection closes it."""
        conn.leases -= 1
        if conn.retired and conn.leases == 0:
            await conn.close()

    def reset_key(self) -> None:
        """Drop the connection that holds the API key: the next request reads the key again.

        Requests in flight are not cut off; their connection is closed when they are done.  Used
        when the key was replaced or deleted (:meth:`refresh_key`) and when DeepSeek rejects it.
        """
        conn, self._conn = self._conn, None
        if conn is None:
            return
        conn.retired = True
        if conn.leases == 0:  # nothing on it: closing is a coroutine, so hand it to the loop
            self._retire_idle(conn)

    def _retire_idle(self, conn: _Connection) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return  # its pool died with its loop
        task = loop.create_task(conn.close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def refresh_key(self) -> bool:
        """Compare the key in the credential store with the one in use; ``True`` if it changed.

        The store is read in a thread (the Windows credential manager blocks) and only a SHA-256
        of what it says is compared with the one of the key in use: no second copy of the key is
        kept and nothing about it is logged.  A key that was deleted or cannot be read counts as
        changed, so the next request fails at once with the store's clear message instead of
        going on with a key that is gone.
        """
        conn = self._conn
        if conn is None:
            return False  # not used yet: the first request reads the key anyway
        try:
            fresh: bytes | None = await asyncio.to_thread(self._digest_of_stored_key)
        except Exception as exc:  # deleted, or the store is unavailable: either way, start over
            log.warning("deepseek_key_unreadable", reason=type(exc).__name__)
            fresh = None
        if conn is not self._conn:
            return False  # replaced while we were reading
        if fresh is not None and hmac.compare_digest(fresh, conn.digest):
            return False
        self.reset_key()
        log.info("deepseek_key_changed", readable=fresh is not None)
        return True

    def _digest_of_stored_key(self) -> bytes:
        return _digest(self._api_key())

    async def on_state_change(self, old_version: int, new_version: int) -> None:
        """The state watcher's call (``twin secrets set`` bumps the version): look at the key."""
        await self.refresh_key()

    async def aclose(self) -> None:
        """Close the connection (after the requests in flight); a later request opens a new one."""
        if self._health is not None:
            self._health.unregister_breaker(self._breaker)
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.retired = True
            if conn.leases == 0:
                await conn.close()
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)

    def _enter(self) -> None:
        """Bind the client to the running event loop and make sure the health record sees it.

        The SDK's connection pool and a contended semaphore belong to the loop they were used on.
        The application has one loop, but a command may call ``asyncio.run`` several times and
        every test has a loop of its own.  When the loop has changed and the old one is over, the
        pool and the limits are made again for the new one; when the old one is still running
        (another thread), sharing would corrupt both, so that is an error (D-701).
        """
        if self._health is not None:
            self._health.register_breaker(self._breaker)  # again, if the client was closed
        loop = asyncio.get_running_loop()
        if loop is self._loop:
            return
        previous = self._loop
        if previous is not None and previous.is_running():
            raise LoopAffinityError(
                "the DeepSeek client is in use on another running event loop; "
                "one process shares one client on one loop"
            )
        if previous is not None:
            self._conn = None  # its connections died with the old loop
        self._loop = loop
        self._semaphore = asyncio.Semaphore(self._config.max_concurrency)
        self._offline_semaphore = asyncio.Semaphore(self._offline_slots)

    @contextlib.asynccontextmanager
    async def _slot(self, purpose: Purpose) -> AsyncIterator[None]:
        """A concurrency slot: the live conversation may use all of them, the rest all but one.

        The offline limit is taken first, so a call that is waiting for it holds no slot of the
        shared limit and cannot keep the reply out (R-NFR-001, D-701).  With a single slot
        nothing can be kept back: everything shares it.
        """
        if purpose in LIVE_PURPOSES or self._offline_slots >= self._config.max_concurrency:
            async with self._semaphore:
                yield
            return
        async with self._offline_semaphore, self._semaphore:
            yield

    def _alert(
        self,
        key: str,
        category: str,
        title: str,
        *,
        severity: str = "warning",
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Raise an alert at most once per cooldown for ``key``."""
        now = self._clock.monotonic()
        last = self._alerted_at.get(key)
        if last is not None and now - last < ALERT_COOLDOWN_S:
            return
        self._alerted_at[key] = now
        if self._alerts is not None:
            self._alerts.raise_alert(
                category, title, severity=severity, detail=detail, dedup_key=key
            )

    def _on_circuit_open(self, failures: int) -> None:
        log.error("deepseek_circuit_open", failures=failures)
        self._alert(
            "llm_circuit_open",
            "llm_circuit",
            f"DeepSeek circuit breaker opened after {failures} consecutive failures",
            severity="critical",
            detail={"failures": failures},
        )

    # -------------------------------------------------------------------- images

    async def _prepare_images(
        self, images: Sequence[ImageInput], existing: int, caps: LlmCapabilities
    ) -> list[PreparedImage]:
        many = existing + len(images) >= MANY_IMAGES_THRESHOLD

        def work() -> list[PreparedImage]:
            return [
                prepare_image(
                    image.read(), detail=image.detail, capabilities=caps, many_images=many
                )
                for image in images
            ]

        return await asyncio.to_thread(work)

    # ---------------------------------------------------------------------- chat

    async def chat(
        self,
        messages: Sequence[ChatMessage] | PromptLayout,
        *,
        purpose: Purpose | str,
        thinking: bool = False,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        max_tokens: int | None = None,
        json_schema: Schema | None = None,
        json_mode: bool = False,
        images: Sequence[ImageInput] | None = None,
        model: str | None = None,
        tag: LedgerTag = DAILY,
        capabilities: LlmCapabilities | None = None,
        redactor: Redactor | None = None,
    ) -> ChatResult:
        """One chat completion.  See the module docstring for everything this does.

        ``json_schema`` (a pydantic model class or a JSON schema mapping) or ``json_mode=True``
        switches on JSON output.  ``tag`` selects the budget account: ``one_time`` batch calls
        are recorded separately and never degrade the daily budget.  ``capabilities`` overrides
        what the M0 probe stored (used by the probe itself).
        """
        requested = Purpose(purpose)
        purpose = requested
        scope = current_call_scope()
        if (
            scope is not None
        ):  # booked as evaluation / a one-time batch; the model stays the caller's
            purpose, tag = scope.purpose, scope.tag
        caps = capabilities or self._capabilities()
        layout = messages if isinstance(messages, PromptLayout) else None
        raw_messages: Sequence[ChatMessage] = (
            messages.messages() if isinstance(messages, PromptLayout) else messages
        )
        working = normalize_messages(raw_messages)

        use_json = json_mode or json_schema is not None
        thinking = self._gate(purpose, thinking, tag)
        if use_json and thinking and not caps.json_in_thinking:
            # the probe found that this API turns JSON down in thinking mode (D-117): ask without
            # thinking instead of failing on every call (the probe itself passes the documented
            # capabilities, so it keeps testing thinking)
            log.info("thinking_disabled_for_json", purpose=purpose.value)
            thinking = False
        effort = reasoning_effort or self._config.reasoning_effort

        image_count = count_images(working)
        prepared: list[PreparedImage] = []
        if images:
            prepared = await self._prepare_images(images, image_count, caps)
            working = attach_images(working, prepared)
        validate_image_placement(working)
        total_images = count_images(working)
        if total_images > MAX_IMAGES_PER_REQUEST:
            raise LlmConfigError(f"a request may hold at most {MAX_IMAGES_PER_REQUEST} images")
        if estimate_encoded_bytes(prepared) > MAX_TOTAL_IMAGE_BYTES:
            raise LlmConfigError("the images of one request may total at most 64 MiB")

        chosen = model or self.model_for(requested, has_images=total_images > 0)
        self._pricing.price_for(chosen)  # refuse early if the cost could not be computed
        if total_images and self._pricing.resolve(chosen) not in VISION_MODELS:
            raise LlmConfigError(f"model {chosen!r} cannot read images")

        if use_json:
            working = with_json_instruction(working, json_schema)
        sent, redacted = redact_messages(working, redactor or self._redactor)
        if redacted:
            log.info("outbound_redacted", purpose=purpose.value, replacements=redacted)

        body = build_request(
            model=chosen,
            messages=sent,
            thinking=thinking,
            reasoning_effort=effort,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            json_mode=use_json,
        )
        for name in body.ignored:
            if name not in self._warned:
                self._warned.add(name)
                log.warning("parameter_ignored_in_thinking_mode", parameter=name)
        log.debug("llm_request", purpose=purpose.value, model=chosen, messages=sent)

        timeout_s = float(
            self._config.timeout_s.thinking if thinking else self._config.timeout_s.non_thinking
        )
        return await self._send(
            body=body,
            purpose=purpose,
            model=chosen,
            thinking=thinking,
            tag=tag,
            timeout_s=timeout_s,
            sent=sent,
            layout=layout,
            image_count=total_images,
        )

    def _gate(self, purpose: Purpose, thinking: bool, tag: LedgerTag) -> bool:
        """Apply the batch and budget rules; returns the thinking flag to use."""
        if tag.batch_id and self._batches is not None:
            self._batches.before_call(tag.batch_id)
        if tag.account != "daily" or self._budget is None:
            return thinking
        if not self._budget.allow(purpose.value):
            raise BudgetDeniedError(purpose.value, self._budget.limits().level)
        if thinking:
            limits = self._budget.limits()
            allowed = (
                limits.chat_thinking_allowed
                if purpose is Purpose.REPLY
                else limits.planner_thinking_allowed
                if purpose in (Purpose.PLAN, Purpose.PROACTIVE)
                else True
            )
            if not allowed:
                log.info("thinking_disabled_by_budget", purpose=purpose.value, level=limits.level)
                return False
        return thinking

    async def _send(
        self,
        *,
        body: RequestBody,
        purpose: Purpose,
        model: str,
        thinking: bool,
        tag: LedgerTag,
        timeout_s: float,
        sent: Sequence[ChatMessage],
        layout: PromptLayout | None,
        image_count: int,
    ) -> ChatResult:
        attempts = 0
        self._enter()
        while True:
            attempts += 1
            self._breaker.before_call()
            started_at = self._clock.now_utc()
            started = self._clock.monotonic()
            conn: _Connection | None = None
            try:
                async with self._slot(purpose):
                    conn = self._lease()
                    try:
                        async with asyncio.timeout(timeout_s + self._deadline_slack_s):
                            response = await conn.sdk.chat.completions.create(
                                timeout=timeout_s, **body.kwargs
                            )
                    finally:
                        await self._return(conn)
            except asyncio.CancelledError:
                self._breaker.abandon()
                raise
            except Exception as exc:
                failure = self._failure_of(exc, conn.secret if conn is not None else "")
                if failure is None:
                    self._breaker.abandon()
                    raise
                if not failure.retryable:
                    self._breaker.record_success()  # the service answered
                    self._note_attempt(failure.kind not in (FailureKind.AUTH, FailureKind.BALANCE))
                    self._raise_for(failure, conn)
                self._breaker.record_failure()
                self._note_attempt(False)
                log.warning(
                    "llm_attempt_failed",
                    purpose=purpose.value,
                    attempt=attempts,
                    kind=failure.kind.value,
                    status=failure.status,
                )
                if attempts > self._retry.max_retries:
                    raise RetriesExhaustedError(
                        f"DeepSeek call failed after {attempts} attempts: {failure.message}",
                        attempts=attempts,
                        last_status=failure.status,
                    ) from None
                await self._clock.sleep(
                    self._retry.delay(attempts, self._rng, failure.retry_after_s)
                )
                continue
            self._breaker.record_success()
            self._note_attempt(True)
            latency_ms = max(0, round((self._clock.monotonic() - started) * 1000))
            return await self._finish(
                response=response,
                purpose=purpose,
                model=model,
                thinking=thinking,
                tag=tag,
                started_at=started_at,
                latency_ms=latency_ms,
                attempts=attempts,
                sent=sent,
                layout=layout,
                image_count=image_count,
            )

    def _note_attempt(self, ok: bool) -> None:
        """Tell the health record how an attempt ended (R-OPS-003)."""
        if self._health is not None:
            self._health.record(ok)

    def _failure_of(self, exc: BaseException, secret: str) -> Failure | None:
        secrets = (secret,) if secret else ()
        if isinstance(exc, TimeoutError):
            return Failure(FailureKind.TIMEOUT, True, None, "request timed out")
        return classify(exc, secrets)

    def _raise_for(self, failure: Failure, conn: _Connection | None) -> None:
        """Turn a non-retryable failure into its exception (and alert where it matters)."""
        if failure.kind is FailureKind.AUTH:
            if conn is not None and conn is self._conn:
                self.reset_key()  # the key may have been replaced: read it again next time
            self._alert(
                "llm_auth",
                "llm_auth",
                "DeepSeek rejected the API key (401/403)",
                severity="critical",
                detail={"status": failure.status},
            )
            raise AuthenticationFailedError(
                f"DeepSeek rejected the API key: {failure.message}",
                status=failure.status,
                request_id=failure.request_id,
            )
        if failure.kind is FailureKind.BALANCE:
            self._alert(
                "llm_balance",
                "llm_balance",
                "DeepSeek account balance is exhausted (402)",
                severity="critical",
                detail={"status": failure.status},
            )
            raise InsufficientBalanceError(
                f"DeepSeek balance exhausted: {failure.message}",
                status=failure.status,
                request_id=failure.request_id,
            )
        if failure.kind is FailureKind.INVALID:
            raise InvalidRequestError(
                f"DeepSeek rejected the request ({failure.status}): {failure.message}",
                status=failure.status,
                request_id=failure.request_id,
            )
        raise ApiError(
            f"DeepSeek answered {failure.status}: {failure.message}",
            status=failure.status,
            request_id=failure.request_id,
        )

    async def _finish(
        self,
        *,
        response: Any,
        purpose: Purpose,
        model: str,
        thinking: bool,
        tag: LedgerTag,
        started_at: datetime,
        latency_ms: int,
        attempts: int,
        sent: Sequence[ChatMessage],
        layout: PromptLayout | None,
        image_count: int,
    ) -> ChatResult:
        raw = response.model_dump()
        choices = raw.get("choices") or []
        if not choices:
            raise ApiError("DeepSeek returned no choices", retryable=False)
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        returned = message.get("reasoning_content") or None
        reasoning = returned if thinking else None
        usage = parse_usage(raw)
        cost = self._pricing.cost(usage, model, started_at)
        request_id = raw.get("id")
        result = ChatResult(
            content=content,
            reasoning_content=reasoning or None,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            model=model,
            purpose=purpose,
            thinking=thinking,
            at=started_at,
            request_id=request_id,
            finish_reason=choice.get("finish_reason"),
            attempts=attempts,
            image_count=image_count,
            reasoning_returned=returned is not None,
        )
        ledger_id = await asyncio.to_thread(self._account, result, tag, sent, layout)
        log.info(
            "llm_call",
            purpose=purpose.value,
            model=model,
            thinking=thinking,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cache_hit_ratio=round(usage.cache_hit_ratio, 3),
            cost_usd=round(cost.total_usd, 6),
            peak=cost.peak,
            latency_ms=latency_ms,
            attempts=attempts,
            account=tag.account,
        )
        if reasoning:
            log.debug("llm_reasoning", purpose=purpose.value, reasoning=reasoning)
        log.debug("llm_reply", purpose=purpose.value, reply=content)
        return replace(result, ledger_id=ledger_id)

    def _account(
        self,
        result: ChatResult,
        tag: LedgerTag,
        sent: Sequence[ChatMessage],
        layout: PromptLayout | None,
    ) -> str | None:
        """Everything that follows a paid call: ledger, budget, estimator, cache statistics.

        A failure here must never lose the reply that was already paid for: it is logged and
        raised as an alert instead.
        """
        ledger_id: str | None = None
        try:
            if self._ledger is not None:
                ledger_id = self._ledger.record(
                    LedgerRecord(
                        provider="deepseek",
                        model=result.model,
                        purpose=result.purpose.value,
                        usage=result.usage,
                        cost=result.cost,
                        thinking=result.thinking,
                        latency_ms=result.latency_ms,
                        at=result.at,
                        request_id=result.request_id,
                        tag=tag,
                        image_count=result.image_count,
                    )
                )
            if self._budget is not None and tag.account == "daily":
                self._budget.note_spend()
            if tag.batch_id and self._batches is not None:
                self._batches.after_call(tag.batch_id)
        except Exception as exc:
            log.error("ledger_write_failed", error=safe_message(str(exc)))
            self._alert(
                "llm_ledger_write",
                "llm_ledger",
                "could not record the cost of a DeepSeek call",
                severity="critical",
                detail={"error": type(exc).__name__},
            )
        if result.usage.prompt_tokens > 0:
            self._estimator.observe(sent, result.usage.prompt_tokens)
            self._unsaved_samples += 1
            if self._save_calibration is not None and (
                self._unsaved_samples >= CALIBRATION_SAVE_EVERY
            ):
                try:
                    self._save_calibration(self._estimator)
                    self._unsaved_samples = 0
                except Exception as exc:  # losing a calibration sample is harmless
                    log.warning("calibration_save_failed", error=safe_message(str(exc)))
        if layout is not None and self._cache is not None:
            self._cache.observe(layout, result.usage, purpose=result.purpose.value)
        return ledger_id

    # ----------------------------------------------------------------- JSON output

    async def chat_json[T: BaseModel](
        self,
        messages: Sequence[ChatMessage] | PromptLayout,
        schema: type[T],
        *,
        purpose: Purpose | str,
        thinking: bool = False,
        max_tokens: int | None = None,
        retries: int = 1,
        **options: Any,
    ) -> JsonResult[T]:
        """A reply validated against a pydantic model (R-LLM-003).

        If the reply is not valid JSON for ``schema``, the error is sent back to the model and
        the call is repeated once (``retries``); a second failure raises
        :class:`StructuredOutputError` (offline jobs are retried by the queue, online callers
        degrade).  Both calls are recorded in the ledger; ``total_cost_usd`` adds them up.
        """
        limit = max_tokens or (JSON_THINKING_MAX_TOKENS if thinking else JSON_MAX_TOKENS)
        base = messages.messages() if isinstance(messages, PromptLayout) else list(messages)
        conversation = normalize_messages(base)
        total_cost = 0.0
        last_problem = "no attempt was made"
        for attempt in range(1, retries + 2):
            result = await self.chat(
                conversation,
                purpose=purpose,
                thinking=thinking,
                max_tokens=limit,
                json_schema=schema,
                **options,
            )
            total_cost += result.cost_usd
            try:
                value = parse_json_reply(result.content, schema)
            except ValueError as problem:
                last_problem = str(problem)
                log.warning(
                    "structured_output_invalid",
                    purpose=Purpose(purpose).value,
                    attempt=attempt,
                    problem=last_problem[:200],
                    finish_reason=result.finish_reason,
                )
                conversation = [
                    *conversation,
                    {
                        "role": "assistant",
                        "content": result.content[:MAX_ECHOED_REPLY_CHARS] or "(empty)",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"That reply could not be used: {last_problem} "
                            "Reply again with only the corrected JSON object."
                        ),
                    },
                ]
                continue
            return JsonResult(value=value, chat=result, attempts=attempt, total_cost_usd=total_cost)
        raise StructuredOutputError(
            f"no valid {schema.__name__} after {retries + 1} attempts: {last_problem}",
            attempts=retries + 1,
        )


def parse_json_reply[T: BaseModel](content: str, schema: type[T]) -> T:
    """Validate ``content`` as ``schema``; raises ``ValueError`` describing what is wrong.

    A reply wrapped in a code fence is accepted.  The message names the fields and error kinds
    but never echoes the reply.
    """
    text = content.strip()
    if not text:
        raise ValueError("the reply was empty.")
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        problems = []
        for error in exc.errors(include_input=False, include_url=False)[:MAX_VALIDATION_ERRORS]:
            where = ".".join(str(part) for part in error["loc"]) or "(root)"
            problems.append(f"{where}: {error['msg']}")
        raise ValueError("invalid JSON for the schema - " + "; ".join(problems)) from exc
