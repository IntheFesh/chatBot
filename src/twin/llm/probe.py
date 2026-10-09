"""The M0 probe of the DeepSeek API (R-LLM-013).

``twin llm probe`` needs a real API key.  It makes a few dozen tiny requests (well under one
US dollar-cent per request; images are drawn by :mod:`twin.llm.synth_images`, no real photo is
ever used) and records seven things:

1. **thinking_toggle** - thinking on and off both succeed; on returns ``reasoning_content``,
   off does not;
2. **cache_hit** - the same long prefix sent again gets ``prompt_cache_hit_tokens`` > 0 (the
   documentation says a cache hit needs a persisted identical prefix and that building takes
   seconds, so the probe waits and tries again a few times);
3. **vision** - a JPEG, a PNG and a multi-frame GIF are each accepted; the GIF result is only
   recorded;
4. **json_output** - JSON output parses with thinking off and with thinking on (the proactive
   planner relies on the latter);
5. **detail_param** - the image ``detail`` parameter is accepted (requests with and without it);
6. **image_tokens** - the billed tokens per image for several sizes, taken as the difference to
   a text-only request;
7. **latency_cost** - time and cost of every request.

M0 passes when checks 1, 2 and 4 succeed and, within check 3, JPEG and PNG succeed.  Checks 5 to
7 are measurements and only have to be recorded.  What the probe learns changes the client's
behaviour through :class:`~twin.llm.capabilities.LlmCapabilities` (``detail`` omitted if
rejected, GIF reduced to a still frame if rejected, measured image tokens for estimates).

Stored result (read by the M0 gate evaluator of round 09b): settings key ``m0.llm_probe`` holds
:meth:`ProbeReport.to_json` with ``schema_version`` 1; :func:`load_probe_summary` returns the
typed view ``ProbeSummary``.  Probe calls are recorded in ``cost_ledger`` on the ``one_time``
account with the batch id ``probe-<run id>`` and the purpose ``probe``.
"""

from __future__ import annotations

import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from twin.clock import Clock
from twin.llm.capabilities import LlmCapabilities, save_capabilities
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import (
    ApiError,
    AuthenticationFailedError,
    CircuitOpenError,
    InsufficientBalanceError,
    InvalidRequestError,
    LlmError,
    StructuredOutputError,
)
from twin.llm.images import ImageInput
from twin.llm.reliability import safe_message
from twin.llm.synth_images import SCENE_COLOR, SCENE_SHAPE, draw_gif, draw_jpeg, draw_png
from twin.llm.types import ChatMessage, ChatResult, LedgerTag, Purpose
from twin.ops.logging import get_logger
from twin.storage.ids import new_id
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.llm.probe")

SCHEMA_VERSION = 1
PROBE_KEY = "m0.llm_probe"
HISTORY_KEY = "m0.llm_probe.history"
HISTORY_LIMIT = 10

CHECKS: dict[int, str] = {
    1: "thinking_toggle",
    2: "cache_hit",
    3: "vision",
    4: "json_output",
    5: "detail_param",
    6: "image_tokens",
    7: "latency_cost",
}
GATE_CHECKS = (1, 2, 3, 4)  # the others are measurements
IMAGE_SIZES = ((64, 64), (256, 256), (512, 512), (1024, 1024), (2048, 2048))
VISION_QUESTION = "What shape and colour is the main object? Answer in a few words."
PROBE_CAPABILITIES = LlmCapabilities()  # the probe always tests the documented behaviour


class ProbeAborted(Exception):
    """The probe cannot continue (bad key, no balance, breaker open)."""


@dataclass(frozen=True)
class ProbeConfig:
    cache_wait_s: float = 5.0
    cache_retry_wait_s: float = 10.0
    cache_retries: int = 2
    json_samples: int = 3
    prefix_sentences: int = 150
    image_sizes: tuple[tuple[int, int], ...] = IMAGE_SIZES


@dataclass(frozen=True)
class RequestRecord:
    """One request of the probe: technical facts only, never content."""

    label: str
    ok: bool
    latency_ms: int
    cost_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    reasoning_tokens: int = 0
    status: int | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> RequestRecord:
        return cls(**raw)


@dataclass
class CheckResult:
    number: int
    id: str
    title: str
    gate: bool  # counts toward the M0 verdict
    passed: bool
    ran: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    requests: list[RequestRecord] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "id": self.id,
            "title": self.title,
            "gate": self.gate,
            "passed": self.passed,
            "ran": self.ran,
            "metrics": self.metrics,
            "notes": self.notes,
            "requests": [r.to_json() for r in self.requests],
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> CheckResult:
        return cls(
            number=int(raw["number"]),
            id=str(raw["id"]),
            title=str(raw["title"]),
            gate=bool(raw["gate"]),
            passed=bool(raw["passed"]),
            ran=bool(raw.get("ran", True)),
            metrics=dict(raw.get("metrics", {})),
            notes=[str(n) for n in raw.get("notes", [])],
            requests=[RequestRecord.from_json(r) for r in raw.get("requests", [])],
        )


@dataclass
class ProbeReport:
    run_id: str
    started_at: str
    finished_at: str
    model: str
    vision_model: str
    checks: list[CheckResult]
    capabilities: LlmCapabilities
    fatal: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def m0_passed(self) -> bool:
        """The M0 verdict: every gate check passed and the probe ran to the end."""
        return self.fatal is None and all(c.passed for c in self.checks if c.gate)

    @property
    def requests(self) -> list[RequestRecord]:
        return [r for c in self.checks for r in c.requests]

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.requests)

    def check(self, check_id: str) -> CheckResult:
        return next(c for c in self.checks if c.id == check_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "live": True,
            "model": self.model,
            "vision_model": self.vision_model,
            "m0_passed": self.m0_passed,
            "fatal": self.fatal,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "capabilities": self.capabilities.to_json(),
            "checks": [c.to_json() for c in self.checks],
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ProbeReport:
        return cls(
            run_id=str(raw["run_id"]),
            started_at=str(raw["started_at"]),
            finished_at=str(raw["finished_at"]),
            model=str(raw["model"]),
            vision_model=str(raw["vision_model"]),
            checks=[CheckResult.from_json(c) for c in raw["checks"]],
            capabilities=LlmCapabilities.from_json(raw.get("capabilities")),
            fatal=raw.get("fatal"),
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
        )


class ProbeAnswer(BaseModel):
    city: str
    population_millions: float


# ------------------------------------------------------------------------ the probe


def _long_prefix(sentences: int) -> str:
    lines = [
        f"Probe filler sentence {i}: the synthetic fox number {i} jumps over the synthetic dog."
        for i in range(sentences)
    ]
    return "This text is synthetic filler for a cache test.\n" + "\n".join(lines)


def _is_rejection(exc: LlmError) -> bool:
    return isinstance(exc, InvalidRequestError)


class LlmProbe:
    """Runs the seven checks against the live API."""

    def __init__(
        self,
        client: DeepSeekClient,
        *,
        clock: Clock,
        config: ProbeConfig | None = None,
        run_id: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._clock = clock
        self._config = config or ProbeConfig()
        self.run_id = run_id or f"probe-{new_id()}"
        self._tag = LedgerTag("one_time", self.run_id)
        self._progress = progress or (lambda _message: None)

    # ----------------------------------------------------------------- plumbing

    async def _call(
        self, label: str, messages: list[ChatMessage], **kwargs: Any
    ) -> tuple[ChatResult | None, RequestRecord, LlmError | None]:
        started = self._clock.monotonic()
        try:
            result = await self._client.chat(
                messages,
                purpose=Purpose.PROBE,
                tag=self._tag,
                capabilities=PROBE_CAPABILITIES,
                **kwargs,
            )
        except (AuthenticationFailedError, InsufficientBalanceError, CircuitOpenError) as exc:
            raise ProbeAborted(str(exc)) from exc
        except LlmError as exc:
            return None, self._failed(label, started, exc), exc
        record = RequestRecord(
            label,
            True,
            result.latency_ms,
            result.cost_usd,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cache_hit_tokens=result.usage.cache_hit_tokens,
            reasoning_tokens=result.usage.reasoning_tokens,
        )
        return result, record, None

    # -------------------------------------------------------------------- checks

    async def _check_thinking(self) -> CheckResult:
        out = CheckResult(1, CHECKS[1], "thinking on and off", True, False)
        question: list[ChatMessage] = [
            {"role": "user", "content": "What is 17 times 23? Reply with the number only."}
        ]
        on, record_on, _ = await self._call("thinking_on", question, thinking=True, max_tokens=2048)
        off, record_off, _ = await self._call(
            "thinking_off", question, thinking=False, max_tokens=64
        )
        out.requests = [record_on, record_off]
        out.metrics = {
            "thinking_on_ok": on is not None,
            "thinking_on_reasoning_returned": bool(on and on.reasoning_returned),
            "thinking_on_reasoning_tokens": on.usage.reasoning_tokens if on else 0,
            "thinking_off_ok": off is not None,
            "thinking_off_reasoning_returned": bool(off and off.reasoning_returned),
        }
        out.passed = bool(on and off and on.reasoning_returned and not off.reasoning_returned)
        if on and not on.reasoning_returned:
            out.notes.append("thinking enabled but no reasoning_content came back")
        if off and off.reasoning_returned:
            out.notes.append("thinking disabled but reasoning_content came back")
        return out

    async def _check_cache(self) -> CheckResult:
        out = CheckResult(2, CHECKS[2], "context cache hit on a repeated prefix", True, False)
        prefix = _long_prefix(self._config.prefix_sentences)
        messages: list[ChatMessage] = [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Reply with the single word: ok"},
        ]
        first, record, _ = await self._call("cache_first", messages, thinking=False, max_tokens=8)
        out.requests.append(record)
        hits: list[int] = []
        if first is not None:
            waits = [self._config.cache_wait_s] + [self._config.cache_retry_wait_s] * (
                self._config.cache_retries
            )
            for index, wait in enumerate(waits, start=1):
                await self._clock.sleep(wait)
                again, record, _ = await self._call(
                    f"cache_repeat_{index}", messages, thinking=False, max_tokens=8
                )
                out.requests.append(record)
                hits.append(again.usage.cache_hit_tokens if again else 0)
                if again is not None and again.usage.cache_hit_tokens > 0:
                    break
        out.metrics = {
            "prompt_tokens": first.usage.prompt_tokens if first else 0,
            "first_cache_hit_tokens": first.usage.cache_hit_tokens if first else 0,
            "repeat_cache_hit_tokens": hits,
            "repeats": len(hits),
            "waited_s": self._config.cache_wait_s
            + self._config.cache_retry_wait_s * max(0, len(hits) - 1),
        }
        out.passed = any(hit > 0 for hit in hits)
        if first is not None and not out.passed:
            out.notes.append("the repeated request never reported cache hits")
        return out

    async def _vision_request(
        self, label: str, image: ImageInput, **kwargs: Any
    ) -> tuple[ChatResult | None, RequestRecord, LlmError | None]:
        messages: list[ChatMessage] = [{"role": "user", "content": VISION_QUESTION}]
        return await self._call(
            label, messages, images=[image], thinking=False, max_tokens=40, **kwargs
        )

    async def _check_vision(self) -> CheckResult:
        out = CheckResult(3, CHECKS[3], "images: JPEG, PNG and animated GIF", True, False)
        gif = draw_gif()
        samples = {
            "jpeg": draw_jpeg(256, 256),
            "png": draw_png(256, 256),
            "gif": gif,
        }
        results: dict[str, ChatResult | None] = {}
        rejection: dict[str, bool | None] = {}
        for name, data in samples.items():
            result, record, error = await self._vision_request(
                f"vision_{name}", ImageInput.from_bytes(data)
            )
            out.requests.append(record)
            results[name] = result
            rejection[name] = None if result is not None or error is None else _is_rejection(error)
        answers = {
            name: bool(
                result
                and (SCENE_COLOR in result.content.lower() or SCENE_SHAPE in result.content.lower())
            )
            for name, result in results.items()
        }
        out.metrics = {
            "jpeg_ok": results["jpeg"] is not None,
            "png_ok": results["png"] is not None,
            "gif_ok": results["gif"] is not None,
            "gif_rejected": rejection["gif"] is True,
            "gif_frames": 4,
            "answer_names_scene": answers,
        }
        out.passed = results["jpeg"] is not None and results["png"] is not None
        if results["gif"] is None:
            out.notes.append("the animated GIF was not accepted; clients reduce GIFs to one frame")
        return out

    async def _check_json(self) -> CheckResult:
        out = CheckResult(4, CHECKS[4], "JSON output with thinking off and on", True, False)
        messages: list[ChatMessage] = [
            {
                "role": "user",
                "content": "Give the city Paris and its population in millions as JSON.",
            }
        ]
        samples = self._config.json_samples
        successes = {False: 0, True: 0}
        parse_failures = {False: 0, True: 0}
        api_failures = {False: 0, True: 0}
        for thinking in (False, True):
            for index in range(samples):
                label = f"json_{'on' if thinking else 'off'}_{index + 1}"
                started = self._clock.monotonic()
                try:
                    result = await self._client.chat_json(
                        messages,
                        ProbeAnswer,
                        purpose=Purpose.PROBE,
                        thinking=thinking,
                        retries=0,
                        tag=self._tag,
                        capabilities=PROBE_CAPABILITIES,
                    )
                except (
                    AuthenticationFailedError,
                    InsufficientBalanceError,
                    CircuitOpenError,
                ) as exc:
                    raise ProbeAborted(str(exc)) from exc
                except StructuredOutputError as exc:
                    parse_failures[thinking] += 1
                    out.requests.append(self._failed(label, started, exc))
                    continue
                except LlmError as exc:
                    api_failures[thinking] += 1
                    out.requests.append(self._failed(label, started, exc))
                    continue
                successes[thinking] += 1
                chat = result.chat
                out.requests.append(
                    RequestRecord(
                        label,
                        True,
                        chat.latency_ms,
                        result.total_cost_usd,
                        prompt_tokens=chat.usage.prompt_tokens,
                        completion_tokens=chat.usage.completion_tokens,
                        reasoning_tokens=chat.usage.reasoning_tokens,
                    )
                )
        out.metrics = {
            "samples_per_mode": samples,
            "thinking_off_parsed": successes[False],
            "thinking_on_parsed": successes[True],
            "thinking_off_ok": successes[False] == samples,
            "thinking_on_ok": successes[True] == samples,
            "thinking_off_parse_failures": parse_failures[False],
            "thinking_on_parse_failures": parse_failures[True],
            "thinking_off_api_failures": api_failures[False],
            "thinking_on_api_failures": api_failures[True],
        }
        out.passed = successes[False] == samples and successes[True] == samples
        if successes[True] != samples:
            out.notes.append(
                f"JSON with thinking enabled parsed {successes[True]} of {samples} times; "
                "the proactive planner depends on it"
            )
        if successes[False] != samples:
            out.notes.append(f"JSON with thinking disabled parsed {successes[False]} of {samples}")
        return out

    def _failed(self, label: str, started: float, exc: LlmError) -> RequestRecord:
        latency = max(0, round((self._clock.monotonic() - started) * 1000))
        status = exc.status if isinstance(exc, ApiError) else None
        return RequestRecord(label, False, latency, 0.0, status=status, error=type(exc).__name__)

    async def _check_detail(self) -> CheckResult:
        out = CheckResult(5, CHECKS[5], "image detail parameter", False, False)
        image = draw_jpeg(256, 256)
        without, record_without, _ = await self._vision_request(
            "detail_absent", ImageInput.from_bytes(image)
        )
        with_low, record_with, error_with = await self._vision_request(
            "detail_low", ImageInput.from_bytes(image, "low")
        )
        out.requests = [record_without, record_with]
        accepted: bool | None
        if with_low is not None:
            accepted = True
        elif error_with is not None and _is_rejection(error_with):
            accepted = False
        else:
            accepted = None
        out.metrics = {
            "without_detail_ok": without is not None,
            "detail_low_ok": with_low is not None,
            "detail_accepted": accepted,
            "prompt_tokens_without_detail": without.usage.prompt_tokens if without else None,
            "prompt_tokens_detail_low": with_low.usage.prompt_tokens if with_low else None,
        }
        out.passed = accepted is not None and without is not None
        if accepted is False:
            out.notes.append("detail was rejected: the client stops sending it")
        return out

    async def _check_image_tokens(self) -> CheckResult:
        out = CheckResult(6, CHECKS[6], "billed tokens per image size", False, False)
        text_only, record, _ = await self._call(
            "image_tokens_baseline",
            [{"role": "user", "content": VISION_QUESTION}],
            thinking=False,
            max_tokens=1,
        )
        out.requests.append(record)
        table: list[dict[str, int]] = []
        complete = text_only is not None
        for width, height in self._config.image_sizes:
            if text_only is None:
                break
            result, record, _ = await self._vision_request(
                f"image_tokens_{width}x{height}", ImageInput.from_bytes(draw_jpeg(width, height))
            )
            out.requests.append(record)
            if result is None:
                complete = False
                continue
            table.append(
                {
                    "width": width,
                    "height": height,
                    "pixels": width * height,
                    "tokens": max(0, result.usage.prompt_tokens - text_only.usage.prompt_tokens),
                }
            )
        out.metrics = {
            "baseline_prompt_tokens": text_only.usage.prompt_tokens if text_only else None,
            "per_image": table,
            "max_tokens_per_image": max((row["tokens"] for row in table), default=None),
        }
        out.passed = complete and bool(table)
        return out

    def _check_latency_cost(self, checks: list[CheckResult]) -> CheckResult:
        out = CheckResult(7, CHECKS[7], "latency and cost of the requests", False, False)
        records = [r for c in checks for r in c.requests]
        done = [r for r in records if r.ok]
        latencies = sorted(r.latency_ms for r in done)
        out.metrics = {
            "requests": len(records),
            "successful": len(done),
            "total_cost_usd": round(sum(r.cost_usd for r in records), 6),
            "median_latency_ms": int(statistics.median(latencies)) if latencies else None,
            "max_latency_ms": latencies[-1] if latencies else None,
            "per_check_cost_usd": {
                c.id: round(sum(r.cost_usd for r in c.requests), 6) for c in checks
            },
        }
        out.passed = bool(done)
        return out

    # ---------------------------------------------------------------------- run

    async def run(self) -> ProbeReport:
        """Run all checks; an unusable key or balance stops the run early (``fatal``)."""
        started = self._clock.now_utc()
        steps: list[Callable[[], Awaitable[CheckResult]]] = [
            self._check_thinking,
            self._check_cache,
            self._check_vision,
            self._check_json,
            self._check_detail,
            self._check_image_tokens,
        ]
        checks: list[CheckResult] = []
        fatal: str | None = None
        for step in steps:
            number = len(checks) + 1
            self._progress(f"check {number}: {CHECKS[number]}")
            try:
                checks.append(await step())
            except ProbeAborted as exc:
                fatal = f"probe stopped at check {number}: {safe_message(str(exc))}"
                break
        for number in range(len(checks) + 1, 7):
            checks.append(
                CheckResult(
                    number, CHECKS[number], CHECKS[number], number in GATE_CHECKS, False, ran=False
                )
            )
        checks.append(self._check_latency_cost(checks))
        finished = self._clock.now_utc()
        report = ProbeReport(
            run_id=self.run_id,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            model=self._client.model_for(Purpose.PROBE),
            vision_model=self._client.model_for(Purpose.PROBE, has_images=True),
            checks=checks,
            capabilities=derive_capabilities(checks, finished),
            fatal=fatal,
        )
        log.info(
            "llm_probe_finished",
            run_id=self.run_id,
            m0_passed=report.m0_passed,
            cost_usd=round(report.total_cost_usd, 6),
        )
        return report


def derive_capabilities(checks: list[CheckResult], at: datetime) -> LlmCapabilities:
    """What the client should do differently, from the probe outcome (unknown stays default)."""
    by_id = {c.id: c for c in checks}
    base = PROBE_CAPABILITIES
    detail = (
        by_id["detail_param"].metrics.get("detail_accepted") if by_id["detail_param"].ran else None
    )
    vision = by_id["vision"]
    gif_ok = vision.metrics.get("gif_ok") if vision.ran else None
    gif_rejected = bool(vision.metrics.get("gif_rejected")) if vision.ran else False
    json_check = by_id["json_output"]
    json_on = (
        json_check.metrics.get("thinking_on_parse_failures", 0) == 0 if json_check.ran else None
    )
    table = by_id["image_tokens"].metrics.get("per_image") if by_id["image_tokens"].ran else None
    return LlmCapabilities(
        detail_supported=base.detail_supported if detail is None else bool(detail),
        gif_supported=False if gif_rejected else (True if gif_ok else base.gif_supported),
        json_in_thinking=base.json_in_thinking if json_on is None else bool(json_on),
        image_tokens=tuple((int(r["pixels"]), int(r["tokens"])) for r in table) if table else (),
        measured_at=at.isoformat(),
    )


# ------------------------------------------------------------------------ storage


@dataclass(frozen=True)
class ProbeSummary:
    """What the M0 gate evaluator (round 09b) reads."""

    run_id: str
    finished_at: str
    m0_passed: bool
    fatal: str | None
    gate_checks: dict[str, bool]  # check id -> passed, for the checks that decide M0
    measurements: dict[str, Any]  # metrics of the measurement checks

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION


def save_probe(session: Session, report: ProbeReport, clock: Clock) -> None:
    """Store the report, keep a short history, and store the learned capabilities."""
    payload = report.to_json()
    put_setting(session, PROBE_KEY, payload, clock=clock, by="llm_probe", record_history=False)
    history = get_setting(session, HISTORY_KEY, [])
    entries = list(history) if isinstance(history, list) else []
    entries.append(
        {
            "run_id": report.run_id,
            "finished_at": report.finished_at,
            "m0_passed": report.m0_passed,
            "total_cost_usd": round(report.total_cost_usd, 6),
        }
    )
    put_setting(
        session,
        HISTORY_KEY,
        entries[-HISTORY_LIMIT:],
        clock=clock,
        by="llm_probe",
        record_history=False,
    )
    if report.fatal is None:
        save_capabilities(session, report.capabilities, clock)


def load_probe_report(session: Session) -> ProbeReport | None:
    """The latest stored probe report, or ``None`` if the probe never ran."""
    raw = get_setting(session, PROBE_KEY, None)
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        return None
    return ProbeReport.from_json(raw)


def load_probe_summary(session: Session) -> ProbeSummary | None:
    report = load_probe_report(session)
    if report is None:
        return None
    return ProbeSummary(
        run_id=report.run_id,
        finished_at=report.finished_at,
        m0_passed=report.m0_passed,
        fatal=report.fatal,
        gate_checks={c.id: c.passed for c in report.checks if c.gate},
        measurements={c.id: c.metrics for c in report.checks if not c.gate},
    )
