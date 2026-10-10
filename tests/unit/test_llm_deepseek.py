"""The DeepSeek client against intercepted HTTP (R-LLM-001 to R-LLM-006, R-LLM-009, R-LLM-010)."""

from __future__ import annotations

import asyncio
import base64
import logging
import random
from collections.abc import Awaitable, Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, TypeVar

import httpx
import pytest
import respx
from pydantic import BaseModel

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, completion, error, ok, request_json
from tests.support.synthetic import mobile
from twin.config.loader import load_settings
from twin.config.settings import Settings
from twin.llm import deepseek as deepseek_module
from twin.llm.budget import BudgetManager
from twin.llm.capabilities import LlmCapabilities
from twin.llm.deepseek import (
    DeepSeekClient,
    build_request,
    normalize_messages,
    parse_json_reply,
    parse_usage,
    redact_messages,
    with_json_instruction,
)
from twin.llm.errors import (
    ApiError,
    AuthenticationFailedError,
    BudgetDeniedError,
    CircuitOpenError,
    ImagePlacementError,
    InsufficientBalanceError,
    InvalidRequestError,
    LlmConfigError,
    RetriesExhaustedError,
    StructuredOutputError,
    UnknownModelError,
)
from twin.llm.health import LlmHealth
from twin.llm.images import ImageInput
from twin.llm.layout import CacheMonitor, PromptLayout
from twin.llm.ledger import LedgerRecord, LedgerStore
from twin.llm.onetime import BatchPausedError
from twin.llm.pricing import Pricing
from twin.llm.redaction import ConsistentRedactor
from twin.llm.reliability import BreakerState, CircuitBreaker, RetryPolicy
from twin.llm.synth_images import draw_gif, draw_jpeg, draw_png
from twin.llm.tokens import TokenEstimator
from twin.llm.types import ChatMessage, CostBreakdown, LedgerTag, Purpose, Usage
from twin.schedule.time_service import BotTimeService
from twin.storage.db import Database
from twin.storage.models import CostLedger

T = TypeVar("T")

FRIDAY_PEAK = datetime(2026, 10, 16, 2, 0, tzinfo=UTC)  # Friday 02:00 UTC: peak hours
SATURDAY = datetime(2026, 10, 17, 2, 0, tzinfo=UTC)  # weekend: off-peak


def user(text: str) -> ChatMessage:
    return {"role": "user", "content": text}


class Rig:
    """A client wired to a real database, ledger and manual clock; HTTP is intercepted."""

    def __init__(
        self,
        db: Database,
        clock: ManualClock,
        *,
        overrides: dict[str, Any] | None = None,
        retry: RetryPolicy | None = None,
        capabilities: LlmCapabilities | None = None,
        with_budget: bool = False,
        batches: Any = None,
        failure_threshold: int = 10,
        deadline_slack_s: float = 5.0,
        health: LlmHealth | None = None,
    ) -> None:
        clock.set_time(FRIDAY_PEAK)
        self.db = db
        self.clock = clock
        self.settings: Settings = load_settings(None, overrides or {})
        self.time = BotTimeService(clock, lambda: "America/Chicago")
        self.ledger = LedgerStore(db, clock, self.time)
        self.alerts = RecordingAlerts()
        self.pricing = Pricing.from_settings(self.settings)
        self.estimator = TokenEstimator()
        self.cache = CacheMonitor(self.estimator)
        self.key_reads = 0
        self.saved: list[float] = []
        self.budget: BudgetManager | None = None
        if with_budget:
            self.budget = BudgetManager(
                self.settings.budget,
                examples_k=self.settings.engine.examples_k,
                ledger=self.ledger,
                time_service=self.time,
                clock=clock,
                db=db,
                alerts=self.alerts,
                ttl_s=0.0,
            )
        caps = capabilities or LlmCapabilities()
        self.breaker = CircuitBreaker(clock, failure_threshold=failure_threshold)
        self.client = DeepSeekClient(
            config=self.settings.deepseek,
            pricing=self.pricing,
            clock=clock,
            api_key=self.read_key,
            ledger=self.ledger,
            alerts=self.alerts,
            capabilities=lambda: caps,
            breaker=self.breaker,
            estimator=self.estimator,
            cache_monitor=self.cache,
            budget=self.budget,
            batches=batches,
            retry=retry or RetryPolicy(),
            rng=random.Random(7),
            save_calibration=lambda est: self.saved.append(est.factor),
            deadline_slack_s=deadline_slack_s,
            health=health,
        )

    def read_key(self) -> str:
        self.key_reads += 1
        return TEST_KEY

    def rows(self) -> list[dict[str, Any]]:
        with self.db.session() as session:
            return [
                {
                    "purpose": r.purpose,
                    "model": r.model,
                    "hit": r.cache_hit_tokens,
                    "miss": r.cache_miss_tokens,
                    "completion": r.completion_tokens,
                    "reasoning": r.reasoning_tokens,
                    "cost": r.cost_usd,
                    "peak": r.peak,
                    "thinking": r.thinking,
                    "account": r.account,
                    "batch": r.batch_id,
                    "images": r.image_count,
                    "request_id": r.request_id,
                }
                for r in session.query(CostLedger).order_by(CostLedger.at).all()
            ]

    async def drive(self, awaitable: Awaitable[T], *, step: float = 10.0) -> T:
        """Run ``awaitable`` while the manual clock is advanced to release backoff sleeps."""
        task = asyncio.ensure_future(awaitable)
        for _ in range(500):
            if task.done():
                break
            await asyncio.wait({task}, timeout=0.01)
            if not task.done():
                await self.clock.advance(step)
        return await task


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def rig(db: Database, clock: ManualClock) -> Rig:
    return Rig(db, clock)


def mock(api: respx.MockRouter, *responses: httpx.Response | Exception) -> respx.Route:
    return api.post(API).mock(side_effect=list(responses))


# ------------------------------------------------------------------- request body


def test_request_without_thinking_switches_it_off_explicitly_and_keeps_temperature() -> None:
    body = build_request(
        model="deepseek-flash",
        messages=[user("hi")],
        thinking=False,
        reasoning_effort="high",
        temperature=0.7,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        max_tokens=50,
    )
    assert body.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert body.kwargs["temperature"] == 0.7
    assert body.kwargs["presence_penalty"] == 0.1 and body.kwargs["frequency_penalty"] == 0.2
    assert body.kwargs["max_tokens"] == 50
    assert "reasoning_effort" not in body.kwargs and body.ignored == ()
    assert "response_format" not in body.kwargs


def test_request_with_thinking_enables_it_sets_the_effort_and_drops_sampling_parameters() -> None:
    body = build_request(
        model="deepseek-flash",
        messages=[user("hi")],
        thinking=True,
        reasoning_effort="max",
        temperature=0.7,
        presence_penalty=0.1,
        frequency_penalty=0.2,
    )
    assert body.kwargs["extra_body"] == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    assert not {"temperature", "presence_penalty", "frequency_penalty"} & set(body.kwargs)
    assert body.ignored == ("temperature", "presence_penalty", "frequency_penalty")


def test_request_validation_and_json_mode() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        build_request(model="m", messages=[user("x")], thinking=True, reasoning_effort="ultra")
    body = build_request(
        model="m", messages=[user("x")], thinking=False, reasoning_effort="high", json_mode=True
    )
    assert body.kwargs["response_format"] == {"type": "json_object"}


async def test_the_http_body_for_both_thinking_states(rig: Rig, api: respx.MockRouter) -> None:
    route = mock(api, ok(), ok(reasoning="想一想", content="嗯"))
    await rig.client.chat([user("你好")], purpose="reply", temperature=0.9)
    plain = request_json(route.calls.last.request)
    assert plain["model"] == "deepseek-flash"
    assert plain["thinking"] == {"type": "disabled"} and plain["temperature"] == 0.9
    assert plain["messages"] == [{"role": "user", "content": "你好"}]
    assert "reasoning_effort" not in plain and "response_format" not in plain
    assert route.calls.last.request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert str(route.calls.last.request.url) == API

    await rig.client.chat([user("你好")], purpose="reply", thinking=True, temperature=0.9)
    thought = request_json(route.calls.last.request)
    assert thought["thinking"] == {"type": "enabled"} and thought["reasoning_effort"] == "high"
    assert "temperature" not in thought


async def test_ignored_parameters_are_warned_about_once(rig: Rig, api: respx.MockRouter) -> None:
    mock(api, ok(), ok(), ok())
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("twin.llm.deepseek")
    handler = Capture(level=logging.WARNING)
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        for _ in range(3):
            await rig.client.chat([user("x")], purpose="reply", thinking=True, temperature=0.5)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    warned = [r for r in records if r.getMessage() == "parameter_ignored_in_thinking_mode"]
    assert len(warned) == 1


async def test_reasoning_content_is_never_sent_back(rig: Rig, api: respx.MockRouter) -> None:
    route = mock(api, ok(content="第一", reasoning="私下的思考过程"), ok(content="第二"))
    first = await rig.client.chat([user("问题一")], purpose="reply", thinking=True)
    assert first.reasoning_content == "私下的思考过程"
    history: list[dict[str, Any]] = [
        user("问题一"),
        # an assembling bug could carry the reasoning along; the client must drop it anyway
        {
            "role": "assistant",
            "content": first.content,
            "reasoning_content": first.reasoning_content,
        },
        user("问题二"),
    ]
    await rig.client.chat(history, purpose="reply", thinking=True)  # type: ignore[arg-type]
    sent = route.calls.last.request.content
    assert "私下的思考过程".encode() not in sent and b"reasoning_content" not in sent
    assert request_json(route.calls.last.request)["messages"][1] == {
        "role": "assistant",
        "content": "第一",
    }


async def test_reasoning_is_only_reported_when_thinking_was_requested(
    rig: Rig, api: respx.MockRouter
) -> None:
    mock(api, ok(reasoning="stray"), ok(reasoning=""), ok(reasoning="why", reasoning_tokens=5))
    off = await rig.client.chat([user("x")], purpose="reply")
    assert off.reasoning_content is None and off.thinking is False
    empty = await rig.client.chat([user("x")], purpose="reply", thinking=True)
    assert empty.reasoning_content is None
    on = await rig.client.chat([user("x")], purpose="reply", thinking=True)
    assert on.reasoning_content == "why" and on.usage.reasoning_tokens == 5


def test_messages_are_reduced_to_role_and_content() -> None:
    cleaned = normalize_messages(
        [{"role": "user", "content": "a", "name": "x", "reasoning_content": "r"}]
    )
    assert cleaned == [{"role": "user", "content": "a"}]
    for bad in (
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": "user"}],
        [{"role": "user", "content": 5}],
    ):
        with pytest.raises(ValueError):
            normalize_messages(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown type"):
        normalize_messages([{"role": "user", "content": [{"type": "audio"}]}])


# ---------------------------------------------------------------- results and cost


async def test_the_result_carries_content_usage_cost_and_ids(
    rig: Rig, api: respx.MockRouter
) -> None:
    mock(api, ok(content="早安", prompt=1000, hit=400, completion_tokens=50, request_id="abc"))
    result = await rig.client.chat([user("早")], purpose=Purpose.REPLY)
    assert result.content == "早安" and result.model == "deepseek-flash"
    assert (result.usage.prompt_tokens, result.usage.cache_hit_tokens) == (1000, 400)
    assert (result.usage.cache_miss_tokens, result.usage.completion_tokens) == (600, 50)
    assert result.request_id == "abc" and result.finish_reason == "stop" and result.attempts == 1
    expected = (400 * 0.006 + 600 * 0.30 + 50 * 1.20) / 1_000_000
    assert result.cost.peak and result.cost_usd == pytest.approx(expected)
    assert result.at == FRIDAY_PEAK and result.purpose is Purpose.REPLY
    assert result.ledger_id is not None


async def test_offpeak_calls_are_charged_at_the_discount(rig: Rig, api: respx.MockRouter) -> None:
    mock(
        api,
        ok(prompt=1000, hit=0, completion_tokens=100),
        ok(prompt=1000, hit=0, completion_tokens=100),
    )
    peak = await rig.client.chat([user("x")], purpose="reply")
    rig.clock.set_time(SATURDAY)
    off = await rig.client.chat([user("x")], purpose="reply")
    assert peak.cost.peak and not off.cost.peak
    assert off.cost_usd == pytest.approx(peak.cost_usd * 0.5)
    assert [row["peak"] for row in rig.rows()] == [True, False]


def test_usage_parsing_covers_missing_cache_fields() -> None:
    plain = parse_usage(completion(prompt=80, hit=None))
    assert (plain.cache_hit_tokens, plain.cache_miss_tokens) == (0, 80)
    only_hit = parse_usage(
        {"usage": {"prompt_tokens": 50, "completion_tokens": 1, "prompt_cache_hit_tokens": 20}}
    )
    assert (only_hit.cache_hit_tokens, only_hit.cache_miss_tokens) == (20, 30)
    assert parse_usage({}).prompt_tokens == 0


async def test_every_call_is_written_to_the_ledger(rig: Rig, api: respx.MockRouter) -> None:
    mock(api, ok(prompt=500, hit=100, completion_tokens=30, reasoning_tokens=12, reasoning="r"))
    await rig.client.chat([user("x")], purpose="summary", thinking=True)
    (row,) = rig.rows()
    assert row["purpose"] == "summary" and row["model"] == "deepseek-flash"
    assert (row["hit"], row["miss"], row["completion"], row["reasoning"]) == (100, 400, 30, 12)
    assert row["thinking"] is True and row["peak"] is True
    assert (row["account"], row["batch"], row["images"]) == ("daily", None, 0)
    assert row["request_id"] == "req-1"


async def test_one_time_calls_are_recorded_on_their_own_account(
    rig: Rig, api: respx.MockRouter
) -> None:
    mock(api, ok())
    await rig.client.chat([user("x")], purpose="persona", tag=LedgerTag("one_time", "persona-1"))
    (row,) = rig.rows()
    assert (row["account"], row["batch"]) == ("one_time", "persona-1")


async def test_a_ledger_failure_does_not_lose_the_paid_reply(
    rig: Rig, api: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock(api, ok(content="还在"))

    def broken(entry: object) -> str:
        raise RuntimeError("disk full")

    monkeypatch.setattr(rig.ledger, "record", broken)
    result = await rig.client.chat([user("x")], purpose="reply")
    assert result.content == "还在" and result.ledger_id is None
    assert [a.category for a in rig.alerts.alerts] == ["llm_ledger"]


# ------------------------------------------------------------------ model choice


def test_models_follow_the_purpose(db: Database, clock: ManualClock) -> None:
    rig = Rig(
        db,
        clock,
        overrides={
            "deepseek": {"offline_model": "deepseek-v4-pro", "chat_model": "deepseek-flash"}
        },
    )
    client = rig.client
    assert client.model_for(Purpose.REPLY) == "deepseek-flash"
    assert client.model_for(Purpose.PLAN) == "deepseek-flash"
    for purpose in (Purpose.EXTRACT, Purpose.SUMMARY, Purpose.PERSONA, Purpose.EVAL):
        assert client.model_for(purpose) == "deepseek-v4-pro"
    assert client.model_for(Purpose.CAPTION) == "deepseek-flash"
    assert client.model_for(Purpose.SUMMARY, has_images=True) == "deepseek-flash"


async def test_unknown_models_are_refused_before_anything_is_sent(
    rig: Rig, api: respx.MockRouter
) -> None:
    route = mock(api, ok())
    with pytest.raises(UnknownModelError, match="mystery"):
        await rig.client.chat([user("x")], purpose="reply", model="mystery")
    assert route.call_count == 0


async def test_retired_model_names_are_accepted_and_priced_as_flash(
    rig: Rig, api: respx.MockRouter
) -> None:
    mock(api, ok(model="deepseek-flash"))
    result = await rig.client.chat([user("x")], purpose="reply", model="deepseek-v4-flash")
    assert result.model == "deepseek-v4-flash" and result.cost_usd > 0


def test_a_vision_model_that_cannot_see_is_a_configuration_error(
    db: Database, clock: ManualClock
) -> None:
    with pytest.raises(LlmConfigError, match="cannot read images"):
        Rig(db, clock, overrides={"deepseek": {"vision_model": "deepseek-v4-pro"}})


# --------------------------------------------------------------------------- JSON


class Plan(BaseModel):
    topic: str
    score: int


async def test_json_mode_sets_the_response_format_and_the_instruction(
    rig: Rig, api: respx.MockRouter
) -> None:
    route = mock(api, ok(content='{"topic": "天气", "score": 3}'))
    result = await rig.client.chat_json([user("给个计划")], Plan, purpose="plan")
    assert result.value == Plan(topic="天气", score=3) and result.attempts == 1
    body = request_json(route.calls.last.request)
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 4096
    text = body["messages"][-1]["content"]
    assert text.startswith("给个计划") and "JSON" in text and '"topic"' in text


async def test_invalid_json_is_sent_back_once_and_the_second_reply_is_used(
    rig: Rig, api: respx.MockRouter
) -> None:
    route = mock(
        api,
        ok(content='{"topic": "天气", "score": "高"}'),
        ok(content='```json\n{"topic": "天气", "score": 4}\n```'),
    )
    result = await rig.client.chat_json([user("给个计划")], Plan, purpose="plan")
    assert result.value.score == 4 and result.attempts == 2
    assert result.total_cost_usd == pytest.approx(2 * result.chat.cost_usd)
    second = request_json(route.calls.last.request)["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "score" in second[-1]["content"] and "高" not in second[-1]["content"]
    assert len(rig.rows()) == 2  # both attempts cost money and are recorded


async def test_two_invalid_replies_fail_with_a_structured_output_error(
    rig: Rig, api: respx.MockRouter
) -> None:
    mock(api, ok(content="not json at all"), ok(content=""))
    with pytest.raises(StructuredOutputError, match="Plan") as info:
        await rig.client.chat_json([user("x")], Plan, purpose="plan")
    assert info.value.attempts == 2 and len(rig.rows()) == 2


async def test_thinking_json_calls_get_a_larger_token_allowance(
    rig: Rig, api: respx.MockRouter
) -> None:
    route = mock(api, ok(content='{"topic": "a", "score": 1}', reasoning="r"))
    await rig.client.chat_json([user("x")], Plan, purpose="plan", thinking=True)
    assert request_json(route.calls.last.request)["max_tokens"] == 16384
    mock_route = api.post(API).mock(return_value=ok(content='{"topic": "a", "score": 1}'))
    await rig.client.chat_json([user("x")], Plan, purpose="plan", max_tokens=99)
    assert request_json(mock_route.calls.last.request)["max_tokens"] == 99


def test_json_helpers() -> None:
    assert parse_json_reply('{"topic": "a", "score": 1}', Plan).score == 1
    assert parse_json_reply('```\n{"topic": "a", "score": 1}\n```', Plan).topic == "a"
    for bad, fragment in (("", "empty"), ("[1]", "invalid"), ('{"topic": "a"}', "score")):
        with pytest.raises(ValueError, match=fragment):
            parse_json_reply(bad, Plan)
    messages = with_json_instruction([user("a"), {"role": "assistant", "content": "b"}], Plan)
    assert messages[0]["content"].startswith("a\n\n") and messages[1]["content"] == "b"
    assert with_json_instruction([{"role": "system", "content": "s"}], None)[-1]["role"] == "user"
    parts: list[ChatMessage] = [
        {"role": "user", "content": [{"type": "text", "text": "t"}]},
    ]
    appended = with_json_instruction(parts, {"type": "object"})[0]["content"]
    assert isinstance(appended, list) and appended[-1]["type"] == "text"  # type: ignore[typeddict-item]


async def test_json_mode_without_a_schema_still_names_json(rig: Rig, api: respx.MockRouter) -> None:
    route = mock(api, ok(content="{}"))
    await rig.client.chat([user("x")], purpose="extract", json_mode=True)
    body = request_json(route.calls.last.request)
    assert "JSON" in body["messages"][-1]["content"]
    assert body["response_format"] == {"type": "json_object"}


# -------------------------------------------------------------------------- images


async def test_images_go_into_the_last_user_message_with_their_detail(
    rig: Rig, api: respx.MockRouter
) -> None:
    route = mock(api, ok())
    await rig.client.chat(
        [{"role": "system", "content": "rules"}, user("看看")],
        purpose="caption",
        images=[
            ImageInput.from_bytes(draw_png(), "low"),
            ImageInput.from_bytes(draw_jpeg(), "auto"),
        ],
    )
    body = request_json(route.calls.last.request)
    assert body["model"] == "deepseek-flash"
    content = body["messages"][1]["content"]
    assert [p["type"] for p in content] == ["text", "image_url", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["detail"] == "low"
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[2]["image_url"]["detail"] == "auto"
    assert body["messages"][0]["content"] == "rules"
    assert rig.rows()[0]["images"] == 2


async def test_images_in_system_or_assistant_messages_are_refused_without_a_request(
    rig: Rig, api: respx.MockRouter
) -> None:
    route = mock(api, ok())
    part: Any = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    for role in ("system", "assistant"):
        with pytest.raises(ImagePlacementError):
            await rig.client.chat(
                [user("x"), {"role": role, "content": [part]}],  # type: ignore[list-item]
                purpose="caption",
            )
    assert route.call_count == 0


async def test_a_text_only_model_cannot_be_given_images(rig: Rig, api: respx.MockRouter) -> None:
    route = mock(api, ok())
    with pytest.raises(LlmConfigError, match="cannot read images"):
        await rig.client.chat(
            [user("x")],
            purpose="caption",
            images=[ImageInput.from_bytes(draw_png())],
            model="deepseek-v4-pro",
        )
    assert route.call_count == 0


async def test_probe_findings_change_what_is_sent(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    caps = LlmCapabilities(detail_supported=False, gif_supported=False)
    rig = Rig(db, clock, capabilities=caps)
    route = mock(api, ok(), ok())
    await rig.client.chat(
        [user("x")],
        purpose="caption",
        images=[ImageInput.from_bytes(draw_gif(), "low")],
    )
    image = request_json(route.calls.last.request)["messages"][0]["content"][1]["image_url"]
    assert image["url"].startswith("data:image/png;base64,") and "detail" not in image
    # the probe can override what was stored
    await rig.client.chat(
        [user("x")],
        purpose="caption",
        images=[ImageInput.from_bytes(draw_gif(), "low")],
        capabilities=LlmCapabilities(),
    )
    image = request_json(route.calls.last.request)["messages"][0]["content"][1]["image_url"]
    assert image["url"].startswith("data:image/gif;base64,") and image["detail"] == "low"


# ------------------------------------------------------------------------ redaction


async def test_personal_identifiers_never_leave_the_machine(
    rig: Rig, api: respx.MockRouter
) -> None:
    route = mock(api, ok(), ok())
    number = mobile()
    await rig.client.chat(
        [
            {"role": "system", "content": f"联系人 {number}"},
            user(f"我的号码 {number}，邮箱 a@b.org"),
        ],
        purpose="reply",
    )
    sent = route.calls.last.request.content.decode()
    assert number not in sent and "a@b.org" not in sent
    body = request_json(route.calls.last.request)
    assert body["messages"][0]["content"] == "联系人 [手机号]"
    assert body["messages"][1]["content"] == "我的号码 [手机号]，邮箱 [邮箱]"

    consistent = ConsistentRedactor()
    await rig.client.chat(
        [user(f"{number} 与 {number}")], purpose="reply", redactor=consistent.redact
    )
    assert request_json(route.calls.last.request)["messages"][0]["content"] == (
        "[手机号#1] 与 [手机号#1]"
    )


async def test_image_data_is_not_touched_by_redaction(rig: Rig, api: respx.MockRouter) -> None:
    route = mock(api, ok())
    data = draw_png(40, 40)
    await rig.client.chat(
        [user(mobile())], purpose="caption", images=[ImageInput.from_bytes(data, "low")]
    )
    content = request_json(route.calls.last.request)["messages"][0]["content"]
    assert content[0]["text"] == "[手机号]"
    assert content[1]["image_url"]["url"].endswith(base64.b64encode(data).decode())


def test_redact_messages_counts_replacements() -> None:
    cleaned, counts = redact_messages(
        [user(f"{mobile()} {mobile()}"), {"role": "assistant", "content": "没有"}]
    )
    assert counts == {"phone": 2} and cleaned[1]["content"] == "没有"


# --------------------------------------------------------------------- reliability


@pytest.mark.parametrize(
    "failure",
    [
        error(429),
        error(500),
        error(503, "overloaded"),
        httpx.ConnectError("no route"),
        httpx.ReadTimeout("slow"),
    ],
    ids=["429", "500", "503", "connection", "timeout"],
)
async def test_retryable_failures_are_retried_with_growing_pauses(
    rig: Rig, api: respx.MockRouter, failure: httpx.Response | Exception
) -> None:
    route = mock(api, failure, failure, ok(content="终于"))
    result = await rig.drive(rig.client.chat([user("x")], purpose="reply"))
    assert result.content == "终于" and result.attempts == 3 and route.call_count == 3
    first, second = rig.clock.sleeps[:2]
    assert 0.5 <= first < 1.5 and 1.0 <= second < 3.0
    assert len(rig.rows()) == 1  # failed attempts cost nothing


async def test_after_four_retries_the_error_is_final(rig: Rig, api: respx.MockRouter) -> None:
    route = mock(api, *[error(500) for _ in range(5)])
    with pytest.raises(RetriesExhaustedError) as info:
        await rig.drive(rig.client.chat([user("x")], purpose="reply"))
    assert route.call_count == 5 and info.value.attempts == 5 and info.value.status == 500
    assert len(rig.clock.sleeps) == 4 and rig.rows() == []


async def test_retry_after_headers_are_respected(rig: Rig, api: respx.MockRouter) -> None:
    mock(api, error(429, **{"retry-after": "20"}), ok())
    await rig.drive(rig.client.chat([user("x")], purpose="reply"))
    assert rig.clock.sleeps[0] >= 20


@pytest.mark.parametrize(
    ("status", "exception"),
    [
        (400, InvalidRequestError),
        (422, InvalidRequestError),
        (401, AuthenticationFailedError),
        (403, AuthenticationFailedError),
        (402, InsufficientBalanceError),
        (404, ApiError),
    ],
)
async def test_client_errors_are_not_retried(
    rig: Rig, api: respx.MockRouter, status: int, exception: type[ApiError]
) -> None:
    route = mock(api, error(status), ok())
    with pytest.raises(exception) as info:
        await rig.client.chat([user("x")], purpose="reply")
    assert route.call_count == 1 and info.value.status == status and rig.clock.sleeps == []


async def test_auth_and_balance_failures_alert_at_once_but_only_once_per_cooldown(
    rig: Rig, api: respx.MockRouter
) -> None:
    mock(api, error(401), error(401), error(402), error(402))
    for expected in (AuthenticationFailedError, AuthenticationFailedError):
        with pytest.raises(expected):
            await rig.client.chat([user("x")], purpose="reply")
    for _ in range(2):
        with pytest.raises(InsufficientBalanceError):
            await rig.client.chat([user("x")], purpose="reply")
    assert [a.category for a in rig.alerts.alerts] == ["llm_auth", "llm_balance"]
    assert all(a.severity == "critical" for a in rig.alerts.alerts)
    rig.clock.tick(601)
    mock(api, error(401))
    with pytest.raises(AuthenticationFailedError):
        await rig.client.chat([user("x")], purpose="reply")
    assert [a.category for a in rig.alerts.alerts].count("llm_auth") == 2


async def test_a_rejected_key_is_read_again_for_the_next_call(
    rig: Rig, api: respx.MockRouter
) -> None:
    mock(api, error(401), ok())
    with pytest.raises(AuthenticationFailedError):
        await rig.client.chat([user("x")], purpose="reply")
    await rig.client.chat([user("x")], purpose="reply")
    assert rig.key_reads == 2


async def test_the_key_never_appears_in_errors_or_alerts(rig: Rig, api: respx.MockRouter) -> None:
    mock(api, error(401, f"Authentication Fails, Your api key: {TEST_KEY} is invalid"))
    with pytest.raises(AuthenticationFailedError) as info:
        await rig.client.chat([user("x")], purpose="reply")
    assert TEST_KEY not in str(info.value) and "***" in str(info.value)
    assert TEST_KEY not in repr([a.detail for a in rig.alerts.alerts])
    mock(api, error(400, "bad sk-abcdefghijklmnop value"))
    with pytest.raises(InvalidRequestError) as bad:
        await rig.client.chat([user("x")], purpose="reply")
    assert "sk-abcdefghijklmnop" not in str(bad.value)


async def test_concurrency_is_limited_by_a_semaphore(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, overrides={"deepseek": {"max_concurrency": 2}})
    state = {"active": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        for _ in range(8):
            await asyncio.sleep(0)
        state["active"] -= 1
        return ok()

    api.post(API).mock(side_effect=handler)
    await asyncio.gather(*[rig.client.chat([user(str(i))], purpose="reply") for i in range(7)])
    assert state["peak"] == 2


async def test_a_call_that_never_answers_times_out_and_is_reported(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(
        db,
        clock,
        overrides={"deepseek": {"timeout_s": {"non_thinking": 1, "thinking": 180}}},
        retry=RetryPolicy(max_retries=0),
        deadline_slack_s=0.0,
    )

    async def never(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        return ok()

    api.post(API).mock(side_effect=never)
    with pytest.raises(RetriesExhaustedError, match="timed out"):
        await rig.client.chat([user("x")], purpose="reply")


async def test_the_circuit_opens_after_ten_failures_and_probes_after_five_minutes(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, retry=RetryPolicy(max_retries=0))
    route = mock(api, *[error(500) for _ in range(10)])
    for _ in range(10):
        with pytest.raises(RetriesExhaustedError):
            await rig.client.chat([user("x")], purpose="reply")
    assert rig.breaker.state is BreakerState.OPEN
    assert [a.category for a in rig.alerts.alerts] == ["llm_circuit"]
    calls = route.call_count
    with pytest.raises(CircuitOpenError) as info:
        await rig.client.chat([user("x")], purpose="reply")
    assert route.call_count == calls and info.value.retry_after_s > 0

    clock.tick(301)
    assert rig.breaker.state is BreakerState.HALF_OPEN
    mock(api, error(500))
    with pytest.raises(RetriesExhaustedError):  # the probe fails: open for another five minutes
        await rig.client.chat([user("x")], purpose="reply")
    assert rig.breaker.state is BreakerState.OPEN
    clock.tick(301)
    mock(api, ok(content="back"))
    result = await rig.client.chat([user("x")], purpose="reply")
    assert result.content == "back" and rig.breaker.state is BreakerState.CLOSED


async def test_every_attempt_and_the_state_of_the_breaker_reach_the_health_record(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    health = LlmHealth(clock)
    rig = Rig(db, clock, retry=RetryPolicy(max_retries=0), failure_threshold=3, health=health)
    mock(api, ok(content="fine"), error(500), error(500), error(500))
    await rig.client.chat([user("x")], purpose="reply")
    assert (health.snapshot(900).calls, health.snapshot(900).failures) == (1, 0)
    for _ in range(3):
        with pytest.raises(RetriesExhaustedError):
            await rig.client.chat([user("x")], purpose="reply")
    snapshot = health.snapshot(900)
    assert (snapshot.calls, snapshot.failures, snapshot.circuit_open) == (4, 3, True)
    assert snapshot.error_rate == 0.75
    clock.tick(301)
    mock(api, ok(content="back"))
    await rig.client.chat([user("x")], purpose="reply")  # the probe works: the breaker closes
    assert not health.snapshot(900).circuit_open and health.snapshot(900).calls == 5


async def test_client_errors_do_not_trip_the_breaker(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, failure_threshold=3)
    mock(api, *[error(400) for _ in range(5)])
    for _ in range(5):
        with pytest.raises(InvalidRequestError):
            await rig.client.chat([user("x")], purpose="reply")
    assert rig.breaker.state is BreakerState.CLOSED


async def test_an_empty_answer_is_an_error(rig: Rig, api: respx.MockRouter) -> None:
    api.post(API).mock(
        return_value=httpx.Response(200, json={"id": "x", "choices": [], "usage": {}})
    )
    with pytest.raises(ApiError, match="no choices"):
        await rig.client.chat([user("x")], purpose="reply")


# ------------------------------------------------------------------- budget, batches


async def test_the_budget_holds_back_proactive_work_but_never_replies(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, with_budget=True)
    route = mock(api, *[ok(prompt=10, completion_tokens=1)] * 3, ok())
    assert rig.budget is not None
    # spend 1.6 dollars today by hand: level 3 (proactive paused)
    rig.ledger.record(
        LedgerRecord(
            "deepseek",
            "deepseek-flash",
            "reply",
            Usage(1, 1, 0, 1, 0),
            CostBreakdown(1.6, 0, 0, True, 1.0),
            False,
            1,
            clock.now_utc(),
        )
    )
    rig.budget.note_spend()
    assert rig.budget.current_level() == 3
    with pytest.raises(BudgetDeniedError, match="proactive"):
        await rig.client.chat([user("x")], purpose="proactive")
    with pytest.raises(BudgetDeniedError):
        await rig.client.chat([user("x")], purpose="plan", thinking=True)
    assert route.call_count == 0
    await rig.client.chat([user("x")], purpose="reply", thinking=True)  # allowed, no thinking
    assert request_json(route.calls.last.request)["thinking"] == {"type": "disabled"}
    await rig.client.chat([user("x")], purpose="summary")
    assert route.call_count == 2


async def test_thinking_stays_available_for_planning_until_proactive_is_paused(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, with_budget=True)
    route = mock(api, ok(reasoning="r"), ok(reasoning="r"), ok(reasoning="r"))

    def spend(usd: float) -> None:
        rig.ledger.record(
            LedgerRecord(
                "deepseek",
                "deepseek-flash",
                "reply",
                Usage(1, 1, 0, 1, 0),
                CostBreakdown(usd, 0, 0, True, 1.0),
                False,
                1,
                clock.now_utc(),
            )
        )

    await rig.client.chat([user("x")], purpose="reply", thinking=True)
    assert request_json(route.calls.last.request)["thinking"] == {"type": "enabled"}
    spend(1.0)  # level 1: chat thinking off, planner thinking still on
    await rig.client.chat([user("x")], purpose="reply", thinking=True)
    assert request_json(route.calls.last.request)["thinking"] == {"type": "disabled"}
    await rig.client.chat([user("x")], purpose="plan", thinking=True)
    assert request_json(route.calls.last.request)["thinking"] == {"type": "enabled"}


async def test_calls_update_the_budget_after_every_paid_reply(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, with_budget=True)
    mock(api, ok(prompt=2_000_000, hit=0, completion_tokens=500_000))  # about $1.20
    assert rig.budget is not None
    await rig.client.chat([user("x")], purpose="reply")
    assert rig.budget.current_level() >= 1
    assert any(a.category == "budget" for a in rig.alerts.alerts)


async def test_one_time_calls_do_not_consult_or_feed_the_daily_budget(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, with_budget=True)
    mock(api, ok(prompt=5_000_000, hit=0, completion_tokens=1_000_000))
    assert rig.budget is not None
    await rig.client.chat([user("x")], purpose="persona", tag=LedgerTag("one_time", "p1"))
    assert rig.budget.current_level() == 0 and rig.budget.status().daily_spent == 0.0


class PausedGuard:
    def __init__(self) -> None:
        self.after: list[str] = []

    def before_call(self, batch_id: str) -> None:
        raise BatchPausedError(batch_id)

    def after_call(self, batch_id: str) -> None:
        self.after.append(batch_id)


class WatchingGuard:
    def __init__(self) -> None:
        self.events: list[str] = []

    def before_call(self, batch_id: str) -> None:
        self.events.append(f"before:{batch_id}")

    def after_call(self, batch_id: str) -> None:
        self.events.append(f"after:{batch_id}")


async def test_a_paused_batch_blocks_its_calls_before_any_request(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, batches=PausedGuard())
    route = mock(api, ok())
    with pytest.raises(BatchPausedError):
        await rig.client.chat([user("x")], purpose="persona", tag=LedgerTag("one_time", "b1"))
    assert route.call_count == 0


async def test_batch_calls_are_checked_before_and_after(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    guard = WatchingGuard()
    rig = Rig(db, clock, batches=guard)
    mock(api, ok(), ok())
    await rig.client.chat([user("x")], purpose="persona", tag=LedgerTag("one_time", "b1"))
    await rig.client.chat([user("x")], purpose="reply")  # daily calls are not batch calls
    assert guard.events == ["before:b1", "after:b1"]


# -------------------------------------------------------------- estimator and cache


async def test_the_token_estimator_learns_from_real_usage_and_saves_periodically(
    rig: Rig, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(prompt=400, completion_tokens=5))
    for _ in range(20):
        await rig.client.chat([user("你好世界" * 30)], purpose="reply")
    assert rig.estimator.calibration.samples == 20
    assert rig.estimator.factor != 1.0
    assert len(rig.saved) == 1  # saved after the twentieth sample


async def test_cache_statistics_are_recorded_for_layouts(rig: Rig, api: respx.MockRouter) -> None:
    mock(api, ok(prompt=1000, hit=0), ok(prompt=1000, hit=900))
    layout = PromptLayout.of([{"role": "system", "content": "规则" * 100}], [user("今天好吗")])
    await rig.client.chat(layout, purpose="reply")
    second = await rig.client.chat(layout, purpose="reply")
    assert second.usage.cache_hit_ratio == pytest.approx(0.9)
    summary = rig.cache.summary("reply")
    assert summary.calls == 2 and summary.hit_ratio == pytest.approx(0.45)


async def test_the_client_can_be_closed_and_reused(rig: Rig, api: respx.MockRouter) -> None:
    mock(api, ok(), ok())
    await rig.client.chat([user("x")], purpose="reply")
    await rig.client.aclose()
    await rig.client.aclose()
    await rig.client.chat([user("x")], purpose="reply")
    assert rig.key_reads == 2


async def test_the_model_cannot_be_called_with_an_invalid_purpose(rig: Rig) -> None:
    with pytest.raises(ValueError, match="not a valid Purpose"):
        await rig.client.chat([user("x")], purpose="chitchat")


# ----------------------------------------------------------- edge cases and limits


async def test_a_cancelled_call_frees_the_half_open_probe_slot(
    db: Database, clock: ManualClock, api: respx.MockRouter
) -> None:
    rig = Rig(db, clock, failure_threshold=1, retry=RetryPolicy(max_retries=0))
    mock(api, error(500))
    with pytest.raises(RetriesExhaustedError):
        await rig.client.chat([user("x")], purpose="reply")
    assert rig.breaker.state is BreakerState.OPEN
    clock.tick(301)
    started = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        return ok()

    api.post(API).mock(side_effect=hang)
    task = asyncio.ensure_future(rig.client.chat([user("x")], purpose="reply"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rig.breaker.before_call()  # not stuck behind a probe that will never report back
    assert rig.breaker.state is BreakerState.HALF_OPEN


async def test_unexpected_errors_are_raised_and_do_not_count_against_the_service(
    db: Database, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(db, clock, failure_threshold=2)

    async def broken_create(**kwargs: Any) -> None:
        raise RuntimeError("a bug in our own code")

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=broken_create)))
    monkeypatch.setattr(rig.client, "_client", lambda: sdk)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="our own code"):
            await rig.client.chat([user("x")], purpose="reply")
    assert rig.breaker.state is BreakerState.CLOSED and rig.clock.sleeps == []


async def test_requests_with_too_many_or_too_large_images_are_refused_locally(
    rig: Rig, api: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = mock(api, ok())
    pictures = [ImageInput.from_bytes(draw_png(32, 32)) for _ in range(2)]
    monkeypatch.setattr(deepseek_module, "MAX_IMAGES_PER_REQUEST", 1)
    with pytest.raises(LlmConfigError, match="at most 1 images"):
        await rig.client.chat([user("x")], purpose="caption", images=pictures)
    monkeypatch.setattr(deepseek_module, "MAX_IMAGES_PER_REQUEST", 600)
    monkeypatch.setattr(deepseek_module, "MAX_TOTAL_IMAGE_BYTES", 10)
    with pytest.raises(LlmConfigError, match="64 MiB"):
        await rig.client.chat([user("x")], purpose="caption", images=pictures)
    assert route.call_count == 0


async def test_a_failing_calibration_save_does_not_disturb_the_calls(
    rig: Rig, api: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(estimator: TokenEstimator) -> None:
        raise OSError("settings table is locked")

    monkeypatch.setattr(rig.client, "_save_calibration", broken)
    api.post(API).mock(return_value=ok(prompt=300))
    for _ in range(21):
        result = await rig.client.chat([user("你好" * 20)], purpose="reply")
    assert result.content == "好的" and rig.estimator.calibration.samples == 21


def test_list_content_with_text_parts_is_normalised() -> None:
    cleaned = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a", "extra": 1},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                ],
            }
        ]
    )
    assert cleaned[0]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
    ]
