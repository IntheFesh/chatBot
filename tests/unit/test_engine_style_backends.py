"""The ``style`` and ``hybrid`` backends in the reply pipeline (R-ENG-006, R-TRN-011, R-LLM-011).

The style model is a scripted client (``tests/support/style_models.py``) or the real HTTP client
against the local test server of round 01; DeepSeek is intercepted with respx.  The pipeline is the
real one: what a backend returns goes through the same parsing, post-processing and retries as the
DeepSeek backend's output.
"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import respx

from tests.support.deepseek import API, TEST_KEY, error, ok
from tests.support.reply_view import StaticDataView
from tests.support.style_models import ScriptedStyleClient, down, register_model
from tests.support.style_server import StyleServer, llama_defaults
from twin.engine.hybrid_backend import NO_REPLY_PLANNED, HybridPlan
from twin.engine.pipeline import ReplyPipeline
from twin.engine.style_runtime import ConfiguredStyleClient, StyleRuntime
from twin.engine.types import InboundItem, ReplyContext
from twin.llm.errors import StyleModelError
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.style_client import RenderedPrompt, StyleParams
from twin.memory.recent import Turn
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.services import Services
from twin.training import lf_template

NOW = datetime(2026, 10, 9, 18, 30, tzinfo=UTC)
CARD_MARKER = "训练时的口头禅是嘿嘿"
OPENER = "<|im_start|>assistant\n"


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def llm(services: Services) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


def stored_card(services: Services) -> None:
    """The pre-holdout card the model below was trained with (version v1)."""
    text = (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block(f"### 风格\n- {CARD_MARKER}\n\n### 基本情况\n- 一件事实\n\n")
        + compose.manual_block([], None)
    )
    PersonaStore(services.db, services.clock).add_version("pre_holdout", text, reason="generate")


@pytest.fixture
def ready(services: Services) -> Services:
    """A registered, active model locked to card v1, and that card in the store."""
    stored_card(services)
    register_model(services, persona_version="v1")
    return services


def style(services: Services, llm: LlmRuntime, client: ScriptedStyleClient) -> StyleRuntime:
    return StyleRuntime.from_services(services, llm, client=client)


def pipeline_of(services: Services, llm: LlmRuntime, runtime: StyleRuntime) -> ReplyPipeline:
    return ReplyPipeline.from_services(
        services, llm, extra_backends=runtime.backends, rng=random.Random(4)
    )


def inbound(text: str, number: int = 1) -> InboundItem:
    return InboundItem(f"in-{number}", NOW + timedelta(seconds=number), "text", text)


def context(text: str = "在吗", **changes: Any) -> ReplyContext:
    changes.setdefault("backend", "style")
    changes.setdefault("seed", 3)
    return ReplyContext(inbound=(inbound(text),), **changes)


def data(**changes: Any) -> StaticDataView:
    return StaticDataView(at=NOW, **changes)


def sent(route: respx.Route) -> list[dict[str, Any]]:
    return [json.loads(call.request.content) for call in route.calls]


def plan_json(**fields: Any) -> str:
    plan = {
        "reply": True,
        "intent": "接住对方的话",
        "facts_to_use": ["对方养了一只猫"],
        "tone": "随意",
        "bubble_hint": "两条短句",
        "sticker_hint": "",
    }
    plan.update(fields)
    return json.dumps(plan, ensure_ascii=False)


# ---------------------------------------------------------------------- the style backend


async def test_the_style_model_gets_only_the_rendered_prompt_and_its_text_becomes_bubbles(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content="不该被调用"))
    client = ScriptedStyleClient(["好呀，我也想去。\n哈哈"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    history = (
        Turn("user", "周末有空吗", NOW - timedelta(hours=2), NOW, ("h1",)),
        Turn("bot", "有呀", NOW - timedelta(hours=2), NOW, ("h2",)),
    )
    draft = await pipeline.run(
        context("去看电影吧", history=history), data(memory_text="【相关的事】\n对方养了一只猫")
    )
    assert draft.usable and draft.backend == "style" and not draft.thinking
    assert [b.text for b in draft.bubbles] == ["好呀", "我也想去", "哈哈"]
    assert draft.cost_usd == 0 and draft.usage.calls == 1 and draft.usage.prompt_tokens == 120
    assert route.call_count == 0  # DeepSeek was not asked
    (prompt,) = client.prompts
    assert isinstance(prompt, RenderedPrompt) and prompt.text.endswith(OPENER)
    assert prompt.text.startswith("<|im_start|>system\n")
    assert CARD_MARKER in prompt.text  # the locked card, not the data view's
    assert "她说话很短" not in prompt.text and "一件事实" not in prompt.text
    assert (
        "<|im_start|>user\n周末有空吗<|im_end|>\n<|im_start|>assistant\n有呀<|im_end|>\n"
        in prompt.text
    )
    assert prompt.text.endswith("<|im_start|>user\n去看电影吧<|im_end|>\n" + OPENER)
    assert "对方养了一只猫" in prompt.text  # the memory block is part of the system segment
    assert "<think" not in prompt.text
    (params,) = client.params
    assert "<|im_end|>" in params.stop_strings() and params.n_predict == 200
    meta = draft.meta
    assert meta["model"] == "r-test-run-Q5_K_M" and meta["locked"] is True
    assert meta["persona"] == "pre_holdout v1" and meta["template"] == lf_template.TEMPLATE_VERSION


async def test_the_context_of_the_prompt_is_the_newest_eight_merged_turns(
    ready: Services, llm: LlmRuntime
) -> None:
    client = ScriptedStyleClient(["嗯"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    history = tuple(
        Turn("user" if i % 2 == 0 else "bot", f"第{i}句", NOW, NOW, (f"h{i}",)) for i in range(20)
    )
    await pipeline.run(context("现在这句", history=history, woke_up=True), data())
    text = client.prompts[0].text
    system, conversation = text.split("<|im_end|>\n", 1)
    assert "第12句" not in text  # older than the newest eight merged turns
    assert "【前文】\n她：第13句" in system  # the turn of hers that would open the context
    assert conversation.startswith("<|im_start|>user\n第14句<|im_end|>\n")
    assert "你刚醒" in system
    meta = client.prompts[0].meta
    assert meta is not None and meta.context_turns == 7 and meta.prelude_turns == 1


async def test_a_think_block_in_the_output_is_removed_and_counts_as_a_violation(
    ready: Services, llm: LlmRuntime
) -> None:
    client = ScriptedStyleClient(["<think>先想想怎么说</think>好呀", "好呀"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context(), data())
    assert [kind.kind for kind in draft.violations] == ["think_tag"]
    assert draft.attempts == 2 and len(client.prompts) == 2
    assert [b.text for b in draft.bubbles] == ["好呀"]
    assert "regenerated" in {a.step for a in draft.actions}
    assert draft.meta["attempt_violations"] == [["think_tag"]]
    assert all("先想想" not in b.text for b in draft.bubbles)


async def test_a_style_model_that_only_thinks_gives_way_to_deepseek_without_thinking(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content="好呀"))
    client = ScriptedStyleClient(["<think>甲</think>", "<think>乙</think>"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context(), data())
    assert draft.usable and draft.backend == "deepseek" and draft.attempts == 3
    assert len(client.prompts) == 2 and route.call_count == 1
    assert sent(route)[0]["thinking"] == {"type": "disabled"}
    attempts = draft.meta["attempt_violations"]
    assert len(attempts) == 2 and all("think_tag" in kinds for kinds in attempts)


async def test_a_style_server_that_is_down_is_replaced_by_deepseek_for_that_reply(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content="好呀"))
    client = ScriptedStyleClient([down("connection refused")])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context(), data())
    assert draft.usable and draft.backend == "deepseek" and route.call_count == 1
    failed = [a for a in draft.actions if a.step == "backend_error"]
    assert [a.detail for a in failed] == ["StyleModelError"]


async def test_no_active_model_or_a_foreign_template_stops_the_style_backend(
    services: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content="好呀"))
    client = ScriptedStyleClient(["不该被用"])
    pipeline = pipeline_of(services, llm, style(services, llm, client))
    draft = await pipeline.run(context(), data())  # nothing registered
    assert draft.backend == "deepseek" and client.prompts == []
    register_model(services, template_version="qwen3_think@1", persona_version="v1")
    again = await pipeline.run(context(), data())
    assert again.backend == "deepseek" and client.prompts == []


async def test_a_reply_that_hit_the_token_limit_loses_its_unfinished_last_line(
    ready: Services, llm: LlmRuntime
) -> None:
    client = ScriptedStyleClient(["好呀\n我们明天去那个新开的"], truncated=True)
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context(), data())
    assert [b.text for b in draft.bubbles] == ["好呀"] and draft.meta["truncated"] is True


async def test_a_memory_that_fails_does_not_stop_the_reply(
    ready: Services, llm: LlmRuntime
) -> None:
    client = ScriptedStyleClient(["好呀"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context(), data(fail_memory=True))
    assert draft.usable and draft.meta["memory_unavailable"] is True
    assert client.prompts[0].text.endswith(OPENER)


# --------------------------------------------------------------------- the hybrid backend


async def test_deepseek_plans_and_the_style_model_follows_the_plan(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content=plan_json(sticker_hint="开心"), hit=50))
    client = ScriptedStyleClient(["好呀，一起去"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context("周末去看电影吧", backend="hybrid"), data())
    assert draft.usable and draft.backend == "hybrid"
    assert [b.text for b in draft.bubbles] == ["好呀", "一起去"]
    assert draft.plan is not None and draft.plan["intent"] == "接住对方的话"
    assert draft.cost_usd > 0 and draft.usage.calls == 2  # the plan and the reply
    (planner_request,) = sent(route)
    system = planner_request["messages"][0]["content"]
    assert "只是先把这一轮想清楚" in system and "她说话很短" in system  # the full card
    assert planner_request["thinking"] == {"type": "disabled"}
    assert "周末去看电影吧" in planner_request["messages"][-1]["content"]
    (prompt,) = client.prompts
    system_segment = prompt.text.split("<|im_end|>\n", 1)[0]
    assert "【规划】\n想表达：接住对方的话\n会用到：对方养了一只猫\n语气：随意" in system_segment
    assert "气泡：两条短句\n表情包：开心" in system_segment
    assert CARD_MARKER in system_segment and prompt.meta is not None and prompt.meta.has_plan


async def test_thinking_belongs_to_the_planning_step(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content=plan_json(), reasoning="她大概想去"))
    client = ScriptedStyleClient(["好呀"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(
        context("周末去哪里玩呢？", backend="hybrid", thinking_mode="auto"), data()
    )
    assert sent(route)[0]["thinking"] == {"type": "enabled"}
    assert draft.thinking and draft.reasoning == "她大概想去"
    assert len(client.prompts) == 1 and "<think" not in client.prompts[0].text


async def test_the_style_backend_with_thinking_on_runs_the_hybrid_planning_step(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content=plan_json(intent="认真回答")))
    client = ScriptedStyleClient(["想去呀"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context("周末去哪里玩呢？", thinking_mode="on"), data())
    assert draft.backend == "hybrid" and draft.thinking and route.call_count == 1
    assert "想表达：认真回答" in client.prompts[0].text
    quiet = await pipeline.run(context("周末去哪里玩呢？", thinking_mode="off"), data())
    assert quiet.backend == "style" and route.call_count == 1  # no planning without thinking


async def test_a_budget_that_switched_thinking_off_also_skips_the_planning_step(
    services: Services, api: respx.MockRouter
) -> None:
    services.settings.budget.daily_usd = 0.00001
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    llm = build_llm_runtime(services)
    try:
        stored_card(services)
        register_model(services, persona_version="v1")
        route = api.post(API).mock(return_value=ok(content="好呀"))
        client = ScriptedStyleClient(["嗯嗯"])
        pipeline = pipeline_of(services, llm, style(services, llm, client))
        spent = await pipeline.run(context("你好", backend="deepseek", thinking_mode="on"), data())
        assert spent.usable and llm.budget.limits().level >= 1
        draft = await pipeline.run(context("你好呀", thinking_mode="on"), data())
        assert draft.backend == "style" and route.call_count == 1  # the style model alone
    finally:
        await llm.client.aclose()


async def test_a_plan_that_says_do_not_answer_sends_nothing_when_the_message_is_a_closing_one(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content=plan_json(reply=False, intent="对方在道别")))
    client = ScriptedStyleClient(["不该说话"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context("好的", backend="hybrid"), data())
    assert draft.usable and draft.no_reply and draft.bubbles == ()
    assert client.prompts == []  # the style model was not asked
    assert draft.plan is not None and draft.plan["reply"] is False
    assert any(a.step == "plan_no_reply" and a.detail == NO_REPLY_PLANNED for a in draft.actions)


async def test_a_plan_that_says_do_not_answer_a_question_is_ignored(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content=plan_json(reply=False)))
    client = ScriptedStyleClient(["在呀"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context("你在吗？", backend="hybrid"), data())
    assert not draft.no_reply and [b.text for b in draft.bubbles] == ["在呀"]
    assert draft.meta["plan_no_reply_ignored"] is True


async def test_a_plan_that_is_not_valid_json_twice_leaves_the_style_model_on_its_own(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content="我觉得她会说好呀"))
    client = ScriptedStyleClient(["好呀"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context("周末去看电影吧", backend="hybrid"), data())
    assert route.call_count == 2  # the second try told the model what was wrong
    assert "invalid JSON" in sent(route)[1]["messages"][-1]["content"]
    assert draft.usable and draft.backend == "hybrid" and draft.plan is None
    assert draft.meta["plan_failed"] is True and "【规划】" not in client.prompts[0].text


async def test_a_planner_that_fails_hands_the_reply_to_deepseek(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(side_effect=[error(400, "bad request"), ok(content="好呀")])
    client = ScriptedStyleClient(["不该被用"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context("周末去看电影吧", backend="hybrid"), data())
    assert draft.usable and draft.backend == "deepseek" and client.prompts == []
    assert [a.detail for a in draft.actions if a.step == "backend_error"] == ["InvalidRequestError"]
    assert route.call_count == 2


async def test_a_planner_that_refuses_is_never_shown_and_ends_in_the_fallback(
    ready: Services, llm: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=error(400, "Content Exists Risk"))
    client = ScriptedStyleClient(["不该被用"])
    pipeline = pipeline_of(ready, llm, style(ready, llm, client))
    draft = await pipeline.run(context("你好", backend="hybrid"), data())
    assert draft.needs_fallback and draft.fallback_reason == "refused" and draft.bubbles == ()
    assert client.prompts == []


def test_the_plan_accepts_what_a_model_writes_loosely() -> None:
    plan = HybridPlan.model_validate_json(
        '{"reply": "true", "intent": null, "facts_to_use": "一件事", "tone": 3, "extra": 1}'
    )
    assert plan.reply is True and plan.intent == "" and plan.tone == "3"
    assert plan.facts_to_use == ["一件事"] and plan.bubble_hint == "" and plan.sticker_hint == ""
    fields = HybridPlan(reply=True, facts_to_use=["a", " ", "b"]).fields()
    assert fields.facts_to_use == ("a", "b")
    with pytest.raises(ValueError, match="reply"):
        HybridPlan.model_validate_json('{"intent": "没有 reply 字段"}')


# ------------------------------------------------------------------- the real client


async def test_the_real_client_sends_one_string_to_the_completion_endpoint(
    ready: Services, llm: LlmRuntime
) -> None:
    server = StyleServer()
    llama_defaults(server, "好呀\n哈哈")
    server.start()
    try:
        ready.settings.style_model.endpoint = server.url
        runtime = StyleRuntime.from_services(ready, llm)
        pipeline = pipeline_of(ready, llm, runtime)
        draft = await pipeline.run(context("在吗"), data())
        await runtime.aclose()
        assert [b.text for b in draft.bubbles] == ["好呀", "哈哈"]
        sent_prompt = server.last("/completion").json
        assert set(sent_prompt) >= {"prompt", "stop", "n_predict", "temperature", "top_p"}
        assert "messages" not in sent_prompt and "chat_template" not in sent_prompt
        assert sent_prompt["prompt"].startswith("<|im_start|>system\n")
        assert sent_prompt["prompt"].endswith("在吗<|im_end|>\n" + OPENER)
        assert "<|im_end|>" in sent_prompt["stop"]
        assert sent_prompt["temperature"] == 0.7 and sent_prompt["n_predict"] == 200
    finally:
        server.stop()


async def test_a_configuration_that_cannot_work_is_an_unavailable_model_not_a_crash(
    services: Services,
) -> None:
    services.settings.style_model.mode = "vllm_completion"  # without the name of the LoRA
    client = ConfiguredStyleClient(services.settings.style_model, services.clock)
    health = await client.health()
    assert not health.ok and "model_id" in health.detail
    with pytest.raises(StyleModelError, match="model_id"):
        await client.generate(RenderedPrompt("x"), StyleParams())
    await client.aclose()


async def test_the_client_is_made_once_and_closed(services: Services, llm: LlmRuntime) -> None:
    server = StyleServer()
    llama_defaults(server)
    server.start()
    try:
        services.settings.style_model.endpoint = server.url
        client = ConfiguredStyleClient(services.settings.style_model, services.clock)
        assert (await client.health()).ok and (await client.health()).ok
        assert (await client.tokenize("ab")) == [97, 98]
        out = await client.generate(RenderedPrompt("<|im_start|>user\nx"), StyleParams())
        assert out.text == "好呀"
        await client.aclose()
        await client.aclose()  # closing twice is fine
    finally:
        server.stop()


async def test_the_runtime_registers_the_two_backends_by_their_names(
    ready: Services, llm: LlmRuntime
) -> None:
    runtime = style(ready, llm, ScriptedStyleClient())
    assert set(runtime.backends) == {"style", "hybrid"}
    assert runtime.backends["style"].name == "style" and runtime.backends["hybrid"].name == "hybrid"
    await runtime.aclose()
    assert isinstance(runtime.client, ScriptedStyleClient) and runtime.client.closed
