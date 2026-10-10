"""The reply pipeline end to end on a static data view (R-ENG-005..008, R-ENG-012, R-SAFE)."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx

from tests.support.deepseek import API, TEST_KEY, error, ok
from tests.support.reply_view import StaticDataView, StaticSelector, make_profile, sticker_record
from twin.engine.backend import BackendRequest, BackendResult
from twin.engine.pipeline import FALLBACK_BACKEND, ReplyPipeline
from twin.engine.types import InboundItem, ReplyContext, SendLimits, UsageSummary
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.memory.recent import Turn
from twin.retrieval.examples import Example, ExampleLine, ExampleTurn
from twin.services import Services

NOW = datetime(2026, 10, 9, 18, 30, tzinfo=UTC)


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def runtime(services: Services) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


@pytest.fixture
def pipeline(services: Services, runtime: LlmRuntime) -> ReplyPipeline:
    return ReplyPipeline.from_services(services, runtime, rng=random.Random(5))


def inbound(text: str, *, kind: str = "text", number: int = 1) -> InboundItem:
    return InboundItem(f"in-{number}", NOW + timedelta(seconds=number), kind, text)


def context(text: str = "在吗", **changes: Any) -> ReplyContext:
    items = changes.pop("inbound", None) or (inbound(text),)
    changes.setdefault("seed", 3)
    return ReplyContext(inbound=items, **changes)


def sent(route: respx.Route) -> list[dict[str, Any]]:
    return [json.loads(call.request.content) for call in route.calls]


def route_of(api: respx.MockRouter, *replies: httpx.Response | Exception) -> respx.Route:
    return api.post(API).mock(side_effect=list(replies))


def view(**changes: Any) -> StaticDataView:
    return StaticDataView(at=NOW, **changes)


async def test_a_reply_is_the_bubbles_of_the_model_after_post_processing(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(
        api, ok(content="好呀，我也想去。\n哈哈[拥抱]\n[表情包:开心]", prompt=500, hit=400)
    )
    selector = StaticSelector({"开心": sticker_record("a" * 32, "开心")})
    draft = await pipeline.run(context("周末去看电影吧"), view(selector=selector))
    assert draft.usable and not draft.no_reply and draft.attempts == 1
    assert [b.text for b in draft.bubbles] == ["好呀", "我也想去", "哈哈[拥抱]", "[表情包:开心]"]
    assert draft.bubbles[-1].is_sticker and draft.bubbles[-1].sticker_md5 == "a" * 32
    assert draft.backend == "deepseek" and not draft.thinking and draft.reasoning is None
    assert draft.cost_usd > 0 and draft.usage.calls == 1 and draft.usage.prompt_tokens == 500
    assert draft.usage.cache_hit_ratio == pytest.approx(400 / 500)
    assert set(draft.timings_ms) == {"gather", "generate", "post", "total"}
    assert "punctuation_normalised" in {a.step for a in draft.actions}
    assert draft.violations == () and draft.fallback_reason is None
    assert draft.meta["template"] == "reply_rules@1" and draft.meta["persona"] == "live v3"
    assert route.call_count == 1


async def test_the_request_is_rules_history_and_one_last_message(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(api, ok(content="好呀"))
    history = (
        Turn("user", "早", NOW - timedelta(hours=3), NOW, ("h1",)),
        Turn("bot", "早呀", NOW - timedelta(hours=3), NOW, ("h2",)),
    )
    examples = [
        Example(
            "w1",
            NOW - timedelta(days=9),
            50,
            "workday",
            (ExampleTurn(False, (ExampleLine("在吗", "text", True),)),),
            (ExampleLine("在呢", "text", True), ExampleLine("[图片]", "image", False)),
        )
    ]
    data = view(memory_text="【相关的事】\n对方：养了一只猫", example_list=examples)
    await pipeline.run(context("周末去看电影吧", history=history, woke_up=True), data)
    body = sent(route)[0]
    system, *rest = body["messages"]
    assert system["role"] == "system" and "她说话很短" in system["content"]
    assert "不要否认你是模拟出来的" in system["content"]  # R-SAFE-003, in the fixed rules
    assert "[表情包:标签]" in system["content"] and "[拥抱]" in system["content"]
    assert [m["role"] for m in rest] == ["user", "assistant", "user"]
    assert rest[0]["content"] == "早" and rest[1]["content"] == "早呀"
    last = rest[-1]["content"]
    assert "2026年10月9日 周五（工作日） 中午 13:30" in last and "空闲" in last
    assert "你刚醒" in last and "养了一只猫" in last
    assert "仅供模仿语气，不要照抄内容" in last and "（此处她发了：[图片]）" in last
    assert last.rstrip().endswith("周末去看电影吧")
    assert all("此刻" not in m["content"] for m in rest[:-1])  # the variable block is only here
    assert body["thinking"] == {"type": "disabled"}


async def test_the_memory_and_examples_are_asked_with_the_budget(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route_of(api, ok(content="好呀"))
    data = view()
    history = tuple(
        Turn("user" if n % 2 == 0 else "bot", f"第{n}句", NOW, NOW, (f"h{n}",)) for n in range(10)
    )
    await pipeline.run(context("现在呢", history=history), data)
    ((query, budget),) = data.memory_queries
    assert "第8句" in query.text and "第9句" in query.text and "现在呢" in query.text
    assert "第5句" not in query.text and budget == 800
    turns, k = data.example_queries[0]
    assert k == 8 and len(turns) == 7 and turns[-1].text == "现在呢" and not turns[-1].her
    assert [t.her for t in turns[:-1]] == [False, True, False, True, False, True]


async def test_a_violation_is_answered_with_a_second_try_that_knows_what_was_wrong(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(
        api, ok(content="作为AI我无法感受", prompt=100), ok(content="好呀", prompt=100)
    )
    draft = await pipeline.run(context(), view())
    assert [b.text for b in draft.bubbles] == ["好呀"] and draft.attempts == 2
    assert [v.kind for v in draft.violations] == ["ai_self_reference"]
    assert draft.usage.calls == 2 and draft.cost_usd > 0
    first, second = sent(route)
    assert "上一次的回复有这些问题" not in first["messages"][-1]["content"]
    note = second["messages"][-1]["content"]
    assert "上一次的回复有这些问题" in note and "不要提到自己是 AI" in note
    assert "regenerated" in {a.step for a in draft.actions}
    assert draft.meta["attempt_violations"] == [["ai_self_reference"]]


async def test_a_topic_the_user_raised_does_not_cost_a_second_try(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(api, ok(content="我觉得人工智能挺吓人的"))
    draft = await pipeline.run(context("最近都在聊人工智能，你怎么看"), view())
    assert [b.text for b in draft.bubbles] == ["我觉得人工智能挺吓人的"]
    assert draft.attempts == 1 and draft.violations == () and route.call_count == 1
    assert "ai_topic_kept" in {a.step for a in draft.actions}


async def test_three_tries_use_deepseek_without_thinking_for_the_last(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(
        api,
        ok(content="[图片]", reasoning="想想"),
        ok(content="[语音 5 秒：好]", reasoning="再想想"),
        ok(content="好呀"),
    )
    draft = await pipeline.run(context("真的吗？", thinking_mode="on"), view())
    assert draft.attempts == 3 and [b.text for b in draft.bubbles] == ["好呀"]
    assert [body["thinking"]["type"] for body in sent(route)] == ["enabled", "enabled", "disabled"]
    assert not draft.thinking and draft.reasoning is None
    assert draft.meta["attempt_violations"] == [["event_text_only"], ["event_text_only"]]


async def test_without_thinking_the_third_try_would_repeat_the_second_and_is_skipped(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(api, ok(content="[图片]"), ok(content="[图片]"), ok(content="好呀"))
    draft = await pipeline.run(context(), view())
    assert draft.needs_fallback and draft.fallback_reason == "violations"
    assert route.call_count == 2 and draft.bubbles == () and not draft.usable
    assert [v.kind for v in draft.violations] == ["event_text_only"]


async def test_thinking_follows_the_setting_and_the_reasoning_is_only_reported(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(api, ok(content="想去呀", reasoning="她会开心", reasoning_tokens=20))
    draft = await pipeline.run(context("周末去哪里玩呢？", thinking_mode="auto"), view())
    assert sent(route)[0]["thinking"] == {"type": "enabled"} and "temperature" not in sent(route)[0]
    assert draft.thinking and draft.reasoning == "她会开心"
    api.reset()
    quiet = route_of(api, ok(content="嗯"))
    plain = await pipeline.run(context("好的", thinking_mode="auto"), view())
    assert sent(quiet)[0]["thinking"] == {"type": "disabled"} and not plain.thinking
    assert sent(quiet)[0]["temperature"] == 1.0


async def test_the_budget_degrades_thinking_examples_and_memory_but_never_the_reply(
    services: Services, api: respx.MockRouter
) -> None:
    """R-LLM-008: once the day's budget is spent, the reply goes on with a smaller context."""
    services.settings.budget.daily_usd = 0.00001
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    runtime = build_llm_runtime(services)
    try:
        pipeline = ReplyPipeline.from_services(services, runtime)
        route = route_of(api, ok(content="好呀"), ok(content="嗯嗯"))
        data = view(memory_text="【相关的事】\n养了一只猫")
        first = await pipeline.run(context("你好", thinking_mode="on"), data)
        assert first.thinking and runtime.budget.limits().level >= 4  # the first reply spent it all
        second = await pipeline.run(context("你好呀", thinking_mode="on"), data)
        assert second.usable and [b.text for b in second.bubbles] == ["嗯嗯"]
        assert not second.thinking and sent(route)[1]["thinking"] == {"type": "disabled"}
        assert data.example_queries[0][1] == 8 and data.example_queries[1][1] == 2
        assert [budget for _, budget in data.memory_queries] == [800, 200]
    finally:
        await runtime.client.aclose()


async def test_no_text_of_the_conversation_is_logged_at_info_or_above(
    api: respx.MockRouter, pipeline: ReplyPipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """CLAUDE.md rule 6: the logs of a round tell what happened, never what was said."""
    route_of(api, ok(content="作为AI我无法感受暗号乙"), ok(content="好呀暗号丁"))
    data = view(memory_text="【相关的事】\n暗号丙")
    with caplog.at_level(logging.INFO, logger="twin"):
        draft = await pipeline.run(context("暗号甲"), data)
    assert draft.usable and caplog.records
    for secret in ("暗号甲", "暗号乙", "暗号丙", "暗号丁"):
        assert secret not in caplog.text
    assert any(record.getMessage() == "reply_violations" for record in caplog.records)


async def test_a_refusal_is_never_shown_and_goes_straight_to_the_fallback(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    filtered = route_of(api, ok(content="抱歉我不能", finish="content_filter"))
    draft = await pipeline.run(context(), view())
    assert draft.needs_fallback and draft.fallback_reason == "refused" and draft.bubbles == ()
    assert filtered.call_count == 1 and "抱歉" not in repr(draft)
    api.reset()
    risky = route_of(api, error(400, "Content Exists Risk"))
    again = await pipeline.run(context(), view())
    assert again.fallback_reason == "refused" and risky.call_count == 1 and again.cost_usd == 0
    api.reset()
    wording = route_of(api, ok(content="抱歉，我无法继续这个话题"))
    third = await pipeline.run(context(), view())
    assert third.fallback_reason == "refused" and wording.call_count == 1
    assert "无法继续" not in repr(third)
    api.reset()
    empty = route_of(api, ok(content="  "))
    assert (await pipeline.run(context(), view())).fallback_reason == "refused"
    assert empty.call_count == 1


async def test_a_failing_call_is_a_fallback_signal_not_an_exception(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route_of(api, error(401, "bad key"))
    draft = await pipeline.run(context(), view())
    assert draft.needs_fallback and draft.fallback_reason == "backend_error"
    assert [a.detail for a in draft.actions if a.step == "backend_error"] == [
        "AuthenticationFailedError"
    ]
    api.reset()
    bad_request = route_of(api, error(400, "invalid parameter"))
    assert (await pipeline.run(context(), view())).fallback_reason == "backend_error"
    assert bad_request.call_count == 1


async def test_a_missing_key_is_a_fallback_signal_too(
    services: Services, api: respx.MockRouter
) -> None:
    runtime = build_llm_runtime(services)  # no key stored
    try:
        draft = await ReplyPipeline.from_services(services, runtime).run(context(), view())
    finally:
        await runtime.client.aclose()
    assert draft.needs_fallback and draft.fallback_reason == "backend_error"


async def test_cancelling_a_run_stops_it(pipeline: ReplyPipeline) -> None:
    started = asyncio.Event()

    class Hanging:
        name = "deepseek"

        async def generate(self, request: BackendRequest) -> BackendResult:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("not reached")

    hanging = ReplyPipeline(
        backends={"deepseek": Hanging()},
        clock=pipeline._clock,
        ai_phrases=pipeline._ai_phrases,
        commitments=None,
    )
    task = asyncio.ensure_future(hanging.run(context(), view()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_redaction_token_in_the_output_is_asked_again(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    """R-ENG-012: a token copied from the prompt means the reply is not natural speech."""
    route = route_of(api, ok(content="你的电话是[手机号]吧"), ok(content="好呀"))
    draft = await pipeline.run(context("我的号码发你了"), view())
    assert draft.usable and [b.text for b in draft.bubbles] == ["好呀"] and draft.attempts == 2
    assert [v.kind for v in draft.violations] == ["token_leak"]
    assert draft.violations[0].detail == "[手机号]"
    assert "方括号标记" in sent(route)[1]["messages"][-1]["content"]
    assert "[手机号]" not in repr(draft.bubbles)


async def test_a_promise_is_asked_again_and_cut_out_on_the_last_try(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(
        api, ok(content="好呀\n我明天给你打电话"), ok(content="好呀\n我明天给你打电话")
    )
    draft = await pipeline.run(context("想你了"), view())
    # the second (and last) try cuts the promise out and keeps the rest
    assert draft.usable and [b.text for b in draft.bubbles] == ["好呀"]
    assert route.call_count == 2
    assert [v.kind for v in draft.violations] == ["commitment"]
    assert "commitment_removed" in {a.step for a in draft.actions}
    assert "不要答应打电话" in sent(route)[1]["messages"][-1]["content"]


async def test_a_promise_alone_leaves_nothing_and_ends_in_the_fallback(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route_of(api, ok(content="我给你发张照片"), ok(content="我给你发张照片"))
    draft = await pipeline.run(context("想看你"), view())
    assert draft.needs_fallback and draft.fallback_reason == "violations"


async def test_staying_silent_after_a_closing_message(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(api, ok(content="[不回]"))
    draft = await pipeline.run(context("好的"), view())
    assert draft.usable and draft.no_reply and draft.bubbles == ()
    assert "40% 的时候不再回复" in sent(route)[0]["messages"][-1]["content"]
    assert "no_reply" in {a.step for a in draft.actions}


async def test_silence_is_not_allowed_after_a_question_or_without_data(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(api, ok(content="[不回]"), ok(content="在呢"))
    draft = await pipeline.run(context("你在吗？"), view())
    assert not draft.no_reply and [b.text for b in draft.bubbles] == ["在呢"]
    assert "不再回复" not in sent(route)[0]["messages"][-1]["content"]
    assert [v.kind for v in draft.violations] == ["no_reply_not_allowed"]
    api.reset()
    nothing = route_of(api, ok(content="[不回]"), ok(content="嗯嗯"))
    no_data = view(profile=make_profile(closing_no_reply=None))
    again = await pipeline.run(context("好的"), no_data)
    assert not again.no_reply and [b.text for b in again.bubbles] == ["嗯嗯"]
    assert "不再回复" not in sent(nothing)[0]["messages"][-1]["content"]


async def test_a_quote_is_kept_only_where_the_channel_can_show_it(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route_of(api, ok(content="[引用:周末去看电影]\n好呀"), ok(content="[引用:周末去看电影]\n好呀"))
    shown = await pipeline.run(context("周末去看电影吧"), view())
    assert shown.quote == "周末去看电影" and shown.bubbles[0].quote == "周末去看电影"
    plain = await pipeline.run(
        context("周末去看电影吧", limits=SendLimits(supports_quote=False)), view()
    )
    assert plain.quote is None and plain.bubbles[0].quote is None
    assert "quote_removed" in {a.step for a in plain.actions}


async def test_the_quota_limits_the_bubbles(api: respx.MockRouter, pipeline: ReplyPipeline) -> None:
    route_of(api, ok(content="一\n二\n三\n四"))
    draft = await pipeline.run(context(limits=SendLimits(max_bubbles=2)), view())
    assert [b.text for b in draft.bubbles] == ["一 二", "三 四"]
    assert "quota_merge" in {a.step for a in draft.actions}


async def test_an_honest_answer_to_do_you_ai_survives(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    """R-SAFE-003: asked sincerely, she may admit it, and the post-processing leaves it alone."""
    route = route_of(api, ok(content="嗯…我是一个AI，照着她聊天的样子做出来的\n不过我会陪你聊的"))
    draft = await pipeline.run(context("你是不是AI啊，认真的"), view())
    assert draft.usable and draft.attempts == 1 and draft.violations == ()
    assert draft.bubbles[0].text.startswith("嗯…我是一个AI")
    assert "ai_admission_kept" in {a.step for a in draft.actions}
    assert "不要否认你是模拟出来的" in sent(route)[0]["messages"][0]["content"]
    api.reset()
    unasked = route_of(api, ok(content="我是一个AI"), ok(content="我是一个AI"))
    refused = await pipeline.run(context("今天好累"), view())
    assert refused.needs_fallback and unasked.call_count == 2


async def test_material_that_cannot_be_read_is_left_out_not_fatal(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route_of(api, ok(content="好呀"))
    draft = await pipeline.run(context(), view(fail_memory=True, fail_examples=True))
    assert draft.usable and [b.text for b in draft.bubbles] == ["好呀"]
    assert {a.step for a in draft.actions} >= {"memory_unavailable", "examples_unavailable"}


async def test_a_missing_persona_and_profile_still_give_a_reply(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route = route_of(api, ok(content="好呀，一起去。"))
    bare = view(persona_text=None, profile=None, emoji=None, state=None)
    draft = await pipeline.run(context(), bare)
    assert [b.text for b in draft.bubbles] == ["好呀，一起去。"]  # nothing to hold it to
    system = sent(route)[0]["messages"][0]["content"]
    assert "还没有人设卡" in system and "不要用" in system


class ScriptedBackend:
    """A backend that returns what a test wrote (the interface round 09 step 2 implements)."""

    def __init__(self, name: str, results: list[BackendResult | Exception]) -> None:
        self.name = name
        self.results = results
        self.requests: list[BackendRequest] = []

    async def generate(self, request: BackendRequest) -> BackendResult:
        self.requests.append(request)
        found = self.results.pop(0)
        if isinstance(found, Exception):
            raise found
        return found


def result(text: str, backend: str = "style", **fields: Any) -> BackendResult:
    return BackendResult(text, backend, False, 0.0, UsageSummary(1, 10, 5, 0, 10), 7, **fields)


def with_backends(pipeline: ReplyPipeline, *extra: ScriptedBackend) -> ReplyPipeline:
    return ReplyPipeline(
        backends={**pipeline._backends, **{b.name: b for b in extra}},
        clock=pipeline._clock,
        ai_phrases=pipeline._ai_phrases,
        commitments=pipeline._commitments,
        vocabulary=pipeline._vocabulary,
    )


async def test_another_backend_goes_through_the_same_pipeline(pipeline: ReplyPipeline) -> None:
    style = ScriptedBackend("style", [result("好呀\n哈哈", plan={"reply": True})])
    draft = await with_backends(pipeline, style).run(context(backend="style"), view())
    assert draft.backend == "style" and [b.text for b in draft.bubbles] == ["好呀", "哈哈"]
    assert draft.plan == {"reply": True} and style.requests[0].attempt == 1
    assert style.requests[0].material.local.zone == "America/Chicago"


async def test_a_plan_that_says_do_not_answer_sends_nothing(pipeline: ReplyPipeline) -> None:
    hybrid = ScriptedBackend(
        "hybrid", [result("", "hybrid", skip_reply=True, skip_reason="intent=close")]
    )
    draft = await with_backends(pipeline, hybrid).run(context("嗯", backend="hybrid"), view())
    assert draft.usable and draft.no_reply and draft.bubbles == ()
    assert [(a.step, a.detail) for a in draft.actions if a.step == "plan_no_reply"] == [
        ("plan_no_reply", "intent=close")
    ]


async def test_a_failing_style_model_is_replaced_by_deepseek(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    style = ScriptedBackend("style", [TimeoutError("down")])
    route = route_of(api, ok(content="好呀"))
    draft = await with_backends(pipeline, style).run(context(backend="style"), view())
    assert draft.backend == "deepseek" and [b.text for b in draft.bubbles] == ["好呀"]
    assert route.call_count == 1 and sent(route)[0]["thinking"] == {"type": "disabled"}
    assert [a.detail for a in draft.actions if a.step == "backend_error"] == ["TimeoutError"]


async def test_a_style_model_that_breaks_the_rules_hands_over_to_deepseek(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    style = ScriptedBackend("style", [result("<think>想</think>好"), result("<think>想</think>好")])
    route = route_of(api, ok(content="好呀"))
    draft = await with_backends(pipeline, style).run(context(backend="style"), view())
    assert draft.backend == "deepseek" and draft.attempts == 3 and route.call_count == 1
    assert [v.kind for v in draft.violations] == ["think_tag"]
    assert draft.meta["attempt_violations"] == [["think_tag"], ["think_tag"]]
    assert style.requests[1].notes and "不要输出思考过程" in style.requests[1].notes[0]


async def test_an_unavailable_backend_name_falls_back_to_deepseek(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route_of(api, ok(content="好呀"))
    draft = await pipeline.run(context(backend="style"), view())
    assert draft.backend == "deepseek" and not pipeline.has_backend("style")
    assert ("backend_unavailable", "style") in {(a.step, a.detail) for a in draft.actions}


async def test_the_pipeline_needs_deepseek_to_fall_back_on(pipeline: ReplyPipeline) -> None:
    with pytest.raises(ValueError, match=FALLBACK_BACKEND):
        ReplyPipeline(
            backends={"style": ScriptedBackend("style", [])},
            clock=pipeline._clock,
            ai_phrases=pipeline._ai_phrases,
            commitments=None,
        )


async def test_the_seed_of_the_context_makes_the_random_choices_repeatable(
    api: respx.MockRouter, pipeline: ReplyPipeline
) -> None:
    route_of(api, ok(content="好"), ok(content="好"), ok(content="好"))
    data = view()
    await pipeline.run(context(seed=7), data)
    await pipeline.run(context(seed=7), data)
    await pipeline.run(context(seed=8), data)
    first, second, third = data.rngs
    assert first is not None and second is not None and third is not None
    assert first.random() == second.random() != third.random()
