"""The style and hybrid backends in the sandbox, once a model is deployed (R-EVAL-001, R-EVAL-009).

No style model exists before round 14; these tests register one in the model registry and put a
scripted server behind it, which is exactly what round 14's ``twin model evaluate`` will do with
a real one.  The point: the sandbox asks the style and hybrid backends with the very same context
as DeepSeek, books what costs money as evaluation, and refuses to count a fall-back as a result.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta

import pytest
import respx
from sqlalchemy import select

from tests.support.eval_world import DeepSeekScript, request_for, sample_of
from tests.support.export_world import LIVE_STYLE, PAST_MARKER, World
from tests.support.memory import add_fact
from tests.support.style_models import ScriptedStyleClient, down, register_model
from twin.engine.style_runtime import StyleRuntime
from twin.eval.blind import blind_report, generate_items, plan_blind
from twin.eval.sandbox import SandboxKit, SandboxMode, backend_status, build_sandbox
from twin.eval.store import EvalStore
from twin.llm.runtime import build_llm_runtime
from twin.memory.recent import Turn
from twin.storage.models import CostLedger

BATCH = "eval-backends-1"


@asynccontextmanager
async def styled(world: World, client: ScriptedStyleClient) -> AsyncIterator[SandboxKit]:
    """A hold-out sandbox whose style backends talk to ``client``."""
    llm = build_llm_runtime(world.services)
    style = StyleRuntime.from_services(world.services, llm, client=client)
    kit = build_sandbox(
        world.services, mode=SandboxMode.HOLDOUT, batch_id=BATCH, llm=llm, style=style
    )
    try:
        yield kit
    finally:
        await kit.aclose()


def test_a_backend_is_available_once_a_model_that_this_code_can_render_is_active(
    world: World,
) -> None:
    services = world.services
    assert not backend_status(services, "style").available
    register_model(services, persona_version="v1")
    assert backend_status(services, "style").available
    assert backend_status(services, "hybrid").available


def test_a_model_bound_to_another_template_is_not_a_deployed_backend(world: World) -> None:
    register_model(world.services, persona_version="v1", template_version="qwen3_think@1")
    for name in ("style", "hybrid"):
        status = backend_status(world.services, name)
        assert not status.available and "qwen3_think@1" in status.message


async def test_the_style_backend_gets_the_same_eight_turns_and_only_what_was_known_then(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    register_model(world.services, persona_version="v1")
    client = ScriptedStyleClient(["好呀好呀\n哈哈"])
    sample = sample_of(world)[0]
    before = "一秒之前才知道的事蓝色自行车"
    after = "一秒之后才知道的事红色风筝"
    add_fact(world.memory, before, sample.at - timedelta(seconds=1), importance=5)
    add_fact(world.memory, after, sample.at + timedelta(seconds=1), importance=5)
    turns = tuple(
        Turn("bot" if i % 2 else "user", f"第{i}轮", sample.at, sample.at, (f"t{i}",))
        for i in range(12)
    )
    request = replace(request_for(sample, "style"), history=turns)
    async with styled(world, client) as kit:
        reply = await kit.sandbox.reply(request)
    assert reply.usable and reply.backend == "style" and not reply.fell_back
    assert [line.text for line in reply.candidate.lines] == ["好呀好呀", "哈哈"]
    assert script.requests == []  # the style model is local: DeepSeek was not asked
    prompt = client.prompts[0].text
    assert PAST_MARKER in prompt and LIVE_STYLE not in prompt  # the pre-holdout card
    assert before in prompt and after not in prompt  # the memory of that moment
    assert all(f"第{i}轮" in prompt for i in range(5, 12))  # the same seven turns DeepSeek gets
    assert not any(f"第{i}轮" in prompt for i in range(5))
    assert prompt.rstrip().endswith("assistant")  # the model continues her turn


async def test_the_hybrid_backend_plans_with_deepseek_booked_as_evaluation(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    register_model(world.services, persona_version="v1")
    plan = (
        '{"reply": true, "intent": "答应", "facts_to_use": [], "tone": "轻松", '
        '"bubble_hint": "两条", "sticker_hint": ""}'
    )
    script.reply = lambda body: plan
    client = ScriptedStyleClient(["好呀"])
    sample = sample_of(world)[0]
    async with styled(world, client) as kit:
        reply = await kit.sandbox.reply(request_for(sample, "hybrid"))
    assert reply.usable and reply.backend == "hybrid" and not reply.fell_back
    assert len(script.requests) == 1 and "【规划】" in client.prompts[0].text
    with world.services.db.session() as session:
        rows = [
            (r.purpose, r.account, r.batch_id) for r in session.scalars(select(CostLedger)).all()
        ]
    assert rows == [("eval", "one_time", BATCH)]  # the planner is not booked as a plan or a reply


async def test_a_style_model_that_is_down_is_a_failed_pair_not_a_deepseek_result(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    register_model(world.services, persona_version="v1")
    client = ScriptedStyleClient([down("connection refused")] * 4)
    sample = sample_of(world)[0]
    async with styled(world, client) as kit:
        reply = await kit.sandbox.reply(request_for(sample, "style"))
    assert reply.fell_back and reply.backend == "deepseek"  # the pipeline kept the bot talking
    assert len(script.requests) == 1  # ... with DeepSeek, which is exactly what must not count


async def test_a_blind_test_of_deepseek_and_the_style_model_on_the_same_contexts(
    world: World,
    api: respx.MockRouter,
    script: DeepSeekScript,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparison round 14 makes: one set of contexts, a pair per backend, one report."""
    register_model(world.services, persona_version="v1")
    client = ScriptedStyleClient(["好呀好呀"])
    # the style runtime a job builds talks to the scripted server instead of a real one
    monkeypatch.setattr(
        "twin.engine.style_runtime.ConfiguredStyleClient",
        lambda config, clock, api_key=None: client,
    )
    planned = await plan_blind(world.services, ("deepseek", "style"), 6, seed=2)
    assert planned.samples == 6 and planned.pairs == 12 and planned.estimated_usd > 0
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(planned.run.id)
    by_backend = {b: [i.sample_key for i in items if i.backend == b] for b in ("deepseek", "style")}
    assert by_backend["deepseek"] == by_backend["style"]  # the same contexts for both backends
    runtime = build_llm_runtime(world.services)
    for batch in planned.batch_ids:
        runtime.batches.approve(batch)
    summary = await generate_items(
        world.services, planned.run.id, [i.id for i in items], planned.batch_ids[0]
    )
    assert summary.generated == 12 and summary.failed == 0
    assert len(script.requests) == 6 and len(client.prompts) == 6  # DeepSeek once per deepseek pair
    report = blind_report(store, store.get_run(planned.run.id))
    assert [b.backend for b in report.backends] == ["deepseek", "style"]
    assert all(b.generated == 6 for b in report.backends)
    assert {
        i.payload["draft"]["backend"] for i in store.items(planned.run.id, backend="style")
    } == {"style"}
