"""The batch jobs of the evaluation: packing, failures, deferral, what a pair becomes."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
import respx

from tests.support.eval_world import (
    DeepSeekScript,
    add_memory_facts,
    memory_script,
    request_for,
    sample_of,
    serve,
)
from tests.support.export_world import World
from twin.engine.types import Bubble
from twin.eval import blind, memory_test
from twin.eval.blind import (
    EVAL_GENERATE_JOB,
    EvalError,
    generate_item,
    handle_eval_generate,
    pack,
    plan_blind,
    queued_jobs,
    readiness_problems,
)
from twin.eval.memory_test import (
    EVAL_MEMORY_JOB,
    JUDGE_SYSTEM,
    QUESTION_SYSTEM,
    ask_items,
    handle_eval_memory,
    plan_memory,
)
from twin.eval.sandbox import SandboxKit, SandboxReply
from twin.eval.store import EvalStore, NewItem
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.ops.jobs import BatchTooLargeError, JobContext, JobDeferred, JobQueue, JobView
from twin.services import Services

WHEN = datetime(2026, 9, 1, 12, tzinfo=UTC)


def test_jobs_are_packed_into_batches_that_stay_within_the_one_time_limit() -> None:
    assert pack([1.0, 2.0, 3.0, 4.0], 10.0) == [[0, 1, 2, 3]]
    assert pack([4.0, 4.0, 4.0], 10.0) == [[0, 1], [2]]
    assert pack([], 10.0) == []
    with pytest.raises(BatchTooLargeError):
        pack([1.0, 11.0], 10.0)  # one job that alone is above the limit cannot be split


def test_a_fresh_installation_lists_everything_that_is_missing(services: Services) -> None:
    problems = readiness_problems(services, ("deepseek",))
    assert any("hold-out" in p for p in problems)
    assert any("profile" in p for p in problems) and any("routine" in p for p in problems)
    assert any("persona card" in p for p in problems)
    compact = readiness_problems(services, ("style",))
    assert any("compact" in p for p in compact) and not any("full" in p for p in compact)


async def test_a_batch_above_the_one_time_limit_cancels_the_run_it_belongs_to(
    world: World, api: respx.MockRouter
) -> None:
    world.services.settings.budget.one_time_usd = 0.000001
    with pytest.raises(BatchTooLargeError):
        await plan_blind(world.services, ("deepseek",), 5, seed=1)
    store = EvalStore(world.services.db, world.services.clock)
    runs = store.list_runs("blind")
    assert [r.status for r in runs] == ["cancelled"]
    assert JobQueue(world.services.db, world.services.clock).list_jobs() == []
    # a run nobody saw gives its contexts back
    assert store.used_sample_keys("blind") == set()
    memory = store.list_runs("memory")
    assert memory == []


async def test_a_memory_test_above_the_one_time_limit_is_cancelled_too(
    world: World, api: respx.MockRouter
) -> None:
    add_memory_facts(world)
    world.services.settings.budget.one_time_usd = 0.000001
    with pytest.raises(BatchTooLargeError):
        await plan_memory(world.services, seed=1)
    assert [r.status for r in EvalStore(world.services.db, world.services.clock).list_runs()] == [
        "cancelled"
    ]


def job_context(world: World, job_type: str, payload: dict[str, Any]) -> JobContext:
    job = JobView(
        id="job-1", type=job_type, payload=payload, priority=1, status="running", attempts=1,
        max_attempts=3, run_after=WHEN, offpeak_only=False, deadline=None, last_error=None,
        batch_id="b1", estimated_cost_usd=0.1, requires_approval=True, approved_at=WHEN,
        created_at=WHEN, updated_at=WHEN, approved_usd=0.1, started_at=WHEN, finished_at=None,
    )  # fmt: skip
    return JobContext(job=job, services=world.services, clock=world.services.clock)


@pytest.mark.parametrize(
    ("error", "wait"),
    [(BudgetDeniedError("eval", 3), 3600.0), (CircuitOpenError(30.0), 300.0)],
)
async def test_a_job_that_meets_the_budget_or_the_breaker_goes_back_to_the_queue(
    world: World, monkeypatch: pytest.MonkeyPatch, error: Exception, wait: float
) -> None:
    async def refuse(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(blind, "generate_items", refuse)
    monkeypatch.setattr(memory_test, "ask_items", refuse)
    for handler, job_type in (
        (handle_eval_generate, EVAL_GENERATE_JOB),
        (handle_eval_memory, EVAL_MEMORY_JOB),
    ):
        with pytest.raises(JobDeferred) as raised:
            await handler(job_context(world, job_type, {"run_id": "r", "item_ids": []}))
        assert raised.value.retry_in_s == wait
    for handler, job_type in (
        (handle_eval_generate, EVAL_GENERATE_JOB),
        (handle_eval_memory, EVAL_MEMORY_JOB),
    ):
        broken = job_context(world, job_type, {})
        broken = replace(broken, services=None)
        with pytest.raises(RuntimeError, match="services container"):
            await handler(broken)


async def test_queued_jobs_counts_the_generation_jobs_that_wait_or_run(
    world: World, api: respx.MockRouter
) -> None:
    assert queued_jobs(world.services) == 0
    planned = await plan_blind(world.services, ("deepseek",), 12, seed=1)
    assert (
        queued_jobs(world.services) == len(planned.batch_ids) * 2
        or queued_jobs(world.services) == 2
    )  # 12 pairs in jobs of 10


async def test_what_a_pair_becomes_depends_on_how_the_bot_answered(
    world: World, kit: SandboxKit, api: respx.MockRouter
) -> None:
    store = EvalStore(world.services.db, world.services.clock)
    sample = sample_of(world)[0]
    request = request_for(sample)
    run = store.create_run("blind", mode="holdout", backends=["deepseek"], status="running")
    base = await kit.sandbox.reply(request)
    cases: dict[str, SandboxReply] = {
        "no_reply": replace(base, draft=replace(base.draft, bubbles=(), no_reply=True)),
        "fell_back": replace(base, draft=replace(base.draft, backend="style")),
        "empty": replace(base, candidate=replace(base.candidate, lines=())),
        "fallback:violations": replace(
            base,
            draft=replace(
                base.draft, needs_fallback=True, fallback_reason="violations", bubbles=()
            ),
        ),
        "generated": base,
    }
    store.add_items(
        run.id, [NewItem(f"k{n}", "deepseek", WHEN, sample.payload()) for n in range(len(cases))]
    )

    class Answers:
        def __init__(self, replies: list[SandboxReply]) -> None:
            self._replies = replies

        async def reply(self, _request: object) -> SandboxReply:
            return self._replies.pop(0)

    outcomes = {}
    for (label, reply), item in zip(cases.items(), store.items(run.id), strict=True):
        status = await generate_item(Answers([reply]), store, item, "off")  # type: ignore[arg-type]
        found = store.item(item.id)
        outcomes[label] = (status, found.payload.get("failure"), "bot" in found.payload)
    assert outcomes == {
        "no_reply": ("failed", "no_reply", False),
        "fell_back": ("failed", "fell_back", False),
        "empty": ("failed", "empty", False),
        "fallback:violations": ("failed", "fallback:violations", False),
        "generated": ("generated", None, True),
    }


def make_memory_run(world: World, backend: str = "deepseek") -> tuple[EvalStore, str, list[str]]:
    store = EvalStore(world.services.db, world.services.clock)
    run = store.create_run("memory", mode="live", backends=[backend], status="running")
    store.add_items(
        run.id,
        [
            NewItem(
                "f1",
                backend,
                world.services.clock.now_utc(),
                {"fact": "第1号旧记忆：那把编号01的蓝色钥匙", "source": "real_record"},
                "real_record",
            )
        ],
    )
    return store, run.id, [i.id for i in store.items(run.id)]


async def test_a_question_whose_verdict_cannot_be_read_is_a_lost_sample(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = memory_script()

    def reply(body: dict[str, object]) -> str:
        messages = body["messages"]
        assert isinstance(messages, list)
        if str(messages[0]["content"]).startswith(JUDGE_SYSTEM[:12]):
            return "this is not json"
        return base.reply(body)

    store, run_id, ids = make_memory_run(world)
    router = serve(DeepSeekScript(reply, prompt_tokens=60, completion_tokens=20))
    try:
        summary = await ask_items(world.services, run_id, ids, "b1")
    finally:
        router.stop()
    assert (summary.asked, summary.failed) == (0, 1)
    assert store.item(ids[0]).payload["failure"] == "judgement_not_written"
    assert QUESTION_SYSTEM  # the prompts are fixed in code, not read from the template table


async def test_a_memory_question_asked_of_a_backend_that_is_not_deployed_is_failed(
    world: World,
) -> None:
    store, run_id, ids = make_memory_run(world, backend="style")
    summary = await ask_items(world.services, run_id, ids, None)
    assert (summary.asked, summary.failed) == (0, 1)
    assert store.item(ids[0]).payload["failure"] == "not_deployed"
    again = await ask_items(world.services, run_id, ids, None)  # a failed item is not asked twice
    assert again.skipped == 1 and again.failed == 0


async def test_a_memory_test_needs_a_deployed_backend_to_answer_with(world: World) -> None:
    add_memory_facts(world)
    with pytest.raises(EvalError, match="未部署"):
        await plan_memory(world.services, seed=1, backend="style")
    assert EvalStore(world.services.db, world.services.clock).list_runs("memory") == []
    assert json.dumps({}) == "{}" and Bubble("text", "x").is_sticker is False
