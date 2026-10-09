"""The blind test: drawing, pricing, generating and counting (R-EVAL-001, R-LLM-014, R-SAFE-006)."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise

import pytest
import respx
from sqlalchemy import select

from tests.support.eval_world import DeepSeekScript, eval_worker, open_kit, sample_of
from tests.support.export_world import World
from twin.eval.blind import (
    EVAL_GENERATE_JOB,
    EvalError,
    assign_sides,
    blind_report,
    generate_items,
    plan_blind,
    price_pairs,
    readiness_problems,
)
from twin.eval.samples import DrawResult
from twin.eval.stats import binomial_z, two_proportion_test
from twin.eval.store import EvalStore, NewItem
from twin.eval.ui import presentation_order
from twin.llm.runtime import build_llm_runtime
from twin.ops.jobs import JobQueue
from twin.services import Services
from twin.storage.models import CostLedger


async def plan(world: World, n: int = 10, seed: int = 1, *backends: str):  # type: ignore[no-untyped-def]
    return await plan_blind(world.services, backends or ("deepseek",), n, seed=seed)


async def test_planning_draws_stores_and_prices_the_pairs_without_spending(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    planned = await plan(world, 10, 4)
    assert planned.samples == 10 and planned.pairs == 10 and not planned.short
    assert planned.estimated_usd > 0 and planned.batch_ids
    assert script.requests == []  # nothing was sent: the estimate is made from the prompts
    store = EvalStore(world.services.db, world.services.clock)
    run = store.get_run(planned.run.id)
    assert run.kind == "blind" and run.mode == "holdout" and run.backends == ("deepseek",)
    assert run.params["seed"] == 4 and run.params["n"] == 10
    assert run.params["holdout_cutoff"] == world.cutoff.isoformat()
    assert run.params["context_turns"] == 8 and run.batch_ids == tuple(planned.batch_ids)
    items = store.items(run.id)
    assert len(items) == 10 and {i.status for i in items} == {"pending"}
    assert all(i.at >= world.cutoff for i in items)  # only the hold-out window
    assert all(i.left_is_bot is not None and i.period and i.length_bin for i in items)
    assert all({"inbound", "history", "shown", "real", "seed"} <= set(i.payload) for i in items)
    # the generation is queued as one-time batches that wait for `twin jobs approve`
    queue = JobQueue(world.services.db, world.services.clock)
    jobs = queue.list_jobs(job_type=EVAL_GENERATE_JOB)
    assert jobs and all(j.requires_approval and j.approved_at is None for j in jobs)
    assert all(j.batch_id in planned.batch_ids for j in jobs)
    status = build_llm_runtime(world.services).batches.status(planned.batch_ids[0])
    assert status.estimated_usd > 0 and not status.approved and status.spent_usd == 0


async def test_the_pairs_are_generated_after_the_approval_and_cost_less_than_estimated(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    planned = await plan(world, 10, 4)
    runtime = build_llm_runtime(world.services)
    worker = eval_worker(world.services)
    summary = await worker.run_until_idle()  # nothing is approved yet: nothing runs
    assert summary.done == 0 and script.requests == []
    for batch in planned.batch_ids:
        runtime.batches.approve(batch)
    summary = await worker.run_until_idle()
    assert summary.done >= 1 and summary.failed == 0
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(planned.run.id)
    assert {i.status for i in items} == {"generated"} and len(script.requests) == 10
    for item in items:
        bot = item.payload["bot"]
        assert [line["t"] for line in bot["lines"]] == ["好呀", "哈哈[拥抱]"]
        assert item.payload["draft"]["backend"] == "deepseek" and item.cost_usd > 0
    spent = sum(runtime.ledger.batch_spent_usd(b) for b in planned.batch_ids)
    assert 0 < spent <= planned.estimated_usd  # the estimate is an upper bound
    with world.services.db.session() as session:
        rows = session.scalars(select(CostLedger)).all()
        assert {(r.purpose, r.account) for r in rows} == {("eval", "one_time")}
        assert {r.batch_id for r in rows} == set(planned.batch_ids)


async def test_a_stopped_generation_goes_on_with_the_pairs_that_are_left(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    planned = await plan(world, 10, 4)
    store = EvalStore(world.services.db, world.services.clock)
    ids = [item.id for item in store.items(planned.run.id)]
    first = await generate_items(world.services, planned.run.id, ids[:4], planned.batch_ids[0])
    assert (first.generated, first.skipped) == (4, 0) and len(script.requests) == 4
    second = await generate_items(world.services, planned.run.id, ids, planned.batch_ids[0])
    assert (second.generated, second.skipped) == (6, 4) and len(script.requests) == 10
    assert store.counts(planned.run.id)["generated"] == 10


async def test_a_pair_the_bot_could_not_answer_is_failed_and_not_judged(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    script.reply = lambda body: "作为一个AI语言模型，我无法回答这个问题"
    planned = await plan(world, 4, 2)
    ids = [i.id for i in EvalStore(world.services.db, world.services.clock).items(planned.run.id)]
    summary = await generate_items(world.services, planned.run.id, ids, planned.batch_ids[0])
    assert summary.failed == 4 and summary.generated == 0
    items = EvalStore(world.services.db, world.services.clock).items(planned.run.id)
    assert {i.status for i in items} == {"failed"}
    assert all(str(i.payload["failure"]).startswith("fallback:") for i in items)
    assert all("bot" not in i.payload for i in items)
    report = blind_report(EvalStore(world.services.db, world.services.clock), planned.run)
    entry = report.backends[0]
    assert (entry.failed, entry.judged, entry.generated) == (4, 0, 0)


async def test_a_reply_by_another_backend_is_failed_instead_of_counted(
    world: World, api: respx.MockRouter
) -> None:
    """The style model is not deployed: a pair asked of it would be DeepSeek's reply in disguise."""
    planned = await plan(world, 2, 2)
    store = EvalStore(world.services.db, world.services.clock)
    sample = store.items(planned.run.id)[0]
    store.add_items(
        planned.run.id,
        [
            NewItem(
                sample.sample_key, "style", sample.at, sample.payload, "x", "night", "short", True
            )
        ],
    )
    last = store.items(planned.run.id)[-1]
    summary = await generate_items(world.services, planned.run.id, [last.id], planned.batch_ids[0])
    assert summary.failed == 1
    assert store.item(last.id).payload["failure"] == "not_deployed"


async def test_contexts_of_an_earlier_test_are_not_drawn_again(
    world: World, api: respx.MockRouter
) -> None:
    first = await plan(world, 12, 1)
    second = await plan(world, 12, 2)
    store = EvalStore(world.services.db, world.services.clock)
    a = {i.sample_key for i in store.items(first.run.id)}
    b = {i.sample_key for i in store.items(second.run.id)}
    assert len(a) == 12 and len(b) == 12 and not a & b
    assert second.excluded["context_used_before"] >= 12
    # a cancelled test that nobody saw gives its contexts back
    store.update_run(first.run.id, status="cancelled")
    third = await plan(world, 40, 3)
    assert {i.sample_key for i in store.items(third.run.id)} & a


async def test_a_backend_that_is_not_deployed_is_refused_before_anything_is_stored(
    world: World, api: respx.MockRouter
) -> None:
    with pytest.raises(EvalError, match="未部署"):
        await plan_blind(world.services, ("deepseek", "style"), 5, seed=1)
    assert EvalStore(world.services.db, world.services.clock).list_runs() == []
    assert JobQueue(world.services.db, world.services.clock).list_jobs() == []


async def test_a_fresh_installation_says_what_to_do_first(services: Services) -> None:
    problems = readiness_problems(services, ("deepseek",))
    assert any("hold-out" in p for p in problems)
    with pytest.raises(EvalError, match="hold-out"):
        await plan_blind(services, ("deepseek",), 5, seed=1)


async def test_the_planner_prices_each_backend_by_what_it_sends(
    world: World, api: respx.MockRouter
) -> None:
    """A style-only pair costs nothing; DeepSeek's reply and the hybrid planner are priced."""

    drawn = sample_of(world, 2)

    async with open_kit(world, batch_id=None) as kit:
        result = DrawResult(samples=drawn)
        prices = await price_pairs(
            world.services, kit, result, ("deepseek", "style", "hybrid"), "off"
        )
    for sample in drawn:
        reply, style, hybrid = (
            prices[(sample.sample_key, b)] for b in ("deepseek", "style", "hybrid")
        )
        assert reply > hybrid > 0 and style == 0.0


def test_the_sides_are_independent_fair_coins() -> None:
    sides = assign_sides(4000, 9)
    assert abs(binomial_z(sum(sides), 4000)) < 3  # the left-right split does not reject 0.5
    assert sides == assign_sides(4000, 9) and sides != assign_sides(4000, 10)
    runs = sum(a != b for a, b in pairwise(sides))
    assert 1800 < runs < 2200  # no pattern in the order either


def _store_with_run(services: Services, backends: tuple[str, ...]) -> tuple[EvalStore, str]:
    store = EvalStore(services.db, services.clock)
    run = store.create_run("blind", mode="holdout", backends=backends, status="running")
    return store, run.id


def _add(
    store: EvalStore, run_id: str, backend: str, count: int, *, period: str, length: str
) -> list[str]:
    at = datetime(2026, 9, 1, 12, tzinfo=UTC)
    start = len(store.items(run_id))
    store.add_items(
        run_id,
        [
            NewItem(f"k{start + n}", backend, at, {"n": n}, None, period, length, n % 2 == 0)
            for n in range(count)
        ],
    )
    return [i.id for i in store.items(run_id)[start:]]


def test_the_report_counts_valid_judgements_the_rates_and_the_tests(services: Services) -> None:
    store, run_id = _store_with_run(services, ("deepseek", "style"))
    deepseek = _add(store, run_id, "deepseek", 10, period="night", length="short")
    style = _add(store, run_id, "style", 10, period="evening", length="long")
    for item_id in deepseek + style:
        store.save_generated(item_id, {"bot": {"lines": [], "quote": None}}, cost_usd=0.0)
    for item_id in deepseek[:7]:  # deepseek: 6 of 7 guessed right
        store.judge(item_id, "correct" if item_id != deepseek[0] else "wrong")
    store.judge(deepseek[7], None)  # a skip is not a judgement
    for position, item_id in enumerate(style[:8]):  # style: 3 of 8
        store.judge(item_id, "correct" if position < 3 else "wrong")
    store.mark_failed(style[9], "no_reply")
    report = blind_report(store, store.get_run(run_id))
    first, second = report.backends
    assert (first.backend, first.judged, first.correct, first.skipped) == ("deepseek", 7, 6, 1)
    assert first.rate.point == pytest.approx(6 / 7) and first.waiting == 2
    assert (second.judged, second.correct, second.failed, second.waiting) == (8, 3, 1, 1)
    low, high = first.rate.interval or (0.0, 0.0)
    assert low < 6 / 7 < high <= 1.0
    assert first.by_period["night"].total == 7 and "evening" not in first.by_period
    assert second.by_length["long"].successes == 3
    assert report.of("style") is second and report.of("hybrid") is None
    expected = two_proportion_test(3, 8, 6, 7)
    assert expected is not None and len(report.comparisons) == 1
    comparison = report.comparisons[0]
    assert (comparison.first, comparison.second) == ("deepseek", "style")
    assert comparison.test == two_proportion_test(6, 7, 3, 8)


def test_pairs_of_one_context_are_a_whole_pass_apart_in_the_order_they_are_shown(
    services: Services,
) -> None:
    store, run_id = _store_with_run(services, ("deepseek", "style", "hybrid"))
    at = datetime(2026, 9, 1, 12, tzinfo=UTC)
    contexts = [f"c{n}" for n in range(6)]
    store.add_items(
        run_id,
        [NewItem(c, b, at, {}) for c in contexts for b in ("deepseek", "style", "hybrid")],
    )
    items = store.items(run_id)
    shown = presentation_order(items)
    assert sorted(i.id for i in shown) == sorted(i.id for i in items)
    for context in contexts:
        positions = [n for n, item in enumerate(shown) if item.sample_key == context]
        assert len(positions) == 3
        assert min(b - a for a, b in pairwise(positions)) >= 3
    first_pass = {item.backend for item in shown[:6]}
    assert len(first_pass) >= 2  # not one backend first for everybody
    assert presentation_order(items) == shown  # the order is stable: a resumed run keeps it
    one = presentation_order([i for i in items if i.backend == "deepseek"])
    assert [i.sample_key for i in one] == contexts  # one backend: the order of the run
