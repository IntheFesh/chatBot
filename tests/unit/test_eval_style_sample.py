"""The style sample of a model: sized by the rarest share, priced, copied for a short hold-out.

R-SRV-005, R-EVAL-002 (round 18, user decision, R18-E5, D-852).  The synthetic world's hold-out
has 76 usable contexts and her shares are about her reference rates, so the rarest (her comma
rate) asks for some 700 replies: every context is answered several times, each copy with a seed of
its own, and nothing is spent before ``twin jobs approve``.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest
import respx

from tests.support.eval_world import DeepSeekScript
from tests.support.export_world import World
from twin.eval.blind import EVAL_GENERATE_JOB, EvalError
from twin.eval.render import Candidate, CandidateLine
from twin.eval.samples import DrawnSample, SampleDrawer
from twin.eval.store import EvalStore
from twin.eval.style_metrics import her_units_per_reply, replies_needed
from twin.eval.style_sample import (
    PLAN_MARGIN,
    EnoughAlready,
    plan_style_sample,
    replica_seed,
    replicate,
    sample_runs,
)
from twin.ops.jobs import JobQueue
from twin.profile.api import load_profile
from twin.services import Services

NO_QUOTES = {"supports_quote": False}


def estimate(world: World) -> int:
    """The replies per backend a first plan asks for: her rarest share, plus the margin."""
    profile = load_profile(world.services, "pre_holdout")
    assert profile is not None
    hers = profile.metrics
    return math.ceil(replies_needed(hers, NO_QUOTES, her_units_per_reply(hers)) * PLAN_MARGIN)


def answer_all(world: World, run_id: str, reply: Candidate) -> int:
    """Every pair of the run answered by ``reply``, as the generation leaves it."""
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(run_id)
    for item in items:
        store.save_generated(
            item.id,
            {
                "bot": reply.to_json(),
                "draft": {"backend": item.backend, "quote_lines": 0},
                "channel": {"supports_quote": False, "supports_typing": True},
            },
            cost_usd=0.0,
        )
    return len(items)


async def test_r18_e5_a_plan_answers_every_usable_context_with_copies_and_seeds_of_their_own(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    """R18-E5 (D-852): the size, the cap at the hold-out, the copies, the price, the approval."""
    plan = await plan_style_sample(world.services, ("deepseek",), model_id="m-1", seed=7)
    store = EvalStore(world.services.db, world.services.clock)
    run = store.get_run(plan.run.id)
    assert run.kind == "style" and run.params["purpose"] == "sample"
    assert run.params["model_id"] == "m-1" and run.params["min_expected"] == 50
    items = store.items(run.id)
    needed = estimate(world)
    assert 600 < needed < 900  # "about 700" at her rates
    assert plan.needed == needed and plan.replies == needed == len(items) and plan.have == 0
    assert plan.rarest == "comma_rate" and run.params["expected_per_reply"]["comma_rate"] > 0
    # the hold-out holds far fewer usable contexts: every one of them, answered again and again
    usable = SampleDrawer(world.services).draw(10_000, seed=1).samples
    assert plan.contexts == len(usable) < needed
    assert {item.sample_key for item in items} == {sample.sample_key for sample in usable}
    copies = Counter(item.sample_key for item in items)
    assert max(copies.values()) - min(copies.values()) <= 1  # spread as evenly as possible
    assert plan.copies == max(copies.values()) > 1
    numbered = {(item.sample_key, item.payload["replica"]) for item in items}
    assert len(numbered) == len(items)  # every copy has its own number...
    assert all(  # ...and the seed that number gives: the sample can be made again
        item.payload["seed"] == replica_seed(item.sample_key, "deepseek", item.payload["replica"])
        for item in items
    )
    assert len({item.payload["seed"] for item in items}) == len(items)
    assert all(item.left_is_bot is None and item.status == "pending" for item in items)
    # a one-time batch, priced from the prompts, waiting for `twin jobs approve`
    assert plan.estimated_usd > 0 and plan.batch_ids and script.requests == []
    jobs = JobQueue(world.services.db, world.services.clock).list_jobs(job_type=EVAL_GENERATE_JOB)
    assert jobs and all(job.requires_approval and job.approved_at is None for job in jobs)
    assert sample_runs(store, "m-1") == [run] and sample_runs(store, "m-2") == []
    # nothing more is planned while these replies wait to be generated
    with pytest.raises(EvalError, match="still has"):
        await plan_style_sample(world.services, ("deepseek",), model_id="m-1", seed=8)
    # the same seed draws the same sample (another model, so no top-up)
    again = await plan_style_sample(world.services, ("deepseek",), model_id="m-2", seed=7)
    assert sorted(
        (i.sample_key, i.payload["replica"], i.payload["seed"]) for i in store.items(again.run.id)
    ) == sorted((i.sample_key, i.payload["replica"], i.payload["seed"]) for i in items)


async def test_r18_e5_a_second_plan_tops_up_from_what_the_bot_really_wrote(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    """R18-E5 (D-852): a bot that writes one bubble a reply needs more replies than her estimate.

    One text a reply, no comma, no code, no sticker: her comma rate (3.1 %) expects some 24 commas
    in ~770 texts - not enough; the top-up is sized from the bot's own denominator (one text per
    reply) and continues the copies where the sample left them.
    """
    first = await plan_style_sample(world.services, ("deepseek",), model_id="m-3", seed=3)
    one_text = Candidate((CandidateLine("text", text="你好呀你好呀你好"),), None)
    have = answer_all(world, first.run.id, one_text)
    store = EvalStore(world.services.db, world.services.clock)
    profile = load_profile(world.services, "pre_holdout")
    assert profile is not None
    per_text = {"comma_rate": 1.0, "emoji_code_rate": 1.0, "quote_rate": 1.0, "sticker_share": 1.0}
    total = math.ceil(replies_needed(profile.metrics, NO_QUOTES, per_text) * PLAN_MARGIN)
    second = await plan_style_sample(world.services, ("deepseek",), model_id="m-3", seed=4)
    assert second.have == have and second.needed == second.replies == total - have
    assert min(item.payload["replica"] for item in store.items(second.run.id)) >= first.copies - 1
    keys = {
        (item.sample_key, item.payload["replica"])
        for run in (first.run, second.run)
        for item in store.items(run.id)
    }
    assert len(keys) == have + second.replies  # no copy was made twice


async def test_r18_e5_a_sample_with_enough_events_for_every_share_plans_nothing(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    """R18-E5 (D-852): five texts and a sticker a reply give every share more than 50 events."""
    first = await plan_style_sample(world.services, ("deepseek",), model_id="m-4", seed=5)
    lines = [CandidateLine("text", text="你好呀你好呀你好") for _ in range(5)]
    lines.append(CandidateLine("sticker", sticker_md5="ab" * 16))
    answer_all(world, first.run.id, Candidate(tuple(lines), None))
    with pytest.raises(EnoughAlready, match="nothing more to generate"):
        await plan_style_sample(world.services, ("deepseek",), model_id="m-4", seed=6)


def test_r18_e5_copies_go_round_the_pool_and_continue_the_numbers(world: World) -> None:
    """R18-E5 (D-852): every context once more before any twice more; numbers carry on."""
    pool = SampleDrawer(world.services).draw(3, seed=2).samples
    assert len(pool) == 3
    first: DrawnSample = pool[0]
    picks = replicate(pool, 7, {first.sample_key: 2})
    assert [replica for _, replica in picks] == [2, 0, 0, 3, 1, 1, 4]
    assert [sample.sample_key for sample, _ in picks[:3]] == [s.sample_key for s in pool]
    assert replicate([], 5, {}) == []


async def test_r18_e5_a_fresh_installation_says_what_to_do_first(services: Services) -> None:
    """R18-E5 (D-852): no hold-out, no profile: nothing is drawn or priced."""
    with pytest.raises(EvalError, match="pre-holdout profile"):
        await plan_style_sample(services, ("deepseek",), model_id="m-5", seed=1)
    with pytest.raises(EvalError, match="unknown backend"):
        await plan_style_sample(services, ("nobody",), model_id="m-5", seed=1)
    assert EvalStore(services.db, services.clock).list_runs("style") == []
