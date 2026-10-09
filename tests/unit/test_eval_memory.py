"""The memory test: twenty questions, ten from each source (R-EVAL-003, R-LLM-014)."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from tests.support.eval_world import (
    KNOWN,
    DeepSeekScript,
    add_memory_facts,
    eval_worker,
    memory_script,
    serve,
)
from tests.support.export_world import World
from twin.eval.isolation import changes, snapshot
from twin.eval.memory_test import (
    JUDGE_SYSTEM,
    PER_SOURCE,
    QUESTION_SYSTEM,
    TOTAL,
    MemorySession,
    MemorySummary,
    eligible_facts,
    finish_run,
    leaked,
    plan_memory,
    summarize,
)
from twin.eval.store import EvalStore, ItemView, NewItem
from twin.eval.ui import LineKeys
from twin.llm.runtime import build_llm_runtime
from twin.ops.jobs import JobQueue


def console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=100, color_system=None, highlight=False), buffer


def item(source: str, outcome: str | None, status: str = "judged") -> ItemView:
    return ItemView(
        "i", "r", 0, "k", "deepseek", source, None, None, KNOWN, status, None, outcome, None,
        None, 0.0, None, {},
    )  # fmt: skip


def summary_of(correct: int, partial: int, wrong: int, *, real: int = 10) -> MemorySummary:
    outcomes = ["correct"] * correct + ["partial"] * partial + ["wrong"] * wrong
    items = [
        item("real_record" if n < real else "user_said", outcome)
        for n, outcome in enumerate(outcomes)
    ]
    return summarize(items)


@pytest.mark.parametrize(
    ("correct", "partial", "wrong", "verdict"),
    [
        (16, 0, 4, "passed"),  # exactly 80 %
        (15, 1, 4, "failed"),  # 15.5 of 20
        (15, 2, 3, "passed"),  # 16 of 20: two halves make a whole
        (14, 4, 2, "passed"),
        (0, 20, 0, "failed"),  # all halves: 50 %
        (20, 0, 0, "passed"),
        (8, 0, 12, "failed"),
    ],
)
def test_the_score_counts_a_partial_answer_as_half_and_eighty_percent_passes(
    correct: int, partial: int, wrong: int, verdict: str
) -> None:
    score = summary_of(correct, partial, wrong)
    assert score.total == TOTAL and score.composition_complete and score.complete
    assert score.points == correct + 0.5 * partial
    assert score.verdict == verdict and score.meets_threshold == (verdict == "passed")


def test_a_run_without_the_full_composition_or_the_review_is_not_passed() -> None:
    lopsided = summary_of(20, 0, 0, real=12)  # twelve real, eight from the bot
    assert lopsided.accuracy == 1.0 and not lopsided.composition_complete
    assert lopsided.verdict == "insufficient"
    unreviewed = summarize(
        [item("real_record" if n < 10 else "bot_invented", None, "generated") for n in range(20)]
    )
    assert unreviewed.reviewed == 0 and unreviewed.verdict == "insufficient"
    assert summarize([]).verdict == "insufficient" and summarize([]).accuracy is None
    broken = summarize(
        [item("real_record", "correct")] * 10
        + [item("bot_invented", "correct")] * 9
        + [item("bot_invented", None, "failed")]
    )
    assert (
        broken.failed == 1 and not broken.composition_complete
    )  # a lost question is a lost sample


def test_a_question_that_contains_its_answer_is_recognised() -> None:
    assert leaked("你家猫叫豆包吗", ["豆包"]) == "豆包"
    assert leaked("你家猫叫什么", ["豆包"]) is None
    assert leaked("你好", ["好"]) is None  # a single character is not an answer


async def test_the_plan_takes_ten_from_each_source_and_prices_the_questions(world: World) -> None:
    add_memory_facts(world, real=12, bot=12)
    script = memory_script()
    router = serve(script)
    try:
        plan = await plan_memory(world.services, seed=3)
    finally:
        router.stop()
    assert not plan.insufficient and plan.estimated_usd > 0 and plan.batch_ids
    assert script.requests == []  # the plan costs nothing: it is priced, not run
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(plan.run.id)
    assert len(items) == TOTAL and {i.status for i in items} == {"pending"}
    sources = [i.source for i in items]
    assert sources.count("real_record") == PER_SOURCE
    assert sum(s in {"user_said", "bot_invented"} for s in sources) == PER_SOURCE
    assert len({i.sample_key for i in items}) == TOTAL
    run = store.get_run(plan.run.id)
    assert run.kind == "memory" and run.mode == "live" and run.params["seed"] == 3
    assert run.params["real_available"] == 14 and run.params["bot_available"] == 12
    queue = JobQueue(world.services.db, world.services.clock)
    assert all(j.requires_approval for j in queue.list_jobs(job_type="eval_memory"))
    # the same seed draws the same facts
    router = serve(memory_script())
    try:
        again = await plan_memory(world.services, seed=3)
    finally:
        router.stop()
    assert {i.sample_key for i in store.items(again.run.id)} == {i.sample_key for i in items}


async def test_too_few_facts_from_the_bots_conversation_is_not_passed_and_nothing_fills_in(
    world: World,
) -> None:
    add_memory_facts(world, real=15, bot=7)
    plan = await plan_memory(world.services, seed=1)
    assert plan.insufficient and plan.bot_available == 7 and plan.real_available >= 15
    store = EvalStore(world.services.db, world.services.clock)
    run = store.get_run(plan.run.id)
    assert run.status == "done" and run.verdict == "insufficient"
    assert run.summary["passed"] is False and run.summary["total"] == 0
    assert any("chat a few more days" in r for r in run.summary["reasons"])
    assert store.items(run.id) == []  # no real record was drawn to make up the number
    assert JobQueue(world.services.db, world.services.clock).list_jobs() == []


async def test_too_few_real_records_is_not_passed_either(world: World) -> None:
    add_memory_facts(world, real=0, bot=12)
    now = world.services.clock.now_utc()
    real, bot = eligible_facts(world.services, now)
    assert len(real) == 2 and len(bot) == 12  # the two facts of the world are all that is real
    plan = await plan_memory(world.services, seed=1)
    assert plan.insufficient and any("real records" in r for r in plan.run.summary["reasons"])


async def run_whole(world: World, script: DeepSeekScript, *, seed: int = 3):  # type: ignore[no-untyped-def]
    router = serve(script)
    try:
        plan = await plan_memory(world.services, seed=seed)
        runtime = build_llm_runtime(world.services)
        for batch in plan.batch_ids:
            runtime.batches.approve(batch)
        summary = await eval_worker(world.services).run_until_idle()
    finally:
        router.stop()
    return plan, summary, runtime


async def test_the_questions_are_asked_in_live_mode_judged_and_booked_as_evaluation(
    world: World,
) -> None:
    add_memory_facts(world)
    script = memory_script()
    router = serve(script)
    try:
        plan = await plan_memory(world.services, seed=3)
        runtime = build_llm_runtime(world.services)
        for batch in plan.batch_ids:
            runtime.batches.approve(batch)
        before = snapshot(world.services.db)
        summary = await eval_worker(world.services).run_until_idle()
        after = snapshot(world.services.db)
    finally:
        router.stop()
    assert summary.failed == 0 and summary.done >= 1
    assert changes(before, after).outside() == []  # evaluation tables, ledger, jobs: nothing else
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(plan.run.id)
    assert {i.status for i in items} == {"generated"}
    for found in items:
        payload = found.payload
        assert payload["question"].endswith("是什么情况") and payload["key_points"]
        assert payload["answered"] is True and payload["answered_by"] == "deepseek"
        assert found.auto_outcome in {"correct", "wrong"}
        # evidence: the fact, its source, when it was known and what it came from
        assert payload["fact"] and payload["source"] == found.source and payload["known_at"]
        assert payload["evidence"]["ids"] and payload["fact_number"] >= 1
    assert sum(i.auto_outcome == "correct" for i in items) >= 15  # live memory finds the facts
    spent = sum(runtime.ledger.batch_spent_usd(b) for b in plan.batch_ids)
    assert 0 < spent <= plan.estimated_usd
    assert script.errors == [] and len(script.requests) == 3 * TOTAL  # question, answer, verdict


async def test_the_user_reviews_the_verdicts_and_can_overturn_them(world: World) -> None:
    add_memory_facts(world)
    plan, summary, _ = await run_whole(world, memory_script())
    assert summary.failed == 0
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(plan.run.id)
    automatic = [i.auto_outcome for i in items]
    # Enter keeps the first, `w` turns the second into wrong, `p` the third into partial, `c`
    # the fourth into correct; the rest are confirmed with `k`
    keys = "\nw\np\nc\n" + "k\n" * (TOTAL - 4)
    out, buffer = console()
    outcome = MemorySession(
        store, store.get_run(plan.run.id), out, LineKeys(io.StringIO(keys))
    ).run()
    assert outcome.reviewed == TOTAL and outcome.finished and not outcome.quit
    reviewed = store.items(plan.run.id)
    assert {i.status for i in reviewed} == {"judged"}
    assert [i.outcome for i in reviewed[:4]] == [automatic[0], "wrong", "partial", "correct"]
    assert reviewed[1].payload["changed"] is (automatic[1] != "wrong")
    assert reviewed[2].score == 0.5 and reviewed[3].score == 1.0
    assert all(r.auto_outcome == a for r, a in zip(reviewed, automatic, strict=True))  # kept
    run = store.get_run(plan.run.id)
    assert run.status == "done" and run.summary["composition_complete"] is True
    assert run.summary["total"] == TOTAL and run.summary["reviewed"] == TOTAL
    score = summarize(reviewed)
    assert run.verdict == score.verdict and run.summary["points"] == score.points
    assert "第 1/20 题" in buffer.getvalue() and "要点：" in buffer.getvalue()


async def test_a_review_that_stops_is_continued_and_counts_only_when_complete(
    world: World,
) -> None:
    add_memory_facts(world)
    plan, _, _ = await run_whole(world, memory_script())
    store = EvalStore(world.services.db, world.services.clock)
    out, _ = console()
    first = MemorySession(
        store, store.get_run(plan.run.id), out, LineKeys(io.StringIO("k\nk\nq\n"))
    ).run()
    assert first.quit and first.reviewed == 2 and not first.finished
    assert store.get_run(plan.run.id).status != "done"
    assert summarize(store.items(plan.run.id)).verdict == "insufficient"  # 2 of 20 reviewed
    rest = MemorySession(
        store, store.get_run(plan.run.id), out, LineKeys(io.StringIO("k\n" * 18))
    ).run()
    assert rest.reviewed == 18 and rest.finished
    assert store.get_run(plan.run.id).status == "done"


async def test_a_question_that_gives_the_answer_away_is_asked_again(world: World) -> None:
    add_memory_facts(world)
    script = memory_script(leak_first=True)
    plan, summary, _ = await run_whole(world, script)
    assert summary.failed == 0
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(plan.run.id)
    assert {i.status for i in items} == {"generated"}
    assert all(leaked(i.payload["question"], i.payload["key_points"]) is None for i in items)
    assert len(script.requests) == 4 * TOTAL  # one more question call for each


async def test_a_question_that_still_gives_the_answer_away_is_a_lost_sample(world: World) -> None:
    add_memory_facts(world)

    def always_leaks(body: dict[str, object]) -> str:
        messages = body["messages"]
        assert isinstance(messages, list)
        if str(messages[0]["content"]).startswith(QUESTION_SYSTEM[:12]):
            return json.dumps({"question": "钥匙在哪", "key_points": ["钥匙"]}, ensure_ascii=False)
        return "好"

    plan, _, _ = await run_whole(world, DeepSeekScript(always_leaks))
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(plan.run.id)
    assert {i.status for i in items} == {"failed"}
    assert {i.payload["failure"] for i in items} == {"question_not_written"}
    assert summarize(items).verdict == "insufficient" and summarize(items).failed == TOTAL


async def test_an_answer_the_bot_did_not_give_is_wrong_whatever_the_judge_says(
    world: World,
) -> None:
    add_memory_facts(world)
    base = memory_script()

    def reply(body: dict[str, object]) -> str:
        messages = body["messages"]
        assert isinstance(messages, list)
        system = str(messages[0]["content"])
        if system.startswith(QUESTION_SYSTEM[:12]):
            return base.reply(body)
        if system.startswith(JUDGE_SYSTEM[:12]):  # a generous judge
            return json.dumps({"verdict": "correct", "reason": "很好"}, ensure_ascii=False)
        return "作为一个AI语言模型，我无法回答这个问题"  # a violation every time: no usable reply

    plan, _, _ = await run_whole(world, DeepSeekScript(reply))
    store = EvalStore(world.services.db, world.services.clock)
    items = store.items(plan.run.id)
    assert {i.status for i in items} == {"generated"}
    assert {i.auto_outcome for i in items} == {"wrong"}
    assert {i.payload["answered"] for i in items} == {False}
    assert {i.payload["judge_reason"] for i in items} == {"没有回答"}


async def test_finishing_a_run_stores_the_verdict_and_the_numbers(world: World) -> None:
    store = EvalStore(world.services.db, world.services.clock)
    run = store.create_run("memory", mode="live", backends=["deepseek"], status="running")
    store.add_items(
        run.id,
        [
            NewItem(
                f"f{n}",
                "deepseek",
                KNOWN,
                {"fact": f"事实{n}"},
                "real_record" if n < 10 else "user_said",
            )
            for n in range(TOTAL)
        ],
    )
    for found in store.items(run.id):
        store.save_auto(found.id, {"question": "问"}, auto_outcome="correct", cost_usd=0.0)
        store.judge(found.id, "correct" if found.seq < 17 else "wrong", score=1.0)
    finished = finish_run(store, run.id)
    assert finished.status == "done" and finished.verdict == "passed"
    assert (
        finished.summary["accuracy"] == pytest.approx(0.85) and finished.summary["passed"] is True
    )
    assert finished.finished_at is not None
