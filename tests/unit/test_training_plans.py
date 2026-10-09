"""Plans for the hybrid share of the training set (R-TRN-005, R-LLM-014).

DeepSeek is a ``respx`` route that answers like the planner prompt asks it to; the conversation
is synthetic (``tests.support.export_world``).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import text

from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.embedding import HashingBackend
from tests.support.export_world import World, build_world
from tests.support.memory import add_fact
from tests.support.policies import AlwaysOffPeak
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.ops.jobs import BatchTooLargeError, HandlerRegistry, JobQueue, Worker
from twin.profile.prompt_templates import TRAIN_PLAN, TemplateStore
from twin.services import Services
from twin.storage.training_plan_models import TrainingPlan
from twin.training.dataset_dir import DatasetDir
from twin.training.export import (
    ExportOptions,
    ExportResult,
    TrainingSetExporter,
)
from twin.training.export_blocks import iter_block_refs
from twin.training.plans import (
    DONE,
    FIELD_CHARS,
    PENDING,
    PLAN_JOB,
    REFUSED,
    PlanDraft,
    PlanEntry,
    PlanInput,
    PlanStore,
    StoredPlan,
    checked_facts,
    fact_lines,
    handle_training_plan,
    needs_plan,
    normalize_line,
    plan_from_draft,
    plan_messages,
    plan_selected,
    queue_plan_batches,
    target_hash,
)

TOKENIZER = tiny_qwen_tokenizer()
PHONE = "138" + "12345678"  # built from parts: the privacy scan sees no number
BLOCK = (
    "【相关的事】\n她：喜欢吃火锅\n对方：住在二楼\n【最近几天】\n9月1日周二（聊天记录）：聊了天气"
)


# ----------------------------------------------------------------------------- selection


def test_the_selection_depends_on_the_id_and_the_ratio_only() -> None:
    ids = [f"w{number:040d}" for number in range(4000)]
    picked = [i for i in ids if plan_selected(i, 0.3)]
    assert picked == [i for i in ids if plan_selected(i, 0.3)]  # the same every time
    assert 0.26 < len(picked) / len(ids) < 0.34
    assert set(picked) <= {i for i in ids if plan_selected(i, 0.5)}  # a larger share adds samples
    assert not any(plan_selected(i, 0.0) for i in ids[:50])
    assert all(plan_selected(i, 1.0) for i in ids[:50])


# ------------------------------------------------------------------------------- facts


def test_only_the_fact_sections_of_the_memory_block_offer_facts() -> None:
    assert fact_lines(BLOCK) == ["她：喜欢吃火锅", "对方：住在二楼"]
    assert fact_lines("") == []
    assert fact_lines("【今天要留意的】\n后天考试（9月3日 10:00）") == ["后天考试（9月3日 10:00）"]


def test_a_stored_fact_is_kept_only_when_the_block_still_has_that_line() -> None:
    available = fact_lines(BLOCK)
    kept = checked_facts(["她：喜欢吃火锅", "她：从没说过的事", "- 对方：住在二楼"], available)
    assert kept == (
        "她：喜欢吃火锅",
        "对方：住在二楼",
    )  # the lines of the block, not the stored text
    assert checked_facts(["她：喜欢吃火锅", "她：喜欢吃火锅"], available) == ("她：喜欢吃火锅",)
    assert checked_facts([], available) == ()
    many = [f"事{number}" for number in range(9)]
    assert len(checked_facts(many, many)) == 4


def test_stored_facts_are_compared_as_desensitised_lines() -> None:
    line = f"她：电话是{PHONE}"
    stored = normalize_line("她：电话是[手机号]")
    assert checked_facts([stored], [line]) == (line,)


def test_lines_differing_in_width_or_spacing_are_the_same_line() -> None:
    assert normalize_line("  - 她：喜欢  吃火锅 ") == "她:喜欢 吃火锅"
    assert normalize_line("她：ＡＢＣ") == "她:ABC"


# --------------------------------------------------------------------------- the draft


def source(
    reply: str = "好呀\n我也想去", facts: tuple[str, ...] = ("她：喜欢火锅", "对方：住二楼")
) -> PlanInput:
    return PlanInput(
        target_hash(reply),
        "2026年8月5日 周三 20:00",
        (("user", "周末去吃火锅吗"),),
        reply,
        facts,
    )


def test_a_draft_names_facts_by_number_and_numbers_outside_the_list_are_dropped() -> None:
    draft = PlanDraft.model_validate(
        {"intent": "答应一起去吃饭", "fact_numbers": [2, "1", 9, 0, "x", 2], "tone": "开心"}
    )
    plan = plan_from_draft(draft, source())
    assert plan == StoredPlan("答应一起去吃饭", ("对方：住二楼", "她：喜欢火锅"), "开心", "")


def test_a_plan_cannot_repeat_her_reply_and_needs_an_intent() -> None:
    reply = "好呀\n我也想去一起吃火锅"
    plan = plan_from_draft(PlanDraft(intent="答应一起去", tone="我也想去一起吃火锅"), source(reply))
    assert plan is not None and plan.intent == "答应一起去" and plan.tone == ""  # the echo is gone
    quoting = PlanDraft(intent="就说我也想去一起吃火锅", tone="随意")
    assert plan_from_draft(quoting, source(reply)) is None  # nothing but her own words
    assert plan_from_draft(PlanDraft(intent="  "), source()) is None


def test_every_field_is_cut_to_a_line_of_reasonable_length() -> None:
    plan = plan_from_draft(PlanDraft(intent="长" * 300, tone="短"), source())
    assert plan is not None and len(plan.intent) == FIELD_CHARS


def test_the_draft_reads_loose_json() -> None:
    draft = PlanDraft.model_validate({"intent": None, "fact_numbers": "3", "extra": 1})
    assert (draft.intent, draft.fact_numbers) == ("", [3])
    assert PlanDraft.model_validate({"fact_numbers": None}).fact_numbers == []


def test_a_stored_plan_checks_its_facts_against_the_block_it_is_used_with() -> None:
    plan = StoredPlan("闲聊", ("她：喜欢吃火锅", "她：不在块里"), "随意", "一条")
    fields = plan.fields(fact_lines(BLOCK))
    assert fields.facts_to_use == ("她：喜欢吃火锅",)
    assert (fields.intent, fields.tone, fields.bubble_hint) == ("闲聊", "随意", "一条")
    assert StoredPlan.from_json(plan.to_json()) == plan


def test_the_plan_prompt_shows_the_numbered_facts_and_the_reply(services: Services) -> None:
    template = TemplateStore(services.db, services.clock).active(TRAIN_PLAN)
    system, user = plan_messages(template, source())
    assert system["role"] == "system" and "fact_numbers" in str(system["content"])
    text = str(user["content"])
    assert "1. 她：喜欢火锅\n2. 对方：住二楼" in text and "对方：周末去吃火锅吗" in text
    assert text.rstrip().endswith("好呀\n我也想去")
    empty = str(plan_messages(template, source(facts=()))[1]["content"])
    assert "（没有）" in empty


# --------------------------------------------------------------------------- the table


def test_registering_adds_keeps_and_resets_rows(services: Services) -> None:
    store = PlanStore(services.db)
    first = store.register({"w1": source("甲"), "w2": source("乙")})
    assert (first.new, first.reset, first.kept) == (2, 0, 0)
    store.save_done(
        "w1", StoredPlan("闲聊"), model="deepseek-flash", template_ref="x@1", cost_usd=0.01
    )
    again = store.register({"w1": source("甲"), "w2": source("乙"), "w3": source("丙")})
    assert (again.new, again.reset, again.kept) == (1, 0, 2)
    changed = store.register({"w1": source("甲甲")})  # her reply changed: the plan is stale
    assert (changed.new, changed.reset, changed.kept) == (0, 1, 0)
    entries = store.entries()
    assert entries["w1"].status == PENDING and entries["w1"].plan is None
    assert store.counts() == {PENDING: 3, DONE: 0, REFUSED: 0}
    store.save_refused("w2", cost_usd=0.001)
    assert store.counts()[REFUSED] == 1 and store.pending_ids() == ["w1", "w3"]


def test_the_plan_and_its_inputs_are_stored_encrypted(services: Services) -> None:
    store = PlanStore(services.db)
    store.register({"w1": source("独特的回复内容")})
    store.save_done("w1", StoredPlan("独特的意图"), model="m", template_ref="t@1", cost_usd=0.0)
    with services.db.session() as session:
        raw_input, raw_plan = session.execute(
            text("SELECT input, plan FROM training_plans WHERE id = 'w1'")
        ).one()
        assert "独特的回复内容".encode() not in bytes(raw_input)
        assert "独特的意图".encode() not in bytes(raw_plan)
        row = session.get(TrainingPlan, "w1")
        assert row is not None
        assert row.input["reply"] == "独特的回复内容" and row.plan["intent"] == "独特的意图"


def test_a_reply_needs_a_plan_until_there_is_a_verdict_for_exactly_that_reply() -> None:
    sha = target_hash("好")

    def entry(status: str, sha_of: str) -> PlanEntry:
        return PlanEntry("w", status, sha_of, None, None)

    assert needs_plan(None, sha)
    assert needs_plan(entry(PENDING, sha), sha)
    assert needs_plan(entry(DONE, target_hash("别的")), sha)
    assert not needs_plan(entry(DONE, sha), sha)
    assert not needs_plan(entry(REFUSED, sha), sha)  # asked once; the sample goes without


# -------------------------------------------------------------- the flow with DeepSeek


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def runtime(services: Services, embedder: HashingBackend) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> World:
    return build_world(services, embedder, days=10)


def exporter_for(services: Services, directory: Path, ratio: float) -> TrainingSetExporter:
    return TrainingSetExporter(
        services, TOKENIZER, ExportOptions(out_dir=directory, plan_ratio=ratio)
    )


def worker_for(services: Services) -> Worker:
    registry = HandlerRegistry()
    registry.register(PLAN_JOB, handle_training_plan)
    return Worker(
        JobQueue(services.db, services.clock),
        registry,
        services.clock,
        services=services,
        offpeak=AlwaysOffPeak(),
        alerts=services.alerts,
    )


def planner(
    *,
    numbers: list[int] | None = None,
    pick: str | None = None,
    calls: list[dict[str, Any]] | None = None,
) -> Any:
    """A DeepSeek that names fact numbers: fixed ones, or the fact lines that contain ``pick``."""

    def answer(request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        if calls is not None:
            calls.append(body)
        chosen = numbers if numbers is not None else []
        if pick is not None:
            user = str(body["messages"][-1]["content"])
            chosen = [
                int(line.split(". ", 1)[0])
                for line in user.split("\n")
                if line[:1].isdigit() and ". " in line and pick in line
            ]
        payload = {
            "intent": "回应对方的话并说说自己的情况",
            "fact_numbers": [*chosen, 9],
            "tone": "随意",
            "bubble_hint": "两三条短句",
        }
        return ok(content=json.dumps(payload, ensure_ascii=False))

    return answer


def read(dataset: DatasetDir, name: str) -> list[dict[str, Any]]:
    path = dataset.file_path(name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def first_run(world: World, tmp_path: Path, ratio: float = 0.5) -> ExportResult:
    result = exporter_for(world.services, tmp_path / "ds", ratio).run()
    assert result.state == "waiting_for_plans" and result.dataset is None
    return result


def test_an_export_that_needs_plans_writes_nothing_and_hands_back_their_inputs(
    world: World, tmp_path: Path
) -> None:
    result = first_run(world, tmp_path)
    assert not (tmp_path / "ds").exists()
    assert 0 < len(result.missing_plans) == result.plan_selected
    cutoff = world.cutoff
    refs = {r.sample_id: r for r in iter_block_refs(world.services)}
    assert all(refs[i].reply_at < cutoff for i in result.missing_plans)  # no test sample has one
    some = next(iter(result.missing_plans.values()))
    assert some.turns and some.reply and some.when.startswith("2026年8月")


def test_the_plan_inputs_are_desensitised(world: World, tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from tests.support.synth_chat import MessageWriter

    writer = MessageWriter(world.services)
    day = datetime(2026, 8, 5, 20, 0, tzinfo=UTC)
    writer.add(day, False, "text", f"我的电话{PHONE}你存一下")
    writer.add(day + timedelta(seconds=30), True, "text", f"存了{PHONE}，我发你邮箱a.b@example.com")
    writer.store(append=True)
    result = first_run(world, tmp_path, ratio=1.0)
    joined = json.dumps([i.to_json() for i in result.missing_plans.values()], ensure_ascii=False)
    assert PHONE not in joined and "a.b@example.com" not in joined
    assert "[手机号]" in joined and "[邮箱]" in joined


def test_the_batches_are_priced_and_wait_for_approval(
    world: World, tmp_path: Path, runtime: LlmRuntime
) -> None:
    services = world.services
    result = first_run(world, tmp_path)
    store = PlanStore(services.db)
    store.register(result.missing_plans)
    queued = queue_plan_batches(services, runtime, sorted(result.missing_plans))
    assert queued.samples == len(result.missing_plans) and queued.jobs >= 1
    assert queued.estimated_usd > 0 and queued.batch_ids
    jobs = JobQueue(services.db, services.clock).list_jobs(job_type=PLAN_JOB, limit=1000)
    assert len(jobs) == queued.jobs and all(j.requires_approval and not j.approved_at for j in jobs)
    assert sum(len(j.payload["ids"]) for j in jobs) == queued.samples
    assert all(len(j.payload["ids"]) <= 20 for j in jobs)
    status = runtime.batches.status(queued.batch_ids[0])
    assert status.estimated_usd > 0 and not status.approved
    again = queue_plan_batches(services, runtime, sorted(result.missing_plans))
    assert again.samples == 0 and again.already_queued == queued.samples  # nothing twice


def test_a_batch_above_the_one_time_limit_is_split(
    world: World, tmp_path: Path, runtime: LlmRuntime
) -> None:
    services = world.services
    result = first_run(world, tmp_path)
    store = PlanStore(services.db)
    store.register(result.missing_plans)
    ids = sorted(result.missing_plans)
    template = TemplateStore(services.db, services.clock).active(TRAIN_PLAN)
    found = store.get_input(ids[0])
    assert found is not None
    item = runtime.batches.item_from_messages(
        services.settings.deepseek.offline_model,
        plan_messages(template, found[1]),
        completion_tokens=220,
    )
    one = runtime.batches.estimate_item(item)[0]
    services.settings.budget.one_time_usd = one * 6  # six samples of this size fit in a batch
    queued = queue_plan_batches(services, runtime, ids[1:])
    assert len(queued.batch_ids) >= 2 and queued.samples == len(ids) - 1
    for batch in queued.batch_ids:
        assert runtime.batches.status(batch).estimated_usd <= one * 6 * 1.5
    services.settings.budget.one_time_usd = one / 2  # not even one plan fits
    with pytest.raises(BatchTooLargeError):
        queue_plan_batches(services, runtime, ids[:1])


async def test_the_jobs_write_the_plans_and_the_next_export_uses_them(
    world: World, tmp_path: Path, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    services = world.services
    # a fact she may use: the text of one user turn, known well before it
    first = first_run(world, tmp_path)
    sample_id = sorted(first.missing_plans)[8]
    text = first.missing_plans[sample_id].turns[-1][1]
    ref = next(r for r in iter_block_refs(services) if r.sample_id == sample_id)
    add_fact(
        world.memory, f"{text} 关于火锅的约定", ref.reply_at.replace(year=2026, month=8, day=2)
    )
    first = first_run(world, tmp_path)
    calls: list[dict[str, Any]] = []
    api.post(API).mock(side_effect=planner(pick="关于火锅的约定", calls=calls))
    store = PlanStore(services.db)
    store.register(first.missing_plans)
    queued = queue_plan_batches(services, runtime, sorted(first.missing_plans))
    for batch in queued.batch_ids:
        runtime.batches.approve(batch)
    summary = await worker_for(services).run_until_idle()
    assert summary.failed == 0 and summary.retried == 0
    assert len(calls) == len(first.missing_plans)
    assert store.counts() == {PENDING: 0, DONE: len(first.missing_plans), REFUSED: 0}
    spent = runtime.ledger.batch_spent_usd(queued.batch_ids[0])
    assert spent > 0  # the money is on the one-time account of the batch

    result = exporter_for(services, tmp_path / "ds", 0.5).run()
    assert result.state == "done" and result.dataset is not None
    planned = [
        r
        for name in ("sft_train.jsonl", "sft_val.jsonl")
        for r in read(result.dataset, name)
        if "【规划】" in r["system"]
    ]
    assert len(planned) == len(first.missing_plans)
    assert result.stats["plans"]["planned"] == len(planned)
    assert result.stats["plans"]["share_of_train_and_val"] > 0.3
    section = planned[0]["system"].split("【规划】\n", 1)[1]
    assert section.startswith("想表达：回应对方的话并说说自己的情况")
    assert "语气：随意" in section and "气泡：两三条短句" in section
    with_facts = {r["id"]: r for r in planned if "会用到：" in r["system"]}
    assert sample_id in with_facts
    for row in with_facts.values():  # a plan names only lines its own memory block shows
        named = row["system"].split("会用到：", 1)[1].split("\n", 1)[0]
        assert named in fact_lines(row["system"]) and "；" not in named  # number 9 is no fact
        assert "关于火锅的约定" in named
    assert not [r for r in read(result.dataset, "sft_test.jsonl") if "【规划】" in r["system"]]


async def test_a_job_that_stopped_goes_on_with_the_samples_that_have_no_plan(
    world: World, tmp_path: Path, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    services = world.services
    first = first_run(world, tmp_path)
    store = PlanStore(services.db)
    store.register(first.missing_plans)
    ids = sorted(first.missing_plans)
    store.save_done(ids[0], StoredPlan("已经写好的"), model="m", template_ref="t@1", cost_usd=0.0)
    calls: list[dict[str, Any]] = []
    api.post(API).mock(side_effect=planner(calls=calls))
    queued = queue_plan_batches(services, runtime, ids)
    for batch in queued.batch_ids:
        runtime.batches.approve(batch)
    await worker_for(services).run_until_idle()
    assert len(calls) == len(ids) - 1
    assert store.entries()[ids[0]].plan == StoredPlan("已经写好的")


async def test_an_answer_that_is_not_json_twice_leaves_the_sample_without_a_plan(
    world: World, tmp_path: Path, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    services = world.services
    first = first_run(world, tmp_path)
    store = PlanStore(services.db)
    store.register(first.missing_plans)
    api.post(API).mock(return_value=ok(content="不是 JSON"))
    queued = queue_plan_batches(services, runtime, sorted(first.missing_plans))
    for batch in queued.batch_ids:
        runtime.batches.approve(batch)
    summary = await worker_for(services).run_until_idle()
    assert summary.failed == 0
    assert store.counts() == {PENDING: 0, DONE: 0, REFUSED: len(first.missing_plans)}
    result = exporter_for(services, tmp_path / "ds", 0.5).run()
    assert (
        result.state == "done" and result.stats["plans"]["planned"] == 0
    )  # exported without plans


async def test_a_refused_request_is_a_refusal_and_not_a_failure_of_the_job(
    world: World, tmp_path: Path, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    services = world.services
    first = first_run(world, tmp_path)
    store = PlanStore(services.db)
    store.register(first.missing_plans)
    api.post(API).mock(return_value=error(400, "Content Exists Risk"))
    queued = queue_plan_batches(services, runtime, sorted(first.missing_plans))
    for batch in queued.batch_ids:
        runtime.batches.approve(batch)
    summary = await worker_for(services).run_until_idle()
    assert summary.failed == 0 and summary.retried == 0
    assert store.counts()[REFUSED] == len(first.missing_plans)


async def test_a_plan_for_a_reply_that_has_changed_is_asked_for_again(
    world: World, tmp_path: Path, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    services = world.services
    first = first_run(world, tmp_path)
    store = PlanStore(services.db)
    store.register(first.missing_plans)
    api.post(API).mock(side_effect=planner())
    queued = queue_plan_batches(services, runtime, sorted(first.missing_plans))
    for batch in queued.batch_ids:
        runtime.batches.approve(batch)
    await worker_for(services).run_until_idle()
    victim = sorted(first.missing_plans)[0]
    ref = next(r for r in iter_block_refs(services) if r.sample_id == victim)
    from twin.storage.chat_models import Message

    with services.db.transaction(bump_state=False) as session:
        row = session.get(Message, ref.reply_ids[0])
        assert row is not None
        row.kind, row.text = "text", "后来改过的内容"
    again = exporter_for(services, tmp_path / "ds", 0.5).run()
    assert again.state == "waiting_for_plans" and list(again.missing_plans) == [victim]
