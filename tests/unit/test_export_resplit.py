"""R18-C3 ② (user decision, D-881..D-884): ``twin train export`` re-splits the hold-out itself.

The user's words: "twin train export 时自动重切（测试集 = 当时最近 10%），pre_holdout 画像/作息/
人设卡/表情包上下文标签随之重建，人设卡重生成费用并入该次导出的批任务估价。" and "重切不作废已存的
门槛结果".  So the export job

* moves the stored cut-off to the newest ``retrieval.holdout_ratio`` of her reply blocks when the
  data has moved past it (an import still never moves it: ``test_holdout.py``);
* reads nothing until the pre-holdout profile, routine and card statistics are rebuilt for the new
  cut-off (it hands itself back while those jobs wait; iron rule 13);
* plans the new pre-holdout description inside its own one-time batch, beside the plans of the
  samples, so one approval pays for both;
* leaves every stored evaluation and gate verdict alone.

The conversation is synthetic; no DeepSeek call is made (estimates only).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.export_world import World, build_world
from tests.support.memory import mark_days_replayed
from tests.support.synth_chat import append_texts
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.eval.gates import check_gate
from twin.eval.store import EvalStore
from twin.memory.asof import AsOfSource
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.profile.holdout import compute_holdout, get_holdout, holdout_share
from twin.profile.jobs import handle_profile_rebuild
from twin.profile.persona.jobs import PERSONA_JOB, REFRESH_JOB, handle_persona_refresh
from twin.profile.persona.refresh import count_her_messages, write_description
from twin.profile.pre_holdout import pre_holdout_state
from twin.profile.queue import PROFILE_JOB
from twin.services import Services
from twin.training import export as export_module
from twin.training import export_job
from twin.training.dataset_dir import load_dataset_dir
from twin.training.dpo_export import dataset_cutoff
from twin.training.export_job import (
    EXPORT_JOB,
    REBUILD_WAIT_S,
    ExportRequest,
    handle_training_export,
    queue_export,
    read_state,
)
from twin.training.plans import PLAN_JOB

DESCRIPTION = "### 风格\n- 句子短，常用语气词\n\n"


@pytest.fixture
def world(services: Services, embedder: HashingBackend, monkeypatch: pytest.MonkeyPatch) -> World:
    """The export world, a description made for its cut-off, then two more weeks of chat."""
    monkeypatch.setattr(
        export_job,
        "ensure_tokenizer",
        lambda cache, explicit=None, fetch=None: tiny_qwen_tokenizer(),
    )
    built = build_world(services, embedder, days=12)
    write_description(
        services,
        "pre_holdout",
        DESCRIPTION,
        provenance={"cutoff": built.cutoff.isoformat(), "scope": "pre_holdout"},
        template_version="t@1",
        her_messages=count_her_messages(services, "pre_holdout"),
        at=services.clock.now_utc(),
    )
    rows = []
    moment = datetime(2026, 8, 16, 9, 0, tzinfo=UTC)
    for _ in range(14 * 5):  # five exchanges a day for two weeks after the world
        rows.append((moment, False, "在干嘛呢"))
        rows.append((moment + timedelta(seconds=40), True, "刚下班在路上"))
        moment += timedelta(hours=4, minutes=48)
    append_texts(services, rows)
    mark_days_replayed(services)  # the export checks the replay of its range (D-817)
    return built


def worker(services: Services, *types: str) -> Worker:
    """A worker for these job types, one job at a time (the order of the queue decides)."""
    handlers = {
        EXPORT_JOB: handle_training_export,
        PROFILE_JOB: handle_profile_rebuild,
        REFRESH_JOB: handle_persona_refresh,
    }
    registry = HandlerRegistry()
    for job_type in types:
        registry.register(job_type, handlers[job_type])
    return Worker(
        JobQueue(services.db, services.clock),
        registry,
        services.clock,
        services=services,
        alerts=services.alerts,
        concurrency=1,
    )


async def settle(services: Services, clock: ManualClock, *types: str) -> None:
    """Run these job types until none of them is left; a job handed back is waited for."""
    for _ in range(10):
        await worker(services, *types).run_until_idle()
        queue = JobQueue(services.db, services.clock)
        if not [job for job in queue.list_jobs(status="pending", limit=500) if job.type in types]:
            return
        clock.tick(REBUILD_WAIT_S + 1)  # the export looks again after this
    raise AssertionError("the jobs did not settle")


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """The cut-off of the pre-holdout profile every exporter was built with."""
    seen: list[str | None] = []

    class Spy(AsOfSource):
        def __init__(self, services: Services, **kwargs: Any) -> None:
            super().__init__(services, **kwargs)
            profile = self.profile
            seen.append(profile.version.data_range.get("cutoff") if profile else None)

    monkeypatch.setattr(export_module, "AsOfSource", Spy)
    return seen


async def test_r18_c3_the_export_moves_the_cutoff_to_the_newest_tenth_first(
    world: World, tmp_path: Path, reads: list[str | None]
) -> None:
    """R18-C3 ② (D-881): the export job re-splits when the data has moved past the cut-off; it
    reads nothing while the pre-holdout rebuild it caused is still queued."""
    services = world.services
    old = world.cutoff
    share = holdout_share(services)
    assert share is not None and share.drifted  # two weeks of chat sit after the old cut-off
    queue_export(services, ExportRequest(out_dir=tmp_path / "ds"))
    summary = await worker(services, EXPORT_JOB).run_until_idle()
    assert summary.deferred == 1 and summary.failed == 0
    stored = get_holdout(services)
    assert stored is not None and stored.cutoff > old
    assert stored.cutoff == compute_holdout(services).cutoff  # the newest 10 % now
    now = holdout_share(services)
    assert now is not None and now.share <= services.settings.retrieval.holdout_ratio + 0.01
    state = read_state(services)
    assert state is not None and state["state"] == "rebuilding"
    assert state["cutoff"] == stored.cutoff.isoformat()
    assert state["previous_cutoff"] == old.isoformat()
    assert PROFILE_JOB in state["waiting_for"]
    assert reads == []  # nothing was read under the old cut-off's profile
    assert not (tmp_path / "ds").exists()


async def test_r18_c3_the_export_reads_only_the_profile_of_the_new_cutoff(
    world: World, tmp_path: Path, clock: ManualClock, reads: list[str | None]
) -> None:
    """R18-C3 ② (D-882, iron rule 13): once the rebuild jobs ran, the export goes on by itself in
    the same run; every exporter it builds reads the pre-holdout profile of the new cut-off."""
    services = world.services
    queue_export(services, ExportRequest(out_dir=tmp_path / "ds"))
    await settle(services, clock, EXPORT_JOB, PROFILE_JOB, REFRESH_JOB)
    stored = get_holdout(services)
    assert stored is not None and stored.cutoff > world.cutoff
    assert reads and set(reads) == {stored.cutoff.isoformat()}
    state = read_state(services)
    assert state is not None and state["state"] == "waiting_for_plans"


async def test_r18_c3_one_batch_pays_for_the_plans_and_the_new_description(
    world: World, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-C3 ② (D-883): the description of the old cut-off is planned again inside the export's
    own batch; its estimate is part of the export's, and one approval releases both."""
    services = world.services
    queue_export(services, ExportRequest(out_dir=tmp_path / "ds"))
    await settle(services, clock, EXPORT_JOB, PROFILE_JOB, REFRESH_JOB)
    state = read_state(services)
    assert state is not None and state["state"] == "waiting_for_plans"
    persona = state["persona"]
    assert persona["batch_id"] in state["batches"] and persona["estimated_usd"] > 0
    queue = JobQueue(services.db, services.clock)
    (description,) = queue.list_jobs(status="pending", job_type=PERSONA_JOB)
    assert description.payload["scope"] == "pre_holdout"
    assert description.batch_id == persona["batch_id"] and description.requires_approval
    plans = queue.list_jobs(status="pending", job_type=PLAN_JOB, limit=500)
    assert plans and {job.batch_id for job in plans} <= set(state["batches"])
    assert any(job.batch_id == persona["batch_id"] for job in plans)  # the same batch
    estimates = sum(job.estimated_cost_usd or 0.0 for job in [description, *plans])
    assert state["estimated_usd"] == pytest.approx(round(estimates, 4), abs=1e-4)
    approval = queue.approve_batch(persona["batch_id"], max_usd=1000.0)
    jobs = queue.list_jobs(batch_id=persona["batch_id"], limit=500)
    assert approval.job_count == len(jobs) and all(job.approved_at for job in jobs)
    assert {job.type for job in jobs} == {PERSONA_JOB, PLAN_JOB}


async def test_r18_c3_a_resplit_by_the_export_keeps_every_stored_verdict(
    world: World, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-C3 ③ (D-884): the re-split does not void a stored gate verdict or any evaluation."""
    services = world.services
    store = EvalStore(services.db, services.clock)
    store.create_run(
        "gate",
        status="done",
        milestone="M1",
        verdict="passed",
        params={"milestone": "M1"},
        summary={"summary": "M1 passed before the re-split", "checks": [], "runs": []},
    )
    runs = len(store.list_runs("gate"))
    queue_export(services, ExportRequest(out_dir=tmp_path / "ds"))
    await settle(services, clock, EXPORT_JOB, PROFILE_JOB, REFRESH_JOB)
    stored = get_holdout(services)
    assert stored is not None and stored.cutoff > world.cutoff
    outcome = check_gate(services, "M1")
    assert outcome.status == "passed" and outcome.message == "M1 passed before the re-split"
    assert len(store.list_runs("gate")) == runs


async def test_r18_c3_the_dpo_export_reads_the_cutoff_the_export_moved_to(
    world: World, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-C3 ② with the DPO decision (D-860, D-881): the dataset of an export that re-split
    records the new cut-off, and the DPO export built on it reads exactly that one."""
    from tests.support.export_plans import fill_plans
    from twin.training.plans import PlanStore

    services = world.services
    queue_export(services, ExportRequest(out_dir=tmp_path / "ds"))
    await settle(services, clock, EXPORT_JOB, PROFILE_JOB, REFRESH_JOB)
    stored = get_holdout(services)
    assert stored is not None and stored.cutoff > world.cutoff
    # what the approved batch does: the plans and the description of the new cut-off
    store = PlanStore(services.db)
    pending = {i: store.get_input(i) for i in store.pending_ids()}
    fill_plans(services, {i: found[1] for i, found in pending.items() if found is not None})
    write_description(
        services,
        "pre_holdout",
        DESCRIPTION,
        provenance={"cutoff": stored.cutoff.isoformat(), "scope": "pre_holdout"},
        template_version="t@1",
        her_messages=count_her_messages(services, "pre_holdout"),
        at=services.clock.now_utc(),
    )
    queue_export(services, ExportRequest(out_dir=tmp_path / "ds"))
    await settle(services, clock, EXPORT_JOB)
    state = read_state(services)
    assert state is not None and state["state"] == "done", state
    dataset = load_dataset_dir(Path(state["directory"]))
    assert dataset.meta.holdout_cutoff == stored.cutoff.isoformat()
    assert dataset_cutoff(dataset) == stored.cutoff


# ------------------------------------------------------------------ the guards and the edges


def drop_waiting_jobs(services: Services) -> None:
    """What a failed rebuild leaves behind: nothing waits any more."""
    queue = JobQueue(services.db, services.clock)
    for job in queue.list_jobs(status="pending", limit=500):
        if job.type in (PROFILE_JOB, REFRESH_JOB):
            queue.cancel(job.id)


def test_r18_c3_the_exporter_never_starts_on_the_profile_of_another_cutoff(world: World) -> None:
    """R18-C3 ② (D-882, iron rule 13): built directly, the exporter refuses a pre-holdout profile
    of another cut-off - naming the queued rebuild while it waits, the command when none does."""
    from twin.profile.holdout import resplit_holdout
    from twin.training.export import ExportError, TrainingSetExporter

    services = world.services
    resplit_holdout(services, by="test")
    state = pre_holdout_state(services)
    assert not state.profile_current and PROFILE_JOB in state.waiting and state.stale() == []
    with pytest.raises(ExportError, match="still queued: profile_rebuild"):
        TrainingSetExporter(services, tiny_qwen_tokenizer())
    drop_waiting_jobs(services)
    assert pre_holdout_state(services).waiting == ()
    with pytest.raises(ExportError, match="twin profile rebuild"):
        TrainingSetExporter(services, tiny_qwen_tokenizer())


def test_r18_c3_card_statistics_of_another_cutoff_name_their_command(world: World) -> None:
    """R18-C3 ② (D-882): the card's statistics rules come from a profile; when that profile is of
    another cut-off and nothing refreshes them, the export names the free refresh."""
    from twin.profile.builder import rebuild
    from twin.profile.holdout import resplit_holdout
    from twin.profile.persona.refresh import refresh_stats
    from twin.training.export import ExportError, TrainingSetExporter

    services = world.services
    refresh_stats(services, "pre_holdout", force=True)  # the card names the profile of the old one
    resplit_holdout(services, by="test")
    rebuild(services, "pre_holdout", reason="resplit", force=True)
    drop_waiting_jobs(services)
    state = pre_holdout_state(services)
    assert state.profile_current and not state.card_current and not state.card_by_hand
    assert state.card_cutoff == world.cutoff
    (problem,) = state.stale()
    assert "--stats-only" in problem
    with pytest.raises(ExportError, match="--stats-only"):
        TrainingSetExporter(services, tiny_qwen_tokenizer())


def test_r18_c3_a_stale_description_is_waited_for_or_recorded_when_allowed(
    world: World, tmp_path: Path
) -> None:
    """R18-C3 ② (D-882): a description of another cut-off does not stop the first pass; ``build``
    refuses it, ``--allow-incomplete`` writes the set and records it."""
    from tests.support.export_plans import fill_plans
    from twin.profile.builder import rebuild
    from twin.profile.holdout import resplit_holdout
    from twin.profile.persona.refresh import refresh_all
    from twin.training.export import ExportError, ExportOptions, TrainingSetExporter

    services = world.services
    resplit_holdout(services, by="test")
    rebuild(services, "pre_holdout", reason="resplit", force=True)
    refresh_all(services, "pre_holdout")
    drop_waiting_jobs(services)
    state = pre_holdout_state(services)
    assert state.description == "stale" and state.description_cutoff == world.cutoff
    exporter = TrainingSetExporter(
        services, tiny_qwen_tokenizer(), ExportOptions(out_dir=tmp_path / "a", plan_ratio=0.0)
    )
    assert exporter.description_pending
    first = exporter.run()
    assert first.state == "waiting_for_plans" and first.describe
    fill_plans(services, first.missing_plans)
    with pytest.raises(ExportError, match="earlier hold-out cut-off"):
        exporter.build()
    allowed = TrainingSetExporter(
        services,
        tiny_qwen_tokenizer(),
        ExportOptions(out_dir=tmp_path / "b", plan_ratio=0.0, allow_incomplete=True),
    )
    result = allowed.run()
    assert result.dataset is not None
    gaps = result.dataset.meta.stats["derived_data"]
    assert gaps["persona_description_current"] is False and gaps["complete"] is False


async def test_r18_c3_a_description_planned_before_is_listed_not_planned_twice(
    world: World, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-C3 ② (D-883): after ``twin retrieval resplit`` planned the description as its own
    batch, the export lists that batch instead of planning a second one."""
    from twin.profile.holdout import resplit_holdout
    from twin.profile.persona.hook import plan_description_after_resplit

    services = world.services
    resplit_holdout(services, by="user")
    assert "twin jobs approve persona-" in plan_description_after_resplit(services)
    (earlier,) = JobQueue(services.db, services.clock).list_jobs(job_type=PERSONA_JOB)
    queue_export(services, ExportRequest(out_dir=tmp_path / "ds"))
    await settle(services, clock, EXPORT_JOB, PROFILE_JOB, REFRESH_JOB)
    state = read_state(services)
    assert state is not None and state["state"] == "waiting_for_plans"
    assert state["persona"]["batch_id"] is None
    assert state["persona"]["waiting"] == [earlier.batch_id]
    assert earlier.batch_id in state["batches"]
    assert len(JobQueue(services.db, services.clock).list_jobs(job_type=PERSONA_JOB)) == 1


def test_r18_c3_the_rebuild_jobs_that_are_waited_for(world: World) -> None:
    """R18-C3 ② (D-882): which queued jobs count as the pre-holdout rebuild - not the live scope,
    not a job still waiting for an approval, a sticker job only with context corrections."""
    from twin.stickers.tag_jobs import TAG_JOB

    services = world.services
    queue = JobQueue(services.db, services.clock)
    queue.enqueue(PROFILE_JOB, {"scope": "live"})
    queue.enqueue(TAG_JOB, {"vision": ["a" * 32], "context": []})
    queue.enqueue(
        TAG_JOB, {"vision": [], "context": ["b" * 32]}, batch_id="tag-1", requires_approval=True
    )
    assert pre_holdout_state(services).waiting == ()
    queue.enqueue(TAG_JOB, {"vision": [], "context": ["c" * 32]})
    queue.enqueue(REFRESH_JOB, {"scope": "all"})
    assert pre_holdout_state(services).waiting == (REFRESH_JOB, "sticker_tag")


def test_r18_c3_the_resplit_of_an_export_needs_a_stored_cutoff_only_once(
    services: Services,
) -> None:
    """R18-C3 ② (D-881): before any cut-off was stored, the export stores the first one (no
    re-split); the same data again moves nothing."""
    from tests.support.synth_chat import ChatSpec, build_chat
    from twin.profile.holdout import resplit_if_behind

    build_chat(services, ChatSpec(days=8))
    assert get_holdout(services) is None
    assert resplit_if_behind(services, by="test") is None
    stored = get_holdout(services)
    assert stored is not None and stored.cutoff == compute_holdout(services).cutoff
    assert resplit_if_behind(services, by="test") is None
    assert get_holdout(services) == stored


def test_r18_c3_the_status_lines_of_a_waiting_export() -> None:
    """R18-C3 ② (D-881, D-883): what ``export`` and ``export-status`` print while it waits."""
    from twin.training.cli import format_state

    rebuilding = format_state(
        {
            "state": "rebuilding",
            "at": "t",
            "cutoff": "2026-08-20T00:00:00+00:00",
            "previous_cutoff": "2026-08-14T00:00:00+00:00",
            "waiting_for": [PROFILE_JOB, REFRESH_JOB],
        }
    )
    text = "\n".join(rebuilding)
    assert "moved from 2026-08-14" in text and "profile_rebuild, persona_refresh" in text
    waiting = "\n".join(
        format_state(
            {
                "state": "waiting_for_plans",
                "at": "t",
                "missing": 3,
                "selected": 9,
                "batches": ["trainplan-1"],
                "estimated_usd": 0.12,
                "persona": {
                    "cutoff": "2026-08-20",
                    "batch_id": "trainplan-1",
                    "estimated_usd": 0.07,
                },
            }
        )
    )
    assert "persona description is generated again for the cut-off 2026-08-20" in waiting
    assert "$0.07 of the estimate" in waiting and "twin jobs approve trainplan-1" in waiting


def test_r18_c3_the_description_has_a_batch_of_its_own_when_it_cannot_join(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-C3 ② (D-883): with no plan to join, or no room left in the first batch of the plans,
    the new description is a batch of its own - still listed with the export's batches."""
    from twin.profile.persona import jobs as persona_jobs
    from twin.training.export import TrainingSetExporter
    from twin.training.export_job import _queue_batches
    from twin.training.plans import PlanStore

    services = world.services
    alone = _queue_batches(services, [], True, ())
    assert alone.plans.batch_ids == [] and alone.persona is not None
    assert alone.persona.batch_id is not None and alone.batch_ids == [alone.persona.batch_id]
    JobQueue(services.db, services.clock).cancel(alone.persona.scopes[0].job_id)
    missing, _ = TrainingSetExporter(services, tiny_qwen_tokenizer()).scan()
    PlanStore(services.db).register(missing)
    limit = services.settings.budget.one_time_usd
    big = limit - 0.001  # the description fills a batch by itself: no plan fits beside it
    monkeypatch.setattr(persona_jobs, "description_estimate", lambda s, r: big)
    monkeypatch.setattr(export_job, "description_estimate", lambda s, r: big)
    queued = _queue_batches(services, sorted(missing), True, ())
    assert queued.plans.batch_ids and queued.persona is not None
    assert queued.persona.batch_id not in queued.plans.batch_ids
    assert queued.batch_ids == [*queued.plans.batch_ids, queued.persona.batch_id]
    assert queued.estimated_usd == pytest.approx(queued.plans.estimated_usd + big)


def test_r18_c3_a_later_look_at_the_same_resplit_keeps_the_cutoff_it_came_from(
    world: World,
) -> None:
    """R18-C3 ② (D-882): the ``rebuilding`` record keeps the old cut-off across the looks of the
    same re-split, and drops it for another one."""
    from twin.training.export_job import SplitPreparation, rebuilding_state

    state = pre_holdout_state(world.services)
    prepared = SplitPreparation(None, state)
    earlier = {"state": "rebuilding", "cutoff": state.cutoff.isoformat(), "previous_cutoff": "x"}
    assert rebuilding_state(prepared, earlier)["previous_cutoff"] == "x"
    other = {**earlier, "cutoff": "2020-01-01T00:00:00+00:00"}
    assert rebuilding_state(prepared, other)["previous_cutoff"] is None
    assert rebuilding_state(prepared, None)["previous_cutoff"] is None


def test_r18_c3_an_export_without_enough_data_fails_on_the_split(services: Services) -> None:
    """R18-C3 ② (D-881): with too little data the split cannot be made - an export error."""
    from twin.training.export import ExportError
    from twin.training.export_job import prepare_split

    with pytest.raises(ExportError, match="hold-out cannot be determined"):
        prepare_split(services)


async def test_r18_c3_a_job_that_cannot_split_fails_with_the_reason(services: Services) -> None:
    """R18-C3 ② (D-881): the export job on too little data ends as failed, not deferred."""
    queue_export(services, ExportRequest())
    summary = await worker(services, EXPORT_JOB).run_until_idle()
    assert summary.failed == 1 and summary.deferred == 0
    state = read_state(services)
    assert state is not None and state["state"] == "failed"
    assert "hold-out cannot be determined" in state["message"]


def test_r18_c3_no_room_for_the_joining_job_is_refused_before_anything_is_queued(
    world: World,
) -> None:
    """R18-C3 ② (D-883): a reserve above the one-time limit cannot be kept for anybody."""
    from twin.llm.runtime import build_llm_runtime
    from twin.ops.jobs import BatchTooLargeError
    from twin.training.export import TrainingSetExporter
    from twin.training.plans import PlanStore, queue_plan_batches

    services = world.services
    missing, _ = TrainingSetExporter(services, tiny_qwen_tokenizer()).scan()
    PlanStore(services.db).register(missing)
    runtime = build_llm_runtime(services)
    limit = services.settings.budget.one_time_usd
    with pytest.raises(BatchTooLargeError):
        queue_plan_batches(services, runtime, sorted(missing), reserve_usd=limit + 1.0)
    assert JobQueue(services.db, services.clock).list_jobs(job_type=PLAN_JOB) == []


def test_r18_c3_a_record_without_a_readable_cutoff_names_none() -> None:
    """R18-C3 ② (D-882): a version record without a cut-off, or with garbage, names none."""
    from twin.profile.pre_holdout import cutoff_of

    assert cutoff_of(None) is None and cutoff_of({}) is None
    assert cutoff_of({"cutoff": "not a moment"}) is None
    assert cutoff_of({"cutoff": "2026-08-14T02:46:19"}) is None  # a naive moment is not one
    moment = cutoff_of({"cutoff": "2026-08-14T02:46:19+00:00"})
    assert moment is not None and moment.isoformat() == "2026-08-14T02:46:19+00:00"
