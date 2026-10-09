"""The persona jobs: one-time batch, approval, refresh, import hook, re-split
(R-LLM-014, R-ARCH-003, R-IMP-011, R-TRN-013)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import respx

from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.ingest import make_export, run_import
from tests.support.persona import sticker_scenario
from tests.support.policies import AlwaysOffPeak
from tests.support.synth_chat import append_texts
from twin.ingest.hooks import HookContext, default_hooks, load_hooks
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.ops.process_model import iter_commands
from twin.profile.builder import rebuild
from twin.profile.holdout import LISTENER_MODULES, holdout_cutoff, resplit_holdout
from twin.profile.persona.hook import persona_after_resplit, queue_persona
from twin.profile.persona.jobs import (
    PERSONA_JOB,
    REFRESH_JOB,
    estimate_items,
    handle_persona_generate,
    handle_persona_refresh,
    plan_generation,
    profile_rebuild_waiting,
    queue_refresh,
    seed_for,
)
from twin.profile.persona.store import PersonaStore
from twin.profile.queue import queue_profile_rebuild
from twin.services import Services
from twin.storage.models import Job

REPLY = json.dumps(
    {
        "tone": [{"text": "说话轻快", "evidence": ["S01", "S02"]}],
        "emotions": [{"emotion": "开心", "text": "哇塞", "evidence": ["S01"]}],
        "facts": [{"text": "养了一只猫", "evidence": ["S02"]}],
    },
    ensure_ascii=False,
)


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def library(services: Services) -> Services:
    sticker_scenario(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    return services


def queue_of(services: Services) -> JobQueue:
    return JobQueue(services.db, services.clock)


def jobs_of(services: Services, job_type: str) -> list[Any]:
    return queue_of(services).list_jobs(job_type=job_type, limit=100)


async def run_persona_jobs(services: Services) -> Any:
    registry = HandlerRegistry()
    registry.register(PERSONA_JOB, handle_persona_generate)
    registry.register(REFRESH_JOB, handle_persona_refresh)
    worker = Worker(
        queue_of(services),
        registry,
        services.clock,
        services=services,
        offpeak=AlwaysOffPeak(),
        alerts=services.alerts,
    )
    return await worker.run_until_idle()


# ---------------------------------------------------------------- planning


def test_planning_prices_both_scopes_into_one_batch_waiting_for_approval(library: Services) -> None:
    plan = plan_generation(library, ["live", "pre_holdout"], reason="manual")
    assert plan.batch_id and plan.batch_id.startswith("persona-")
    assert [item.scope for item in plan.scopes] == ["live", "pre_holdout"]
    assert 0 < plan.estimated_usd < 1.0
    jobs = jobs_of(library, PERSONA_JOB)
    assert len(jobs) == 2
    for job in jobs:
        assert job.requires_approval and job.approved_at is None and job.offpeak_only
        assert (
            job.batch_id == plan.batch_id and job.estimated_cost_usd and job.estimated_cost_usd > 0
        )
        assert job.payload["batch_id"] == plan.batch_id and job.payload["reason"] == "manual"
    assert len({job.payload["seed"] for job in jobs}) == 2
    assert queue_of(library).claim_next({PERSONA_JOB}, offpeak_allowed=True, worker_id="w") is None


def test_the_estimate_is_an_upper_bound_from_the_settings_alone(library: Services) -> None:
    runtime = build_llm_runtime(library)
    items = estimate_items(library, runtime)
    config = library.settings.persona
    assert len(items) == 6 + 1  # sixty segments ten at a time, then the merge
    assert items[0].prompt_tokens > config.batch_segments * 100
    assert items[-1].prompt_tokens >= 6 * 1500  # the merge reads what the map calls answered
    small = library.settings.persona.model_copy(update={"sample_segments": 10})
    library.settings.persona = small
    assert len(estimate_items(library, runtime)) == 1  # one batch needs no merge


def test_a_scope_that_is_already_waiting_is_not_queued_twice(library: Services) -> None:
    first = plan_generation(library, ["live"], reason="manual")
    second = plan_generation(library, ["live", "pre_holdout"], reason="manual")
    assert second.already_queued == ["live"] and [s.scope for s in second.scopes] == ["pre_holdout"]
    third = plan_generation(library, ["live", "pre_holdout"], reason="manual")
    assert third.batch_id is None and third.already_queued == ["live", "pre_holdout"]
    assert first.batch_id != second.batch_id and len(jobs_of(library, PERSONA_JOB)) == 2


def test_a_batch_above_the_one_time_limit_is_refused(library: Services) -> None:
    from twin.ops.jobs import BatchTooLargeError

    library.settings.budget.one_time_usd = 0.0001
    with pytest.raises(BatchTooLargeError):
        plan_generation(library, ["live"], reason="manual")
    assert jobs_of(library, PERSONA_JOB) == []


def test_seeds_depend_on_the_batch_and_the_scope() -> None:
    assert seed_for("persona-1", "live") == seed_for("persona-1", "live")
    assert seed_for("persona-1", "live") != seed_for("persona-1", "pre_holdout")
    assert seed_for("persona-1", "live") != seed_for("persona-2", "live")


# ----------------------------------------------------------------- running


async def test_nothing_runs_before_the_approval_and_everything_after(
    library: Services, api: respx.MockRouter
) -> None:
    rebuild(library, "all")
    route = api.post(API).mock(return_value=ok(content=REPLY, prompt=500, completion_tokens=90))
    plan = plan_generation(library, ["live", "pre_holdout"], reason="manual")
    assert plan.batch_id
    waiting = await run_persona_jobs(library)
    assert waiting.done == 0 and route.call_count == 0
    runtime = build_llm_runtime(library)
    runtime.batches.approve(plan.batch_id)
    done = await run_persona_jobs(library)
    assert done.done == 2 and done.failed == 0
    store = PersonaStore(library.db, library.clock)
    live, past = store.active("live"), store.active("pre_holdout")
    assert live is not None and past is not None
    assert live.reason == past.reason == "generate"
    assert "说话轻快" in live.text and "说话轻快" in past.text
    # one-time account, with the batch id; the daily budget is untouched (R-LLM-014)
    assert runtime.ledger.batch_spent_usd(plan.batch_id) > 0
    now = library.clock.now_utc()
    assert (
        runtime.ledger.total_usd(now - timedelta(days=1), now + timedelta(days=1), account="daily")
        == 0
    )
    body = request_json(route.calls[0].request)
    assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "disabled"}


async def test_a_failing_model_leaves_the_job_for_a_retry_and_writes_no_card(
    library: Services, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=error(400))  # not retried by the client
    plan = plan_generation(library, ["live"], reason="manual")
    assert plan.batch_id
    build_llm_runtime(library).batches.approve(plan.batch_id)
    summary = await run_persona_jobs(library)
    assert summary.done == 0
    assert PersonaStore(library.db, library.clock).active("live") is None
    assert jobs_of(library, PERSONA_JOB)[0].status in {"pending", "failed"}


@pytest.mark.parametrize("failure", ["circuit", "budget"])
async def test_an_open_circuit_or_a_denied_budget_hands_the_job_back(
    library: Services, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from twin.llm.deepseek import DeepSeekClient
    from twin.llm.errors import BudgetDeniedError, CircuitOpenError

    async def refuse(self: DeepSeekClient, *args: Any, **kwargs: Any) -> Any:
        raise CircuitOpenError(60.0) if failure == "circuit" else BudgetDeniedError("persona", 3)

    monkeypatch.setattr(DeepSeekClient, "chat_json", refuse)
    plan = plan_generation(library, ["live"], reason="manual")
    assert plan.batch_id
    build_llm_runtime(library).batches.approve(plan.batch_id)
    summary = await run_persona_jobs(library)
    assert summary.deferred == 1 and summary.failed == 0 and summary.retried == 0
    job = jobs_of(library, PERSONA_JOB)[0]
    assert job.status == "pending" and job.attempts == 0  # a hand-back costs no attempt
    assert PersonaStore(library.db, library.clock).active("live") is None


# ----------------------------------------------------------------- refreshing


async def test_the_refresh_waits_for_the_profile_of_the_same_import(library: Services) -> None:
    queue_profile_rebuild(library, scope="all", reason="import")
    assert profile_rebuild_waiting(library)
    queue_refresh(library, scope="all", reason="import")
    waiting = await run_persona_jobs(library)
    assert waiting.deferred == 1 and PersonaStore(library.db, library.clock).active("live") is None
    # once the profile is there the refresh writes the cards
    with library.db.transaction() as session:
        for job in session.query(Job).filter(Job.type == "profile_rebuild"):
            session.delete(job)
    rebuild(library, "all")
    library.clock.set_time(library.clock.now_utc() + timedelta(minutes=5))
    done = await run_persona_jobs(library)
    assert done.done == 1
    store = PersonaStore(library.db, library.clock)
    assert store.active("live") is not None and store.active("pre_holdout") is not None


def test_a_refresh_is_queued_once_per_scope(library: Services) -> None:
    first = queue_refresh(library, scope="all")
    assert queue_refresh(library, scope="all") == first
    assert queue_refresh(library, scope="live") != first
    assert len(jobs_of(library, REFRESH_JOB)) == 2


# --------------------------------------------------------------- the hook


def context(
    services: Services, *, inserted: int = 10, changed: int = 0, first: bool = False
) -> HookContext:
    return HookContext(services, "run", "conv", None, inserted, changed, first)


def test_the_hook_is_registered_with_its_backfill_command() -> None:
    registry = load_hooks()
    by_name = {hook.name: hook for hook in registry.hooks()}
    assert by_name["persona"].backfill_command == "persona regenerate"
    names = registry.names()
    assert names.index("profile") < names.index("persona") < names.index("sticker_tag")
    from twin.cli import app

    commands = {name for name, _ in iter_commands(app)}
    assert registry.missing_commands(commands) == []
    assert default_hooks is registry


def test_a_first_import_queues_the_free_refresh_and_a_priced_description(library: Services) -> None:
    result = queue_persona(context(library, first=True))
    assert result.status == "queued" and result.jobs == 3
    assert (
        "twin jobs approve persona-" in result.detail
        and "live: the description was never" in result.detail
    )
    assert len(jobs_of(library, REFRESH_JOB)) == 1 and len(jobs_of(library, PERSONA_JOB)) == 2


def test_an_import_without_news_does_nothing(library: Services) -> None:
    result = queue_persona(context(library, inserted=0))
    assert result.status == "skipped" and jobs_of(library, REFRESH_JOB) == []


def test_a_small_growth_only_refreshes_the_statistics(library: Services) -> None:
    from twin.profile.persona.refresh import count_her_messages, write_description

    holdout_cutoff(library)
    for scope in ("live", "pre_holdout"):
        write_description(
            library, scope, "### 风格\n- 语气：x\n\n", provenance={}, template_version="t@1",
            her_messages=count_her_messages(library, scope), at=library.clock.now_utc(),
        )  # fmt: skip
    result = queue_persona(context(library))
    assert result.status == "queued" and result.jobs == 1 and "refreshed" in result.detail
    assert jobs_of(library, PERSONA_JOB) == []
    # but a hand-made growth beyond ten percent also queues a description of the live scope
    last = library.clock.now_utc()
    append_texts(library, [(last - timedelta(minutes=i + 1), True, "新") for i in range(30)])
    again = queue_persona(context(library))
    assert again.jobs == 2 and "live: her messages grew" in again.detail
    assert [j.payload["scope"] for j in jobs_of(library, PERSONA_JOB)] == ["live"]
    third = queue_persona(context(library))
    assert "already waiting" in third.detail


def test_after_a_resplit_the_past_card_is_refreshed_and_described_again(library: Services) -> None:
    holdout_cutoff(library)
    append_texts(
        library,
        [(library.clock.now_utc() - timedelta(days=1, minutes=i), True, "后来") for i in range(60)],
    )
    assert "twin.profile.persona.hook" in LISTENER_MODULES
    outcome = resplit_holdout(library)
    notes = " | ".join(outcome.notes)
    assert "persona: pre-holdout statistics refresh queued" in notes
    assert "twin jobs approve persona-" in notes
    assert [j.payload["scope"] for j in jobs_of(library, PERSONA_JOB)] == ["pre_holdout"]
    assert jobs_of(library, REFRESH_JOB)[0].payload["scope"] == "pre_holdout"
    again = persona_after_resplit(library, None, outcome.current)
    assert "already waiting" in again


async def test_a_real_import_queues_the_persona_work(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=300)
    outcome = run_import(services, export)
    result = outcome.run.hooks["persona"]
    assert result["status"] == "queued" and result["backfill_command"] == "persona regenerate"
    assert len(jobs_of(services, REFRESH_JOB)) == 1
    assert len(jobs_of(services, PERSONA_JOB)) == 2


async def test_the_whole_way_from_an_import_to_the_two_renderings(
    services: Services, tmp_path: Path, api: respx.MockRouter
) -> None:
    import re

    from twin.profile.persona.api import render_compact, render_full

    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    run_import(services, make_export(tmp_path, target_messages=400))
    with services.db.transaction() as session:  # the profile job of the import: do it here
        for job in session.query(Job).filter(Job.type == "profile_rebuild"):
            session.delete(job)
    rebuild(services, "all")

    def answer(request: Any) -> Any:
        prompt = request_json(request)["messages"][1]["content"]
        first = (re.findall(r"【片段 (S\d+)】", prompt) or ["S01"])[0]
        if "份归纳结果" in prompt:
            return ok(
                content=json.dumps(
                    {
                        "tone": [{"text": "说话轻快", "evidence": ["S01"]}],
                        "facts": [{"text": "养了一只猫", "evidence": ["S01"]}],
                    },
                    ensure_ascii=False,
                )
            )
        return ok(
            content=json.dumps(
                {
                    "tone": [{"text": "说话轻快", "evidence": [first]}],
                    "facts": [{"text": "养了一只猫", "evidence": [first]}],
                },
                ensure_ascii=False,
            )
        )

    api.post(API).mock(side_effect=answer)
    jobs = jobs_of(services, PERSONA_JOB)
    assert len(jobs) == 2 and jobs[0].batch_id == jobs[1].batch_id
    build_llm_runtime(services).batches.approve(str(jobs[0].batch_id))
    library_summary = await run_persona_jobs(
        services
    )  # the statistics refresh and both descriptions
    assert library_summary.failed == 0 and library_summary.done == 3
    for scope in ("live", "pre_holdout"):
        full = render_full(services, scope)
        compact = render_compact(services, scope)
        assert full is not None and compact is not None
        assert "养了一只猫" in full.text and "说话轻快" in full.text
        assert "养了一只猫" not in compact.text and "说话轻快" in compact.text
        assert full.tokens <= 1500 and compact.tokens <= 400 and compact.dropped == 0
    store = PersonaStore(services.db, services.clock)
    live = store.active("live")
    assert live is not None and live.described_her_messages and live.described_her_messages >= 30
