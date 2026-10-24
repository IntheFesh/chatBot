"""``twin train export``, ``export-status``, ``retrain-check`` (R-TRN-002, R-TRN-012, R-ARCH-006).

The commands run through the real CLI against a database in the test's data directory; the
conversation is synthetic, DeepSeek is a ``respx`` route, and the tokenizer is the small test one
(the real file is pinned by hash and is checked in ``test_training_tokenizer.py``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import respx
from sqlalchemy import select
from typer.testing import CliRunner

from tests.support.deepseek import API, TEST_KEY, ok
from tests.support.embedding import HashingBackend
from tests.support.export_plans import fill_missing_plans
from tests.support.export_world import build_world
from tests.support.policies import AlwaysOffPeak
from tests.support.synth_chat import MessageWriter
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from tests.support.training_history import record_training
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.services import Services, build_services
from twin.storage.models import Alert
from twin.storage.training_models import DatasetVersion
from twin.training import cli as training_cli
from twin.training import export_job
from twin.training.export_job import EXPORT_JOB, handle_training_export, read_state
from twin.training.plans import PLAN_JOB, handle_training_plan

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv("TWIN_RETRIEVAL__MODEL", "test/hashing-bigram")
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        export_job,
        "ensure_tokenizer",
        lambda cache, explicit=None, fetch=None: tiny_qwen_tokenizer(),
    )


def open_services() -> Services:
    settings = load_settings()
    return build_services(settings, root=resolve_paths(settings).root)


@pytest.fixture
def world_in_cli_database(data_dir: Path, embedder: HashingBackend, tokenizer: None) -> None:
    services = open_services()
    try:
        build_world(services, embedder, days=8)
    finally:
        services.close()


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def with_services[T](work: Any) -> T:
    services = open_services()
    try:
        return work(services)  # type: ignore[no-any-return]
    finally:
        services.close()


# ----------------------------------------------------------------------------- export


def test_the_command_queues_a_job_and_the_foreground_runs_it(
    world_in_cli_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_TRAINING__HYBRID_PLAN_RATIO", "0")
    # ratio 0 asks for no plans of the ordinary samples; the proactive ones still have theirs
    assert with_services(lambda s: fill_missing_plans(s, tiny_qwen_tokenizer())) > 0
    queued = invoke("train", "export", "--out", str(tmp_path / "ds"))
    assert queued.exit_code == 0, queued.output
    assert "queued (job " in queued.output and "twin train export-status" in queued.output
    assert not (tmp_path / "ds").exists()  # nothing ran: the application would do it
    again = invoke("train", "export", "--out", str(tmp_path / "ds"), "--foreground")
    assert again.exit_code == 0, again.output
    assert "an export is already waiting" in again.output  # not queued twice, but run now
    assert "export: done" in again.output
    assert "next: twin train bundle --profile" in again.output
    assert (tmp_path / "ds" / "dataset_meta.json").is_file()

    def rows(services: Services) -> list[tuple[str, int, int, int, dict[str, Any]]]:
        with services.db.session() as session:
            return [
                (r.id, r.train_count, r.val_count, r.test_count, dict(r.stats))
                for r in session.scalars(select(DatasetVersion))
            ]

    ((version, train, val, test, stats),) = with_services(rows)
    meta = json.loads((tmp_path / "ds" / "dataset_meta.json").read_text(encoding="utf-8"))
    assert version == meta["dataset_version"] and (train, val, test) == (
        meta["counts"]["train"],
        meta["counts"]["val"],
        meta["counts"]["test"],
    )
    assert stats["her_messages_covered"] > 0 and stats["samples"]["total"] == train + val + test


def test_r18_f7_allow_incomplete_reaches_the_job(data_dir: Path) -> None:
    """R18-F7 (D-817): ``--allow-incomplete`` travels in the payload of the queued job; a payload
    written before the option existed means "stop on a gap"."""
    queued = invoke("train", "export", "--allow-incomplete")
    assert queued.exit_code == 0, queued.output

    def payloads(services: Services) -> list[dict[str, Any]]:
        queue = JobQueue(services.db, services.clock)
        return [job.payload for job in queue.list_jobs(status="pending", job_type=EXPORT_JOB)]

    (payload,) = with_services(payloads)
    assert payload["allow_incomplete"] is True
    request = export_job.ExportRequest.from_payload(payload)
    assert request.allow_incomplete is True
    older = {key: value for key, value in payload.items() if key != "allow_incomplete"}
    assert export_job.ExportRequest.from_payload(older).allow_incomplete is False


def test_the_status_command_tells_how_the_last_export_ended(
    world_in_cli_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "no export has run yet" in invoke("train", "export-status").output
    monkeypatch.setenv("TWIN_TRAINING__HYBRID_PLAN_RATIO", "0")
    written = with_services(lambda s: fill_missing_plans(s, tiny_qwen_tokenizer()))
    invoke("train", "export", "--out", str(tmp_path / "ds"), "--foreground")
    shown = invoke("train", "export-status")
    assert shown.exit_code == 0
    assert "export: done" in shown.output and "train" in shown.output
    assert f"plans: pending 0, done {written}, refused 0" in shown.output
    assert "planning figures" in shown.output and "5090-8b" in shown.output


def test_r17_6_at_ratio_zero_the_proactive_samples_still_wait_for_their_plans(
    world_in_cli_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-643: the ratio is the share of the ordinary samples; a proactive one always has a plan."""
    monkeypatch.setenv("TWIN_TRAINING__HYBRID_PLAN_RATIO", "0")
    waiting = invoke("train", "export", "--out", str(tmp_path / "ds"), "--foreground")
    assert waiting.exit_code == 0, waiting.output
    assert "export: waiting_for_plans" in waiting.output
    assert "twin jobs approve " in waiting.output and not (tmp_path / "ds").exists()


def test_missing_plans_are_queued_as_a_priced_batch_and_the_second_export_completes(
    world_in_cli_database: None,
    tmp_path: Path,
    api: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "ds"
    waiting = invoke("train", "export", "--out", str(out), "--foreground")
    assert waiting.exit_code == 0, waiting.output
    assert "export: waiting_for_plans" in waiting.output
    assert "planned samples have no plan yet" in waiting.output
    assert "estimated $" in waiting.output
    batches = [
        line.split("twin jobs approve ")[1].strip()
        for line in waiting.output.splitlines()
        if "twin jobs approve " in line
    ]
    assert batches and not out.exists()  # nothing is written before the plans exist

    api.post(API).mock(
        return_value=ok(
            content=json.dumps(
                {"intent": "闲聊几句", "fact_numbers": [], "tone": "随意", "bubble_hint": "两条"},
                ensure_ascii=False,
            )
        )
    )

    def run_plan_jobs(services: Services) -> int:
        services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
        registry = HandlerRegistry()
        registry.register(PLAN_JOB, handle_training_plan)
        worker = Worker(
            JobQueue(services.db, services.clock),
            registry,
            services.clock,
            services=services,
            offpeak=AlwaysOffPeak(),
            alerts=services.alerts,
        )
        return asyncio.run(worker.run_until_idle()).done

    pending = invoke("train", "export-status")
    assert "plans: pending" in pending.output and "pending 0" not in pending.output
    for batch in batches:
        approved = invoke("jobs", "approve", batch, "--yes")
        assert approved.exit_code == 0, approved.output
    done_jobs = with_services(run_plan_jobs)
    assert done_jobs >= len(batches)
    finished = invoke("train", "export", "--out", str(out), "--foreground")
    assert finished.exit_code == 0, finished.output
    assert "export: done" in finished.output and out.is_dir()
    assert "plans " in finished.output
    state = with_services(read_state)
    assert state is not None and state["state"] == "done"
    assert state["stats"]["plans"]["planned"] > 0


def test_r18_c3_the_foreground_export_resplits_rebuilds_and_plans_one_batch(
    data_dir: Path, embedder: HashingBackend, tokenizer: None, tmp_path: Path
) -> None:
    """R18-C3 ② (D-881..D-883): after two weeks of new chat, ``twin train export --foreground``
    moves the cut-off, runs the pre-holdout rebuild it waits for, and plans the plans and the new
    description as one priced batch."""
    from tests.support.memory import mark_days_replayed
    from tests.support.synth_chat import append_texts
    from twin.profile.holdout import get_holdout
    from twin.profile.persona.refresh import count_her_messages, write_description

    def prepare(services: Services) -> datetime:
        built = build_world(services, embedder, days=8)
        write_description(
            services,
            "pre_holdout",
            "### 风格\n- 句子短\n\n",
            provenance={"cutoff": built.cutoff.isoformat()},
            template_version="t@1",
            her_messages=count_her_messages(services, "pre_holdout"),
            at=services.clock.now_utc(),
        )
        moment = datetime(2026, 8, 16, 9, 0, tzinfo=UTC)
        rows = []
        for _ in range(70):
            rows.append((moment, False, "在干嘛呢"))
            rows.append((moment + timedelta(seconds=40), True, "刚下班在路上"))
            moment += timedelta(hours=4, minutes=48)
        append_texts(services, rows)
        mark_days_replayed(services)
        return built.cutoff

    old = with_services(prepare)
    result = invoke("train", "export", "--out", str(tmp_path / "ds"), "--foreground")
    assert result.exit_code == 0, result.output
    assert "the export first moves it to the newest 10%" in result.output
    assert "export: waiting_for_plans" in result.output
    assert "persona description is generated again for the cut-off" in result.output
    assert "twin jobs approve " in result.output and not (tmp_path / "ds").exists()
    stored = with_services(get_holdout)
    assert stored is not None and stored.cutoff > old


def test_r18_c3_a_foreground_export_that_still_waits_says_who_runs_it(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-C3 ② (D-882): when the rebuild an export waits for cannot run here (a sticker job
    waiting for the off-peak hours), the foreground says so and how it goes on."""

    async def nothing(services: Services) -> None:
        export_job.save_state(
            services,
            {
                "state": "rebuilding",
                "cutoff": "2026-08-20T00:00:00+00:00",
                "previous_cutoff": None,
                "waiting_for": ["sticker_tag"],
            },
        )

    monkeypatch.setattr(training_cli, "_export_in_foreground", nothing)
    result = invoke("train", "export", "--foreground")
    assert result.exit_code == 0, result.output
    assert "export: rebuilding" in result.output and "sticker_tag" in result.output
    assert "the application is not running: `twin train export --foreground`" in result.output


def test_a_run_that_fails_on_the_data_says_why_and_is_not_retried(
    world_in_cli_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_TRAINING__HYBRID_PLAN_RATIO", "0")
    failed = invoke(
        "train", "export", "--from", "2031-01-01", "--out", str(tmp_path / "ds"), "--foreground"
    )
    assert failed.exit_code == 1, failed.output
    assert "export: failed" in failed.output and "too few samples" in failed.output
    assert not (tmp_path / "ds").exists()

    def jobs(services: Services) -> list[tuple[str, int]]:
        found = JobQueue(services.db, services.clock).list_jobs(job_type=EXPORT_JOB, limit=10)
        return [(job.status, job.attempts) for job in found]

    assert with_services(jobs) == [("failed", 1)]
    status = invoke("train", "export-status")
    assert "export: failed" in status.output


def test_the_foreground_does_not_run_beside_the_application(
    world_in_cli_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_cli, "app_is_running", lambda services: True)
    result = invoke("train", "export", "--out", str(tmp_path / "ds"), "--foreground")
    assert result.exit_code == 0
    assert "the application is running and will execute the job" in result.output
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--from", "not-a-date"],
        ["--to", "2026-13-40"],
        ["--from", "2026-09-05", "--to", "2026-09-01"],
    ],
)
def test_bad_dates_are_a_usage_error(data_dir: Path, args: list[str]) -> None:
    result = invoke("train", "export", *args)
    assert result.exit_code == 2, result.output
    assert "date" in result.output or "before" in result.output


def test_the_handler_is_registered_for_the_application_and_its_job_type_is_unique() -> None:
    from twin.ops.jobs import load_handlers

    registry = load_handlers()
    assert registry.has(EXPORT_JOB) and registry.has(PLAN_JOB)
    assert registry.get(EXPORT_JOB) is handle_training_export


# ---------------------------------------------------------------------- retrain-check


def test_the_check_command_reports_and_raises_the_reminder(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def prepare(services: Services) -> None:
        writer = MessageWriter(services)
        for number in range(130):
            writer.add(
                datetime(2026, 8, 1, tzinfo=UTC) + timedelta(minutes=number),
                True,
                "text",
                f"她的第{number}句",
            )
        writer.store()

    with_services(prepare)
    nothing = invoke("train", "retrain-check")
    assert nothing.exit_code == 0 and "no style model has been trained yet" in nothing.output
    with_services(lambda services: record_training(services, covered=100))
    due = invoke("train", "retrain-check")
    assert due.exit_code == 0, due.output
    assert "30 of her messages are new" in due.output and "retraining is suggested" in due.output
    assert "twin train export" in due.output

    def alerts(services: Services) -> int:
        with services.db.session() as session:
            return len(
                list(session.scalars(select(Alert).where(Alert.category == "retrain_suggested")))
            )

    assert with_services(alerts) == 1
    invoke("train", "retrain-check")
    assert with_services(alerts) == 1  # once per training
