"""R-TRN-010, R-PRIV-003: dataset versions and the records of the training runs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.support.clock import ManualClock
from tests.support.training_data import write_synthetic_dataset
from twin.services import Services
from twin.storage.training_models import DatasetVersion
from twin.training.profiles import PROFILES
from twin.training.runs import (
    RunError,
    RunStore,
    ensure_dataset_version,
    hyperparameters,
    new_run_id,
    uncleaned_runs,
)

NOW = datetime(2026, 10, 9, 12, 30, 5, tzinfo=UTC)


def make_run(services: Services, tmp_path: Path, run_id: str = "r1") -> RunStore:
    dataset = write_synthetic_dataset(tmp_path / "ds")
    ensure_dataset_version(services.db, dataset, "ds")
    store = RunStore(services.db)
    store.create(
        run_id=run_id,
        profile="5090-8b",
        dataset_version="ds-test-01",
        bundle_sha256="a" * 64,
        bundle_size=10,
        parameters=hyperparameters(PROFILES["5090-8b"], 40),
    )
    return store


def test_run_ids_say_when_and_which_profile_and_differ_every_time() -> None:
    first, second = new_run_id("5090-8b", NOW), new_run_id("5090-8b", NOW)
    assert first.startswith("r20261009-123005-5090-8b-") and len(first) == len(second)
    other_zone = datetime(2026, 10, 9, 7, 30, 5, tzinfo=UTC) - timedelta(hours=0)
    assert new_run_id("pro6000-14b", other_zone).startswith("r20261009-073005-pro6000-14b-")
    assert len({new_run_id("5090-8b", NOW) for _ in range(20)}) > 1


def test_the_hyperparameters_record_the_settings_of_the_profile() -> None:
    values = hyperparameters(PROFILES["5090-14b"], 25_000)
    assert values["epochs"] == 2 and values["quantization_bit"] == 4
    assert values["template"] == "qwen3_nothink" and values["lora_rank"] == 32
    assert values["train_samples"] == 25_000 and values["cutoff_len"] == 2048
    assert hyperparameters(PROFILES["5090-8b"], 100)["epochs"] == 3


def test_a_dataset_version_is_recorded_once_and_names_exactly_one_content(
    services: Services, tmp_path: Path
) -> None:
    dataset = write_synthetic_dataset(tmp_path / "ds")
    assert ensure_dataset_version(services.db, dataset, "ds") == "ds-test-01"
    assert ensure_dataset_version(services.db, dataset, "ds") == "ds-test-01"
    with services.db.session() as session:
        row = session.get(DatasetVersion, "ds-test-01")
        assert row is not None
        assert (row.train_count, row.val_count, row.test_count, row.dpo_count) == (40, 6, 6, 0)
        assert row.scope == "pre_holdout" and row.directory == "ds"
        assert row.holdout_cutoff == datetime(2026, 9, 1, tzinfo=UTC)
        assert row.persona_version == "v3" and row.template_version.startswith("qwen3_nothink@")
        assert set(row.files) == {"sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"}
    changed = write_synthetic_dataset(tmp_path / "ds2", train=41)  # same id, other data
    with pytest.raises(RunError, match="different files"):
        ensure_dataset_version(services.db, changed, "ds2")


def test_a_run_starts_created_and_each_step_moves_it_on(services: Services, tmp_path: Path) -> None:
    store = make_run(services, tmp_path)
    run = store.get("r1")
    assert run.status == "created" and run.steps == {} and run.bundle_sha256 == "a" * 64
    for step, status in (
        ("upload", "uploaded"),
        ("setup", "set_up"),
        ("train", "trained"),
        ("dpo", "dpo_done"),
        ("eval", "evaluated"),
        ("export", "exported"),
        ("download", "downloaded"),
    ):
        store.begin_step("r1", step, NOW)
        assert store.get("r1").steps[step]["ok"] is False
        store.end_step("r1", step, NOW + timedelta(minutes=5), exit_code=0)
        assert store.get("r1").status == status
        assert store.get("r1").steps[step]["ok"] is True
    final = store.get("r1")
    assert final.started_at == NOW and final.cleaned_at is None


def test_a_failed_step_marks_the_run_and_keeps_the_reason(
    services: Services, tmp_path: Path
) -> None:
    store = make_run(services, tmp_path)
    store.begin_step("r1", "train", NOW)
    store.end_step("r1", "train", NOW, exit_code=3)
    failed = store.get("r1")
    assert failed.status == "failed" and failed.last_error == "train exited with code 3"
    assert failed.steps["train"]["exit_code"] == 3 and failed.steps["train"]["ok"] is False
    store.begin_step("r1", "train", NOW)  # trying again clears the old reason
    assert store.get("r1").last_error is None
    store.end_step("r1", "train", NOW, exit_code=1, error="it was lost")
    assert store.get("r1").last_error == "it was lost"


def test_the_columns_learned_on_the_way_are_stored_with_the_step(
    services: Services, tmp_path: Path
) -> None:
    store = make_run(services, tmp_path)
    store.begin_step("r1", "train", NOW)
    store.end_step(
        "r1",
        "train",
        NOW,
        exit_code=0,
        fields={
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "peak_vram_mib": 21000,
            "best_val_loss": 1.25,
            "best_checkpoint": "checkpoint-8",
        },
    )
    run = store.get("r1")
    assert (run.gpu_model, run.peak_vram_mib, run.best_val_loss, run.best_checkpoint) == (
        "NVIDIA GeForce RTX 5090",
        21000,
        1.25,
        "checkpoint-8",
    )
    store.update("r1", artifacts={"gguf/a.gguf": {"sha256": "b" * 64, "size": 3}})
    assert store.get("r1").artifacts["gguf/a.gguf"]["size"] == 3


def test_the_log_offset_survives_a_new_attempt_only_when_asked(
    services: Services, tmp_path: Path
) -> None:
    store = make_run(services, tmp_path)
    store.begin_step("r1", "train", NOW)
    store.set_step_fields("r1", "train", log_offset=4096)
    store.begin_step("r1", "train", NOW, keep_offset=True)
    assert store.get("r1").steps["train"]["log_offset"] == 4096
    store.begin_step("r1", "train", NOW)
    assert store.get("r1").steps["train"]["log_offset"] == 0
    store.end_step("r1", "train", NOW, exit_code=0)
    assert store.get("r1").steps["train"]["ok"] is True


def test_a_run_with_data_on_an_instance_is_uncleaned_until_the_cleanup_is_recorded(
    services: Services, tmp_path: Path, clock: ManualClock
) -> None:
    store = make_run(services, tmp_path)
    assert uncleaned_runs(services.db) == []  # nothing was uploaded yet
    store.begin_step("r1", "upload", NOW)
    store.end_step("r1", "upload", NOW, exit_code=1, error="the connection broke")
    assert uncleaned_runs(services.db) == []  # a failed upload put nothing there
    store.begin_step("r1", "upload", NOW)
    store.end_step("r1", "upload", NOW, exit_code=0)
    assert [run.id for run in uncleaned_runs(services.db)] == ["r1"]
    store.begin_step("r1", "cleanup", NOW + timedelta(hours=3))
    store.end_step("r1", "cleanup", NOW + timedelta(hours=3), exit_code=0)
    done = store.get("r1")
    assert done.cleaned_at == NOW + timedelta(hours=3) and done.status == "cleaned"
    assert uncleaned_runs(services.db) == []


def test_runs_are_listed_newest_first_and_an_unknown_run_is_an_error(
    services: Services, tmp_path: Path, clock: ManualClock
) -> None:
    store = make_run(services, tmp_path, "r1")
    assert store.latest() is not None and store.latest().id == "r1"  # type: ignore[union-attr]
    clock.set_time(NOW + timedelta(days=1))
    store.create(
        run_id="r2",
        profile="5090-14b",
        dataset_version="ds-test-01",
        bundle_sha256="c" * 64,
        bundle_size=1,
        parameters={},
    )
    assert [run.id for run in store.all()] == ["r2", "r1"]
    assert store.latest().id == "r2"  # type: ignore[union-attr]
    with pytest.raises(RunError, match="no training run"):
        store.get("r9")
    with pytest.raises(RunError, match="no training run"):
        store.begin_step("r9", "train", NOW)
