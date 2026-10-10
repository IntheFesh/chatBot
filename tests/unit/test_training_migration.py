"""R-STO-006 (round 13): the migration of the training tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.unit.test_migrations import table_names
from twin.storage import migrate


def test_round_13_migration_adds_the_training_tables_and_their_constraints(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0008_daily_plans_timezone")
    before = table_names(path)
    migrate.upgrade(path, "0009_training_tables")
    assert table_names(path) - before == {"dataset_versions", "training_runs", "model_registry"}
    assert "0009_training_tables" in migrate.revision_history()
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        dataset = (
            "INSERT INTO dataset_versions (id, scope, directory, holdout_cutoff, train_count, "
            "val_count, test_count, dpo_count, plan_ratio, persona_version, profile_version, "
            "template_version, files, stats, created_at, updated_at) VALUES ('{id}', '{scope}', "
            "'d', '2026-01-01', 10, 1, 1, 0, 0.3, 'v1', 'p1', 't1', '{{}}', '{{}}', "
            "'2026-01-01', '2026-01-01')"
        )
        connection.execute(dataset.format(id="ds1", scope="pre_holdout"))
        with pytest.raises(sqlite3.IntegrityError):  # the scopes of R-TRN-013
            connection.execute(dataset.format(id="ds2", scope="all"))
        run = (
            "INSERT INTO training_runs (id, profile, dataset_version, status, hyperparameters, "
            "steps, artifacts, created_at, updated_at) VALUES ('{id}', '5090-8b', '{ds}', "
            "'{status}', '{{}}', '{{}}', '{{}}', '2026-01-01', '2026-01-01')"
        )
        connection.execute(run.format(id="r1", ds="ds1", status="created"))
        with pytest.raises(sqlite3.IntegrityError):  # a run belongs to a known dataset version
            connection.execute(run.format(id="r2", ds="missing", status="created"))
        with pytest.raises(sqlite3.IntegrityError):  # statuses are a closed vocabulary
            connection.execute(run.format(id="r3", ds="ds1", status="melted"))
        model = (
            "INSERT INTO model_registry (id, run_id, kind, profile, base_model, quant, path, "
            "sha256, size, template_version, persona_version, profile_version, dataset_version, "
            "eval, enabled, active, created_at, updated_at) VALUES ('{id}', 'r1', '{kind}', "
            "'5090-8b', 'Qwen/Qwen3-8B', '{quant}', 'p', 'h', 1, 't', 'v', 'p', 'ds1', '{{}}', "
            "0, 0, '2026-01-01', '2026-01-01')"
        )
        connection.execute(model.format(id="m1", kind="gguf", quant="Q4_K_M"))
        with pytest.raises(sqlite3.IntegrityError):  # one row per run and quantisation
            connection.execute(model.format(id="m2", kind="gguf", quant="Q4_K_M"))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(model.format(id="m3", kind="onnx", quant="Q8_0"))
    finally:
        connection.close()
    migrate.downgrade(path, "0008_daily_plans_timezone")
    assert table_names(path) == before
