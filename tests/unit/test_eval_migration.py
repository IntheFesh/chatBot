"""R-STO-006 (round 09b): the migration of the evaluation tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.unit.test_migrations import table_names
from twin.storage import migrate


def test_round_09b_migration_adds_the_evaluation_tables_and_their_constraints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0011_training_plans")
    before = table_names(path)
    migrate.upgrade(path, "0012_eval_tables")
    assert table_names(path) - before == {"eval_runs", "eval_items"}
    assert "0012_eval_tables" in migrate.revision_history()
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        run = (
            "INSERT INTO eval_runs (id, kind, status, mode, milestone, verdict, backends, "
            "batch_ids, params, summary, created_at, updated_at) VALUES ('{id}', '{kind}', "
            "'{status}', {mode}, NULL, {verdict}, '[]', '[]', '{{}}', '{{}}', "
            "'2026-03-08', '2026-03-08')"
        )
        defaults = {"kind": "blind", "status": "planned", "mode": "'holdout'", "verdict": "NULL"}
        connection.execute(run.format(id="r1", **defaults))
        for name, bad in (
            ("r2", {"kind": "essay"}),
            ("r3", {"status": "stalled"}),
            ("r4", {"mode": "'dream'"}),
            ("r5", {"verdict": "'maybe'"}),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(run.format(id=name, **(defaults | bad)))
        connection.execute(
            run.format(id="r6", **(defaults | {"kind": "gate", "verdict": "'passed'"}))
        )

        item = (
            "INSERT INTO eval_items (id, run_id, seq, sample_key, backend, at, status, outcome, "
            "auto_outcome, cost_usd, payload, created_at, updated_at) VALUES ('{id}', '{run}', "
            "{seq}, 'k', 'deepseek', '2026-03-08', '{status}', {outcome}, NULL, 0, x'00', "
            "'2026-03-08', '2026-03-08')"
        )
        fields = {"run": "r1", "seq": 0, "status": "pending", "outcome": "NULL"}
        connection.execute(item.format(id="i1", **fields))
        for name, bad in (
            ("i2", {"seq": 1, "status": "thinking"}),
            ("i3", {"seq": 1, "outcome": "'maybe'"}),
            ("i4", {"seq": -1}),
            ("i5", {"seq": 0}),  # a position of a run is used once
            ("i6", {"seq": 1, "run": "nope"}),  # the run must exist
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(item.format(id=name, **(fields | bad)))
        for number, outcome in enumerate(("'correct'", "'partial'", "'wrong'"), start=1):
            connection.execute(
                item.format(id=f"ok{number}", **(fields | {"seq": number, "outcome": outcome}))
            )
        connection.execute("DELETE FROM eval_runs WHERE id = 'r1'")  # the items go with their run
        assert connection.execute("SELECT COUNT(*) FROM eval_items").fetchone() == (0,)
    finally:
        connection.close()
    migrate.downgrade(path, "0011_training_plans")
    assert table_names(path) == before
