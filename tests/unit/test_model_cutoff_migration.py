"""R18-C3 ③ (user decision, D-885): the migration ``0020_model_holdout_cutoff``.

``model_registry.holdout_cutoff`` locks the hold-out cut-off a model was trained with, so that M5
draws its contexts only after it.  The migration adds the column (nullable), fills it for the rows
that exist from the dataset version each row is locked to (``dataset_versions.holdout_cutoff`` -
the cut-off of that export), leaves a row whose dataset is not recorded here empty, and the way
down removes the column and keeps the rows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.unit.test_migrations import table_names
from tests.unit.test_ops_migration import columns, insert
from twin.storage import migrate

AFTER = "0020_model_holdout_cutoff"
PARENT = "0019_sticker_first_frame"
CUTOFF = "2026-08-14 02:46:19.000000"


def test_the_revision_follows_the_sticker_first_frame_in_the_one_linear_history() -> None:
    history = migrate.revision_history()
    assert len(history) == len(set(history))
    assert history.index(AFTER) == history.index(PARENT) + 1
    assert migrate.head_revision() in history


def _rows(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(path)
    try:
        return dict(connection.execute("SELECT id, holdout_cutoff FROM model_registry").fetchall())
    finally:
        connection.close()


def test_the_column_is_added_and_filled_from_the_dataset_of_each_model(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, PARENT)
    old = columns(path, "model_registry")
    tables = table_names(path)
    connection = sqlite3.connect(path)
    try:
        insert(
            connection, "dataset_versions", id="ds-1", holdout_cutoff=CUTOFF, scope="pre_holdout"
        )
        for run, dataset in (("r-1", "ds-1"), ("r-2", "ds-gone")):
            insert(
                connection,
                "model_registry",
                id=f"{run}-Q5_K_M",
                run_id=run,
                quant="Q5_K_M",
                dataset_version=dataset,
                kind="gguf",
            )
        connection.commit()
    finally:
        connection.close()
    migrate.upgrade(path, AFTER)
    new = columns(path, "model_registry")
    assert table_names(path) == tables
    assert set(new) - set(old) == {"holdout_cutoff"}
    kind, not_null, default = new["holdout_cutoff"]
    assert kind.upper() == "DATETIME" and not not_null and default is None
    assert {name: new[name] for name in old} == old
    assert _rows(path) == {"r-1-Q5_K_M": CUTOFF, "r-2-Q5_K_M": None}


def test_the_way_down_removes_the_column_and_keeps_the_rows(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        insert(connection, "model_registry", id="r-1-Q5_K_M", holdout_cutoff=CUTOFF, kind="gguf")
        connection.commit()
    finally:
        connection.close()
    migrate.downgrade(path, PARENT)
    assert "holdout_cutoff" not in columns(path, "model_registry")
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT id FROM model_registry").fetchall() == [("r-1-Q5_K_M",)]
    finally:
        connection.close()
    migrate.upgrade(path, AFTER)
    assert _rows(path) == {"r-1-Q5_K_M": None}
