"""R-STO-006 (round 15): the migration ``0017_eval_kinds`` - three kinds, two tables.

``eval_runs.kind`` gains ``consistency``, ``cost`` and ``report``; the constraint is rebuilt
from the kinds the table holds, so what other rounds added survives both ways, and the evaluation
items of the existing runs are kept.  The tables of the findings and their fixes are made after
the rebuild and dropped before it on the way down.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.unit.test_migrations import table_names
from tests.unit.test_ops_migration import columns, insert
from twin.storage import migrate

BEFORE = "0016_proactive_tables"
AFTER = "0017_eval_kinds"
NEW_KINDS = ("consistency", "cost", "report")
OLD_KINDS = ("blind", "style", "memory", "gate", "stability", "proactive_audit")


def test_the_migration_follows_the_proactive_tables_and_is_the_head() -> None:
    history = migrate.revision_history()
    assert history.index(AFTER) == history.index(BEFORE) + 1
    assert history[-1] == AFTER


def test_it_adds_the_two_tables_and_nothing_else(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    before = table_names(path)
    migrate.upgrade(path, AFTER)
    assert table_names(path) - before == {"consistency_findings", "consistency_fixes"}
    assert before - table_names(path) == set()
    assert columns(path, "eval_runs")["kind"][0].upper().startswith("VARCHAR(16)")


def test_the_three_kinds_are_accepted_and_the_old_ones_still_are(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        for kind in (*OLD_KINDS, *NEW_KINDS):
            insert(connection, "eval_runs", id=f"R{kind}".ljust(26, "0"), kind=kind, status="done")
        with pytest.raises(sqlite3.IntegrityError):
            insert(connection, "eval_runs", id="Z" * 26, kind="essay", status="done")
    finally:
        connection.close()


def test_the_vocabularies_of_the_new_tables_are_enforced_by_the_database(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        insert(connection, "eval_runs", id="R" * 26, kind="consistency", status="running")
        good = {"run_id": "R" * 26, "seq": 0, "fingerprint": "f" * 64, "model_severity": "minor"}
        insert(connection, "consistency_findings", id="A" * 26, status="proposed", **good)
        insert(
            connection, "consistency_findings", id="B" * 26, status="confirmed", severity="obvious",
            **{**good, "seq": 1},
        )  # fmt: skip
        for bad in (
            {"status": "maybe", "seq": 2},
            {"status": "proposed", "model_severity": "huge", "seq": 3},
            {"status": "confirmed", "seq": 4},  # confirmed needs a severity
            {"status": "proposed", "severity": "obvious", "seq": 5},  # only a confirmed one has it
            {"status": "rejected", "severity": "minor", "seq": 6},
            {"status": "proposed", "seq": 0},  # the number of a run's finding is unique
            {"status": "proposed", "seq": -1},
        ):
            with pytest.raises(sqlite3.IntegrityError):
                insert(connection, "consistency_findings", id="C" * 26, **{**good, **bad})
        fix = {"finding_id": "A" * 26, "seq": 0, "target_id": "T" * 26}
        insert(
            connection,
            "consistency_fixes",
            id="F" * 26,
            action="rewrite_fact",
            status="proposed",
            **fix,
        )
        insert(
            connection, "consistency_fixes", id="G" * 26, action="invalidate_lifeline",
            status="applied", applied_at="2026-10-10 00:00:00", **{**fix, "seq": 1},
        )  # fmt: skip
        for bad in (
            {"action": "delete_everything", "status": "proposed", "seq": 2},
            {"action": "rewrite_fact", "status": "done", "seq": 3},
            {"action": "rewrite_fact", "status": "applied", "seq": 4},  # applied needs its time
            {"action": "rewrite_fact", "status": "proposed", "applied_at": "2026-10-10", "seq": 5},
            {"action": "rewrite_fact", "status": "proposed", "seq": 0},
        ):
            with pytest.raises(sqlite3.IntegrityError):
                insert(connection, "consistency_fixes", id="H" * 26, **{**fix, **bad})
        with pytest.raises(sqlite3.IntegrityError):  # a finding belongs to a run that exists
            insert(
                connection,
                "consistency_findings",
                id="D" * 26,
                status="proposed",
                **{**good, "run_id": "X" * 26, "seq": 9},
            )
    finally:
        connection.close()


def test_the_findings_and_fixes_go_with_their_run(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        insert(connection, "eval_runs", id="R" * 26, kind="consistency", status="done")
        insert(
            connection, "consistency_findings", id="A" * 26, run_id="R" * 26, seq=0,
            fingerprint="f" * 64, model_severity="minor", status="proposed",
        )  # fmt: skip
        insert(
            connection, "consistency_fixes", id="F" * 26, finding_id="A" * 26, seq=0,
            action="rewrite_fact", target_id="T" * 26, status="proposed",
        )  # fmt: skip
        connection.execute("DELETE FROM eval_runs WHERE id = ?", ("R" * 26,))
        assert connection.execute("SELECT COUNT(*) FROM consistency_findings").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM consistency_fixes").fetchone()[0] == 0
    finally:
        connection.close()


def test_the_runs_and_the_items_of_earlier_rounds_survive_the_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        for kind in OLD_KINDS:
            insert(connection, "eval_runs", id=f"R{kind}".ljust(26, "0"), kind=kind, status="done")
        insert(
            connection, "eval_items", id="I" * 26, run_id="Rblind".ljust(26, "0"), seq=0,
            sample_key="k", backend="deepseek", at="2026-10-01 00:00:00", status="pending",
            cost_usd=0.0, payload=b"\x01\x02",
        )  # fmt: skip
        connection.commit()
    finally:
        connection.close()
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        kinds = {row[0] for row in connection.execute("SELECT kind FROM eval_runs")}
        assert kinds == set(OLD_KINDS)
        assert connection.execute("SELECT id, payload FROM eval_items").fetchall() == [
            ("I" * 26, b"\x01\x02")
        ]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='eval_items'"
        ).fetchone()[0]
        assert "REFERENCES eval_runs" in sql
    finally:
        connection.close()


def test_a_kind_another_round_added_is_kept_both_ways(tmp_path: Path) -> None:
    """The constraint is read from the table, not written down (as rounds 10 and 12 do)."""
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    connection = sqlite3.connect(path)
    sql = str(
        connection.execute("SELECT sql FROM sqlite_master WHERE name = 'eval_runs'").fetchone()[0]
    )
    assert "'proactive_audit'" in sql
    connection.execute("PRAGMA writable_schema = ON")
    connection.execute(
        "UPDATE sqlite_master SET sql = ? WHERE name = 'eval_runs'",
        (sql.replace("'proactive_audit'", "'proactive_audit', 'elsewhere'"),),
    )
    connection.commit()
    connection.close()

    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    try:
        for kind in ("elsewhere", "consistency", "proactive_audit"):
            insert(connection, "eval_runs", id=f"K{kind}".ljust(26, "0"), kind=kind, status="done")
        connection.commit()
    finally:
        connection.close()

    migrate.downgrade(path, BEFORE)
    connection = sqlite3.connect(path)
    try:
        kinds = {row[0] for row in connection.execute("SELECT kind FROM eval_runs")}
        assert kinds == {"elsewhere", "proactive_audit"}  # only the new kinds went with it
        insert(connection, "eval_runs", id="Y" * 26, kind="elsewhere", status="done")
        for kind in NEW_KINDS:
            with pytest.raises(sqlite3.IntegrityError):
                insert(connection, "eval_runs", id="W" * 26, kind=kind, status="done")
    finally:
        connection.close()


def test_the_migration_can_be_undone_and_done_again(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, BEFORE)
    old_tables = table_names(path)
    old_eval = columns(path, "eval_runs")
    migrate.upgrade(path, AFTER)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    insert(connection, "eval_runs", id="R" * 26, kind="consistency", status="done")
    insert(connection, "eval_runs", id="S" * 26, kind="blind", status="done")
    insert(
        connection, "consistency_findings", id="A" * 26, run_id="R" * 26, seq=0,
        fingerprint="f" * 64, model_severity="minor", status="proposed",
    )  # fmt: skip
    connection.commit()
    connection.close()

    migrate.downgrade(path, BEFORE)
    assert table_names(path) == old_tables and columns(path, "eval_runs") == old_eval
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT id FROM eval_runs").fetchall() == [("S" * 26,)]
    finally:
        connection.close()
    migrate.upgrade(path, AFTER)
    assert {"consistency_findings", "consistency_fixes"} <= table_names(path)
    assert migrate.current_revision(path) == AFTER
