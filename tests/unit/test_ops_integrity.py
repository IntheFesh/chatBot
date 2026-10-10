"""The weekly integrity check: SQLite and the vector index (R-OPS-003)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import delete

from tests.support.embedding import HashingBackend
from tests.support.memory import add_fact, make_memory, utc
from twin.ops.integrity import (
    REPAIRS,
    VECTOR_SOURCES,
    TableCheck,
    check_database,
    check_vectors,
    run_integrity,
)
from twin.retrieval.vector_store import VectorStore
from twin.services import Services
from twin.storage.db import Database
from twin.storage.memory_models import Fact
from twin.storage.vector_schema import FACT_SCHEMA


def facts_with_vectors(services: Services, embedder: HashingBackend, count: int) -> None:
    memory = make_memory(services)
    for number in range(count):
        add_fact(memory, f"合成事实 {number} 号", utc(2026, 3, 10 + number))


def test_a_clean_database_and_index_are_ok(services: Services, embedder: HashingBackend) -> None:
    facts_with_vectors(services, embedder, 3)
    now = services.clock.now_utc()
    report = run_integrity(services.db, VectorStore(services.paths.vectors_dir), now)
    assert report.ok and report.database_ok and report.problems() == []
    assert report.at == now
    (facts,) = report.vectors  # only the tables that exist are checked
    assert facts == TableCheck("memory_facts", 3, 0) and facts.ok
    assert report.to_json()["vectors"] == [{"table": "memory_facts", "rows": 3, "orphans": 0}]
    assert report.to_json()["ok"] is True


def test_a_database_without_any_index_is_ok(services: Services) -> None:
    report = run_integrity(
        services.db, VectorStore(services.paths.vectors_dir), services.clock.now_utc()
    )
    assert report.ok and report.vectors == ()


def test_vector_ids_without_a_database_row_are_reported_with_the_repair(
    services: Services, embedder: HashingBackend
) -> None:
    facts_with_vectors(services, embedder, 4)
    with services.db.transaction() as session:
        session.execute(delete(Fact))  # the database forgot them, the index did not
    checks = check_vectors(services.db, VectorStore(services.paths.vectors_dir))
    assert checks == (TableCheck("memory_facts", 4, 4),)
    report = run_integrity(
        services.db, VectorStore(services.paths.vectors_dir), services.clock.now_utc()
    )
    assert report.database_ok and not report.ok
    (line,) = report.problems()
    assert line == "memory_facts: 4 id(s) not in the database; run twin memory reindex"
    assert report.to_json()["ok"] is False


def test_every_vector_table_names_the_command_that_repairs_it() -> None:
    assert {source.schema.name for source in VECTOR_SOURCES} == set(REPAIRS)
    assert FACT_SCHEMA.name in REPAIRS


def test_a_damaged_file_is_a_result_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "damaged.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)")
    connection.execute("CREATE INDEX ib ON t (b)")
    connection.executemany(
        "INSERT INTO t VALUES (?, ?)", [(i, "x" * 200 + str(i)) for i in range(400)]
    )
    connection.commit()
    connection.close()
    data = bytearray(path.read_bytes())
    data[4096 + 8 : 4096 + 200] = b"\xff" * 192  # the header of the second page
    path.write_bytes(bytes(data))
    db = Database(path)
    try:
        clean, messages = check_database(db)
    finally:
        db.dispose()
    assert not clean and messages != ("ok",) and "database disk image is malformed" in messages[0]


def test_a_healthy_file_passes_the_check(services: Services) -> None:
    assert check_database(services.db) == (True, ("ok",))
