"""The vectors of the memory: what is encoded, when it became known, and nothing from the future.

R-MEM-003 (the memory is searched by meaning), R-TRN-013 (no peeking at what was not yet known),
R-RET-006 (one encoding per table), R-NFR-005 (on an injected clock).  The vector tables hold the
ids, the vectors and the moment each record became known - never a word of text (R-STO-005).
"""

from __future__ import annotations

import os
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.memory import add_fact, add_summary, make_memory, utc
from twin.clock import to_epoch
from twin.memory.vectors import MemoryVectors, StaleVectorsError, VectorSync, encoding_of
from twin.retrieval import embedder as embedder_module
from twin.retrieval.embedder import EmbedderError, embedding_service, reset_embedding_services
from twin.retrieval.vector_store import IndexMeta, VectorStore
from twin.services import Services
from twin.storage.vector_schema import FACT_SCHEMA, VectorKind


def vectors_of(services: Services) -> MemoryVectors:
    memory = make_memory(services)
    assert isinstance(memory.vectors, MemoryVectors)
    return memory.vectors


def test_a_fact_is_stored_with_the_moment_it_became_known_and_no_text(
    services: Services, embedder: HashingBackend, clock: ManualClock
) -> None:
    memory = make_memory(services)
    known = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)  # 01:30 in Chicago, the repeated hour
    fact = add_fact(memory, "她下周三有考试", known)
    table = memory.vectors.table(VectorKind.FACT)
    [row] = table.rows()
    assert row["id"] == fact.id and row["at"] == int(to_epoch(known))
    assert row["kind"] == "fact" and "考试" not in str(row)  # an id and a vector, not a word
    meta = table.read_meta()
    assert meta is not None and meta.encoding == encoding_of(embedding_service(services).info)
    written = datetime.fromisoformat(meta.written_at)
    assert written.tzinfo is not None and written == clock.now_utc()  # the injected clock's now


def test_a_search_never_finds_what_was_not_yet_known_at_the_bound(
    services: Services, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    early = add_fact(memory, "她喜欢吃火锅", utc(2026, 10, 1))
    late = add_fact(memory, "她喜欢吃火锅和烧烤", utc(2026, 10, 20))
    vectors = memory.vectors
    bound = int(to_epoch(utc(2026, 10, 10)))
    before = vectors.search(VectorKind.FACT, "她喜欢吃什么", 5, at_or_before=bound)
    assert [hit.id for hit in before] == [early.id]  # the later fact is not in that past
    after = vectors.search(VectorKind.FACT, "她喜欢吃什么", 5)
    assert {hit.id for hit in after} == {early.id, late.id}
    assert (
        vectors.search(VectorKind.FACT, "  ", 5) == []
        and vectors.search(VectorKind.FACT, "x", 0) == []
    )


def test_a_summary_is_known_at_the_end_of_its_local_day(
    services: Services, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    day = date(2026, 11, 1)  # 25 hours long in Chicago
    summary = add_summary(memory, "real", day, "两个人聊了考试")
    [row] = memory.vectors.table(VectorKind.SUMMARY).rows()
    assert row["id"] == summary.id and row["at"] == int(to_epoch(summary.utc_end))
    start, end = memory.clock.real_bounds(day)
    assert end - start == timedelta(hours=25) and summary.utc_end == end
    inside = int(to_epoch(end)) - 1
    assert memory.vectors.search(VectorKind.SUMMARY, "考试", 3, at_or_before=inside) == []
    assert [h.id for h in memory.vectors.search(VectorKind.SUMMARY, "考试", 3)] == [summary.id]


def test_what_is_encoded_already_is_not_encoded_again_and_another_model_starts_over(
    services: Services, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    add_fact(memory, "她有一只猫", utc(2026, 10, 1))
    [fact] = memory.store.all_facts()  # as stored now: it carries the encoding it was made with
    vectors = memory.vectors
    current = encoding_of(embedding_service(services).info)
    assert vectors.pending_facts([fact], current) == []  # already shown by the current vectors
    assert vectors.pending_facts([fact], "another|8|abc|memory-v1") == [fact]
    assert vectors.usable(VectorKind.FACT)
    table = vectors.table(VectorKind.FACT)
    meta = table.read_meta()
    assert meta is not None
    table.write_meta(type(meta)(**{**meta.to_json(), "encoding": "another|8|abc|memory-v1"}))
    assert not vectors.usable(VectorKind.FACT)  # a table of another model is not searched
    result = vectors.sync_facts([fact])
    assert result == VectorSync(1, 0, True)  # dropped and written again with the current model
    assert vectors.usable(VectorKind.FACT) and table.count() == 1


def test_removed_records_leave_the_index_and_a_reset_forgets_everything(
    services: Services, embedder: HashingBackend
) -> None:
    memory = make_memory(services)
    first = add_fact(memory, "她喜欢咖啡", utc(2026, 10, 1))
    second = add_fact(memory, "她喜欢茶", utc(2026, 10, 2))
    vectors = memory.vectors
    vectors.remove(VectorKind.FACT, [first.id])
    vectors.remove(VectorKind.FACT, [])  # nothing to do
    assert [row["id"] for row in vectors.table(VectorKind.FACT).rows()] == [second.id]
    assert vectors.sync_facts([]) == VectorSync(0, 0, False)
    assert vectors.sync_summaries([]) == VectorSync(0, 0, False)
    vectors.rebuild_reset()
    assert not vectors.usable(VectorKind.FACT) and not vectors.usable(VectorKind.SUMMARY)
    assert not vectors.table(VectorKind.FACT).exists()


# ------------------------------------------------- another embedding model (R18-C2, D-781)


def switch_model(monkeypatch: pytest.MonkeyPatch, backend: HashingBackend) -> None:
    """What changing ``retrieval.model`` (or its weights) and restarting does to the memory."""
    monkeypatch.setattr(embedder_module, "backend_factory", lambda config, paths: backend)
    reset_embedding_services()


def test_a_model_switch_re_encodes_every_old_record_on_the_next_write(
    services: Services, embedder: HashingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-C2 (D-781): the first write after the switch encodes the old facts and summaries too.

    Only the new fact used to be encoded: the table of the old model was dropped and the old
    records, which still carried a vector id, never got a vector again.
    """
    memory = make_memory(services)
    exam = add_fact(memory, "她下周三有期末考试", utc(2026, 10, 1))
    food = add_fact(memory, "她最喜欢吃火锅", utc(2026, 10, 2))
    day = add_summary(memory, "real", date(2026, 10, 1), "两个人聊了期末考试")
    other = HashingBackend(model="test/other-model", weights="w2")
    switch_model(monkeypatch, other)
    after = make_memory(services)
    cat = add_fact(after, "她养了一只猫", utc(2026, 10, 3))
    facts = after.vectors.table(VectorKind.FACT)
    summaries = after.vectors.table(VectorKind.SUMMARY)
    assert {row["id"] for row in facts.rows()} == {exam.id, food.id, cat.id}
    assert {row["id"] for row in summaries.rows()} == {day.id}
    current = encoding_of(other.info)
    assert {f.embed_version for f in after.store.all_facts()} == {current}
    assert {s.embed_version for s in after.store.current_summaries()} == {current}
    assert exam.id in {hit.id for hit in after.vectors.search(VectorKind.FACT, "期末考试", 5)}


def test_a_lost_table_is_written_again_by_the_next_write(
    services: Services, embedder: HashingBackend
) -> None:
    """R18-C2 (D-781): records that say they have a vector get one when the table is gone."""
    memory = make_memory(services)
    first = add_fact(memory, "她喜欢咖啡", utc(2026, 10, 1))
    vectors = memory.vectors
    vectors.rebuild_reset()  # e.g. a restore without the vector folder
    second = add_fact(memory, "她喜欢茶", utc(2026, 10, 2))
    rows = {row["id"] for row in vectors.table(VectorKind.FACT).rows()}
    assert rows == {first.id, second.id}


def test_a_search_on_vectors_of_another_model_raises_instead_of_finding_nothing(
    services: Services, embedder: HashingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-C2 (D-781): until the next write, the table is of the old model: say so, loudly."""
    memory = make_memory(services)
    add_fact(memory, "她下周三有期末考试", utc(2026, 10, 1))
    switch_model(monkeypatch, HashingBackend(model="test/other-model", weights="w2"))
    after = make_memory(services)
    assert not after.vectors.usable(VectorKind.FACT)
    with pytest.raises(StaleVectorsError, match="twin memory reindex"):
        after.vectors.search(VectorKind.FACT, "考试", 5)
    with pytest.raises(EmbedderError):  # what the callers that can go on without vectors catch
        after.vectors.search(VectorKind.FACT, "考试", 5)
    assert after.vectors.search(VectorKind.SUMMARY, "考试", 5) == []  # no table: nothing to find


def test_a_reader_never_sees_the_meta_of_a_table_half_written(tmp_path: Path) -> None:
    """R18-C2 follow-up (D-781): the meta file was rewritten in place (truncate, then write).

    Since D-781 a search reads it to tell "made with another model" from "current", so a replay
    job that read it while a job beside it was syncing saw an empty file, decided the vectors
    were stale and failed (the Windows CI run of
    ``test_spending_above_the_estimate_pauses_the_batch_and_approving_again_continues``).  The
    file is now replaced as a whole: a reader sees the old meta or the new one, never nothing.
    """
    table = VectorStore(tmp_path).table(FACT_SCHEMA)
    metas = [
        IndexMeta(
            table="facts",
            encoding=f"model-{n}|512|{'0' * 16}|v1",
            model=f"model-{n}",
            dimension=512,
            revision="r",
            weights_sha256="0" * 64,
            written_at=f"2026-10-09T12:00:{n:02d}+00:00",
        )
        for n in range(2)
    ]
    table.write_meta(metas[0])
    stop = threading.Event()
    missed: list[int] = []

    def read_while_written() -> None:
        reads = 0
        while not stop.is_set():
            reads += 1
            if table.read_meta() not in metas:
                missed.append(reads)

    reader = threading.Thread(target=read_while_written)
    reader.start()
    try:
        for n in range(3000):
            table.write_meta(metas[n % 2])
    finally:
        stop.set()
        reader.join()
    assert missed == []
    assert (
        sorted(p.name for p in table.meta_path.parent.iterdir() if p.name.endswith(".part")) == []
    )


def test_a_meta_file_held_open_for_a_moment_is_replaced_once_it_is_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-781 follow-up: Windows refuses to replace a file another thread holds open; the write
    tries again, briefly, and a file that stays busy is an error (no stray ``.part`` left)."""
    from twin.retrieval import vector_store

    monkeypatch.setattr(vector_store, "META_PAUSE_S", 0.0)
    table = VectorStore(tmp_path).table(FACT_SCHEMA)
    meta = IndexMeta(
        table="facts",
        encoding="m|512|0000000000000000|v1",
        model="m",
        dimension=512,
        revision="r",
        weights_sha256="0" * 64,
        written_at="2026-10-09T12:00:00+00:00",
    )
    real_replace = os.replace
    busy = {"left": 3}

    def replace_when_free(source: str | Path, target: str | Path) -> None:
        if busy["left"] > 0:
            busy["left"] -= 1
            raise PermissionError(13, "the file is in use by another process")
        real_replace(source, target)

    monkeypatch.setattr(vector_store.os, "replace", replace_when_free)
    table.write_meta(meta)
    assert table.read_meta() == meta and busy["left"] == 0
    busy["left"] = vector_store.META_ATTEMPTS  # never free in time
    with pytest.raises(PermissionError):
        table.write_meta(meta)
    assert [p.name for p in tmp_path.rglob("*.part")] == []
    assert table.read_meta() == meta  # the old meta is still whole
