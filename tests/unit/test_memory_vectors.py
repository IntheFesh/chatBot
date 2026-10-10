"""The vectors of the memory: what is encoded, when it became known, and nothing from the future.

R-MEM-003 (the memory is searched by meaning), R-TRN-013 (no peeking at what was not yet known),
R-RET-006 (one encoding per table), R-NFR-005 (on an injected clock).  The vector tables hold the
ids, the vectors and the moment each record became known - never a word of text (R-STO-005).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.memory import add_fact, add_summary, make_memory, utc
from twin.clock import to_epoch
from twin.memory.vectors import MemoryVectors, VectorSync, encoding_of
from twin.retrieval.embedder import embedding_service
from twin.services import Services
from twin.storage.vector_schema import VectorKind


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
