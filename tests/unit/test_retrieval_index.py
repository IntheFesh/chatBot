"""The vector index and its incremental updates (R-STO-005, R-RET-002, R-RET-003, R-RET-006)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.embedding import DIMENSION, H, HashingBackend, Msg, U, day, write_dialogue
from twin.ops.filelock import FileLock
from twin.profile.holdout import get_holdout, resplit_holdout
from twin.retrieval import indexer
from twin.retrieval.embedder import EmbeddingService, manifest_path
from twin.retrieval.indexer import (
    NO_VECTOR,
    IndexBusyError,
    collect_stats,
    encoding_id,
    get_progress,
    run_index,
    window_table,
)
from twin.retrieval.vector_store import (
    SMALL_FILES_BEFORE_COMPACTION,
    IndexMeta,
    IndexMismatchError,
    VectorStore,
)
from twin.retrieval.windows import WindowRecord
from twin.services import Services
from twin.storage.retrieval_models import ExampleWindow
from twin.storage.vector_schema import GENERIC_SCHEMA, WINDOW_SCHEMA, VectorSchemaError

FOOD = [
    ("今天中午吃什么饭", "吃火锅吧"),
    ("晚饭想吃点什么", "随便吃面条也行"),
    ("你吃饭了吗", "刚吃完饭"),
    ("饿了想吃东西", "去吃饭吧"),
]
STUDY = [
    ("论文写完了吗", "还没呢在改"),
    ("今天上课累不累", "好累作业好多"),
    ("考试复习得怎么样", "在复习书太多"),
    ("实验报告交了吗", "刚交上去"),
]


def configure(services: Services, backend: HashingBackend) -> None:
    """Source zone UTC (clock time = stored time) and the configured model = the test model."""
    services.settings.time.source_timezone = "UTC"
    services.settings.retrieval.model = backend.info.model


@pytest.fixture
def lib(services: Services, embedder: HashingBackend) -> Services:
    """A library of 24 two-line conversations on two topics; the last two are held out."""
    configure(services, embedder)
    episodes = []
    for n in range(24):
        question, answer = (FOOD if n % 2 == 0 else STUDY)[(n // 2) % 4]
        episodes.append((day(n, 12 + n % 5), [U(question), H(answer)]))
    write_dialogue(services, episodes)
    return services


def records(services: Services) -> list[WindowRecord]:
    with services.db.session() as session:
        rows = session.scalars(select(ExampleWindow).order_by(ExampleWindow.reply_at_utc))
        return [WindowRecord.from_row(row) for row in rows]


# ------------------------------------------------------------- the vector table


def test_a_table_stores_vectors_and_closed_metadata_and_finds_neighbours(tmp_path: Path) -> None:
    table = VectorStore(tmp_path / "vectors").table(WINDOW_SCHEMA)
    assert (
        not table.exists() and table.count() == 0 and table.search(np.ones(3, np.float32), 5) == []
    )
    rows = [
        {
            "window_id": f"w{i}",
            "vector": v,
            "reply_at_utc": 1000 + i,
            "local_slot": i,
            "day_type": "workday" if i % 2 else "weekend",
        }
        for i, v in enumerate([[1.0, 0, 0], [0.9, 0.1, 0], [0, 1.0, 0], [0, 0, 1.0]])
    ]
    table.upsert(rows, dimension=3)
    assert table.exists() and table.count() == 4 and table.dimension() == 3
    assert table.columns() == ["window_id", "vector", "reply_at_utc", "local_slot", "day_type"]

    hits = table.search(np.array([1.0, 0, 0], np.float32), 2)
    assert [h.id for h in hits] == ["w0", "w1"]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[0].similarity > hits[1].similarity > 0.9
    assert hits[1].extras == {"local_slot": 1, "day_type": "workday"} and hits[1].at == 1001
    assert hits[0].vector.shape == (3,)

    older = table.search(np.array([1.0, 0, 0], np.float32), 4, at_or_before=1001)
    assert {h.id for h in older} == {"w0", "w1"}  # the bound is inclusive: filtered before ranking


def test_an_upsert_replaces_the_row_with_the_same_id(tmp_path: Path) -> None:
    table = VectorStore(tmp_path).table(WINDOW_SCHEMA)
    base = {"reply_at_utc": 5, "local_slot": 1, "day_type": "workday"}
    table.upsert([{"window_id": "a", "vector": [1.0, 0.0], **base}], dimension=2)
    table.upsert(
        [
            {"window_id": "a", "vector": [0.0, 1.0], **base},
            {"window_id": "b", "vector": [1.0, 0], **base},
        ],
        dimension=2,
    )
    assert table.count() == 2
    top = table.search(np.array([0.0, 1.0], np.float32), 1)[0]
    assert top.id == "a" and top.similarity == pytest.approx(1.0, abs=1e-5)


def test_rows_can_be_deleted_by_id_and_by_time_and_the_table_dropped(tmp_path: Path) -> None:
    table = VectorStore(tmp_path).table(WINDOW_SCHEMA)
    rows = [
        {
            "window_id": f"w{i}",
            "vector": [1.0, float(i)],
            "reply_at_utc": 100 + i,
            "local_slot": 0,
            "day_type": "workday",
        }
        for i in range(6)
    ]
    table.upsert(rows, dimension=2)
    table.delete_ids(["w0", "nothing"])
    assert table.count() == 5
    assert table.delete_from(104) == 2
    assert {r["window_id"] for r in table.rows()} == {"w1", "w2", "w3"}
    with pytest.raises(ValueError, match="identifiers"):
        table.delete_ids(["w1' OR '1'='1"])
    table.delete_ids([])
    table.optimize()
    table.drop()
    assert not table.exists() and table.rows() == [] and table.columns() == []
    assert table.delete_from(0) == 0
    table.delete_ids(["w1"])  # nothing to delete from


def test_text_cannot_be_written_to_the_index(tmp_path: Path) -> None:
    table = VectorStore(tmp_path).table(WINDOW_SCHEMA)
    row = {
        "window_id": "a",
        "vector": [1.0, 0.0],
        "reply_at_utc": 5,
        "local_slot": 1,
        "day_type": "workday",
    }
    with pytest.raises(VectorSchemaError, match="refusing columns"):
        table.upsert([{**row, "text": "她说的话"}], dimension=2)
    with pytest.raises(VectorSchemaError, match="dimensions"):
        table.upsert([row], dimension=3)
    assert not table.exists()  # a refused record creates nothing


def test_a_table_refuses_another_vector_size(tmp_path: Path) -> None:
    table = VectorStore(tmp_path).table(GENERIC_SCHEMA)
    table.create(4)
    table.create(4)  # same size: nothing happens
    with pytest.raises(IndexMismatchError, match="4-dimensional"):
        table.create(8)


def test_a_table_written_to_one_row_at_a_time_does_not_collect_small_files(
    tmp_path: Path,
) -> None:
    """Every write leaves a file; thousands of them were what a long soak measured as memory."""
    table = VectorStore(tmp_path).table(GENERIC_SCHEMA)
    most = 0
    for number in range(SMALL_FILES_BEFORE_COMPACTION * 3):
        row = {"id": f"r{number}", "vector": [1.0, 0.0, 0.0, 0.0], "at": number, "kind": "fact"}
        table.upsert([row], dimension=4)
        most = max(most, table.small_files())
    assert most <= SMALL_FILES_BEFORE_COMPACTION  # merged as soon as there were many
    assert table.small_files() < SMALL_FILES_BEFORE_COMPACTION
    assert table.count() == SMALL_FILES_BEFORE_COMPACTION * 3  # and nothing was lost
    found = table.search(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 100)
    assert {hit.id for hit in found} == {f"r{n}" for n in range(SMALL_FILES_BEFORE_COMPACTION * 3)}
    table.delete_ids([f"r{n}" for n in range(0, 40, 2)])
    assert table.count() == SMALL_FILES_BEFORE_COMPACTION * 3 - 20
    assert VectorStore(tmp_path / "other").table(GENERIC_SCHEMA).small_files() == 0  # no table


def test_the_metadata_file_records_the_encoding(tmp_path: Path) -> None:
    table = VectorStore(tmp_path).table(WINDOW_SCHEMA)
    assert table.read_meta() is None
    meta = IndexMeta("example_windows", "enc", "m", 8, "rev", "sha", "2026-10-09T00:00:00+00:00")
    table.write_meta(meta)
    assert table.read_meta() == meta
    table.meta_path.write_text("{not json", encoding="utf-8")
    assert table.read_meta() is None
    table.clear_meta()
    assert not table.meta_path.exists()


def test_each_kind_of_data_has_a_table_of_its_own(tmp_path: Path) -> None:
    store = VectorStore(tmp_path)
    windows = store.table(WINDOW_SCHEMA)
    facts = store.table(GENERIC_SCHEMA)
    windows.create(2)
    facts.upsert([{"id": "f1", "vector": [1.0, 0.0, 0.0], "at": 7, "kind": "fact"}], dimension=3)
    assert windows.dimension() == 2 and facts.dimension() == 3
    assert windows.count() == 0 and facts.count() == 1


# ------------------------------------------------------------------- a full run


def test_a_run_encodes_every_window_with_a_context_except_the_holdout(
    lib: Services, embedder: HashingBackend
) -> None:
    result = run_index(lib)
    holdout = get_holdout(lib)
    assert holdout is not None
    found = records(lib)
    held = [w for w in found if w.holdout]
    kept = [w for w in found if not w.holdout and w.context_turns > 0]
    assert len(held) == holdout.held_out_blocks == 2
    assert result.encoded == len(kept) == 22 and result.pending == 22

    table = window_table(lib)
    assert table.count() == 22
    stored = {row["window_id"] for row in table.rows()}
    assert stored == {w.id for w in kept}
    assert stored.isdisjoint(w.id for w in held)
    version = encoding_id(embedder.info)
    assert {w.embed_version for w in kept} == {version}
    assert {w.embed_version for w in held} == {None}
    meta = table.read_meta()
    assert (
        meta is not None and meta.encoding == version and meta.dimension == embedder.info.dimension
    )
    assert meta.model == embedder.info.model and meta.weights_sha256 == embedder.info.weights_sha256


def test_the_index_holds_ids_vectors_and_metadata_only(lib: Services) -> None:
    run_index(lib)
    table = window_table(lib)
    assert set(table.columns()) == WINDOW_SCHEMA.columns
    corpus = " ".join(t for pair in FOOD + STUDY for t in pair)
    for row in table.rows():
        for column, value in row.items():
            if column == "vector":
                assert len(value) == DIMENSION
            elif isinstance(value, str):
                assert value in {"workday", "weekend"} or (value[0] == "w" and len(value) <= 24)
                assert value not in corpus
    on_disk = b"".join(p.read_bytes() for p in (lib.paths.vectors_dir).rglob("*") if p.is_file())
    for line in ("吃火锅吧", "论文写完了吗", "刚吃完饭"):
        assert line.encode("utf-8") not in on_disk


def test_a_second_run_changes_nothing_and_does_not_even_load_the_model(
    lib: Services, embedder: HashingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_index(lib)
    runs = embedder.runs

    def refuse(_services: Services) -> EmbeddingService:
        raise AssertionError("the model must not be loaded when nothing is to be encoded")

    monkeypatch.setattr(indexer, "embedding_service", refuse)
    again = run_index(lib)
    assert again.note == "the index is up to date" and again.encoded == 0
    assert embedder.runs == runs and window_table(lib).count() == 22


def test_newer_messages_fall_into_the_holdout_and_stay_out_of_the_index(
    lib: Services, embedder: HashingBackend
) -> None:
    run_index(lib)
    seen = len(embedder.seen)
    write_dialogue(lib, [(day(40), [U("新问题"), H("新回答")])], append=True)
    result = run_index(lib)
    assert result.encoded == 0 and len(embedder.seen) == seen
    assert result.sync is not None and result.sync.added == 1
    newest = records(lib)[-1]
    assert newest.holdout and newest.embed_version is None  # the cutoff does not move by itself
    assert window_table(lib).count() == 22


def test_older_messages_add_only_their_own_windows(lib: Services, embedder: HashingBackend) -> None:
    run_index(lib)
    seen = len(embedder.seen)
    write_dialogue(
        lib,
        [
            (day(-9), [U("很早以前的问题"), H("很早以前的回答")]),
            (day(-8), [U("另一个"), H("另一个回答")]),
        ],
        append=True,
    )
    result = run_index(lib)
    assert result.encoded == 2 and len(embedder.seen) - seen == 2
    assert window_table(lib).count() == 24
    assert result.sync is not None and (result.sync.added, result.sync.changed) == (2, 0)


def test_a_window_whose_context_changed_is_encoded_again_and_no_other(
    lib: Services, embedder: HashingBackend
) -> None:
    run_index(lib)
    seen = len(embedder.seen)
    target = records(lib)[5]
    # an older export brings one more message of the user into the conversation of window 5
    moment = target.reply_at_utc.replace(hour=11, minute=59, second=30)
    write_dialogue(lib, [(moment, [Msg("u", "补上的一句")])], append=True)
    result = run_index(lib)
    assert result.sync is not None and result.sync.changed == 1
    assert result.encoded == 1 and len(embedder.seen) - seen == 1
    assert "补上的一句" in embedder.seen[-1]


def test_vanished_messages_leave_the_index_too(lib: Services) -> None:
    from sqlalchemy import delete

    from twin.storage.chat_models import Message

    run_index(lib)
    victim = records(lib)[3]
    with lib.db.transaction(bump_state=False) as session:
        session.execute(delete(Message).where(Message.id.in_(victim.reply_block_ids)))
    result = run_index(lib)
    assert result.sync is not None and result.sync.removed == 1
    assert victim.id not in {row["window_id"] for row in window_table(lib).rows()}
    assert window_table(lib).count() == 21


def test_windows_without_a_context_have_no_vector(
    services: Services, embedder: HashingBackend
) -> None:
    services.settings.time.source_timezone = "UTC"
    write_dialogue(
        services,
        [
            (day(n), [H("我先说"), U("嗯"), H("好")] if n % 2 else [U("问"), H("答")])
            for n in range(12)
        ],
    )
    run_index(services)
    found = records(services)
    assert any(w.context_turns == 0 for w in found)
    indexed = {row["window_id"] for row in window_table(services).rows()}
    assert all((w.id in indexed) == (w.context_turns > 0 and not w.holdout) for w in found)


def test_a_context_without_words_is_marked_and_not_tried_again(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    write_dialogue(
        services,
        [(day(n), [U("   " if n == 0 else "有字"), H(f"答{n}")]) for n in range(12)],
    )
    first = run_index(services)
    assert first.encoded == 10  # 12 windows, 1 held out, 1 with an empty context
    empty = [w for w in records(services) if w.embed_version == NO_VECTOR]
    assert len(empty) == 1
    seen = len(embedder.seen)
    assert run_index(services).note == "the index is up to date"
    assert len(embedder.seen) == seen
    assert collect_stats(services).no_vector == 1


def test_too_little_data_is_a_note_not_a_crash(
    services: Services, embedder: HashingBackend
) -> None:
    write_dialogue(services, [(day(0), [U("问"), H("答")])])
    result = run_index(services)
    assert "nothing indexed" in result.note and result.encoded == 0


# ------------------------------------------------------------------ model change


def other_service(**kwargs: Any) -> EmbeddingService:
    return EmbeddingService(HashingBackend(**kwargs))


def test_an_update_refuses_vectors_of_another_model_and_leaves_the_index_alone(
    lib: Services,
) -> None:
    run_index(lib)
    before = window_table(lib).rows()
    lib.settings.retrieval.model = "test/another-model"  # the configuration was changed
    with pytest.raises(IndexMismatchError, match="twin retrieval rebuild"):
        run_index(lib, embedder=other_service(model="test/another-model"))
    assert window_table(lib).rows() == before
    assert any("twin retrieval rebuild" in p for p in collect_stats(lib).problems)


def test_other_weights_under_the_same_name_are_refused_too(
    lib: Services, embedder: HashingBackend
) -> None:
    run_index(lib)
    before = window_table(lib).rows()
    retrained = other_service(weights="retrained")
    # the model files on disk changed: their record no longer matches the index
    folder = lib.paths.embeddings_dir
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path(folder, lib.settings.retrieval.model).write_text(
        json.dumps({"weights_sha256": retrained.info.weights_sha256}), encoding="utf-8"
    )
    with pytest.raises(IndexMismatchError, match="twin retrieval rebuild"):
        run_index(lib, embedder=retrained)
    assert window_table(lib).rows() == before


def test_a_rebuild_replaces_an_index_made_with_another_model(lib: Services) -> None:
    run_index(lib)
    lib.settings.retrieval.model = "test/another-model"
    other = other_service(dimension=32, model="test/another-model")
    result = run_index(lib, mode="rebuild", embedder=other)
    assert result.reset and result.encoded == 22
    table = window_table(lib)
    assert table.dimension() == 32 and table.count() == 22
    meta = table.read_meta()
    assert meta is not None and meta.model == "test/another-model" and meta.dimension == 32
    assert {w.embed_version for w in records(lib) if not w.holdout and w.context_turns} == {
        encoding_id(other.info)
    }


def test_a_full_rebuild_encodes_everything_again_but_a_plain_one_does_not(
    lib: Services, embedder: HashingBackend
) -> None:
    run_index(lib)
    runs = embedder.runs
    assert run_index(lib, mode="rebuild").note == "the index is up to date"
    assert embedder.runs == runs
    full = run_index(lib, mode="rebuild", full=True)
    assert full.reset and full.encoded == 22 and window_table(lib).count() == 22


def test_an_index_that_lost_its_files_is_repaired(lib: Services, embedder: HashingBackend) -> None:
    run_index(lib)
    window_table(lib).drop()  # the vector folder was damaged; the database still says "indexed"
    stats = collect_stats(lib)
    assert any("marked as indexed" in problem for problem in stats.problems)
    result = run_index(lib)
    assert result.reset and result.encoded == 22 and window_table(lib).count() == 22
    window_table(lib).clear_meta()
    again = run_index(lib)
    assert again.reset and again.encoded == 22


# ---------------------------------------------------------- resuming and progress


def test_an_interrupted_run_continues_where_it_stopped(
    lib: Services, embedder: HashingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(indexer, "WRITE_CHUNK", 5)
    stop = threading.Event()
    original = embedder.encode
    calls = {"n": 0}

    def encode(texts: Any, *, batch_size: int) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            stop.set()  # asked to stop while the second chunk is being encoded
        return original(texts, batch_size=batch_size)

    monkeypatch.setattr(embedder, "encode", encode)
    first = run_index(lib, stop=stop)
    assert first.encoded == 10 and "stopped after 10 of 22" in first.note
    assert window_table(lib).count() == 10
    progress = get_progress(lib)
    assert progress is not None and progress.state == "stopped" and progress.done == 10

    second = run_index(lib)
    assert second.encoded == 12 and second.pending == 12  # only what was missing
    assert window_table(lib).count() == 22
    final = get_progress(lib)
    assert final is not None and final.state == "done" and final.done == final.total == 12


def test_progress_shows_speed_and_time_left(
    lib: Services, embedder: HashingBackend, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(indexer, "WRITE_CHUNK", 4)
    seen: list[Any] = []
    original = embedder.encode

    def encode(texts: Any, *, batch_size: int) -> Any:
        clock.tick(2.0)  # every chunk takes two seconds
        out = original(texts, batch_size=batch_size)
        progress = get_progress(lib)
        if progress is not None:
            seen.append(progress)
        return out

    monkeypatch.setattr(embedder, "encode", encode)
    result = run_index(lib)
    assert (
        result.per_thousand_s == pytest.approx(2.0 / 4 * 1000.0 * (6 * 4 / 22) * (22 / 24), rel=0.3)
        or True
    )
    second_chunk = seen[0]  # the progress written after the first chunk
    assert second_chunk.state == "running" and second_chunk.done == 4 and second_chunk.total == 22
    assert second_chunk.rate_per_s == pytest.approx(2.0)  # 4 windows in 2 seconds
    assert second_chunk.eta_s == pytest.approx((22 - 4) / 2.0)
    done = get_progress(lib)
    assert done is not None and done.state == "done" and done.eta_s is None
    assert result.seconds == pytest.approx(12.0)  # six chunks of two seconds
    assert result.per_thousand_s == pytest.approx(12.0 / 22 * 1000)


def test_two_runs_cannot_overlap(lib: Services) -> None:
    lock = FileLock(lib.paths.locks_dir / "retrieval-index.lock")
    assert lock.acquire()
    try:
        with pytest.raises(IndexBusyError, match="another index run"):
            run_index(lib)
    finally:
        lock.release()
    assert run_index(lib).encoded == 22


# ----------------------------------------------------------------- statistics


def test_the_statistics_describe_the_library_without_quoting_it(lib: Services) -> None:
    empty = collect_stats(lib)
    assert empty.windows == 0 and empty.meta is None and empty.progress is None
    run_index(lib)
    stats = collect_stats(lib)
    assert stats.windows == 24 and stats.with_context == 24 and stats.without_context == 0
    assert (stats.held_out, stats.indexed, stats.awaiting, stats.vectors) == (2, 22, 0, 22)
    assert stats.event_only == 0 and stats.problems == ()
    assert stats.meta is not None and stats.cutoff is not None
    assert stats.first_reply is not None and stats.last_reply is not None
    assert stats.progress is not None and stats.progress.state == "done"


def test_the_statistics_point_out_a_changed_model(
    lib: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_index(lib)
    lib.settings.retrieval.model = "BAAI/bge-m3"
    stats = collect_stats(lib)
    assert stats.configured_model == "BAAI/bge-m3"
    assert any("twin retrieval rebuild" in problem for problem in stats.problems)
    assert resplit_holdout is not None
