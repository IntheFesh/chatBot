"""``eval_runs`` and ``eval_items`` through ``EvalStore`` (R-STO-006, R-EVAL-001)."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from tests.support.clock import ManualClock
from twin.eval import api
from twin.eval.store import EvalStore, EvalStoreError, NewItem
from twin.services import Services
from twin.storage.eval_models import EvalItem

AT = datetime(2026, 9, 1, 12, tzinfo=UTC)


@pytest.fixture
def store(services: Services) -> EvalStore:
    return EvalStore(services.db, services.clock)


def item(key: str, backend: str = "deepseek", **fields: object) -> NewItem:
    return NewItem(key, backend, AT, {"text": f"正文{key}"}, **fields)  # type: ignore[arg-type]


def test_a_run_is_found_by_its_id_or_by_the_start_of_it(
    store: EvalStore, clock: ManualClock
) -> None:
    first = store.create_run("blind", mode="holdout", backends=["deepseek"], params={"n": 5})
    clock.tick(1)
    second = store.create_run("memory", mode="live")
    # how much of the id the two runs share depends on the process-wide id generator (a clock that
    # went backwards makes it count on from the last id), so the prefixes are made from the ids
    shared = len(os.path.commonprefix([first.id, second.id]))
    assert store.get_run(first.id) == first
    assert store.get_run(first.id[: shared + 1]).id == first.id  # the shortest start of it
    assert first.params == {"n": 5} and first.backends == ("deepseek",) and not first.finished
    with pytest.raises(EvalStoreError, match="more than one run"):
        store.get_run(first.id[:shared])  # what the two ids share starts both of them
    with pytest.raises(EvalStoreError, match="no evaluation run"):
        store.get_run("nothere")
    assert [r.id for r in store.list_runs()] == [second.id, first.id]  # newest first
    assert [r.id for r in store.list_runs("blind")] == [first.id]
    assert store.list_runs(limit=1)[0].id == second.id


def test_the_latest_run_can_be_asked_for_by_kind_status_backend_and_milestone(
    store: EvalStore, clock: ManualClock
) -> None:
    older = store.create_run("blind", backends=["deepseek"], status="done")
    clock.tick(1)
    newer = store.create_run("blind", backends=["style"], status="running")
    clock.tick(1)
    gate = store.create_run("gate", status="done", milestone="M1", verdict="passed")
    assert store.latest_run("blind") == newer
    assert store.latest_run("blind", status="done") == older
    assert store.latest_run("blind", backend="deepseek") == older
    assert store.latest_run("blind", backend="hybrid") is None
    assert store.latest_run("gate", milestone="M1") == gate
    assert store.latest_run("gate", milestone="M2") is None
    assert store.latest_run("memory") is None


def test_updating_a_run_merges_its_numbers_and_stamps_the_finish(
    store: EvalStore, clock: ManualClock
) -> None:
    run = store.create_run("blind", params={"a": 1}, summary={"x": 1})
    updated = store.update_run(
        run.id, status="done", verdict="passed", batch_ids=["b1"], params={"b": 2}, summary={"y": 2}
    )
    assert updated.params == {"a": 1, "b": 2} and updated.summary == {"x": 1, "y": 2}
    assert updated.verdict == "passed" and updated.batch_ids == ("b1",)
    assert updated.finished_at == clock.now_utc() and updated.finished
    assert store.update_run(run.id, status="running").finished_at is None  # reopened
    with pytest.raises(EvalStoreError):
        store.update_run("nothere", status="done")


def test_items_are_numbered_filtered_and_their_text_is_sealed(
    store: EvalStore, services: Services
) -> None:
    run = store.create_run("blind", backends=["deepseek", "style"])
    assert store.add_items(run.id, [item("a"), item("b", "style")]) == 2
    assert store.add_items(run.id, [item("c")]) == 1  # numbered after the last one
    found = store.items(run.id)
    assert [(i.seq, i.sample_key, i.backend) for i in found] == [
        (0, "a", "deepseek"),
        (1, "b", "style"),
        (2, "c", "deepseek"),
    ]
    assert found[0].payload == {"text": "正文a"} and found[0].status == "pending"
    assert store.items(run.id, backend="style")[0].sample_key == "b"
    assert store.items(run.id, status="generated") == []
    assert {i.sample_key for i in store.items(run.id, status=["pending", "failed"])} == {
        "a",
        "b",
        "c",
    }
    assert store.items(run.id, with_payload=False)[0].payload == {}
    with services.db.session() as session:  # the text is not readable in the table
        raw = bytes(session.get(EvalItem, found[0].id).payload_ct)  # type: ignore[union-attr,arg-type]
    assert "正文".encode() not in raw
    with pytest.raises(EvalStoreError):
        store.add_items("nothere", [item("z")])
    assert store.counts(run.id)["pending"] == 3


def test_an_item_moves_from_pending_to_generated_to_judged_and_keeps_its_cost(
    store: EvalStore, clock: ManualClock
) -> None:
    run = store.create_run("blind")
    store.add_items(run.id, [item("a"), item("b"), item("c"), item("d")])
    a, b, c, d = (i.id for i in store.items(run.id))
    generated = store.save_generated(a, {"bot": {"lines": []}}, cost_usd=0.002)
    assert generated.status == "generated" and generated.payload["bot"] == {"lines": []}
    assert generated.payload["text"] == "正文a"  # the earlier keys stay
    assert store.save_generated(a, {"note": 1}, cost_usd=0.003).cost_usd == pytest.approx(0.005)
    judged = store.judge(a, "correct", score=1.0, payload={"chosen": "left"})
    assert (judged.status, judged.outcome, judged.score) == ("judged", "correct", 1.0)
    assert (
        judged.valid and judged.judged_at == clock.now_utc() and judged.payload["chosen"] == "left"
    )
    skipped = store.judge(b, None, score=1.0)  # a score on a skip is dropped
    assert (skipped.status, skipped.outcome, skipped.score) == ("skipped", None, None)
    assert not skipped.valid
    failed = store.mark_failed(c, "no_reply")
    assert failed.status == "failed" and failed.payload["failure"] == "no_reply"
    auto = store.save_auto(d, {"answer": "x"}, auto_outcome="partial", cost_usd=0.01)
    assert (auto.status, auto.auto_outcome, auto.outcome) == ("generated", "partial", None)
    assert store.counts(run.id) == {"judged": 1, "skipped": 1, "failed": 1, "generated": 1}
    for call in (
        lambda: store.item("nothere"),
        lambda: store.judge("nothere", "wrong"),
        lambda: store.mark_failed("nothere", "x"),
        lambda: store.save_generated("nothere", {}, cost_usd=0.0),
        lambda: store.save_auto("nothere", {}, auto_outcome=None, cost_usd=0.0),
    ):
        with pytest.raises(EvalStoreError):
            call()


def test_a_context_counts_as_used_unless_its_run_was_cancelled_unseen(store: EvalStore) -> None:
    done = store.create_run("blind", status="done")
    planned = store.create_run("blind", status="planned")
    cancelled = store.create_run("blind", status="cancelled")
    seen = store.create_run("blind", status="cancelled")
    memory = store.create_run("memory", status="done")
    for run, key in (
        (done, "k-done"),
        (planned, "k-planned"),
        (cancelled, "k-cancelled"),
        (seen, "k-seen"),
        (memory, "k-memory"),
    ):
        store.add_items(run.id, [item(key)])
    store.judge(store.items(seen.id)[0].id, "wrong", score=0.0)  # a user saw it, whatever happened
    assert store.used_sample_keys("blind") == {"k-done", "k-planned", "k-seen"}
    assert store.used_sample_keys("blind", except_run=done.id) == {"k-planned", "k-seen"}
    assert store.used_sample_keys("memory") == {"k-memory"}


def test_the_api_module_exposes_what_later_rounds_import() -> None:
    assert len(api.__all__) == len(set(api.__all__))
    for name in api.__all__:
        assert hasattr(api, name), name
    assert {
        "EvalSandbox",
        "InMemoryChannel",
        "gate_judge",
        "plan_blind",
        "render_candidate",
    } <= set(api.__all__)
