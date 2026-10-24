"""``ProbeHold``: what the sender asks of the probe, and how it fails safe (R17-4, D-684, D-685)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.clock import ManualClock
from twin.channel.probe.hold import REASON_REFUSED, REASON_URGENT, ProbeHold
from twin.channel.probe.model import (
    Attempt,
    AttemptPhase,
    PlanStatus,
    ProbePlan,
    StepId,
    StepStatus,
)
from twin.channel.probe.store import ProbeStore
from twin.storage.db import Database


@pytest.fixture
def store(db: Database, clock: ManualClock) -> ProbeStore:
    return ProbeStore(db, clock)


def open_attempt(store: ProbeStore) -> None:
    def change(plan: ProbePlan) -> None:
        step = plan.step(StepId.COUNT)
        step.status = StepStatus.ACTIVE
        step.attempts.append(
            Attempt(
                n=1,
                phase=AttemptPhase.RUNNING,
                armed_at=plan.created_at,
                baseline_inbound_at=None,
            )
        )

    store.update(change)


def test_the_hold_follows_the_stored_plan(store: ProbeStore) -> None:
    hold = ProbeHold(store)
    assert hold.active() is False
    store.create()
    assert hold.active() is True
    store.finish(PlanStatus.STOPPED, "done")
    assert hold.active() is False


def test_a_refusal_is_audited_with_the_run_and_the_step_but_never_the_words(
    store: ProbeStore, clock: ManualClock
) -> None:
    plan = store.create()
    open_attempt(store)
    ProbeHold(store).refused("text", "我今天好累", clock.now_utc())
    [entry] = store.audit()
    assert (entry["decision"], entry["why"]) == ("refused", REASON_REFUSED)
    assert (entry["run_id"], entry["step"]) == (plan.run_id, "count")
    assert entry["text_chars"] == 5 and entry["text_head"] is None


def test_an_urgent_send_is_audited_and_voids_the_attempt_under_way(
    store: ProbeStore, clock: ManualClock
) -> None:
    store.create()
    open_attempt(store)
    ProbeHold(store).urgent_sent("text", "你不是一个人", clock.now_utc())
    [entry] = store.audit()
    assert (entry["decision"], entry["why"]) == ("urgent", REASON_URGENT)
    plan = store.load()
    assert plan is not None and plan.steps[0].attempts[0].outcome == "void"


def test_a_plan_that_cannot_be_read_never_stands_in_the_way_of_a_safety_message(
    store: ProbeStore, clock: ManualClock
) -> None:
    """A broken plan: the audit has no run id, the attempt cannot be voided, nothing raises."""
    store.create()
    store.state.put("probe.plan", {"garbage": True})
    hold = ProbeHold(store)
    hold.urgent_sent("text", "你不是一个人", clock.now_utc())
    [entry] = store.audit()
    assert entry["decision"] == "urgent" and entry["run_id"] is None and entry["step"] is None


def test_a_failing_audit_is_logged_and_the_send_goes_on(
    store: ProbeStore, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create()

    def broken(entry: dict[str, Any]) -> None:
        raise OSError("the database is locked")

    monkeypatch.setattr(store, "add_audit", broken)
    hold = ProbeHold(store)
    hold.refused("text", "嗨", clock.now_utc())  # no exception: the audit is a record only
    hold.urgent_sent("text", "嗨", clock.now_utc())


def test_a_failing_void_is_logged_and_the_send_goes_on(
    store: ProbeStore, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create()

    def broken(reason: str) -> bool:
        raise OSError("the database is locked")

    monkeypatch.setattr(store, "void_current_attempt", broken)
    ProbeHold(store).urgent_sent("text", "嗨", clock.now_utc())
    assert [e["decision"] for e in store.audit()] == ["urgent"]
