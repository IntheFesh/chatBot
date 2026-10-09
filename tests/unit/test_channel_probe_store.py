"""The stored probe plan: atomic changes, answers, the final summary (R-CH-009)."""

from __future__ import annotations

import pytest

from tests.support.clock import ManualClock
from twin.channel.probe.model import (
    Action,
    ActionKind,
    ActionStatus,
    Attempt,
    AttemptPhase,
    PlanStatus,
    ProbeOptions,
    ProbePlan,
    Question,
    QuestionKind,
    StepId,
    StepStatus,
)
from twin.channel.probe.store import (
    AnswerRejected,
    NoProbePlan,
    ProbeAlreadyRunning,
    ProbeStore,
    validate_answer,
)
from twin.channel.probe.summary import VERDICT_UNDETERMINED, load_channel_probe_summary
from twin.storage.db import Database


@pytest.fixture
def store(db: Database, clock: ManualClock) -> ProbeStore:
    return ProbeStore(db, clock)


def ask(store: ProbeStore, question: Question) -> None:
    def change(plan: ProbePlan) -> None:
        step = plan.step(StepId.COUNT)
        step.status = StepStatus.ACTIVE
        step.attempts.append(
            Attempt(
                n=1,
                phase=AttemptPhase.RUNNING,
                armed_at=plan.created_at,
                baseline_inbound_at=None,
                actions=[
                    Action(
                        id="ask:count",
                        kind=ActionKind.ASK,
                        label="ask",
                        status=ActionStatus.ACTIVE,
                        question=question,
                    )
                ],
            )
        )

    store.update(change)


def count_question(maximum: int | None = 8) -> Question:
    return Question("q-count", QuestionKind.COUNT, "how many?", maximum=maximum)


# ------------------------------------------------------------------- the plan


def test_there_is_no_plan_until_one_is_created(store: ProbeStore) -> None:
    assert store.load() is None
    with pytest.raises(NoProbePlan):
        store.update(lambda plan: None)
    with pytest.raises(NoProbePlan):
        store.answer("x", "1")


def test_a_created_plan_is_stored_and_named_by_time_and_a_random_suffix(
    store: ProbeStore, clock: ManualClock
) -> None:
    plan = store.create()
    assert plan.status is PlanStatus.RUNNING and plan.run_id.startswith("20261009T120000Z-")
    assert store.load() == plan
    assert [step.id for step in plan.steps] == [StepId.COUNT, StepId.MEDIA, StepId.WINDOW]
    assert plan.created_at == clock.now_utc()


def test_a_second_plan_is_refused_while_one_runs_and_allowed_afterwards(store: ProbeStore) -> None:
    first = store.create()
    with pytest.raises(ProbeAlreadyRunning, match=first.run_id):
        store.create()
    store.finish(PlanStatus.STOPPED, "enough")
    second = store.create(ProbeOptions(empty_token_experiment=True))
    assert second.run_id != first.run_id and len(second.steps) == 4
    loaded = store.load()
    assert loaded is not None and loaded.run_id == second.run_id


def test_a_change_that_raises_leaves_the_stored_plan_untouched(store: ProbeStore) -> None:
    store.create()

    def broken(plan: ProbePlan) -> None:
        plan.notice = "half done"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        store.update(broken)
    loaded = store.load()
    assert loaded is not None and loaded.notice is None


def test_a_change_sees_the_latest_stored_plan(store: ProbeStore) -> None:
    store.create()
    store.update(lambda plan: setattr(plan, "notice", "first"))
    seen: list[str | None] = []
    store.update(lambda plan: seen.append(plan.notice))
    assert seen == ["first"]


# ------------------------------------------------------------------- finishing


def test_finishing_stores_the_summary_with_the_plan_in_one_step(
    store: ProbeStore, db: Database
) -> None:
    store.create()
    with db.session() as session:
        assert load_channel_probe_summary(session) is None
    plan = store.finish(PlanStatus.STOPPED, "stopped by the user")
    assert plan.status is PlanStatus.STOPPED and plan.stop_reason == "stopped by the user"
    assert plan.finished_at is not None and plan.notice is None
    with db.session() as session:
        stored = load_channel_probe_summary(session)
    assert stored is not None and stored.run_id == plan.run_id
    assert stored.status == "stopped" and not stored.complete
    assert stored.verdict == VERDICT_UNDETERMINED and stored.n_messages is None


def test_finishing_twice_changes_nothing(store: ProbeStore) -> None:
    store.create()
    first = store.finish(PlanStatus.COMPLETED, None)
    second = store.finish(PlanStatus.STOPPED, "late")
    assert second.status is PlanStatus.COMPLETED and second.finished_at == first.finished_at
    assert second.stop_reason is None


def test_finishing_needs_a_final_status(store: ProbeStore) -> None:
    store.create()
    with pytest.raises(ValueError, match="final status"):
        store.finish(PlanStatus.RUNNING, None)


# ------------------------------------------------------------------ questions


def test_a_pending_question_is_found_and_an_answer_is_recorded(
    store: ProbeStore, clock: ManualClock
) -> None:
    store.create()
    ask(store, count_question())
    plan = store.load()
    assert plan is not None
    [pending] = store.pending_questions(plan)
    assert pending.id == "q-count" and not pending.answered
    answered = store.answer("q-count", " 5 ")
    assert answered.answer == "5" and answered.answered_at == clock.now_utc()
    loaded = store.load()
    assert loaded is not None and store.pending_questions(loaded) == []
    assert any("q-count" in event.text for event in loaded.events)


def test_an_unknown_or_already_answered_question_is_rejected(store: ProbeStore) -> None:
    store.create()
    ask(store, count_question())
    with pytest.raises(AnswerRejected, match="no pending question"):
        store.answer("nope", "1")
    store.answer("q-count", "1")
    with pytest.raises(AnswerRejected, match="no pending question"):
        store.answer("q-count", "2")


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("0", True),
        ("8", True),
        ("9", False),
        ("-1", False),
        ("three", False),
        ("", False),
        ("4.5", False),
    ],
)
def test_a_count_answer_is_a_whole_number_within_what_was_sent(text: str, ok: bool) -> None:
    question = count_question(8)
    if ok:
        assert validate_answer(question, text) == text
    else:
        with pytest.raises(AnswerRejected):
            validate_answer(question, text)


def test_a_count_answer_without_a_maximum_only_needs_to_be_a_number() -> None:
    assert validate_answer(count_question(None), "99") == "99"


def test_a_choice_answer_matches_ignoring_case_and_returns_the_canonical_word() -> None:
    question = Question("q", QuestionKind.CHOICE, "?", choices=["moving", "still", "missing"])
    assert validate_answer(question, " MOVING ") == "moving"
    with pytest.raises(AnswerRejected, match="moving, still, missing"):
        validate_answer(question, "yes")


def test_a_ready_answer_is_anything_even_nothing() -> None:
    question = Question("q", QuestionKind.READY, "ready?")
    assert validate_answer(question, "") == "ready"
    assert validate_answer(question, "whatever") == "ready"
