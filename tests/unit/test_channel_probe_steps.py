"""What each probe step sends, asks and concludes, as pure functions (R-CH-009)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from twin.channel.base import TEST_PREFIX
from twin.channel.probe import steps
from twin.channel.probe.model import (
    Action,
    ActionKind,
    ActionStatus,
    Attempt,
    AttemptPhase,
    ProbeOptions,
    ProbePlan,
    SendRecord,
    StepId,
    StepState,
    new_plan,
)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def plan_with(n: int | None, **options: object) -> ProbePlan:
    plan = new_plan("run", NOW, ProbeOptions(**options))  # type: ignore[arg-type]
    if n is not None:
        plan.step(StepId.COUNT).data = {"n": n}
    return plan


def attempt_for(items: list[str] | None = None, budget: int | None = None) -> Attempt:
    return Attempt(
        n=1,
        phase=AttemptPhase.RUNNING,
        armed_at=NOW,
        baseline_inbound_at=None,
        inbound_at=NOW,
        started_at=NOW + timedelta(seconds=3),
        items=items or [],
        budget=budget,
    )


def accepted(at: datetime = NOW, hours: float = 0.0) -> SendRecord:
    return SendRecord(ok=True, outcome="ok", at=at, elapsed_h=hours)


def rejected(code: int = -2) -> SendRecord:
    return SendRecord(ok=False, outcome="window_rejected", reason="platform_rejected", code=code)


def done(action: Action, record: SendRecord) -> None:
    action.status = ActionStatus.DONE if record.ok else ActionStatus.FAILED
    action.send = record


# ----------------------------------------------------------------- activation


def test_the_count_always_starts_and_has_no_budget() -> None:
    plan = plan_with(None)
    setup = steps.activation(plan, plan.step(StepId.COUNT))
    assert isinstance(setup, steps.AttemptSetup) and setup.budget is None and not setup.from_inbound


def test_later_steps_cannot_start_before_the_count_is_known() -> None:
    plan = plan_with(None)
    for step_id in (StepId.MEDIA, StepId.WINDOW):
        reason = steps.activation(plan, plan.step(step_id))
        assert isinstance(reason, str) and "did not measure" in reason


@pytest.mark.parametrize("n", [0, 1])
def test_a_count_below_two_leaves_no_room_and_says_what_arrived(n: int) -> None:
    plan = plan_with(n)
    for step_id in (StepId.MEDIA, StepId.WINDOW):
        reason = steps.activation(plan, plan.step(step_id))
        assert isinstance(reason, str) and f"only {n} message(s) reached the phone" in reason


def test_media_takes_as_many_pictures_as_the_budget_allows_in_order() -> None:
    plan = plan_with(3)  # budget 2
    media = plan.step(StepId.MEDIA)
    setup = steps.activation(plan, media)
    assert isinstance(setup, steps.AttemptSetup)
    assert (setup.budget, setup.items, setup.from_inbound) == (2, ["jpg", "png"], False)
    assert media.remaining == ["jpg", "png", "gif"]
    media.remaining = ["gif"]  # the first attempt is done
    media.attempts.append(attempt_for(["jpg", "png"]))
    again = steps.activation(plan, media)
    assert isinstance(again, steps.AttemptSetup) and again.items == ["gif"]


def test_the_window_keeps_the_latest_points_that_fit_the_budget() -> None:
    plan = plan_with(8)  # budget 7: all six
    setup = steps.activation(plan, plan.step(StepId.WINDOW))
    assert isinstance(setup, steps.AttemptSetup) and setup.from_inbound
    assert setup.items == ["1", "6", "12", "20", "23", "25"]
    narrow = plan_with(4)  # budget 3
    setup = steps.activation(narrow, narrow.step(StepId.WINDOW))
    assert isinstance(setup, steps.AttemptSetup) and setup.items == ["20", "23", "25"]
    assert steps.dropped_hours(narrow, setup.items) == [1.0, 6.0, 12.0]


def test_the_experiment_sends_a_single_message() -> None:
    plan = plan_with(8, empty_token_experiment=True)
    setup = steps.activation(plan, plan.step(StepId.EMPTY_TOKEN))
    assert isinstance(setup, steps.AttemptSetup) and setup.budget == 1


# ------------------------------------------------------------------ the actions


def test_the_count_sends_fifteen_texts_two_minutes_apart_then_asks() -> None:
    plan = plan_with(None)
    actions = steps.build_actions(plan, plan.step(StepId.COUNT), attempt_for())
    sends, ask = actions[:-1], actions[-1]
    assert len(sends) == 15 and ask.kind is ActionKind.ASK and ask.id == "ask:count"
    assert [a.due_offset_s for a in sends[:3]] == [0.0, 120.0, 240.0]
    assert all(a.text and a.text.startswith(TEST_PREFIX) for a in sends)
    assert sends[0].text is not None and "1/15" in sends[0].text


def test_the_window_sends_at_hour_offsets_and_warns_in_every_message() -> None:
    plan = plan_with(8)
    attempt = attempt_for(["1", "6", "25"])
    actions = steps.build_actions(plan, plan.step(StepId.WINDOW), attempt)
    sends = [a for a in actions if a.kind is ActionKind.SEND_TEXT]
    assert [a.due_offset_s for a in sends] == [3600.0, 21600.0, 90000.0]
    assert all(a.text and "不要回复" in a.text and a.text.startswith(TEST_PREFIX) for a in sends)
    assert actions[-1].id == "ask:window"


def test_only_the_last_media_attempt_tests_the_typing() -> None:
    plan = plan_with(8)
    media = plan.step(StepId.MEDIA)
    media.remaining = ["jpg", "png", "gif"]
    first = steps.build_actions(plan, media, attempt_for(["jpg", "png"]))
    assert [a.id for a in first] == ["send:jpg", "send:png", "ask:jpg", "ask:png"]
    last = steps.build_actions(plan, media, attempt_for(["jpg", "png", "gif"]))
    assert [a.id for a in last][-3:] == ["ask:typing_ready", "typing", "ask:typing_seen"]
    assert [a.due_offset_s for a in last[:3]] == [0.0, 5.0, 10.0]  # the pictures are spaced out


def test_the_experiment_text_says_it_has_no_context_token() -> None:
    plan = plan_with(8, empty_token_experiment=True)
    actions = steps.build_actions(plan, plan.step(StepId.EMPTY_TOKEN), attempt_for())
    assert actions[0].empty_token and actions[0].text and actions[0].text.startswith(TEST_PREFIX)


# ---------------------------------------------------------------- the messages


def test_the_announcement_names_the_step_and_for_the_window_the_silence() -> None:
    plan = plan_with(8)
    first = attempt_for()
    count_text = steps.announce_text(plan, plan.step(StepId.COUNT), first)
    assert (
        count_text.startswith(TEST_PREFIX) and "第1步" in count_text and "25小时" not in count_text
    )
    window_text = steps.announce_text(plan, plan.step(StepId.WINDOW), first)
    assert "第3步" in window_text and "25小时" in window_text and "作废" in window_text
    redo = attempt_for()
    redo.n = 2
    assert "这是重做" in steps.announce_text(plan, plan.step(StepId.COUNT), redo)
    notice = steps.terminal_notice(plan, plan.step(StepId.WINDOW), redo)
    assert "Step 3/3" in notice and "25 hours" in notice and "attempt 2" in notice


# ---------------------------------------------------------------------- questions


def question(plan: ProbePlan, step: StepState, attempt: Attempt, action_id: str) -> object:
    action = next(a for a in attempt.actions if a.id == action_id)
    return steps.question_for(plan, step, attempt, action)


def test_nothing_is_asked_when_nothing_was_accepted() -> None:
    plan = plan_with(None)
    step = plan.step(StepId.COUNT)
    attempt = attempt_for()
    attempt.actions = steps.build_actions(plan, step, attempt)
    assert question(plan, step, attempt, "ask:count") is None


def test_the_count_question_is_bounded_by_what_the_server_accepted() -> None:
    plan = plan_with(None)
    step = plan.step(StepId.COUNT)
    attempt = attempt_for()
    attempt.actions = steps.build_actions(plan, step, attempt)
    for action in attempt.actions[:4]:
        done(action, accepted())
    asked = question(plan, step, attempt, "ask:count")
    assert asked is not None and asked.maximum == 4  # type: ignore[attr-defined]
    assert asked.id == "count-1-count" and "条数测试" in asked.prompt  # type: ignore[attr-defined]


def test_the_experiment_question_is_a_yes_or_no_count() -> None:
    plan = plan_with(8, empty_token_experiment=True)
    step = plan.step(StepId.EMPTY_TOKEN)
    attempt = attempt_for()
    attempt.actions = steps.build_actions(plan, step, attempt)
    assert question(plan, step, attempt, "ask:empty") is None  # nothing was accepted
    done(attempt.actions[0], accepted())
    asked = question(plan, step, attempt, "ask:empty")
    assert asked is not None and asked.maximum == 1  # type: ignore[attr-defined]


def test_picture_questions_exist_only_for_accepted_pictures_and_the_gif_asks_about_motion() -> None:
    plan = plan_with(8)
    step = plan.step(StepId.MEDIA)
    step.remaining = ["jpg", "png", "gif"]
    attempt = attempt_for(["jpg", "png", "gif"])
    attempt.actions = steps.build_actions(plan, step, attempt)
    done(attempt.actions[0], accepted())
    done(attempt.actions[1], rejected())
    assert question(plan, step, attempt, "ask:png") is None
    assert question(plan, step, attempt, "ask:gif") is None
    jpg = question(plan, step, attempt, "ask:jpg")
    assert jpg is not None and jpg.choices == ["arrived", "missing"]  # type: ignore[attr-defined]
    done(attempt.actions[2], accepted())
    gif = question(plan, step, attempt, "ask:gif")
    assert gif is not None and gif.choices == ["moving", "still", "missing"]  # type: ignore[attr-defined]


def test_the_typing_questions_follow_the_typing_itself() -> None:
    plan = plan_with(8)
    step = plan.step(StepId.MEDIA)
    step.remaining = ["jpg"]
    attempt = attempt_for(["jpg"])
    attempt.actions = steps.build_actions(plan, step, attempt)
    done(attempt.actions[0], accepted())
    assert question(plan, step, attempt, "ask:typing_ready") is not None
    assert question(plan, step, attempt, "ask:typing_seen") is None  # not shown yet
    typing = next(a for a in attempt.actions if a.id == "typing")
    typing.status = ActionStatus.DONE
    assert question(plan, step, attempt, "ask:typing_seen") is not None
    done(attempt.actions[0], rejected())  # after a failed send there is nothing to watch for
    assert question(plan, step, attempt, "ask:typing_ready") is None


def test_an_unknown_action_asks_nothing() -> None:
    plan = plan_with(8)
    step = plan.step(StepId.COUNT)
    attempt = attempt_for()
    stray = Action(id="ask:other", kind=ActionKind.ASK, label="?")
    assert steps.question_for(plan, step, attempt, stray) is None


# --------------------------------------------------------------------- judging


def test_a_count_of_fifteen_on_the_phone_is_a_lower_bound() -> None:
    plan = plan_with(None)
    step = plan.step(StepId.COUNT)
    attempt = attempt_for()
    attempt.actions = steps.build_actions(plan, step, attempt)
    for action in attempt.actions[:15]:
        done(action, accepted())
    ask = attempt.actions[-1]
    ask.question = steps.question_for(plan, step, attempt, ask)
    assert ask.question is not None
    ask.question.answer = "15"
    result = steps.finalize(plan, step, attempt)
    assert not result.void and step.data["capped"] is True and step.data["n"] == 15
    ask.question.answer = "14"  # one never reached the phone: not a clean 15 any more
    steps.finalize(plan, step, attempt)
    assert step.data["capped"] is False and step.data["n"] == 14
    assert step.data["mismatch"] is True and step.data["first_problem_index"] == 15


def test_a_failed_send_that_is_not_the_platforms_answer_voids_the_attempt() -> None:
    plan = plan_with(None)
    step = plan.step(StepId.COUNT)
    attempt = attempt_for()
    attempt.actions = steps.build_actions(plan, step, attempt)
    done(attempt.actions[0], SendRecord(ok=False, outcome="network", reason="connect_error"))
    result = steps.finalize(plan, step, attempt)
    assert result.void and "network" in (result.void_reason or "")
    assert step.data == {}  # nothing is recorded from a void attempt


def test_the_experiment_is_judged_by_the_phone() -> None:
    plan = plan_with(8, empty_token_experiment=True)
    step = plan.step(StepId.EMPTY_TOKEN)
    attempt = attempt_for()
    attempt.actions = steps.build_actions(plan, step, attempt)
    done(attempt.actions[0], accepted())
    ask = attempt.actions[1]
    ask.question = steps.question_for(plan, step, attempt, ask)
    assert ask.question is not None
    ask.question.answer = "0"  # accepted by the server, not on the phone
    steps.finalize(plan, step, attempt)
    assert step.data["api_ok"] is True and step.data["delivered"] is False
    assert step.data["failure"] is None
