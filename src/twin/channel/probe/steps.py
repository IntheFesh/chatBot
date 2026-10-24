"""What each measurement sends, what it asks, and how its outcome is judged (R-CH-009).

The runner is generic: it executes the :class:`~twin.channel.probe.model.Action` lists built
here and asks :func:`finalize` what an attempt proved.  Everything about the *meaning* of the
four measurements lives in this module, as pure functions of the plan, so it can be tested
without a channel or a clock:

1. **count**  one message every 2 minutes after a fresh inbound message, until the first
   failure or 15; the phone's count (not the server's) is N.
2. **media**  a JPEG, a PNG and a multi-frame GIF, then the typing indicator; at most N-1
   messages per inbound message, so the pictures are split over several attempts when N is
   small.  Quoting is not tested: the protocol has no way to send one (D-009).
3. **window** a text at 1, 6, 12, 20, 23 and 25 hours after a fresh inbound message (the
   last N-1 points when N is small), then the phone's count.
4. **quota**  whether replies and proactive messages share the count is concluded in the
   summary from the protocol and step 1; it needs no messages of its own.

An attempt is **void** (redone from the start) when the platform did not answer (no network,
timeout, expired login), when the user wrote to the bot during the measurement, or when the
application restarted in the middle of a send.  A failure that *is* the platform's answer (an
error number, an HTTP error) is a measurement, not a reason to redo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twin.channel.base import TEST_PREFIX
from twin.channel.probe.model import (
    Action,
    ActionKind,
    ActionStatus,
    Attempt,
    ProbePlan,
    Question,
    QuestionKind,
    SendRecord,
    StepId,
    StepState,
    to_json,
)

IMAGE_NAMES = ("jpg", "png", "gif")
IMAGE_LABELS = {"jpg": "JPEG", "png": "PNG", "gif": "GIF"}

STEP_TITLES = {
    StepId.COUNT: "how many messages follow one message from you",
    StepId.MEDIA: "pictures, animated GIF and the typing indicator",
    StepId.WINDOW: "how long after your message the bot can still write",
    StepId.EMPTY_TOKEN: "experiment: a message with an empty context token",
}
STEP_TITLES_ZH = {
    StepId.COUNT: "连发条数",
    StepId.MEDIA: "图片、GIF 与正在输入",
    StepId.WINDOW: "主动发送窗口",
    StepId.EMPTY_TOKEN: "空令牌实验",
}


# ---------------------------------------------------------------- activation


@dataclass(frozen=True)
class AttemptSetup:
    """What the next attempt of a step is allowed to do."""

    budget: int | None  # the most messages it may send
    items: list[str]  # pictures (media) or hour marks (window) it will cover
    from_inbound: bool  # timed from the user's message (window) or from the moment we saw it


def activation(plan: ProbePlan, step: StepState) -> AttemptSetup | str:
    """The setup of the next attempt, or the reason the step must be skipped (a ``str``)."""
    if step.id is StepId.COUNT:
        return AttemptSetup(None, [], False)
    budget = plan.message_budget()
    if budget is None:
        return "step 1 did not measure how many messages follow one inbound message"
    if budget < 1:
        reached = plan.step(StepId.COUNT).data.get("n")
        return (
            f"only {reached} message(s) reached the phone after one inbound message, so there "
            "is no room to send anything besides the one that proves it"
        )
    if step.id is StepId.MEDIA:
        if not step.remaining and not step.attempts:
            step.remaining = list(IMAGE_NAMES)
        return AttemptSetup(budget, step.remaining[:budget], False)
    if step.id is StepId.WINDOW:
        hours = sorted(plan.options.window_hours)
        kept = hours[-budget:] if budget < len(hours) else hours
        return AttemptSetup(budget, [f"{hour:g}" for hour in kept], True)
    return AttemptSetup(min(budget, 1), [], False)


def dropped_hours(plan: ProbePlan, kept: list[str]) -> list[float]:
    """Window points left out because the message budget was too small."""
    keep = {float(hour) for hour in kept}
    return [hour for hour in sorted(plan.options.window_hours) if hour not in keep]


# ------------------------------------------------------------------ messages


def announce_text(plan: ProbePlan, step: StepState, attempt: Attempt) -> str:
    """The WeChat message asking for a fresh inbound message (always a test message)."""
    position = [s.id for s in plan.steps].index(step.id) + 1
    text = (
        f"{TEST_PREFIX} 通道探针:第{position}步「{STEP_TITLES_ZH[step.id]}」即将开始。"
        "请现在给我发一条任意消息,我收到后才会开始。"
    )
    if step.id is StepId.WINDOW:
        longest = max(plan.options.window_hours)
        text += f"发出之后约{longest:g}小时内请不要再给我发任何消息,否则这一步作废、需要重做。"
    if attempt.n > 1:
        text += "(上一次作废了,这是重做。)"
    return text


def terminal_notice(plan: ProbePlan, step: StepState, attempt: Attempt) -> str:
    """What the person at the keyboard is told while the probe waits for a fresh message."""
    position = [s.id for s in plan.steps].index(step.id) + 1
    total = len(plan.steps)
    text = (
        f"Step {position}/{total} ({step.id.value}): {STEP_TITLES[step.id]}. "
        "Send the bot ONE message from your phone now (anything); the step starts when it arrives."
    )
    if step.id is StepId.WINDOW:
        longest = max(plan.options.window_hours)
        text += (
            f" After that send NOTHING for about {longest:g} hours, or the step is void and "
            "must be redone."
        )
    if attempt.n > 1:
        text += f" (attempt {attempt.n}: the previous one was void)"
    return text


def _text_action(index: int, label: str, text: str, offset_s: float) -> Action:
    return Action(
        id=f"send:{index}", kind=ActionKind.SEND_TEXT, label=label, due_offset_s=offset_s, text=text
    )


def build_actions(plan: ProbePlan, step: StepState, attempt: Attempt) -> list[Action]:
    """The ordered actions of ``attempt``, created when the user's fresh message arrived."""
    options = plan.options
    if step.id is StepId.COUNT:
        actions = [
            _text_action(
                i,
                f"count {i}/{options.max_messages}",
                f"{TEST_PREFIX} 条数测试 {i}/{options.max_messages}(先别回复我)",
                (i - 1) * options.interval_s,
            )
            for i in range(1, options.max_messages + 1)
        ]
        actions.append(Action(id="ask:count", kind=ActionKind.ASK, label="phone count"))
        return actions
    if step.id is StepId.WINDOW:
        total = len(attempt.items)
        actions = []
        for index, mark in enumerate(attempt.items, start=1):
            hours = float(mark)
            actions.append(
                _text_action(
                    index,
                    f"window {hours:g} h",
                    f"{TEST_PREFIX} 窗口测试 {index}/{total}(你上次发消息后约{hours:g}小时)。"
                    "请不要回复,也不要给我发消息,直到测试结束。",
                    hours * 3600.0,
                )
            )
        actions.append(Action(id="ask:window", kind=ActionKind.ASK, label="phone count"))
        return actions
    if step.id is StepId.MEDIA:
        actions = [
            Action(
                id=f"send:{name}",
                kind=ActionKind.SEND_IMAGE,
                label=f"picture {IMAGE_LABELS[name]}",
                due_offset_s=index * options.image_gap_s,
                image=name,
            )
            for index, name in enumerate(attempt.items)
        ]
        actions += [
            Action(id=f"ask:{name}", kind=ActionKind.ASK, label=f"{IMAGE_LABELS[name]} arrived?")
            for name in attempt.items
        ]
        if attempt.items == step.remaining:  # the last attempt of this step also tests typing
            actions += [
                Action(id="ask:typing_ready", kind=ActionKind.ASK, label="ready to watch"),
                Action(id="typing", kind=ActionKind.TYPING, label="typing indicator"),
                Action(id="ask:typing_seen", kind=ActionKind.ASK, label="typing visible?"),
            ]
        return actions
    return [
        Action(
            id="send:1",
            kind=ActionKind.SEND_TEXT,
            label="empty context token",
            text=f"{TEST_PREFIX} 空令牌实验:这条消息没有带 context_token。",
            empty_token=True,
        ),
        Action(id="ask:empty", kind=ActionKind.ASK, label="phone count"),
    ]


# ----------------------------------------------------------------- questions


def _sends(attempt: Attempt) -> list[Action]:
    return [a for a in attempt.actions if a.kind in (ActionKind.SEND_TEXT, ActionKind.SEND_IMAGE)]


def _accepted(attempt: Attempt) -> int:
    return sum(1 for a in _sends(attempt) if a.send is not None and a.send.ok)


def _action(attempt: Attempt, action_id: str) -> Action | None:
    return next((a for a in attempt.actions if a.id == action_id), None)


def question_for(
    plan: ProbePlan, step: StepState, attempt: Attempt, action: Action
) -> Question | None:
    """The question an ASK action puts to the user now, or ``None`` if there is nothing to ask."""
    accepted = _accepted(attempt)
    if action.id == "ask:count":
        if accepted == 0:
            return None
        return Question(
            id=f"{step.id.value}-{attempt.n}-count",
            kind=QuestionKind.COUNT,
            maximum=accepted,
            prompt=(
                "On your phone, in the ClawBot chat: how many messages starting with "
                f"'{TEST_PREFIX} 条数测试' arrived? (the server accepted {accepted}; "
                "answer with what you actually see, 0 if none)"
            ),
        )
    if action.id == "ask:window":
        if accepted == 0:
            return None
        return Question(
            id=f"{step.id.value}-{attempt.n}-count",
            kind=QuestionKind.COUNT,
            maximum=accepted,
            prompt=(
                "On your phone, in the ClawBot chat: how many messages starting with "
                f"'{TEST_PREFIX} 窗口测试' arrived? (the server accepted {accepted}; "
                "answer with what you actually see, 0 if none)"
            ),
        )
    if action.id == "ask:empty":
        if accepted == 0:
            return None
        return Question(
            id=f"{step.id.value}-{attempt.n}-count",
            kind=QuestionKind.COUNT,
            maximum=1,
            prompt=(
                f"Did the message '{TEST_PREFIX} 空令牌实验' arrive on your phone? "
                "(1 = yes, 0 = no)"
            ),
        )
    if action.id.startswith("ask:") and action.id[4:] in IMAGE_NAMES:
        name = action.id[4:]
        sent = _action(attempt, f"send:{name}")
        if sent is None or sent.send is None or not sent.send.ok:
            return None
        if name == "gif":
            return Question(
                id=f"{step.id.value}-{attempt.n}-gif",
                kind=QuestionKind.CHOICE,
                choices=["moving", "still", "missing"],
                prompt=(
                    "Did the GIF test picture ('[TEST] GIF', a yellow square on dark blue) "
                    "arrive, and does the square MOVE across the picture? "
                    "(moving / still / missing)"
                ),
            )
        colours = "blue to yellow gradient" if name == "jpg" else "red to green gradient"
        return Question(
            id=f"{step.id.value}-{attempt.n}-{name}",
            kind=QuestionKind.CHOICE,
            choices=["arrived", "missing"],
            prompt=(
                f"Did the {IMAGE_LABELS[name]} test picture ('[TEST] {name.upper()}', "
                f"{colours}) arrive on your phone? (arrived / missing)"
            ),
        )
    if action.id == "ask:typing_ready":
        if any(a.status is ActionStatus.FAILED for a in _sends(attempt)):
            return None
        return Question(
            id=f"{step.id.value}-{attempt.n}-typing-ready",
            kind=QuestionKind.READY,
            prompt=(
                "Open the ClawBot chat on your phone and keep it in front of you. Press Enter "
                f"when you are looking: the typing indicator is then sent for "
                f"{plan.options.typing_hold_s:g} seconds."
            ),
        )
    if action.id == "ask:typing_seen":
        typing = _action(attempt, "typing")
        if typing is None or typing.status is not ActionStatus.DONE:
            return None
        return Question(
            id=f"{step.id.value}-{attempt.n}-typing-seen",
            kind=QuestionKind.CHOICE,
            choices=["yes", "no", "unsure"],
            prompt=(
                "Did the chat show the other side typing (the 'is typing' line under the "
                "bot's name) while the indicator was sent? (yes / no / unsure)"
            ),
        )
    return None


# --------------------------------------------------------------- judging


@dataclass(frozen=True)
class Finalization:
    """What an attempt proved: redo it, or accept it (and say whether the step is finished)."""

    void_reason: str | None = None
    step_done: bool = True

    @property
    def void(self) -> bool:
        return self.void_reason is not None


def send_row(record: SendRecord | None) -> dict[str, Any] | None:
    """A send as stored in a step's result (only numbers and fixed words)."""
    if record is None:
        return None
    row = to_json(record)
    return {k: v for k, v in row.items() if v not in (None, "", False) or k == "ok"}


def _answer(attempt: Attempt, action_id: str) -> str | None:
    action = _action(attempt, action_id)
    if action is None or action.question is None:
        return None
    return action.question.answer


def _interruption(attempt: Attempt) -> str | None:
    for action in _sends(attempt):
        record = action.send
        if (
            action.status is ActionStatus.FAILED
            and record is not None
            and not record.is_platform_answer()
        ):
            return f"{action.label}: {record.outcome} ({record.reason or 'no detail'})"
    return None


def finalize(plan: ProbePlan, step: StepState, attempt: Attempt) -> Finalization:
    """Judge a finished attempt and fold its measurements into ``step.data``."""
    interrupted = _interruption(attempt)
    if interrupted is not None:
        return Finalization(void_reason=f"interrupted - {interrupted}")
    if step.id is StepId.COUNT:
        return _finalize_count(plan, step, attempt)
    if step.id is StepId.MEDIA:
        return _finalize_media(step, attempt)
    if step.id is StepId.WINDOW:
        return _finalize_window(plan, step, attempt)
    return _finalize_empty_token(step, attempt)


def _first_failure(attempt: Attempt) -> SendRecord | None:
    for action in _sends(attempt):
        if action.status is ActionStatus.FAILED:
            return action.send
    return None


def _finalize_count(plan: ProbePlan, step: StepState, attempt: Attempt) -> Finalization:
    accepted = _accepted(attempt)
    answer = _answer(attempt, "ask:count")
    phone = int(answer) if answer is not None else 0
    failure = _first_failure(attempt)
    first_problem = accepted + 1 if failure is not None else None
    if phone < accepted:
        first_problem = phone + 1 if first_problem is None else min(first_problem, phone + 1)
    step.data = {
        "n": phone,
        "capped": failure is None and accepted == plan.options.max_messages and phone == accepted,
        "api_ok": accepted,
        "phone_received": phone,
        "mismatch": phone != accepted,
        "first_problem_index": first_problem,
        "first_failure": send_row(failure),
        "max_messages": plan.options.max_messages,
        "interval_s": plan.options.interval_s,
        "inbound_at": attempt.inbound_at.isoformat() if attempt.inbound_at else None,
        "context_fp": attempt.context_fp,
    }
    return Finalization()


def _finalize_media(step: StepState, attempt: Attempt) -> Finalization:
    images: dict[str, Any] = dict(step.data.get("images", {}))
    for name in attempt.items:
        sent = _action(attempt, f"send:{name}")
        record = sent.send if sent else None
        row: dict[str, Any] = {"api_ok": bool(record and record.ok)}
        if record is not None and not record.ok:
            row["failure"] = send_row(record)
            row["phone"] = "not_delivered"
        elif record is not None:
            row["phone"] = _answer(attempt, f"ask:{name}") or "unanswered"
        else:
            row["phone"] = "not_tested"
        images[name] = row
    step.data["images"] = images
    gif = images.get("gif")
    if isinstance(gif, dict):
        state = gif.get("phone")
        step.data["gif_animated"] = {"moving": True, "still": False}.get(str(state))
    step.data.setdefault("gif_animated", None)
    typing = _action(attempt, "typing")
    if typing is not None:
        seen = _answer(attempt, "ask:typing_seen")
        step.data["typing"] = {
            "sent": typing.status is ActionStatus.DONE,
            "visible": seen,
            "hold_s": plan_hold(typing),
        }
    step.data["quote"] = "not_supported"
    step.data["budget_per_inbound"] = attempt.budget
    failed = any(a.status is ActionStatus.FAILED for a in _sends(attempt))
    step.remaining = [] if failed else [n for n in step.remaining if n not in attempt.items]
    if failed:
        for name in IMAGE_NAMES:
            images.setdefault(name, {"api_ok": False, "phone": "not_tested"})
    return Finalization(step_done=not step.remaining)


def plan_hold(action: Action) -> float | None:
    """How long the typing indicator was held (recorded in the action's note)."""
    return float(action.note) if action.note else None


def _finalize_window(plan: ProbePlan, step: StepState, attempt: Attempt) -> Finalization:
    sends = _sends(attempt)
    accepted_rows = [a for a in sends if a.send is not None and a.send.ok]
    answer = _answer(attempt, "ask:window")
    phone = int(answer) if answer is not None else 0
    points: list[dict[str, Any]] = []
    accepted_seen = 0
    upper: float | None = None
    lower: float | None = None
    for action in sends:
        record = action.send
        point: dict[str, Any] = {"hours_planned": action.due_offset_s / 3600.0}
        if record is None:
            point.update(api_ok=None, phone_delivered=None)  # never sent: an earlier one failed
        elif record.ok:
            accepted_seen += 1
            delivered = accepted_seen <= phone
            point.update(
                api_ok=True,
                phone_delivered=delivered,
                hours_actual=record.elapsed_h,
                late_s=round(record.late_s),
            )
            if delivered and record.elapsed_h is not None:
                lower = max(lower or 0.0, record.elapsed_h)
            if not delivered and record.elapsed_h is not None:
                upper = record.elapsed_h if upper is None else min(upper, record.elapsed_h)
        else:
            point.update(
                api_ok=False,
                phone_delivered=False,
                hours_actual=record.elapsed_h,
                failure=send_row(record),
            )
            if record.elapsed_h is not None:
                upper = record.elapsed_h if upper is None else min(upper, record.elapsed_h)
        points.append(point)
    failure = _first_failure(attempt)
    step.data = {
        "points": points,
        "dropped_hours": dropped_hours(plan, attempt.items),
        "lower_bound_h": lower,
        "upper_bound_h": upper,
        "api_ok": len(accepted_rows),
        "phone_received": phone,
        "mismatch": phone != len(accepted_rows),
        "first_failure": send_row(failure),
        "inbound_at": attempt.inbound_at.isoformat() if attempt.inbound_at else None,
        "context_fp": attempt.context_fp,
        "budget_per_inbound": attempt.budget,
    }
    return Finalization()


def _finalize_empty_token(step: StepState, attempt: Attempt) -> Finalization:
    action = _action(attempt, "send:1")
    record = action.send if action else None
    answer = _answer(attempt, "ask:empty")
    delivered = answer == "1"
    step.data = {
        "api_ok": bool(record and record.ok),
        "phone_received": int(answer) if answer is not None else 0,
        "delivered": delivered,
        "failure": send_row(record) if record is not None and not record.ok else None,
        "inbound_at": attempt.inbound_at.isoformat() if attempt.inbound_at else None,
    }
    return Finalization()
