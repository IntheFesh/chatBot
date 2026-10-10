"""The structured result of the channel probe and the R-CH-010 verdict.

:func:`summarize` condenses a :class:`~twin.channel.probe.model.ProbePlan` into a
:class:`ChannelProbeSummary`: the numbers the milestone check (round 09b) needs, the verdict
on R-CH-010 and the suggested values for ``channel.proactive_window_safe_h`` and
``channel.outbound_quota_safe``.  It is stored in the ``settings`` table under
:data:`SUMMARY_KEY` (format: ``docs/DECISIONS.md`` D-152) and read back with
:func:`load_channel_probe_summary`.

The verdict (R-CH-010) is deliberately blunt:

* ``not_met`` when a measurement proves the window shorter than 12 hours or fewer than 3
  messages can follow one inbound message;
* ``met`` when both were measured and both are fine;
* ``undetermined`` otherwise (something was not measured, or the points tested cannot tell).

Nothing here invents a number: every value comes from a step's measured data, and a missing
measurement stays ``None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from twin.channel.probe.model import (
    PlanStatus,
    ProbePlan,
    StepId,
    StepStatus,
    from_json,
    to_json,
)
from twin.clock import Clock
from twin.storage.settings_store import get_setting, put_setting

SUMMARY_SCHEMA_VERSION = 1
SUMMARY_KEY = "m0.channel_probe"
HISTORY_KEY = "m0.channel_probe.history"
HISTORY_LIMIT = 10

MIN_WINDOW_H = 12.0  # R-CH-010: a shorter window cannot carry the proactive messages
MIN_MESSAGES = 3  # R-CH-010: fewer messages after one inbound cannot carry chasing
MARGIN = 0.9  # the suggested safe values keep 10 % in reserve

VERDICT_MET = "met"
VERDICT_NOT_MET = "not_met"
VERDICT_UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class ChannelProbeSummary:
    """What the probe found out (and did not)."""

    run_id: str
    status: str  # running | completed | stopped
    started_at: str
    finished_at: str | None
    complete: bool  # every planned step reached a conclusion
    verdict: str  # met | not_met | undetermined
    reasons: list[str]
    n_messages: int | None  # step 1: messages that reached the phone after one inbound
    n_capped: bool  # step 1 hit its ceiling without a failure: N is "at least" this
    window_lower_bound_h: float | None  # the longest delay at which a message was delivered
    window_upper_bound_h: float | None  # the shortest delay at which one was not
    gif_animated: bool | None
    typing_visible: bool | None
    quote_supported: bool
    quota_shared: bool | None
    quota_basis: str
    suggestions: dict[str, float | int | None]
    steps: dict[str, dict[str, Any]]
    failures: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)
    schema_version: int = SUMMARY_SCHEMA_VERSION

    @property
    def meets_requirement(self) -> bool | None:
        """``True`` / ``False`` for a definite verdict, ``None`` while undetermined."""
        if self.verdict == VERDICT_MET:
            return True
        if self.verdict == VERDICT_NOT_MET:
            return False
        return None

    def to_json(self) -> dict[str, Any]:
        return to_json(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ChannelProbeSummary:
        return from_json(cls, data)


# ----------------------------------------------------------------- the verdict


def window_verdict(lower: float | None, upper: float | None) -> bool | None:
    """Is the window at least 12 hours?  ``None`` when the points tested cannot tell."""
    if lower is not None and lower >= MIN_WINDOW_H:
        return True
    if upper is not None and upper <= MIN_WINDOW_H:
        return False  # a message failed at or before 12 hours: the window is shorter
    return None


def judge(
    n_messages: int | None, lower: float | None, upper: float | None
) -> tuple[str, list[str]]:
    """The R-CH-010 verdict and the reasons behind it."""
    reasons: list[str] = []
    count_ok: bool | None = None if n_messages is None else n_messages >= MIN_MESSAGES
    window_ok = window_verdict(lower, upper)
    if count_ok is False:
        reasons.append(
            f"only {n_messages} message(s) reached the phone after one inbound message "
            f"(at least {MIN_MESSAGES} are needed)"
        )
    if window_ok is False:
        bound = f"{upper:g}" if upper is not None else "12"
        reasons.append(
            f"a message failed {bound} hour(s) after the inbound message: the window is "
            f"shorter than {MIN_WINDOW_H:g} hours"
        )
    if count_ok is False or window_ok is False:
        return VERDICT_NOT_MET, reasons
    if count_ok is None:
        reasons.append("the message count (step 1) was not measured")
    if window_ok is None:
        reasons.append("the window (step 3) was not measured far enough to say")
    if count_ok and window_ok:
        return VERDICT_MET, reasons
    return VERDICT_UNDETERMINED, reasons


def suggest_window_h(lower: float | None) -> float | None:
    """The measured window with 10 % in reserve, rounded down to a tenth of an hour."""
    if lower is None or lower <= 0:
        return None
    return math.floor(lower * MARGIN * 10) / 10


def suggest_quota(n_messages: int | None) -> int | None:
    """The measured message count with 10 % in reserve (at least 1 when any got through)."""
    if n_messages is None or n_messages < 1:
        return None
    return max(1, math.floor(n_messages * MARGIN))


# ---------------------------------------------------------------- the summary


def _failure_rows(plan: ProbePlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in plan.steps:
        for attempt in step.attempts:
            for action in attempt.actions:
                record = action.send
                if record is None or record.ok:
                    continue
                rows.append(
                    {
                        "step": step.id.value,
                        "attempt": attempt.n,
                        "action": action.label,
                        "outcome": record.outcome,
                        "reason": record.reason,
                        "code": record.code,
                        "ret": record.ret,
                        "errcode": record.errcode,
                        "errmsg": record.errmsg,
                        "http_status": record.http_status,
                        "hours_after_inbound": record.elapsed_h,
                        "at": record.at.isoformat() if record.at else None,
                    }
                )
    return rows


def _step_rows(plan: ProbePlan) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for step in plan.steps:
        rows[step.id.value] = {
            "status": step.status.value,
            "skip_reason": step.skip_reason,
            "attempts": len(step.attempts),
            "voided_attempts": step.voided_attempts,
            "finished_at": step.finished_at.isoformat() if step.finished_at else None,
            **step.data,
        }
    return rows


def _typing_visible(media: dict[str, Any]) -> bool | None:
    typing = media.get("typing")
    answer = typing.get("visible") if isinstance(typing, dict) else None
    return {"yes": True, "no": False}.get(answer) if isinstance(answer, str) else None


def summarize(plan: ProbePlan) -> ChannelProbeSummary:
    """Condense ``plan`` (finished or not) into the stored result."""
    count = plan.step(StepId.COUNT)
    media = plan.step(StepId.MEDIA)
    window = plan.step(StepId.WINDOW)
    n_raw = count.data.get("n") if count.status is StepStatus.DONE else None
    n_messages = n_raw if isinstance(n_raw, int) else None
    lower_raw = window.data.get("lower_bound_h") if window.status is StepStatus.DONE else None
    upper_raw = window.data.get("upper_bound_h") if window.status is StepStatus.DONE else None
    lower = float(lower_raw) if isinstance(lower_raw, int | float) else None
    upper = float(upper_raw) if isinstance(upper_raw, int | float) else None
    verdict, reasons = judge(n_messages, lower, upper)
    animated = media.data.get("gif_animated")
    notes: list[str] = []
    for step in plan.steps:
        if step.status is StepStatus.SKIPPED and step.skip_reason:
            notes.append(f"step {step.id.value} skipped: {step.skip_reason}")
        if step.voided_attempts:
            notes.append(
                f"step {step.id.value}: {step.voided_attempts} attempt(s) voided and redone"
            )
    dropped = window.data.get("dropped_hours")
    if isinstance(dropped, list) and dropped:
        shown = ", ".join(f"{float(hours):g}" for hours in dropped)
        notes.append(f"window points left out for lack of message budget: {shown} h")
    shared = True if n_messages is not None else None
    return ChannelProbeSummary(
        run_id=plan.run_id,
        status=plan.status.value,
        started_at=plan.created_at.isoformat(),
        finished_at=plan.finished_at.isoformat() if plan.finished_at else None,
        complete=plan.status is PlanStatus.COMPLETED,
        verdict=verdict,
        reasons=reasons,
        n_messages=n_messages,
        n_capped=bool(count.data.get("capped")) if n_messages is not None else False,
        window_lower_bound_h=lower,
        window_upper_bound_h=upper,
        gif_animated=animated if isinstance(animated, bool) else None,
        typing_visible=_typing_visible(media.data),
        quote_supported=False,
        quota_shared=shared,
        quota_basis=(
            "the protocol has one send call with one context_token for replies and for "
            "proactive messages, so the count measured in step 1 applies to both "
            "(inferred from the source and step 1, not measured separately)"
            if shared
            else "step 1 was not measured"
        ),
        suggestions={
            "channel.proactive_window_safe_h": suggest_window_h(lower),
            "channel.outbound_quota_safe": suggest_quota(n_messages),
        },
        steps=_step_rows(plan),
        failures=_failure_rows(plan),
        notes=notes,
    )


# ------------------------------------------------------------------ storage


def save_summary(session: Session, summary: ChannelProbeSummary, clock: Clock) -> None:
    """Store ``summary`` as the latest result and keep a short history."""
    put_setting(
        session,
        SUMMARY_KEY,
        summary.to_json(),
        clock=clock,
        by="channel_probe",
        record_history=False,
    )
    history = get_setting(session, HISTORY_KEY, [])
    entries = list(history) if isinstance(history, list) else []
    entries.append(
        {
            "run_id": summary.run_id,
            "finished_at": summary.finished_at,
            "status": summary.status,
            "verdict": summary.verdict,
            "n_messages": summary.n_messages,
            "window_lower_bound_h": summary.window_lower_bound_h,
        }
    )
    put_setting(
        session,
        HISTORY_KEY,
        entries[-HISTORY_LIMIT:],
        clock=clock,
        by="channel_probe",
        record_history=False,
    )


def measured_capabilities(
    summary: ChannelProbeSummary | None,
) -> tuple[float | None, int | None, bool | None]:
    """``(window hours, message count, GIF moves)`` as measured; ``None`` where unknown."""
    if summary is None:
        return None, None, None
    return summary.window_lower_bound_h, summary.n_messages, summary.gif_animated


def load_channel_probe_summary(session: Session) -> ChannelProbeSummary | None:
    """The latest stored probe result, or ``None`` if the probe never finished or was stopped.

    This is the read interface for the milestone check (round 09b): the M0 channel part is
    passed when ``summary.complete and summary.meets_requirement is True``.
    """
    raw = get_setting(session, SUMMARY_KEY, None)
    if not isinstance(raw, dict) or raw.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        return None
    return ChannelProbeSummary.from_json(raw)
