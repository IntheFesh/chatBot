"""Where the probe plan and its audit trail live: the encrypted ``channel_state`` table.

One key holds the whole plan (:data:`KEY_PLAN`), one the audit trail of bypassed sends
(:data:`KEY_AUDIT`).  Several writers touch the plan: the running application (the state
machine) and the command line (``answer``, ``stop``), possibly from different processes, so
every change is a read-modify-write inside one ``BEGIN IMMEDIATE`` transaction
(:meth:`ProbeStore.update`); a change function only touches the fields it owns.

When a change leaves the plan in a final state (completed or stopped) the structured summary
is written to the ``settings`` table in the same transaction, so the plan and the result the
milestone check reads can never disagree.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from twin.channel.base import TEST_PREFIX
from twin.channel.probe.model import (
    ActionKind,
    ActionStatus,
    PlanStatus,
    ProbeOptions,
    ProbePlan,
    Question,
    QuestionKind,
    end_plan,
    new_plan,
    plan_from_json,
    plan_to_json,
)
from twin.channel.probe.summary import save_summary, summarize
from twin.channel.state import ChannelStateStore, StateTx
from twin.clock import Clock, ensure_aware
from twin.storage.db import Database

T = TypeVar("T")

KEY_PLAN = "probe.plan"
KEY_AUDIT = "probe.audit"
AUDIT_MAX = 500


class ProbeError(Exception):
    """Base class of probe errors that are the user's to fix."""


class ProbeAlreadyRunning(ProbeError):
    """A plan is still running; stop it before starting another."""


class NoProbePlan(ProbeError):
    """There is no probe plan."""


class AnswerRejected(ProbeError):
    """The answer is not valid for the question."""


def _load(tx: StateTx) -> ProbePlan | None:
    data = tx.get(KEY_PLAN)
    return plan_from_json(data) if isinstance(data, dict) else None


class ProbeStore:
    """The stored probe plan, with atomic changes."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock
        self._state = ChannelStateStore(db)

    @property
    def state(self) -> ChannelStateStore:
        return self._state

    # -------------------------------------------------------------- the plan

    def load(self) -> ProbePlan | None:
        data = self._state.get(KEY_PLAN)
        return plan_from_json(data) if isinstance(data, dict) else None

    def create(self, options: ProbeOptions | None = None) -> ProbePlan:
        """Start a new plan; refuses while another one is running."""
        now = self._clock.now_utc()
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"
        with self._db.transaction() as session:
            tx = StateTx(session)
            current = _load(tx)
            if current is not None and current.status is PlanStatus.RUNNING:
                raise ProbeAlreadyRunning(
                    f"probe {current.run_id} is still running (`twin channel probe stop` ends it)"
                )
            plan = new_plan(run_id, now, options or ProbeOptions())
            plan.add_event(now, "probe plan created")
            tx.put(KEY_PLAN, plan_to_json(plan))
        return plan

    def update(self, change: Callable[[ProbePlan], T]) -> T:
        """Apply ``change`` to the stored plan atomically; returns what ``change`` returns."""
        with self._db.transaction() as session:
            tx = StateTx(session)
            plan = _load(tx)
            if plan is None:
                raise NoProbePlan("there is no probe plan (`twin channel probe start`)")
            was_running = plan.status is PlanStatus.RUNNING
            result = change(plan)
            tx.put(KEY_PLAN, plan_to_json(plan))
            if was_running and plan.status is not PlanStatus.RUNNING:
                save_summary(session, summarize(plan), self._clock)
            return result

    def finish(self, status: PlanStatus, reason: str | None) -> ProbePlan:
        """End the plan (``completed`` or ``stopped``); a plan already ended is left alone."""
        if status is PlanStatus.RUNNING:
            raise ValueError("finish() needs a final status")
        now = self._clock.now_utc()

        def change(plan: ProbePlan) -> ProbePlan:
            end_plan(plan, status, reason, now)
            return plan

        return self.update(change)

    # ------------------------------------------------------------- questions

    @staticmethod
    def pending_questions(plan: ProbePlan) -> list[Question]:
        """The questions the person at the keyboard has to answer now."""
        step = plan.active_step()
        attempt = step.current_attempt() if step else None
        if attempt is None:
            return []
        return [
            action.question
            for action in attempt.actions
            if action.kind is ActionKind.ASK
            and action.status is ActionStatus.ACTIVE
            and action.question is not None
            and not action.question.answered
        ]

    def answer(self, question_id: str, text: str) -> Question:
        """Record the answer to a pending question (validated against its kind)."""
        now = self._clock.now_utc()

        def change(plan: ProbePlan) -> Question:
            for question in self.pending_questions(plan):
                if question.id != question_id:
                    continue
                question.answer = validate_answer(question, text)
                question.answered_at = now
                plan.add_event(now, f"question {question.id} answered")
                return question
            raise AnswerRejected(f"no pending question {question_id!r}")

        return self.update(change)

    # ----------------------------------------------------------------- audit

    def add_audit(self, entry: dict[str, Any]) -> None:
        with self._state.transaction() as tx:
            entries = list(tx.get(KEY_AUDIT, []))
            entries.append(entry)
            tx.put(KEY_AUDIT, entries[-AUDIT_MAX:])

    def audit(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._state.get(KEY_AUDIT, [])]


def validate_answer(question: Question, text: str) -> str:
    """The canonical answer text for ``question``, or :class:`AnswerRejected`."""
    cleaned = text.strip()
    if question.kind is QuestionKind.READY:
        return "ready"
    if question.kind is QuestionKind.COUNT:
        if not cleaned.isdigit():
            raise AnswerRejected("answer with a whole number such as 0, 3 or 8")
        value = int(cleaned)
        if question.maximum is not None and value > question.maximum:
            raise AnswerRejected(
                f"the program sent at most {question.maximum} message(s) here; "
                f"{value} cannot have arrived"
            )
        return str(value)
    for choice in question.choices:
        if cleaned.lower() == choice.lower():
            return choice
    raise AnswerRejected("answer with one of: " + ", ".join(question.choices))


def audit_entry(
    *,
    at: datetime,
    decision: str,
    why: str,
    kind: str,
    text: str | None,
    gate_reason: str | None,
    run_id: str | None,
    step: str | None,
    empty_token: bool = False,
) -> dict[str, Any]:
    """One audit record.  Probe texts are fixed templates, so storing their start is safe."""
    return {
        "at": ensure_aware(at).isoformat(),
        "decision": decision,
        "why": why,
        "kind": kind,
        "text_chars": len(text) if text is not None else None,
        # only a probe text is quoted; anything else could be chat content, so only its length
        "text_head": text[:24] if text is not None and text.startswith(TEST_PREFIX) else None,
        "gate_reason": gate_reason,
        "run_id": run_id,
        "step": step,
        "empty_token": empty_token,
    }
