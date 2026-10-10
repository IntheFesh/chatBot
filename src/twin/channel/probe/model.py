"""The persisted plan of the channel probe (R-CH-009).

The probe runs for more than a day and must survive restarts, so everything it needs to
continue is data: a :class:`ProbePlan` with one :class:`StepState` per measurement, each step
with its :class:`Attempt` s (an attempt is "ask the user for a fresh message, wait for it, send
what the step sends, ask what arrived"), each attempt with its :class:`Action` s.

The classes are plain dataclasses; :func:`encode` and :func:`decode` turn them into the JSON
that ``channel_state`` stores (and back), so adding a field with a default never breaks a plan
that is already stored.  Nothing here touches the database or the clock.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, Union, get_args, get_origin, get_type_hints

from twin.clock import ensure_aware

SCHEMA_VERSION = 1
EVENTS_MAX = 400

# the order of the measurements (R-CH-009): count first, because its result sizes the others
STEP_ORDER = ("count", "media", "window", "empty_token")


class PlanStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


class StepId(StrEnum):
    COUNT = "count"  # 1. how many messages after one inbound
    MEDIA = "media"  # 2. pictures, GIF, typing (quotes: not supported by the protocol)
    WINDOW = "window"  # 3. how long after the inbound a message still gets through
    EMPTY_TOKEN = "empty_token"  # noqa: S105 - a step name, not a password; optional experiment: a text with an empty context_token


class StepStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    SKIPPED = "skipped"


class AttemptPhase(StrEnum):
    ANNOUNCE = "announce"  # tell the user to send a fresh message
    WAITING = "waiting"  # waiting for that message
    RUNNING = "running"  # the step's actions are being carried out
    FINISHED = "finished"


class ActionKind(StrEnum):
    SEND_TEXT = "send_text"
    SEND_IMAGE = "send_image"
    TYPING = "typing"
    ASK = "ask"


class ActionStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"  # being carried out right now (a restart finds it and marks it unknown)
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class QuestionKind(StrEnum):
    COUNT = "count"  # a number from 0 to ``maximum``
    CHOICE = "choice"  # one of ``choices``
    READY = "ready"  # "press enter when you are looking at the phone"


# ------------------------------------------------------------------- records


@dataclass
class SendRecord:
    """The outcome of one send, with every number the server gave back (message redacted)."""

    ok: bool
    outcome: str  # an ``OutboundKind`` value, or "unknown" after a restart in mid-send
    reason: str = ""
    code: int | None = None
    ret: int | None = None
    errcode: int | None = None
    errmsg: str | None = None
    http_status: int | None = None
    session_expired: bool = False
    message_id: str | None = None
    client_id: str | None = None
    at: datetime | None = None
    elapsed_h: float | None = None  # hours since the inbound message that started the attempt
    late_s: float = 0.0  # how long after its planned time the send really happened

    def is_platform_answer(self) -> bool:
        """True when the server itself said no (a code or an HTTP status), not this machine."""
        if self.ok or self.outcome not in ("window_rejected", "rejected"):
            return False
        return self.code is not None or self.http_status is not None


@dataclass
class Question:
    """A question for the person at the keyboard; ``answer`` is text (numbers are digits)."""

    id: str
    kind: QuestionKind
    prompt: str
    choices: list[str] = field(default_factory=list)
    maximum: int | None = None
    asked_at: datetime | None = None
    answered_at: datetime | None = None
    answer: str | None = None

    @property
    def answered(self) -> bool:
        return self.answer is not None


@dataclass
class Action:
    """One thing an attempt does, in order: send a text or picture, show typing, or ask."""

    id: str
    kind: ActionKind
    label: str
    due_offset_s: float = 0.0  # seconds after the attempt's anchor
    due_at: datetime | None = None
    status: ActionStatus = ActionStatus.PENDING
    text: str | None = None  # SEND_TEXT: the message (always starts with the test prefix)
    image: str | None = None  # SEND_IMAGE: "jpg", "png" or "gif"
    empty_token: bool = False  # SEND_TEXT: the empty context_token experiment
    tries: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    send: SendRecord | None = None
    question: Question | None = None
    note: str | None = None


@dataclass
class Attempt:
    """One round of a step: announce, wait for the user's fresh message, act, ask, judge."""

    n: int
    phase: AttemptPhase
    armed_at: datetime
    baseline_inbound_at: datetime | None  # the last inbound message when this attempt began
    announced_at: datetime | None = None
    announce_note: str | None = None
    candidate_inbound_at: datetime | None = None  # a newer inbound seen, letting it settle
    inbound_at: datetime | None = None  # the inbound message that started the measuring
    started_at: datetime | None = None
    context_fp: str | None = None  # a short fingerprint of that message's context_token
    budget: int | None = None  # the most messages this attempt may send (N - 1)
    items: list[str] = field(default_factory=list)  # media: the pictures sent in this attempt
    actions: list[Action] = field(default_factory=list)
    outcome: str | None = None  # "complete" or "void"
    void_reason: str | None = None
    finished_at: datetime | None = None


@dataclass
class StepState:
    id: StepId
    status: StepStatus = StepStatus.PENDING
    attempts: list[Attempt] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)  # media: pictures still to be tested
    data: dict[str, Any] = field(default_factory=dict)  # the measured result, built up
    skip_reason: str | None = None
    finished_at: datetime | None = None

    def current_attempt(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def voided_attempts(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.outcome == "void")


@dataclass
class ProbeOptions:
    """Timing and size of the measurements.  The defaults are the ones SPEC R-CH-009 names."""

    interval_s: float = 120.0  # step 1: one message every 2 minutes
    max_messages: int = 15  # step 1: stop here even without a failure
    window_hours: list[float] = field(default_factory=lambda: [1.0, 6.0, 12.0, 20.0, 23.0, 25.0])
    empty_token_experiment: bool = False
    network_retries: int = 5  # a send that never left this machine may be repeated
    network_retry_s: float = 60.0
    typing_hold_s: float = 30.0
    image_gap_s: float = 5.0
    max_attempts_per_step: int = 10
    settle_s: float = 3.0  # wait this long after a new inbound message before measuring
    poll_s: float = 2.0  # while waiting for the user (a message, an answer)
    watch_s: float = 60.0  # while waiting for the next timed send: how often to look for a message
    idle_poll_s: float = 5.0  # while no plan is running


@dataclass
class Event:
    at: datetime
    text: str


@dataclass
class ProbePlan:
    run_id: str
    status: PlanStatus
    created_at: datetime
    options: ProbeOptions
    steps: list[StepState]
    schema_version: int = SCHEMA_VERSION
    finished_at: datetime | None = None
    stop_reason: str | None = None
    notice: str | None = None  # what the person at the keyboard should do right now
    events: list[Event] = field(default_factory=list)

    def step(self, step_id: StepId) -> StepState:
        for state in self.steps:
            if state.id is step_id:
                return state
        raise KeyError(step_id)

    def active_step(self) -> StepState | None:
        """The first step that is not finished, or ``None`` when all are."""
        for state in self.steps:
            if state.status in (StepStatus.PENDING, StepStatus.ACTIVE):
                return state
        return None

    def add_event(self, at: datetime, text: str) -> None:
        self.events.append(Event(ensure_aware(at), text))
        del self.events[:-EVENTS_MAX]

    def message_budget(self) -> int | None:
        """How many messages the later steps may send per inbound: ``N - 1`` (R-CH-009)."""
        measured = self.step(StepId.COUNT).data.get("n")
        if not isinstance(measured, int):
            return None
        return max(0, measured - 1)


def end_plan(plan: ProbePlan, status: PlanStatus, reason: str | None, now: datetime) -> None:
    """Put a running plan into a final state (completed or stopped)."""
    if plan.status is not PlanStatus.RUNNING:
        return
    plan.status = status
    plan.finished_at = ensure_aware(now)
    plan.stop_reason = reason
    plan.notice = None
    plan.add_event(now, f"probe {status.value}" + (f": {reason}" if reason else ""))


def new_plan(run_id: str, now: datetime, options: ProbeOptions) -> ProbePlan:
    """A plan with its steps in the order of :data:`STEP_ORDER`."""
    ids = [StepId.COUNT, StepId.MEDIA, StepId.WINDOW]
    if options.empty_token_experiment:
        ids.append(StepId.EMPTY_TOKEN)
    return ProbePlan(
        run_id=run_id,
        status=PlanStatus.RUNNING,
        created_at=ensure_aware(now),
        options=options,
        steps=[StepState(step_id) for step_id in ids],
    )


# --------------------------------------------------------------------- codec


def encode(value: Any) -> Any:
    """Dataclasses, enums and datetimes to JSON-friendly values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_json(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return ensure_aware(value).isoformat()
    if isinstance(value, list | tuple):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    return value


def to_json(obj: Any) -> dict[str, Any]:
    """A dataclass instance as a JSON object."""
    return {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(tp: type) -> dict[str, Any]:
    """The resolved field annotations of a dataclass (resolving them is slow, so once)."""
    found = _HINTS.get(tp)
    if found is None:
        found = _HINTS[tp] = get_type_hints(tp)
    return found


def from_json[D](tp: type[D], data: dict[str, Any]) -> D:
    """Build the dataclass ``tp`` from the JSON object ``data`` (missing keys use defaults)."""
    hints = _hints(tp)
    return tp(**{key: decode(hints[key], value) for key, value in data.items() if key in hints})


def decode(tp: Any, data: Any) -> Any:
    """The inverse of :func:`encode` for the annotation ``tp``."""
    if tp is Any or data is None:
        return data
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        options = [arg for arg in get_args(tp) if arg is not type(None)]
        return decode(options[0], data) if len(options) == 1 else data
    if origin is list:
        (item_type,) = get_args(tp)
        return [decode(item_type, item) for item in data]
    if origin is dict:
        _key_type, item_type = get_args(tp)
        return {str(key): decode(item_type, item) for key, item in data.items()}
    if isinstance(tp, type):
        if dataclasses.is_dataclass(tp):
            return from_json(tp, data)
        if issubclass(tp, Enum):
            return tp(data)
        if tp is datetime:
            return ensure_aware(datetime.fromisoformat(data))
        if tp is float:
            return float(data)
    return data


def plan_to_json(plan: ProbePlan) -> dict[str, Any]:
    return to_json(plan)


def plan_from_json(data: dict[str, Any]) -> ProbePlan:
    return from_json(ProbePlan, data)
