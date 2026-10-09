"""The state machine that carries the probe plan out (R-CH-009).

The plan lives in the database; this class only moves it forward.  :meth:`ProbeRunner.tick`
looks at the stored plan, does at most one thing (announce, notice the user's message, send one
message, put one question, judge an attempt) and returns how long to wait before looking
again.  Everything it decides is saved before and after each send, so a restart in the middle
of a 26-hour run continues where it was; a send whose result is lost is never repeated (the
attempt is voided instead).

The rules, in one place:

* every attempt starts with the user's fresh message (the terminal and WeChat both ask for
  it); a message from the user *during* an attempt voids it, because it resets the platform's
  count and window;
* a send that fails stops the attempt's sends at once and is never retried, except a send
  that never left this machine (no network), which is repeated a few times a minute apart;
* what counts is what reached the phone: after each attempt the user is asked, and the answer
  (not the server's "ok") is the measurement;
* the probe never talks to anyone but the bound user, and only through the channel's own
  ``send_*`` methods with a :class:`~twin.channel.probe.policy.ProbeSendPolicy`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, TypeVar

from twin.channel.base import (
    AuthState,
    BypassRefused,
    CapabilityNotSupported,
    ChannelError,
    MediaNotAllowed,
    OutboundResult,
    RecipientNotAllowed,
    SendBypass,
)
from twin.channel.console import AlertBanner
from twin.channel.probe import steps
from twin.channel.probe.images import ProbeImage, make_probe_images
from twin.channel.probe.model import (
    Action,
    ActionKind,
    ActionStatus,
    Attempt,
    AttemptPhase,
    PlanStatus,
    ProbePlan,
    SendRecord,
    StepId,
    StepState,
    StepStatus,
    end_plan,
)
from twin.channel.probe.store import NoProbePlan, ProbeStore
from twin.clock import Clock
from twin.llm.redaction import redact_text
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger

log = get_logger("twin.channel.probe.runner")

T = TypeVar("T")

ALERT_CATEGORY = "channel.probe"
ERRMSG_LIMIT = 200
LOGIN_PROBLEM = (
    "The WeChat login is not valid, so your message cannot arrive: run `twin channel login "
    "--force`. The probe carries on by itself once it works. "
)


@dataclass(frozen=True)
class InboundMarker:
    """What the probe needs to know about the user's messages and the login."""

    last_inbound_at: datetime | None
    context_fingerprint: str | None
    auth: AuthState
    bound: bool


class ProbeChannel(Protocol):
    """The part of a channel the probe uses (the WeChat channel through an adapter)."""

    def inbound_marker(self) -> InboundMarker: ...

    def register_image(self, data: bytes) -> str: ...

    async def send_text(self, text: str, *, bypass: SendBypass) -> OutboundResult: ...

    async def send_image(self, data: bytes, mime: str, *, bypass: SendBypass) -> OutboundResult: ...

    async def send_typing(self, active: bool) -> None: ...


class PlanChanged(Exception):
    """The plan is no longer what the tick looked at (replaced, or the attempt is gone)."""


def _clean(errmsg: str | None) -> str | None:
    return redact_text(errmsg)[:ERRMSG_LIMIT] if errmsg else None


def send_record(
    result: OutboundResult,
    *,
    at: datetime,
    inbound_at: datetime | None,
    due_at: datetime | None,
) -> SendRecord:
    """The numbers of one send, with the server's message redacted."""
    elapsed = (at - inbound_at).total_seconds() / 3600.0 if inbound_at else None
    late = max(0.0, (at - due_at).total_seconds()) if due_at else 0.0
    return SendRecord(
        ok=result.ok,
        outcome=result.kind.value,
        reason=result.reason,
        code=result.code,
        ret=result.ret,
        errcode=result.errcode,
        errmsg=_clean(result.errmsg),
        http_status=result.http_status,
        session_expired=result.session_expired,
        message_id=result.message_id,
        client_id=result.client_id,
        at=at,
        elapsed_h=round(elapsed, 4) if elapsed is not None else None,
        late_s=round(late, 1),
    )


def refused_record(reason: str, *, at: datetime, inbound_at: datetime | None) -> SendRecord:
    """A send the channel or the policy refused before anything was sent."""
    elapsed = (at - inbound_at).total_seconds() / 3600.0 if inbound_at else None
    return SendRecord(
        ok=False,
        outcome="refused",
        reason=reason,
        at=at,
        elapsed_h=round(elapsed, 4) if elapsed is not None else None,
    )


class ProbeRunner:
    """Moves the stored plan forward, one decision per :meth:`tick`."""

    def __init__(
        self,
        *,
        store: ProbeStore,
        channel: ProbeChannel,
        policy: SendBypass,
        clock: Clock,
        alerts: AlertSink,
        banner: AlertBanner,
        images: dict[str, ProbeImage] | None = None,
    ) -> None:
        self._store = store
        self._channel = channel
        self._policy = policy
        self._clock = clock
        self._alerts = alerts
        self._banner = banner
        self._images = images
        self._executing: str | None = None  # the action this process is carrying out right now

    # ------------------------------------------------------------- main loop

    async def run_forever(self) -> None:
        """Tick until cancelled; each tick says how long to wait."""
        while True:
            try:
                delay = await self.tick()
            except (PlanChanged, NoProbePlan):
                delay = 1.0
            await self._clock.sleep(delay)

    async def tick(self) -> float:
        """Do the next thing the stored plan needs; returns the seconds to wait."""
        plan = await asyncio.to_thread(self._store.load)
        if plan is None or plan.status is not PlanStatus.RUNNING:
            return plan.options.idle_poll_s if plan else 5.0
        step = plan.active_step()
        if step is None:
            await asyncio.to_thread(self._store.finish, PlanStatus.COMPLETED, None)
            await self._tell(
                "Channel probe finished", ["Run `twin channel probe report`."], plan, None
            )
            return plan.options.idle_poll_s
        attempt = step.current_attempt()
        if step.status is StepStatus.PENDING or attempt is None or attempt.outcome is not None:
            return await self._open_attempt(plan, step)
        if attempt.phase is AttemptPhase.ANNOUNCE:
            return await self._announce(plan, step, attempt)
        if attempt.phase is AttemptPhase.WAITING:
            return await self._wait_for_inbound(plan, step, attempt)
        return await self._run_attempt(plan, step, attempt)

    # --------------------------------------------------------------- helpers

    async def _update(self, change: Callable[[ProbePlan], T]) -> T:
        return await asyncio.to_thread(self._store.update, change)

    @staticmethod
    def _find(plan: ProbePlan, step_id: StepId, n: int) -> tuple[StepState, Attempt]:
        step = plan.step(step_id)
        if n < 1 or n > len(step.attempts):
            raise PlanChanged(f"{step_id.value} attempt {n} no longer exists")
        return step, step.attempts[n - 1]

    @staticmethod
    def _find_action(attempt: Attempt, action_id: str) -> Action:
        for action in attempt.actions:
            if action.id == action_id:
                return action
        raise PlanChanged(f"action {action_id} no longer exists")

    async def _tell(
        self, title: str, lines: list[str], plan: ProbePlan, step: StepState | None
    ) -> None:
        """Show a message in the console of the running application and note it as an alert."""
        self._banner.show(title, lines)
        key = f"{ALERT_CATEGORY}:{plan.run_id}:{step.id.value if step else 'end'}:{title}"

        def raise_alert() -> None:
            self._alerts.raise_alert(
                ALERT_CATEGORY,
                title,
                severity="info",
                detail={"run_id": plan.run_id, "lines": lines},
                dedup_key=key,
            )

        await asyncio.to_thread(raise_alert)

    def _image(self, name: str) -> ProbeImage:
        if self._images is None:
            self._images = make_probe_images()
        return self._images[name]

    # -------------------------------------------------------------- attempts

    async def _open_attempt(self, plan: ProbePlan, step: StepState) -> float:
        """Begin the next attempt of ``step`` (or skip the step, or give up on the plan)."""
        marker = await asyncio.to_thread(self._channel.inbound_marker)
        now = self._clock.now_utc()

        def change(p: ProbePlan) -> None:
            state = p.step(step.id)
            if state.status in (StepStatus.DONE, StepStatus.SKIPPED):
                raise PlanChanged("step already finished")
            current = state.current_attempt()
            if current is not None and current.outcome is None:
                raise PlanChanged("an attempt is already open")
            if len(state.attempts) >= p.options.max_attempts_per_step:
                end_plan(
                    p,
                    PlanStatus.STOPPED,
                    f"step {state.id.value} was void {state.voided_attempts} times in a row",
                    now,
                )
                return
            setup = steps.activation(p, state)
            if isinstance(setup, str):
                state.status = StepStatus.SKIPPED
                state.skip_reason = setup
                state.finished_at = now
                p.add_event(now, f"step {state.id.value} skipped: {setup}")
                return
            state.status = StepStatus.ACTIVE
            attempt = Attempt(
                n=len(state.attempts) + 1,
                phase=AttemptPhase.ANNOUNCE,
                armed_at=now,
                baseline_inbound_at=marker.last_inbound_at,
                budget=setup.budget,
                items=setup.items,
            )
            state.attempts.append(attempt)
            p.notice = steps.terminal_notice(p, state, attempt)
            p.add_event(now, f"step {state.id.value} attempt {attempt.n}: waiting for your message")

        await self._update(change)
        return 0.0

    async def _announce(self, plan: ProbePlan, step: StepState, attempt: Attempt) -> float:
        """Ask for the fresh message in the terminal and in WeChat, then wait for it."""
        text = steps.announce_text(plan, step, attempt)
        await self._tell(
            "Channel probe: send the bot a message",
            [
                steps.terminal_notice(plan, step, attempt),
                "The same request goes to WeChat while the platform still accepts messages.",
            ],
            plan,
            step,
        )
        note = "sent"
        try:
            result = await self._channel.send_text(text, bypass=self._policy)
            if not result.ok:
                note = f"not delivered ({result.kind.value}: {result.reason})"
        except BypassRefused:
            raise PlanChanged("the plan is no longer running") from None
        except RecipientNotAllowed:
            await asyncio.to_thread(
                self._store.finish, PlanStatus.STOPPED, "no user is bound to the channel"
            )
            raise PlanChanged("nobody is bound") from None
        except ChannelError as exc:
            note = f"not sent ({type(exc).__name__})"
        now = self._clock.now_utc()

        def change(p: ProbePlan) -> None:
            _state, current = self._find(p, step.id, attempt.n)
            current.phase = AttemptPhase.WAITING
            current.announced_at = now
            current.announce_note = note

        await self._update(change)
        return plan.options.poll_s

    async def _wait_for_inbound(self, plan: ProbePlan, step: StepState, attempt: Attempt) -> float:
        """Notice the user's fresh message; start measuring once it has settled."""
        marker = await asyncio.to_thread(self._channel.inbound_marker)
        last = marker.last_inbound_at
        baseline = attempt.baseline_inbound_at
        options = plan.options
        notice = steps.terminal_notice(plan, step, attempt)
        if marker.auth is not AuthState.OK:
            notice = LOGIN_PROBLEM + notice
        if plan.notice != notice:
            await self._update(lambda p: setattr(p, "notice", notice))
        if last is None or (baseline is not None and last <= baseline):
            return options.poll_s
        now = self._clock.now_utc()
        if attempt.candidate_inbound_at is None or last > attempt.candidate_inbound_at:

            def settle(p: ProbePlan) -> None:
                _state, current = self._find(p, step.id, attempt.n)
                current.candidate_inbound_at = last

            await self._update(settle)
            return options.settle_s  # a second message right behind the first moves the start

        def begin(p: ProbePlan) -> None:
            state, current = self._find(p, step.id, attempt.n)
            current.phase = AttemptPhase.RUNNING
            current.inbound_at = last
            current.started_at = now
            current.context_fp = marker.context_fingerprint
            current.actions = steps.build_actions(p, state, current)
            anchor = last if state.id is StepId.WINDOW else now
            for action in current.actions:
                action.due_at = anchor + timedelta(seconds=action.due_offset_s)
            p.notice = self._running_notice(p, state, current)
            p.add_event(now, f"step {state.id.value} attempt {current.n}: your message arrived")

        await self._update(begin)
        log.info("probe_attempt_started", probe_step=step.id.value, attempt=attempt.n)
        return 0.0

    @staticmethod
    def _running_notice(plan: ProbePlan, step: StepState, attempt: Attempt) -> str:
        if step.id is StepId.WINDOW and attempt.inbound_at is not None:
            end = attempt.inbound_at + timedelta(hours=max(plan.options.window_hours))
            return (
                "Window test running: do NOT message the bot until "
                f"{end:%Y-%m-%d %H:%M} UTC (a message voids the test). Questions come at the end."
            )
        return f"Step {step.id.value} is running; do not message the bot until it asks you."

    # ------------------------------------------------------------ the actions

    async def _run_attempt(self, plan: ProbePlan, step: StepState, attempt: Attempt) -> float:
        marker = await asyncio.to_thread(self._channel.inbound_marker)
        sending = any(
            a.kind in (ActionKind.SEND_TEXT, ActionKind.SEND_IMAGE)
            and a.status in (ActionStatus.PENDING, ActionStatus.ACTIVE)
            for a in attempt.actions
        )
        if (
            sending  # once every send is over, a message from the user can no longer disturb it
            and marker.last_inbound_at is not None
            and attempt.inbound_at is not None
            and marker.last_inbound_at > attempt.inbound_at
        ):
            await self._void(plan, step, attempt, "you wrote to the bot during the measurement")
            return 0.0
        action = next(
            (a for a in attempt.actions if a.status in (ActionStatus.PENDING, ActionStatus.ACTIVE)),
            None,
        )
        if action is None:
            await self._finish_attempt(plan, step, attempt)
            return 0.0
        if (
            action.status is ActionStatus.ACTIVE
            and action.kind is not ActionKind.ASK
            and self._executing != action.id
        ):
            await self._recover(plan, step, attempt, action)  # the process restarted mid-send
            return 0.0
        if action.kind is ActionKind.ASK:
            return await self._ask(plan, step, attempt, action)
        if action.kind is ActionKind.TYPING:
            return await self._typing(plan, step, attempt, action)
        due = action.due_at or self._clock.now_utc()
        wait = (due - self._clock.now_utc()).total_seconds()
        if wait > 0:
            return min(wait, plan.options.watch_s)
        return await self._send(plan, step, attempt, action)

    async def _send(
        self, plan: ProbePlan, step: StepState, attempt: Attempt, action: Action
    ) -> float:
        started = self._clock.now_utc()

        def mark(p: ProbePlan) -> None:
            if p.status is not PlanStatus.RUNNING:
                raise PlanChanged("the plan was stopped")
            _s, current = self._find(p, step.id, attempt.n)
            found = self._find_action(current, action.id)
            found.status = ActionStatus.ACTIVE
            found.started_at = started
            found.tries += 1

        await self._update(mark)
        self._executing = action.id
        try:
            result: OutboundResult | None = None
            refusal: str | None = None
            try:
                if action.kind is ActionKind.SEND_IMAGE and action.image is not None:
                    image = self._image(action.image)
                    await asyncio.to_thread(self._channel.register_image, image.data)
                    result = await self._channel.send_image(
                        image.data, image.mime, bypass=self._policy
                    )
                else:
                    result = await self._channel.send_text(action.text or "", bypass=self._policy)
            except (
                BypassRefused,
                RecipientNotAllowed,
                MediaNotAllowed,
                CapabilityNotSupported,
            ) as exc:
                refusal = type(exc).__name__
        finally:
            self._executing = None
        record = (
            send_record(result, at=started, inbound_at=attempt.inbound_at, due_at=action.due_at)
            if result is not None
            else refused_record(refusal or "refused", at=started, inbound_at=attempt.inbound_at)
        )
        await self._update(lambda p: self._apply_send(p, step.id, attempt.n, action.id, record))
        return 0.0

    def _apply_send(
        self, plan: ProbePlan, step_id: StepId, n: int, action_id: str, record: SendRecord
    ) -> None:
        """Store the outcome of a send; a failure ends the attempt's sends."""
        _state, attempt = self._find(plan, step_id, n)
        action = self._find_action(attempt, action_id)
        now = self._clock.now_utc()
        action.send = record
        action.finished_at = now
        if record.ok:
            action.status = ActionStatus.DONE
            plan.add_event(now, f"{action.label}: accepted by the server")
            return
        retry = record.outcome == "network" and action.tries <= plan.options.network_retries
        if retry:  # nothing left this machine, so sending again cannot double a message
            action.status = ActionStatus.PENDING
            action.due_at = now + timedelta(seconds=plan.options.network_retry_s)
            action.note = f"no network, try {action.tries}"
            plan.add_event(now, f"{action.label}: no network, trying again in a minute")
            return
        action.status = ActionStatus.FAILED
        detail = record.outcome + (f" code {record.code}" if record.code is not None else "")
        plan.add_event(now, f"{action.label}: failed ({detail}); no more sends in this attempt")
        self._skip_after_failure(attempt, action)

    @staticmethod
    def _skip_after_failure(attempt: Attempt, failed: Action) -> None:
        """After a failed send nothing else is sent or shown in this attempt."""
        after = False
        for action in attempt.actions:
            if action is failed:
                after = True
                continue
            if (
                after
                and action.kind in (ActionKind.SEND_TEXT, ActionKind.SEND_IMAGE, ActionKind.TYPING)
                and action.status is ActionStatus.PENDING
            ):
                action.status = ActionStatus.SKIPPED

    async def _recover(
        self, plan: ProbePlan, step: StepState, attempt: Attempt, action: Action
    ) -> None:
        """A send was in flight when the application stopped: its result is unknown."""
        now = self._clock.now_utc()
        record = SendRecord(ok=False, outcome="unknown", reason="interrupted_by_restart", at=now)

        def change(p: ProbePlan) -> None:
            _s, current = self._find(p, step.id, attempt.n)
            found = self._find_action(current, action.id)
            found.send = record
            found.status = ActionStatus.FAILED
            found.finished_at = now
            self._skip_after_failure(current, found)
            p.add_event(now, f"{found.label}: the application restarted during this step")

        await self._update(change)

    async def _ask(
        self, plan: ProbePlan, step: StepState, attempt: Attempt, action: Action
    ) -> float:
        if action.status is ActionStatus.PENDING:
            question = steps.question_for(plan, step, attempt, action)
            now = self._clock.now_utc()
            if question is None:

                def skip(p: ProbePlan) -> None:
                    _s, current = self._find(p, step.id, attempt.n)
                    self._find_action(current, action.id).status = ActionStatus.SKIPPED

                await self._update(skip)
                return 0.0
            question.asked_at = now

            def ask(p: ProbePlan) -> None:
                _s, current = self._find(p, step.id, attempt.n)
                found = self._find_action(current, action.id)
                found.status = ActionStatus.ACTIVE
                found.question = question
                p.notice = "Answer the question: `twin channel probe answer`"
                p.add_event(now, f"question for you: {question.id}")

            await self._update(ask)
            await self._tell(
                "Channel probe: a question for you",
                [question.prompt, "Answer it with `twin channel probe answer`."],
                plan,
                step,
            )
            return plan.options.poll_s
        if action.question is not None and action.question.answered:
            now = self._clock.now_utc()

            def done(p: ProbePlan) -> None:
                _s, current = self._find(p, step.id, attempt.n)
                found = self._find_action(current, action.id)
                found.status = ActionStatus.DONE
                found.finished_at = now
                p.notice = None

            await self._update(done)
            return 0.0
        return plan.options.poll_s

    async def _typing(
        self, plan: ProbePlan, step: StepState, attempt: Attempt, action: Action
    ) -> float:
        started = self._clock.now_utc()
        hold = plan.options.typing_hold_s

        def mark(p: ProbePlan) -> None:
            if p.status is not PlanStatus.RUNNING:
                raise PlanChanged("the plan was stopped")
            _s, current = self._find(p, step.id, attempt.n)
            found = self._find_action(current, action.id)
            found.status = ActionStatus.ACTIVE
            found.started_at = started

        await self._update(mark)
        self._executing = action.id
        try:
            await self._channel.send_typing(True)
            await self._clock.sleep(hold)
        finally:
            self._executing = None
            await asyncio.shield(self._channel.send_typing(False))
        now = self._clock.now_utc()

        def done(p: ProbePlan) -> None:
            _s, current = self._find(p, step.id, attempt.n)
            found = self._find_action(current, action.id)
            found.status = ActionStatus.DONE
            found.finished_at = now
            found.note = f"{hold:g}"
            p.add_event(now, f"typing indicator held for {hold:g} s")

        await self._update(done)
        return 0.0

    # --------------------------------------------------------------- judging

    async def _void(self, plan: ProbePlan, step: StepState, attempt: Attempt, reason: str) -> None:
        now = self._clock.now_utc()

        def change(p: ProbePlan) -> None:
            _s, current = self._find(p, step.id, attempt.n)
            if current.outcome is not None:
                return
            current.outcome = "void"
            current.void_reason = reason
            current.phase = AttemptPhase.FINISHED
            current.finished_at = now
            p.notice = None
            for action in current.actions:
                if action.status in (ActionStatus.PENDING, ActionStatus.ACTIVE):
                    action.status = ActionStatus.SKIPPED
            p.add_event(now, f"step {step.id.value} attempt {current.n} void: {reason}")

        await self._update(change)
        log.warning("probe_attempt_void", probe_step=step.id.value, attempt=attempt.n, why=reason)
        await self._tell(
            "Channel probe: this attempt is void",
            [f"Reason: {reason}.", "It will be redone: the probe asks for a fresh message again."],
            plan,
            step,
        )

    async def _finish_attempt(self, plan: ProbePlan, step: StepState, attempt: Attempt) -> None:
        """All actions are over: judge the attempt, fold its result into the step."""
        now = self._clock.now_utc()

        def change(p: ProbePlan) -> str | None:
            state, current = self._find(p, step.id, attempt.n)
            if current.outcome is not None:
                return None
            verdict = steps.finalize(p, state, current)
            current.phase = AttemptPhase.FINISHED
            current.finished_at = now
            p.notice = None
            if verdict.void:
                current.outcome = "void"
                current.void_reason = verdict.void_reason
                p.add_event(
                    now, f"step {state.id.value} attempt {current.n} void: {verdict.void_reason}"
                )
                return verdict.void_reason
            current.outcome = "complete"
            if verdict.step_done:
                state.status = StepStatus.DONE
                state.finished_at = now
            p.add_event(now, f"step {state.id.value} attempt {current.n} complete")
            return None

        void_reason = await self._update(change)
        log.info(
            "probe_attempt_finished",
            probe_step=step.id.value,
            attempt=attempt.n,
            outcome="void" if void_reason else "complete",
        )
        if void_reason:
            await self._tell(
                "Channel probe: this attempt is void",
                [f"Reason: {void_reason}.", "The probe asks for a fresh message again."],
                plan,
                step,
            )
