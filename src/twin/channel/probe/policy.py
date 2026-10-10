"""``ProbeSendPolicy``: the one object allowed to skip the safe thresholds (R-CH-009).

The probe has to send past ``channel.proactive_window_safe_h`` and
``channel.outbound_quota_safe``, otherwise it could never find the real edge.  It does so by
passing this policy as ``bypass=`` to ``IlinkChannel.send_text`` / ``send_image``.  The policy
authorises a send only when all of these hold, and writes an audit record either way:

* a probe plan is running (stopping the plan switches the permission off at once);
* the plan is in an attempt that is announcing or measuring (never between attempts);
* a text starts with ``[测试]``; a picture carries no text (its bytes were already checked
  against the probe's own picture manifest by the media allow list).

It bypasses *thresholds* only: the channel still refuses an expired login, a missing
``context_token`` and a window the platform already closed.  Nothing in the engine or in the
proactive path holds a policy object, and ``tests/unit/test_channel_probe_policy.py`` scans
the source to keep it that way.
"""

from __future__ import annotations

from twin.channel.base import (
    TEST_PREFIX,
    BypassGrant,
    BypassRefused,
    BypassRequest,
)
from twin.channel.probe.model import (
    ActionKind,
    ActionStatus,
    AttemptPhase,
    PlanStatus,
    ProbePlan,
    StepId,
)
from twin.channel.probe.store import ProbeStore, audit_entry
from twin.ops.logging import get_logger

log = get_logger("twin.channel.probe.policy")


class ProbeSendPolicy:
    """Authorises the probe's own sends; refuses everything else."""

    def __init__(self, store: ProbeStore) -> None:
        self._store = store

    def authorize(self, request: BypassRequest) -> BypassGrant | None:
        plan: ProbePlan | None = None
        try:
            plan = self._store.load()
            why = self._refusal(plan, request)
        except Exception:  # an unreadable plan must never turn into a permission
            why = "plan_unreadable"
        step = plan.active_step() if plan else None
        step_id = step.id.value if step else None
        empty_token = why is None and plan is not None and self._empty_token_requested(plan)
        decision = "refused" if why else "allowed"
        self._store.add_audit(
            audit_entry(
                at=request.now,
                decision=decision,
                why=why or "ok",
                kind=request.kind,
                text=request.text,
                gate_reason=request.gate_reason,
                run_id=plan.run_id if plan else None,
                step=step_id,
                empty_token=empty_token,
            )
        )
        log.info(
            "probe_bypass",
            decision=decision,
            why=why or "ok",
            send_kind=request.kind,
            gate_reason=request.gate_reason,
            step=step_id,
        )
        if why is not None:
            raise BypassRefused(f"the probe policy does not allow this send ({why})")
        return BypassGrant(empty_context_token=True) if empty_token else None

    @staticmethod
    def _refusal(plan: ProbePlan | None, request: BypassRequest) -> str | None:
        if plan is None or plan.status is not PlanStatus.RUNNING:
            return "probe_not_active"
        step = plan.active_step()
        attempt = step.current_attempt() if step else None
        if attempt is None or attempt.phase not in (AttemptPhase.ANNOUNCE, AttemptPhase.RUNNING):
            return "no_attempt_in_progress"
        if request.kind == "text":
            if request.text is None or not request.text.startswith(TEST_PREFIX):
                return "missing_test_prefix"
            return None
        if request.kind == "image":
            return "text_with_picture" if request.text is not None else None
        return "unknown_kind"

    @staticmethod
    def _empty_token_requested(plan: ProbePlan) -> bool:
        """True only while the optional experiment's own send is the action being carried out."""
        step = plan.active_step()
        attempt = step.current_attempt() if step else None
        if step is None or attempt is None or step.id is not StepId.EMPTY_TOKEN:
            return False
        return plan.options.empty_token_experiment and any(
            action.kind is ActionKind.SEND_TEXT
            and action.empty_token
            and action.status is ActionStatus.ACTIVE
            for action in attempt.actions
        )
