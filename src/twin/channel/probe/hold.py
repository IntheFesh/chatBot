"""``ProbeHold``: while a probe measures, the channel sends nothing but the probe's own messages.

The probe counts how many messages the platform lets the bot send after one message of the user
(R-CH-009).  A reply, a message of her own or the answer to a command, sent meanwhile, would be
counted with the probe's and spoil the number.  The engine and the proactive scheduler therefore
hold their work while :meth:`ProbeStore.active` says a plan runs; this object is the last line, in
the sender itself (D-684):

* a send without the probe's permission is refused with ``window_rejected`` / ``probe_running`` -
  a result the engine already knows how to treat (it stores no bubble for it) - and the refusal is
  written to the probe's audit trail (the length of the text, never the text);
* the one exception is the **urgent** send of the crisis answer (R-SAFE-001, D-685): safety wins
  over the measurement.  It goes out and the attempt it disturbed is void: the probe says so in
  ``twin channel probe status`` and asks the user for a fresh message to measure again.

The sends of the probe itself do not come here: they carry the ``ProbeSendPolicy`` and are judged
by it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from twin.channel.probe.model import ProbePlan
from twin.channel.probe.store import ProbeStore, audit_entry
from twin.ops.logging import get_logger

log = get_logger("twin.channel.probe.hold")

REASON_REFUSED = "probe_running"
REASON_URGENT = "urgent_during_probe"
VOID_REASON = (
    "the bot sent a safety message (the answer to a crisis) during the attempt, "
    "so its count cannot be trusted"
)


class ProbeHold:
    """What the WeChat sender asks of the probe (see the module description)."""

    def __init__(self, store: ProbeStore) -> None:
        self._store = store

    def active(self) -> bool:
        """Is a probe plan running (one small read of the database)?"""
        return self._store.active()

    def refused(self, kind: str, text: str | None, now: datetime) -> None:
        """Write down that a send was held back for the probe."""
        self._audit("refused", REASON_REFUSED, kind, text, now)

    def urgent_sent(self, kind: str, text: str | None, now: datetime) -> None:
        """A safety message goes out while the probe runs: write it down and void the attempt."""
        self._audit("urgent", REASON_URGENT, kind, text, now)
        try:
            voided = self._store.void_current_attempt(VOID_REASON)
        except Exception as exc:  # a broken plan must not stand in the way of a safety message
            log.warning("probe_attempt_not_voided", error_type=type(exc).__name__)
            return
        log.warning("probe_attempt_void_by_urgent_send", voided=voided)

    def _audit(self, decision: str, why: str, kind: str, text: str | None, now: datetime) -> None:
        plan = self._plan()
        step = plan.active_step() if plan is not None else None
        entry: dict[str, Any] = audit_entry(
            at=now,
            decision=decision,
            why=why,
            kind=kind,
            text=text,
            gate_reason=None,
            run_id=plan.run_id if plan is not None else None,
            step=step.id.value if step is not None else None,
        )
        try:
            self._store.add_audit(entry)
        except Exception as exc:  # the audit is a record, never a reason to lose the answer
            log.warning("probe_audit_not_written", error_type=type(exc).__name__)
        log.info("probe_hold", decision=decision, why=why, send_kind=kind)

    def _plan(self) -> ProbePlan | None:
        try:
            return self._store.load()
        except Exception:
            return None
