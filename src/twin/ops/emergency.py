"""The message to the emergency contact: the mail that can only say "he may need care" (R-SAFE-001).

:class:`SmtpEmergencyNotifier` is the delivery that round 09 left as an interface
(:class:`~twin.engine.safety.notifier.EmergencyNotifier`).  What it sends is decided by the types,
not by care: an :class:`~twin.engine.safety.notifier.EmergencyNotice` has no field for text, and
:func:`~twin.engine.safety.notifier.render_notice` fills the one fixed template from the moment
alone - so there is no path by which chat content could get into this mail.

Three rules on top:

* the recipient is the address in ``safety.emergency_contact.email`` and nothing else - a notice
  naming another address is refused (R-SAFE-004);
* **one mail per event**: a crisis goes on for several messages, and the contact is told once for
  it.  A second notice within ``EPISODE_GAP`` of the last one belongs to the same event and is not
  sent (the moment is kept in the setting ``safety.emergency.last``);
* it is only used when ``safety.emergency_contact.enabled`` is on and an address is set; the
  crisis handler checks that before it asks, and ``twin setup`` explains what is sent when the
  user switches it on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from twin.clock import Clock
from twin.config.settings import SafetyConfig
from twin.engine.safety.notifier import EmergencyNotice, render_notice
from twin.ops.logging import get_logger
from twin.ops.mail import Mailer, MailError, OutgoingMail
from twin.storage.db import Database
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.emergency")

LAST_KEY = "safety.emergency.last"
EPISODE_GAP = timedelta(hours=12)


class SmtpEmergencyNotifier:
    """Sends the fixed notice by e-mail, once per event (see the module description)."""

    def __init__(
        self,
        mailer: Mailer,
        db: Database,
        clock: Clock,
        safety: SafetyConfig,
        zone: Callable[[], ZoneInfo],
    ) -> None:
        self._mailer = mailer
        self._db = db
        self._clock = clock
        self._safety = safety
        self._zone = zone

    def _last(self) -> datetime | None:
        with self._db.session() as session:
            raw = get_setting(session, LAST_KEY)
        return datetime.fromisoformat(raw) if isinstance(raw, str) else None

    def _remember(self, at: datetime) -> None:
        with self._db.transaction(bump_state=False) as session:
            put_setting(
                session,
                LAST_KEY,
                at.isoformat(),
                clock=self._clock,
                by="safety",
                record_history=False,
            )

    async def notify(self, notice: EmergencyNotice) -> bool:
        """Send it (``True``), skip it as the same event (``False``), or raise ``MailError``."""
        configured = (self._safety.emergency_contact.email or "").strip()
        if not self._safety.emergency_contact.enabled or not configured:
            return False
        if notice.recipient.strip().lower() != configured.lower():
            raise MailError("recipient")  # the one address is the configured one (R-SAFE-004)
        last = await asyncio.to_thread(self._last)
        if last is not None and notice.at - last < EPISODE_GAP:
            log.info("emergency_notice_skipped", reason="same_event")
            return False
        message = render_notice(notice, self._zone())
        await asyncio.to_thread(
            self._mailer.send, OutgoingMail(configured, message.subject, message.body)
        )
        await asyncio.to_thread(self._remember, notice.at)
        log.info("emergency_notice_sent")
        return True
