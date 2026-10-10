"""Delivering alerts: the notification and the e-mail (R-OPS-004).

:class:`AlertDelivery` is the application component that takes what
:class:`~twin.ops.alerts.AlertService` queued and sends it:

* the **notification** (Windows toast, or the console panel elsewhere) is shown at once, in a
  worker thread;
* the **e-mail** (``ops.smtp``; HTML and plain text) goes out right after; if it fails the alert
  stays pending for it and is tried again after 1, 5, 15 and 60 minutes (then hourly), eight
  attempts in all; a failing mail server never delays the notification and never reaches the
  caller of ``raise_alert``.

It runs in ``twin run`` and in ``twin supervise`` (the supervisor must be able to say that
``twin run`` died while no application is there to say it).  Two processes may run it at once:
a delivery is leased per alert (``claimed_at``), so each notice is sent by one of them.

When no e-mail account is set up the mail part is skipped (and said so once in the log); the
notification still works.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from datetime import timedelta
from zoneinfo import ZoneInfo

from twin.app import ComponentHealth, HealthStatus, TaskSupervisor
from twin.clock import Clock
from twin.ops.alert_text import render_alert
from twin.ops.alerts import AlertService, AlertView
from twin.ops.logging import get_logger
from twin.ops.mail import Mailer, MailError, OutgoingMail
from twin.ops.notify import Notifier

log = get_logger("twin.alert_delivery")

POLL_S = 2.0
RETRY_AFTER = (
    timedelta(minutes=1),
    timedelta(minutes=5),
    timedelta(minutes=15),
    timedelta(minutes=60),
)
MAX_MAIL_ATTEMPTS = 8


class AlertDelivery:
    """Sends the queued notices (see the module description)."""

    name = "alert_delivery"
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        service: AlertService,
        clock: Clock,
        *,
        notifier: Notifier | None,
        mailer: Mailer | None,
        recipient: Callable[[], str | None],
        zone: Callable[[], ZoneInfo],
        poll_s: float = POLL_S,
    ) -> None:
        self._service = service
        self._clock = clock
        self._notifier = notifier
        self._mailer = mailer
        self._recipient = recipient
        self._zone = zone
        self._poll_s = poll_s
        self._supervisor = TaskSupervisor(self.name, clock, None)
        self._event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._warned_mail = False
        self.delivered = 0

    # -------------------------------------------------------------------- one pass

    async def deliver_once(self) -> int:
        """Send everything that is due; returns how many alerts were looked at."""
        views = await asyncio.to_thread(self._service.claim_due)
        for view in views:
            await self._deliver(view)
        return len(views)

    async def _deliver(self, view: AlertView) -> None:
        rendered = render_alert(view, self._zone())
        if view.toast_state == "pending":
            await self._toast(view, rendered.toast_title, rendered.toast_body)
        if view.mail_state == "pending":
            await self._mail(view, rendered.subject, rendered.text, rendered.html)
        self.delivered += 1

    async def _toast(self, view: AlertView, title: str, body: str) -> None:
        notifier = self._notifier
        if notifier is None:
            await asyncio.to_thread(self._service.finish_toast, view.id, ok=None)
            return
        try:
            await asyncio.to_thread(notifier.notify, title, body)
        except Exception as exc:  # a notice that cannot be shown must not stop the others
            log.warning("notification_failed", category=view.category, reason=type(exc).__name__)
            await asyncio.to_thread(self._service.finish_toast, view.id, ok=False)
            return
        await asyncio.to_thread(self._service.finish_toast, view.id, ok=True)

    async def _mail(self, view: AlertView, subject: str, text: str, page: str) -> None:
        mailer, to = self._mailer, self._recipient()
        if mailer is None or not mailer.configured or not to:
            if not self._warned_mail:
                self._warned_mail = True
                log.warning("alert_mail_not_configured")
            await asyncio.to_thread(self._service.finish_mail, view.id, ok=None)
            return
        now = self._clock.now_utc()
        if view.mail_next_at is not None and view.mail_next_at > now:
            await asyncio.to_thread(self._service.release, view.id)
            return
        try:
            await asyncio.to_thread(mailer.send, OutgoingMail(to, subject, text, page))
        except MailError as exc:
            await self._mail_failed(view, exc.code)
            return
        except Exception as exc:  # whatever else a mail library raises
            await self._mail_failed(view, type(exc).__name__)
            return
        await asyncio.to_thread(self._service.finish_mail, view.id, ok=True)

    async def _mail_failed(self, view: AlertView, code: str) -> None:
        attempts = view.mail_attempts + 1
        retry = None
        if attempts < MAX_MAIL_ATTEMPTS:
            retry = RETRY_AFTER[min(attempts - 1, len(RETRY_AFTER) - 1)]
        log.warning(
            "alert_mail_failed",
            category=view.category,
            reason=code,
            attempt=attempts,
            gave_up=retry is None,
        )
        await asyncio.to_thread(
            self._service.finish_mail, view.id, ok=False, error=code, retry_in=retry
        )

    # --------------------------------------------------------------------- component

    def _poke(self) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._event.set)

    async def _wait(self) -> None:
        sleeper = asyncio.ensure_future(self._clock.sleep(self._poll_s))
        waker = asyncio.ensure_future(self._event.wait())
        try:
            await asyncio.wait({sleeper, waker}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sleeper, waker):
                task.cancel()
            await asyncio.gather(sleeper, waker, return_exceptions=True)
        self._event.clear()

    async def _run(self) -> None:
        while True:
            await self.deliver_once()
            await self._wait()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._service.reset_claims)
        self._service.set_wake(self._poke)
        self._supervisor.spawn("deliver", self._run, restart_on_exit=True)
        self._poke()

    async def stop(self) -> None:
        self._service.set_wake(None)
        with contextlib.suppress(Exception):
            await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        base = self._supervisor.health()
        if base.status is not HealthStatus.OK:
            return base
        return ComponentHealth(HealthStatus.OK)
