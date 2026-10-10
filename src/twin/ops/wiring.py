"""Putting the operations of round 12 into the application (R-OPS-001 to R-OPS-006).

:func:`register_ops` adds to a :class:`~twin.app.Application`:

* ``alert_delivery`` - the notifications and e-mails of :mod:`twin.ops.alerts`;
* ``health_monitor`` - the minute-by-minute check (:mod:`twin.ops.monitor`);
* ``ops_scheduler`` - the daily backup and the monthly cost mail (:mod:`twin.ops.scheduler`);
* ``login_recovery`` - the QR window when the WeChat login is lost (WeChat channel only).

The builders (:func:`build_mailer`, :func:`build_emergency_notifier`) are the one place where the
SMTP account of ``ops.smtp`` and the credential ``smtp_password`` are put together.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from twin.app import Application
from twin.engine.backend_select import STYLE_BACKENDS, BackendSelector
from twin.engine.safety.notifier import EmergencyNotifier
from twin.ops.alert_delivery import AlertDelivery
from twin.ops.backup.service import BackupService
from twin.ops.emergency import SmtpEmergencyNotifier
from twin.ops.health import HealthCollector, LiveSources, StyleReading
from twin.ops.login_recovery import LoginRecovery
from twin.ops.mail import SMTP_PASSWORD_SECRET, Mailer, SmtpMailer
from twin.ops.monitor import HealthMonitor
from twin.ops.notify import Notifier, default_notifier
from twin.ops.scheduler import OpsScheduler, backup_task, cost_report_task
from twin.schedule.component import ScheduleComponent
from twin.schedule.events import Resumed
from twin.schedule.service import time_service_for
from twin.services import Services

__all__ = [
    "OpsKit",
    "build_emergency_notifier",
    "build_mailer",
    "register_ops",
    "style_reading_of",
]


def build_mailer(services: Services) -> SmtpMailer:
    """The SMTP client of ``ops.smtp`` with the password from the credential store."""
    return SmtpMailer(
        services.settings.ops.smtp, lambda: services.secrets.get(SMTP_PASSWORD_SECRET)
    )


def build_emergency_notifier(services: Services, mailer: Mailer | None = None) -> EmergencyNotifier:
    """The delivery of the one fixed message to the emergency contact (R-SAFE-001)."""
    return SmtpEmergencyNotifier(
        mailer or build_mailer(services),
        services.db,
        services.clock,
        services.settings.safety,
        time_service_for(services).bot_timezone,
    )


def style_reading_of(selector: BackendSelector) -> Callable[[], Awaitable[StyleReading]]:
    """How the health check asks the style model: through the selector's cached probe."""

    async def read() -> StyleReading:
        needed = selector.requested() in STYLE_BACKENDS or selector.fallback() is not None
        if not needed:
            return StyleReading(False, True)
        probe = await selector.probe()
        return StyleReading(True, probe.usable, probe.code)

    return read


@dataclass
class OpsKit:
    """The components :func:`register_ops` added (the CLI and the tests reach them here)."""

    delivery: AlertDelivery
    monitor: HealthMonitor
    collector: HealthCollector
    scheduler: OpsScheduler
    backup: BackupService
    login: LoginRecovery | None = None


def register_ops(
    application: Application,
    services: Services,
    *,
    style: BackendSelector | None = None,
    mailer: Mailer | None = None,
    notifier: Notifier | None = None,
    started_at: datetime | None = None,
    busy: Callable[[], bool] | None = None,
    schedule: ScheduleComponent | None = None,
) -> OpsKit:
    """Add the operations components to ``application`` (see the module description)."""
    clock = services.clock
    mailer = mailer if mailer is not None else build_mailer(services)
    time = time_service_for(services)
    delivery = AlertDelivery(
        services.alerts,
        clock,
        notifier=notifier if notifier is not None else default_notifier(),
        mailer=mailer,
        recipient=lambda: services.settings.ops.smtp.to,
        zone=time.bot_timezone,
    )
    collector = HealthCollector(
        services,
        started_at=started_at,
        live=LiveSources(
            components=application.health,
            style=style_reading_of(style) if style is not None else None,
        ),
    )
    monitor = HealthMonitor(services, collector, started_at=started_at)
    backup = BackupService.from_services(services)
    scheduler = OpsScheduler(
        services,
        [backup_task(services, backup, busy=busy), cost_report_task(services, mailer)],
        zone=time.bot_timezone,
    )
    if schedule is not None:

        async def announce_wake(event: Resumed) -> None:
            if event.kind != "wake":
                return
            minutes = event.gap_s / 60
            await asyncio.to_thread(
                services.alerts.raise_alert,
                "system_resumed",
                f"the computer was asleep for {minutes:.0f} minutes",
                severity="warning",
                detail={"gap_s": round(event.gap_s)},
                dedup_key="system_resumed",
            )

        schedule.events.subscribe(Resumed, announce_wake)
    application.register(delivery)
    application.register(monitor)
    application.register(scheduler)
    login = None
    if services.settings.channel.kind == "ilink":
        login = LoginRecovery(services)
        application.register(login)
    return OpsKit(delivery, monitor, collector, scheduler, backup, login)
