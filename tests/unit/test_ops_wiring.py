"""The operations of round 12 inside the application (R-OPS-001 to R-OPS-006, R-SAFE-001)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from tests.support.clock import ManualClock
from tests.support.ops import RecordingMailer, RecordingNotifier
from tests.support.routine import Rig, fixed_model
from twin.app import Application
from twin.engine.backend_select import Probe
from twin.engine.safety.notifier import EmergencyNotice
from twin.ops.emergency import SmtpEmergencyNotifier
from twin.ops.health import HealthCheck, HealthLevel, StyleReading
from twin.ops.mail import SMTP_PASSWORD_SECRET, SmtpMailer
from twin.ops.wiring import (
    build_emergency_notifier,
    build_mailer,
    register_ops,
    style_reading_of,
)
from twin.schedule.component import ScheduleComponent
from twin.schedule.events import Resumed
from twin.services import Services

AT = datetime(2026, 10, 9, 12, tzinfo=UTC)


def test_the_four_components_are_added_with_the_wechat_channel(services: Services) -> None:
    application = Application()
    kit = register_ops(
        application, services, mailer=RecordingMailer(), notifier=RecordingNotifier()
    )
    assert sorted(application.components) == [
        "alert_delivery",
        "health_monitor",
        "login_recovery",
        "ops_scheduler",
    ]
    assert kit.login is not None and kit.delivery.name == "alert_delivery"
    assert kit.monitor.name == "health_monitor" and kit.scheduler.name == "ops_scheduler"
    assert kit.backup.backups_dir == services.paths.backups_dir
    assert kit.collector is not None


def test_without_the_wechat_channel_there_is_no_login_window(services: Services) -> None:
    services.settings.channel.kind = "console"
    application = Application()
    kit = register_ops(
        application, services, mailer=RecordingMailer(), notifier=RecordingNotifier()
    )
    assert kit.login is None and "login_recovery" not in application.components


def test_the_defaults_are_the_smtp_mailer_and_the_console_or_toast_notifier(
    services: Services,
) -> None:
    application = Application()
    kit = register_ops(application, services)
    assert isinstance(build_mailer(services), SmtpMailer)
    assert kit.delivery is not None and len(application.components) == 4


async def test_a_wake_from_sleep_is_an_alert_and_a_start_is_not(
    services: Services, clock: ManualClock
) -> None:
    rig = Rig.build(services, clock, fixed_model())
    schedule = ScheduleComponent(services, kit=rig.kit)
    register_ops(
        Application(),
        services,
        mailer=RecordingMailer(),
        notifier=RecordingNotifier(),
        schedule=schedule,
    )
    await schedule.events.publish(Resumed(AT, "startup", 0.0, None))
    assert services.alerts.recent() == []
    await schedule.events.publish(Resumed(AT, "wake", 1800.0, None))
    (alert,) = services.alerts.recent()
    assert alert.category == "system_resumed" and alert.severity == "warning"
    assert "30 minutes" in alert.title and alert.detail == {"gap_s": 1800}


def test_the_password_comes_from_the_credential_store_only(services: Services) -> None:
    services.settings.ops.smtp.host = "smtp.example.org"
    services.settings.ops.smtp.user = "bot@example.org"
    services.settings.ops.smtp.to = "me@example.org"
    mailer = build_mailer(services)
    assert mailer.configured
    services.secrets.set(SMTP_PASSWORD_SECRET, "synthetic-app-password")
    assert mailer._password() == "synthetic-app-password"  # read when used, not when built


async def test_the_emergency_notifier_is_the_smtp_one_with_the_safety_settings(
    services: Services,
) -> None:
    mailer = RecordingMailer()
    notifier = build_emergency_notifier(services, mailer)
    assert isinstance(notifier, SmtpEmergencyNotifier)
    notice = EmergencyNotice(AT, "friend@example.org")
    assert await notifier.notify(notice) is False  # off by default: nothing is sent
    services.settings.safety.emergency_contact.enabled = True
    services.settings.safety.emergency_contact.email = "friend@example.org"
    assert await notifier.notify(notice) is True and mailer.sent[0].to == "friend@example.org"
    assert isinstance(build_emergency_notifier(services), SmtpEmergencyNotifier)


class FakeSelector:
    """What the health check asks of the backend selector."""

    def __init__(self, requested: str, fallback: Any, probe: Probe) -> None:
        self._requested, self._fallback, self._probe = requested, fallback, probe
        self.probes = 0

    def requested(self) -> str:
        return self._requested

    def fallback(self) -> Any:
        return self._fallback

    async def probe(self) -> Probe:
        self.probes += 1
        return self._probe


async def test_the_style_endpoint_is_looked_at_only_when_the_backend_needs_it() -> None:
    good = Probe(True, "ok", "fine", AT)
    bad = Probe(False, "unreachable", "no answer", AT)
    deepseek = FakeSelector("deepseek", None, good)
    assert await style_reading_of(deepseek)() == StyleReading(False, True)  # type: ignore[arg-type]
    assert deepseek.probes == 0  # not even asked
    style = FakeSelector("style", None, good)
    assert await style_reading_of(style)() == StyleReading(True, True, "ok")  # type: ignore[arg-type]
    down = FakeSelector("hybrid", None, bad)
    assert await style_reading_of(down)() == StyleReading(True, False, "unreachable")  # type: ignore[arg-type]
    fallen_back = FakeSelector(
        "deepseek", {"since": "x"}, bad
    )  # asked for DeepSeek, but in a fallback
    assert await style_reading_of(fallen_back)() == StyleReading(True, False, "unreachable")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["alert_delivery", "health_monitor", "ops_scheduler"])
def test_the_components_depend_on_nothing_so_they_start_first_and_stop_last(
    services: Services, name: str
) -> None:
    application = Application()
    register_ops(application, services, mailer=RecordingMailer(), notifier=RecordingNotifier())
    assert list(application.components[name].depends_on) == []


async def test_a_probe_of_another_component_is_part_of_the_health_report(
    services: Services,
) -> None:
    """R-SRV-002: the llama-server process of round 14 registers its check here."""

    async def serving() -> HealthCheck:
        return HealthCheck("style_serving", HealthLevel.WARN, "llama-server is restarting", 2.0)

    async def broken() -> HealthCheck:
        raise RuntimeError("probe bug")

    kit = register_ops(
        Application(),
        services,
        mailer=RecordingMailer(),
        notifier=RecordingNotifier(),
        probes=(("style_serving", serving), ("broken_probe", broken)),
    )
    report = await kit.collector.collect()
    found = {check.name: check for check in report.checks}
    assert found["style_serving"].level is HealthLevel.WARN and found["style_serving"].value == 2.0
    assert found["broken_probe"].level is HealthLevel.FAIL
    assert "probe crashed" in found["broken_probe"].detail
