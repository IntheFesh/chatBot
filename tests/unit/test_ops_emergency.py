"""The mail to the emergency contact: fixed text, one address, once an event (R-SAFE-001)."""

from __future__ import annotations

import dataclasses
import email
import email.policy
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.support.clock import ManualClock
from tests.support.ops import (
    Certificate,
    RecordingMailer,
    SmtpServer,
    make_certificate,
)
from twin.config.settings import SafetyConfig, SmtpConfig
from twin.engine.safety.notifier import (
    BODY_TEMPLATE,
    SUBJECT,
    EmergencyEmail,
    EmergencyNotice,
    render_notice,
)
from twin.ops.emergency import EPISODE_GAP, LAST_KEY, SmtpEmergencyNotifier
from twin.ops.mail import MailError, SmtpMailer
from twin.services import Services
from twin.storage.settings_store import get_setting

CHICAGO = ZoneInfo("America/Chicago")
FRIEND = "friend@example.org"
CANARY = "我真的不想活了"

pytestmark = pytest.mark.filterwarnings("ignore:Requiring AUTH while not requiring TLS:UserWarning")


def safety(*, enabled: bool = True, address: str | None = FRIEND) -> SafetyConfig:
    config = SafetyConfig()
    config.emergency_contact.enabled = enabled
    config.emergency_contact.email = address
    return config


def notifier(
    services: Services, mailer: RecordingMailer, config: SafetyConfig
) -> SmtpEmergencyNotifier:
    return SmtpEmergencyNotifier(mailer, services.db, services.clock, config, lambda: CHICAGO)


def notice(services: Services, recipient: str = FRIEND) -> EmergencyNotice:
    return EmergencyNotice(services.clock.now_utc(), recipient)


def test_the_notice_has_no_field_that_could_carry_what_was_said() -> None:
    assert {f.name for f in dataclasses.fields(EmergencyNotice)} == {"at", "recipient"}
    assert {f.name for f in dataclasses.fields(EmergencyEmail)} == {"to", "subject", "body"}
    assert BODY_TEMPLATE.count("{") == 1 and "{time}" in BODY_TEMPLATE  # the time is all there is


async def test_the_mail_is_the_fixed_template_with_the_time(
    services: Services, clock: ManualClock
) -> None:
    mailer = RecordingMailer()
    assert await notifier(services, mailer, safety()).notify(notice(services)) is True
    (mail,) = mailer.sent
    expected = render_notice(notice(services), CHICAGO)
    assert (mail.to, mail.subject, mail.text) == (expected.to, expected.subject, expected.body)
    assert mail.to == FRIEND and mail.subject == SUBJECT and mail.html is None
    assert "2026-10-09 07:00（America/Chicago）" in mail.text
    assert "他可能需要关心" in mail.text and "不包含任何聊天内容" in mail.text
    assert CANARY not in mail.text


async def test_one_mail_per_event_and_a_new_one_after_the_gap(
    services: Services, clock: ManualClock
) -> None:
    mailer = RecordingMailer()
    contact = notifier(services, mailer, safety())
    assert await contact.notify(notice(services)) is True
    clock.tick(30 * 60)  # the crisis goes on: more messages, the same event
    assert await contact.notify(notice(services)) is False
    clock.tick(EPISODE_GAP.total_seconds() - 30 * 60 + 1)
    assert await contact.notify(notice(services)) is True
    assert len(mailer.sent) == 2
    with services.db.session() as session:
        assert get_setting(session, LAST_KEY) == clock.now_utc().isoformat()


async def test_the_memory_of_the_last_mail_survives_a_restart(
    services: Services, clock: ManualClock
) -> None:
    mailer = RecordingMailer()
    await notifier(services, mailer, safety()).notify(notice(services))
    clock.tick(60)
    again = notifier(services, mailer, safety())  # a new object, as after a restart
    assert await again.notify(notice(services)) is False and len(mailer.sent) == 1


async def test_a_failed_mail_is_not_remembered_so_the_next_try_may_send(
    services: Services, clock: ManualClock
) -> None:
    mailer = RecordingMailer()
    mailer.errors = ["network"]
    contact = notifier(services, mailer, safety())
    with pytest.raises(MailError) as caught:
        await contact.notify(notice(services))
    assert caught.value.code == "network"
    with services.db.session() as session:
        assert get_setting(session, LAST_KEY) is None
    clock.tick(60)
    assert await contact.notify(notice(services)) is True


async def test_only_the_configured_address_is_ever_written_to(services: Services) -> None:
    mailer = RecordingMailer()
    contact = notifier(services, mailer, safety())
    with pytest.raises(MailError) as caught:
        await contact.notify(notice(services, "someone-else@example.org"))
    assert caught.value.code == "recipient" and mailer.sent == []
    assert await contact.notify(notice(services, "  FRIEND@Example.org ")) is True  # same address
    assert mailer.sent[0].to == FRIEND  # written to the configured address, as configured


@pytest.mark.parametrize(
    "config",
    [
        safety(enabled=False),
        safety(address=None),
        safety(address="   "),
        safety(enabled=False, address=None),
    ],
)
async def test_nothing_is_sent_unless_switched_on_with_an_address(
    services: Services, config: SafetyConfig
) -> None:
    mailer = RecordingMailer()
    assert await notifier(services, mailer, config).notify(notice(services)) is False
    assert mailer.sent == [] and mailer.attempts == 0


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> Certificate:
    return make_certificate(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def server(certificate: Certificate) -> Iterator[SmtpServer]:
    with SmtpServer(certificate, implicit_tls=False) as running:
        yield running


async def test_the_mail_arrives_through_a_real_smtp_server(
    services: Services, server: SmtpServer, certificate: Certificate, tmp_path: Path
) -> None:
    config = SmtpConfig(host="127.0.0.1", port=server.port, user=server.user, to="me@example.org")
    mailer = SmtpMailer(config, lambda: server.password, tls_context=certificate.client)
    assert await notifier(services, mailer, safety()).notify(notice(services))  # type: ignore[arg-type]
    (envelope,) = server.inbox.envelopes
    assert envelope.rcpt_tos == [FRIEND] and envelope.mail_from == server.user
    assert envelope.content is not None
    message = email.message_from_bytes(envelope.content, policy=email.policy.default)
    body = message.get_body(("plain",)).get_content()  # type: ignore[union-attr]
    assert message["Subject"] == SUBJECT and "他可能需要关心" in body
    assert "me@example.org" not in body  # the user's own address is not in her friend's mail
    assert timedelta(hours=12) == EPISODE_GAP
