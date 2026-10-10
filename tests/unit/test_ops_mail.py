"""E-mail through a real SMTP server (aiosmtpd): both TLS modes, login, errors (R-OPS-004)."""

from __future__ import annotations

import email
import email.policy
from collections.abc import Iterator
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.support.ops import Certificate, SmtpServer, free_port, make_certificate
from twin.config.settings import SmtpConfig
from twin.ops.alert_delivery import AlertDelivery
from twin.ops.mail import (
    MailError,
    OutgoingMail,
    SmtpMailer,
    build_message,
    security_for,
)
from twin.services import Services

CANARY = "今晚想吃火锅"

# aiosmtpd counts only a STARTTLS upgrade as "TLS", so the implicit-TLS test server warns
pytestmark = pytest.mark.filterwarnings("ignore:Requiring AUTH while not requiring TLS:UserWarning")


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> Certificate:
    return make_certificate(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def starttls(certificate: Certificate) -> Iterator[SmtpServer]:
    with SmtpServer(certificate, implicit_tls=False) as server:
        yield server


@pytest.fixture
def implicit(certificate: Certificate) -> Iterator[SmtpServer]:
    with SmtpServer(certificate, implicit_tls=True) as server:
        yield server


def mailer_for(
    server: SmtpServer, certificate: Certificate, security: str, password: str | None = None
) -> SmtpMailer:
    config = SmtpConfig(
        host="127.0.0.1",
        port=server.port,
        user=server.user,
        to="me@example.org",
        security=security,  # type: ignore[arg-type]
    )
    secret = server.password if password is None else password
    return SmtpMailer(config, lambda: secret, tls_context=certificate.client)


def test_the_security_follows_the_port_unless_it_is_set() -> None:
    assert security_for(SmtpConfig(port=465)) == "ssl"
    assert security_for(SmtpConfig(port=587)) == "starttls"
    assert security_for(SmtpConfig(port=25)) == "starttls"
    assert security_for(SmtpConfig(port=465, security="starttls")) == "starttls"
    assert security_for(SmtpConfig(port=587, security="ssl")) == "ssl"


def test_a_mail_goes_through_starttls_with_text_and_html(
    starttls: SmtpServer, certificate: Certificate
) -> None:
    mailer = mailer_for(starttls, certificate, "starttls")
    assert mailer.configured
    mailer.send(
        OutgoingMail("me@example.org", "[wechat-twin] 测试", "plain body", "<p>html body</p>")
    )
    (envelope,) = starttls.inbox.envelopes
    assert envelope.mail_from == starttls.user and envelope.rcpt_tos == ["me@example.org"]
    message = email.message_from_bytes(envelope.content, policy=email.policy.default)  # type: ignore[arg-type]
    assert message["Subject"] == "[wechat-twin] 测试"
    assert message.get_body(("plain",)).get_content().strip() == "plain body"  # type: ignore[union-attr]
    assert "html body" in message.get_body(("html",)).get_content()  # type: ignore[union-attr]
    assert message["Message-ID"] and message["Date"]


def test_a_mail_goes_through_implicit_tls(implicit: SmtpServer, certificate: Certificate) -> None:
    mailer_for(implicit, certificate, "ssl").send(OutgoingMail("me@example.org", "s", "body"))
    assert len(implicit.inbox.envelopes) == 1


def test_a_wrong_password_is_an_auth_error_without_the_password_in_it(
    starttls: SmtpServer, certificate: Certificate
) -> None:
    mailer = mailer_for(starttls, certificate, "starttls", password="wrong-password-123")
    with pytest.raises(MailError) as caught:
        mailer.send(OutgoingMail("me@example.org", "s", "body"))
    assert caught.value.code == "auth"
    assert "wrong-password-123" not in str(caught.value) and not starttls.inbox.envelopes


def test_a_server_that_is_not_there_is_a_network_error(certificate: Certificate) -> None:
    config = SmtpConfig(host="127.0.0.1", port=free_port(), user="bot@example.org", to="x@y.z")
    mailer = SmtpMailer(config, lambda: "p", tls_context=certificate.client, timeout_s=2.0)
    with pytest.raises(MailError) as caught:
        mailer.send(OutgoingMail("me@example.org", "s", "body"))
    assert caught.value.code == "network"


def test_a_certificate_the_client_does_not_trust_is_a_tls_error(
    starttls: SmtpServer, tmp_path: Path
) -> None:
    stranger = make_certificate(tmp_path)  # another self-signed certificate: not trusted
    config = SmtpConfig(host="127.0.0.1", port=starttls.port, user=starttls.user, to="x@y.z")
    mailer = SmtpMailer(config, lambda: starttls.password, tls_context=stranger.client)
    with pytest.raises(MailError) as caught:
        mailer.send(OutgoingMail("me@example.org", "s", "body"))
    assert caught.value.code == "tls"


def test_an_account_without_a_host_or_an_address_is_not_configured() -> None:
    assert not SmtpMailer(SmtpConfig(), lambda: "p").configured
    assert not SmtpMailer(
        SmtpConfig(host="smtp.example.org", user="not-an-address"), lambda: "p"
    ).configured
    with pytest.raises(MailError) as caught:
        SmtpMailer(SmtpConfig(), lambda: "p").send(OutgoingMail("a@b.c", "s", "t"))
    assert caught.value.code == "not_configured"


def test_the_message_has_the_headers_of_a_proper_mail() -> None:
    message = build_message("bot@example.org", OutgoingMail("me@example.org", "主题", "text"))
    assert message["From"] == "bot@example.org" and message["To"] == "me@example.org"
    assert message.is_multipart() is False
    html = build_message("bot@example.org", OutgoingMail("me@example.org", "s", "text", "<p>x</p>"))
    assert html.is_multipart()


async def test_an_alert_arrives_by_mail_and_holds_no_chat_content(
    services: Services, starttls: SmtpServer, certificate: Certificate
) -> None:
    """The whole path: the alert service, the delivery, a real SMTP round trip."""
    mailer = mailer_for(starttls, certificate, "starttls")
    delivery = AlertDelivery(
        services.alerts,
        services.clock,
        notifier=None,
        mailer=mailer,
        recipient=lambda: "me@example.org",
        zone=lambda: ZoneInfo("America/Chicago"),
    )
    services.alerts.raise_alert(
        "crisis",
        f"he wrote: {CANARY}",
        severity="critical",
        detail={"text": CANARY, "severity": "high", "reply": CANARY},
    )
    await delivery.deliver_once()
    (text,) = starttls.inbox.texts
    assert CANARY not in text and "火锅" not in text
    raw = starttls.inbox.envelopes[0].content
    assert raw is not None
    message = email.message_from_bytes(raw, policy=email.policy.default)  # type: ignore[arg-type]
    assert "检测到可能需要关心的信号" in message["Subject"] or "[wechat-twin]" in message["Subject"]
    body = message.get_body(("plain",)).get_content()  # type: ignore[union-attr]
    assert CANARY not in body and "不包含任何聊天内容" in body
