"""Sending e-mail through the user's SMTP account (R-OPS-004, R-SAFE-001).

The account is ``ops.smtp`` (host, port, user, ``security``) and the password in the credential
store (``smtp_password``, CLAUDE.md rule 10).  Port 465 uses implicit TLS and every other port
STARTTLS (``security: auto``); the connection is never unencrypted.  The sender is ``ops.smtp.user``
(an address), the recipient of alerts ``ops.smtp.to``.

A failure is reported as a :class:`MailError` carrying a short code (``auth``, ``network``,
``tls``, ``recipient``, ``smtp``, ``not_configured``) and nothing else: neither the server's
reply text nor the password can reach a log or the ``alerts`` table through it.  The callers
(the alert delivery, the emergency notice) record the code and try again later; a mail that fails
never fails the program.
"""

from __future__ import annotations

import smtplib
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Protocol

from twin.config.settings import SmtpConfig

SMTP_PASSWORD_SECRET = "smtp_password"  # noqa: S105 - the credential store entry name, not a secret
SSL_PORT = 465
DEFAULT_TIMEOUT_S = 20.0


class MailError(Exception):
    """A mail could not be sent; ``code`` says why in one word."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OutgoingMail:
    """A message: recipient, subject, plain text and, optionally, an HTML alternative."""

    to: str
    subject: str
    text: str
    html: str | None = None


class Mailer(Protocol):
    """Delivers :class:`OutgoingMail` (blocking: call it from a worker thread)."""

    @property
    def configured(self) -> bool: ...

    def send(self, mail: OutgoingMail) -> None:
        """Send ``mail`` or raise :class:`MailError`."""
        ...


def security_for(config: SmtpConfig) -> str:
    """``ssl`` or ``starttls`` for the configured account."""
    if config.security == "auto":
        return "ssl" if config.port == SSL_PORT else "starttls"
    return config.security


def build_message(sender: str, mail: OutgoingMail) -> EmailMessage:
    """The RFC 5322 message: plain text, plus HTML when there is some."""
    message = EmailMessage()
    message["From"] = sender
    message["To"] = mail.to
    message["Subject"] = mail.subject
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = make_msgid(domain=sender.rpartition("@")[2] or "localhost")
    message.set_content(mail.text)
    if mail.html is not None:
        message.add_alternative(mail.html, subtype="html")
    return message


class SmtpMailer:
    """An SMTP client for one account (see the module description)."""

    def __init__(
        self,
        config: SmtpConfig,
        password: Callable[[], str | None],
        *,
        tls_context: ssl.SSLContext | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._config = config
        self._password = password
        self._context = tls_context
        self._timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        user = self._config.user or ""
        return bool(self._config.host and "@" in user)

    def _tls(self) -> ssl.SSLContext:
        return self._context or ssl.create_default_context()

    def send(self, mail: OutgoingMail) -> None:
        if not self.configured:
            raise MailError("not_configured")
        host = str(self._config.host)
        user = str(self._config.user)
        secret = self._password()
        message = build_message(user, mail)
        try:
            with self._connect(host) as client:
                if secret:
                    client.login(user, secret)
                client.send_message(message)
        except smtplib.SMTPAuthenticationError:
            raise MailError("auth") from None
        except smtplib.SMTPRecipientsRefused:
            raise MailError("recipient") from None
        except ssl.SSLError:
            raise MailError("tls") from None
        except smtplib.SMTPException:
            raise MailError("smtp") from None
        except OSError:
            raise MailError("network") from None

    def _connect(self, host: str) -> smtplib.SMTP:
        port = self._config.port
        if security_for(self._config) == "ssl":
            return smtplib.SMTP_SSL(host, port, timeout=self._timeout_s, context=self._tls())
        client = smtplib.SMTP(host, port, timeout=self._timeout_s)
        try:
            client.starttls(context=self._tls())
        except BaseException:
            client.close()
            raise
        return client
