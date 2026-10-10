"""Test doubles and harnesses of round 12: mail, notices, schtasks, children."""

from __future__ import annotations

import asyncio
import datetime as dt
import ipaddress
import socket
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiosmtpd.controller import Controller
from aiosmtpd.smtp import SMTP, AuthResult, Envelope, LoginPassword, Session
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from twin.ops.mail import MailError, OutgoingMail
from twin.ops.taskscheduler import CommandResult, TaskSchedulerError, TaskSpec, build_task_xml


class RecordingNotifier:
    """A notifier that remembers what it was asked to show (``fail`` makes it raise)."""

    name = "recording"

    def __init__(self) -> None:
        self.shown: list[tuple[str, str]] = []
        self.fail = False

    def notify(self, title: str, body: str) -> None:
        if self.fail:
            from twin.ops.notify import NotifyError

            raise NotifyError("scripted")
        self.shown.append((title, body))


class RecordingMailer:
    """A mailer that keeps the mails; ``errors`` is a queue of codes to fail with, first."""

    def __init__(self, *, configured: bool = True) -> None:
        self._configured = configured
        self.sent: list[OutgoingMail] = []
        self.errors: list[str] = []
        self.attempts = 0

    @property
    def configured(self) -> bool:
        return self._configured

    def send(self, mail: OutgoingMail) -> None:
        self.attempts += 1
        if self.errors:
            raise MailError(self.errors.pop(0))
        self.sent.append(mail)


class ScriptedRunner:
    """A :class:`~twin.ops.taskscheduler.CommandRunner` with scripted answers.

    ``answers`` maps the first words of a command (``"schtasks.exe /Query"``) to a result or to a
    function of the argument list; a command nobody scripted fails like a missing program.
    """

    def __init__(
        self, answers: Mapping[str, CommandResult | Callable[[list[str]], CommandResult]] | None
    ) -> None:
        self.answers = dict(answers or {})
        self.calls: list[list[str]] = []

    def run(self, args: list[str], *, timeout_s: float = 30.0) -> CommandResult:
        self.calls.append(list(args))
        for prefix, answer in self.answers.items():
            words = prefix.split()
            if args[: len(words)] == words:
                return answer(args) if callable(answer) else answer
        raise TaskSchedulerError(f"cannot run {args[0]}: FileNotFoundError")


def ok(text: str = "", code: int = 0) -> CommandResult:
    return CommandResult(code, text.encode("utf-8"), b"")


def failed(text: str = "ERROR: The system cannot find the file specified.") -> CommandResult:
    return CommandResult(1, b"", text.encode("utf-8"))


# the scheduled task and the power plan of a machine that is set up as the bot wants it
TASK_SPEC = TaskSpec(
    r"C:\repo\.venv\Scripts\twin.exe", "supervise --from-task", r"C:\repo", r"PC\me"
)
POWERCFG = """
Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)
  Subgroup GUID: 238c9fa8-0aad-41ed-83f4-97be242c8f20  (Sleep)
    Power Setting GUID: 29f6c1db-86da-48c5-9fdb-f2b67b1f44da  (Sleep after)
      Possible Setting Index: 0x00000000
      Possible Settings Description: Never
    Current AC Power Setting Index: 0x{ac:08x}
    Current DC Power Setting Index: 0x{dc:08x}
"""


def healthy_machine_runner() -> ScriptedRunner:
    """Answers of ``schtasks``, ``powercfg`` and ``nvidia-smi`` on a machine that is set up right.

    The doctor's Windows checks ask the machine they run on; a test that wants a verdict about
    its own subject, not about the CI computer, hands the context this runner.
    """
    return ScriptedRunner(
        {
            "schtasks.exe /Query": ok(build_task_xml(TASK_SPEC)),
            "powercfg": ok(POWERCFG.format(ac=0, dc=0)),
            "nvidia-smi": ok("NVIDIA GeForce RTX 5090, 570.86, 32607 MiB\n"),
        }
    )


# ------------------------------------------------------------------ a real SMTP server


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass
class Certificate:
    """A self-signed certificate for 127.0.0.1 (server context and a client that trusts it)."""

    server: ssl.SSLContext
    client: ssl.SSLContext


def make_certificate(folder: Path) -> Certificate:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=2))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_file, key_file = folder / "cert.pem", folder / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(str(cert_file), str(key_file))
    client = ssl.create_default_context(cafile=str(cert_file))
    return Certificate(server, client)


class Inbox:
    """The mail handler of the test server."""

    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    async def handle_DATA(self, server: SMTP, session: Session, envelope: Envelope) -> str:
        self.envelopes.append(envelope)
        return "250 Message accepted for delivery"

    @property
    def texts(self) -> list[str]:
        return [e.content.decode("utf-8", errors="replace") for e in self.envelopes if e.content]  # type: ignore[union-attr]


class SmtpServer:
    """An :mod:`aiosmtpd` server on 127.0.0.1 that wants TLS and a login (``password``)."""

    def __init__(
        self,
        certificate: Certificate,
        *,
        implicit_tls: bool,
        user: str = "bot@example.org",
        password: str = "app-password",  # noqa: S107 - a made-up test password
    ) -> None:
        self.inbox = Inbox()
        self.port = free_port()
        self.user, self.password = user, password

        def authenticate(
            server: SMTP, session: Session, envelope: Envelope, mechanism: str, data: Any
        ) -> AuthResult:
            if isinstance(data, LoginPassword):
                good = data.login == self.user.encode() and data.password == self.password.encode()
                return AuthResult(success=good, handled=False)
            return AuthResult(success=False, handled=False)

        options: dict[str, Any] = {
            "authenticator": authenticate,
            "auth_required": True,
            "auth_require_tls": True,
        }
        if implicit_tls:
            options["ssl_context"] = certificate.server
            # the whole connection is TLS; aiosmtpd only counts a STARTTLS upgrade as "TLS"
            options["auth_require_tls"] = False
        else:
            options["tls_context"] = certificate.server
            options["require_starttls"] = True
        self._controller = Controller(self.inbox, hostname="127.0.0.1", port=self.port, **options)

    def __enter__(self) -> SmtpServer:
        self._controller.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._controller.stop()


# --------------------------------------------------------------------- child processes


class FakeChild:
    """A started ``twin run`` that ends when the test says (or when it is asked to stop)."""

    def __init__(self, pid: int = 4242, *, stops_gracefully: bool = True) -> None:
        self.pid: int | None = pid
        self._exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self.stop_requests = 0
        self.killed = False
        self._stops_gracefully = stops_gracefully

    async def wait(self) -> int:
        return await asyncio.shield(self._exit)

    def finish(self, code: int) -> None:
        if not self._exit.done():
            self._exit.set_result(code)

    def request_stop(self) -> None:
        self.stop_requests += 1
        if self._stops_gracefully:
            self.finish(0)

    def kill(self) -> None:
        self.killed = True
        self.finish(-9)


@dataclass
class FakeLauncher:
    """Hands out :class:`FakeChild` objects and remembers them."""

    stops_gracefully: bool = True
    children: list[FakeChild] = field(default_factory=list)
    environments: list[dict[str, str]] = field(default_factory=list)

    async def start(self, env: Mapping[str, str]) -> FakeChild:
        child = FakeChild(1000 + len(self.children), stops_gracefully=self.stops_gracefully)
        self.children.append(child)
        self.environments.append(dict(env))
        return child
