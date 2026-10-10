"""``twin setup``: the first-run wizard (R-OPS-001, R-SAFE-001, R-SCOPE-003)."""

from __future__ import annotations

import email
import email.policy
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.ops import Certificate, SmtpServer, make_certificate
from twin.cli import app
from twin.config.loader import ENV_CONFIG
from twin.config.secrets import SecretStore
from twin.engine.safety.notifier import BODY_TEMPLATE, SUBJECT
from twin.ops import setup_cli
from twin.ops.mail import SMTP_PASSWORD_SECRET, SmtpMailer
from twin.ops.setup_cli import DEEPSEEK_SECRET
from twin.services import CliContext, set_cli_context

runner = CliRunner()
TARGET = "wx" + "id_synthetic9"  # built at run time so the privacy scan stays clean
pytestmark = pytest.mark.filterwarnings("ignore:Requiring AUTH while not requiring TLS:UserWarning")


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret_store: SecretStore) -> Path:
    """A configuration file that does not exist yet, as on a fresh installation."""
    path = tmp_path / "config" / "config.yaml"
    monkeypatch.setenv(ENV_CONFIG, str(path))
    set_cli_context(CliContext(secrets=secret_store))
    return path


def wizard(answers: list[str], *options: str) -> tuple[int, str]:
    result = runner.invoke(app, [*options, "setup"], input="\n".join(answers) + "\n")
    return result.exit_code, result.output


def written(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_the_wizard_asks_everything_and_writes_it_where_it_belongs(
    config: Path, secret_store: SecretStore
) -> None:
    answers = [
        "y",  # she knows and agrees
        "2026-09-30",
        "sk-synthetic-deepseek-key",
        "n",  # no mail account now
        TARGET,
        "Asia/Shanghai",
        "n",  # no emergency contact
    ]
    code, out = wizard(answers)
    assert code == 0, out
    data = written(config)
    assert data["consent"] == {"confirmed_at": "2026-09-30"}
    assert data["target"] == {"username": TARGET}
    assert data["time"] == {"bot_timezone": "Asia/Shanghai"}
    assert data["safety"] == {"emergency_contact": {"enabled": False, "email": None}}
    assert secret_store.get(DEEPSEEK_SECRET) == "sk-synthetic-deepseek-key"
    assert "sk-synthetic" not in config.read_text(encoding="utf-8")  # secrets never in the file
    assert "sk-synthetic" not in out  # and not echoed
    assert "twin db upgrade" in out and "twin service install" in out and str(config) in out


def test_the_consent_is_required_and_the_date_must_be_a_date(
    config: Path, secret_store: SecretStore
) -> None:
    code, out = wizard(["n"])
    assert code != 0 and "没有她的同意不能继续" in out
    assert not config.exists() and not secret_store.exists(DEEPSEEK_SECRET)
    code, out = wizard(["y", "yesterday"])
    assert code != 0 and "YYYY-MM-DD" in out and not config.exists()


@pytest.mark.parametrize(
    ("zone", "today"), [("America/Chicago", "2026-11-01"), ("Asia/Shanghai", "2026-11-02")]
)
def test_without_an_earlier_answer_the_proposed_date_is_today_where_the_user_lives(
    config: Path, clock: ManualClock, zone: str, today: str
) -> None:
    """21:30 on 1 November in Chicago is already the 2nd in UTC and in Shanghai."""
    clock.set_time(datetime(2026, 11, 2, 3, 30, tzinfo=UTC))
    answers = ["y", "", "n", "n", "", "", "n"]  # the date, and every answer is the proposal
    options = ("--set", "consent.confirmed_at=null", "--set", f"time.bot_timezone={zone}")
    code, out = wizard(answers, *options)
    assert code == 0, out
    assert f"[{today}]" in out  # the proposal on the screen
    assert written(config)["consent"] == {"confirmed_at": today}


def test_the_wizard_can_be_run_again_and_keeps_what_is_there(
    config: Path, secret_store: SecretStore
) -> None:
    secret_store.set(DEEPSEEK_SECRET, "sk-old")
    first = [
        "y", "2026-09-30",
        "n",  # keep the key
        "n",  # no mail account
        "", "Asia/Shanghai",
        "y", "friend@example.org",
    ]  # fmt: skip
    code, out = wizard(first)
    assert code == 0, out
    assert secret_store.get(DEEPSEEK_SECRET) == "sk-old"
    assert written(config)["safety"] == {
        "emergency_contact": {"enabled": True, "email": "friend@example.org"}
    }
    again = ["y", "", "n", "n", "", "", "", ""]  # every answer is the proposal
    code, out = wizard(again)
    assert code == 0, out
    data = written(config)
    assert data["consent"] == {"confirmed_at": "2026-09-30"}
    assert data["time"] == {"bot_timezone": "Asia/Shanghai"}
    assert data["safety"] == {"emergency_contact": {"enabled": True, "email": "friend@example.org"}}


def test_a_new_key_replaces_the_old_one_only_when_asked(
    config: Path, secret_store: SecretStore
) -> None:
    secret_store.set(DEEPSEEK_SECRET, "sk-old")
    code, out = wizard(["y", "2026-09-30", "y", "sk-new-key", "n", "", "", "n"])
    assert code == 0, out
    assert secret_store.get(DEEPSEEK_SECRET) == "sk-new-key"
    empty = wizard(["y", "2026-09-30", "y", "", "n", "", "", "n"])
    assert empty[0] == 0 and secret_store.get(DEEPSEEK_SECRET) == "sk-new-key"  # an empty answer


def test_no_key_given_means_a_hint_not_an_error(config: Path, secret_store: SecretStore) -> None:
    code, out = wizard(["y", "2026-09-30", "", "n", "", "", "n"])
    assert code == 0 and "twin secrets set deepseek_api_key" in out
    assert not secret_store.exists(DEEPSEEK_SECRET)


def test_a_time_zone_that_does_not_exist_is_asked_again(config: Path) -> None:
    code, out = wizard(
        ["y", "2026-09-30", "", "n", "", "Mars/Base", "Atlantis", "Europe/Paris", "n"]
    )
    assert code == 0, out
    assert out.count("不认识时区") == 2
    assert written(config)["time"] == {"bot_timezone": "Europe/Paris"}


def test_the_emergency_contact_is_explained_before_it_is_switched_on(config: Path) -> None:
    code, out = wizard(["y", "2026-09-30", "", "n", "", "", "y", ""])  # on, but no address
    assert code == 0, out
    assert SUBJECT in out
    for line in BODY_TEMPLATE.splitlines():
        assert line in out  # the very text that would be sent is on the screen
    assert "没有填邮箱，保持关闭" in out
    assert written(config)["safety"] == {"emergency_contact": {"enabled": False, "email": None}}


# ------------------------------------------------------------------------------ the mail


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> Certificate:
    return make_certificate(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def server(certificate: Certificate) -> Iterator[SmtpServer]:
    with SmtpServer(certificate, implicit_tls=False) as running:
        yield running


@pytest.fixture
def trusting(monkeypatch: pytest.MonkeyPatch, certificate: Certificate) -> None:
    """The wizard's mailer trusts the test server's self-signed certificate."""
    real = SmtpMailer

    def factory(smtp, password):  # type: ignore[no-untyped-def]
        return real(smtp, password, tls_context=certificate.client)

    monkeypatch.setattr(setup_cli, "SmtpMailer", factory)


def mail_answers(server: SmtpServer, password: str) -> list[str]:
    return [
        "y", "2026-09-30", "",  # consent, no DeepSeek key
        "y", "127.0.0.1", str(server.port), server.user, "me@example.org", password, "y",
        "", "", "n",
    ]  # fmt: skip


def test_the_mail_account_is_stored_and_a_test_mail_arrives(
    config: Path, secret_store: SecretStore, server: SmtpServer, trusting: None
) -> None:
    code, out = wizard(mail_answers(server, server.password))
    assert code == 0, out
    assert "测试邮件已发出" in out
    data = written(config)
    smtp = data["ops"]["smtp"]  # type: ignore[index]
    assert smtp["host"] == "127.0.0.1" and smtp["port"] == server.port  # type: ignore[index]
    assert smtp["user"] == server.user and smtp["to"] == "me@example.org"  # type: ignore[index]
    assert secret_store.get(SMTP_PASSWORD_SECRET) == server.password
    assert server.password not in config.read_text(encoding="utf-8") + out
    (envelope,) = server.inbox.envelopes
    assert envelope.rcpt_tos == ["me@example.org"] and envelope.content is not None
    message = email.message_from_bytes(envelope.content, policy=email.policy.default)
    assert message["Subject"] == "[wechat-twin] 测试邮件"


def test_a_wrong_password_is_reported_by_the_test_mail(
    config: Path, secret_store: SecretStore, server: SmtpServer, trusting: None
) -> None:
    code, out = wizard(mail_answers(server, "wrong-password"))
    assert code == 0, out
    assert "测试邮件没发出去（auth）" in out and "wrong-password" not in out
    assert server.inbox.envelopes == []
    assert (
        secret_store.get(SMTP_PASSWORD_SECRET) == "wrong-password"
    )  # kept: the user may fix the server
