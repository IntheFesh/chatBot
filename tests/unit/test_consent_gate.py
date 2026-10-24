"""R18-F10 (D-825): her consent has no default date; without it the commands refuse (R-SCOPE-003).

The user's decision (round 18): ``ConsentConfig`` carries no date; ``twin setup`` asks whether she
knows and agrees and then writes the date; without it every command except ``twin setup`` and
``twin doctor`` refuses with exit code 3.  ``twin purge``, ``twin service stop|uninstall`` and
``twin secrets migrate`` stay usable as well (D-825: stopping the bot and deleting her data must
never need the consent that may just have been withdrawn).  R-CFG-004 and
``config/config.example.yaml`` say ``null``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.support.consent import CONSENT_DATE, CONSENT_ENV
from tests.support.network import OfflineTransport
from twin.cli import app
from twin.config.loader import ENV_CONFIG, load_settings
from twin.config.secrets import SecretStore
from twin.config.settings import ConsentConfig, Settings
from twin.ops.process_model import ExitCode, get_spec, iter_commands
from twin.services import CliContext, set_cli_context

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]

# what may run without her consent (D-825); everything else is refused with exit code 3
EXEMPT = {
    "doctor",
    "setup",
    "purge",
    "service stop",
    "service uninstall",
}


@pytest.fixture
def fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret_store: SecretStore) -> Path:
    """A fresh installation: no configuration file and no consent anywhere."""
    monkeypatch.delenv(CONSENT_ENV, raising=False)
    path = tmp_path / "config" / "config.yaml"
    monkeypatch.setenv(ENV_CONFIG, str(path))
    set_cli_context(CliContext(secrets=secret_store, http_transport=OfflineTransport()))
    return path


def test_the_settings_carry_no_consent_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """R18-F10: no default date in the code, the example file or the SPEC block."""
    monkeypatch.delenv(CONSENT_ENV, raising=False)
    assert ConsentConfig().confirmed_at is None
    assert Settings().consent.confirmed_at is None
    assert load_settings().consent.confirmed_at is None
    example = yaml.safe_load((ROOT / "config" / "config.example.yaml").read_text(encoding="utf-8"))
    assert example["consent"] == {"confirmed_at": None}


def test_without_consent_commands_refuse_with_exit_code_3(fresh: Path) -> None:
    """R18-F10: a missing consent is refused, and the message names `twin setup`."""
    for args in (["jobs", "list"], ["db", "upgrade"], ["config", "show"], ["secrets", "list"]):
        result = runner.invoke(app, args)
        assert result.exit_code == int(ExitCode.CONSENT), (args, result.output)
        assert "twin setup" in result.output


def test_only_the_named_commands_are_exempt_from_the_consent_gate() -> None:
    """R18-F10: the list of commands that run without her consent is exactly ``EXEMPT``."""
    exempt = {
        name
        for name, callback in iter_commands(app)
        if (spec := get_spec(callback)) is not None and not spec.consent
    }
    assert exempt == EXEMPT


def test_doctor_runs_without_consent_and_points_to_setup(fresh: Path) -> None:
    """R18-F10: `twin doctor` still runs; its configuration check fails with the setup hint."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code != int(ExitCode.CONSENT), result.output
    assert "consent.confirmed_at is missing" in result.output
    assert "config: run `twin setup`" in result.output


def test_setup_records_the_consent_and_the_commands_run_afterwards(fresh: Path) -> None:
    """R18-F10: the wizard asks, writes the date the user confirms, and the gate opens."""
    answers = ["y", CONSENT_DATE, "", "n", "", "", "n"]
    done = runner.invoke(app, ["setup"], input="\n".join(answers) + "\n")
    assert done.exit_code == 0, done.output
    written = yaml.safe_load(fresh.read_text(encoding="utf-8"))
    assert written["consent"] == {"confirmed_at": CONSENT_DATE}
    upgraded = runner.invoke(app, ["db", "upgrade"])
    assert upgraded.exit_code == 0, upgraded.output
    listed = runner.invoke(app, ["jobs", "list"])
    assert listed.exit_code == 0, listed.output
