"""``twin run``, ``twin supervise`` and ``twin service install`` refuse the file store on Windows.

R-CFG-002, D-744 (round 17, item 53b): ``TWIN_KEYRING_BACKEND=file`` is a development and test
switch on every platform.  On Windows - the machine the bot runs on - it needs the explicit
``TWIN_ALLOW_FILE_KEYRING=1`` too, which the fixtures of the suite set (never inferred from
``TWIN_HOME`` or from running under pytest).  The platform is a seam
(``twin.ops.credential_guard.current_platform``) so that this is tested on every machine.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from twin import cli as twin_cli
from twin.cli import app
from twin.config.secrets import SecretStore, SecretStoreError
from twin.ops import credential_guard
from twin.ops.process_model import CliError, ExitCode

runner = CliRunner()


@pytest.fixture
def windows_without_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credential_guard, "current_platform", lambda: "win32")
    monkeypatch.delenv("TWIN_ALLOW_FILE_KEYRING", raising=False)

    # whatever the command would do after the guard must not happen
    def never(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the command went on although the credential store was refused")

    monkeypatch.setattr(twin_cli, "_serve", never)


def test_the_guard_refuses_the_file_store_on_windows(windows_without_the_switch: None) -> None:
    with pytest.raises(CliError) as refused:
        credential_guard.require_credential_store(None)  # the default store of this environment
    assert refused.value.code is ExitCode.SECRETS
    assert "TWIN_ALLOW_FILE_KEYRING" in refused.value.message


def test_the_explicit_switch_lets_the_file_store_through(
    windows_without_the_switch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_ALLOW_FILE_KEYRING", "1")
    credential_guard.require_credential_store(None)


def test_the_guard_never_refuses_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TWIN_ALLOW_FILE_KEYRING", raising=False)
    monkeypatch.setattr(credential_guard, "current_platform", lambda: "linux")
    credential_guard.require_credential_store(None)


def test_a_store_that_cannot_be_built_is_a_secrets_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> SecretStore:
        raise SecretStoreError("no store")

    monkeypatch.setattr(SecretStore, "default", staticmethod(broken))
    with pytest.raises(CliError) as failed:
        credential_guard.require_credential_store(None)
    assert failed.value.code is ExitCode.SECRETS


def test_run_refuses_to_start(windows_without_the_switch: None) -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code == int(ExitCode.SECRETS), result.output
    assert "TWIN_ALLOW_FILE_KEYRING=1" in result.output
    assert "effective configuration" not in result.output  # it stopped before anything started


def test_supervise_refuses_to_start(windows_without_the_switch: None) -> None:
    result = runner.invoke(app, ["supervise"])
    assert result.exit_code == int(ExitCode.SECRETS), result.output
    assert "TWIN_ALLOW_FILE_KEYRING=1" in result.output


def test_service_install_refuses_but_may_still_print_the_definition(
    windows_without_the_switch: None,
) -> None:
    refused = runner.invoke(app, ["service", "install"])
    assert refused.exit_code == int(ExitCode.SECRETS), refused.output
    assert "TWIN_ALLOW_FILE_KEYRING=1" in refused.output
    printed = runner.invoke(app, ["service", "install", "--print-xml"])
    assert printed.exit_code == 0 and "<Task" in printed.output  # nothing is registered by it
