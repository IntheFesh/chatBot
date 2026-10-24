"""Every command refuses the file store on Windows; ``twin secrets migrate`` is the way out.

R-CFG-002, D-744 (round 17, item 53b): ``TWIN_KEYRING_BACKEND=file`` is a development and test
switch on every platform.  On Windows - the machine the bot runs on - it needs the explicit
``TWIN_ALLOW_FILE_KEYRING=1`` too, which the fixtures of the suite set (never inferred from
``TWIN_HOME`` or from running under pytest).  Round 18 (item A2, D-827, D-828, the user's
decision): the check sits where the store is opened, so every command except ``twin doctor`` and
``twin secrets migrate`` refuses it, and ``twin secrets migrate`` moves the entries into the
Windows Credential Manager (a test double here).  The platform is a seam
(``twin.config.secrets.current_platform``) so that this is tested on every machine.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.support.credentials import BrokenCredentials, CredentialManagerDouble
from tests.support.network import OfflineTransport
from twin import cli as twin_cli
from twin.cli import app
from twin.config import secrets as secrets_module
from twin.config.loader import load_settings
from twin.config.secrets import SERVICE_NAME, EncryptedFileKeyring, SecretStore, SecretStoreError
from twin.ops import credential_guard, doctor
from twin.ops.doctor import CheckStatus, DoctorContext
from twin.ops.process_model import CliError, ExitCode
from twin.services import CliContext, set_cli_context

runner = CliRunner()


class BrokenSystemCredentials(BrokenCredentials):
    """A system credential store that is there but answers every call with an error."""

    priority = 5.0


@pytest.fixture
def windows_without_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets_module, "current_platform", lambda: "win32")
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
    monkeypatch.setattr(secrets_module, "current_platform", lambda: "linux")
    credential_guard.require_credential_store(None)


def test_a_store_that_cannot_be_built_is_a_secrets_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**_options: object) -> SecretStore:
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


# ------------------------------------------------- R18-A2: every command, and the way out


def _file_store() -> EncryptedFileKeyring:
    return EncryptedFileKeyring(Path(os.environ["TWIN_SECRETS_DIR"]))


def _file_entries() -> dict[str, str]:
    store = _file_store()
    names = ("deepseek_api_key", "db-key-index", *(f"db-key-{n}" for n in range(1, 5)))
    return {
        name: value
        for name in names
        if (value := store.get_password(SERVICE_NAME, name)) is not None
    }


@pytest.fixture
def upgraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A migrated database; ``db upgrade`` itself never opens the credential store."""
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0


@pytest.mark.parametrize(
    ("args", "answers"),
    [
        (["secrets", "set", "deepseek_api_key", "--stdin"], "sk-synthetic-key\n"),
        (["secrets", "list"], None),
        (["jobs", "run", "--until-idle"], None),
        (["jobs", "list"], None),
        (["chat", "--local"], "/quit\n"),
        (["setup"], "y\n2026-10-08\n"),
    ],
)
def test_every_command_that_opens_the_credential_store_refuses_the_file_store(
    upgraded: None, windows_without_the_switch: None, args: list[str], answers: str | None
) -> None:
    """R18-A2 (D-827): the check sits where the store is opened, not in three commands only."""
    result = runner.invoke(app, args, input=answers)
    assert result.exit_code == int(ExitCode.SECRETS), result.output
    assert "twin secrets migrate" in result.output
    assert _file_entries() == {}  # no database key was made, no secret was written
    assert "她已经知道并同意了吗" not in result.output  # the wizard did not even start


def test_doctor_still_runs_on_the_file_store_and_names_the_way_out(
    upgraded: None, windows_without_the_switch: None
) -> None:
    """R18-A2 (D-827): `twin doctor` is exempt and points to `twin secrets migrate`."""
    set_cli_context(CliContext(http_transport=OfflineTransport()))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code != int(ExitCode.SECRETS), result.output
    assert "keyring" in result.output and "db-key" in result.output  # it looked at the store
    # what it says on Windows (its checks take the platform as an input)
    checked = doctor.check_keyring(DoctorContext(load_settings(), platform="win32"))
    assert checked.status is CheckStatus.FAIL
    assert "twin secrets migrate" in checked.detail and "twin secrets migrate" in checked.hint


# ------------------------------------------------------------ R18-A2: `twin secrets migrate`


@pytest.fixture
def filled_file_store(upgraded: None) -> dict[str, str]:
    """The file store of a development run: a key, data sealed with it, a retired key, a secret."""
    stored = runner.invoke(app, ["secrets", "set", "deepseek_api_key", "--stdin"], input="sk-x\n")
    assert stored.exit_code == 0, stored.output
    assert runner.invoke(app, ["settings", "set", "paused", "true"]).exit_code == 0  # a sealed row
    rotated = runner.invoke(app, ["secrets", "rotate-db-key"])  # db-key-1 is retired by this
    assert rotated.exit_code == 0, rotated.output
    assert runner.invoke(app, ["settings", "set", "paused", "false"]).exit_code == 0
    entries = _file_entries()
    assert {"deepseek_api_key", "db-key-index", "db-key-1", "db-key-2"} <= set(entries)
    return entries


@pytest.fixture
def credential_manager(monkeypatch: pytest.MonkeyPatch) -> CredentialManagerDouble:
    """The Windows Credential Manager, as ``keyring`` would hand it out on Windows."""
    double = CredentialManagerDouble()
    monkeypatch.setattr(secrets_module.keyring, "get_keyring", lambda: double)
    return double


def _migrated(double: CredentialManagerDouble) -> dict[str, str]:
    return {
        name: value for (service, name), value in double.items.items() if service == SERVICE_NAME
    }


def test_migrate_moves_every_entry_checks_it_and_removes_the_file(
    filled_file_store: dict[str, str],
    windows_without_the_switch: None,
    credential_manager: CredentialManagerDouble,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R18-A2 (D-828): every entry, the retired key too; decrypted before the file goes."""
    directory = Path(os.environ["TWIN_SECRETS_DIR"])
    result = runner.invoke(app, ["secrets", "migrate"])
    assert result.exit_code == 0, result.output
    assert _migrated(credential_manager) == filled_file_store
    assert re.search(r"the moved database keys opened [1-9]\d* encrypted value", result.output)
    assert not (directory / "secrets.enc").exists() and not (directory / "secrets.key").exists()
    assert "TWIN_KEYRING_BACKEND" in result.output  # what to remove now
    for value in filled_file_store.values():
        assert value not in result.output  # names only, never a value
    # the next command (a new process) uses the credential manager and reads the sealed data
    monkeypatch.delenv("TWIN_KEYRING_BACKEND")
    set_cli_context(CliContext())
    listed = runner.invoke(app, ["settings", "list"])
    assert listed.exit_code == 0, listed.output
    assert "db-key-1" in runner.invoke(app, ["secrets", "list"]).output


def test_a_failed_write_keeps_the_file(
    filled_file_store: dict[str, str],
    windows_without_the_switch: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R18-A2 (D-828): the third write (the probe, one entry, then this) fails: nothing removed."""
    double = CredentialManagerDouble(fail_on_write=3)
    monkeypatch.setattr(secrets_module.keyring, "get_keyring", lambda: double)
    result = runner.invoke(app, ["secrets", "migrate"])
    assert result.exit_code == int(ExitCode.SECRETS), result.output
    assert "the file store was kept" in result.output
    assert _file_entries() == filled_file_store
    double.fail_on_write = None  # the credential manager works again: the second run finishes
    again = runner.invoke(app, ["secrets", "migrate"])
    assert again.exit_code == 0, again.output
    assert "already there with the same value" in again.output
    assert _migrated(double) == filled_file_store


def test_an_entry_that_differs_in_the_credential_manager_stops_the_move(
    filled_file_store: dict[str, str],
    windows_without_the_switch: None,
    credential_manager: CredentialManagerDouble,
) -> None:
    """R18-A2 (D-828): a different value already there is never overwritten."""
    credential_manager.set_password(SERVICE_NAME, "deepseek_api_key", "sk-other")
    result = runner.invoke(app, ["secrets", "migrate"])
    assert result.exit_code == int(ExitCode.SECRETS), result.output
    assert "deepseek_api_key" in result.output and "sk-other" not in result.output
    assert _migrated(credential_manager) == {"deepseek_api_key": "sk-other"}  # nothing written
    assert _file_entries() == filled_file_store


def test_keys_that_do_not_open_the_data_keep_the_file(
    filled_file_store: dict[str, str],
    windows_without_the_switch: None,
    credential_manager: CredentialManagerDouble,
) -> None:
    """R18-A2 (D-828): the data is decrypted with the moved keys before anything is removed."""
    _file_store().set_password(SERVICE_NAME, "db-key-2", "00" * 32)  # not the key of the data
    result = runner.invoke(app, ["secrets", "migrate"])
    assert result.exit_code == int(ExitCode.SECRETS), result.output
    assert "do not open" in result.output and "the file store was kept" in result.output
    assert (Path(os.environ["TWIN_SECRETS_DIR"]) / "secrets.enc").exists()


def test_migrate_without_a_usable_credential_manager_changes_nothing(
    filled_file_store: dict[str, str],
    windows_without_the_switch: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R18-A2 (D-828): no system store to move to: refused, the file stays."""
    monkeypatch.setattr(secrets_module.keyring, "get_keyring", BrokenSystemCredentials)
    result = runner.invoke(app, ["secrets", "migrate"])
    assert result.exit_code == int(ExitCode.SECRETS), result.output
    assert _file_entries() == filled_file_store


def test_migrate_with_no_file_store_has_nothing_to_do(
    upgraded: None, windows_without_the_switch: None, credential_manager: CredentialManagerDouble
) -> None:
    """R18-A2 (D-828): nothing to move is not an error."""
    result = runner.invoke(app, ["secrets", "migrate"])
    assert result.exit_code == 0, result.output
    assert "nothing to migrate" in result.output and _migrated(credential_manager) == {}
