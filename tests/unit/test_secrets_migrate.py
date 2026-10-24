"""``twin.ops.secrets_migrate.migrate_file_store`` step by step (R18-A2, D-828).

The command-line tests (``tests/unit/test_credential_guard.py``) go through a real database; these
take the corners: no database, a database at an older schema, an entry that is already there, a
read-back that differs and a credential manager that breaks with something that is not a keyring
error.  In every failing case the file store is left exactly as it was.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.credentials import CredentialManagerDouble
from twin.cli import app
from twin.config import secrets as secrets_module
from twin.config.secrets import (
    SERVICE_NAME,
    BackendInfo,
    EncryptedFileKeyring,
)
from twin.ops.process_model import ExitCode
from twin.ops.secrets_migrate import MigrationError, migrate_file_store
from twin.storage import migrate

runner = CliRunner()
USABLE = BackendInfo(name="CredentialManagerDouble", kind="system", usable=True, detail="test")


@pytest.fixture
def source(tmp_path: Path) -> EncryptedFileKeyring:
    store = EncryptedFileKeyring(tmp_path / "secrets")
    store.set_password(SERVICE_NAME, "deepseek_api_key", "sk-synthetic")
    store.set_password(SERVICE_NAME, "smtp_password", "mail-synthetic")
    return store


def test_without_a_database_the_entries_are_moved_and_compared(
    source: EncryptedFileKeyring, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-A2: nothing to decrypt yet (a new installation): the read-back is the check."""
    target = CredentialManagerDouble()
    report = migrate_file_store(
        source, target, USABLE, db_path=tmp_path / "data" / "twin.db", clock=clock
    )
    assert sorted(report.moved) == ["deepseek_api_key", "smtp_password"]
    assert not report.database_checked and report.values_opened == 0
    assert not source.exists() and report.removed


def test_a_database_without_keys_yet_is_not_checked(
    source: EncryptedFileKeyring, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-A2: a database made by `twin db upgrade` before any key exists has nothing sealed."""
    db_path = tmp_path / "data" / "twin.db"
    migrate.upgrade(db_path)
    report = migrate_file_store(source, CredentialManagerDouble(), USABLE, db_path=db_path)
    assert not report.database_checked and not source.exists()


def test_an_entry_already_there_with_the_same_value_is_not_written_again(
    source: EncryptedFileKeyring, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-A2: the second run after a failed one finishes the move."""
    target = CredentialManagerDouble()
    target.set_password(SERVICE_NAME, "deepseek_api_key", "sk-synthetic")
    report = migrate_file_store(source, target, USABLE, db_path=tmp_path / "none.db", clock=clock)
    assert report.already_there == ["deepseek_api_key"] and report.moved == ["smtp_password"]


def test_a_database_at_an_older_schema_is_upgraded_first(
    source: EncryptedFileKeyring, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-A2: the keys are checked against the current schema only; nothing is removed."""
    db_path = tmp_path / "data" / "twin.db"
    migrate.upgrade(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("UPDATE alembic_version SET version_num = '0001'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(MigrationError, match="twin db upgrade"):
        migrate_file_store(source, CredentialManagerDouble(), USABLE, db_path=db_path, clock=clock)
    assert source.exists()


def test_a_read_back_that_differs_keeps_the_file(
    source: EncryptedFileKeyring, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-A2: a credential manager that stores something else is caught before the delete."""

    class Mangling(CredentialManagerDouble):
        def set_password(self, service_name: str, username: str, password: str, /) -> None:
            super().set_password(service_name, username, password[::-1])

    with pytest.raises(MigrationError, match="read back differently"):
        migrate_file_store(source, Mangling(), USABLE, db_path=tmp_path / "none.db", clock=clock)
    assert source.exists()


def test_any_failure_of_the_credential_manager_keeps_the_file(
    source: EncryptedFileKeyring, tmp_path: Path, clock: ManualClock
) -> None:
    """R18-A2: an error that is not a keyring error (pywin32 raises its own) is reported too."""

    class Broken(CredentialManagerDouble):
        def get_password(self, service_name: str, username: str, /) -> str | None:
            raise OSError("the vault is locked")

    with pytest.raises(MigrationError, match="OSError"):
        migrate_file_store(source, Broken(), USABLE, db_path=tmp_path / "none.db", clock=clock)
    assert source.exists()


def test_the_command_without_a_database_says_what_it_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R18-A2: a store with secrets but no database yet is moved, and the output says so."""
    stored = runner.invoke(app, ["secrets", "set", "smtp_password", "--stdin"], input="m-x\n")
    assert stored.exit_code == 0, stored.output
    double = CredentialManagerDouble()
    monkeypatch.setattr(secrets_module.keyring, "get_keyring", lambda: double)
    first = runner.invoke(app, ["secrets", "migrate"])
    assert first.exit_code == 0, first.output
    assert "no database data to check the keys against" in first.output
    assert double.items == {(SERVICE_NAME, "smtp_password"): "m-x"}


def test_the_command_reports_a_failure_with_exit_code_6(monkeypatch: pytest.MonkeyPatch) -> None:
    """R18-A2: a refused move is an exit code, not a traceback, and names no value."""
    stored = runner.invoke(app, ["secrets", "set", "smtp_password", "--stdin"], input="m-x\n")
    assert stored.exit_code == 0, stored.output
    double = CredentialManagerDouble(fail_on_write=2)  # the probe passes, the entry fails
    monkeypatch.setattr(secrets_module.keyring, "get_keyring", lambda: double)
    result = runner.invoke(app, ["secrets", "migrate"])
    assert result.exit_code == int(ExitCode.SECRETS), result.output
    assert "Traceback" not in result.output and "m-x" not in result.output
