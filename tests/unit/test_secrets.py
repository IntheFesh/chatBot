"""Credential storage and the encrypted-file fallback (R-CFG-002, CLAUDE.md rule 10)."""

from __future__ import annotations

import os
import stat
import sys
import threading
from pathlib import Path

import pytest
from keyring.errors import PasswordDeleteError

import twin.config.secrets as secrets_module
from tests.support.credentials import BrokenCredentials, MemoryCredentials
from twin.config.secrets import (
    ENV_BACKEND,
    ENV_PASSPHRASE,
    KNOWN_SECRETS,
    SERVICE_NAME,
    EncryptedFileKeyring,
    SecretStore,
    SecretStoreError,
    default_secrets_dir,
    register_secret_name,
    select_backend,
)

VALUE = "sk-synthetic-0123456789abcdef"


def test_secret_store_set_get_exists_delete() -> None:
    store = SecretStore(MemoryCredentials())
    assert store.get("deepseek_api_key") is None
    assert not store.exists("deepseek_api_key")
    store.set("deepseek_api_key", VALUE)
    assert store.get("deepseek_api_key") == VALUE
    assert store.exists("deepseek_api_key")
    assert store.require("deepseek_api_key") == VALUE
    assert store.delete("deepseek_api_key") is True
    assert store.delete("deepseek_api_key") is False


def test_require_names_the_command_to_fix_a_missing_secret() -> None:
    with pytest.raises(SecretStoreError, match="twin secrets set smtp_password"):
        SecretStore(MemoryCredentials()).require("smtp_password")


@pytest.mark.parametrize("name", ["", "has space", "a/b", "x" * 65, "ключ"])
def test_invalid_secret_names_are_rejected(name: str) -> None:
    store = SecretStore(MemoryCredentials())
    with pytest.raises(ValueError, match="invalid secret name"):
        store.set(name, "v")


def test_empty_secret_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        SecretStore(MemoryCredentials()).set("deepseek_api_key", "")


def test_backend_failures_become_actionable_errors() -> None:
    class Failing(MemoryCredentials):
        def get_password(self, service_name: str, username: str, /) -> str | None:
            from keyring.errors import KeyringError

            raise KeyringError("locked")

    store = SecretStore(Failing())
    with pytest.raises(SecretStoreError, match="cannot read secret"):
        store.get("deepseek_api_key")


def test_register_secret_name_extends_the_known_list() -> None:
    register_secret_name("ilink_credentials", "iLink login credentials")
    try:
        assert "ilink_credentials" in KNOWN_SECRETS
    finally:
        del KNOWN_SECRETS["ilink_credentials"]
    with pytest.raises(ValueError):
        register_secret_name("bad name", "x")


# ----------------------------------------------------- encrypted file backend


def make_file_backend(tmp_path: Path, passphrase: str | None = None) -> EncryptedFileKeyring:
    return EncryptedFileKeyring(tmp_path / "store", passphrase)


def test_file_backend_round_trip_across_instances(tmp_path: Path) -> None:
    first = make_file_backend(tmp_path)
    first.set_password(SERVICE_NAME, "alpha", VALUE)
    first.set_password(SERVICE_NAME, "beta", "second")
    second = make_file_backend(tmp_path)  # new object, same files
    assert second.get_password(SERVICE_NAME, "alpha") == VALUE
    assert second.get_password(SERVICE_NAME, "beta") == "second"
    assert second.get_password(SERVICE_NAME, "missing") is None
    assert second.get_password("other-service", "alpha") is None


def test_file_backend_never_stores_plaintext(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path)
    backend.set_password(SERVICE_NAME, "deepseek_api_key", VALUE)
    blob = (tmp_path / "store" / "secrets.enc").read_bytes()
    assert blob.startswith(b"TWSEC1")
    assert VALUE.encode() not in blob
    assert b"deepseek_api_key" not in blob
    assert SERVICE_NAME.encode() not in blob


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_file_backend_files_are_owner_only(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path)
    backend.set_password(SERVICE_NAME, "a", "b")
    for name in ("secrets.enc", "secrets.key"):
        mode = stat.S_IMODE(os.stat(tmp_path / "store" / name).st_mode)
        assert mode == 0o600
    assert stat.S_IMODE(os.stat(tmp_path / "store").st_mode) == 0o700


def test_file_backend_detects_tampering(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path)
    backend.set_password(SERVICE_NAME, "a", "b")
    path = tmp_path / "store" / "secrets.enc"
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0x01
    path.write_bytes(bytes(blob))
    with pytest.raises(SecretStoreError, match="cannot decrypt"):
        backend.get_password(SERVICE_NAME, "a")


def test_file_backend_rejects_foreign_files(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path)
    backend.directory.mkdir(parents=True)
    (backend.directory / "secrets.enc").write_bytes(b"this is not an encrypted store at all")
    with pytest.raises(SecretStoreError, match="not a valid encrypted secret store"):
        backend.get_password(SERVICE_NAME, "a")


def test_file_backend_rejects_a_damaged_key_file(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path)
    backend.set_password(SERVICE_NAME, "a", "b")
    (tmp_path / "store" / "secrets.key").write_bytes(b"short")
    with pytest.raises(SecretStoreError, match="corrupted"):
        backend.get_password(SERVICE_NAME, "a")


def test_file_backend_passphrase_mode(tmp_path: Path) -> None:
    protected = make_file_backend(tmp_path, "correct horse battery staple")
    protected.set_password(SERVICE_NAME, "k", VALUE)
    assert not (tmp_path / "store" / "secrets.key").exists()  # no key file in this mode
    assert make_file_backend(tmp_path, "correct horse battery staple").get_password(
        SERVICE_NAME, "k"
    ) == VALUE
    with pytest.raises(SecretStoreError, match="cannot decrypt"):
        make_file_backend(tmp_path, "wrong passphrase").get_password(SERVICE_NAME, "k")
    with pytest.raises(SecretStoreError, match=ENV_PASSPHRASE):
        make_file_backend(tmp_path).get_password(SERVICE_NAME, "k")


def test_file_backend_refuses_to_switch_modes_silently(tmp_path: Path) -> None:
    make_file_backend(tmp_path).set_password(SERVICE_NAME, "k", "v")
    with pytest.raises(SecretStoreError, match="different mode"):
        make_file_backend(tmp_path, "now with a passphrase").set_password(SERVICE_NAME, "k2", "v")


def test_file_backend_delete(tmp_path: Path) -> None:
    backend = make_file_backend(tmp_path)
    backend.set_password(SERVICE_NAME, "k", "v")
    backend.delete_password(SERVICE_NAME, "k")
    assert backend.get_password(SERVICE_NAME, "k") is None
    with pytest.raises(PasswordDeleteError):
        backend.delete_password(SERVICE_NAME, "k")


def test_file_backend_concurrent_writers_do_not_lose_updates(tmp_path: Path) -> None:
    errors: list[BaseException] = []

    def writer(index: int) -> None:
        try:
            backend = make_file_backend(tmp_path)
            for step in range(5):
                backend.set_password(SERVICE_NAME, f"w{index}-{step}", f"v{index}-{step}")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    backend = make_file_backend(tmp_path)
    for index in range(4):
        for step in range(5):
            assert backend.get_password(SERVICE_NAME, f"w{index}-{step}") == f"v{index}-{step}"


# ---------------------------------------------------------- backend selection


def test_default_secrets_dir_prefers_explicit_then_xdg(tmp_path: Path) -> None:
    assert default_secrets_dir({secrets_module.ENV_SECRETS_DIR: str(tmp_path)}) == tmp_path
    assert default_secrets_dir({"XDG_CONFIG_HOME": str(tmp_path)}) == tmp_path / "wechat-twin"
    assert default_secrets_dir({}).name == "wechat-twin"


def test_forced_file_backend_is_usable(tmp_path: Path) -> None:
    backend, info = select_backend({ENV_BACKEND: "file", secrets_module.ENV_SECRETS_DIR: str(tmp_path)})
    assert info.kind == "file" and info.usable
    assert isinstance(backend, EncryptedFileKeyring)
    assert "AES-256-GCM" in info.detail
    SecretStore(backend, info).set("deepseek_api_key", VALUE)


def test_unusable_system_keyring_falls_back_on_linux_with_a_diagnosis(tmp_path: Path) -> None:
    # In the sandbox keyring's own default is the "fail" backend: nothing is forced here.
    _backend, info = select_backend({secrets_module.ENV_SECRETS_DIR: str(tmp_path)}, "linux")
    assert info.kind == "file" and info.usable
    assert "system keyring unavailable" in info.detail


def test_windows_never_falls_back_to_a_file(tmp_path: Path) -> None:
    _backend, info = select_backend({secrets_module.ENV_SECRETS_DIR: str(tmp_path)}, "win32")
    assert info.kind == "system"
    assert not info.usable
    assert "unavailable" in info.detail


def test_working_system_keyring_is_preferred(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class SystemKeyring(MemoryCredentials):
        priority = 5

    monkeypatch.setattr(secrets_module.keyring, "get_keyring", SystemKeyring)
    backend, info = select_backend({secrets_module.ENV_SECRETS_DIR: str(tmp_path)}, "linux")
    assert info.kind == "system" and info.usable
    assert isinstance(backend, SystemKeyring)


def test_broken_system_keyring_is_probed_and_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BrokenSystem(BrokenCredentials):
        priority = 5

    monkeypatch.setattr(secrets_module.keyring, "get_keyring", BrokenSystem)
    _backend, info = select_backend({secrets_module.ENV_SECRETS_DIR: str(tmp_path)}, "linux")
    assert info.kind == "file"
    assert "credential store unavailable" in info.detail


def test_forcing_an_unknown_backend_name_is_an_error() -> None:
    with pytest.raises(SecretStoreError, match="must be 'system' or 'file'"):
        select_backend({ENV_BACKEND: "registry"})


def test_forced_system_backend_does_not_fall_back(tmp_path: Path) -> None:
    _backend, info = select_backend(
        {ENV_BACKEND: "system", secrets_module.ENV_SECRETS_DIR: str(tmp_path)}, "linux"
    )
    assert info.kind == "system" and not info.usable


def test_secret_store_default_uses_environment(tmp_path: Path) -> None:
    store = SecretStore.default()  # conftest forces the file backend inside tmp_path
    store.set("smtp_password", VALUE)
    assert store.info.kind == "file"
    assert SecretStore.default().get("smtp_password") == VALUE
