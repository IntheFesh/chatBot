"""Secret storage (R-CFG-002, CLAUDE.md rule 10).

Secrets (DeepSeek key, SMTP password, database master keys, AutoDL password,
iLink credentials) live only in the operating system's credential store via
``keyring`` (Windows Credential Manager on the target machine), service name
``wechat-twin``.  They are never written to configuration files or logs.

Linux/macOS without a usable keyring daemon (headless servers, CI, sandboxes)
fall back to :class:`EncryptedFileKeyring`: a real AES-256-GCM encrypted file,
keyed either by a passphrase (``TWIN_KEYRING_PASSPHRASE``, scrypt) or by a
random key file kept next to it with mode 0600.  Windows always uses the native
credential manager; if that is unavailable ``twin doctor`` reports a failure.
"""

from __future__ import annotations

import base64
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import keyring
import orjson
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from keyring.errors import KeyringError, PasswordDeleteError

from twin.ops.filelock import FileLock

SERVICE_NAME = "wechat-twin"
ENV_BACKEND = "TWIN_KEYRING_BACKEND"
ENV_PASSPHRASE = "TWIN_KEYRING_PASSPHRASE"  # noqa: S105 - environment variable name, not a secret
ENV_SECRETS_DIR = "TWIN_SECRETS_DIR"

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAGIC = b"TWSEC1"
_MODE_KEYFILE = 1
_MODE_PASSPHRASE = 2

# secrets a user may set with `twin secrets set <name>`  (name -> description)
KNOWN_SECRETS: dict[str, str] = {
    "deepseek_api_key": "DeepSeek API key",
    "smtp_password": "SMTP password for alert e-mail",
    "autodl_password": "AutoDL SSH password",
}


def register_secret_name(name: str, description: str) -> None:
    """Make ``name`` accepted by ``twin secrets set|delete|check`` (later rounds)."""
    _validate_name(name)
    KNOWN_SECRETS[name] = description


class SecretStoreError(RuntimeError):
    """The credential store could not be read or written."""


def _validate_name(name: str) -> None:
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid secret name {name!r}: use letters, digits, '_', '.', '-'")


def default_secrets_dir(env: dict[str, str] | None = None) -> Path:
    """Directory of the encrypted fallback store (outside the data directory)."""
    environ = os.environ if env is None else env
    explicit = environ.get(ENV_SECRETS_DIR)
    if explicit:
        return Path(explicit)
    config_home = environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "wechat-twin"


class CredentialBackend(Protocol):
    """The three operations of a ``keyring`` backend that the secret store uses."""

    def get_password(self, service_name: str, username: str, /) -> str | None: ...

    def set_password(self, service_name: str, username: str, password: str, /) -> None: ...

    def delete_password(self, service_name: str, username: str, /) -> None: ...


class EncryptedFileKeyring:
    """File-backed credential backend with real authenticated encryption.

    Not a ``keyring.backend.KeyringBackend`` subclass on purpose: keyring auto-registers every
    subclass and would try to instantiate it during backend discovery.

    Format: ``MAGIC | mode | salt(16) | nonce(12) | AES-256-GCM(JSON items)`` with
    ``MAGIC | mode | salt`` as associated data.  Writes are atomic and serialised
    across processes with a file lock.
    """

    def __init__(self, directory: Path, passphrase: str | None = None) -> None:
        self.directory = directory
        self._passphrase = passphrase or None
        self._data_path = directory / "secrets.enc"
        self._key_path = directory / "secrets.key"
        self._lock_path = directory / "secrets.lock"

    # -- keyring API -------------------------------------------------------

    def get_password(self, service: str, username: str) -> str | None:
        with self._exclusive():
            return self._load().get(service, {}).get(username)

    def set_password(self, service: str, username: str, password: str) -> None:
        with self._exclusive():
            items = self._load()
            items.setdefault(service, {})[username] = password
            self._store(items)

    def delete_password(self, service: str, username: str) -> None:
        with self._exclusive():
            items = self._load()
            if username not in items.get(service, {}):
                raise PasswordDeleteError(f"{username} not found in the encrypted file keyring")
            del items[service][username]
            if not items[service]:
                del items[service]
            self._store(items)

    # -- internals ---------------------------------------------------------

    def _exclusive(self) -> FileLock:
        self.directory.mkdir(parents=True, exist_ok=True)
        _restrict(self.directory, 0o700)
        return FileLock(self._lock_path)

    def _derive_key(self, mode: int, salt: bytes) -> bytes:
        if mode == _MODE_PASSPHRASE:
            if not self._passphrase:
                raise SecretStoreError(
                    "the encrypted secret store was created with a passphrase; "
                    f"set {ENV_PASSPHRASE}"
                )
            kdf = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1)
            return kdf.derive(self._passphrase.encode())
        return self._read_or_create_keyfile()

    def _read_or_create_keyfile(self) -> bytes:
        if self._key_path.exists():
            key = self._key_path.read_bytes()
            if len(key) != 32:
                raise SecretStoreError(f"{self._key_path} is corrupted (expected 32 bytes)")
            return key
        key = os.urandom(32)
        fd = os.open(self._key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(key)
        return key

    def _load(self) -> dict[str, dict[str, str]]:
        if not self._data_path.exists():
            return {}
        blob = self._data_path.read_bytes()
        header_len = len(_MAGIC) + 1 + 16
        if len(blob) < header_len + 12 + 16 or not blob.startswith(_MAGIC):
            raise SecretStoreError(f"{self._data_path} is not a valid encrypted secret store")
        mode = blob[len(_MAGIC)]
        salt = blob[len(_MAGIC) + 1 : header_len]
        nonce = blob[header_len : header_len + 12]
        key = self._derive_key(mode, salt)
        try:
            plain = AESGCM(key).decrypt(nonce, blob[header_len + 12 :], blob[:header_len])
        except InvalidTag:
            raise SecretStoreError(
                f"cannot decrypt {self._data_path}: wrong passphrase / key file, or the file "
                "was modified"
            ) from None
        decoded = orjson.loads(plain)
        return {
            str(svc): {str(u): str(p) for u, p in users.items()} for svc, users in decoded.items()
        }

    def _store(self, items: dict[str, dict[str, str]]) -> None:
        mode = _MODE_PASSPHRASE if self._passphrase else _MODE_KEYFILE
        if self._data_path.exists():
            existing_mode = self._data_path.read_bytes()[len(_MAGIC)]
            if existing_mode != mode:
                raise SecretStoreError(
                    "the secret store was created in a different mode (passphrase vs key file); "
                    f"{'unset' if existing_mode == _MODE_KEYFILE else 'set'} {ENV_PASSPHRASE} "
                    "to match it"
                )
        salt = os.urandom(16)
        header = _MAGIC + bytes([mode]) + salt
        nonce = os.urandom(12)
        key = self._derive_key(mode, salt)
        sealed = AESGCM(key).encrypt(nonce, orjson.dumps(items), header)
        tmp = self._data_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(header + nonce + sealed)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._data_path)


def _restrict(path: Path, mode: int) -> None:
    if sys.platform != "win32":
        path.chmod(mode)


@dataclass(frozen=True)
class BackendInfo:
    """Which credential store is in use and why (shown by ``twin doctor``)."""

    name: str
    kind: Literal["system", "file"]
    usable: bool
    detail: str


def _probe(backend: CredentialBackend) -> str | None:
    """Round-trip a throw-away item; return an error description or ``None``."""
    probe_name = "twin-probe-" + base64.b32encode(os.urandom(5)).decode().rstrip("=").lower()
    try:
        backend.set_password(SERVICE_NAME, probe_name, "probe")
        if backend.get_password(SERVICE_NAME, probe_name) != "probe":
            return "read-back mismatch"
        backend.delete_password(SERVICE_NAME, probe_name)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _is_system_candidate(backend: CredentialBackend) -> bool:
    from keyring.backends.fail import Keyring as FailKeyring

    return not isinstance(backend, FailKeyring) and getattr(backend, "priority", 0) > 0


def select_backend(
    env: dict[str, str] | None = None,
    platform: str | None = None,
    *,
    probe_system: bool = True,
) -> tuple[CredentialBackend, BackendInfo]:
    """Choose the credential backend.

    ``TWIN_KEYRING_BACKEND=system|file`` forces a choice.  Otherwise Windows uses
    the Credential Manager and other platforms use the system keyring when it
    works, else the encrypted file store.  ``probe_system=False`` skips the
    write/read/delete round trip against the system keyring (``twin doctor`` does
    its own, so ordinary commands do not churn the Windows Credential Manager).
    """
    environ = dict(os.environ) if env is None else env
    plat = sys.platform if platform is None else platform
    forced = environ.get(ENV_BACKEND, "").strip().lower()
    if forced not in ("", "system", "file"):
        raise SecretStoreError(f"{ENV_BACKEND} must be 'system' or 'file', got {forced!r}")

    system_error: str | None = None
    if forced != "file":
        system = keyring.get_keyring()
        if _is_system_candidate(system):
            system_error = _probe(system) if probe_system else None
            if system_error is None:
                return system, BackendInfo(
                    name=type(system).__name__, kind="system", usable=True, detail="system keyring"
                )
        else:
            system_error = (
                f"no usable system keyring backend "
                f"({type(system).__module__}.{type(system).__qualname__})"
            )
        if plat == "win32" or forced == "system":
            return system, BackendInfo(
                name=type(system).__name__,
                kind="system",
                usable=False,
                detail=f"system keyring unavailable: {system_error}",
            )

    directory = default_secrets_dir(environ)
    file_backend = EncryptedFileKeyring(directory, environ.get(ENV_PASSPHRASE))
    mode = "passphrase" if environ.get(ENV_PASSPHRASE) else "key file"
    reason = f"system keyring unavailable ({system_error}); " if system_error else "forced; "
    file_error = _probe(file_backend)
    return file_backend, BackendInfo(
        name="EncryptedFileKeyring",
        kind="file",
        usable=file_error is None,
        detail=(
            f"{reason}using AES-256-GCM encrypted file {directory / 'secrets.enc'} ({mode})"
            + (f"; file store error: {file_error}" if file_error else "")
        ),
    )


class SecretStore:
    """Named secrets in the credential store (service ``wechat-twin``)."""

    def __init__(self, backend: CredentialBackend, info: BackendInfo | None = None) -> None:
        self._backend = backend
        self.info = info or BackendInfo(
            name=type(backend).__name__, kind="system", usable=True, detail="explicit backend"
        )

    @classmethod
    def default(cls) -> SecretStore:
        # Windows always uses the native Credential Manager: no probe on every command.
        backend, info = select_backend(probe_system=sys.platform != "win32")
        return cls(backend, info)

    def get(self, name: str) -> str | None:
        _validate_name(name)
        try:
            return self._backend.get_password(SERVICE_NAME, name)
        except KeyringError as exc:
            raise SecretStoreError(f"cannot read secret {name!r}: {exc}") from exc

    def require(self, name: str) -> str:
        value = self.get(name)
        if value is None:
            raise SecretStoreError(f"secret {name!r} is not set. Run: twin secrets set {name}")
        return value

    def set(self, name: str, value: str) -> None:
        _validate_name(name)
        if not value:
            raise ValueError("refusing to store an empty secret")
        try:
            self._backend.set_password(SERVICE_NAME, name, value)
        except KeyringError as exc:
            raise SecretStoreError(f"cannot write secret {name!r}: {exc}") from exc

    def delete(self, name: str) -> bool:
        """Delete ``name``; returns ``False`` if it did not exist."""
        _validate_name(name)
        try:
            self._backend.delete_password(SERVICE_NAME, name)
        except PasswordDeleteError:
            return False
        except KeyringError as exc:
            raise SecretStoreError(f"cannot delete secret {name!r}: {exc}") from exc
        return True

    def exists(self, name: str) -> bool:
        return self.get(name) is not None
