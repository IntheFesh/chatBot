"""``twin secrets migrate``: from the encrypted file store into the Windows Credential Manager.

The user's decision of round 18 (item A2, D-828): the file store (``TWIN_KEYRING_BACKEND=file``)
is refused by every command on Windows (D-827), and this is the way out.  In order, and nothing is
removed unless every step worked:

1. read every entry of the file store - the DeepSeek key, the SMTP and AutoDL passwords, the key
   index and every database key, the retired ones included (old rows and old backups need them);
2. refuse before writing anything if the credential manager already holds a *different* value
   under one of those names (an entry with the same value is fine: a second run after a failed
   one);
3. write each entry into the credential manager and read each back: it must be the same;
4. load the database keys from the credential manager and decrypt the first rows of every
   encrypted table with them (:func:`twin.storage.rotate.open_samples`);
5. only then delete the file store (``secrets.enc``, ``secrets.key``).

The report names entries, never values.  The command is EXCLUSIVE (the application must not run
while its keys move) and exempt from the weak-store refusal and the consent gate (D-825, D-827).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from twin.clock import Clock, get_clock
from twin.config.secrets import (
    BackendInfo,
    CredentialBackend,
    EncryptedFileKeyring,
    SecretStore,
    SecretStoreError,
)
from twin.ops.logging import get_logger
from twin.storage.crypto import CryptoError
from twin.storage.db import Database
from twin.storage.keystore import KeyStore, KeyStoreError
from twin.storage.migrate import SchemaState, schema_status
from twin.storage.rotate import open_samples

log = get_logger("twin.ops.secrets_migrate")

KEPT = "the file store was kept"


class MigrationError(SecretStoreError):
    """The entries could not be moved safely; the file store was not touched (exit code 6)."""


@dataclass
class MigrationReport:
    """What :func:`migrate_file_store` did (names and counts, never values)."""

    moved: list[str] = field(default_factory=list)
    already_there: list[str] = field(default_factory=list)
    values_opened: int = 0
    database_checked: bool = False
    removed: list[Path] = field(default_factory=list)


def _names(keys: list[tuple[str, str]]) -> str:
    return ", ".join(sorted(name for _service, name in keys))


def _check_database(db_path: Path, target: SecretStore, clock: Clock) -> tuple[bool, int]:
    """Open the data with the keys now in ``target``: ``(checked, values opened)``."""
    state = schema_status(db_path).state
    if state in (SchemaState.MISSING, SchemaState.EMPTY):
        return False, 0
    if state is not SchemaState.CURRENT:
        raise MigrationError(
            f"the database is not at the current schema ({state.value}); run `twin db upgrade` "
            f"first, so that the moved keys can be checked against it; {KEPT}"
        )
    keystore = KeyStore(target)
    if not keystore.exists():
        return False, 0
    db = Database(db_path, clock=clock)
    try:
        return True, open_samples(db, keystore.load())
    except (CryptoError, KeyStoreError) as exc:
        raise MigrationError(
            f"the moved database keys do not open the data ({type(exc).__name__}); {KEPT}"
        ) from exc
    finally:
        db.dispose()


def migrate_file_store(
    source: EncryptedFileKeyring,
    target: CredentialBackend,
    target_info: BackendInfo,
    *,
    db_path: Path,
    clock: Clock | None = None,
) -> MigrationReport:
    """Move every entry of ``source`` into ``target``, check it, then remove ``source``.

    ``clock`` is the one of the database that is opened for the check (default: the process's).

    Raises :class:`MigrationError` - with ``source`` untouched - when the target is not usable,
    holds a different value under one of the names, a write or a read-back fails, or the moved
    keys do not decrypt the data.
    """
    report = MigrationReport()
    if not target_info.usable:
        raise MigrationError(
            f"the Windows Credential Manager is not usable here ({target_info.detail}); {KEPT}"
        )
    entries = source.entries()
    try:
        present = {key: target.get_password(*key) for key in entries}
        conflicts = [key for key, value in present.items() if value not in (None, entries[key])]
        if conflicts:
            raise MigrationError(
                f"the credential manager already holds a different value for {_names(conflicts)}; "
                f"nothing was written, {KEPT}. Compare them and delete the wrong one, then run "
                "the command again"
            )
        for key, value in sorted(entries.items()):
            if present[key] == value:
                report.already_there.append(key[1])
                continue
            target.set_password(*key, value)
            report.moved.append(key[1])
        differing = [key for key, value in entries.items() if target.get_password(*key) != value]
    except MigrationError:
        raise
    except Exception as exc:  # whatever the credential manager raises: the file stays (no value)
        raise MigrationError(
            f"the credential manager failed ({type(exc).__name__}: {exc}); {KEPT}"
        ) from exc
    if differing:
        raise MigrationError(f"read back differently: {_names(differing)}; {KEPT}")
    report.database_checked, report.values_opened = _check_database(
        db_path, SecretStore(target, target_info), clock or get_clock()
    )
    report.removed = source.remove()
    log.info(
        "secrets_migrated",
        moved=len(report.moved),
        already_there=len(report.already_there),
        values_opened=report.values_opened,
        removed=len(report.removed),
    )
    return report
