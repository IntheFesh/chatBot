"""Persistence of database master keys in the credential store (R-STO-003).

Layout inside the ``wechat-twin`` service of ``keyring``:

* ``db-key-<id>``   hex of one 256-bit key;
* ``db-key-index``  small JSON ``{"v":1,"current":N,"ids":[...],"retired":[...]}``.

One credential per key keeps every entry far below the Windows Credential
Manager size limit.  Retired keys are kept (they still decrypt old rows and old
backups) until a later round's backup policy deletes them.
"""

from __future__ import annotations

import binascii
from collections.abc import Iterable
from typing import Any

import orjson

from twin.config.secrets import SecretStore
from twin.storage.crypto import KEY_BYTES, KeyRing, generate_key

INDEX_NAME = "db-key-index"
_KEY_PREFIX = "db-key-"


class KeyStoreError(RuntimeError):
    """The master key material is missing or inconsistent."""


def key_secret_name(key_id: int) -> str:
    return f"{_KEY_PREFIX}{key_id}"


class KeyStore:
    """Reads and writes the database key ring in a :class:`SecretStore`."""

    def __init__(self, secrets: SecretStore) -> None:
        self._secrets = secrets

    def exists(self) -> bool:
        return self._secrets.exists(INDEX_NAME)

    def load(self) -> KeyRing:
        index = self._read_index()
        keys: dict[int, bytes] = {}
        for key_id in index["ids"]:
            raw = self._secrets.get(key_secret_name(key_id))
            if raw is None:
                raise KeyStoreError(
                    f"database key {key_id} is listed in the key index but its credential "
                    f"'{key_secret_name(key_id)}' is missing from the credential store "
                    f"({self._secrets.info.detail}). Data encrypted with it cannot be read. "
                    "Restore the credential, or restore a backup that was made with the keys "
                    "you still have."
                )
            keys[key_id] = _decode_key(raw, key_id)
        return KeyRing(keys, index["current"], index["retired"])

    def create_initial(self) -> KeyRing:
        """Generate key id 1.  Refuses to overwrite an existing index."""
        if self.exists():
            raise KeyStoreError("a database key already exists; refusing to replace it")
        key = generate_key()
        self._secrets.set(key_secret_name(1), key.hex())
        self._write_index(current=1, ids=[1], retired=[])
        return KeyRing({1: key}, 1)

    def load_or_create(self, *, allow_create: bool) -> KeyRing:
        """Load the ring; create the first key only if ``allow_create``.

        Creating a key when data already exists would silently make that data
        unreadable, so callers pass ``allow_create=False`` unless the database
        is new or empty.
        """
        if self.exists():
            return self.load()
        if not allow_create:
            raise KeyStoreError(
                "no database master key was found in the credential store "
                f"({self._secrets.info.detail}), but the database already contains data. "
                "Generating a new key would make that data unreadable. Restore the original "
                "credential 'db-key-*' (or the machine's credential store) or restore a backup; "
                "if the data is disposable, delete the database file and start again."
            )
        return self.create_initial()

    def add_key(self, ring: KeyRing, *, make_current: bool = True) -> int:
        """Create the next key, persist it, and (optionally) make it current."""
        new_id = max(ring.key_ids) + 1
        key = generate_key()
        self._secrets.set(key_secret_name(new_id), key.hex())
        ring.add_key(new_id, key)
        if make_current:
            ring.set_current(new_id)
        self._write_index_from(ring)
        return new_id

    def retire(self, ring: KeyRing, key_ids: list[int]) -> None:
        """Mark keys as retired (kept for decryption; never deleted here)."""
        for key_id in key_ids:
            ring.retire(key_id)
        self._write_index_from(ring)

    def delete_retired(self, ring: KeyRing, key_ids: Iterable[int]) -> list[int]:
        """Delete retired keys from the credential store (R-STO-003).

        Only for keys nothing needs any more: the caller (the backup retention) has checked that
        no kept backup and no live value depends on them.  The key index is rewritten first, so
        a credential that cannot be deleted is left behind unlisted instead of listed and gone.
        """
        wanted = sorted(set(key_ids))
        for key_id in wanted:
            if key_id not in ring.retired_ids:
                raise KeyStoreError(f"key {key_id} is not retired; only retired keys are deleted")
        for key_id in wanted:
            ring.forget(key_id)
        self._write_index_from(ring)
        for key_id in wanted:
            self._secrets.delete(key_secret_name(key_id))
        return wanted

    def delete_all(self) -> int:
        """Delete every database key credential (used by the purge command).

        Also when the key index is damaged: the credentials ``db-key-1`` to ``db-key-99`` are then
        deleted by name, so that a purge never leaves a usable key behind.
        """
        try:
            ids = list(self._read_index()["ids"]) if self.exists() else []
        except KeyStoreError:
            ids = list(range(1, 100))
        deleted = 0
        for key_id in ids:
            deleted += int(self._secrets.delete(key_secret_name(key_id)))
        self._secrets.delete(INDEX_NAME)
        return deleted

    # ------------------------------------------------------------- internals

    def _read_index(self) -> dict[str, Any]:
        raw = self._secrets.get(INDEX_NAME)
        if raw is None:
            raise KeyStoreError("the database key index is missing from the credential store")
        try:
            index = orjson.loads(raw)
            ids = [int(item) for item in index["ids"]]
            current = int(index["current"])
            retired = [int(item) for item in index["retired"]]
        except (orjson.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise KeyStoreError(f"the database key index is corrupted: {exc}") from exc
        if current not in ids:
            raise KeyStoreError("the database key index names a current key that is not listed")
        return {"ids": ids, "current": current, "retired": retired}

    def _write_index(self, *, current: int, ids: list[int], retired: list[int]) -> None:
        payload = orjson.dumps({"v": 1, "current": current, "ids": ids, "retired": retired})
        self._secrets.set(INDEX_NAME, payload.decode())

    def _write_index_from(self, ring: KeyRing) -> None:
        self._write_index(
            current=ring.current_id,
            ids=list(ring.key_ids),
            retired=sorted(ring.retired_ids),
        )


def _decode_key(raw: str, key_id: int) -> bytes:
    try:
        key = bytes.fromhex(raw)
    except (ValueError, binascii.Error) as exc:
        raise KeyStoreError(f"database key {key_id} is not valid hex") from exc
    if len(key) != KEY_BYTES:
        raise KeyStoreError(f"database key {key_id} has the wrong length")
    return key
