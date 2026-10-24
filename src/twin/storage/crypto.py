"""AES-256-GCM sealing of database fields (R-STO-002, R-STO-003, R-PRIV-004).

Sealed blob layout (all integers big-endian)::

    +---------+------------+-------------+----------------------------+
    | version | key_id     | nonce       | ciphertext || GCM tag (16B) |
    | 1 byte  | 4 bytes    | 12 bytes    | len(plaintext) + 16        |
    +---------+------------+-------------+----------------------------+

* a fresh random 96-bit nonce is drawn for every write;
* the header ``version || key_id`` plus the caller-supplied associated data
  (table, primary key, column; see :func:`build_aad`) are authenticated, so a
  ciphertext copied to another row or column, or re-labelled with another
  ``key_id``, fails to decrypt;
* ``key_id`` selects the key at decrypt time, which is what makes key rotation
  possible: a :class:`KeyRing` holds the current key and every retired key.
"""

from __future__ import annotations

import os
import struct
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BLOB_VERSION = 1
KEY_BYTES = 32
KEY_ID_BYTES = 4
NONCE_BYTES = 12
TAG_BYTES = 16
HEADER_BYTES = 1 + KEY_ID_BYTES
MIN_BLOB_BYTES = HEADER_BYTES + NONCE_BYTES + TAG_BYTES

_AAD_PREFIX = b"twin-aad-v1"


class CryptoError(Exception):
    """Base class for encryption-layer failures."""


class DecryptionError(CryptoError):
    """The ciphertext could not be authenticated or parsed."""


class UnknownKeyError(DecryptionError):
    """The ciphertext names a ``key_id`` that is not in the key ring."""

    def __init__(self, key_id: int) -> None:
        self.key_id = key_id
        super().__init__(
            f"encryption key id {key_id} is not available in the key ring. "
            "Restore the key from the credential store (Windows Credential Manager entry "
            "'wechat-twin' / db-key-N), or restore a backup together with its keys; "
            "without the key this data cannot be decrypted."
        )


class KeyRingNotConfiguredError(CryptoError):
    """No key ring has been installed for this process."""


class SealedBlob(bytes):
    """Marker type for already-encrypted bytes.

    Encrypted column types accept only this type, which makes it impossible to
    store a plaintext value in an encrypted column by accident.
    """

    __slots__ = ()

    @property
    def key_id(self) -> int:
        return blob_key_id(self)


def _lp(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def build_aad(table: str, pk: Sequence[object], column: str) -> bytes:
    """Associated data binding a ciphertext to ``(table, primary key, column)``."""
    parts = [_AAD_PREFIX, _lp(table.encode()), struct.pack(">H", len(pk))]
    parts.extend(_lp(str(item).encode()) for item in pk)
    parts.append(_lp(column.encode()))
    return b"".join(parts)


def blob_key_id(blob: bytes) -> int:
    """Return the ``key_id`` recorded in a sealed blob without decrypting it."""
    if len(blob) < MIN_BLOB_BYTES:
        raise DecryptionError("ciphertext is too short to be a sealed value")
    if blob[0] != BLOB_VERSION:
        raise DecryptionError(f"unsupported ciphertext version {blob[0]}")
    return int.from_bytes(blob[1:HEADER_BYTES], "big")


class KeyRing:
    """The current encryption key plus retired keys kept for decryption."""

    def __init__(
        self, keys: Mapping[int, bytes], current_id: int, retired: Iterable[int] = ()
    ) -> None:
        self._keys: dict[int, bytes] = {}
        self._ciphers: dict[int, AESGCM] = {}
        self._retired: set[int] = set(retired)
        for key_id, key in keys.items():
            self.add_key(key_id, key)
        if current_id not in self._keys:
            raise ValueError(f"current key id {current_id} is not in the key ring")
        if current_id in self._retired:
            raise ValueError("the current key cannot be retired")
        self._current_id = current_id

    def add_key(self, key_id: int, key: bytes) -> None:
        if not 0 < key_id < 2 ** (8 * KEY_ID_BYTES):
            raise ValueError(f"invalid key id {key_id}")
        if len(key) != KEY_BYTES:
            raise ValueError("encryption keys must be exactly 32 bytes (AES-256)")
        self._keys[key_id] = bytes(key)
        self._ciphers[key_id] = AESGCM(bytes(key))

    @property
    def current_id(self) -> int:
        return self._current_id

    @property
    def key_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._keys))

    @property
    def retired_ids(self) -> frozenset[int]:
        return frozenset(self._retired)

    def key_bytes(self, key_id: int) -> bytes:
        try:
            return self._keys[key_id]
        except KeyError:
            raise UnknownKeyError(key_id) from None

    def set_current(self, key_id: int) -> None:
        if key_id not in self._keys:
            raise UnknownKeyError(key_id)
        if key_id in self._retired:
            raise ValueError("a retired key cannot become current")
        self._current_id = key_id

    def retire(self, key_id: int) -> None:
        if key_id == self._current_id:
            raise ValueError("the current key cannot be retired")
        if key_id not in self._keys:
            raise UnknownKeyError(key_id)
        self._retired.add(key_id)

    def forget(self, key_id: int) -> None:
        """Remove a retired key from memory (the caller deletes it from storage)."""
        if key_id == self._current_id:
            raise ValueError("the current key cannot be removed")
        self._keys.pop(key_id, None)
        self._ciphers.pop(key_id, None)
        self._retired.discard(key_id)

    def seal(self, plaintext: bytes, aad: bytes, *, key_id: int | None = None) -> SealedBlob:
        """Encrypt ``plaintext`` with the current key (or an explicit active key)."""
        use_id = self._current_id if key_id is None else key_id
        if use_id in self._retired:
            raise ValueError("refusing to seal with a retired key")
        cipher = self._ciphers.get(use_id)
        if cipher is None:
            raise UnknownKeyError(use_id)
        header = bytes([BLOB_VERSION]) + use_id.to_bytes(KEY_ID_BYTES, "big")
        nonce = os.urandom(NONCE_BYTES)
        return SealedBlob(header + nonce + cipher.encrypt(nonce, plaintext, header + aad))

    def open(self, blob: bytes, aad: bytes) -> bytes:
        """Decrypt and authenticate ``blob`` using the key it names."""
        key_id = blob_key_id(blob)
        cipher = self._ciphers.get(key_id)
        if cipher is None:
            raise UnknownKeyError(key_id)
        header = bytes(blob[:HEADER_BYTES])
        nonce = bytes(blob[HEADER_BYTES : HEADER_BYTES + NONCE_BYTES])
        try:
            return cipher.decrypt(nonce, bytes(blob[HEADER_BYTES + NONCE_BYTES :]), header + aad)
        except InvalidTag:
            raise DecryptionError(
                "authentication failed: wrong key, corrupted data, or the ciphertext was "
                "moved to a different row or column"
            ) from None


# --------------------------------------------------------------- active ring

_lock = threading.Lock()
_active_ring: KeyRing | None = None


def get_keyring() -> KeyRing:
    """Return the process-wide key ring."""
    ring = _active_ring
    if ring is None:
        raise KeyRingNotConfiguredError(
            "no encryption key ring is installed; build the services container first"
        )
    return ring


def set_active_keyring(ring: KeyRing | None) -> KeyRing | None:
    """Install ``ring`` as the process-wide key ring; return the previous one."""
    global _active_ring
    with _lock:
        previous = _active_ring
        _active_ring = ring
    return previous


@contextmanager
def use_keyring(ring: KeyRing) -> Iterator[KeyRing]:
    """Temporarily install ``ring`` (used by tests and short-lived tools)."""
    previous = set_active_keyring(ring)
    try:
        yield ring
    finally:
        set_active_keyring(previous)


def generate_key() -> bytes:
    """A fresh random AES-256 key."""
    return os.urandom(KEY_BYTES)
