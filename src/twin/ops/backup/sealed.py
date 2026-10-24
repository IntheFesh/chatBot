"""The sealed stream a backup is written in (R-OPS-006, R-STO-003).

A backup is a compressed tar archive, sealed with AES-256-GCM in 1 MiB chunks.  Layout, integers
big-endian::

    magic      6 bytes    b"TWBAK1"
    version    1 byte     1
    key_id     4 bytes    the database key the backup key is derived from (R-STO-003)
    salt       16 bytes   random, per backup
    prefix     7 bytes    random nonce prefix
    chunk      1 byte     log2 of the plaintext chunk size (20 = 1 MiB)
    chunks     ciphertext || 16-byte tag, one per plaintext chunk

The backup key is ``HKDF-SHA256(master key of key_id, salt, info = "twin-backup-v1")``: it is
**derived from** the database master key, so the one set of credentials protects both, and
deleting every database key (``twin purge``) makes every backup undecryptable.  A backup names
the key it needs in its header, so a backup made before a key rotation is restored with the
retired key as long as that key is kept (R-STO-003).

Chunk ``i`` is sealed with the 12-byte nonce ``prefix || i (4 bytes) || final (1 byte)`` and the
whole header as associated data: a chunk cannot be moved, repeated, dropped from the end (the
last one is flagged) or taken under another header, and any change to the file - a flipped
byte anywhere, a truncated tail - fails authentication.  This is what ``twin backup verify``
relies on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from twin.storage.crypto import CryptoError, KeyRing

MAGIC = b"TWBAK1"
VERSION = 1
SALT_SIZE = 16
PREFIX_SIZE = 7
TAG_SIZE = 16
HEADER_SIZE = len(MAGIC) + 1 + 4 + SALT_SIZE + PREFIX_SIZE + 1
DEFAULT_CHUNK_LOG2 = 20
MAX_CHUNK_LOG2 = 26
INFO = b"twin-backup-v1"


class BackupFormatError(CryptoError):
    """The file is not a backup, or it is damaged."""


class BackupDecryptionError(BackupFormatError):
    """The backup failed authentication: wrong key material, or the file was changed."""


@dataclass(frozen=True)
class BackupHeader:
    """The unencrypted start of a backup."""

    key_id: int
    salt: bytes
    prefix: bytes
    chunk_log2: int
    raw: bytes

    @property
    def block(self) -> int:
        return (1 << self.chunk_log2) + TAG_SIZE


def derive_key(master: bytes, salt: bytes) -> bytes:
    """The backup key of one backup."""
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=INFO).derive(master)


def _nonce(prefix: bytes, index: int, final: bool) -> bytes:
    return prefix + index.to_bytes(4, "big") + (b"\x01" if final else b"\x00")


def parse_header(blob: bytes) -> BackupHeader:
    if len(blob) < HEADER_SIZE or not blob.startswith(MAGIC):
        raise BackupFormatError("this is not a wechat-twin backup")
    version = blob[len(MAGIC)]
    if version != VERSION:
        raise BackupFormatError(f"unsupported backup version {version}")
    offset = len(MAGIC) + 1
    key_id = int.from_bytes(blob[offset : offset + 4], "big")
    offset += 4
    salt = blob[offset : offset + SALT_SIZE]
    offset += SALT_SIZE
    prefix = blob[offset : offset + PREFIX_SIZE]
    chunk_log2 = blob[offset + PREFIX_SIZE]
    if not 10 <= chunk_log2 <= MAX_CHUNK_LOG2:
        raise BackupFormatError("the backup header holds an invalid chunk size")
    return BackupHeader(key_id, salt, prefix, chunk_log2, bytes(blob[:HEADER_SIZE]))


def read_header(path: Path) -> BackupHeader:
    """The header of a backup file (nothing is decrypted)."""
    with path.open("rb") as handle:
        return parse_header(handle.read(HEADER_SIZE))


class SealWriter:
    """A write-only file that seals what is written into ``raw`` (``close`` ends the stream)."""

    def __init__(
        self,
        raw: BinaryIO,
        ring: KeyRing,
        *,
        key_id: int | None = None,
        chunk_log2: int = DEFAULT_CHUNK_LOG2,
    ) -> None:
        use_id = ring.current_id if key_id is None else key_id
        salt = os.urandom(SALT_SIZE)
        self._prefix = os.urandom(PREFIX_SIZE)
        self._header = (
            MAGIC
            + bytes([VERSION])
            + use_id.to_bytes(4, "big")
            + salt
            + self._prefix
            + bytes([chunk_log2])
        )
        self.key_id = use_id
        self._aead = AESGCM(derive_key(ring.key_bytes(use_id), salt))
        self._chunk = 1 << chunk_log2
        self._raw = raw
        self._buffer = bytearray()
        self._index = 0
        self._closed = False
        raw.write(self._header)

    def writable(self) -> bool:
        return True

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self._closed:
            raise ValueError("write to a closed SealWriter")
        self._buffer += data
        while len(self._buffer) > self._chunk:
            self._seal(bytes(self._buffer[: self._chunk]), final=False)
            del self._buffer[: self._chunk]
        return len(data)

    def flush(self) -> None:
        self._raw.flush()

    def _seal(self, plain: bytes, *, final: bool) -> None:
        nonce = _nonce(self._prefix, self._index, final)
        self._raw.write(self._aead.encrypt(nonce, plain, self._header))
        self._index += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._seal(bytes(self._buffer), final=True)
        self._buffer.clear()


class SealReader:
    """A read-only file that yields the plaintext of a sealed stream, chunk by chunk."""

    def __init__(self, src: BinaryIO, ring: KeyRing) -> None:
        self.header = parse_header(src.read(HEADER_SIZE))
        master = ring.key_bytes(self.header.key_id)  # UnknownKeyError if the key is gone
        self._aead = AESGCM(derive_key(master, self.header.salt))
        self._src = src
        self._pending: bytes | None = src.read(self.header.block)
        self._index = 0
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def _next_chunk(self) -> bytes:
        pending = self._pending
        if pending is None:
            return b""
        following = self._src.read(self.header.block)
        final = not following
        try:
            plain = self._aead.decrypt(
                _nonce(self.header.prefix, self._index, final), pending, self.header.raw
            )
        except InvalidTag:
            raise BackupDecryptionError(
                "the backup cannot be decrypted: the key does not fit, or the file is damaged "
                "or was changed"
            ) from None
        self._index += 1
        self._pending = None if final else following
        return plain

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self._buffer) < size:
            if self._pending is None:
                break
            self._buffer += self._next_chunk()
        if size < 0:
            data, self._buffer = self._buffer, b""
        else:
            data, self._buffer = self._buffer[:size], self._buffer[size:]
        return data

    def close(self) -> None:
        self._pending = None
        self._buffer = b""
