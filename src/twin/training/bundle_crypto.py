"""Passphrase encryption of the training package (R-TRN-008, R-PRIV-003).

This file depends on the standard library and ``cryptography`` only: the training bundle ships a
copy of it next to ``decrypt_bundle.py`` so that the AutoDL instance, which has no ``twin``
package, decrypts with the very code that encrypted.

Format (version 1), all integers big-endian::

    magic      8 bytes   b"TWINBNDL"
    version    1 byte    1
    scrypt     3 bytes   log2(N), r, p          (N = 2**17, r = 8, p = 1 by default)
    salt       16 bytes
    prefix     7 bytes   random nonce prefix
    chunk      1 byte    log2 of the plaintext chunk size (20 = 1 MiB)
    chunks     ciphertext || 16-byte tag, one per plaintext chunk

The key is ``scrypt(passphrase, salt)``, 32 bytes, used with AES-256-GCM.  Chunk ``i`` is sealed
with the 12-byte nonce ``prefix || i (4 bytes) || final (1 byte)`` and the whole header as
associated data, so a chunk cannot be moved, repeated, dropped from the end (the last chunk is
flagged as final) or sealed under another header.  Every chunk but the last is exactly one chunk
size long; the last one may be shorter (or empty for an empty message).  A wrong passphrase and a
damaged file both fail at the first chunk with :class:`BundleDecryptionError`.
"""

from __future__ import annotations

import os
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"TWINBNDL"
VERSION = 1
TAG_SIZE = 16
SALT_SIZE = 16
PREFIX_SIZE = 7
HEADER_SIZE = len(MAGIC) + 1 + 3 + SALT_SIZE + PREFIX_SIZE + 1
DEFAULT_LOG_N = 17
DEFAULT_R = 8
DEFAULT_P = 1
DEFAULT_CHUNK_LOG2 = 20
MAX_LOG_N = 22
MIN_PASSPHRASE_CHARS = 12


class BundleDecryptionError(Exception):
    """The package is damaged or the passphrase is wrong."""


def derive_key(passphrase: str, salt: bytes, log_n: int, r: int, p: int) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=1 << log_n, r=r, p=p)
    return kdf.derive(passphrase.encode("utf-8"))


def _nonce(prefix: bytes, index: int, final: bool) -> bytes:
    return prefix + index.to_bytes(4, "big") + (b"\x01" if final else b"\x00")


class EncryptWriter:
    """A write-only binary file that seals what is written to it into ``raw``.

    ``close()`` writes the final chunk; the underlying file is left open for the caller.
    """

    def __init__(
        self,
        raw: BinaryIO,
        passphrase: str,
        *,
        log_n: int = DEFAULT_LOG_N,
        r: int = DEFAULT_R,
        p: int = DEFAULT_P,
        chunk_log2: int = DEFAULT_CHUNK_LOG2,
    ) -> None:
        if len(passphrase) < MIN_PASSPHRASE_CHARS:
            raise ValueError(f"the passphrase needs at least {MIN_PASSPHRASE_CHARS} characters")
        salt = os.urandom(SALT_SIZE)
        self._prefix = os.urandom(PREFIX_SIZE)
        self._header = (
            MAGIC + bytes([VERSION, log_n, r, p]) + salt + self._prefix + bytes([chunk_log2])
        )
        self._aead = AESGCM(derive_key(passphrase, salt, log_n, r, p))
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
            raise ValueError("write to a closed EncryptWriter")
        self._buffer += data
        while len(self._buffer) > self._chunk:
            self._seal(bytes(self._buffer[: self._chunk]), final=False)
            del self._buffer[: self._chunk]
        return len(data)

    def flush(self) -> None:
        """Flush what is sealed so far; the partial chunk is sealed when it is full or at close."""
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


class DecryptReader:
    """A read-only binary file that yields the plaintext of a sealed package, chunk by chunk."""

    def __init__(self, src: BinaryIO, passphrase: str) -> None:
        header = src.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE or not header.startswith(MAGIC):
            raise BundleDecryptionError("this is not a training package")
        version, log_n, r, p = header[8], header[9], header[10], header[11]
        if version != VERSION:
            raise BundleDecryptionError(f"unsupported package version {version}")
        if not 10 <= log_n <= MAX_LOG_N or r == 0 or p == 0:
            raise BundleDecryptionError("the package header holds invalid key derivation settings")
        salt_end = 12 + SALT_SIZE
        salt = header[12:salt_end]
        self._prefix = header[salt_end : salt_end + PREFIX_SIZE]
        self._header = header
        self._aead = AESGCM(derive_key(passphrase, salt, log_n, r, p))
        self._block = (1 << header[-1]) + TAG_SIZE
        self._src = src
        self._pending: bytes | None = src.read(self._block)
        self._index = 0
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def _next_chunk(self) -> bytes:
        pending = self._pending
        if pending is None:
            return b""
        following = self._src.read(self._block)
        final = not following
        try:
            plain = self._aead.decrypt(
                _nonce(self._prefix, self._index, final), pending, self._header
            )
        except InvalidTag:
            raise BundleDecryptionError("wrong passphrase, or the package is damaged") from None
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
        """Stop reading; the underlying file stays open, the caller owns it."""
        self._pending = None
        self._buffer = b""


def decrypt_stream(src: BinaryIO, dst: BinaryIO, passphrase: str) -> int:
    """Decrypt ``src`` into ``dst``; returns the number of plaintext bytes written."""
    reader = DecryptReader(src, passphrase)
    written = 0
    while block := reader.read(1 << 20):
        dst.write(block)
        written += len(block)
    return written
