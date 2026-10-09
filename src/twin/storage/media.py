"""Encrypted media store (R-STO-004).

Files live in ``<media_dir>/<sha256>.enc`` where the name is the SHA-256 of the
*plaintext*.  A file is a header followed by independently sealed chunks::

    header (62 bytes)
      magic "TWM1" | version | key_id(4) | chunk_size(4) | nonce_prefix(8)
      | plaintext_len(8) | plaintext_sha256(32) | kind(1)
    chunk i (i = 0..n-1):  AES-256-GCM(chunk_plaintext) + 16-byte tag

* chunk nonce = ``nonce_prefix || i`` (unique per chunk, per file);
* chunk associated data = fixed header fields || chunk index || final flag; the
  *final* chunk additionally authenticates ``plaintext_len`` and the plaintext
  SHA-256 written in the header, so truncation, reordering, splicing chunks
  from another file and editing the header are all detected;
* writing and reading are streaming: memory use is one chunk, independent of
  file size.  The header's length/hash are filled in after the stream ends.

Decrypted bytes are only ever exposed as a stream or, for tools that need a
path, through :meth:`MediaStore.temp_file`, which wipes and removes the file.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import struct
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from twin.storage.crypto import DecryptionError, KeyRing, get_keyring

MAGIC = b"TWM1"
FORMAT_VERSION = 1
DEFAULT_CHUNK_SIZE = 1024 * 1024
TAG_BYTES = 16
_HEADER = struct.Struct(">4sBII8sQ32sB")
HEADER_BYTES = _HEADER.size
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class MediaKind(StrEnum):
    IMAGE = "image"
    STICKER = "sticker"
    AVATAR = "avatar"
    VOICE = "voice"
    VIDEO = "video"
    FILE = "file"
    OTHER = "other"


_KIND_CODES = {kind: index for index, kind in enumerate(MediaKind)}
_CODE_KINDS = {code: kind for kind, code in _KIND_CODES.items()}


class MediaError(Exception):
    """Base class for media store errors."""


class MediaNotFoundError(MediaError):
    """No file with the requested hash is stored."""


class MediaIntegrityError(MediaError):
    """The stored file failed authentication or its hash does not match."""


@dataclass(frozen=True)
class StoredMedia:
    sha256: str
    size: int
    kind: MediaKind
    created: bool  # False when identical content was already stored


@dataclass(frozen=True)
class _Header:
    key_id: int
    chunk_size: int
    nonce_prefix: bytes
    plain_len: int
    sha256: bytes
    kind: MediaKind

    def pack(self) -> bytes:
        return _HEADER.pack(
            MAGIC,
            FORMAT_VERSION,
            self.key_id,
            self.chunk_size,
            self.nonce_prefix,
            self.plain_len,
            self.sha256,
            _KIND_CODES[self.kind],
        )

    @classmethod
    def unpack(cls, data: bytes) -> _Header:
        if len(data) < HEADER_BYTES:
            raise MediaIntegrityError("media file is truncated (incomplete header)")
        magic, version, key_id, chunk_size, prefix, plain_len, sha, kind_code = _HEADER.unpack(
            data[:HEADER_BYTES]
        )
        if magic != MAGIC or version != FORMAT_VERSION:
            raise MediaIntegrityError("not a supported encrypted media file")
        if chunk_size < 1024 or kind_code not in _CODE_KINDS:
            raise MediaIntegrityError("media header is corrupted")
        return cls(key_id, chunk_size, prefix, plain_len, sha, _CODE_KINDS[kind_code])

    def fixed_aad(self) -> bytes:
        return struct.pack(
            ">4sBII8sB",
            MAGIC,
            FORMAT_VERSION,
            self.key_id,
            self.chunk_size,
            self.nonce_prefix,
            _KIND_CODES[self.kind],
        )

    def chunk_count(self) -> int:
        return max(1, -(-self.plain_len // self.chunk_size))

    def chunk_aad(self, index: int, *, final: bool) -> bytes:
        aad = self.fixed_aad() + struct.pack(">QB", index, 1 if final else 0)
        if final:
            aad += struct.pack(">Q", self.plain_len) + self.sha256
        return aad

    def nonce(self, index: int) -> bytes:
        return self.nonce_prefix + struct.pack(">I", index)


def _pieces(data: bytes | BinaryIO | Iterable[bytes], read_size: int) -> Iterator[bytes]:
    if isinstance(data, bytes | bytearray | memoryview):
        yield bytes(data)
    elif hasattr(data, "read"):
        reader = cast(BinaryIO, data)
        while block := reader.read(read_size):
            yield bytes(block)
    else:
        for item in data:
            if item:
                yield bytes(item)


def _check_sha(sha256: str) -> str:
    if not _SHA_RE.fullmatch(sha256):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    return sha256


class MediaReader:
    """Streaming decrypting reader.  Verifies the plaintext hash at end of stream."""

    def __init__(self, path: Path, expected_sha256: str, keyring: KeyRing) -> None:
        self._file: BinaryIO | None = path.open("rb")
        self._path = path
        self._expected = expected_sha256
        try:
            self._header = _Header.unpack(self._file.read(HEADER_BYTES))
            if self._header.sha256.hex() != expected_sha256:
                raise MediaIntegrityError("media header hash does not match the file name")
            self._cipher = AESGCM(keyring.key_bytes(self._header.key_id))
        except BaseException:
            self._file.close()
            self._file = None
            raise
        self._next_chunk = 0
        self._buffer = b""
        self._hash = hashlib.sha256()
        self._produced = 0
        self._done = False

    @property
    def size(self) -> int:
        return self._header.plain_len

    @property
    def kind(self) -> MediaKind:
        return self._header.kind

    @property
    def key_id(self) -> int:
        return self._header.key_id

    def _decrypt_next(self) -> bytes:
        file = self._file
        if file is None:
            raise ValueError("I/O operation on a closed media reader")
        header = self._header
        index = self._next_chunk
        total = header.chunk_count()
        final = index == total - 1
        if final:
            plain_in_chunk = header.plain_len - index * header.chunk_size
        else:
            plain_in_chunk = header.chunk_size
        sealed = file.read(plain_in_chunk + TAG_BYTES)
        if len(sealed) != plain_in_chunk + TAG_BYTES:
            raise MediaIntegrityError("media file is truncated")
        try:
            chunk = self._cipher.decrypt(
                header.nonce(index), sealed, header.chunk_aad(index, final=final)
            )
        except InvalidTag:
            raise MediaIntegrityError(
                f"media chunk {index} failed authentication (corrupted or tampered)"
            ) from None
        self._next_chunk += 1
        self._hash.update(chunk)
        self._produced += len(chunk)
        if final:
            if file.read(1):
                raise MediaIntegrityError("unexpected trailing data after the final chunk")
            if self._hash.hexdigest() != self._expected:
                raise MediaIntegrityError("decrypted content does not match its SHA-256")
            self._done = True
        return chunk

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` plaintext bytes (all remaining if negative)."""
        out = bytearray()
        want = size if size >= 0 else None
        while want is None or len(out) < want:
            if self._buffer:
                take = self._buffer if want is None else self._buffer[: want - len(out)]
                out += take
                self._buffer = self._buffer[len(take) :]
                continue
            if self._done:
                break
            self._buffer = self._decrypt_next()
        return bytes(out)

    def chunks(self) -> Iterator[bytes]:
        """Iterate over plaintext chunks until the end of the file."""
        while True:
            block = self.read(self._header.chunk_size)
            if not block:
                return
            yield block

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class MediaStore:
    """Content-addressed encrypted file store."""

    def __init__(
        self,
        root: Path,
        tmp_dir: Path,
        *,
        keyring_provider: Callable[[], KeyRing] = get_keyring,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        if chunk_size < 1024:
            raise ValueError("chunk_size must be at least 1024 bytes")
        self.root = root
        self.tmp_dir = tmp_dir
        self._keyring_provider = keyring_provider
        self.chunk_size = chunk_size

    # ----------------------------------------------------------------- paths

    def path_for(self, sha256: str) -> Path:
        return self.root / f"{_check_sha(sha256)}.enc"

    def exists(self, sha256: str) -> bool:
        return self.path_for(sha256).is_file()

    def iter_hashes(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for entry in sorted(self.root.glob("*.enc")):
            if _SHA_RE.fullmatch(entry.stem):
                yield entry.stem

    def _ensure_dirs(self) -> None:
        for directory in (self.root, self.tmp_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            self.tmp_dir.chmod(0o700)

    # ------------------------------------------------------------------ put

    def _write_part(
        self, data: bytes | BinaryIO | Iterable[bytes], kind: MediaKind
    ) -> tuple[Path, str, int]:
        """Encrypt ``data`` into a temporary part file; returns (path, sha256, size)."""
        self._ensure_dirs()
        ring = self._keyring_provider()
        key_id = ring.current_id
        nonce_prefix = secrets.token_bytes(8)
        draft = _Header(key_id, self.chunk_size, nonce_prefix, 0, b"\x00" * 32, kind)
        cipher = AESGCM(ring.key_bytes(key_id))
        part = self.root / f".put-{secrets.token_hex(8)}.part"
        digest = hashlib.sha256()
        total = 0
        try:
            with part.open("wb") as out:
                out.write(draft.pack())
                pending = bytearray()
                index = 0
                for piece in _pieces(data, self.chunk_size):
                    pending += piece
                    while len(pending) > self.chunk_size:
                        block = bytes(pending[: self.chunk_size])
                        del pending[: self.chunk_size]
                        digest.update(block)
                        total += len(block)
                        out.write(
                            cipher.encrypt(
                                draft.nonce(index), block, draft.chunk_aad(index, final=False)
                            )
                        )
                        index += 1
                final_block = bytes(pending)
                digest.update(final_block)
                total += len(final_block)
                final = _Header(key_id, self.chunk_size, nonce_prefix, total, digest.digest(), kind)
                out.write(
                    cipher.encrypt(
                        final.nonce(index), final_block, final.chunk_aad(index, final=True)
                    )
                )
                out.seek(0)
                out.write(final.pack())
                out.flush()
                os.fsync(out.fileno())
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        return part, digest.hexdigest(), total

    def put(self, data: bytes | BinaryIO | Iterable[bytes], kind: MediaKind) -> StoredMedia:
        """Encrypt ``data`` (bytes, binary stream or iterable of bytes) and store it."""
        part, sha_hex, total = self._write_part(data, kind)
        target = self.path_for(sha_hex)
        created = not target.exists()
        if created:
            os.replace(part, target)
        else:
            part.unlink()
        return StoredMedia(sha_hex, total, kind, created)

    # ----------------------------------------------------------------- read

    def open(self, sha256: str) -> MediaReader:
        """Open ``sha256`` for streaming, decrypted reads."""
        path = self.path_for(sha256)
        if not path.is_file():
            raise MediaNotFoundError(f"no stored media with sha256 {sha256}")
        return MediaReader(path, sha256, self._keyring_provider())

    def read_bytes(self, sha256: str) -> bytes:
        """Read a whole (small) file into memory."""
        with self.open(sha256) as reader:
            return reader.read()

    def verify(self, sha256: str) -> bool:
        """True if the file exists, authenticates and matches its SHA-256."""
        try:
            with self.open(sha256) as reader:
                for _ in reader.chunks():
                    pass
        except (MediaError, DecryptionError, OSError):
            return False
        return True

    @contextmanager
    def temp_file(self, sha256: str, suffix: str = "") -> Iterator[Path]:
        """Decrypt to a controlled temporary file; overwrite and delete it on exit."""
        self._ensure_dirs()
        path = self.tmp_dir / f"{secrets.token_hex(12)}{suffix}"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        written = 0
        try:
            with os.fdopen(fd, "wb") as out, self.open(sha256) as reader:
                for block in reader.chunks():
                    out.write(block)
                    written += len(block)
            yield path
        finally:
            _wipe(path, written)

    def delete(self, sha256: str) -> bool:
        path = self.path_for(sha256)
        if not path.exists():
            return False
        path.unlink()
        return True

    # -------------------------------------------------------------- rekeying

    def key_id_of(self, sha256: str) -> int:
        with self.path_for(sha256).open("rb") as handle:
            return _Header.unpack(handle.read(HEADER_BYTES)).key_id

    def rekey(self, sha256: str) -> bool:
        """Re-encrypt with the current key.  Returns ``False`` if already current."""
        ring = self._keyring_provider()
        if self.key_id_of(sha256) == ring.current_id:
            return False
        with self.open(sha256) as reader:
            part, new_sha, _ = self._write_part(reader.chunks(), reader.kind)
        if new_sha != sha256:
            part.unlink(missing_ok=True)
            raise MediaIntegrityError("re-encryption changed the content hash")
        os.replace(part, self.path_for(sha256))
        return True

    def rekey_all(self, on_progress: Callable[[int, int], None] | None = None) -> int:
        """Re-encrypt every file that is not on the current key; returns the count."""
        hashes = list(self.iter_hashes())
        changed = 0
        for position, sha in enumerate(hashes, start=1):
            if self.rekey(sha):
                changed += 1
            if on_progress:
                on_progress(position, len(hashes))
        return changed


def _wipe(path: Path, size: int) -> None:
    """Overwrite ``path`` with zeros, flush to disk and delete it."""
    try:
        if path.exists():
            with path.open("r+b") as handle:
                remaining = max(size, handle.seek(0, os.SEEK_END))
                handle.seek(0)
                block = b"\x00" * 65536
                while remaining > 0:
                    handle.write(block[: min(len(block), remaining)])
                    remaining -= len(block)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        path.unlink(missing_ok=True)
