"""The export's ``_integrity/`` folder (R-IMP-001).

What the folder contains could not be observed while this was written (there was no real
export in the development environment), so the reader accepts the shapes such manifests
usually have and says so when it meets none of them:

* a JSON document whose ``files`` / ``entries`` / ``items`` / ``manifest`` / ``checksums`` /
  ``hashes`` member (or the document itself) is either a list of objects or a mapping;
  an entry names a relative path (``path``, ``file``, ``name`` or ``relativePath``, or the
  mapping key) and any of ``sha256``, ``sha1``, ``md5``, ``hash``, ``digest``, ``checksum``
  and ``size`` / ``bytes`` / ``length``;
* a text file with ``<hex digest>  <relative path>`` lines (``sha256sum`` style).

``twin import inspect`` lists the file names and JSON keys of the real ``_integrity/``
folder, which is how the actual format is confirmed (docs/PENDING_USER_ACTIONS.md).

A file that is listed and does not match its size or digest is *failed*: the importer
reports it and does not import it.  A file the manifest does not list is *unlisted*, which
is not a failure.  An ``_integrity/`` folder in which no entry could be read is reported as
*unrecognized*; the import goes on, with a warning in the report.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import orjson

from twin.ingest.paths import long_path, nfc

INTEGRITY_DIR = "_integrity"
_LIST_MEMBERS = ("files", "entries", "items", "manifest", "checksums", "hashes")
_PATH_KEYS = ("path", "file", "name", "relativePath", "filename")
_SIZE_KEYS = ("size", "bytes", "length")
_DIGEST_KEYS = ("sha256", "sha1", "md5", "hash", "digest", "checksum")
_HEX = re.compile(r"^[0-9a-fA-F]+$")
_LINE = re.compile(r"^([0-9a-fA-F]{32,64})\s+\*?(.+?)\s*$")
_ALGORITHMS = {32: "md5", 40: "sha1", 64: "sha256"}
MAX_MANIFEST_BYTES = 256 * 1024 * 1024
HASH_BLOCK = 1024 * 1024


class IntegrityState(StrEnum):
    ABSENT = "absent"  # no _integrity/ folder
    READ = "read"  # entries were found
    UNRECOGNIZED = "unrecognized"  # the folder exists but no entry could be read


class FileVerdict(StrEnum):
    OK = "ok"
    FAILED = "failed"
    UNLISTED = "unlisted"


@dataclass(frozen=True)
class IntegrityEntry:
    path: str
    algorithm: str | None = None
    digest: str | None = None
    size: int | None = None


@dataclass
class IntegrityIndex:
    state: IntegrityState
    entries: dict[str, IntegrityEntry] = field(default_factory=dict)
    files_read: int = 0
    problems: list[str] = field(default_factory=list)

    def lookup(self, relative: str) -> IntegrityEntry | None:
        return self.entries.get(normalize_key(relative))


@dataclass(frozen=True)
class Verification:
    verdict: FileVerdict
    reason: str = ""


def normalize_key(relative: str) -> str:
    text = nfc(relative).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def digest_algorithm(digest: str) -> str | None:
    """``md5`` / ``sha1`` / ``sha256`` from the length of a hex digest."""
    if not _HEX.match(digest):
        return None
    return _ALGORITHMS.get(len(digest))


def _entry_from_object(item: Mapping[str, Any], key: str | None = None) -> IntegrityEntry | None:
    path = key
    if path is None:
        for name in _PATH_KEYS:
            value = item.get(name)
            if isinstance(value, str) and value:
                path = value
                break
    if path is None:
        return None
    algorithm: str | None = None
    digest: str | None = None
    for name in _DIGEST_KEYS:
        value = item.get(name)
        if not isinstance(value, str) or not value:
            continue
        if name in ("sha256", "sha1", "md5"):
            algorithm, digest = name, value.lower()
        else:
            algorithm, digest = digest_algorithm(value), value.lower()
        break
    size: int | None = None
    for name in _SIZE_KEYS:
        value = item.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            size = value
            break
    if digest is None and size is None:
        return None
    return IntegrityEntry(normalize_key(path), algorithm, digest, size)


def _entry_from_value(key: str, value: Any) -> IntegrityEntry | None:
    if isinstance(value, str):
        algorithm = digest_algorithm(value)
        if algorithm is None:
            return None
        return IntegrityEntry(normalize_key(key), algorithm, value.lower(), None)
    if isinstance(value, Mapping):
        return _entry_from_object(value, key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return IntegrityEntry(normalize_key(key), None, None, value)
    return None


def _entries_of(document: Any) -> Iterator[IntegrityEntry]:
    if isinstance(document, list):
        for item in document:
            if isinstance(item, Mapping):
                entry = _entry_from_object(item)
                if entry is not None:
                    yield entry
        return
    if not isinstance(document, Mapping):
        return
    for member in _LIST_MEMBERS:
        inner = document.get(member)
        if isinstance(inner, list | dict):
            yield from _entries_of(inner)
            return
    for key, value in document.items():
        if isinstance(key, str):
            entry = _entry_from_value(key, value)
            if entry is not None:
                yield entry


def parse_manifest_text(name: str, data: bytes) -> list[IntegrityEntry]:
    """Entries of one manifest file (JSON or ``sha256sum`` text)."""
    if name.lower().endswith(".json"):
        return list(_entries_of(orjson.loads(data)))
    entries: list[IntegrityEntry] = []
    for line in data.decode("utf-8-sig", errors="replace").splitlines():
        match = _LINE.match(line.strip())
        if match is None:
            continue
        digest, path = match.group(1).lower(), match.group(2)
        algorithm = digest_algorithm(digest)
        if algorithm is not None:
            entries.append(IntegrityEntry(normalize_key(path), algorithm, digest, None))
    return entries


def load_integrity(root: Path) -> IntegrityIndex:
    """Read every manifest in ``<root>/_integrity``."""
    folder = root / INTEGRITY_DIR
    if not long_path(folder).is_dir():
        return IntegrityIndex(IntegrityState.ABSENT)
    index = IntegrityIndex(IntegrityState.UNRECOGNIZED)
    for path in sorted(p for p in folder.rglob("*") if long_path(p).is_file()):
        try:
            if path.stat().st_size > MAX_MANIFEST_BYTES:
                index.problems.append("a manifest file is too large to read")
                continue
            entries = parse_manifest_text(path.name, long_path(path).read_bytes())
        except (OSError, ValueError):
            index.problems.append("a file in _integrity could not be parsed")
            continue
        index.files_read += 1
        for entry in entries:
            index.entries[entry.path] = entry
    if index.entries:
        index.state = IntegrityState.READ
    return index


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with long_path(path).open("rb") as handle:
        while block := handle.read(HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def verify_file(index: IntegrityIndex, relative: str, path: Path) -> Verification:
    """Check ``path`` against the manifest entry for ``relative`` (size, then digest)."""
    entry = index.lookup(relative)
    if entry is None:
        return Verification(FileVerdict.UNLISTED)
    try:
        if entry.size is not None and long_path(path).stat().st_size != entry.size:
            return Verification(FileVerdict.FAILED, "size differs from the manifest")
        if entry.digest and entry.algorithm and hash_file(path, entry.algorithm) != entry.digest:
            return Verification(FileVerdict.FAILED, f"{entry.algorithm} differs")
    except OSError:
        return Verification(FileVerdict.FAILED, "file could not be read")
    return Verification(FileVerdict.OK)


class StreamHasher:
    """Wraps a binary stream and hashes what is read (one pass for import and verification)."""

    def __init__(self, stream: Any, algorithms: tuple[str, ...]) -> None:
        self._stream = stream
        self._hashes = {name: hashlib.new(name) for name in algorithms}
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        block: bytes = self._stream.read(size)
        self.size += len(block)
        for digest in self._hashes.values():
            digest.update(block)
        return block

    def hexdigest(self, algorithm: str) -> str:
        return self._hashes[algorithm].hexdigest()


def check_stream(entry: IntegrityEntry | None, hasher: StreamHasher) -> Verification:
    """Compare a fully read stream with its manifest entry."""
    if entry is None:
        return Verification(FileVerdict.UNLISTED)
    if entry.size is not None and hasher.size != entry.size:
        return Verification(FileVerdict.FAILED, "size differs from the manifest")
    if entry.digest and entry.algorithm and hasher.hexdigest(entry.algorithm) != entry.digest:
        return Verification(FileVerdict.FAILED, f"{entry.algorithm} differs")
    return Verification(FileVerdict.OK)


def stream_algorithms(entry: IntegrityEntry | None, *extra: str) -> tuple[str, ...]:
    """The hash algorithms a stream has to compute for ``entry`` (plus ``extra``)."""
    names = list(extra)
    if entry is not None and entry.algorithm and entry.algorithm not in names:
        names.append(entry.algorithm)
    return tuple(names)
