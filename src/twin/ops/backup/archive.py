"""The archive inside a backup: writing, reading and checking it (R-OPS-006).

Content of one archive (a tar stream, zstd-compressed, sealed by :mod:`twin.ops.backup.sealed`):

``manifest.json``
    first, so a reader learns what is inside after the first chunk: the format, when and in which
    local day it was made, the schema revision of the database, **every key id the archive
    depends on** (``key_ids``: the key it is sealed with, the keys its encrypted database values
    carry, the keys of the media files it lists), the row count of each table, the list of media
    files (``sha256``, size, key id) and the SHA-256 of the database snapshot;
``twin.db``
    a consistent snapshot of the database, made with SQLite's online backup API while the
    application keeps running (pages are copied in small steps, writers are not blocked);
``vectors/...``
    the files of the vector index, copied as they are.  They are derived data and are copied
    without stopping the writers, so they may be a moment apart from the database; after a restore
    the integrity check says so and names the repair command (R-OPS-003).

The media files themselves (``data/media/<sha256>.enc``, already encrypted and named by their
hash) are not put into the archive: they are copied once into a **media pool** next to the
backups, and the manifest says which ones belong to the backup.  A restore takes the files it
lacks from the pool.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

import zstandard

from twin.ops.backup.sealed import (
    SealReader,
    SealWriter,
)
from twin.storage.crypto import KeyRing
from twin.storage.media import MediaStore
from twin.storage.rotate import encrypted_tables

FORMAT = 1
MANIFEST_NAME = "manifest.json"
DB_NAME = "twin.db"
VECTORS_PREFIX = "vectors/"
ZSTD_LEVEL = 3
COPY_BLOCK = 1 << 20
BACKUP_PAGES = 1000  # database pages copied per step of the online backup
BACKUP_SLEEP_S = 0.01  # pause between steps, so writers of the application get their turn


class ArchiveError(Exception):
    """An archive does not hold what a backup holds, or does not match its manifest."""


@dataclass(frozen=True)
class MediaEntry:
    """One media file the backup belongs to."""

    sha256: str
    size: int
    key_id: int

    def to_json(self) -> list[Any]:
        return [self.sha256, self.size, self.key_id]

    @classmethod
    def from_json(cls, data: list[Any]) -> MediaEntry:
        return cls(str(data[0]), int(data[1]), int(data[2]))


@dataclass
class Manifest:
    """``manifest.json``."""

    created_at: str
    local_date: str
    kind: str
    schema_revision: str | None
    key_id: int
    key_ids: list[int]
    tables: dict[str, int]
    media: list[MediaEntry]
    db_sha256: str
    vector_files: int = 0
    app_version: str = ""
    format: int = FORMAT

    @property
    def row_count(self) -> int:
        return sum(self.tables.values())

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "format": self.format,
                "created_at": self.created_at,
                "local_date": self.local_date,
                "kind": self.kind,
                "schema_revision": self.schema_revision,
                "app_version": self.app_version,
                "key_id": self.key_id,
                "key_ids": sorted(self.key_ids),
                "tables": self.tables,
                "media": [entry.to_json() for entry in self.media],
                "db_sha256": self.db_sha256,
                "vector_files": self.vector_files,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, blob: bytes) -> Manifest:
        try:
            data = json.loads(blob)
            if int(data["format"]) != FORMAT:
                raise ArchiveError(f"unsupported archive format {data['format']}")
            return cls(
                created_at=str(data["created_at"]),
                local_date=str(data["local_date"]),
                kind=str(data["kind"]),
                schema_revision=data.get("schema_revision"),
                key_id=int(data["key_id"]),
                key_ids=[int(item) for item in data["key_ids"]],
                tables={str(k): int(v) for k, v in data["tables"].items()},
                media=[MediaEntry.from_json(item) for item in data["media"]],
                db_sha256=str(data["db_sha256"]),
                vector_files=int(data.get("vector_files", 0)),
                app_version=str(data.get("app_version", "")),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ArchiveError(f"the archive manifest is unreadable: {exc}") from None


@dataclass(frozen=True)
class ArchiveSource:
    """What a backup is made from."""

    db_path: Path
    vectors_dir: Path
    media: MediaStore
    pool_dir: Path
    tmp_dir: Path


# ------------------------------------------------------------------ the database


def snapshot_database(db_path: Path, dest: Path) -> None:
    """A consistent copy of the live database (SQLite online backup API, small steps)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(str(db_path), timeout=30)
    target = sqlite3.connect(str(dest))
    try:
        source.backup(target, pages=BACKUP_PAGES, sleep=BACKUP_SLEEP_S)
    finally:
        target.close()
        source.close()


def _tables_of(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(name) for (name,) in rows}


def _key_ids_in(connection: sqlite3.Connection, present: set[str]) -> set[int]:
    """The key ids that the encrypted values of the database carry (a SQL scan, no decryption)."""
    key_ids: set[int] = set()
    for spec in encrypted_tables():
        name = str(spec.table.name)
        if name not in present:
            continue
        for column in spec.columns:
            found = connection.execute(
                f'SELECT DISTINCT substr("{column.name}", 2, 4) FROM "{name}" '
                f'WHERE "{column.name}" IS NOT NULL'
            ).fetchall()
            key_ids.update(int.from_bytes(bytes(blob), "big") for (blob,) in found if blob)
    return key_ids


def database_key_ids(path: Path) -> set[int]:
    """The key ids used by the encrypted values of a database file."""
    connection = sqlite3.connect(str(path), timeout=30)
    try:
        return _key_ids_in(connection, _tables_of(connection))
    finally:
        connection.close()


def database_facts(snapshot: Path) -> tuple[dict[str, int], set[int], str | None]:
    """``(rows per table, key ids of the encrypted values, schema revision)`` of a snapshot."""
    connection = sqlite3.connect(str(snapshot))
    try:
        present = _tables_of(connection)
        counts = {
            name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in sorted(present - {"alembic_version", "sqlite_sequence"})
        }
        revision = None
        if "alembic_version" in present:
            row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
            revision = str(row[0]) if row else None
        return counts, _key_ids_in(connection, present), revision
    finally:
        connection.close()


def check_database(path: Path) -> list[str]:
    """``PRAGMA integrity_check`` of a database file: the problems, empty when it is clean."""
    connection = sqlite3.connect(str(path))
    try:
        rows = [str(row[0]) for row in connection.execute("PRAGMA integrity_check").fetchall()]
    except sqlite3.DatabaseError as exc:  # too damaged for SQLite to run the check at all
        return [f"SQLite cannot read the database: {exc}"]
    finally:
        connection.close()
    return [] if rows == ["ok"] else rows[:20]


# ---------------------------------------------------------------------- the media


def scan_media(media: MediaStore) -> list[MediaEntry]:
    """Every media file with its size and the key it is sealed with."""
    entries: list[MediaEntry] = []
    for sha in media.iter_hashes():
        path = media.path_for(sha)
        try:
            entries.append(MediaEntry(sha, path.stat().st_size, media.key_id_of(sha)))
        except OSError:  # deleted while the backup runs
            continue
    return entries


def sync_pool(media: MediaStore, pool_dir: Path, entries: list[MediaEntry]) -> int:
    """Copy the media files the pool lacks (or holds under another key); returns how many."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool = MediaStore(pool_dir, media.tmp_dir)
    copied = 0
    for entry in entries:
        target = pool.path_for(entry.sha256)
        if target.is_file():
            try:
                if pool.key_id_of(entry.sha256) == entry.key_id:
                    continue
            except (OSError, ValueError):
                pass  # a damaged pool copy is replaced
        part = pool_dir / f".{entry.sha256}.part"
        try:
            shutil.copyfile(media.path_for(entry.sha256), part)
            os.replace(part, target)
        except OSError:
            part.unlink(missing_ok=True)
            continue
        copied += 1
    return copied


# ---------------------------------------------------------------------- writing


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(COPY_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    """SHA-256 of a file."""
    return _file_sha256(path)


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mtime: float) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(mtime)
    info.mode = 0o600
    tar.addfile(info, io.BytesIO(data))


def vector_files(vectors_dir: Path) -> list[Path]:
    if not vectors_dir.is_dir():
        return []
    return sorted(path for path in vectors_dir.rglob("*") if path.is_file())


@dataclass(frozen=True)
class WrittenArchive:
    """The result of :func:`write_archive`."""

    manifest: Manifest
    size_bytes: int
    sha256: str


def write_archive(
    dest: Path,
    ring: KeyRing,
    source: ArchiveSource,
    *,
    created_at: datetime,
    local_date: str,
    kind: str,
    app_version: str,
) -> WrittenArchive:
    """Make one backup archive at ``dest`` (written under a temporary name, then moved)."""
    work = source.tmp_dir / f"backup-{os.urandom(4).hex()}"
    work.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        snapshot = work / DB_NAME
        snapshot_database(source.db_path, snapshot)
        counts, db_key_ids, revision = database_facts(snapshot)
        entries = scan_media(source.media)
        sync_pool(source.media, source.pool_dir, entries)
        files = vector_files(source.vectors_dir)
        manifest = Manifest(
            created_at=created_at.isoformat(),
            local_date=local_date,
            kind=kind,
            schema_revision=revision,
            key_id=ring.current_id,
            key_ids=sorted({ring.current_id, *db_key_ids, *(e.key_id for e in entries)}),
            tables=counts,
            media=entries,
            db_sha256=_file_sha256(snapshot),
            vector_files=len(files),
            app_version=app_version,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        with part.open("wb") as raw:
            sealer = SealWriter(raw, ring, key_id=ring.current_id)
            compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
            with (
                compressor.stream_writer(cast(BinaryIO, sealer), closefd=False) as compressed,
                tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar,
            ):
                _add_bytes(tar, MANIFEST_NAME, manifest.to_bytes(), created_at.timestamp())
                tar.add(snapshot, arcname=DB_NAME)
                for path in files:
                    relative = path.relative_to(source.vectors_dir).as_posix()
                    with contextlib.suppress(FileNotFoundError):  # the index moved on
                        tar.add(path, arcname=VECTORS_PREFIX + relative, recursive=False)
            sealer.close()
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(part, dest)
        return WrittenArchive(manifest, dest.stat().st_size, _file_sha256(dest))
    finally:
        part.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------- reading


@contextlib.contextmanager
def _open_tar(path: Path, ring: KeyRing, *, full: bool) -> Iterator[tarfile.TarFile]:
    """The tar stream of a backup.  With ``full`` the whole sealed stream is read to its last
    chunk when the block ends, which authenticates every byte (and notices a cut-off tail)."""
    with path.open("rb") as raw:
        reader = SealReader(raw, ring)
        try:
            with zstandard.ZstdDecompressor().stream_reader(
                cast(BinaryIO, reader), closefd=False
            ) as plain:
                with tarfile.open(fileobj=plain, mode="r|") as tar:
                    yield tar
                if full:
                    while plain.read(COPY_BLOCK):
                        pass
            if full:
                while reader.read(COPY_BLOCK):
                    pass
        finally:
            reader.close()


def _manifest_of(tar: tarfile.TarFile, members: Iterator[tarfile.TarInfo]) -> Manifest:
    first = next(members, None)
    if first is not None and first.name == MANIFEST_NAME:
        stream = tar.extractfile(first)
        if stream is not None:
            return Manifest.from_bytes(stream.read())
    raise ArchiveError("the archive does not start with its manifest")


def read_manifest(path: Path, ring: KeyRing) -> Manifest:
    """The manifest of a backup (only the first chunk is decrypted)."""
    with _open_tar(path, ring, full=False) as tar:
        return _manifest_of(tar, iter(tar))


def _safe_target(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArchiveError(f"the archive holds a path that leaves its folder: {name!r}")
    return root.joinpath(*relative.parts)


@dataclass
class Extracted:
    """What :func:`extract_archive` found."""

    manifest: Manifest
    database: Path | None = None
    vectors_dir: Path | None = None
    vector_files: int = 0
    database_sha256: str = ""
    problems: list[str] = field(default_factory=list)


def extract_archive(path: Path, ring: KeyRing, dest: Path) -> Extracted:
    """Decrypt the whole archive into ``dest`` (``twin.db`` and ``vectors/``), checking it.

    The stream is read to its last chunk, so every byte is authenticated; the SHA-256 of the
    extracted database must be the one in the manifest.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with _open_tar(path, ring, full=True) as tar:
        members = iter(tar)
        result = Extracted(_manifest_of(tar, members))
        for member in members:
            if not member.isfile():
                continue
            if member.name == DB_NAME:
                target = dest / DB_NAME
            elif member.name.startswith(VECTORS_PREFIX):
                target = _safe_target(dest, member.name)
                result.vector_files += 1
            else:
                result.problems.append(f"unexpected entry {member.name!r}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                continue
            with target.open("wb") as out:
                shutil.copyfileobj(stream, out, COPY_BLOCK)
            if member.name == DB_NAME:
                result.database = target
    if result.database is None:
        raise ArchiveError("the archive holds no database")
    result.database_sha256 = _file_sha256(result.database)
    if result.database_sha256 != result.manifest.db_sha256:
        raise ArchiveError("the database in the archive does not match its manifest")
    if (dest / "vectors").is_dir():
        result.vectors_dir = dest / "vectors"
    return result
