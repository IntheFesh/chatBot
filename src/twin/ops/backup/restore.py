"""Restoring a backup (``twin backup restore``, R-OPS-006, R-STO-003).

An EXCLUSIVE operation (the application must be stopped).  In this order, each step able to stop
the whole thing before anything of the current data has been touched:

1. the archive is read completely into a work folder under ``data/tmp``: every byte is
   authenticated, the database matches the hash in the manifest and passes SQLite's own check;
2. every key the backup depends on (``manifest.key_ids``) must still be in the key ring - a
   backup made before a key rotation needs the retired key, which is kept as long as that backup
   is kept (R-STO-003) - and the schema must not be newer than this program;
3. the current data is backed up first, as ``pre-restore-<UTC time>.bak.enc`` (the same format,
   sealed with the current key) so that a restore never destroys the only copy of anything;
4. the current database files and vector index are moved aside, the restored ones put in place,
   the media files the backup lists and the data folder lacks are copied from the media pool;
   if anything fails here the moved-aside files are put back;
5. the migrations run (a backup of an older schema is brought up to date), the restored database
   is checked again and a sample of its encrypted values is decrypted with the key ring.

Afterwards the vector index may be a little behind or ahead of the database (it is copied while
the application runs); ``twin health`` reports that with the command that repairs it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from twin import __version__
from twin.clock import Clock
from twin.config.loader import DataPaths
from twin.ops.backup.archive import (
    ArchiveError,
    ArchiveSource,
    Manifest,
    check_database,
    extract_archive,
    read_manifest,
    write_archive,
)
from twin.ops.backup.layout import (
    POOL_DIRNAME,
    archive_name,
    sidecar_name,
    write_sidecar,
)
from twin.storage import migrate
from twin.storage.crypto import KeyRing, build_aad
from twin.storage.db import Database
from twin.storage.media import MediaStore
from twin.storage.migrate import SchemaState
from twin.storage.rotate import encrypted_tables

SAMPLE_VALUES = 40


class RestoreError(Exception):
    """The restore was refused or failed; what is on disk is as it was before."""


@dataclass
class RestoreReport:
    archive: str
    created_at: str
    key_id: int
    rows: int
    schema_before: str | None
    schema_after: str | None
    pre_restore: str | None
    media_listed: int
    media_restored: int
    media_missing: int
    vector_files: int
    sampled_values: int
    problems: list[str] = field(default_factory=list)


def _moved(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.name}.replaced-{tag}")


def database_files(db_path: Path) -> list[Path]:
    """The database file and its write-ahead log files, those that exist."""
    return [
        path for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")) if path.exists()
    ]


def sample_decrypt(db_path: Path, ring: KeyRing, count: int = SAMPLE_VALUES) -> int:
    """Decrypt up to ``count`` encrypted values of a database with ``ring``; returns how many.

    Raises the decryption error of the first value that does not open: a restored database whose
    values cannot be read with the available keys is worse than no restore.
    """
    db = Database(db_path)
    chosen = 0
    try:
        specs = encrypted_tables()
        with db.session() as session:
            for spec in specs:
                if chosen >= count:
                    break
                stmt = select(*spec.pk, *spec.columns).limit(max(1, count // max(len(specs), 1)))
                for row in session.execute(stmt).all():
                    pk_values = tuple(row[: len(spec.pk)])
                    for offset, column in enumerate(spec.columns):
                        blob = row[len(spec.pk) + offset]
                        if blob is None:
                            continue
                        ring.open(blob, build_aad(str(spec.table.name), pk_values, column.name))
                        chosen += 1
    finally:
        db.dispose()
    return chosen


def make_pre_restore(
    paths: DataPaths, ring: KeyRing, clock: Clock, media: MediaStore, zone_date: str
) -> Path:
    """Back up what is there now, before a restore replaces it."""
    now = clock.now_utc()
    name = archive_name("pre_restore", zone_date, now)
    dest = paths.backups_dir / name
    source = ArchiveSource(
        db_path=paths.db_path,
        vectors_dir=paths.vectors_dir,
        media=media,
        pool_dir=paths.backups_dir / POOL_DIRNAME,
        tmp_dir=paths.tmp_dir,
    )
    try:
        written = write_archive(
            dest,
            ring,
            source,
            created_at=now,
            local_date=zone_date,
            kind="pre_restore",
            app_version=__version__,
        )
        write_sidecar(paths.backups_dir / sidecar_name(name), written.manifest)
    except Exception as exc:
        raise RestoreError(
            f"the current data cannot be backed up first ({type(exc).__name__}); nothing was "
            "changed.  If the current database is damaged, move it away by hand and restore again"
        ) from None
    return dest


def check_requirements(manifest: Manifest, ring: KeyRing) -> None:
    """Refuse a backup whose keys are gone or whose schema is newer than this program."""
    missing = sorted(set(manifest.key_ids) - set(ring.key_ids))
    if missing:
        raise RestoreError(
            "the backup needs encryption key(s) "
            + ", ".join(str(k) for k in missing)
            + " that are not in the credential store, so its data cannot be read; nothing was "
            "changed"
        )
    revision = manifest.schema_revision
    if revision is not None and revision not in migrate.revision_history():
        raise RestoreError(
            f"the backup has database schema {revision}, which is newer than this program; "
            "update the program first"
        )


def restore_archive(
    archive: Path,
    *,
    paths: DataPaths,
    ring: KeyRing,
    clock: Clock,
    media: MediaStore,
    local_date: str,
) -> RestoreReport:
    """Replace the data with the contents of ``archive`` (see the module description)."""
    try:
        manifest = read_manifest(archive, ring)
    except ArchiveError as exc:
        raise RestoreError(str(exc)) from None
    check_requirements(manifest, ring)
    paths.tmp_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="restore-", dir=paths.tmp_dir))
    try:
        try:
            extracted = extract_archive(archive, ring, work)
        except ArchiveError as exc:
            raise RestoreError(str(exc)) from None
        damaged = check_database(extracted.database) if extracted.database else ["no database"]
        if damaged:
            raise RestoreError(
                "the database inside the backup fails SQLite's integrity check: "
                + "; ".join(damaged[:3])
            )
        report = _swap(
            extracted.database,
            extracted.vectors_dir,
            manifest,
            archive,
            paths,
            ring,
            clock,
            media,
            local_date,
        )
        report.vector_files = extracted.vector_files
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _swap(
    database: Path | None,
    vectors: Path | None,
    manifest: Manifest,
    archive: Path,
    paths: DataPaths,
    ring: KeyRing,
    clock: Clock,
    media: MediaStore,
    local_date: str,
) -> RestoreReport:
    if database is None:
        raise RestoreError("the backup holds no database")
    before = migrate.current_revision(paths.db_path)
    pre = None
    if paths.db_path.exists():
        pre = make_pre_restore(paths, ring, clock, media, local_date).name
    tag = f"{clock.now_utc():%Y%m%dT%H%M%S}"
    aside: list[tuple[Path, Path]] = []
    try:
        for current in database_files(paths.db_path):
            target = _moved(current, tag)
            os.replace(current, target)
            aside.append((current, target))
        if paths.vectors_dir.exists():
            target = _moved(paths.vectors_dir, tag)
            os.replace(paths.vectors_dir, target)
            aside.append((paths.vectors_dir, target))
        paths.db_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(database, paths.db_path)
        if vectors is not None and vectors.is_dir():
            os.replace(vectors, paths.vectors_dir)
        status = migrate.schema_status(paths.db_path)
        if status.state is SchemaState.AHEAD:
            raise RestoreError("the restored database is newer than this program")
        if status.state is not SchemaState.CURRENT:
            migrate.upgrade(paths.db_path)
        problems = check_database(paths.db_path)
        if problems:
            raise RestoreError("the restored database fails the integrity check")
        sampled = sample_decrypt(paths.db_path, ring)
        restored, missing = _restore_media(manifest, media, paths.backups_dir / POOL_DIRNAME)
    except BaseException:
        _put_back(aside, paths)
        raise
    for _current, moved in aside:  # the pre-restore backup holds them
        if moved.is_dir():
            shutil.rmtree(moved, ignore_errors=True)
        else:
            moved.unlink(missing_ok=True)
    return RestoreReport(
        archive=archive.name,
        created_at=manifest.created_at,
        key_id=manifest.key_id,
        rows=manifest.row_count,
        schema_before=before,
        schema_after=migrate.current_revision(paths.db_path),
        pre_restore=pre,
        media_listed=len(manifest.media),
        media_restored=restored,
        media_missing=missing,
        vector_files=0,
        sampled_values=sampled,
    )


def _put_back(aside: list[tuple[Path, Path]], paths: DataPaths) -> None:
    """Undo a restore that failed half way: delete what was put in place, move the old back."""
    for path in database_files(paths.db_path):
        path.unlink(missing_ok=True)
    if paths.vectors_dir.exists():
        shutil.rmtree(paths.vectors_dir, ignore_errors=True)
    for current, moved in reversed(aside):
        if moved.exists():
            os.replace(moved, current)


def _restore_media(manifest: Manifest, media: MediaStore, pool: Path) -> tuple[int, int]:
    """Copy the listed media files the data folder lacks from the pool: ``(restored, missing)``."""
    media.root.mkdir(parents=True, exist_ok=True)
    restored = missing = 0
    for entry in manifest.media:
        target = media.path_for(entry.sha256)
        if target.is_file():
            continue
        source = pool / f"{entry.sha256}.enc"
        if not source.is_file():
            missing += 1
            continue
        part = target.with_name(target.name + ".part")
        shutil.copyfile(source, part)
        os.replace(part, target)
        restored += 1
    return restored, missing
