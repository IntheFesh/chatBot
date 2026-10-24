"""Making backups, keeping them, deleting what is old (R-OPS-006, R-STO-003).

:class:`BackupService` is what ``twin backup now``, the daily schedule and the restore use:

* :meth:`create` writes one archive (the database snapshot, the vector index, the media list, all
  sealed with a key derived from the current database key), its index file, the ``backup_records``
  row, then applies the retention policy and brings the off-site copy up to date.  A backup that
  fails leaves a ``failed`` record, raises the ``backup_failed`` alert and re-raises; one that
  works closes that alert;
* :meth:`apply_retention` keeps 14 daily and 8 weekly backups (:mod:`twin.ops.backup.retention`),
  removes the rest with their index files, empties the media pool of files no kept backup lists,
  and then deletes the retired database keys that nothing needs any more (R-STO-003);
* :meth:`sync_mirror` copies to ``ops.backup_mirror_dir``; a folder that is not there raises the
  ``backup_mirror_unavailable`` alert and changes nothing about the local backup;
* :meth:`reconcile` makes the records agree with the files in the folder (after a restore the
  database is an old one that does not know the newer backups).

Two processes never write the same archive at once: ``data/locks/backup.lock``.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from twin import __version__
from twin.clock import Clock
from twin.config.loader import DataPaths
from twin.config.settings import OpsConfig
from twin.ops.alerts import AlertService
from twin.ops.backup.archive import (
    ArchiveSource,
    Manifest,
    check_database,
    database_key_ids,
    extract_archive,
    read_manifest,
    scan_media,
    sha256_of,
    write_archive,
)
from twin.ops.backup.layout import (
    POOL_DIRNAME,
    archive_name,
    is_archive,
    kind_of,
    read_sidecar,
    sidecar_name,
    write_sidecar,
)
from twin.ops.backup.ledger import BackupLedger, BackupView
from twin.ops.backup.mirror import MirrorReport, MirrorUnavailableError, sync_mirror
from twin.ops.backup.retention import Candidate, plan_retention, releasable_keys
from twin.ops.filelock import FileLock
from twin.ops.logging import get_logger
from twin.services import Services
from twin.storage.crypto import KeyRing
from twin.storage.db import Database
from twin.storage.keystore import KeyStore
from twin.storage.media import MediaStore

log = get_logger("twin.backup")


class BackupError(Exception):
    """A backup could not be made; the message is a code, never content."""


class BackupBusyError(BackupError):
    """Another process is making a backup right now."""


@dataclass(frozen=True)
class RetentionOutcome:
    deleted: tuple[str, ...]
    released_keys: tuple[int, ...]
    pool_removed: int


@dataclass(frozen=True)
class VerifyReport:
    """What ``twin backup verify`` found."""

    file: str
    size_bytes: int
    sha256: str
    manifest: Manifest
    database_problems: tuple[str, ...]
    vector_files: int
    media_missing: int
    recorded_sha256: str | None

    @property
    def ok(self) -> bool:
        return (
            not self.database_problems
            and self.media_missing == 0
            and (self.recorded_sha256 in (None, self.sha256))
        )


class BackupService:
    """See the module description."""

    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        paths: DataPaths,
        ring: KeyRing,
        keystore: KeyStore,
        media: MediaStore,
        config: OpsConfig,
        zone: Callable[[], ZoneInfo],
        alerts: AlertService | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._paths = paths
        self._ring = ring
        self._keystore = keystore
        self._media = media
        self._config = config
        self._zone = zone
        self._alerts = alerts
        self.ledger = BackupLedger(db, clock)

    @classmethod
    def from_services(cls, services: Services) -> BackupService:
        from twin.schedule.service import time_service_for

        time = time_service_for(services)
        return cls(
            db=services.db,
            clock=services.clock,
            paths=services.paths,
            ring=services.keyring,
            keystore=services.keystore,
            media=services.media,
            config=services.settings.ops,
            zone=time.bot_timezone,
            alerts=services.alerts,
        )

    # --------------------------------------------------------------------- folders

    @property
    def backups_dir(self) -> Path:
        return self._paths.backups_dir

    @property
    def pool_dir(self) -> Path:
        return self.backups_dir / POOL_DIRNAME

    def _lock(self) -> FileLock:
        return FileLock(self._paths.locks_dir / "backup.lock")

    def _source(self) -> ArchiveSource:
        return ArchiveSource(
            db_path=self._paths.db_path,
            vectors_dir=self._paths.vectors_dir,
            media=self._media,
            pool_dir=self.pool_dir,
            tmp_dir=self._paths.tmp_dir,
        )

    # ----------------------------------------------------------------------- create

    def create(self, kind: str = "daily") -> BackupView:
        """Make a backup now (blocking).  ``kind``: ``daily``, ``manual`` or ``pre_restore``."""
        lock = self._lock()
        if not lock.acquire(blocking=False):
            raise BackupBusyError("another backup is running")
        try:
            return self._create_locked(kind)
        finally:
            lock.release()

    def _create_locked(self, kind: str) -> BackupView:
        started = self._clock.monotonic()
        now = self._clock.now_utc()
        local_date = now.astimezone(self._zone()).date().isoformat()
        name = archive_name(kind, local_date, now)
        dest = self.backups_dir / name
        try:
            written = write_archive(
                dest,
                self._ring,
                self._source(),
                created_at=now,
                local_date=local_date,
                kind=kind,
                app_version=__version__,
            )
            write_sidecar(self.backups_dir / sidecar_name(name), written.manifest)
        except Exception as exc:
            reason = type(exc).__name__
            elapsed = round((self._clock.monotonic() - started) * 1000)
            log.error("backup_failed", reason=reason, kind=kind)
            self._record_failure(kind, local_date, reason, elapsed)
            raise BackupError(f"the backup failed ({reason}); see the log") from None
        elapsed = round((self._clock.monotonic() - started) * 1000)
        view = self.ledger.add_ok(
            written.manifest,
            kind=kind,
            file_name=name,
            size_bytes=written.size_bytes,
            sha256=written.sha256,
            duration_ms=elapsed,
            created_at=now,
        )
        log.info(
            "backup_made",
            kind=kind,
            size_mb=round(written.size_bytes / 1e6, 1),
            media=len(written.manifest.media),
            seconds=round(elapsed / 1000, 1),
        )
        if self._alerts is not None:
            self._alerts.recover("backup_failed", "backups work again")
        self._afterwards(view)
        return view

    def _record_failure(self, kind: str, local_date: str, reason: str, elapsed_ms: int) -> None:
        try:
            self.ledger.add_failed(kind, local_date, reason, elapsed_ms)
        except Exception:  # the record is a courtesy; the alert below matters more
            log.warning("backup_failure_not_recorded")
        if self._alerts is not None:
            self._alerts.raise_alert(
                "backup_failed",
                f"the {kind} backup failed ({reason})",
                severity="critical",
                detail={"reason": reason, "kind": kind},
                dedup_key=f"backup_failed:{local_date}",
            )

    def _afterwards(self, view: BackupView) -> None:
        """Retention and the off-site copy; a problem there never fails the backup."""
        try:
            self.apply_retention()
        except Exception as exc:
            log.warning("backup_retention_failed", reason=type(exc).__name__)
        if view.kind != "pre_restore":
            self.sync_mirror()

    # ---------------------------------------------------------------- retention, keys

    def _candidates(self, views: list[BackupView]) -> list[Candidate]:
        return [Candidate(v.id, v.kind, v.local_date, v.created_at.isoformat()) for v in views]

    def kept_archives(self) -> list[BackupView]:
        """The backups the policy keeps (their files exist and stay)."""
        views = self.ledger.usable()
        plan = plan_retention(
            self._candidates(views),
            keep_daily=self._config.backup_keep_daily,
            keep_weekly=self._config.backup_keep_weekly,
        )
        return [v for v in views if v.id in plan.keep]

    def apply_retention(self) -> RetentionOutcome:
        """Delete backups beyond the policy, empty the pool, release unneeded keys."""
        views = self.ledger.usable()
        plan = plan_retention(
            self._candidates(views),
            keep_daily=self._config.backup_keep_daily,
            keep_weekly=self._config.backup_keep_weekly,
        )
        deleted: list[str] = []
        for view in views:
            if view.id not in plan.drop or view.file_name is None:
                continue
            (self.backups_dir / view.file_name).unlink(missing_ok=True)
            (self.backups_dir / sidecar_name(view.file_name)).unlink(missing_ok=True)
            self.ledger.mark_deleted(view.id)
            deleted.append(view.file_name)
        removed = self._collect_pool([v for v in views if v.id in plan.keep])
        released = self.release_retired_keys()
        if deleted:
            log.info("backups_deleted", count=len(deleted))
        return RetentionOutcome(tuple(deleted), tuple(released), removed)

    def _referenced_media(self, kept: list[BackupView]) -> set[str] | None:
        """Pool files some kept backup lists (``None``: an index is missing, keep everything)."""
        referenced: set[str] = set()
        for view in kept:
            if view.file_name is None:
                continue
            side = read_sidecar(self.backups_dir / sidecar_name(view.file_name))
            if side is None:
                return None
            referenced.update(side.media)
        referenced.update(self._media.iter_hashes())
        return referenced

    def _collect_pool(self, kept: list[BackupView]) -> int:
        referenced = self._referenced_media(kept)
        if referenced is None or not self.pool_dir.is_dir():
            return 0
        removed = 0
        for entry in self.pool_dir.iterdir():
            if entry.is_file() and entry.suffix == ".enc" and entry.stem not in referenced:
                entry.unlink(missing_ok=True)
                removed += 1
        return removed

    def live_key_ids(self) -> set[int]:
        """Keys the live database and media files are sealed with."""
        keys = {self._ring.current_id}
        if self._paths.db_path.is_file():
            keys |= database_key_ids(self._paths.db_path)
        keys |= {entry.key_id for entry in scan_media(self._media)}
        return keys

    def release_retired_keys(self) -> list[int]:
        """Delete retired database keys that no kept backup and no live value needs (R-STO-003)."""
        self.reconcile()
        free = releasable_keys(
            self._ring.retired_ids, self.ledger.kept_key_ids(), self.live_key_ids()
        )
        if not free:
            return []
        gone = self._keystore.delete_retired(self._ring, free)
        log.info("retired_keys_released", count=len(gone))
        return gone

    # ---------------------------------------------------------------------- mirror

    def sync_mirror(self) -> MirrorReport | None:
        """Bring the off-site copy up to date; ``None`` if none is configured or it failed."""
        target = self._config.backup_mirror_dir
        if not target:
            return None
        kept = self.kept_archives()
        try:
            referenced = self._referenced_media(kept)
            report = sync_mirror(
                Path(target).expanduser(),
                self.backups_dir,
                self.pool_dir,
                keep_archives={v.file_name for v in kept if v.file_name},
                pool_shas=referenced if referenced is not None else set(),
            )
        except MirrorUnavailableError as exc:
            log.warning("backup_mirror_unavailable", reason=exc.code)
            if self._alerts is not None:
                self._alerts.raise_alert(
                    "backup_mirror_unavailable",
                    f"the off-site backup folder cannot be used ({exc.code})",
                    severity="warning",
                    detail={"reason": exc.code},
                    dedup_key="backup_mirror",
                )
            return None
        if self._alerts is not None:
            self._alerts.recover("backup_mirror_unavailable", "the off-site backup folder works")
        for view in kept:
            if view.mirrored_at is None and view.kind != "pre_restore":
                self.ledger.mark_mirrored(view.id)
        return report

    # ------------------------------------------------------------------- reconcile

    def reconcile(self) -> int:
        """Make the records agree with the files; returns how many records were added."""
        folder = self.backups_dir
        if not folder.is_dir():
            return 0
        known = {v.file_name for v in self.ledger.usable() if v.file_name}
        for view in self.ledger.usable():
            if view.file_name and not (folder / view.file_name).is_file():
                self.ledger.mark_deleted(view.id)
        added = 0
        for entry in sorted(folder.iterdir()):
            if not entry.is_file() or not is_archive(entry.name) or entry.name in known:
                continue
            try:
                manifest = read_manifest(entry, self._ring)
            except Exception as exc:  # another installation's backup, or a damaged file
                log.warning("backup_not_adopted", file=entry.name, reason=type(exc).__name__)
                continue
            self.ledger.add_ok(
                manifest,
                kind=kind_of(entry.name),
                file_name=entry.name,
                size_bytes=entry.stat().st_size,
                sha256=sha256_of(entry),
                duration_ms=0,
                created_at=datetime.fromisoformat(manifest.created_at),
            )
            side = folder / sidecar_name(entry.name)
            if not side.is_file():
                write_sidecar(side, manifest)
            added += 1
        return added

    # ---------------------------------------------------------------------- verify

    def verify(self, path: Path) -> VerifyReport:
        """Decrypt a backup completely and check what is in it (blocking)."""
        recorded = next((v.sha256 for v in self.ledger.usable() if v.file_name == path.name), None)
        self._paths.tmp_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="verify-", dir=self._paths.tmp_dir))
        try:
            extracted = extract_archive(path, self._ring, work)
            problems = tuple(check_database(extracted.database)) if extracted.database else ()
            problems += tuple(extracted.problems)
            missing = sum(
                1
                for entry in extracted.manifest.media
                if not (self._media.exists(entry.sha256))
                and not (self.pool_dir / f"{entry.sha256}.enc").is_file()
            )
            return VerifyReport(
                file=path.name,
                size_bytes=path.stat().st_size,
                sha256=sha256_of(path),
                manifest=extracted.manifest,
                database_problems=problems,
                vector_files=extracted.vector_files,
                media_missing=missing,
                recorded_sha256=recorded,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
