"""Restoring a backup: round trip, refusals, rollback, old schemas, pre-restore copy (R-OPS-006)."""

from __future__ import annotations

import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support.backup_world import (
    BackupWorld,
    build_backup_world,
    fact_texts,
    on_day,
    table_counts,
    vector_ids,
)
from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from twin.ops.backup import restore as restore_module
from twin.ops.backup.archive import ArchiveSource, Manifest, write_archive
from twin.ops.backup.layout import POOL_DIRNAME, is_archive
from twin.ops.backup.restore import (
    RestoreError,
    RestoreReport,
    check_requirements,
    restore_archive,
    sample_decrypt,
)
from twin.ops.backup.sealed import BackupDecryptionError, BackupFormatError
from twin.services import Services
from twin.storage import migrate
from twin.storage.crypto import KeyRing, UnknownKeyError
from twin.storage.models import Job
from twin.storage.rotate import rotate_db_key


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> BackupWorld:
    return build_backup_world(services)


def do_restore(world: BackupWorld, archive: Path) -> RestoreReport:
    services = world.services
    services.db.dispose()  # the CLI closes everything first; the engine reconnects on demand
    return restore_archive(
        archive,
        paths=services.paths,
        ring=services.keyring,
        clock=services.clock,
        media=services.media,
        local_date="2026-10-09",
    )


def mutate(world: BackupWorld) -> None:
    """Change the installation after the backup: facts gone, jobs added, a media file lost."""
    from sqlalchemy import delete

    from twin.storage.memory_models import Fact

    services = world.services
    with services.db.transaction() as session:
        session.execute(delete(Fact))
        for number in range(5):
            session.add(Job(type="later", payload={"n": number}))
    services.media.path_for(next(iter(world.media))).unlink()
    shutil.rmtree(services.paths.vectors_dir)


def archives_in(world: BackupWorld) -> list[str]:
    return sorted(p.name for p in world.backup.backups_dir.iterdir() if is_archive(p.name))


def test_a_restore_brings_back_rows_content_vectors_and_media(world: BackupWorld) -> None:
    services = world.services
    counts = table_counts(services.paths.db_path)
    texts = fact_texts(services)
    vectors = vector_ids(services)
    assert len(texts) == 6 and len(vectors) == 6
    view = world.backup.create("manual")
    archive = world.backup.backups_dir / str(view.file_name)
    mutate(world)
    assert fact_texts(services) == [] and vector_ids(services) == []

    report = do_restore(world, archive)

    assert table_counts(services.paths.db_path) == counts  # every table, row for row
    assert fact_texts(services) == texts  # decrypted content
    sample = random.Random(7).sample(texts, 3)
    assert set(sample) <= set(fact_texts(services))
    assert vector_ids(services) == vectors
    for sha, content in world.media.items():
        assert services.media.read_bytes(sha) == content
    assert report.rows == view.row_count and report.key_id == 1
    assert report.media_listed == 2 and report.media_restored == 1 and report.media_missing == 0
    assert report.vector_files > 0 and report.sampled_values > 0 and report.problems == []
    assert report.schema_before == report.schema_after == migrate.head_revision()
    assert report.pre_restore is not None and report.pre_restore.startswith("pre-restore-")


def test_what_was_there_before_is_kept_and_can_be_restored(
    world: BackupWorld, clock: ManualClock
) -> None:
    services = world.services
    view = world.backup.create("manual")
    archive = world.backup.backups_dir / str(view.file_name)
    mutate(world)
    after_mutation = table_counts(services.paths.db_path)
    report = do_restore(world, archive)
    assert report.pre_restore in archives_in(world)
    clock.tick(60)  # the next pre-restore copy gets a name of its own
    # the pre-restore copy holds the state at the moment of the restore: restore it again
    do_restore(world, world.backup.backups_dir / str(report.pre_restore))
    again = table_counts(services.paths.db_path)
    assert again["facts"] == 0 and again["jobs"] == 13
    assert {k: v for k, v in again.items() if k != "backup_records"} == {
        k: v for k, v in after_mutation.items() if k != "backup_records"
    }


def test_the_pre_restore_copy_is_found_by_the_ledger_after_a_reconcile(
    world: BackupWorld,
) -> None:
    view = world.backup.create("manual")
    report = do_restore(world, world.backup.backups_dir / str(view.file_name))
    world.backup.reconcile()
    kinds = {v.file_name: v.kind for v in world.backup.ledger.usable()}
    assert kinds[str(report.pre_restore)] == "pre_restore"


def test_a_tampered_backup_changes_nothing(world: BackupWorld) -> None:
    services = world.services
    view = world.backup.create("manual")
    archive = world.backup.backups_dir / str(view.file_name)
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0xFF
    archive.write_bytes(bytes(data))
    mutate(world)
    state = table_counts(services.paths.db_path)
    files_before = archives_in(world)
    with pytest.raises(BackupDecryptionError):
        do_restore(world, archive)
    assert table_counts(services.paths.db_path) == state
    assert archives_in(world) == files_before  # no pre-restore copy was made: nothing was touched
    assert not list(services.paths.data_dir.glob("**/*.replaced-*"))


def test_a_file_that_is_not_a_backup_changes_nothing(world: BackupWorld, tmp_path: Path) -> None:
    junk = tmp_path / "junk.bak.enc"
    junk.write_bytes(b"this is not a backup at all" * 20)
    state = table_counts(world.services.paths.db_path)
    with pytest.raises(BackupFormatError):
        do_restore(world, junk)
    assert table_counts(world.services.paths.db_path) == state


def test_keys_the_backup_needs_must_be_in_the_credential_store(world: BackupWorld) -> None:
    services = world.services
    rotate_db_key(services.db, services.keystore, services.keyring, services.clock, None)
    view = world.backup.create("manual")  # sealed with key 2; the media files list key 1
    archive = world.backup.backups_dir / str(view.file_name)
    only_new = KeyRing({2: services.keyring.key_bytes(2)}, 2)
    with pytest.raises(RestoreError, match=r"key.*1.*not in the credential store"):
        restore_archive(
            archive,
            paths=services.paths,
            ring=only_new,
            clock=services.clock,
            media=services.media,
            local_date="2026-10-09",
        )
    assert archives_in(world) == [str(view.file_name)]


def test_a_schema_newer_than_the_program_is_refused() -> None:
    ring = KeyRing({1: bytes(32)}, 1)
    manifest = Manifest(
        "2026-10-09T12:00:00+00:00", "2026-10-09", "daily", None, 1, [1], {}, [], "x"
    )
    check_requirements(manifest, ring)  # no schema revision recorded: nothing to compare
    manifest.schema_revision = migrate.head_revision()
    check_requirements(manifest, ring)
    manifest.schema_revision = "9999_from_the_future"
    with pytest.raises(RestoreError, match="newer than this program"):
        check_requirements(manifest, ring)


def test_a_restore_that_fails_half_way_puts_everything_back(
    world: BackupWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = world.services
    view = world.backup.create("manual")
    archive = world.backup.backups_dir / str(view.file_name)
    services.db.dispose()
    with services.db.transaction() as session:
        session.add(Job(type="later", payload={"n": 1}))
    before = table_counts(services.paths.db_path)
    vectors = vector_ids(services)
    texts = fact_texts(services)

    def broken(db_path: Path, ring: KeyRing, count: int = 40) -> int:
        raise RuntimeError("the sample could not be decrypted")

    monkeypatch.setattr(restore_module, "sample_decrypt", broken)
    with pytest.raises(RuntimeError, match="sample"):
        do_restore(world, archive)
    assert table_counts(services.paths.db_path) == before  # the old database is back
    assert vector_ids(services) == vectors and fact_texts(services) == texts
    assert not list(services.paths.data_dir.glob("**/*.replaced-*"))
    pre = [name for name in archives_in(world) if name.startswith("pre-restore-")]
    assert len(pre) == 1  # the copy of the current data was made first and stays


def test_media_the_pool_lacks_is_reported_not_fatal(world: BackupWorld) -> None:
    services = world.services
    view = world.backup.create("manual")
    archive = world.backup.backups_dir / str(view.file_name)
    sha = next(iter(world.media))
    services.media.path_for(sha).unlink()
    (world.backup.backups_dir / POOL_DIRNAME / f"{sha}.enc").unlink()
    report = do_restore(world, archive)
    assert report.media_missing == 1 and report.media_restored == 0
    assert not services.media.exists(sha)
    other = next(s for s in world.media if s != sha)
    assert services.media.read_bytes(other) == world.media[other]


def test_a_backup_of_an_older_schema_is_brought_up_to_date(world: BackupWorld) -> None:
    services = world.services
    db_path = services.paths.db_path
    head = migrate.head_revision()
    services.db.dispose()
    migrate.downgrade(db_path, "-1")  # the installation as it was one migration ago
    older = migrate.current_revision(db_path)
    assert older != head
    archive = world.backup.backups_dir / "twin-2026-09-01.bak.enc"
    written = write_archive(
        archive,
        services.keyring,
        ArchiveSource(
            db_path=db_path,
            vectors_dir=services.paths.vectors_dir,
            media=services.media,
            pool_dir=world.backup.pool_dir,
            tmp_dir=services.paths.tmp_dir,
        ),
        created_at=datetime(2026, 9, 1, 15, tzinfo=UTC),
        local_date="2026-09-01",
        kind="daily",
        app_version="test",
    )
    assert written.manifest.schema_revision == older
    migrate.upgrade(db_path)
    assert migrate.current_revision(db_path) == head
    report = do_restore(world, archive)
    assert report.schema_before == head and report.schema_after == head
    assert migrate.current_revision(db_path) == head  # migrated again after the swap
    assert "backup_records" in table_counts(db_path)  # a table the older schema did not have
    assert fact_texts(services) == sorted(world.fact_texts)


def test_the_sample_check_opens_values_with_the_available_keys(world: BackupWorld) -> None:
    services = world.services
    assert sample_decrypt(services.paths.db_path, services.keyring) > 0
    assert sample_decrypt(services.paths.db_path, services.keyring, count=1) >= 1
    rotate_db_key(services.db, services.keystore, services.keyring, services.clock, services.media)
    only_new = KeyRing({2: services.keyring.key_bytes(2)}, 2)
    assert sample_decrypt(services.paths.db_path, only_new) > 0  # everything is on key 2 now
    # a database whose values need a key that is gone cannot be sampled
    with pytest.raises(UnknownKeyError):
        sample_decrypt(services.paths.db_path, KeyRing({7: services.keyring.key_bytes(2)}, 7))


def test_a_restore_after_a_key_rotation_reads_the_old_backup_with_the_retired_key(
    world: BackupWorld, clock: ManualClock
) -> None:
    services = world.services
    clock.set_time(on_day(0))
    old = world.backup.create("daily")
    rotate_db_key(services.db, services.keystore, services.keyring, services.clock, services.media)
    assert services.keyring.current_id == 2
    texts = fact_texts(services)
    report = do_restore(world, world.backup.backups_dir / str(old.file_name))
    assert report.key_id == 1  # the database inside is still sealed with key 1
    assert fact_texts(services) == texts
    assert sample_decrypt(services.paths.db_path, services.keyring) > 0
    # the restored database values are on key 1 again: the retired key is needed again
    assert 1 in world.backup.live_key_ids()


def test_restoring_over_nothing_makes_no_pre_restore_copy(
    world: BackupWorld, tmp_path: Path
) -> None:
    services = world.services
    view = world.backup.create("manual")
    archive = world.backup.backups_dir / str(view.file_name)
    services.db.dispose()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{services.paths.db_path}{suffix}").unlink(missing_ok=True)
    report = do_restore(world, archive)
    assert report.pre_restore is None
    assert fact_texts(services) == sorted(world.fact_texts)
