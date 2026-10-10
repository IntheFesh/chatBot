"""Encrypted backups: content, sealing, retention, keys, off-site copy (R-OPS-006)."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete, select

from tests.support.backup_world import (
    MARKER,
    BackupWorld,
    build_backup_world,
    on_day,
    table_counts,
)
from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from twin.ops.backup import archive as archive_module
from twin.ops.backup.archive import (
    ArchiveError,
    Manifest,
    MediaEntry,
    check_database,
    extract_archive,
    read_manifest,
    sha256_of,
)
from twin.ops.backup.layout import (
    archive_name,
    date_of,
    is_archive,
    kind_of,
    read_sidecar,
    sidecar_name,
)
from twin.ops.backup.mirror import MirrorUnavailableError, check_mirror, sync_mirror
from twin.ops.backup.retention import Candidate, plan_retention, releasable_keys
from twin.ops.backup.sealed import (
    HEADER_SIZE,
    BackupDecryptionError,
    BackupFormatError,
    SealReader,
    SealWriter,
    read_header,
)
from twin.ops.backup.service import BackupBusyError, BackupError, BackupService
from twin.ops.filelock import FileLock
from twin.services import Services
from twin.storage.crypto import KeyRing, UnknownKeyError, generate_key
from twin.storage.keystore import key_secret_name
from twin.storage.models import Job
from twin.storage.rotate import rotate_db_key


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> BackupWorld:
    return build_backup_world(services)


def backup_files(service: BackupService) -> list[str]:
    return sorted(p.name for p in service.backups_dir.iterdir() if is_archive(p.name))


def flip(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


# ----------------------------------------------------------------------------- the content


def test_a_backup_holds_the_database_the_vectors_and_the_list_of_media(world: BackupWorld) -> None:
    services = world.services
    view = world.backup.create("manual")
    path = world.backup.backups_dir / str(view.file_name)
    assert view.file_name == "twin-2026-10-09.bak.enc" and path.is_file()
    assert view.status == "ok" and view.kind == "manual" and view.key_id == 1
    assert view.size_bytes == path.stat().st_size and view.sha256 == sha256_of(path)
    manifest = read_manifest(path, services.keyring)
    assert manifest.tables["facts"] == 6 and manifest.tables["jobs"] == 8
    assert manifest.row_count == view.row_count == sum(manifest.tables.values())
    assert manifest.key_id == 1 and manifest.key_ids == [1] and manifest.vector_files > 0
    assert {entry.sha256 for entry in manifest.media} == set(world.media)
    assert manifest.schema_revision and manifest.local_date == "2026-10-09"
    # the media files are not in the archive: they are in the pool, listed in the index next to it
    sidecar = read_sidecar(path.with_name(sidecar_name(path.name)))
    assert (
        sidecar is not None and set(sidecar.media) == set(world.media) and sidecar.key_ids == (1,)
    )
    pool = world.backup.pool_dir
    assert {p.stem for p in pool.glob("*.enc")} == set(world.media)
    assert [r.id for r in world.backup.ledger.usable()] == [view.id]


def test_nothing_readable_is_in_the_file(world: BackupWorld) -> None:
    view = world.backup.create("manual")
    blobs = [world.backup.backups_dir / str(view.file_name)]
    blobs += list(world.backup.pool_dir.glob("*.enc"))
    blobs += [world.backup.backups_dir / sidecar_name(str(view.file_name))]
    for path in blobs:
        raw = path.read_bytes()
        assert MARKER.encode("utf-8") not in raw and b"STICKER" not in raw, path.name
        assert b"SQLite format" not in raw and b"facts" not in raw, path.name
    header = read_header(blobs[0])
    assert header.key_id == 1 and len(header.raw) == HEADER_SIZE


def test_the_snapshot_is_the_database_as_of_its_moment_not_a_later_write(
    world: BackupWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = world.services
    original = archive_module.snapshot_database

    def writing_snapshot(db_path: Path, dest: Path) -> None:
        original(db_path, dest)
        with services.db.transaction() as session:  # a write right after the copy was taken
            session.add(Job(type="demo", payload={"n": 99}))

    monkeypatch.setattr(archive_module, "snapshot_database", writing_snapshot)
    view = world.backup.create("manual")
    path = world.backup.backups_dir / str(view.file_name)
    assert read_manifest(path, services.keyring).tables["jobs"] == 8  # not torn, not 9
    report = world.backup.verify(path)
    assert report.ok and not report.database_problems


def test_a_backup_made_while_the_application_writes_is_consistent(world: BackupWorld) -> None:
    """The online backup API copies pages in steps; writers keep going and the copy is whole."""
    services = world.services
    stop = threading.Event()
    written = [0]

    def writer() -> None:
        while not stop.is_set() and written[0] < 400:
            with services.db.transaction() as session:
                session.add(Job(type="demo", payload={"n": 1000 + written[0], "text": MARKER}))
            written[0] += 1
            time.sleep(0.002)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        view = world.backup.create("manual")
    finally:
        stop.set()
        thread.join()
    path = world.backup.backups_dir / str(view.file_name)
    work = services.paths.tmp_dir / "extracted"
    extracted = extract_archive(path, services.keyring, work)
    assert extracted.database is not None
    assert check_database(extracted.database) == []
    assert table_counts(extracted.database)["jobs"] == extracted.manifest.tables["jobs"]
    assert 8 <= extracted.manifest.tables["jobs"] <= 8 + written[0]


def test_every_table_of_the_database_is_counted_like_the_database_itself(
    world: BackupWorld,
) -> None:
    before = table_counts(world.services.paths.db_path)
    view = world.backup.create("manual")
    path = world.backup.backups_dir / str(view.file_name)
    manifest = read_manifest(path, world.services.keyring)
    assert manifest.tables == before


def test_a_second_backup_on_the_same_day_replaces_the_first(
    world: BackupWorld, clock: ManualClock
) -> None:
    first = world.backup.create("daily")
    clock.tick(3600)
    second = world.backup.create("manual")
    assert first.file_name == second.file_name and backup_files(world.backup) == [first.file_name]
    usable = world.backup.ledger.usable()
    assert [v.id for v in usable] == [second.id]  # the first record is marked replaced
    states = {v.id: v.status for v in world.backup.ledger.recent()}
    assert states[first.id] == "deleted" and states[second.id] == "ok"


def test_a_failed_backup_is_recorded_alerted_and_forgotten_when_one_works(
    world: BackupWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr("twin.ops.backup.service.write_archive", explode)
        with pytest.raises(BackupError, match="OSError"):
            world.backup.create("daily")
    (failed,) = world.backup.ledger.recent()
    assert failed.status == "failed" and failed.error == "OSError" and failed.file_name is None
    (alert,) = world.services.alerts.open_alerts()
    assert alert.category == "backup_failed" and alert.severity == "critical"
    assert "disk full" not in alert.title
    world.backup.create("daily")
    assert world.services.alerts.open_alerts() == []  # the recovery closed it


def test_two_backups_never_write_at_once(world: BackupWorld) -> None:
    lock = FileLock(world.services.paths.locks_dir / "backup.lock")
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(BackupBusyError):
            world.backup.create("daily")
    finally:
        lock.release()
    assert world.backup.create("daily").status == "ok"


def test_the_media_pool_is_copied_once_and_follows_the_live_files(world: BackupWorld) -> None:
    world.backup.create("daily")
    pool_before = {p.name: p.stat().st_mtime_ns for p in world.backup.pool_dir.glob("*.enc")}
    assert len(pool_before) == 2
    world.backup.create("manual")  # nothing new: the pool files are not copied again
    assert {
        p.name: p.stat().st_mtime_ns for p in world.backup.pool_dir.glob("*.enc")
    } == pool_before


# ----------------------------------------------------------------------------- the sealing


def sealed(key_ring: KeyRing, payload: bytes, chunk_log2: int = 10) -> bytes:
    import io

    raw = io.BytesIO()
    writer = SealWriter(raw, key_ring, chunk_log2=chunk_log2)
    writer.write(payload)
    writer.close()
    return raw.getvalue()


def opened(key_ring: KeyRing, blob: bytes) -> bytes:
    import io

    return SealReader(io.BytesIO(blob), key_ring).read()


@pytest.fixture
def ring() -> KeyRing:
    return KeyRing({1: generate_key(), 2: generate_key()}, 2)


def test_a_sealed_stream_round_trips_in_pieces_and_names_its_key(ring: KeyRing) -> None:
    payload = os.urandom(5000)  # five chunks of one KiB, the last one partial
    blob = sealed(ring, payload)
    assert opened(ring, blob) == payload
    assert blob[: len(b"TWBAK1")] == b"TWBAK1" and payload[:32] not in blob
    import io

    reader = SealReader(io.BytesIO(blob), ring)
    assert reader.header.key_id == 2
    pieces = [reader.read(700) for _ in range(8)]
    assert b"".join(pieces) == payload


def test_an_empty_payload_is_still_authenticated(ring: KeyRing) -> None:
    blob = sealed(ring, b"")
    assert opened(ring, blob) == b""
    flip_at = len(blob) - 3
    broken = bytearray(blob)
    broken[flip_at] ^= 1
    with pytest.raises(BackupDecryptionError):
        opened(ring, bytes(broken))


def test_any_changed_byte_fails_authentication(ring: KeyRing) -> None:
    payload = os.urandom(3500)
    blob = sealed(ring, payload)
    block = (1 << 10) + 16
    for offset in (HEADER_SIZE + 5, HEADER_SIZE + block + 7, len(blob) - 1):
        broken = bytearray(blob)
        broken[offset] ^= 0x80
        with pytest.raises(BackupDecryptionError):
            opened(ring, bytes(broken))
    # the header is associated data: the salt, the nonce prefix, the chunk size
    for offset in (12, 30, HEADER_SIZE - 1):
        broken = bytearray(blob)
        broken[offset] ^= 0x01
        with pytest.raises(BackupFormatError):
            opened(ring, bytes(broken))


def test_a_cut_off_tail_a_dropped_or_swapped_chunk_is_noticed(ring: KeyRing) -> None:
    payload = os.urandom(4000)
    blob = sealed(ring, payload)
    block = (1 << 10) + 16
    chunks = [blob[HEADER_SIZE + i * block : HEADER_SIZE + (i + 1) * block] for i in range(4)]
    header = blob[:HEADER_SIZE]
    assert len(chunks[-1]) < block
    with pytest.raises(BackupDecryptionError):  # the last chunk is gone: the new last is not final
        opened(ring, header + b"".join(chunks[:-1]))
    with pytest.raises(BackupDecryptionError):  # cut in the middle of a chunk
        opened(ring, blob[: len(blob) - 5])
    with pytest.raises(BackupDecryptionError):  # two chunks exchanged
        opened(ring, header + chunks[1] + chunks[0] + chunks[2] + chunks[3])
    with pytest.raises(BackupDecryptionError):  # one chunk repeated
        opened(ring, header + chunks[0] + chunks[0] + chunks[1] + chunks[2] + chunks[3])


def test_other_key_material_or_no_key_cannot_open_it(ring: KeyRing) -> None:
    blob = sealed(ring, b"secret" * 100)
    stranger = KeyRing({2: generate_key()}, 2)  # key id 2 with other bytes
    with pytest.raises(BackupDecryptionError):
        opened(stranger, blob)
    missing = KeyRing({1: generate_key()}, 1)
    with pytest.raises(UnknownKeyError):
        opened(missing, blob)


def test_other_files_are_not_backups(ring: KeyRing, tmp_path: Path) -> None:
    for blob in (b"", b"TWBAK", b"NOTABACKUP" * 10, b"TWBAK1\x09" + bytes(40)):
        with pytest.raises(BackupFormatError):
            opened(ring, blob)
    odd = bytearray(sealed(ring, b"x"))
    odd[HEADER_SIZE - 1] = 3  # a chunk size of 8 bytes is nonsense
    with pytest.raises(BackupFormatError, match="chunk size"):
        opened(ring, bytes(odd))
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")
    with pytest.raises(BackupFormatError):
        read_header(note)


def test_a_damaged_backup_file_fails_verification_in_the_service(world: BackupWorld) -> None:
    view = world.backup.create("manual")
    path = world.backup.backups_dir / str(view.file_name)
    flip(path, path.stat().st_size // 2)
    with pytest.raises(BackupDecryptionError):
        world.backup.verify(path)
    with pytest.raises(BackupDecryptionError):
        extract_archive(path, world.services.keyring, world.services.paths.tmp_dir / "x")


def test_a_truncated_or_extended_backup_file_fails_verification(world: BackupWorld) -> None:
    view = world.backup.create("manual")
    path = world.backup.backups_dir / str(view.file_name)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 20])
    with pytest.raises(BackupDecryptionError):
        world.backup.verify(path)
    path.write_bytes(data + b"appended")
    with pytest.raises(BackupDecryptionError):
        world.backup.verify(path)


def test_verify_finds_what_a_listed_media_file_lacks(world: BackupWorld) -> None:
    view = world.backup.create("manual")
    path = world.backup.backups_dir / str(view.file_name)
    sha = next(iter(world.media))
    world.services.media.path_for(sha).unlink()
    world.backup.pool_dir.joinpath(f"{sha}.enc").unlink()
    report = world.backup.verify(path)
    assert report.media_missing == 1 and not report.ok
    assert report.recorded_sha256 == view.sha256 == report.sha256


def test_verify_compares_with_the_recorded_hash_when_there_is_one(world: BackupWorld) -> None:
    view = world.backup.create("manual")
    path = world.backup.backups_dir / str(view.file_name)
    other = world.backup.backups_dir / "twin-2025-01-01.bak.enc"
    other.write_bytes(path.read_bytes())  # a copy under another name: no record, nothing to compare
    report = world.backup.verify(other)
    assert report.recorded_sha256 is None and report.ok
    assert world.backup.verify(path).recorded_sha256 == view.sha256


# ----------------------------------------------------------------------------- retention


def candidates(first: datetime, days: int, kind: str = "daily") -> list[Candidate]:
    result = []
    for number in range(days):
        moment = first + timedelta(days=number)
        result.append(
            Candidate(f"b{number:03d}", kind, moment.date().isoformat(), moment.isoformat())
        )
    return result


def test_the_policy_keeps_fourteen_days_and_eight_sundays() -> None:
    start = datetime(2026, 6, 1, 15, tzinfo=UTC)  # a Monday; 130 daily backups
    plan = plan_retention(candidates(start, 130), keep_daily=14, keep_weekly=8)
    last = start + timedelta(days=129)
    newest_days = {(last - timedelta(days=n)).date().isoformat() for n in range(14)}
    sundays = [
        (last - timedelta(days=n)).date()
        for n in range(130)
        if (last - timedelta(days=n)).weekday() == 6
    ][:8]
    expected = newest_days | {d.isoformat() for d in sundays}
    kept = {c.local_date for c in candidates(start, 130) if c.id in plan.keep}
    assert kept == expected
    assert len(plan.keep) + len(plan.drop) == 130 and not plan.keep & plan.drop
    assert 14 <= len(plan.keep) <= 22


def test_a_young_history_keeps_everything() -> None:
    start = datetime(2026, 9, 20, 15, tzinfo=UTC)
    plan = plan_retention(candidates(start, 5), keep_daily=14, keep_weekly=8)
    assert plan.drop == frozenset() and len(plan.keep) == 5


def test_a_day_has_one_backup_and_the_newest_of_the_restores_stay() -> None:
    day = datetime(2026, 9, 20, 15, tzinfo=UTC)
    twice = [
        Candidate("early", "daily", "2026-09-20", day.isoformat()),
        Candidate("late", "manual", "2026-09-20", (day + timedelta(hours=3)).isoformat()),
    ]
    restores = [
        Candidate(f"r{n}", "pre_restore", "2026-09-20", (day + timedelta(minutes=n)).isoformat())
        for n in range(5)
    ]
    plan = plan_retention(twice + restores, keep_daily=14, keep_weekly=8)
    assert "late" in plan.keep and "early" in plan.drop
    assert plan.keep & {c.id for c in restores} == {"r4", "r3", "r2"}  # the newest three


def test_a_key_is_released_only_when_nothing_needs_it() -> None:
    assert releasable_keys([1, 2, 3], kept_backup_keys=[2], live_keys=[3]) == [1]
    assert releasable_keys([1, 2], kept_backup_keys=[], live_keys=[]) == [1, 2]
    assert releasable_keys([], kept_backup_keys=[1], live_keys=[1]) == []


def test_the_names_of_the_files_say_what_they_are() -> None:
    at = datetime(2026, 10, 9, 17, 30, 5, tzinfo=UTC)
    assert archive_name("daily", "2026-10-09", at) == "twin-2026-10-09.bak.enc"
    assert archive_name("manual", "2026-10-09", at) == "twin-2026-10-09.bak.enc"
    restore = archive_name("pre_restore", "2026-10-09", at)
    assert restore == "pre-restore-20261009T173005Z.bak.enc"
    assert kind_of(restore) == "pre_restore" and kind_of("twin-2026-10-09.bak.enc") == "daily"
    assert date_of("twin-2026-10-09.bak.enc") == "2026-10-09" and date_of(restore) is None
    assert date_of("twin-oops.bak.enc") is None
    assert is_archive("twin-2026-10-09.bak.enc") and not is_archive("twin-2026-10-09.bak.enc.part")
    assert sidecar_name("twin-2026-10-09.bak.enc") == "twin-2026-10-09.media.zst"
    assert read_sidecar(Path("/nonexistent/x.media.zst")) is None


def test_old_backups_are_deleted_with_their_index_and_unneeded_media(
    world: BackupWorld, clock: ManualClock
) -> None:
    config = world.services.settings.ops
    config.backup_keep_daily, config.backup_keep_weekly = 3, 1
    gone_sha = next(iter(world.media))
    views = []
    for offset in range(12):  # 2026-09-01 (Tuesday) .. 2026-09-12
        clock.set_time(on_day(offset))
        if offset == 4:  # a media file disappears from the live store
            world.services.media.path_for(gone_sha).unlink()
        views.append(world.backup.create("daily"))
    kept = sorted(p.name for p in world.backup.backups_dir.glob("*.bak.enc"))
    # the newest three days and the last Sunday (09-06) stay
    assert kept == [
        "twin-2026-09-06.bak.enc",
        "twin-2026-09-10.bak.enc",
        "twin-2026-09-11.bak.enc",
        "twin-2026-09-12.bak.enc",
    ]
    assert sorted(p.name for p in world.backup.backups_dir.glob("*.media.zst")) == [
        "twin-2026-09-06.media.zst",
        "twin-2026-09-10.media.zst",
        "twin-2026-09-11.media.zst",
        "twin-2026-09-12.media.zst",
    ]
    states = {v.local_date: v.status for v in world.backup.ledger.recent(limit=50) if v.file_name}
    assert states["2026-09-01"] == "deleted" and states["2026-09-12"] == "ok"
    # the file that left the live store on 09-05 is in the 09-06 backup's list? no: it was gone
    # before that backup, and the older backups that had it are deleted: the pool lost it too
    assert not (world.backup.pool_dir / f"{gone_sha}.enc").exists()
    assert (world.backup.pool_dir / f"{next(s for s in world.media if s != gone_sha)}.enc").exists()


def test_a_backup_file_deleted_by_hand_is_noticed(world: BackupWorld, clock: ManualClock) -> None:
    first = world.backup.create("daily")
    (world.backup.backups_dir / str(first.file_name)).unlink()
    assert world.backup.reconcile() == 0
    assert world.backup.ledger.usable() == []  # the record knows the file is gone
    assert world.backup.ledger.newest_ok() is None


# ----------------------------------------------------------------------------- keys


def rotate(world: BackupWorld, *, media: bool = True) -> None:
    s = world.services
    rotate_db_key(s.db, s.keystore, s.keyring, s.clock, s.media if media else None)


def test_a_backup_before_a_rotation_keeps_its_key_and_restores_afterwards(
    world: BackupWorld, clock: ManualClock
) -> None:
    services = world.services
    clock.set_time(on_day(0))
    old = world.backup.create("daily")
    old_path = world.backup.backups_dir / str(old.file_name)
    assert read_header(old_path).key_id == 1
    rotate(world)
    assert services.keyring.current_id == 2 and services.keyring.retired_ids == frozenset({1})
    clock.set_time(on_day(1))
    new = world.backup.create("daily")
    new_path = world.backup.backups_dir / str(new.file_name)
    assert read_header(new_path).key_id == 2
    assert read_manifest(new_path, services.keyring).key_ids == [2]
    # both backups still open: the retired key is kept as long as the old backup is
    assert world.backup.verify(old_path).ok and world.backup.verify(new_path).ok
    assert read_manifest(old_path, services.keyring).key_ids == [1]
    assert services.secrets.exists(key_secret_name(1))
    assert world.backup.release_retired_keys() == []


def test_a_retired_key_is_deleted_when_the_last_backup_that_needs_it_is_gone(
    world: BackupWorld, clock: ManualClock
) -> None:
    services = world.services
    config = services.settings.ops
    config.backup_keep_daily, config.backup_keep_weekly = 2, 0
    clock.set_time(on_day(0))
    old = world.backup.create("daily")
    old_path = world.backup.backups_dir / str(old.file_name)
    rotate(world)
    clock.set_time(on_day(1))
    world.backup.create("daily")  # two days kept: the old backup stays, so does its key
    assert services.secrets.exists(key_secret_name(1)) and old_path.exists()
    clock.set_time(on_day(2))
    world.backup.create("daily")  # the old backup falls out, nothing references key 1 any more
    assert not old_path.exists()
    assert not services.secrets.exists(key_secret_name(1))
    assert services.keystore.load().key_ids == (2,) and services.keyring.key_ids == (2,)
    assert services.secrets.exists(key_secret_name(2))


def test_live_data_and_kept_backups_on_a_retired_key_keep_the_key(
    world: BackupWorld, clock: ManualClock
) -> None:
    services = world.services
    services.settings.ops.backup_keep_daily = 1
    services.settings.ops.backup_keep_weekly = 0
    clock.set_time(on_day(0))
    old = world.backup.create("daily")
    rotate(world, media=False)  # the media files stay on key 1
    assert services.keyring.retired_ids == frozenset({1})
    clock.set_time(on_day(1))
    world.backup.create("daily")  # the old backup is dropped, but the media files need key 1
    assert not (world.backup.backups_dir / str(old.file_name)).exists()
    assert services.secrets.exists(key_secret_name(1)) and 1 in world.backup.live_key_ids()
    services.media.rekey_all()  # now nothing live is on key 1 - but the kept backup lists it
    assert 1 not in world.backup.live_key_ids()
    assert world.backup.release_retired_keys() == []
    clock.set_time(on_day(2))
    world.backup.create("daily")  # the next backup lists the media under key 2 and replaces it
    assert not services.secrets.exists(key_secret_name(1))
    assert services.keyring.key_ids == (2,)


def test_a_backup_made_between_a_rotation_and_the_media_rekey_lists_both_keys(
    world: BackupWorld,
) -> None:
    rotate(world, media=False)
    view = world.backup.create("daily")
    manifest = read_manifest(world.backup.backups_dir / str(view.file_name), world.services.keyring)
    assert manifest.key_id == 2 and manifest.key_ids == [1, 2]
    assert {e.key_id for e in manifest.media} == {1}
    assert world.backup.ledger.kept_key_ids() == {1, 2}


def test_the_key_of_a_backup_is_derived_from_the_database_key(world: BackupWorld) -> None:
    """Without a database key there is no backup key: what purge relies on."""
    view = world.backup.create("daily")
    path = world.backup.backups_dir / str(view.file_name)
    other = KeyRing({1: generate_key()}, 1)  # a new installation: key id 1, other bytes
    with pytest.raises(BackupDecryptionError):
        read_manifest(path, other)


# ----------------------------------------------------------------------------- off-site copy


def test_the_off_site_folder_follows_the_local_policy(
    world: BackupWorld, clock: ManualClock, tmp_path: Path
) -> None:
    mirror = tmp_path / "usb"
    mirror.mkdir()
    config = world.services.settings.ops
    config.backup_mirror_dir = str(mirror)
    config.backup_keep_daily, config.backup_keep_weekly = 2, 0
    clock.set_time(on_day(0))
    first = world.backup.create("daily")
    assert sorted(p.name for p in mirror.glob("*.bak.enc")) == [str(first.file_name)]
    assert (mirror / sidecar_name(str(first.file_name))).is_file()
    assert {p.stem for p in (mirror / "media-pool").glob("*.enc")} == set(world.media)
    assert world.backup.ledger.usable()[0].mirrored_at is not None
    for offset in (1, 2):
        clock.set_time(on_day(offset))
        world.backup.create("daily")
    assert sorted(p.name for p in mirror.glob("*.bak.enc")) == [
        "twin-2026-09-02.bak.enc",
        "twin-2026-09-03.bak.enc",
    ]
    assert not (mirror / "twin-2026-09-01.bak.enc").exists()
    # the mirror's copy is the same file
    assert sha256_of(mirror / "twin-2026-09-03.bak.enc") == sha256_of(
        world.backup.backups_dir / "twin-2026-09-03.bak.enc"
    )
    assert world.backup.verify(mirror / "twin-2026-09-03.bak.enc").media_missing == 0


def test_a_missing_off_site_folder_is_an_alert_and_never_a_failed_backup(
    world: BackupWorld, tmp_path: Path
) -> None:
    config = world.services.settings.ops
    config.backup_mirror_dir = str(tmp_path / "unplugged")
    view = world.backup.create("daily")
    assert view.status == "ok" and (world.backup.backups_dir / str(view.file_name)).is_file()
    (alert,) = world.services.alerts.open_alerts()
    assert alert.category == "backup_mirror_unavailable" and alert.severity == "warning"
    assert world.backup.ledger.usable()[0].mirrored_at is None
    assert not (tmp_path / "unplugged").exists()  # a missing drive is not created
    (tmp_path / "unplugged").mkdir()  # plugged in again
    world.backup.sync_mirror()
    assert world.services.alerts.open_alerts() == []
    assert (tmp_path / "unplugged" / str(view.file_name)).is_file()
    assert world.backup.ledger.usable()[0].mirrored_at is not None


def test_no_mirror_configured_means_nothing_to_do(world: BackupWorld) -> None:
    assert world.backup.sync_mirror() is None


def test_pre_restore_backups_stay_on_this_machine(
    world: BackupWorld, clock: ManualClock, tmp_path: Path
) -> None:
    mirror = tmp_path / "usb"
    mirror.mkdir()
    daily = world.backup.backups_dir / "twin-2026-10-09.bak.enc"
    world.backup.create("daily")
    restore = world.backup.backups_dir / "pre-restore-20261009T120000Z.bak.enc"
    restore.write_bytes(daily.read_bytes())
    report = sync_mirror(
        mirror,
        world.backup.backups_dir,
        world.backup.pool_dir,
        keep_archives={daily.name, restore.name},
        pool_shas=set(world.media),
    )
    assert report.copied >= 3 and not (mirror / restore.name).exists()
    assert (mirror / daily.name).is_file()


def test_the_mirror_check_names_why_a_folder_cannot_be_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(MirrorUnavailableError) as missing:
        check_mirror(tmp_path / "nowhere")
    assert missing.value.code == "missing"
    folder = tmp_path / "drive"
    folder.mkdir()
    check_mirror(folder)
    assert list(folder.iterdir()) == []  # the probe file is removed again

    def refuse(self: Path, data: bytes) -> int:
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "write_bytes", refuse)
    with pytest.raises(MirrorUnavailableError) as readonly:
        check_mirror(folder)
    assert readonly.value.code == "not_writable"


def test_a_copy_that_fails_half_way_is_reported_and_leaves_no_partial_file(
    world: BackupWorld, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror = tmp_path / "usb"
    mirror.mkdir()
    world.backup.create("daily")

    def broken(source: object, target: object) -> None:
        raise OSError("device removed")

    monkeypatch.setattr("twin.ops.backup.mirror.shutil.copyfile", broken)
    with pytest.raises(MirrorUnavailableError) as failed:
        sync_mirror(
            mirror,
            world.backup.backups_dir,
            world.backup.pool_dir,
            keep_archives={"twin-2026-10-09.bak.enc"},
            pool_shas=set(world.media),
        )
    assert failed.value.code == "copy_failed"
    assert not list(mirror.glob("*.part")) and not list(mirror.glob("*.bak.enc"))


# ----------------------------------------------------------------------------- the records


def test_the_records_are_rebuilt_from_the_files(world: BackupWorld, clock: ManualClock) -> None:
    services = world.services
    clock.set_time(on_day(0))
    first = world.backup.create("daily")
    clock.set_time(on_day(1))
    second = world.backup.create("daily")
    from twin.storage.ops_models import BackupRecord

    with services.db.transaction() as session:
        session.execute(delete(BackupRecord))  # as after a restore of an older database
    assert world.backup.ledger.usable() == []
    stranger = world.backup.backups_dir / "twin-2020-01-01.bak.enc"
    stranger.write_bytes(os.urandom(500))  # another installation's file, or a damaged one
    assert world.backup.reconcile() == 2
    usable = {v.file_name: v for v in world.backup.ledger.usable()}
    assert set(usable) == {str(first.file_name), str(second.file_name)}
    assert usable[str(second.file_name)].sha256 == sha256_of(
        world.backup.backups_dir / str(second.file_name)
    )
    assert (
        usable[str(first.file_name)].key_ids == (1,) and usable[str(first.file_name)].row_count > 0
    )
    assert world.backup.reconcile() == 0  # nothing more to adopt
    assert stranger.exists()  # never deleted, never adopted


def test_the_manifest_is_refused_when_it_is_not_one() -> None:
    with pytest.raises(ArchiveError, match="unreadable"):
        Manifest.from_bytes(b"{not json")
    with pytest.raises(ArchiveError, match="unsupported"):
        Manifest.from_bytes(b'{"format": 99}')
    with pytest.raises(ArchiveError, match="unreadable"):
        Manifest.from_bytes(b'{"format": 1}')
    manifest = Manifest(
        "2026-10-09T12:00:00+00:00", "2026-10-09", "daily", "rev", 2, [1, 2], {"a": 3}, [], "ab"
    )
    again = Manifest.from_bytes(manifest.to_bytes())
    assert again == manifest and again.row_count == 3
    assert MediaEntry.from_json(MediaEntry("aa", 5, 2).to_json()) == MediaEntry("aa", 5, 2)


def test_the_database_of_the_installation_is_not_changed_by_a_backup(world: BackupWorld) -> None:
    before = table_counts(world.services.paths.db_path)
    world.backup.create("daily")
    after = table_counts(world.services.paths.db_path)
    after["backup_records"] -= 1  # the record of the backup itself
    assert after == before
    with world.services.db.session() as session:
        assert session.scalars(select(Job)).all()
