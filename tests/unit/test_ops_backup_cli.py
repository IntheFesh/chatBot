"""``twin backup now|list|verify|restore`` (R-OPS-006)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import delete
from typer.testing import CliRunner

from tests.support.backup_world import (
    BackupWorld,
    build_backup_world,
    fact_texts,
    table_counts,
)
from tests.support.cli_runner import invoke
from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from twin.clock import set_active_clock
from twin.config.secrets import SecretStore
from twin.ops.backup import cli as backup_cli
from twin.ops.backup.archive import read_manifest
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.process_model import CommandKind, ExitCode, get_spec
from twin.services import CliContext, Services, set_cli_context
from twin.storage.memory_models import Fact
from twin.storage.models import Job

runner = CliRunner()


@pytest.fixture
def world(services: Services, embedder: HashingBackend, secret_store: SecretStore) -> BackupWorld:
    set_cli_context(CliContext(secrets=secret_store))  # the root callback keeps the secret store
    return build_backup_world(services)


def twin(world: BackupWorld, *args: str, answer: str | None = None) -> tuple[int, str]:
    if args[0] == "restore":
        # Restoring replaces the database files, which the application must not be holding open
        # (R-ARCH-006: it is stopped).  The fixture's own container stands in for it here, and
        # Windows refuses to replace a file that a connection of this process still has open.
        world.services.db.dispose()
    options = ["--set", f"paths.data_dir={world.services.paths.data_dir}"]
    return invoke(runner, [*options, "backup", *args], answer=answer)


def make(world: BackupWorld) -> str:
    code, out = twin(world, "now")
    assert code == 0, out
    return next(line.split(":")[0] for line in out.splitlines() if ".bak.enc" in line)


def test_backup_now_makes_a_backup_and_says_how_to_check_it(world: BackupWorld) -> None:
    code, out = twin(world, "now")
    assert code == 0, out
    files = [p.name for p in world.services.paths.backups_dir.glob("*.bak.enc")]
    assert len(files) == 1
    assert files[0] in out and "media files" in out and "sealed with key 1" in out
    assert f"twin backup verify {files[0]}" in out
    assert "MB" in out and "rows" in out


def test_backup_now_refuses_while_another_backup_runs(world: BackupWorld) -> None:
    from twin.ops.filelock import FileLock

    lock = FileLock(world.services.paths.locks_dir / "backup.lock")
    assert lock.acquire(blocking=False)
    try:
        code, out = twin(world, "now")
    finally:
        lock.release()
    assert code == ExitCode.BUSY and "another backup" in out


def test_backup_list_shows_records_files_and_failures(world: BackupWorld) -> None:
    code, out = twin(world, "list")
    assert code == 0 and "no good backup yet" in out
    name = make(world)
    world.backup.ledger.add_failed("daily", "2026-10-08", "OSError", 5)
    code, out = twin(world, "list")
    assert code == 0 and name in out and "(yes)" in out
    assert "failed (OSError)" in out and "newest good backup" in out
    (world.services.paths.backups_dir / name).unlink()
    code, out = twin(world, "list")
    assert "(-)" in out  # the record is there, the file is not


def test_backup_verify_decrypts_everything_and_reports(world: BackupWorld) -> None:
    name = make(world)
    code, out = twin(world, "verify", name)
    assert code == 0, out
    assert "OK: the backup is complete" in out and "sealed with key 1" in out
    assert "media files listed: 2, missing: 0" in out and "sha256 " in out
    path = world.services.paths.backups_dir / name
    code2, out2 = twin(world, "verify", str(path))  # a path works too
    assert code2 == 0 and "OK" in out2


def test_backup_verify_fails_on_a_damaged_file_a_missing_file_and_a_missing_media_file(
    world: BackupWorld,
) -> None:
    name = make(world)
    path = world.services.paths.backups_dir / name
    sha = next(iter(world.media))
    world.services.media.path_for(sha).unlink()
    (world.backup.pool_dir / f"{sha}.enc").unlink()
    code, out = twin(world, "verify", name)
    assert code == 1 and "missing: 1" in out and "OK" not in out
    code, out = twin(world, "verify", "twin-1999-01-01.bak.enc")
    assert code != 0 and "there is no backup file" in out
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x10
    path.write_bytes(bytes(data))
    code, out = twin(world, "verify", name)
    assert code != 0 and "OK" not in out


def test_backup_restore_replaces_the_data_after_a_question(world: BackupWorld) -> None:
    services = world.services
    counts = table_counts(services.paths.db_path)
    texts = fact_texts(services)
    name = make(world)
    with services.db.transaction() as session:
        session.execute(delete(Fact))
        session.add(Job(type="later", payload={"n": 1}))
    changed = table_counts(services.paths.db_path)
    code, out = twin(world, "restore", name, answer="n\n")  # the answer is no
    assert code == 1 and table_counts(services.paths.db_path) == changed
    code, out = twin(world, "restore", name, answer="y\n")
    assert code == 0, out
    assert "the current data is backed up first" in out and "integrity check: clean" in out
    assert "pre-restore-" in out and "encrypted values decrypted as a check" in out
    restored = table_counts(services.paths.db_path)
    assert {k: v for k, v in restored.items() if k != "backup_records"} == {
        k: v for k, v in counts.items() if k != "backup_records"
    }
    services.db.dispose()  # this process' connections still point at the replaced file
    assert fact_texts(services) == texts
    # the records were made to agree with the files: the new database knows the pre-restore copy
    code, out = twin(world, "list")
    assert "pre_restore" in out


@pytest.mark.parametrize(
    ("zone", "local_day"), [("America/Chicago", "2026-11-01"), ("Asia/Shanghai", "2026-11-02")]
)
def test_the_copy_made_before_a_restore_is_dated_by_the_day_of_the_bots_zone(
    world: BackupWorld, clock: ManualClock, zone: str, local_day: str
) -> None:
    """21:30 on 1 November in Chicago is the 2nd in Shanghai: the same instant, two dates."""
    name = make(world)  # (a command line builds its own services, and with them the system clock)
    clock.set_time(datetime(2026, 11, 2, 3, 30, tzinfo=UTC))
    set_active_clock(clock)  # the machine's clock for the next command
    world.services.db.dispose()
    options = ["--set", f"paths.data_dir={world.services.paths.data_dir}"]
    options += ["--set", f"time.bot_timezone={zone}"]
    code, out = invoke(runner, [*options, "backup", "restore", name, "--yes"])
    assert code == 0, out
    [pre] = sorted(world.services.paths.backups_dir.glob("pre-restore-*.bak.enc"))
    assert pre.name == "pre-restore-20261102T033000Z.bak.enc"  # (named by the instant, in UTC)
    manifest = read_manifest(pre, world.services.keyring)
    assert manifest.local_date == local_day and manifest.kind == "pre_restore"
    assert manifest.created_at.startswith("2026-11-02T03:30:00")


def test_the_restore_is_an_exclusive_command_and_backup_now_only_writes() -> None:
    restore, now = get_spec(backup_cli.backup_restore), get_spec(backup_cli.backup_now)
    assert restore is not None and restore.kind is CommandKind.EXCLUSIVE
    assert now is not None and now.kind is not CommandKind.EXCLUSIVE


def test_restore_yes_skips_the_question_and_missing_files_are_refused(world: BackupWorld) -> None:
    name = make(world)
    code, out = twin(world, "restore", name, "--yes")
    assert code == 0, out
    code, out = twin(world, "restore", "nope.bak.enc", "--yes")
    assert code != 0 and "there is no backup file" in out


def test_restore_is_refused_while_the_application_runs(world: BackupWorld) -> None:
    name = make(world)
    lock = InstanceLock(LOCK_RUN, locks_dir=world.services.paths.locks_dir)
    assert lock.acquire()
    try:
        code, out = twin(world, "restore", name, "--yes")
    finally:
        lock.release()
    assert code == ExitCode.BUSY and "twin service stop" in out


def test_restore_reports_an_index_that_does_not_match_the_database(world: BackupWorld) -> None:
    """The vector files are copied while the application runs: they may drift from the database."""
    services = world.services
    with services.db.transaction() as session:
        session.execute(delete(Fact))  # the index still holds the six facts
    name = make(world)
    code, out = twin(world, "restore", name, "--yes")
    assert code == 1
    assert "PROBLEM: memory_facts: 6 id(s) not in the database; run twin memory reindex" in out
    assert "integrity check: clean" not in out


def test_restore_refuses_a_damaged_backup_and_changes_nothing(world: BackupWorld) -> None:
    name = make(world)
    path: Path = world.services.paths.backups_dir / name
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x04
    path.write_bytes(bytes(data))
    before = table_counts(world.services.paths.db_path)
    code, out = twin(world, "restore", name, "--yes")
    assert code != 0 and "cannot be decrypted" in out
    assert table_counts(world.services.paths.db_path) == before
    assert [p.name for p in path.parent.glob("pre-restore-*")] == []
