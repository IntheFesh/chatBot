"""CLI: ``twin backup now|list|verify|restore`` (R-OPS-006).

``now``
    LIGHT: a backup right now (``twin-<local date>.bak.enc``; replaces the day's earlier one), then
    the retention and the off-site copy as for the daily one.
``list``
    READ: the records, newest first - failures and deletions too - and whether the file is there.
``verify <file>``
    READ: decrypt the whole backup, check every byte, the database inside it (SQLite's own check)
    and that the media files it lists exist.  Exit code 1 when anything is wrong.
``restore <file>``
    EXCLUSIVE: replace the data with the backup's.  The current data is backed up first
    (``pre-restore-<time>.bak.enc``), the migrations and the integrity checks run afterwards.
    ``<file>`` is a path, or the name of a file in ``data/backups``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.table import Table

from twin.clock import get_clock
from twin.ops.backup.archive import ArchiveError
from twin.ops.backup.restore import RestoreError, restore_archive
from twin.ops.backup.service import BackupBusyError, BackupError, BackupService
from twin.ops.integrity import run_integrity
from twin.ops.process_model import CliError, CommandKind, ExitCode, command
from twin.retrieval.vector_store import VectorStore
from twin.services import get_cli_context
from twin.storage.crypto import set_active_keyring
from twin.storage.keystore import KeyStore
from twin.storage.media import MediaStore

backup_app = typer.Typer(help="Encrypted backups of the bot's data.", no_args_is_help=True)


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _megabytes(size: int) -> str:
    return f"{size / 1e6:.1f} MB"


def _resolve(service: BackupService, name: str) -> Path:
    given = Path(name).expanduser()
    for candidate in (given, service.backups_dir / name):
        if candidate.is_file():
            return candidate
    raise CliError(f"there is no backup file {name!r} (looked in {service.backups_dir})")


@backup_app.command("now")
@command(CommandKind.LIGHT)
def backup_now() -> None:
    """Make a backup now."""
    services = get_cli_context().services()
    service = BackupService.from_services(services)
    typer.echo("backing up (the database is copied while the application keeps running) ...")
    try:
        view = service.create("manual")
    except BackupBusyError as exc:
        raise CliError(str(exc), ExitCode.BUSY) from exc
    except BackupError as exc:
        raise CliError(str(exc)) from exc
    typer.echo(f"{view.file_name}: {_megabytes(view.size_bytes)}, {view.row_count} rows,")
    typer.echo(
        f"  {view.media_count} media files, sealed with key {view.key_id}, "
        f"{view.duration_ms / 1000:.1f} s"
    )
    typer.echo(f"  in {service.backups_dir}")
    typer.echo("  check it with: twin backup verify " + str(view.file_name))


@backup_app.command("list")
@command(CommandKind.READ)
def backup_list() -> None:
    """The backups, newest first."""
    services = get_cli_context().services()
    service = BackupService.from_services(services)
    table = Table("date", "kind", "status", "file", "size", "rows", "media", "key", "off-site")
    for view in service.ledger.recent(limit=40):
        present = (
            "yes" if view.file_name and (service.backups_dir / view.file_name).is_file() else "-"
        )
        table.add_row(
            view.local_date,
            view.kind,
            view.status if view.status != "failed" else f"failed ({view.error})",
            f"{view.file_name or '-'} ({present})",
            _megabytes(view.size_bytes) if view.size_bytes else "-",
            str(view.row_count),
            str(view.media_count),
            ",".join(str(k) for k in view.key_ids) or "-",
            "yes" if view.mirrored_at else "-",
        )
    _console().print(table)
    newest = service.ledger.newest_ok()
    typer.echo(
        f"newest good backup: {newest.created_at:%Y-%m-%d %H:%M} UTC"
        if newest
        else "no good backup yet: run `twin backup now`"
    )


@backup_app.command("verify")
@command(CommandKind.READ)
def backup_verify(
    file: Annotated[str, typer.Argument(help="Backup file (path, or name in data/backups)")],
) -> None:
    """Decrypt a backup completely and check what is in it."""
    services = get_cli_context().services()
    service = BackupService.from_services(services)
    path = _resolve(service, file)
    typer.echo(f"verifying {path.name} ({_megabytes(path.stat().st_size)}) ...")
    try:
        report = service.verify(path)
    except ArchiveError as exc:
        raise CliError(str(exc)) from exc
    manifest = report.manifest
    typer.echo(f"  sealed with key {manifest.key_id}; depends on keys {manifest.key_ids}")
    typer.echo(f"  made {manifest.created_at} (local date {manifest.local_date})")
    typer.echo(f"  database: schema {manifest.schema_revision}, {manifest.row_count} rows in")
    typer.echo(f"  {len(manifest.tables)} tables; vector index files: {report.vector_files}")
    typer.echo(f"  media files listed: {len(manifest.media)}, missing: {report.media_missing}")
    typer.echo(f"  sha256 {report.sha256}")
    if report.recorded_sha256 not in (None, report.sha256):
        typer.echo("  PROBLEM: the file differs from the hash recorded when it was made")
    for problem in report.database_problems:
        typer.echo(f"  PROBLEM: {problem}")
    if not report.ok:
        raise typer.Exit(1)
    typer.echo("OK: the backup is complete and decrypts with the keys in the credential store")


@backup_app.command("restore")
@command(CommandKind.EXCLUSIVE)
def backup_restore(
    file: Annotated[str, typer.Argument(help="Backup file (path, or name in data/backups)")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
) -> None:
    """Replace the data with a backup (the application must be stopped)."""
    context = get_cli_context()
    settings = context.settings()
    paths = context.paths()
    candidate = Path(file).expanduser()
    path = candidate if candidate.is_file() else paths.backups_dir / file
    if not path.is_file():
        raise CliError(f"there is no backup file {file!r}")
    keystore = KeyStore(context.secret_store())
    ring = keystore.load()  # a restore never invents a key
    set_active_keyring(ring)
    typer.echo(f"restoring {path.name}: the current data is backed up first, then replaced")
    if not yes and not typer.confirm("Replace the current data with this backup?"):
        raise typer.Exit(1)
    clock = get_clock()  # the machine's clock; a test installs its own
    local_date = clock.now_utc().astimezone(ZoneInfo(settings.time.bot_timezone)).date().isoformat()
    context.reset()  # nothing of the old database may stay open
    try:
        report = restore_archive(
            path,
            paths=paths,
            ring=ring,
            clock=clock,
            media=MediaStore(paths.media_dir, paths.tmp_dir),
            local_date=local_date,
        )
    except RestoreError as exc:
        raise CliError(str(exc)) from exc
    typer.echo(f"restored {report.archive} (made {report.created_at}):")
    typer.echo(
        f"  {report.rows} rows; schema {report.schema_before} -> {report.schema_after}; "
        f"{report.sampled_values} encrypted values decrypted as a check"
    )
    typer.echo(
        f"  media: {report.media_listed} listed, {report.media_restored} copied back from the "
        f"pool, {report.media_missing} not available"
    )
    typer.echo(f"  the data that was there is in {report.pre_restore or '(there was none)'}")
    services = context.services()
    BackupService.from_services(services).reconcile()
    integrity = run_integrity(services.db, VectorStore(paths.vectors_dir), services.clock.now_utc())
    for problem in integrity.problems():
        typer.echo(f"  PROBLEM: {problem}")
    if not integrity.ok:
        raise typer.Exit(1)
    typer.echo("integrity check: clean")
