"""CLI: ``twin stickers download`` (R-IMP-008)."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CommandKind, app_is_running, command
from twin.services import Services, get_cli_context
from twin.stickers.download import STICKER_JOB, handle_sticker_download, queue_sticker_download
from twin.storage.chat_models import Sticker

stickers_app = typer.Typer(help="The sticker library.", no_args_is_help=True)


def sticker_counts(services: Services) -> dict[str, int]:
    """Stickers per status."""
    with services.db.session() as session:
        rows = session.execute(select(Sticker.status, func.count()).group_by(Sticker.status)).all()
    return {status: int(count) for status, count in rows}


@stickers_app.command("download")
@command(CommandKind.HEAVY)
def stickers_download(
    retry_failed: Annotated[
        bool, typer.Option("--retry-failed", help="Also try the stickers marked unavailable again")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run here when the application is stopped")
    ] = False,
) -> None:
    """Download the stickers whose files are not in the export (queued as a job)."""
    services = get_cli_context().services()
    queued = queue_sticker_download(services, retry_failed=retry_failed)
    if queued.job_id is None:
        typer.echo("no sticker is waiting for a download")
        return
    verb = "a download job is already waiting" if queued.already_queued else "queued a download job"
    typer.echo(f"{verb} for {queued.pending} sticker(s) (job {queued.job_id})")
    if not foreground:
        typer.echo("the running application executes it; `twin jobs list --type sticker_download`")
        return
    if app_is_running(services):
        typer.echo("the application is running and will execute the job")
        return
    registry = HandlerRegistry()
    registry.register(STICKER_JOB, handle_sticker_download)
    asyncio.run(run_jobs_until_idle(services, registry))
    table = Table("status", "stickers")
    for status, count in sorted(sticker_counts(services).items()):
        table.add_row(status, str(count))
    Console(highlight=False).print(table)
