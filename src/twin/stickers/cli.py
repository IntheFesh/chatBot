"""CLI: ``twin stickers`` - download, list, show, tag, untag, disable, enable, tag-all.

``download`` and ``tag-all`` are HEAVY (they queue jobs).  ``tag-all`` is the backfill command
of the tagging hook: it plans the tagging of every sticker that still needs it as a one-time
batch, prints the estimated cost and waits for ``twin jobs approve <batch>`` (R-LLM-014).
``list`` and ``show`` only read; ``tag``, ``untag``, ``disable`` and ``enable`` are LIGHT.
Hand-set tags win over everything the models found (R-STK-003).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.services import Services, get_cli_context
from twin.stickers.catalog import StickerCatalog, StickerRecord, UnknownStickerError
from twin.stickers.download import STICKER_JOB, handle_sticker_download, queue_sticker_download
from twin.stickers.tag_jobs import plan_tagging
from twin.stickers.tags import TagError
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


# ---------------------------------------------------------------------- the library


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _resolve(catalog: StickerCatalog, md5: str) -> StickerRecord:
    try:
        return catalog.resolve(md5)
    except UnknownStickerError as exc:
        raise CliError(str(exc)) from exc


def _shorten(text: str | None, limit: int = 24) -> str:
    if not text:
        return "-"
    return text if len(text) <= limit else text[: limit - 1] + "…"


@stickers_app.command("list")
@command(CommandKind.READ)
def stickers_list(
    untagged: Annotated[
        bool, typer.Option("--untagged", help="Only stickers without tags")
    ] = False,
    tag: Annotated[str | None, typer.Option("--tag", help="Only stickers with this tag")] = None,
    limit: Annotated[int, typer.Option(min=1, help="Most used stickers to list")] = 30,
) -> None:
    """List the library, her most used stickers first, with the library statistics."""
    services = get_cli_context().services()
    catalog = StickerCatalog(services)
    counts = catalog.counts()
    typer.echo(
        f"{counts['total']} sticker(s): {counts['available']} usable files, "
        f"{counts['hers']} used by her, {counts['tagged']} tagged "
        f"({counts['manual']} by hand, {counts['context_corrected']} corrected from her use), "
        f"{counts['with_vector']} with a description vector, {counts['disabled']} disabled"
    )
    records = catalog.records(tagged=False if untagged else None, limit=None if tag else limit)
    if tag is not None:
        records = [r for r in records if tag in r.tags][:limit]
    table = Table("md5", "status", "her", "user", "tags", "source", "description")
    for record in records:
        flags = " (off)" if record.disabled else ""
        table.add_row(
            record.md5[:8],
            record.status + flags,
            str(record.her_uses),
            str(record.user_uses),
            "、".join(record.tags) or "-",
            record.tag_source or "-",
            _shorten(record.description),
        )
    _console().print(table)


@stickers_app.command("show")
@command(CommandKind.READ)
def stickers_show(md5: Annotated[str, typer.Argument(help="MD5 or its first characters")]) -> None:
    """Show everything the library knows about one sticker."""
    services = get_cli_context().services()
    catalog = StickerCatalog(services)
    record = _resolve(catalog, md5)
    last = f"{record.last_used_at:%Y-%m-%d}" if record.last_used_at else "-"
    size = f"{record.width or '?'}x{record.height or '?'}"
    lines = [
        f"md5          {record.md5}",
        f"file         {record.status}, {record.mime or '-'}, {size}; origin {record.origin}"
        + ("; switched off" if record.disabled else ""),
        f"uses         her {record.her_uses}, user {record.user_uses}, her last use {last}",
        f"tags         {'、'.join(record.tags) or '-'} (decided by: {record.tag_source or '-'})",
        f"  picture    {'、'.join(record.vision_tags) or '-'}",
        f"  her use    {'、'.join(record.context_tags) or '-'}"
        + (f" ({record.context_uses} uses before the cutoff)" if record.context_tagged_at else ""),
        f"  by hand    {'、'.join(record.manual_tags) or '-'}",
        f"description  {record.description or '-'}",
        f"use cases    {record.use_cases or '-'}",
        f"her meaning  {record.context_note or '-'}",
        f"vector       {'yes' if record.desc_vector_id else 'no'}",
        f"windows      {len(set(catalog.use_windows(record.md5).values()))} example window(s) "
        "hold her use of it",
    ]
    console = _console()
    for line in lines:
        console.print(line, markup=False)


@stickers_app.command("tag")
@command(CommandKind.LIGHT)
def stickers_tag(
    md5: Annotated[str, typer.Argument(help="MD5 or its first characters")],
    tags: Annotated[list[str], typer.Argument(help="One to three tags of the vocabulary")],
) -> None:
    """Set the tags of a sticker by hand (they win over the automatic ones)."""
    services = get_cli_context().services()
    catalog = StickerCatalog(services)
    record = _resolve(catalog, md5)
    try:
        changed = catalog.set_manual(record.md5, tags)
    except (TagError, ValueError) as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    typer.echo(f"{changed.md5[:8]}: tags are now {'、'.join(changed.tags)} (by hand)")


@stickers_app.command("untag")
@command(CommandKind.LIGHT)
def stickers_untag(md5: Annotated[str, typer.Argument(help="MD5 or its first characters")]) -> None:
    """Remove the hand-set tags; the automatic ones apply again."""
    services = get_cli_context().services()
    catalog = StickerCatalog(services)
    changed = catalog.clear_manual(_resolve(catalog, md5).md5)
    typer.echo(f"{changed.md5[:8]}: tags are now {'、'.join(changed.tags) or '(none)'}")


def _switch(md5: str, disabled: bool) -> None:
    services = get_cli_context().services()
    catalog = StickerCatalog(services)
    changed = catalog.set_disabled(_resolve(catalog, md5).md5, disabled)
    typer.echo(f"{changed.md5[:8]}: {'disabled' if disabled else 'enabled'}")


@stickers_app.command("disable")
@command(CommandKind.LIGHT)
def stickers_disable(
    md5: Annotated[str, typer.Argument(help="MD5 or its first characters")],
) -> None:
    """Never choose this sticker (it stays in the library)."""
    _switch(md5, True)


@stickers_app.command("enable")
@command(CommandKind.LIGHT)
def stickers_enable(
    md5: Annotated[str, typer.Argument(help="MD5 or its first characters")],
) -> None:
    """Allow a disabled sticker again."""
    _switch(md5, False)


@stickers_app.command("tag-all")
@command(CommandKind.HEAVY)
def stickers_tag_all() -> None:
    """Queue the tagging of every sticker that needs it; the price waits for your approval."""
    services = get_cli_context().services()
    plan = plan_tagging(services, batch=True)
    if plan.mode == "none":
        reason = f" ({plan.already_queued} already queued)" if plan.already_queued else ""
        typer.echo(f"every sticker is tagged and described{reason}")
        return
    typer.echo(
        f"{plan.stickers} sticker(s) to tag ({plan.contexts} also to be corrected from her use) "
        f"in {plan.jobs} job(s), estimated ${plan.estimated_usd:.2f} (an upper bound)"
    )
    for batch_id in plan.batch_ids:
        typer.echo(f"approve with: twin jobs approve {batch_id}")
    typer.echo(
        "after the approval the running application tags them; or `twin jobs run --until-idle`"
    )
