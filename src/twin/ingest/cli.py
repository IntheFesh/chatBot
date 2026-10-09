"""CLI: ``twin import`` (start, status, inspect) and ``twin images caption-backfill``.

``twin import <directory>`` is a HEAVY command (R-ARCH-006): it checks the export, lets the
person choose the target conversation the first time, records the run and queues the
``import`` job; the running application executes it.  ``--foreground`` runs it here when the
application is stopped, with a progress bar; ``--resume`` continues the last unfinished run.
``twin import status`` and ``twin import inspect`` only read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from typer.core import TyperGroup

from twin.clock import get_clock
from twin.config.mask import mask_value
from twin.config.runtime import TARGET_USERNAME
from twin.ingest.captions import plan_caption_batches
from twin.ingest.importer import ImportFailure, prepare_import, resumable_run
from twin.ingest.inspect import (
    DEFAULT_SAMPLE,
    inspect_export,
    render_inspect,
    write_inspect_report,
)
from twin.ingest.jobs import IMPORT_JOB, handle_import, queue_import
from twin.ingest.layout import ConversationEntry, ExportLayout, ExportLayoutError
from twin.ingest.runs import RunView, get_run, latest_run
from twin.ingest.schema import UnsupportedSchema
from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.services import Services, get_cli_context

DEFAULT_COMMAND = "start"
TERMINAL_STATUSES = ("done", "failed", "superseded")


class ImportGroup(TyperGroup):
    """``twin import <directory>`` works like ``twin import start <directory>``.

    A first argument that is not a sub-command name (a path), or that is an option such as
    ``--resume``, selects the default ``start`` command.  A directory that is literally
    called ``status`` or ``inspect`` must be written ``./status``.
    """

    def resolve_command(self, ctx: Any, args: list[str]) -> tuple[str | None, Any, list[str]]:
        if args and args[0] not in self.commands and args[0] not in ctx.help_option_names:
            return super().resolve_command(ctx, [DEFAULT_COMMAND, *args])
        return super().resolve_command(ctx, args)


import_app = typer.Typer(
    cls=ImportGroup,
    help="Import the chat records of an export (twin import <directory>).",
    no_args_is_help=True,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
images_app = typer.Typer(help="Picture descriptions.", no_args_is_help=True)


def _console() -> Console:
    return Console(highlight=False)


# ------------------------------------------------------------------- targets


def _describe(entry: ConversationEntry) -> str:
    count = entry.meta.messageCount
    return (
        f"{entry.meta.displayName or '(no name)'}  "
        f"{count if count is not None else '?'} messages  [{mask_value(entry.username)}]"
    )


def choose_target(
    services: Services, layout: ExportLayout, requested: str | None, *, assume_yes: bool
) -> str:
    """The wxid to import: the stored target, ``--target``, or the person's choice (R-IMP-003).

    Only the ``meta.json`` of each conversation is read.  The first choice is stored as the
    ``target.username`` runtime setting.
    """
    candidates = layout.direct_conversations()
    stored = services.runtime.get(TARGET_USERNAME)
    wanted = requested or stored
    if wanted:
        if wanted.isdigit() and requested and 1 <= int(wanted) <= len(candidates):
            return candidates[int(wanted) - 1].username
        for entry in candidates:
            if entry.username == wanted:
                return entry.username
        listing = "\n".join(f"  {i}. {_describe(e)}" for i, e in enumerate(candidates, start=1))
        raise CliError(
            "the target conversation is not in this export; one-to-one conversations here:\n"
            f"{listing}\nuse --target <number> or `twin settings set target.username <id>`",
            ExitCode.USAGE,
        )
    if not candidates:
        raise CliError("this export has no one-to-one conversation", ExitCode.USAGE)
    table = Table("#", "conversation", "messages", "id (masked)")
    for index, entry in enumerate(candidates, start=1):
        count = entry.meta.messageCount
        table.add_row(
            str(index),
            entry.meta.displayName or "-",
            str(count if count is not None else "?"),
            mask_value(entry.username),
        )
    _console().print(table)
    if len(candidates) == 1 and assume_yes:
        chosen = candidates[0]
    else:
        number = typer.prompt("Number of the conversation to imitate", type=int)
        if not 1 <= number <= len(candidates):
            raise CliError(f"choose a number from 1 to {len(candidates)}", ExitCode.USAGE)
        chosen = candidates[number - 1]
    services.runtime.set(TARGET_USERNAME, chosen.username, by="import")
    typer.echo("saved as the target conversation (twin settings set target.username to change)")
    return chosen.username


# ------------------------------------------------------------------- status


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_status(view: RunView) -> list[str]:
    """Plain-text status of a run: phase, progress, speed, ETA and the post-import hooks."""
    percent = ""
    if view.total:
        percent = f" ({100.0 * view.processed / view.total:.1f}%)"
    total = view.total if view.total is not None else "?"
    speed = f"{view.speed_per_s:,.0f} messages/s" if view.speed_per_s else "-"
    lines = [
        f"import run {view.id}: {view.status}, phase {view.phase}",
        f"  processed {view.processed:,} / {total}{percent}   speed {speed}   "
        f"remaining {_duration(view.eta_seconds)}",
        f"  new {view.inserted:,}  duplicates {view.duplicates:,}  "
        f"conflicts updated {view.conflict_updated:,} kept {view.conflict_kept:,}  "
        f"invalid {view.invalid:,}",
        f"  last update {view.updated_at:%Y-%m-%d %H:%M:%S} UTC",
    ]
    if view.error:
        lines.append(f"  error: {view.error}")
    if view.hooks:
        lines.append("  post-import hooks:")
        for name, entry in view.hooks.items():
            detail = f" - {entry.get('detail')}" if entry.get("detail") else ""
            lines.append(f"    {name}: {entry.get('status')}{detail}")
    if view.report_path:
        lines.append(f"  report: {view.report_path}")
    return lines


@import_app.command("status")
@command(CommandKind.READ)
def import_status(
    watch: Annotated[bool, typer.Option("--watch", help="Keep refreshing until it ends")] = False,
    interval: Annotated[float, typer.Option(help="Seconds between refreshes with --watch")] = 1.0,
    run_id: Annotated[str | None, typer.Option("--run", help="Show this run id")] = None,
) -> None:
    """Show the phase, progress, speed, remaining time and hook states of the latest import."""
    services = get_cli_context().services()

    def current() -> RunView | None:
        return get_run(services.db, run_id) if run_id else latest_run(services.db)

    view = current()
    if view is None:
        raise CliError("no import has been started yet; run `twin import <directory>`")
    if not watch:
        typer.echo("\n".join(format_status(view)))
        return

    async def loop() -> None:
        shown = view
        with Live(
            Text("\n".join(format_status(shown))), console=_console(), auto_refresh=False
        ) as live:
            while True:
                fresh = await asyncio.to_thread(current)
                if fresh is not None:
                    shown = fresh
                live.update(Text("\n".join(format_status(shown))), refresh=True)
                if shown.status in TERMINAL_STATUSES:
                    return
                await services.clock.sleep(interval)

    asyncio.run(loop())


# -------------------------------------------------------------------- inspect


@import_app.command("inspect")
@command(CommandKind.READ)
def import_inspect(
    export_dir: Annotated[Path | None, typer.Argument(help="Export directory")] = None,
    sample: Annotated[
        int,
        typer.Option(help="Messages looked at per messages.json (0 = all of them)"),
    ] = DEFAULT_SAMPLE,
) -> None:
    """Describe the structure of an export without printing any value (R-IMP-014)."""
    context = get_cli_context()
    paths = context.paths()
    directory = export_dir or paths.export_dir
    if directory is None:
        raise CliError("give the export directory or set paths.export_dir", ExitCode.USAGE)
    try:
        result = inspect_export(directory, sample=sample)
    except ExportLayoutError as exc:
        raise CliError(str(exc)) from exc
    now = get_clock().now_utc()
    text = render_inspect(result, now)
    path = write_inspect_report(paths.reports_dir, text, now)
    typer.echo(text)
    typer.echo(f"structure report written to {path}")


# ---------------------------------------------------------------------- start


def _export_directory(services: Services, given: Path | None) -> Path:
    directory = given or services.paths.export_dir
    if directory is None:
        raise CliError(
            "give the export directory (twin import <directory>) or set paths.export_dir",
            ExitCode.USAGE,
        )
    return directory


async def _foreground(services: Services, run_id: str) -> None:
    registry = HandlerRegistry()
    registry.register(IMPORT_JOB, handle_import)
    console = _console()
    with Progress(
        TextColumn("[bold]import[/bold] {task.fields[phase]}"),
        BarColumn(),
        TextColumn("{task.completed:,.0f}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task: TaskID = progress.add_task("import", total=None, phase="starting")

        async def tick() -> None:
            view = await asyncio.to_thread(get_run, services.db, run_id)
            if view is not None:
                progress.update(task, completed=view.processed, total=view.total, phase=view.phase)

        summary = await run_jobs_until_idle(services, registry, on_tick=tick)
    if summary.failed or summary.retried:
        failures = f"failed={summary.failed} retried={summary.retried}"
        typer.echo(f"the import job did not finish: {failures}")


@import_app.command(DEFAULT_COMMAND)
@command(CommandKind.HEAVY)
def import_start(
    export_dir: Annotated[Path | None, typer.Argument(help="Export directory")] = None,
    foreground: Annotated[
        bool,
        typer.Option("--foreground", help="Run here, with a progress bar (application stopped)"),
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue the last unfinished import")
    ] = False,
    target: Annotated[
        str | None, typer.Option("--target", help="Conversation number or id (this run only)")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Choose the only candidate without asking")
    ] = False,
) -> None:
    """Import an export: queue the import job (default) or run it here (--foreground)."""
    services = get_cli_context().services()
    if resume:
        unfinished = resumable_run(services)
        if unfinished is None:
            raise CliError("there is no unfinished import to resume")
        run_id = unfinished.id
        typer.echo(f"resuming import run {run_id} at {unfinished.processed:,} messages")
    else:
        directory = _export_directory(services, export_dir)
        try:
            layout = ExportLayout(directory)
            layout.validate()
            username = choose_target(services, layout, target, assume_yes=yes)
            prepared = prepare_import(services, directory, target_username=username)
        except (ExportLayoutError, UnsupportedSchema, ImportFailure) as exc:
            raise CliError(str(exc)) from exc
        run_id = prepared.run_id
        if prepared.resumed:
            typer.echo(f"continuing the unfinished run {run_id} of the same export files")
    job_id = queue_import(services, run_id)
    if not foreground:
        typer.echo(
            f"queued import run {run_id} (job {job_id}); the running application executes it"
        )
        typer.echo("check progress with `twin import status` (or `twin import status --watch`)")
        return
    if app_is_running(services):
        typer.echo(
            "the application is running and will execute the import; see `twin import status`"
        )
        return
    asyncio.run(_foreground(services, run_id))
    view = get_run(services.db, run_id)
    if view is None or view.status != "done":
        detail = view.error if view and view.error else "see `twin import status`"
        raise CliError(f"the import did not finish: {detail}")
    typer.echo("\n".join(format_status(view)))
    if view.report_path:
        typer.echo("")
        typer.echo(Path(view.report_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- caption backfill


@images_app.command("caption-backfill")
@command(CommandKind.HEAVY)
def images_caption_backfill(
    days: Annotated[
        int | None, typer.Option("--days", min=1, help="Pictures of the last N days")
    ] = None,
) -> None:
    """Queue picture descriptions as a one-time batch; approve it with `twin jobs approve`."""
    services = get_cli_context().services()
    window = days or services.settings.ingest.caption_recent_days
    plan = plan_caption_batches(services, days=window)
    if plan.images == 0:
        reason = f"; {plan.already_queued} already queued" if plan.already_queued else ""
        typer.echo(f"no undescribed pictures in the last {window} days{reason}")
        return
    table = Table("batch", "jobs", "pictures", "estimated cost (USD)")
    for batch in plan.batches:
        table.add_row(
            batch.batch_id, str(batch.jobs), str(batch.images), f"{batch.estimated_usd:.2f}"
        )
    _console().print(table)
    typer.echo(
        f"{plan.images} picture(s) queued, estimated ${plan.estimated_usd:.2f} in total "
        f"(one-time limit ${services.settings.budget.one_time_usd:.2f} per batch)"
    )
    for batch in plan.batches:
        typer.echo(f"approve with: twin jobs approve {batch.batch_id}")
