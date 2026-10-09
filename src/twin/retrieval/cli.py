"""CLI: ``twin retrieval`` (R-RET-003, R-RET-006, R-IMP-011).

``rebuild`` and ``resplit`` are HEAVY (they queue the index job; ``--foreground`` runs it here
when the application is stopped, with a progress bar and the time left).  ``stats`` only reads.
The commands print counts, times and model names - never what she wrote.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn
from rich.table import Table

from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.profile.holdout import HoldoutError, resplit_holdout
from twin.profile.jobs import handle_profile_rebuild
from twin.profile.queue import PROFILE_JOB
from twin.retrieval.embedder import MODEL_PROFILES, cpu_speed_warning
from twin.retrieval.indexer import (
    INDEX_JOB,
    IndexStats,
    collect_stats,
    get_progress,
)
from twin.retrieval.jobs import handle_retrieval_index
from twin.retrieval.queue import queue_retrieval_index
from twin.services import Services, get_cli_context

retrieval_app = typer.Typer(
    help="The library of her real replies for retrieval.", no_args_is_help=True
)


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


async def _run_in_foreground(services: Services, registry: HandlerRegistry) -> None:
    console = _console()
    with Progress(
        TextColumn("[bold]retrieval[/bold]"),
        BarColumn(),
        TextColumn("{task.completed:,.0f}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task: TaskID = progress.add_task("index", total=None)

        async def tick() -> None:
            view = await asyncio.to_thread(get_progress, services)
            if view is not None and view.state == "running":
                progress.update(task, completed=view.done, total=view.total)

        summary = await run_jobs_until_idle(services, registry, on_tick=tick)
    if summary.failed or summary.retried:
        typer.echo(
            f"the job did not finish: failed={summary.failed} retried={summary.retried}; "
            "see `twin jobs list`"
        )


def _foreground_or_hint(services: Services, registry: HandlerRegistry) -> None:
    if app_is_running(services):
        typer.echo("the application is running and will execute the job(s)")
        return
    asyncio.run(_run_in_foreground(services, registry))


# ------------------------------------------------------------------- rebuild


@retrieval_app.command("rebuild")
@command(CommandKind.HEAVY)
def retrieval_rebuild(
    full: Annotated[
        bool, typer.Option("--full", help="Encode every window again, even if nothing changed")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run here when the application is stopped")
    ] = False,
) -> None:
    """Build or repair the library of her real replies (queued as a job; resumes if stopped)."""
    services = get_cli_context().services()
    config = services.settings.retrieval
    warning = cpu_speed_warning(config, "cpu" if config.device != "cuda" else "cuda", None)
    if warning:
        typer.echo(f"note: {warning}")
    queued = queue_retrieval_index(services, mode="rebuild", full=full, reason="manual")
    verb = "an index run is already waiting" if queued.already_queued else "queued"
    typer.echo(f"{verb} (job {queued.job_id}{', full' if full else ''})")
    if not foreground:
        typer.echo("the running application executes it; see progress with `twin retrieval stats`")
        return
    registry = HandlerRegistry()
    registry.register(INDEX_JOB, handle_retrieval_index)
    _foreground_or_hint(services, registry)
    _print_stats(collect_stats(services))


# --------------------------------------------------------------------- stats


def _print_stats(stats: IndexStats) -> None:
    table = Table("item", "value", show_header=False)
    table.add_row("windows (her reply blocks)", f"{stats.windows:,}")
    table.add_row("  with a context", f"{stats.with_context:,}")
    table.add_row("  without context (she spoke first)", f"{stats.without_context:,}")
    table.add_row("  replies of events only", f"{stats.event_only:,}")
    table.add_row("held out (evaluation set)", f"{stats.held_out:,}")
    cutoff = f"{stats.cutoff:%Y-%m-%d %H:%M} UTC" if stats.cutoff else "not computed yet"
    table.add_row("hold-out cutoff", cutoff)
    if stats.first_reply and stats.last_reply:
        table.add_row(
            "replies from / to", f"{stats.first_reply:%Y-%m-%d} / {stats.last_reply:%Y-%m-%d}"
        )
    table.add_row("windows with a vector", f"{stats.indexed:,}")
    table.add_row("waiting for encoding", f"{stats.awaiting:,}")
    table.add_row("nothing to encode", f"{stats.no_vector:,}")
    table.add_row("vectors in the index", f"{stats.vectors:,}")
    if stats.meta is not None:
        table.add_row("index model", f"{stats.meta.model} ({stats.meta.dimension} dimensions)")
        table.add_row("index weights", stats.meta.weights_sha256[:16] or "unknown")
        table.add_row("index updated", stats.meta.written_at[:19])
    table.add_row("configured model", stats.configured_model)
    progress = stats.progress
    if progress is not None:
        line = f"{progress.state}: {progress.done:,}/{progress.total:,}"
        if progress.rate_per_s:
            line += f", {1000 / progress.rate_per_s:,.0f} s per 1,000 windows"
        if progress.state == "running":
            line += f", about {_duration(progress.eta_s)} left"
        table.add_row("last index run", line)
    _console().print(table)
    for problem in stats.problems:
        typer.echo(f"problem: {problem}")


@retrieval_app.command("stats")
@command(CommandKind.READ)
def retrieval_stats() -> None:
    """Show the size of the library, the hold-out, the model and the progress of the last run."""
    services = get_cli_context().services()
    stats = collect_stats(services)
    if stats.windows == 0:
        typer.echo("the library is empty; run `twin retrieval rebuild` after importing messages")
        return
    _print_stats(stats)
    meta = stats.meta
    profile = MODEL_PROFILES.get(meta.model if meta else stats.configured_model)
    if profile is not None and profile.relative_cost >= 5:
        typer.echo("note: this model is slow on a CPU; bge-small-zh-v1.5 is the light choice")


# ------------------------------------------------------------------- resplit


@retrieval_app.command("resplit")
@command(CommandKind.HEAVY)
def retrieval_resplit(
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation")] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run the queued jobs here (application stopped)")
    ] = False,
) -> None:
    """Move the hold-out cutoff to the newest 10 % of today's data (evaluation changes)."""
    services = get_cli_context().services()
    if not yes:
        typer.echo(
            "A re-split moves the evaluation set: results from before it are not comparable with "
            "later ones, and everything derived from the pre-holdout data is recomputed."
        )
        typer.confirm("Re-split the hold-out now?", abort=True)
    try:
        result = resplit_holdout(services)
    except HoldoutError as exc:
        raise CliError(str(exc), ExitCode.FAILURE) from exc
    previous = f"{result.previous.cutoff:%Y-%m-%d %H:%M} UTC" if result.previous else "none"
    typer.echo(
        f"cutoff: {previous} -> {result.current.cutoff:%Y-%m-%d %H:%M} UTC; "
        f"{result.current.held_out_blocks:,} of {result.current.blocks:,} reply blocks held out"
    )
    for note in result.notes:
        typer.echo(f"  {note}")
    if not foreground:
        typer.echo("the running application executes the queued jobs")
        return
    registry = HandlerRegistry()
    registry.register(INDEX_JOB, handle_retrieval_index)
    registry.register(PROFILE_JOB, handle_profile_rebuild)
    _foreground_or_hint(services, registry)
