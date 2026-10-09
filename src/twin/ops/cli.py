"""CLI: ``twin jobs``."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from twin.ops.jobs import (
    BatchNotFoundError,
    BatchTooLargeError,
    JobQueue,
    JobView,
    Worker,
    default_registry,
    load_handlers,
)
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.services import Services, get_cli_context

jobs_app = typer.Typer(help="The persistent job queue.", no_args_is_help=True)


def _queue(services: Services) -> JobQueue:
    return JobQueue(services.db, services.clock)


def _stamp(value: object) -> str:
    return "-" if value is None else str(value)[:19]


def _note(job: JobView) -> str:
    notes = []
    if job.status == "pending" and not default_registry.has(job.type):
        notes.append("无处理器")
    if job.requires_approval and job.approved_at is None:
        notes.append("待批准")
    return " ".join(notes)


@jobs_app.command("list")
@command(CommandKind.READ)
def jobs_list(
    status: Annotated[
        str | None, typer.Option(help="pending|running|done|failed|cancelled")
    ] = None,
    job_type: Annotated[str | None, typer.Option("--type", help="Filter by job type")] = None,
    batch: Annotated[str | None, typer.Option(help="Filter by batch id")] = None,
    limit: Annotated[int, typer.Option(help="Maximum rows")] = 50,
) -> None:
    """List jobs, newest first."""
    services = get_cli_context().services()
    load_handlers()
    try:
        jobs = _queue(services).list_jobs(
            status=status, job_type=job_type, batch_id=batch, limit=limit
        )
    except ValueError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    table = Table("id", "type", "status", "tries", "prio", "run after (UTC)", "batch", "note")
    for job in jobs:
        table.add_row(
            job.id,
            job.type,
            job.status,
            f"{job.attempts}/{job.max_attempts}",
            str(job.priority),
            _stamp(job.run_after),
            job.batch_id or "-",
            _note(job),
        )
    Console(highlight=False).print(table)
    counts = _queue(services).counts()
    typer.echo("  ".join(f"{name}={count}" for name, count in counts.items()))


@jobs_app.command("show")
@command(CommandKind.READ)
def jobs_show(
    job_id: Annotated[str, typer.Argument(help="Job id")],
    payload: Annotated[bool, typer.Option("--payload", help="Also print the payload")] = False,
) -> None:
    """Show one job (the payload is hidden unless --payload is given)."""
    services = get_cli_context().services()
    job = _queue(services).get(job_id)
    if job is None:
        raise CliError(f"no job {job_id}")
    load_handlers()
    rows = {
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "attempts": f"{job.attempts}/{job.max_attempts}",
        "priority": job.priority,
        "run_after": job.run_after,
        "offpeak_only": job.offpeak_only,
        "deadline": job.deadline,
        "batch": job.batch_id,
        "estimated_usd": job.estimated_cost_usd,
        "requires_approval": job.requires_approval,
        "approved_at": job.approved_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "created_at": job.created_at,
        "last_error": job.last_error,
        "note": _note(job),
    }
    for key, value in rows.items():
        typer.echo(f"{key:18} {value if value is not None else '-'}")
    if payload:
        typer.echo(f"{'payload':18} {job.payload!r}")


@jobs_app.command("retry")
@command(CommandKind.LIGHT)
def jobs_retry(job_id: Annotated[str, typer.Argument(help="Job id")]) -> None:
    """Re-queue a failed or cancelled job with a fresh attempt budget."""
    services = get_cli_context().services()
    if not _queue(services).retry(job_id):
        raise CliError(f"job {job_id} does not exist or is not failed/cancelled")
    typer.echo(f"job {job_id} queued again")


@jobs_app.command("cancel")
@command(CommandKind.LIGHT)
def jobs_cancel(job_id: Annotated[str, typer.Argument(help="Job id")]) -> None:
    """Cancel a pending or running job."""
    services = get_cli_context().services()
    if not _queue(services).cancel(job_id):
        raise CliError(f"job {job_id} does not exist or is already finished")
    typer.echo(f"job {job_id} cancelled")


@jobs_app.command("run")
@command(CommandKind.HEAVY)
def jobs_run(
    until_idle: Annotated[
        bool, typer.Option("--until-idle", help="Run queued jobs in the foreground, then exit")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Run even though the application is running")
    ] = False,
) -> None:
    """Execute queued jobs here (used when the application is not running)."""
    import asyncio

    if not until_idle:
        raise CliError(
            "pass --until-idle; the long-running worker is part of `twin run`", ExitCode.USAGE
        )
    services = get_cli_context().services()
    app_running = app_is_running(services)
    if app_running and not force:
        raise CliError(
            "the application is running and executes the queue itself; "
            "use --force to run a second worker anyway",
            ExitCode.BUSY,
        )
    load_handlers()
    worker = Worker(
        _queue(services),
        default_registry,
        services.clock,
        services=services,
        alerts=services.alerts,
        concurrency=services.settings.jobs.concurrency,
    )

    async def _go() -> None:
        if not app_running:  # with --force the running application owns its `running` jobs
            await worker.recover()
        summary = await worker.run_until_idle()
        typer.echo(
            f"done={summary.done} retried={summary.retried} "
            f"failed={summary.failed} discarded={summary.discarded}"
        )

    asyncio.run(_go())


@jobs_app.command("approve")
@command(CommandKind.LIGHT)
def jobs_approve(
    batch: Annotated[str, typer.Argument(help="Batch id")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
) -> None:
    """Approve a one-time batch of jobs after reviewing its cost estimate (R-LLM-014)."""
    services = get_cli_context().services()
    queue = _queue(services)
    try:
        summary = queue.batch_summary(batch)
    except BatchNotFoundError as exc:
        raise CliError(str(exc)) from exc
    limit = services.settings.budget.one_time_usd
    typer.echo(
        f"batch {batch}: {summary.jobs} job(s), {summary.awaiting_approval} awaiting approval, "
        f"estimated ${summary.estimated_usd:.2f} (one-time limit ${limit:.2f})"
    )
    if not yes and not typer.confirm("Approve this batch?"):
        raise typer.Exit(1)
    try:
        approval = queue.approve_batch(batch, max_usd=limit)
    except (BatchNotFoundError, BatchTooLargeError) as exc:
        raise CliError(str(exc)) from exc
    typer.echo(f"approved {approval.job_count} job(s), ${approval.total_usd:.2f}")
