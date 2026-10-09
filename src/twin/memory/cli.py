"""CLI: ``twin memory`` (R-MEM-009, R-MEM-010, R-IMP-011, R-LLM-014).

``replay estimate``, ``replay status``, ``list`` and ``block`` (the memory block a reply would
get, to judge the recall) only read.  ``replay start`` is HEAVY: it
queues the replay of the days that need it as one-time batches and prints the estimate; nothing
runs before ``twin jobs approve <batch>`` (R-LLM-014).  ``remember``, ``forget`` and ``reindex``
are short changes (LIGHT).  ``summarize`` queues the summary of one day (HEAVY).
The commands print counts, dates and money; ``list``, ``remember`` and ``forget`` show memory
text to the person who asked, on their own terminal, and nothing of it reaches a log.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from twin.llm.runtime import build_llm_runtime
from twin.memory.assemble import MemoryAssembler
from twin.memory.blocks import MemoryQuery
from twin.memory.jobs import queue_daily_summary
from twin.memory.manage import MemoryManager
from twin.memory.memory import Memory
from twin.memory.replay import (
    ReplayRequest,
    estimate_replay,
    plan_replay,
    replay_status,
)
from twin.ops.process_model import CliError, CommandKind, ExitCode, command
from twin.services import Services, get_cli_context

memory_app = typer.Typer(
    help="The memory: facts, summaries, replay of the records.", no_args_is_help=True
)
replay_app = typer.Typer(help="Replay the real records into the memory.", no_args_is_help=True)
memory_app.add_typer(replay_app, name="replay")


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _day(value: str | None, option: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CliError(f"{option} must be a date like 2026-03-05", ExitCode.USAGE) from exc


def _request(start: str | None, end: str | None, force: bool) -> ReplayRequest:
    first, last = _day(start, "--from"), _day(end, "--to")
    if first and last and last < first:
        raise CliError("--to is before --from", ExitCode.USAGE)
    return ReplayRequest(first, last, force)


FROM_OPTION = typer.Option("--from", help="First local day (YYYY-MM-DD); default: the first record")
TO_OPTION = typer.Option("--to", help="Last local day (YYYY-MM-DD); default: the last record")


# ------------------------------------------------------------------------ replay


@replay_app.command("estimate")
@command(CommandKind.READ)
def replay_estimate_command(
    start: Annotated[str | None, FROM_OPTION] = None,
    end: Annotated[str | None, TO_OPTION] = None,
    force: Annotated[bool, typer.Option("--force", help="Count the days already done too")] = False,
) -> None:
    """Show which days a replay would do and what it would cost (an upper bound; nothing runs)."""
    services = get_cli_context().services()
    estimate = estimate_replay(services, _request(start, end, force))
    limit = services.settings.budget.one_time_usd
    table = Table("item", "value", show_header=False)
    table.add_row("days to replay", f"{len(estimate.days):,}")
    table.add_row("days already done", f"{estimate.already_done:,}")
    table.add_row("dialogue lines", f"{estimate.lines:,}")
    table.add_row("estimated dialogue tokens", f"{estimate.tokens:,}")
    table.add_row("estimated cost (USD, peak prices)", f"{estimate.estimated_usd:.2f}")
    table.add_row("one-time limit per batch (USD)", f"{limit:.2f}")
    if estimate.days:
        table.add_row("days from / to", f"{estimate.days[0].day} / {estimate.days[-1].day}")
    _console().print(table)
    if not estimate.days:
        typer.echo("nothing to replay: every day of the records is done")
        return
    if estimate.estimated_usd > limit:
        typer.echo("more than one batch will be needed (each stays within the one-time limit)")
    typer.echo("start it with: twin memory replay start (then twin jobs approve <batch>)")


@replay_app.command("start")
@command(CommandKind.HEAVY)
def replay_start_command(
    start: Annotated[str | None, FROM_OPTION] = None,
    end: Annotated[str | None, TO_OPTION] = None,
    force: Annotated[bool, typer.Option("--force", help="Replay days that are done too")] = False,
) -> None:
    """Queue the replay of the days that need it; it waits for `twin jobs approve <batch>`."""
    services = get_cli_context().services()
    plan = plan_replay(services, _request(start, end, force), reason="manual")
    if plan.already_queued:
        typer.echo(f"{plan.already_queued} day(s) are already waiting in the queue")
    if not plan.batches:
        typer.echo("nothing new to queue")
        return
    table = Table("batch", "days", "jobs", "estimated cost (USD)")
    for batch in plan.batches:
        table.add_row(
            batch.batch_id, str(batch.days), str(batch.jobs), f"{batch.estimated_usd:.2f}"
        )
    _console().print(table)
    typer.echo(f"estimated ${plan.estimated_usd:.2f} in total (an upper bound)")
    for batch in plan.batches:
        typer.echo(f"approve with: twin jobs approve {batch.batch_id}")
    typer.echo(
        "after the approval the running application replays the days off peak; with the "
        "application stopped use `twin jobs run --until-idle`"
    )


@replay_app.command("status")
@command(CommandKind.READ)
def replay_status_command() -> None:
    """Show how far the replay is: days done, batches, jobs and what the memory holds."""
    services = get_cli_context().services()
    status = replay_status(services)
    table = Table("item", "value", show_header=False)
    table.add_row("days with messages", f"{status.days_with_messages:,}")
    table.add_row("days replayed", f"{status.days_replayed:,}")
    table.add_row("days waiting", f"{status.days_waiting:,}")
    table.add_row("jobs", ", ".join(f"{name} {count}" for name, count in status.jobs.items()))
    table.add_row("daily summaries", f"{status.summaries:,}")
    sources = ", ".join(f"{name} {count}" for name, count in sorted(status.facts_by_source.items()))
    table.add_row("current facts by source", sources or "none")
    table.add_row("open follow-ups", f"{status.followups_open:,}")
    _console().print(table)
    for batch in status.batches:
        state = (
            "paused" if batch.paused else ("approved" if batch.approved else "waiting for approval")
        )
        typer.echo(
            f"batch {batch.batch_id}: {state}; estimated ${batch.estimated_usd:.2f}, "
            f"spent ${batch.spent_usd:.2f}"
            + (f" of a cap of ${batch.cap_usd:.2f}" if batch.cap_usd is not None else "")
        )


# ---------------------------------------------------------------------- the rest


def _manager(services: Services) -> MemoryManager:
    runtime = build_llm_runtime(services)
    return MemoryManager(Memory(services), runtime.client)


@memory_app.command("list")
@command(CommandKind.READ)
def memory_list(
    page: Annotated[int, typer.Option("--page", min=1, help="Page number")] = 1,
    keyword: Annotated[
        str | None, typer.Option("--keyword", help="Only facts with this word")
    ] = None,
) -> None:
    """List what the bot remembers (current facts, newest first)."""
    services = get_cli_context().services()
    listing = MemoryManager(Memory(services)).list_items(page, keyword)
    if not listing.items:
        typer.echo("nothing remembered yet" if not keyword else "no fact contains that word")
        return
    console = _console()
    for item in listing.items:
        when = f" [{item.event_date}]" if item.event_date else ""
        console.print(f"{item.number}. {item.text}{when}  ({item.source})", markup=False)
    for text in listing.followups:
        console.print(f"待跟进：{text}", markup=False)
    typer.echo(f"page {listing.page}/{listing.pages}, {listing.total} fact(s)")


@memory_app.command("block")
@command(CommandKind.READ)
def memory_block(
    topic: Annotated[str, typer.Argument(help="The current topic: what the user just said")],
    at: Annotated[
        str | None,
        typer.Option(
            "--at", help="A past moment with its UTC offset, e.g. 2026-03-05T09:00:00-06:00"
        ),
    ] = None,
    budget: Annotated[
        int | None,
        typer.Option("--budget", min=0, help="Token budget (default: memory.block_tokens)"),
    ] = None,
) -> None:
    """Show the memory block a reply about ``topic`` would get (to judge what is recalled)."""
    services = get_cli_context().services()
    moment = services.clock.now_utc()
    if at is not None:
        try:
            parsed = datetime.fromisoformat(at)
        except ValueError as exc:
            raise CliError("--at must look like 2026-03-05T09:00:00-06:00", ExitCode.USAGE) from exc
        if parsed.tzinfo is None:
            raise CliError("--at needs a UTC offset, e.g. -06:00 or +08:00", ExitCode.USAGE)
        moment = parsed.astimezone(UTC)
    runtime = build_llm_runtime(services)
    assembler = MemoryAssembler(
        Memory(services), estimator=runtime.estimator, limits=runtime.budget.limits
    )
    block = assembler.build(MemoryQuery(topic), moment, budget)
    if block.empty:
        typer.echo("the memory has nothing to say about that")
    else:
        _console().print(block.text, markup=False)
    typer.echo(
        f"{len(block.items)} item(s), {block.tokens}/{block.budget_tokens} tokens, "
        f"{block.dropped} left out"
    )


@memory_app.command("remember")
@command(CommandKind.LIGHT)
def memory_remember(
    text: Annotated[str, typer.Argument(help="What to remember")],
) -> None:
    """Remember something (source: user command, the highest rank)."""
    services = get_cli_context().services()
    manager = _manager(services)
    try:
        result = asyncio.run(manager.remember(text))
    except ValueError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    for item in result.facts:
        typer.echo(f"remembered as number {item.number}")
    if result.replaced:
        typer.echo(f"it replaced {result.replaced} older fact(s)")
    if not result.enriched:
        typer.echo("stored as written (the text could not be analysed)")


@memory_app.command("forget")
@command(CommandKind.LIGHT)
def memory_forget(
    query: Annotated[str, typer.Argument(help="A number, an id or some words")],
    all_matches: Annotated[
        bool, typer.Option("--all", help="Delete every fact the words match")
    ] = False,
) -> None:
    """Forget a fact for good, with what was derived only from it."""
    services = get_cli_context().services()
    try:
        result = MemoryManager(Memory(services)).forget(query, all_matches=all_matches)
    except ValueError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    if result.ambiguous:
        typer.echo("that matches several facts; nothing was deleted. Name one by number:")
        for item in result.ambiguous:
            _console().print(f"{item.number}. {item.text}", markup=False)
        return
    if result.followup_matches:
        typer.echo("that matches several follow-ups; use --all to delete them:")
        for text in result.followup_matches:
            _console().print(text, markup=False)
        return
    if not result.deleted:
        raise CliError("nothing matches that")
    for gone in result.deleted:
        label = f"number {gone.number}" if gone.number is not None else gone.kind
        _console().print(f"deleted {gone.kind} ({label}): {gone.text}", markup=False)
    if result.restored:
        typer.echo("became current again: " + ", ".join(str(n) for n in result.restored))


@memory_app.command("summarize")
@command(CommandKind.HEAVY)
def memory_summarize(
    day: Annotated[str, typer.Argument(help="The local day, YYYY-MM-DD")],
    scope: Annotated[str, typer.Option("--scope", help="real, bot or both")] = "both",
    force: Annotated[
        bool, typer.Option("--force", help="Write a new version even if unchanged")
    ] = False,
) -> None:
    """Queue the summary of one day (off peak)."""
    services = get_cli_context().services()
    when = _day(day, "DAY")
    if when is None:
        raise CliError("give the day as YYYY-MM-DD", ExitCode.USAGE)
    if scope not in ("real", "bot", "both"):
        raise CliError("--scope must be real, bot or both", ExitCode.USAGE)
    scopes = ("real", "bot") if scope == "both" else (scope,)
    ids = queue_daily_summary(services, when, scopes, force=force)
    typer.echo(f"queued {len(ids)} summary job(s)" if ids else "those summaries are already queued")


@memory_app.command("reindex")
@command(CommandKind.LIGHT)
def memory_reindex() -> None:
    """Encode every fact and summary again (after the embedding model changed)."""
    services = get_cli_context().services()
    result = Memory(services).reindex()
    typer.echo(f"encoded {result.encoded:,} record(s)")
