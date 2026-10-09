"""CLI: ``twin profile`` and ``twin routine`` (R-PROF-004, R-ACT-005, R-ACT-006).

``twin profile rebuild`` is a HEAVY command (queues the recomputation, ``--foreground`` runs
it here when the application is stopped).  ``history``, ``show`` and ``diff`` only read;
``rollback`` and the ``twin routine`` changes are LIGHT (they bump ``settings.state_version`` so
a running application reloads).  The routine corrections go through
:class:`~twin.profile.overrides.RoutineOverrides`, the API the chat commands of round 11 use too.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.profile.diffing import diff_metrics, summarize
from twin.profile.holdout import get_holdout
from twin.profile.jobs import handle_profile_rebuild
from twin.profile.overrides import OverrideError, RoutineOverrides, parse_weekdays
from twin.profile.queue import PROFILE_JOB, SCOPE_CHOICES, queue_profile_rebuild
from twin.profile.show import profile_overview
from twin.profile.store import VersionError, VersionStore
from twin.services import Services, get_cli_context

profile_app = typer.Typer(help="Style profile and routine model.", no_args_is_help=True)
routine_app = typer.Typer(help="Manual corrections of the routine.", no_args_is_help=True)
routine_add_app = typer.Typer(help="Add a correction.", no_args_is_help=True)
routine_app.add_typer(routine_add_app, name="add")

SHOWN_SCOPES = ("live", "pre_holdout")


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _store(services: Services) -> VersionStore:
    return VersionStore(services.db, services.clock)


def _check_scope(scope: str, allowed: tuple[str, ...]) -> str:
    if scope not in allowed:
        raise CliError(f"--scope must be one of {', '.join(allowed)}", ExitCode.USAGE)
    return scope


# ------------------------------------------------------------------ profile


@profile_app.command("rebuild")
@command(CommandKind.HEAVY)
def profile_rebuild(
    scope: Annotated[str, typer.Option("--scope", help="live, pre_holdout or all")] = "all",
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run here when the application is stopped")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Write a new version even if nothing changed")
    ] = False,
) -> None:
    """Recompute the style profile and the routine model (queued as a job)."""
    services = get_cli_context().services()
    _check_scope(scope, SCOPE_CHOICES)
    queued = queue_profile_rebuild(services, scope=scope, reason="manual", force=force)
    verb = "a recomputation is already waiting" if queued.already_queued else "queued"
    typer.echo(f"{verb} (job {queued.job_id}, scope {scope})")
    if not foreground:
        typer.echo("the running application executes it; `twin jobs list --type profile_rebuild`")
        return
    if app_is_running(services):
        typer.echo("the application is running and will execute the job")
        return
    registry = HandlerRegistry()
    registry.register(PROFILE_JOB, handle_profile_rebuild)
    summary = asyncio.run(run_jobs_until_idle(services, registry))
    if summary.failed or summary.retried:
        raise CliError(
            f"the recomputation did not finish: {'; '.join(summary.failures) or 'see the log'}"
        )
    store = _store(services)
    for name in SHOWN_SCOPES:
        if scope not in ("all", name):
            continue
        version = store.active_profile(name)
        if version is None:
            typer.echo(f"{name}: nothing computed (see `twin profile show`)")
            continue
        typer.echo(
            f"{name}: version {version.id} (her {version.her_messages:,} messages, "
            f"{len(version.changes)} metrics changed over 10%)"
        )
    typer.echo("run `twin profile show` and confirm the sleep time it infers")


@profile_app.command("history")
@command(CommandKind.READ)
def profile_history(
    scope: Annotated[str | None, typer.Option("--scope", help="live or pre_holdout")] = None,
    limit: Annotated[int, typer.Option(min=1, help="Newest versions to list")] = 20,
) -> None:
    """List the stored profile versions, newest first."""
    services = get_cli_context().services()
    if scope is not None:
        _check_scope(scope, SHOWN_SCOPES)
    versions = _store(services).history(scope, limit)
    if not versions:
        raise CliError("no profile has been computed yet; run `twin profile rebuild`")
    table = Table(
        "#", "version", "scope", "created (UTC)", "reason", "her msgs", "changes", "active"
    )
    for rank, version in enumerate(versions):
        table.add_row(
            f"~{rank}",
            version.id,
            version.scope,
            f"{version.created_at:%Y-%m-%d %H:%M}",
            version.reason,
            f"{version.her_messages:,}",
            str(len(version.changes)),
            "yes" if version.active else "",
        )
    _console().print(table)


@profile_app.command("show")
@command(CommandKind.READ)
def profile_show(
    version: Annotated[
        str | None, typer.Argument(help="Version id (prefix) or ~N; default: the active one")
    ] = None,
    scope: Annotated[str, typer.Option("--scope", help="live or pre_holdout")] = "live",
) -> None:
    """Show the style numbers and the routine overview (local time); confirm the sleep time."""
    services = get_cli_context().services()
    _check_scope(scope, SHOWN_SCOPES)
    store = _store(services)
    try:
        chosen = store.resolve(version, scope) if version else store.active_profile(scope)
    except VersionError as exc:
        raise CliError(str(exc)) from exc
    if chosen is None:
        raise CliError(f"no {scope} profile yet; run `twin profile rebuild --scope {scope}`")
    activity = store.activity_for_profile(chosen.id)
    model = store.activity_model(activity.id) if activity else None
    overrides = RoutineOverrides(services.db, services.clock).entries()
    if model is not None:
        model = model.with_overrides([o for o in overrides if o.enabled])
    holdout = get_holdout(services)
    lines = profile_overview(chosen, store.metrics(chosen.id), model, overrides)
    if holdout is not None:
        lines.insert(
            5 if chosen.data_range.get("cutoff") else 4,
            f"hold-out 切分点：{holdout.cutoff:%Y-%m-%d %H:%M} UTC"
            f"（她的回复块 {holdout.blocks:,} 个，其中留出 {holdout.held_out_blocks:,} 个）",
        )
    console = _console()
    for line in lines:
        if line.startswith("!!!"):
            console.print(line, style="bold red", markup=False)
        else:
            console.print(line, markup=False)


@profile_app.command("phrases")
@command(CommandKind.READ)
def profile_phrases(
    scope: Annotated[str, typer.Option("--scope", help="live or pre_holdout")] = "live",
    top: Annotated[int, typer.Option(min=1, help="Entries to show per list")] = 15,
) -> None:
    """Show her frequent sentences, n-grams and form-of-address candidates (real text, local)."""
    services = get_cli_context().services()
    _check_scope(scope, SHOWN_SCOPES)
    store = _store(services)
    version = store.active_profile(scope)
    if version is None:
        raise CliError(f"no {scope} profile yet; run `twin profile rebuild --scope {scope}`")
    phrases = store.phrases(version.id)
    if not phrases:
        raise CliError("this profile version has no phrase data")
    her = phrases.get("her", {})
    typer.echo(
        f"profile {version.id} ({scope}): text of her real messages, shown on this screen only"
    )
    typer.echo("form-of-address candidates (term, at start, at end, total uses, share at an edge):")
    for item in her.get("address_candidates", [])[:top]:
        columns = (item["term"], item["start"], item["end"], item["total"])
        typer.echo("  " + "\t".join(str(c) for c in columns) + f"\t{item['edge_share']:.0%}")
    typer.echo("frequent sentences:")
    for text, count in her.get("sentences", [])[:top]:
        typer.echo(f"  {count}\t{text}")
    for size in sorted(her.get("ngrams", {})):
        typer.echo(f"frequent {size}-character groups:")
        for text, count in her["ngrams"][size][:top]:
            typer.echo(f"  {count}\t{text}")


@profile_app.command("diff")
@command(CommandKind.READ)
def profile_diff(
    first: Annotated[str, typer.Argument(help="Older version (id prefix or ~N)")],
    second: Annotated[str, typer.Argument(help="Newer version (id prefix or ~N)")],
    scope: Annotated[
        str | None, typer.Option("--scope", help="Scope that ~N counts in (live or pre_holdout)")
    ] = None,
) -> None:
    """List the metrics that differ by more than 10 % between two versions."""
    services = get_cli_context().services()
    if scope is not None:
        _check_scope(scope, SHOWN_SCOPES)
    store = _store(services)
    try:
        a, b = store.resolve(first, scope), store.resolve(second, scope)
        changes = diff_metrics(store.metrics(a.id), store.metrics(b.id))
    except VersionError as exc:
        raise CliError(str(exc)) from exc
    typer.echo(
        f"{a.id} ({a.scope}) -> {b.id} ({b.scope}): {len(changes)} metric(s) changed over 10%"
    )
    for line in summarize(changes, limit=len(changes) or 1):
        typer.echo(f"  {line}")


@profile_app.command("rollback")
@command(CommandKind.LIGHT)
def profile_rollback(
    version: Annotated[str, typer.Argument(help="Version id (prefix) or ~N")],
    scope: Annotated[
        str | None, typer.Option("--scope", help="Scope that ~N counts in (live or pre_holdout)")
    ] = None,
) -> None:
    """Make an older version (and the routine model computed with it) the active one."""
    services = get_cli_context().services()
    if scope is not None:
        _check_scope(scope, SHOWN_SCOPES)
    store = _store(services)
    try:
        target = store.resolve(version, scope)
        chosen, activity = store.rollback(target.id)
    except VersionError as exc:
        raise CliError(str(exc)) from exc
    typer.echo(f"{chosen.scope} profile is now version {chosen.id}")
    if activity is not None:
        typer.echo(f"routine model is now version {activity.id}")


# ------------------------------------------------------------------ routine


@routine_app.command("list")
@command(CommandKind.READ)
def routine_list() -> None:
    """List the manual routine corrections."""
    services = get_cli_context().services()
    entries = RoutineOverrides(services.db, services.clock).entries()
    if not entries:
        typer.echo("no manual corrections; the routine is what `twin profile show` infers")
        return
    table = Table("id", "correction", "state", "added (UTC)")
    for item in entries:
        table.add_row(
            item.id, item.describe(), "on" if item.enabled else "off", f"{item.created_at:%Y-%m-%d}"
        )
    _console().print(table)


def _api(services: Services) -> RoutineOverrides:
    return RoutineOverrides(services.db, services.clock)


@routine_add_app.command("sleep")
@command(CommandKind.LIGHT)
def routine_add_sleep(
    start: Annotated[str, typer.Argument(help="Falling asleep, HH:MM local time")],
    end: Annotated[str, typer.Argument(help="Waking up, HH:MM local time")],
    days: Annotated[
        str | None,
        typer.Option("--days", help="workday, weekend and/or holiday, comma separated"),
    ] = None,
) -> None:
    """Set the sleep interval (wins over the inferred one)."""
    services = get_cli_context().services()
    types = [part.strip() for part in days.split(",") if part.strip()] if days else None
    try:
        item = _api(services).add_sleep(start, end, day_types=types)
    except OverrideError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    typer.echo(f"added {item.id}: {item.describe()}")


@routine_add_app.command("busy")
@command(CommandKind.LIGHT)
def routine_add_busy(
    start: Annotated[str, typer.Argument(help="Start, HH:MM local time")],
    end: Annotated[str, typer.Argument(help="End, HH:MM local time")],
    weekdays: Annotated[
        str, typer.Option("--weekdays", help="For example mon-fri, sat,sun or 周一至周五")
    ] = "mon-fri",
) -> None:
    """Mark a weekly busy period."""
    services = get_cli_context().services()
    try:
        item = _api(services).add_busy(parse_weekdays(weekdays), start, end)
    except OverrideError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    typer.echo(f"added {item.id}: {item.describe()}")


@routine_add_app.command("holiday")
@command(CommandKind.LIGHT)
def routine_add_holiday(
    first: Annotated[str, typer.Argument(help="First day, YYYY-MM-DD")],
    last: Annotated[str | None, typer.Argument(help="Last day (default: the first day)")] = None,
) -> None:
    """Mark a range of dates as holidays (takes effect for learning at the next rebuild)."""
    services = get_cli_context().services()
    try:
        begin = date.fromisoformat(first)
        finish = date.fromisoformat(last) if last else begin
        item = _api(services).add_holiday(begin, finish)
    except (ValueError, OverrideError) as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    typer.echo(f"added {item.id}: {item.describe()}")
    typer.echo("run `twin profile rebuild` so the routine is learnt with these days as holidays")


@routine_app.command("remove")
@command(CommandKind.LIGHT)
def routine_remove(
    correction_id: Annotated[str, typer.Argument(help="Id from `twin routine list`")],
) -> None:
    """Delete a correction."""
    services = get_cli_context().services()
    if not _api(services).remove(correction_id):
        raise CliError(f"no correction {correction_id}")
    typer.echo(f"removed {correction_id}")


@routine_app.command("enable")
@command(CommandKind.LIGHT)
def routine_enable(
    correction_id: Annotated[str, typer.Argument(help="Id from `twin routine list`")],
) -> None:
    """Switch a correction on again."""
    services = get_cli_context().services()
    if not _api(services).set_enabled(correction_id, True):
        raise CliError(f"no correction {correction_id}")
    typer.echo(f"enabled {correction_id}")


@routine_app.command("disable")
@command(CommandKind.LIGHT)
def routine_disable(
    correction_id: Annotated[str, typer.Argument(help="Id from `twin routine list`")],
) -> None:
    """Switch a correction off without deleting it."""
    services = get_cli_context().services()
    if not _api(services).set_enabled(correction_id, False):
        raise CliError(f"no correction {correction_id}")
    typer.echo(f"disabled {correction_id}")
