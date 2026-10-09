"""CLI: ``twin timezone`` and ``twin plan`` (R-SCH-002, R-SCH-004, R-MEM-005).

``timezone show`` / ``timezone history`` / ``plan show`` / ``plan lifeline`` only read (READ).
``timezone set`` and ``plan rebuild`` are short changes (LIGHT): they write the settings and the
plan rows in the same
transactions that raise ``state_version``, so a running application notices within two seconds,
drops what it had read and, for a switch, announces it (the application derives the event from the
history table, so a switch made here and one made by ``/时区`` look the same to the other rounds).

``plan show`` never writes: for a day that has no plan yet it shows the plan the day will get (the
seed is a hash of the date and the installation's salt; before the first plan exists the salt does
not either and the preview says so).  ``plan lifeline-generate`` (HEAVY) only queues the job that
draws the life line of a day; the running application, or ``twin jobs run --until-idle``, does it.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from twin.memory.lifeline import LifelineStore
from twin.memory.memory import Memory
from twin.ops.process_model import (
    CliError,
    CommandKind,
    ExitCode,
    app_is_running,
    command,
)
from twin.schedule.jobs import queue_lifeline
from twin.schedule.planner import TimezoneError, validate_zone
from twin.schedule.service import schedule_kit
from twin.schedule.show import DAY_TYPES, STATES, WEEKDAYS, render_plan
from twin.schedule.wallclock import next_transition
from twin.services import Services, get_cli_context

timezone_app = typer.Typer(help="The bot's time zone.", no_args_is_help=True)
plan_app = typer.Typer(help="The day plan.", no_args_is_help=True)


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _day(value: str | None, services: Services) -> date:
    if value is None:
        return schedule_kit(services).time.local_date()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CliError(
            f"the date must look like 2026-10-09, got {value!r}", ExitCode.USAGE
        ) from exc


def _local(moment: datetime, services: Services) -> str:
    return f"{moment.astimezone(schedule_kit(services).time.bot_timezone()):%Y-%m-%d %H:%M}"


# --------------------------------------------------------------------- timezone


@timezone_app.command("show")
@command(CommandKind.READ)
def timezone_show() -> None:
    """The bot's time zone, the time there and her state now."""
    services = get_cli_context().services()
    kit = schedule_kit(services)
    time = kit.time
    zone = time.bot_timezone()
    now = time.now_local()
    offset = now.utcoffset()
    hours = (offset.total_seconds() / 3600) if offset is not None else 0.0
    typer.echo(f"bot time zone: {zone.key} (UTC{hours:+g}, {now.tzname()})")
    typer.echo(f"local time:    {now:%Y-%m-%d %H:%M:%S} {WEEKDAYS[now.weekday()]}")
    typer.echo(f"day type:      {DAY_TYPES[time.day_type()]}")
    source = services.settings.time
    typer.echo(f"routine learnt in: {source.source_timezone} (the clock times move over unchanged)")
    change = next_transition(zone, time.now_utc())
    if change is None:
        typer.echo("clock changes: none (this zone does not use daylight saving time)")
    else:
        when = change.at.astimezone(zone)
        typer.echo(
            f"next clock change: {change.at:%Y-%m-%d %H:%M} UTC "
            f"({change.before} -> {change.after}, {change.shift.total_seconds() / 3600:+g} h; "
            f"on the wall clock {when:%Y-%m-%d %H:%M} {change.after})"
        )
    planner = kit.planner
    if planner.store.covering(time.now_utc()) is None:
        typer.echo("her state:     no plan yet (start the application or run `twin plan rebuild`)")
        return
    state = time.her_state()
    typer.echo(
        f"her state:     {STATES[state.kind]} since {_local(state.since, services)} "
        f"until {_local(state.until, services)}"
    )


@timezone_app.command("set")
@command(CommandKind.LIGHT)
def timezone_set(
    name: Annotated[str, typer.Argument(help="IANA name, e.g. Asia/Shanghai or America/Chicago")],
) -> None:
    """Switch the bot to another time zone and plan the rest of the day there (R-SCH-002)."""
    services = get_cli_context().services()
    try:
        validate_zone(name)
    except TimezoneError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    outcome = schedule_kit(services).planner.switch_timezone(name, source="cli")
    if not outcome.changed or outcome.plan is None:
        typer.echo(f"the bot is already in {outcome.new_timezone}; nothing changed")
        return
    plan = outcome.plan
    typer.echo(f"switched {outcome.old_timezone} -> {outcome.new_timezone}")
    for line in render_plan(plan):
        typer.echo(line)
    typer.echo(
        "A running application notices within two seconds; messages already waiting for their "
        "delay keep their time, proactive candidates are drawn again from the new plan."
    )


@timezone_app.command("history")
@command(CommandKind.READ)
def timezone_history(
    limit: Annotated[int, typer.Option("--limit", min=1, max=200, help="How many switches")] = 20,
) -> None:
    """The switches of the bot's time zone, newest first."""
    services = get_cli_context().services()
    records = schedule_kit(services).planner.history.recent(limit)
    if not records:
        typer.echo("the time zone was never switched")
        return
    table = Table("when (UTC)", "from", "to", "route", "plan", "last greeting (UTC)")
    for record in records:
        greeting = f"{record.last_greeting_at:%Y-%m-%d %H:%M}" if record.last_greeting_at else "—"
        table.add_row(
            f"{record.changed_at:%Y-%m-%d %H:%M}",
            record.from_timezone,
            record.to_timezone,
            record.source,
            record.plan_id or "—",
            greeting,
        )
    _console().print(table)


# ------------------------------------------------------------------------- plan


@plan_app.command("show")
@command(CommandKind.READ)
def plan_show(
    day: Annotated[
        str | None, typer.Argument(help="Local date YYYY-MM-DD (default: today)")
    ] = None,
) -> None:
    """The plan of a local day in the clock times of the bot's zone."""
    services = get_cli_context().services()
    planner = schedule_kit(services).planner
    target = _day(day, services)
    zone = schedule_kit(services).time.bot_timezone()
    stored = planner.store.current_for(target, zone.key)
    plans = planner.store.for_date(target)
    if stored is not None:
        lines = render_plan(stored, replaced=max(0, len(plans) - 1))
    else:
        lines = render_plan(planner.preview(target), stored=False)
    for line in lines:
        typer.echo(line)


@plan_app.command("rebuild")
@command(CommandKind.LIGHT)
def plan_rebuild(
    force: Annotated[
        bool, typer.Option("--force", help="Make a new plan even if its inputs did not change")
    ] = False,
) -> None:
    """Make today's plan again from now on (after a changed routine, holiday or range)."""
    services = get_cli_context().services()
    outcome = schedule_kit(services).planner.refresh("manual", force=force)
    if outcome.changed:
        typer.echo(
            f"new plan {outcome.plan.id} replaces {len(outcome.superseded)} plan(s) from now"
        )
    else:
        typer.echo("the plan already fits its inputs; nothing changed (use --force to redraw)")
    for line in render_plan(outcome.plan):
        typer.echo(line)


LIFELINE_SOURCES = {"plan": "planned", "improvised": "improvised in a chat"}


@plan_app.command("lifeline")
@command(CommandKind.READ)
def plan_lifeline(
    day: Annotated[
        str | None, typer.Argument(help="Local date YYYY-MM-DD (default: today)")
    ] = None,
) -> None:
    """The life line of a day: what she does, where, in which mood (R-MEM-005)."""
    services = get_cli_context().services()
    kit = schedule_kit(services)
    target = _day(day, services)
    entries = LifelineStore(Memory(services), time_service=kit.time).day(target)
    if not entries:
        typer.echo(
            f"no life line for {target.isoformat()} yet: it is drawn when she wakes "
            "(`twin plan lifeline-generate` queues it now)"
        )
        return
    table = Table("time", "what she does", "where", "mood", "origin")
    for entry in entries:
        span = " - ".join(part for part in (entry.start_local, entry.end_local) if part) or "—"
        table.add_row(
            span,
            entry.activity,
            entry.place or "—",
            entry.mood or "—",
            LIFELINE_SOURCES.get(entry.source, entry.source),
        )
    typer.echo(f"life line of {target.isoformat()} ({entries[0].timezone}, local times)")
    _console().print(table)


@plan_app.command("lifeline-generate")
@command(CommandKind.HEAVY)
def plan_lifeline_generate(
    day: Annotated[
        str | None, typer.Argument(help="Local date YYYY-MM-DD (default: today)")
    ] = None,
) -> None:
    """Queue the life line of a day (it replaces the planned entries of that day)."""
    services = get_cli_context().services()
    kit = schedule_kit(services)
    target = _day(day, services)
    zone = kit.time.bot_timezone()
    plan = kit.planner.store.current_for(target, zone.key)
    if plan is None:
        raise CliError(
            f"there is no plan for {target.isoformat()} in {zone.key} yet; "
            "start the application or run `twin plan rebuild` first",
            ExitCode.USAGE,
        )
    job_id = queue_lifeline(services, plan)
    kit.planner.store.mark_lifeline_queued(plan.id, job_id)
    typer.echo(f"queued the life line of {target.isoformat()} as job {job_id}")
    if app_is_running(services):
        typer.echo("the running application will draw it; see it with `twin plan lifeline`")
    else:
        typer.echo("the application is stopped: run `twin jobs run --until-idle` to draw it now")
