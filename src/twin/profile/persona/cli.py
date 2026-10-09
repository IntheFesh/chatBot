"""CLI: ``twin persona`` (R-PERS-001 to R-PERS-005).

``rules`` (``list``, ``delete``, ``consolidate``) is the view of the ``[不要这样]`` rules that the
learning writes (round 11): ``twin persona rules`` lists them, ``delete`` removes one by its
number (a new version of the card), ``consolidate`` queues the weekly consolidation now.
``show``, ``history``, ``diff`` and ``templates`` only read.  ``rollback`` and ``edit`` change
the card in force (LIGHT: the running application reloads it).  ``regenerate`` is a HEAVY
command: without ``--stats-only`` it plans the generation of the automatic description as a
one-time batch - it prints the estimated cost and waits for ``twin jobs approve <batch>`` - and
with ``--stats-only`` it only refreshes the statistics rules, which costs nothing.
"""

from __future__ import annotations

import asyncio
import difflib
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from twin.learning.jobs import RULES_JOB, handle_rules, queue_rules_job
from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.profile.holdout import HoldoutError
from twin.profile.persona import edit as editing
from twin.profile.persona.api import (
    read_corrections,
    render_compact,
    render_full,
    write_corrections,
)
from twin.profile.persona.jobs import plan_generation, queue_refresh
from twin.profile.persona.refresh import description_due, refresh_all, sync_manual_style
from twin.profile.persona.store import PersonaStore, PersonaVersionError
from twin.profile.prompt_templates import ACTIVE_KEY, template_files
from twin.services import Services, get_cli_context
from twin.storage.settings_store import get_setting

persona_app = typer.Typer(help="The persona card.", no_args_is_help=True)
rules_app = typer.Typer(
    help="The rules of [不要这样] (from /不像 and /重来): list, delete one, consolidate.",
    invoke_without_command=True,
)
persona_app.add_typer(rules_app, name="rules")
SCOPES = ("live", "pre_holdout")
SCOPE_CHOICES = (*SCOPES, "all")


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _store(services: Services) -> PersonaStore:
    return PersonaStore(services.db, services.clock)


def _scope(value: str, allowed: tuple[str, ...] = SCOPES) -> str:
    if value not in allowed:
        raise CliError(f"--scope must be one of {', '.join(allowed)}", ExitCode.USAGE)
    return value


@persona_app.command("show")
@command(CommandKind.READ)
def persona_show(
    version: Annotated[
        str | None, typer.Argument(help="vN, ~N or an id prefix; default: the version in force")
    ] = None,
    scope: Annotated[str, typer.Option("--scope", help="live or pre_holdout")] = "live",
    full: Annotated[bool, typer.Option("--full", help="Show the full rendering")] = False,
    compact: Annotated[bool, typer.Option("--compact", help="Show the compact rendering")] = False,
    evidence: Annotated[
        bool, typer.Option("--evidence", help="List the segments behind each statement")
    ] = False,
) -> None:
    """Show a persona card (Markdown), its renderings, or the evidence of its statements."""
    services = get_cli_context().services()
    _scope(scope)
    store = _store(services)
    try:
        card = store.resolve(version, scope) if version else store.active(scope)
    except PersonaVersionError as exc:
        raise CliError(str(exc)) from exc
    if card is None:
        raise CliError(f"no {scope} card yet; run `twin persona regenerate --stats-only`")
    console = _console()
    console.print(
        f"persona card {card.label} ({card.scope}) {card.id}, {card.created_at:%Y-%m-%d %H:%M} UTC,"
        f" {card.reason}{', in force' if card.active else ''}",
        markup=False,
    )
    if evidence:
        record = store.provenance(card.id)
        if not record:
            console.print(
                "this version has no evidence record (its description was never generated)"
            )
            return
        for statement in record.get("statements", []):
            console.print(
                f"{statement['label']}：{statement['text']}  <- {', '.join(statement['evidence'])}",
                markup=False,
            )
        return
    if full or compact:
        pick = render_full if full else render_compact
        rendered = pick(services, scope, version=card.id)
        if rendered is None:
            raise CliError("the card cannot be rendered")
        console.print(rendered.text, markup=False)
        console.print(
            f"-- {rendered.kind} rendering of {rendered.scope} {card.label}: "
            f"{rendered.tokens}/{rendered.budget} tokens, {rendered.dropped} line(s) left out",
            markup=False,
        )
        return
    console.print(card.text, markup=False)
    counts = []
    for name, pick in (("full", render_full), ("compact", render_compact)):
        rendered = pick(services, scope, version=card.id)
        if rendered is not None:
            counts.append(f"{name} {rendered.tokens}/{rendered.budget} tokens")
    console.print("-- " + "; ".join(counts), markup=False)


@persona_app.command("history")
@command(CommandKind.READ)
def persona_history(
    scope: Annotated[str | None, typer.Option("--scope", help="live or pre_holdout")] = None,
    limit: Annotated[int, typer.Option(min=1, help="Newest versions to list")] = 20,
) -> None:
    """List the stored versions of the card, newest first."""
    services = get_cli_context().services()
    if scope is not None:
        _scope(scope)
    versions = _store(services).history(scope, limit)
    if not versions:
        raise CliError("there is no persona card yet; run `twin persona regenerate`")
    table = Table("#", "version", "scope", "created (UTC)", "reason", "described on", "in force")
    for rank, item in enumerate(versions):
        table.add_row(
            f"~{rank}",
            f"{item.label} {item.id}",
            item.scope,
            f"{item.created_at:%Y-%m-%d %H:%M}",
            item.reason,
            f"{item.described_her_messages:,} msgs" if item.described_her_messages else "-",
            "yes" if item.active else "",
        )
    _console().print(table)


@persona_app.command("diff")
@command(CommandKind.READ)
def persona_diff(
    first: Annotated[str, typer.Argument(help="Older version (vN, ~N or id prefix)")],
    second: Annotated[str, typer.Argument(help="Newer version (vN, ~N or id prefix)")],
    scope: Annotated[str, typer.Option("--scope", help="live or pre_holdout")] = "live",
) -> None:
    """Show what differs between two versions of the card."""
    services = get_cli_context().services()
    _scope(scope)
    store = _store(services)
    try:
        a, b = store.resolve(first, scope), store.resolve(second, scope)
    except PersonaVersionError as exc:
        raise CliError(str(exc)) from exc
    lines = list(
        difflib.unified_diff(
            a.text.splitlines(),
            b.text.splitlines(),
            fromfile=f"{a.label} {a.id}",
            tofile=f"{b.label} {b.id}",
            lineterm="",
        )
    )
    if not lines:
        typer.echo("the two versions are identical")
        return
    console = _console()
    for line in lines:
        console.print(line, markup=False)


@persona_app.command("rollback")
@command(CommandKind.LIGHT)
def persona_rollback(
    version: Annotated[str, typer.Argument(help="vN, ~N or an id prefix")],
    scope: Annotated[str, typer.Option("--scope", help="live or pre_holdout")] = "live",
) -> None:
    """Make an older version of the card the one in force."""
    services = get_cli_context().services()
    _scope(scope)
    store = _store(services)
    try:
        target = store.resolve(version, scope)
        chosen = store.rollback(target.id)
    except PersonaVersionError as exc:
        raise CliError(str(exc)) from exc
    typer.echo(f"the {chosen.scope} card in force is now {chosen.label} ({chosen.id})")
    if chosen.scope == "live":
        result = sync_manual_style(services)
        if result.status == "created":
            typer.echo("the pre-holdout card now carries the style lines of this version")


@persona_app.command("edit")
@command(CommandKind.LIGHT)
def persona_edit() -> None:
    """Edit the [手动] section of the live card in the system editor."""
    services = get_cli_context().services()
    outcome = editing.edit_manual(services, editing.make_editor())
    if outcome.status == "rejected":
        raise CliError(outcome.message)
    typer.echo(outcome.message)


@persona_app.command("regenerate")
@command(CommandKind.HEAVY)
def persona_regenerate(
    scope: Annotated[str, typer.Option("--scope", help="live, pre_holdout or all")] = "all",
    stats_only: Annotated[
        bool, typer.Option("--stats-only", help="Only refresh the statistics rules (free)")
    ] = False,
    foreground: Annotated[
        bool,
        typer.Option(
            "--foreground", help="With --stats-only: do it here when the application is stopped"
        ),
    ] = False,
) -> None:
    """Regenerate the automatic part of the card; the description waits for your approval."""
    services = get_cli_context().services()
    _scope(scope, SCOPE_CHOICES)
    scopes = list(SCOPES) if scope == "all" else [scope]
    if stats_only:
        if foreground and not app_is_running(services):
            for result in refresh_all(services, scope):
                typer.echo(f"{result.scope}: {result.status} - {result.note}")
            return
        job_id = queue_refresh(services, scope=scope, reason="manual")
        typer.echo(f"queued a statistics refresh (job {job_id})")
        return
    try:
        plan = plan_generation(services, scopes, reason="manual")
    except HoldoutError as exc:
        raise CliError(f"{exc}") from exc
    for name in plan.already_queued:
        typer.echo(f"{name}: a description is already waiting for approval or running")
    if plan.batch_id is None:
        return
    table = Table("scope", "job", "estimated cost (USD)")
    for item in plan.scopes:
        table.add_row(item.scope, item.job_id, f"{item.estimated_usd:.3f}")
    _console().print(table)
    limit = services.settings.budget.one_time_usd
    typer.echo(
        f"estimated ${plan.estimated_usd:.2f} in total (an upper bound; one-time limit "
        f"${limit:.2f} per batch)"
    )
    typer.echo(f"approve with: twin jobs approve {plan.batch_id}")
    typer.echo(
        "after the approval the running application writes the card; with the application "
        "stopped use `twin jobs run --until-idle`"
    )


@persona_app.command("status")
@command(CommandKind.READ)
def persona_status() -> None:
    """Show for each scope whether the automatic description is due to be written again."""
    services = get_cli_context().services()
    table = Table("scope", "version", "her messages", "described on", "description due")
    store = _store(services)
    for name in SCOPES:
        due = description_due(services, name, split=False)
        card = store.active(name)
        table.add_row(
            name,
            card.label if card else "-",
            f"{due.her_messages:,}",
            f"{due.described_messages:,}" if due.described_messages else "-",
            f"{'yes' if due.due else 'no'}: {due.reason}",
        )
    _console().print(table)


@persona_app.command("templates")
@command(CommandKind.READ)
def persona_templates() -> None:
    """List the prompt template files and which version of each is in force."""
    services = get_cli_context().services()
    table = Table("template", "files", "in force")
    with services.db.session() as session:
        for name, versions in sorted(template_files().items()):
            active = get_setting(session, ACTIVE_KEY + name)
            table.add_row(
                name,
                ", ".join(f"v{v}" for v in sorted(versions)),
                f"v{active}" if active else f"not loaded yet (v{max(versions)} on first use)",
            )
    _console().print(table)


# --------------------------------------------------------------------------------- rules


def _print_rules(services: Services) -> None:
    rules = read_corrections(services)
    if not rules:
        typer.echo("no rules yet: they come from /不像 and /重来, consolidated once a week")
        return
    console = _console()
    for number, rule in enumerate(rules, start=1):
        console.print(f"{number}. {rule}", markup=False)
    typer.echo(f"{len(rules)} of at most {services.settings.learning.rules_max} rule(s)")


@rules_app.callback()
@command(CommandKind.READ)
def rules_default(ctx: typer.Context) -> None:
    """List the rules (`twin persona rules` alone is `twin persona rules list`)."""
    if ctx.invoked_subcommand is None:
        _print_rules(get_cli_context().services())


@rules_app.command("list")
@command(CommandKind.READ)
def rules_list() -> None:
    """List the rules of [不要这样] with their numbers."""
    _print_rules(get_cli_context().services())


@rules_app.command("delete")
@command(CommandKind.LIGHT)
def rules_delete(
    number: Annotated[int, typer.Argument(min=1, help="Number from `twin persona rules`")],
) -> None:
    """Delete one rule (a new version of the live card; only [不要这样] changes)."""
    services = get_cli_context().services()
    rules = read_corrections(services)
    if not 1 <= number <= len(rules):
        raise CliError(f"there is no rule {number}; there are {len(rules)}", ExitCode.USAGE)
    gone = rules[number - 1]
    card = write_corrections(
        services,
        [rule for index, rule in enumerate(rules, start=1) if index != number],
        reason="rule_deleted",
    )
    typer.echo(f"deleted rule {number}: {gone}")
    typer.echo(f"the live card is now {card.label}")


@rules_app.command("consolidate")
@command(CommandKind.HEAVY)
def rules_consolidate(
    foreground: Annotated[
        bool,
        typer.Option("--foreground", help="Run it here when the application is stopped"),
    ] = False,
) -> None:
    """Queue the consolidation of the rules now (it also runs once a week by itself)."""
    services = get_cli_context().services()
    job_id = queue_rules_job(services)
    if job_id is None:
        typer.echo("a consolidation is already waiting or running")
    else:
        typer.echo(f"queued the consolidation (job {job_id}); it runs off peak")
    if not foreground:
        return
    if app_is_running(services):
        typer.echo("the application is running and will execute the job")
        return
    registry = HandlerRegistry()
    registry.register(RULES_JOB, handle_rules)
    summary = asyncio.run(run_jobs_until_idle(services, registry))
    if summary.failed or summary.retried:
        raise CliError("the consolidation did not finish; see `twin jobs list`")
    typer.echo("done; see `twin persona rules`")
