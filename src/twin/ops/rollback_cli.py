"""CLI: ``twin rollback profile|persona|prompt-template|style-model`` (R-OPS-010).

All LIGHT.  Each command calls the rollback of the module that owns the versions
(:mod:`twin.ops.rollback`) and records an audit entry; a running application notices the change
within two seconds.
"""

from __future__ import annotations

from typing import Annotated

import typer

from twin.ops.process_model import CliError, CommandKind, ExitCode, command
from twin.ops.rollback import (
    RollbackError,
    RollbackResult,
    record_audit,
    rollback_persona,
    rollback_profile,
    rollback_prompt_template,
    rollback_style_model,
)
from twin.services import get_cli_context

rollback_app = typer.Typer(
    help="Go back to an earlier version of the profile, persona card, a template or the model.",
    no_args_is_help=True,
)


def _finish(result: RollbackResult) -> None:
    services = get_cli_context().services()
    record_audit(services, result)
    typer.echo(f"{result.subject}: {result.previous or '(none)'} -> {result.current}")
    if result.detail:
        typer.echo(f"  {result.detail}")
    typer.echo("  recorded in the rollback audit log (setting ops.rollback.log)")


@rollback_app.command("profile")
@command(CommandKind.LIGHT)
def rollback_profile_command(
    version: Annotated[str, typer.Argument(help="Version id (prefix) or ~N")],
    scope: Annotated[str | None, typer.Option("--scope", help="live or pre_holdout")] = None,
) -> None:
    """Make an older statistical profile (and its routine model) the active one."""
    services = get_cli_context().services()
    try:
        _finish(rollback_profile(services, version, scope))
    except RollbackError as exc:
        raise CliError(str(exc)) from exc


@rollback_app.command("persona")
@command(CommandKind.LIGHT)
def rollback_persona_command(
    version: Annotated[str, typer.Argument(help="vN, ~N or an id prefix")],
    scope: Annotated[str, typer.Option("--scope", help="live or pre_holdout")] = "live",
) -> None:
    """Make an older persona card the one in force."""
    if scope not in ("live", "pre_holdout"):
        raise CliError("--scope must be live or pre_holdout", ExitCode.USAGE)
    services = get_cli_context().services()
    try:
        _finish(rollback_persona(services, version, scope))
    except RollbackError as exc:
        raise CliError(str(exc)) from exc


@rollback_app.command("prompt-template")
@command(CommandKind.LIGHT)
def rollback_template_command(
    name: Annotated[str, typer.Argument(help="Template name, e.g. memory_extract")],
    version: Annotated[int, typer.Argument(help="Version number to go back to, e.g. 1")],
) -> None:
    """Make an older version of a prompt template the one in force."""
    services = get_cli_context().services()
    try:
        _finish(rollback_prompt_template(services, name, version))
    except RollbackError as exc:
        raise CliError(str(exc)) from exc


@rollback_app.command("style-model")
@command(CommandKind.LIGHT)
def rollback_model_command(
    model: Annotated[str, typer.Argument(help="Model id (see `twin model list`) or its run id")],
) -> None:
    """Make an earlier registered style model the one in use."""
    services = get_cli_context().services()
    try:
        _finish(rollback_style_model(services, model))
    except RollbackError as exc:
        raise CliError(str(exc)) from exc
