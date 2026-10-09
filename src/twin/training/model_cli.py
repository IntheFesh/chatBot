"""CLI: ``twin model register|list|show`` (R-SRV-001).

Registering takes the directory ``twin train remote download`` filled (``data/models/<run_id>/``):
it checks the manifest and every sha256, then records the models with the versions they were
trained with.  Enabling, serving and the release gate come with round 14.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from twin.ops.process_model import CliError, CommandKind, ExitCode, command
from twin.services import get_cli_context
from twin.training.registry import (
    ModelView,
    RegistryError,
    get_model,
    list_models,
    register_artifacts,
)

model_app = typer.Typer(help="Registered style models.", no_args_is_help=True)


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _flags(view: ModelView) -> str:
    parts = ["active" if view.active else "", "enabled" if view.enabled else ""]
    gate = {True: "gate passed", False: "gate failed", None: ""}[view.gate_passed]
    return ", ".join(part for part in (*parts, gate) if part) or "-"


@model_app.command("register")
@command(CommandKind.LIGHT)
def model_register_command(
    directory: Annotated[Path, typer.Argument(help="Artifact directory (data/models/<run_id>)")],
    run: Annotated[
        str | None, typer.Option("--run-id", help="Run id, when the manifest has none")
    ] = None,
) -> None:
    """Verify an artifact directory and register its models with their locked versions."""
    services = get_cli_context().services()
    try:
        result = register_artifacts(
            services.db, directory, models_dir=services.paths.models_dir, run_id=run
        )
    except RegistryError as exc:
        raise CliError(str(exc), ExitCode.FAILURE) from exc
    table = Table("id", "kind", "quant", "size (MiB)", "sha256")
    for model in result.models:
        table.add_row(
            model.id, model.kind, model.quant, f"{model.size / (1 << 20):.1f}", model.sha256[:16]
        )
    _console().print(table)
    versions = result.models[0].versions
    typer.echo(
        f"run {result.run_id}: {result.created} registered, {result.unchanged} already registered"
    )
    typer.echo(
        f"locked to template {versions.template_version}, persona card {versions.persona_version}, "
        f"profile {versions.profile_version}, dataset {versions.dataset_version}"
    )


@model_app.command("list")
@command(CommandKind.READ)
def model_list_command() -> None:
    """List the registered models, newest first."""
    models = list_models(get_cli_context().services().db)
    if not models:
        typer.echo("no models registered yet: `twin model register data/models/<run_id>`")
        return
    table = Table("id", "profile", "kind", "quant", "dataset", "state")
    for model in models:
        table.add_row(
            model.id,
            model.profile,
            model.kind,
            model.quant,
            model.versions.dataset_version,
            _flags(model),
        )
    _console().print(table)


@model_app.command("show")
@command(CommandKind.READ)
def model_show_command(
    model_id: Annotated[str, typer.Argument(help="Model id from `twin model list`")],
) -> None:
    """Show one registered model with its locked versions and evaluation numbers."""
    try:
        model = get_model(get_cli_context().services().db, model_id)
    except RegistryError as exc:
        raise CliError(str(exc), ExitCode.FAILURE) from exc
    table = Table("field", "value", show_header=False)
    rows = [
        ("id", model.id),
        ("run", model.run_id),
        ("profile", model.profile),
        ("base model", model.base_model),
        ("kind / quantisation", f"{model.kind} / {model.quant}"),
        ("path", model.path),
        ("sha256", model.sha256),
        ("size (bytes)", f"{model.size:,}"),
        ("template version", model.versions.template_version),
        ("persona card version", model.versions.persona_version),
        ("profile version", model.versions.profile_version),
        ("dataset version", model.versions.dataset_version),
        ("state", _flags(model)),
        ("registered", model.created_at),
    ]
    for key, value in model.eval.items():
        rows.append((f"eval: {key}", "-" if value is None else str(value)))
    for key, value in rows:
        table.add_row(key, value)
    _console().print(table)
