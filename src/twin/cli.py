"""``twin`` command line entry point."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from twin import __version__
from twin.app import ComponentStartError, ShutdownSignals
from twin.config.cli import config_app, secrets_app, settings_app
from twin.config.loader import ConfigError, parse_overrides
from twin.config.mask import masked_settings
from twin.config.settings import ConfigFileError
from twin.llm.cli import llm_app
from twin.ops.cli import jobs_app
from twin.ops.components import build_application
from twin.ops.console import ensure_utf8
from twin.ops.doctor import CheckStatus, DoctorContext, exit_code, run_checks
from twin.ops.instance_lock import LOCK_RUN, LOCK_SUPERVISOR
from twin.ops.logging import configure_logging, get_logger, shutdown_logging
from twin.ops.power import default_power_manager
from twin.ops.process_model import CliError, CommandKind, command
from twin.services import CliContext, Services, get_cli_context, set_cli_context
from twin.storage.cli import db_app

app = typer.Typer(
    name="twin",
    help="wechat-twin: a single-user WeChat persona bot (local-first, encrypted storage).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(config_app, name="config")
app.add_typer(settings_app, name="settings")
app.add_typer(secrets_app, name="secrets")
app.add_typer(db_app, name="db")
app.add_typer(jobs_app, name="jobs")
app.add_typer(llm_app, name="llm")

log = get_logger("twin.cli")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"wechat-twin {__version__}")
        raise typer.Exit


@app.callback()
def main(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Configuration file (default: config/config.yaml)"),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a setting for this run: key.path=value (repeatable)"),
    ] = None,
    log_level: Annotated[
        str, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR")
    ] = "INFO",
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show the version"
        ),
    ] = False,
) -> None:
    ensure_utf8()
    try:
        overrides = parse_overrides(set_ or [])
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(7) from None
    previous = get_cli_context()
    set_cli_context(
        CliContext(
            config_path=config,
            overrides=overrides,
            log_level=log_level,
            secrets=previous.secrets,
        )
    )


# ---------------------------------------------------------------------- run


async def _serve(services: Services) -> None:
    application, _watcher = build_application(services)
    stop = asyncio.Event()
    signals = ShutdownSignals(asyncio.get_running_loop(), stop)
    power = default_power_manager()
    power.start()
    try:
        await application.run(stop, signals=signals)
    finally:
        power.stop()


@app.command("run")
@command(CommandKind.EXCLUSIVE, acquires=(LOCK_RUN,), tolerates=(LOCK_SUPERVISOR,))
def run() -> None:
    """Start the application (job queue, state watcher, heartbeat)."""
    context = get_cli_context()
    services = context.services()
    log_path = configure_logging(services.paths.logs_dir, level=context.log_level, role="run")
    try:
        typer.echo("effective configuration:")
        typer.echo(
            yaml.safe_dump(masked_settings(services.settings), allow_unicode=True, sort_keys=False)
        )
        seeded = services.runtime.initialize()
        log.info("starting", pid_log=str(log_path), seeded_settings=len(seeded))
        asyncio.run(_serve(services))
    except ComponentStartError as exc:
        raise CliError(str(exc)) from exc
    finally:
        shutdown_logging()


# ------------------------------------------------------------------- doctor


_STATUS_STYLE = {CheckStatus.OK: "green", CheckStatus.WARN: "yellow", CheckStatus.FAIL: "red"}


@app.command("doctor")
@command(CommandKind.READ, consent=False)
def doctor() -> None:
    """Check the installation: Python, dependencies, time zones, keyring, disk, database."""
    context = get_cli_context()
    settings = None
    error = None
    try:
        settings = context.settings()
    except (ConfigError, ConfigFileError) as exc:
        error = str(exc)
    results = run_checks(
        DoctorContext(
            settings=settings,
            settings_error=error,
            secrets=context.secrets,
            root=context.paths().root if settings else None,
        )
    )
    table = Table("check", "status", "detail")
    for result in results:
        style = _STATUS_STYLE[result.status]
        table.add_row(result.name, f"[{style}]{result.status.value}[/{style}]", result.detail)
    console = Console(highlight=False)
    console.print(table)
    for result in results:
        if result.hint and result.status is not CheckStatus.OK:
            console.print(f"[bold]{result.name}[/bold]: {result.hint}", markup=True)
    code = exit_code(results)
    if code:
        raise typer.Exit(code)
