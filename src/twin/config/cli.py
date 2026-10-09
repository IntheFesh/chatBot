"""CLI: ``twin config``, ``twin settings`` and ``twin secrets``."""

import sys
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from twin.config.loader import ConfigError
from twin.config.mask import masked_settings
from twin.config.runtime import (
    SettingSpec,
    SettingValueError,
    registered_settings,
)
from twin.config.secrets import KNOWN_SECRETS, SecretStoreError
from twin.ops.process_model import CliError, CommandKind, ExitCode, command
from twin.services import get_cli_context
from twin.storage.keystore import KeyStore
from twin.storage.migrate import schema_status
from twin.storage.rotate import RotationError, rotate_db_key

config_app = typer.Typer(help="Show the effective configuration.", no_args_is_help=True)
settings_app = typer.Typer(help="Runtime settings stored in the database.", no_args_is_help=True)
secrets_app = typer.Typer(help="Secrets in the credential store.", no_args_is_help=True)


def _console() -> Console:
    return Console(highlight=False)


# ------------------------------------------------------------------- config


@config_app.command("show")
@command(CommandKind.READ, consent=False)
def config_show() -> None:
    """Print the effective configuration (secrets and personal ids masked)."""
    settings = get_cli_context().settings()
    typer.echo(yaml.safe_dump(masked_settings(settings), allow_unicode=True, sort_keys=False))


# ----------------------------------------------------------------- settings


def _services_for_runtime() -> Any:
    return get_cli_context().services()


def _find_spec(key: str) -> SettingSpec[Any]:
    specs = registered_settings()
    if key not in specs:
        raise CliError(f"unknown setting {key!r}; known: {', '.join(sorted(specs))}")
    return specs[key]


@settings_app.command("list")
@command(CommandKind.READ)
def settings_list() -> None:
    """List runtime settings with their current values."""
    services = _services_for_runtime()
    table = Table("key", "value", "description")
    for key, spec in sorted(registered_settings().items()):
        table.add_row(key, str(services.runtime.get(spec)), spec.description)
    _console().print(table)


@settings_app.command("set")
@command(CommandKind.LIGHT)
def settings_set(
    key: Annotated[str, typer.Argument(help="Setting key, e.g. time.bot_timezone")],
    value: Annotated[str, typer.Argument(help="New value (YAML scalar)")],
) -> None:
    """Change a runtime setting; a running application picks it up within 2 seconds."""
    spec = _find_spec(key)
    services = _services_for_runtime()
    try:
        parsed = yaml.safe_load(value)
        changed = services.runtime.set(spec, parsed, by="cli")
    except (yaml.YAMLError, SettingValueError) as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    typer.echo(f"{key} = {parsed!r}" if changed else f"{key} unchanged")


@settings_app.command("history")
@command(CommandKind.READ)
def settings_history(key: Annotated[str, typer.Argument(help="Setting key")]) -> None:
    """Show who changed a setting, when, and from what to what."""
    spec = _find_spec(key)
    services = _services_for_runtime()
    table = Table("at (UTC)", "by", "old", "new")
    for change in services.runtime.history(spec):
        table.add_row(change.at, change.by, repr(change.old), repr(change.new))
    _console().print(table)


# ------------------------------------------------------------------ secrets


def _notify_running_app() -> None:
    """Bump ``state_version`` (inside a LIGHT command) if the database is initialised."""
    context = get_cli_context()
    if schema_status(context.paths().db_path).ok:
        with context.services().db.transaction():
            pass


def _known(name: str) -> str:
    if name not in KNOWN_SECRETS:
        raise CliError(
            f"unknown secret {name!r}; known: {', '.join(sorted(KNOWN_SECRETS))}", ExitCode.USAGE
        )
    return name


@secrets_app.command("set")
@command(CommandKind.LIGHT, consent=False)
def secrets_set(
    name: Annotated[str, typer.Argument(help="Secret name, see `twin secrets list`")],
    from_stdin: Annotated[
        bool, typer.Option("--stdin", help="Read the value from standard input (one line)")
    ] = False,
) -> None:
    """Store a secret in the credential store (never echoed, never logged)."""
    _known(name)
    if from_stdin:
        value = sys.stdin.readline().rstrip("\r\n")
    else:
        value = typer.prompt(f"{KNOWN_SECRETS[name]}", hide_input=True, confirmation_prompt=True)
    if not value:
        raise CliError("empty value; nothing stored", ExitCode.USAGE)
    get_cli_context().secret_store().set(name, value)
    typer.echo(f"stored {name}")
    _notify_running_app()


@secrets_app.command("delete")
@command(CommandKind.LIGHT, consent=False)
def secrets_delete(name: Annotated[str, typer.Argument(help="Secret name")]) -> None:
    """Remove a secret from the credential store."""
    _known(name)
    removed = get_cli_context().secret_store().delete(name)
    typer.echo(f"deleted {name}" if removed else f"{name} was not set")
    _notify_running_app()


@secrets_app.command("list")
@command(CommandKind.READ, consent=False)
def secrets_list() -> None:
    """List secret names and whether they are set (values are never shown)."""
    store = get_cli_context().secret_store()
    table = Table("name", "set", "purpose")
    for name, description in sorted(KNOWN_SECRETS.items()):
        table.add_row(name, "yes" if store.exists(name) else "no", description)
    keystore = KeyStore(store)
    if keystore.exists():
        ring = keystore.load()
        for key_id in ring.key_ids:
            state = "current" if key_id == ring.current_id else "retired"
            table.add_row(f"db-key-{key_id}", "yes", f"database master key ({state})")
    else:
        table.add_row("db-key-*", "no", "database master key (created on first start)")
    _console().print(table)
    typer.echo(f"credential store: {store.info.name} - {store.info.detail}")


@secrets_app.command("check")
@command(CommandKind.READ, consent=False)
def secrets_check(name: Annotated[str, typer.Argument(help="Secret name")]) -> None:
    """Exit 0 if the secret is set, 1 if not (the value is never printed)."""
    _known(name)
    if get_cli_context().secret_store().exists(name):
        typer.echo(f"{name}: set")
        return
    typer.echo(f"{name}: not set")
    raise typer.Exit(1)


@secrets_app.command("rotate-db-key")
@command(CommandKind.EXCLUSIVE)
def secrets_rotate_db_key() -> None:
    """Re-encrypt all data with a new database key (resumable; stop the application first)."""
    services = get_cli_context().services()
    typer.echo("rotating the database key; this can be interrupted and run again")

    def progress(table: str, rows: int) -> None:
        typer.echo(f"  {table}: {rows} row(s) scanned")

    try:
        report = rotate_db_key(
            services.db,
            services.keystore,
            services.keyring,
            services.clock,
            services.media,
            on_batch=progress,
        )
    except (RotationError, SecretStoreError, ConfigError) as exc:
        raise CliError(str(exc)) from exc
    resumed = " (resumed)" if report.resumed else ""
    typer.echo(
        f"done{resumed}: new key id {report.new_key_id}; {report.values_reencrypted} value(s) and "
        f"{report.media_reencrypted} media file(s) re-encrypted; retired keys "
        f"{report.retired_key_ids or 'none'} are kept for old backups"
    )
