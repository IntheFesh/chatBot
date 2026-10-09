"""CLI: ``twin db``."""

import typer

from twin.ops.process_model import CommandKind, command
from twin.services import get_cli_context
from twin.storage import migrate

db_app = typer.Typer(help="Database migrations.", no_args_is_help=True)


@db_app.command("upgrade")
@command(CommandKind.EXCLUSIVE, consent=False)
def db_upgrade() -> None:
    """Create or migrate the database to the latest schema (stop the application first)."""
    paths = get_cli_context().paths()
    paths.ensure()
    before = migrate.schema_status(paths.db_path)
    migrate.upgrade(paths.db_path)
    after = migrate.schema_status(paths.db_path)
    typer.echo(f"database {paths.db_path}: {before.current or 'new'} -> {after.current}")


@db_app.command("status")
@command(CommandKind.READ, consent=False)
def db_status() -> None:
    """Show the applied and the latest schema revision."""
    paths = get_cli_context().paths()
    status = migrate.schema_status(paths.db_path)
    typer.echo(
        f"{paths.db_path}: {status.state.value} (applied {status.current}, latest {status.head})"
    )
    typer.echo(status.hint())
