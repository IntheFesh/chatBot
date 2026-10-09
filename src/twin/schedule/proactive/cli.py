"""CLI: ``twin proactive log`` (R-PRO-008).

``log`` lists what the proactive scheduler decided in the last days, one row per decision: the
local time, the kind, the outcome and its reason, her state, the chase number and how many
bubbles went out.  It only reads (READ) and **shows no words**: the content of the messages and the
planner's reasons are sealed chat content.  ``--show-text`` adds them to the table, after two
confirmations.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

import typer
from rich.console import Console

from twin.eval.proactive_report import print_log
from twin.ops.process_model import CommandKind, command
from twin.schedule.proactive.store import ProactiveLogStore
from twin.schedule.service import schedule_kit
from twin.services import get_cli_context

proactive_app = typer.Typer(help="Proactive messages: the audit log.", no_args_is_help=True)

CONFIRM_FIRST = "Show the words of the messages on this screen? (they are chat content)"
CONFIRM_SECOND = "Really show them? Anyone who can see this screen can read them"


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


@proactive_app.command("log")
@command(CommandKind.READ)
def proactive_log(
    days: Annotated[int, typer.Option("--days", min=1, help="How many local days back")] = 7,
    show_text: Annotated[
        bool, typer.Option("--show-text", help="Also show the words (asks twice)")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Newest rows to show")] = 200,
) -> None:
    """List the proactive scheduler's decisions of the last days (no words unless asked)."""
    services = get_cli_context().services()
    if show_text:
        typer.confirm(CONFIRM_FIRST, abort=True)
        typer.confirm(CONFIRM_SECOND, abort=True)
    time = schedule_kit(services).time
    today = time.local_date()
    entries = ProactiveLogStore(services.db, services.clock).entries(
        first_day=today - timedelta(days=days - 1), last_day=today, with_text=show_text
    )
    print_log(_console(), entries[-limit:], show_text=show_text)
