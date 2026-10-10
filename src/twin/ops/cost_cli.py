"""CLI: ``twin cost report [--month YYYY-MM] [--json]`` (R-OPS-005).  READ."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from twin.llm.ledger import LedgerStore
from twin.ops.cost import (
    CostReportError,
    build_report,
    parse_month,
    render_text,
    report_to_json,
)
from twin.ops.process_model import CliError, CommandKind, ExitCode, command
from twin.schedule.service import time_service_for
from twin.services import get_cli_context

cost_app = typer.Typer(help="What the model calls cost.", no_args_is_help=True)


@cost_app.command("report")
@command(CommandKind.READ, consent=False)
def cost_report(
    month: Annotated[
        str | None, typer.Option("--month", help="Month as YYYY-MM (default: this month)")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON")] = False,
) -> None:
    """The month by day, purpose and model, cache hits, peak share, against the budget."""
    services = get_cli_context().services()
    time = time_service_for(services)
    try:
        start = parse_month(month) if month else time.local_date().replace(day=1)
    except CostReportError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    ledger = LedgerStore(services.db, services.clock, time)
    report = build_report(ledger, time, services.settings.budget, start)
    if json_output:
        typer.echo(json.dumps(report_to_json(report), ensure_ascii=False, indent=2))
    else:
        typer.echo(render_text(report))
