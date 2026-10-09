"""CLI: ``twin llm probe`` (R-LLM-013) and ``twin llm status`` (R-LLM-006, R-LLM-008, R-LLM-012)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from twin.config.runtime import BOT_TIMEZONE
from twin.llm.capabilities import load_capabilities
from twin.llm.probe import (
    GATE_CHECKS,
    LlmProbe,
    ProbeConfig,
    ProbeReport,
    load_probe_report,
    save_probe,
)
from twin.llm.probe_report import render_report, write_report
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.ops.process_model import CliError, CommandKind, command
from twin.schedule.time_service import ConfiguredTimeService
from twin.services import get_cli_context

llm_app = typer.Typer(help="DeepSeek: M0 probe and status.", no_args_is_help=True)

REPORT_NAME = "LLM_REPORT.md"
PROBE_COST_NOTE = "about thirty small requests with synthetic content, well under US$0.10"


def _verdict(passed: bool) -> str:
    return "[green]pass[/green]" if passed else "[red]fail[/red]"


@llm_app.command("probe")
@command(CommandKind.LIGHT)
def llm_probe(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Where to write the report (default: docs/LLM_REPORT.md)"),
    ] = None,
    no_report: Annotated[
        bool, typer.Option("--no-report", help="Only store the result in the database")
    ] = False,
    cache_wait: Annotated[
        float, typer.Option(help="Seconds to wait before repeating the cache request")
    ] = 5.0,
) -> None:
    """Run the M0 checks against the real DeepSeek API and write docs/LLM_REPORT.md.

    Needs the API key (`twin secrets set deepseek_api_key`).  Sends synthetic content only.
    """
    context = get_cli_context()
    services = context.services()
    services.secrets.require(DEEPSEEK_SECRET)  # fails early with the command to run
    if not yes:
        typer.echo(f"This sends {PROBE_COST_NOTE}, billed to the one-time account.")
        if not typer.confirm("Run the probe now?"):
            raise typer.Exit(1)
    runtime = build_llm_runtime(services)
    probe = LlmProbe(
        runtime.client,
        clock=services.clock,
        config=ProbeConfig(cache_wait_s=cache_wait),
        progress=typer.echo,
    )

    async def go() -> ProbeReport:
        try:
            return await probe.run()
        finally:
            await runtime.client.aclose()

    result = asyncio.run(go())
    with services.db.transaction() as session:
        save_probe(session, result, services.clock)
    if not no_report:
        path = report or (services.paths.root / "docs" / REPORT_NAME)
        write_report(path, render_report(result))
        typer.echo(f"report written to {path}")

    table = Table("#", "check", "result", "gate")
    for check in result.checks:
        table.add_row(
            str(check.number),
            check.id,
            _verdict(check.passed) if check.ran else "not run",
            "M0" if check.number in GATE_CHECKS else "measurement",
        )
    Console(highlight=False).print(table)
    typer.echo(f"total cost: ${result.total_cost_usd:.4f}")
    if result.fatal:
        raise CliError(result.fatal)
    json_check = result.check("json_output")
    if json_check.ran and not json_check.metrics.get("thinking_on_ok"):
        raise CliError(
            "STOP: JSON output with thinking enabled is not reliable. The proactive message "
            "planner depends on it; tell the maintainer before building further (see the report).",
        )
    if not result.m0_passed:
        raise CliError("M0 check failed: see the report for the failing checks")
    typer.echo("M0 (DeepSeek) passed")


@llm_app.command("status")
@command(CommandKind.READ)
def llm_status() -> None:
    """Show models, learned capabilities, the last probe and the budget level."""
    context = get_cli_context()
    services = context.services()
    settings = services.settings
    console = Console(highlight=False)
    key_state = "set" if services.secrets.exists(DEEPSEEK_SECRET) else "NOT SET"
    typer.echo(f"API key ({DEEPSEEK_SECRET}): {key_state}")
    typer.echo(
        f"models: chat={settings.deepseek.chat_model} offline={settings.deepseek.offline_model} "
        f"vision={settings.deepseek.vision_model}"
    )
    with services.db.session() as session:
        caps = load_capabilities(session)
        last = load_probe_report(session)
    source = (
        f"measured {caps.measured_at}" if caps.measured else "documented defaults (no probe yet)"
    )
    typer.echo(
        f"capabilities ({source}): detail={caps.detail_supported} gif={caps.gif_supported} "
        f"json_in_thinking={caps.json_in_thinking} image_token_samples={len(caps.image_tokens)}"
    )
    if last is None:
        typer.echo("last probe: never run (twin llm probe)")
    else:
        typer.echo(
            f"last probe: {last.run_id} at {last.finished_at}: "
            f"M0 {'passed' if last.m0_passed else 'NOT passed'}"
        )

    time_service = ConfiguredTimeService(services.clock, lambda: services.runtime.get(BOT_TIMEZONE))
    runtime = build_llm_runtime(services)
    status = runtime.budget.compute()  # read only: no alerts, no events
    table = Table("period", "spent USD", "budget USD", "ratio", "level")
    table.add_row(
        f"today ({status.day})",
        f"{status.daily_spent:.4f}",
        f"{status.daily_budget:.2f}",
        f"{status.daily_ratio:.0%}",
        str(status.daily_level),
    )
    table.add_row(
        status.day.strftime("%Y-%m"),
        f"{status.monthly_spent:.4f}",
        f"{status.monthly_budget:.2f}",
        f"{status.monthly_ratio:.0%}",
        str(status.monthly_level),
    )
    console.print(table)
    start, end = time_service.day_bounds_utc(status.day)
    one_time = runtime.ledger.total_usd(start, end, account="one_time")
    typer.echo(
        f"degradation level: {status.level} (one-time spending today: ${one_time:.4f}, not counted)"
    )
    typer.echo(f"cache hit ratio today: {runtime.ledger.cache_hit_ratio(start, end):.0%}")
