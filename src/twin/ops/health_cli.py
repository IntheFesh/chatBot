"""CLI: ``twin health [--json]`` (R-OPS-003).

READ.  The checks that need nothing but the database, the disk and the credential store (the
channel, disk, queue, backup and budget) are made afresh, here, now.  The ones that only the
running application can see (DeepSeek errors and the circuit breaker, the style model's endpoint,
the application's components) are taken from the newest ``health_snapshots`` row, marked with
its age; if the application is not running, or its newest snapshot is old, they are reported as
unknown instead of being guessed.  The exit code is 1 when a check fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from twin.ops.health import HealthCheck, HealthCollector, HealthLevel, HealthReport
from twin.ops.process_model import CommandKind, command
from twin.ops.service import running
from twin.services import Services, get_cli_context
from twin.storage.ops_models import HealthSnapshot

IN_APPLICATION = ("deepseek", "style_model", "app")
STYLES = {HealthLevel.OK: "green", HealthLevel.WARN: "yellow", HealthLevel.FAIL: "red"}


@dataclass(frozen=True)
class SnapshotView:
    """The part of the newest ``health_snapshots`` row that ``twin health`` shows."""

    at: datetime
    started_at: datetime | None
    checks: dict[str, Any]


def newest_snapshot(services: Services) -> SnapshotView | None:
    with services.db.session() as session:
        row = session.scalars(
            select(HealthSnapshot).order_by(HealthSnapshot.at.desc()).limit(1)
        ).first()
        return None if row is None else SnapshotView(row.at, row.started_at, dict(row.checks))


def merged_report(
    services: Services, snapshot: SnapshotView | None, supervisor: bool, run: bool
) -> tuple[HealthReport, datetime | None]:
    """Fresh checks plus the application's own, from its newest snapshot (see the module text)."""
    now = services.clock.now_utc()
    # the poll of a running application gets the grace period of its start; a second process
    # has no start to wait for - a poll that stopped is a poll that stopped
    started = snapshot.started_at if run and snapshot is not None and snapshot.started_at else None
    collector = HealthCollector(services, started_at=started or datetime.min.replace(tzinfo=UTC))
    stored, channel_ok, last_ok = collector.stored_checks()
    interval = services.settings.ops.health.interval_s
    fresh = snapshot is not None and now - snapshot.at <= timedelta(seconds=3 * interval)
    checks = list(stored)
    if run:
        where = "supervised" if supervisor else "started by hand"
        checks.append(HealthCheck("process", HealthLevel.OK, f"twin run is running ({where})"))
    else:
        checks.append(
            HealthCheck(
                "process",
                HealthLevel.WARN,
                "twin run is not running (`twin service status`, `twin service start`)",
            )
        )
    for name in IN_APPLICATION:
        if snapshot is not None and fresh and name in snapshot.checks:
            checks.append(HealthCheck.from_json(name, snapshot.checks[name]))
        else:
            checks.append(
                HealthCheck(
                    name,
                    HealthLevel.OK,
                    "unknown: only the running application can tell"
                    if not run
                    else "unknown: no recent snapshot yet",
                )
            )
    return HealthReport(now, tuple(checks), channel_ok, last_ok), snapshot.at if snapshot else None


@command(CommandKind.READ, consent=False)
def health_command(
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON")] = False,
) -> None:
    """How the bot is doing: channel, DeepSeek, style model, disk, queue, backup, budget."""
    services = get_cli_context().services()
    supervisor, run = running(services.paths.locks_dir)
    report, snapshot_at = merged_report(services, newest_snapshot(services), supervisor, run)
    age = (report.at - snapshot_at).total_seconds() if snapshot_at else None
    if json_output:
        payload = report.to_json()
        payload["snapshot_at"] = snapshot_at.isoformat() if snapshot_at else None
        payload["snapshot_age_s"] = None if age is None else round(age)
        payload["running"] = {"supervisor": supervisor, "run": run}
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        table = Table("check", "status", "detail")
        for check in report.checks:
            style = STYLES[check.level]
            table.add_row(check.name, f"[{style}]{check.level.value}[/{style}]", check.detail)
        console = Console(highlight=False)
        console.print(table)
        snapshot_text = "no snapshot yet" if age is None else f"newest snapshot {age:.0f} s old"
        typer.echo(f"overall: {report.status} ({snapshot_text})")
    if any(check.level is HealthLevel.FAIL for check in report.checks):
        raise typer.Exit(1)
