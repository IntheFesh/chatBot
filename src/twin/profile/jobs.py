"""The ``profile_rebuild`` job: recompute the profile and the routine in the job queue.

The computation is synchronous work over the stored messages, so it runs in a worker thread.
When the job was queued by an import, the style-change section of the newest import report is
rewritten afterwards.  A sleep time that falls in the daytime raises an alert (the usual cause
is a wrong ``time.source_timezone``); the alert carries clock times and a zone name only.
"""

from __future__ import annotations

import asyncio

from twin.ops.jobs import JobContext, job_handler
from twin.ops.logging import get_logger
from twin.profile.builder import BuildReport, rebuild
from twin.profile.queue import PROFILE_JOB
from twin.profile.report_section import refresh_latest_import_report
from twin.services import Services

log = get_logger("twin.profile")
TIMEZONE_ALERT = "routine_timezone"


def raise_routine_alerts(services: Services, report: BuildReport) -> None:
    for result in report.results:
        if result.status != "created" or result.scope != "live":
            continue
        suspicious = [w for w in result.warnings if "source_timezone" in w]
        if suspicious:
            services.alerts.raise_alert(
                TIMEZONE_ALERT,
                "推断的睡眠时段落在当地白天，请核对 time.source_timezone",
                detail={"scope": result.scope, "warnings": suspicious},
                dedup_key=f"{TIMEZONE_ALERT}-{result.scope}",
            )


def run_rebuild(services: Services, scope: str, reason: str, force: bool) -> BuildReport:
    """The synchronous body of the job (also used by the foreground command)."""
    report = rebuild(services, scope, reason=reason, force=force)
    raise_routine_alerts(services, report)
    if reason == "import":
        refresh_latest_import_report(services)
    for result in report.results:
        log.info(
            "profile_rebuilt",
            scope=result.scope,
            status=result.status,
            her_messages=result.her_messages,
            changes=len(result.changes),
        )
    return report


@job_handler(PROFILE_JOB)
async def handle_profile_rebuild(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("the profile job needs the services container")
    payload = ctx.job.payload
    await asyncio.to_thread(
        run_rebuild,
        services,
        str(payload.get("scope", "all")),
        str(payload.get("reason", "manual")),
        bool(payload.get("force", False)),
    )
