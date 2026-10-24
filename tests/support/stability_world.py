"""Health snapshots, alerts and evaluation runs of an observation week (round 12 tests).

Everything is synthetic: a bot that was started by the scheduled task, a process that keeps
writing a snapshot every minute, a network drill, a blind test, a learning job, an import.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from sqlalchemy import insert

from tests.support.clock import ManualClock
from twin.eval.store import EvalStore, NewItem
from twin.ops.stability import Snap
from twin.services import Services
from twin.storage.chat_models import ImportRun
from twin.storage.models import Job
from twin.storage.ops_models import HealthSnapshot

INTERVAL = 60.0
PROCESS = datetime(2026, 10, 1, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 9, 12, tzinfo=UTC)  # the default time of the test clock
WEEK_START = NOW - timedelta(days=7)
DRILL = NOW - timedelta(days=3)  # a network drill started at this moment


def steady(
    start: datetime,
    end: datetime,
    *,
    launch: str = "task",
    started_at: datetime | None = PROCESS,
    step: float = INTERVAL,
) -> list[Snap]:
    """A snapshot every ``step`` seconds from ``start`` to ``end``, the channel working."""
    out = []
    moment = start
    while moment <= end:
        out.append(Snap(moment, started_at, launch, True, moment))
        moment += timedelta(seconds=step)
    return out


def without(snaps: list[Snap], start: datetime, end: datetime) -> list[Snap]:
    """The snapshots minus those strictly between ``start`` and ``end`` (the process was down)."""
    return [s for s in snaps if not (start < s.at < end)]


def channel_down(
    snaps: list[Snap], start: datetime, end: datetime, *, last_ok: datetime
) -> list[Snap]:
    """The snapshots in ``[start, end)`` say the channel is down since ``last_ok``."""
    return [
        replace(s, channel_ok=False, channel_last_ok_at=last_ok) if start <= s.at < end else s
        for s in snaps
    ]


def store_snapshots(services: Services, snaps: list[Snap]) -> None:
    """Write the snapshots into ``health_snapshots`` (one bulk insert)."""
    rows = [
        {
            "id": f"S{number:025d}",
            "at": s.at,
            "status": "ok",
            "started_at": s.started_at,
            "launch": s.launch,
            "pid": 1,
            "channel_ok": s.channel_ok,
            "channel_last_ok_at": s.channel_last_ok_at,
            "checks": {},
            "created_at": s.at,
            "updated_at": s.at,
        }
        for number, s in enumerate(snaps)
    ]
    with services.db.transaction(bump_state=False) as session:
        session.execute(insert(HealthSnapshot), rows)


def announce(
    services: Services, clock: ManualClock, at: datetime, category: str, *, after_s: float
) -> None:
    """Raise the alert at ``at`` and let its notification be shown ``after_s`` seconds later."""
    clock.set_time(at)
    services.alerts.raise_alert(
        category, "synthetic", severity="critical", dedup_key=f"{category}-{at}"
    )
    for queued in services.alerts.claim_due():
        clock.set_time(at + timedelta(seconds=after_s))
        services.alerts.finish_toast(queued.id, ok=True)


def blind_run(
    services: Services,
    clock: ManualClock,
    at: datetime,
    *,
    backend: str = "deepseek",
    valid: int = 60,
    correct: int = 30,
    keys: str = "k",
) -> str:
    """A blind run made at ``at`` whose judged pairs are exactly these numbers."""
    clock.set_time(at)
    store = EvalStore(services.db, services.clock)
    run = store.create_run("blind", mode="holdout", backends=[backend], status="running")
    store.add_items(
        run.id,
        [
            NewItem(f"{keys}{n}", backend, at, {}, None, "night", "short", n % 2 == 0)
            for n in range(valid)
        ],
    )
    for found in store.items(run.id):
        store.save_generated(found.id, {"bot": {"lines": [], "quote": None}}, cost_usd=0.0)
        store.judge(found.id, "correct" if found.seq < correct else "wrong", score=1.0)
    return run.id


def finished_job(services: Services, job_type: str, at: datetime) -> None:
    with services.db.transaction(bump_state=False) as session:
        session.add(Job(type=job_type, payload={}, status="done", started_at=at, finished_at=at))


def finished_import(services: Services, at: datetime) -> None:
    with services.db.transaction(bump_state=False) as session:
        session.add(
            ImportRun(
                fingerprint="f" * 64,
                source_dir="/synthetic/export",
                target_username="synthetic",
                status="done",
                phase="done",
                stats={},
                started_at=at,
                finished_at=at,
            )
        )


@dataclass
class Observation:
    """What :func:`observed_week` put in place (to change one thing at a time)."""

    snaps: list[Snap]
    drill_alert: bool = True


def observed_week(
    services: Services,
    clock: ManualClock,
    *,
    snaps: list[Snap] | None = None,
    drill: bool = True,
    drill_minutes: int = 15,
    alert_after_min: float = 6,
    alert_toast_s: float = 20,
    blind: bool = True,
    blind_valid: int = 60,
    blind_correct: int = 30,
    blind_at: datetime | None = None,
    learning: bool = True,
    imported: bool = True,
) -> None:
    """Everything the M4 judge needs for a pass, with switches to take one thing away."""
    services.settings.ops.health.interval_s = INTERVAL
    week = steady(WEEK_START, NOW) if snaps is None else snaps
    if drill:
        week = channel_down(
            week,
            DRILL,
            DRILL + timedelta(minutes=drill_minutes),
            last_ok=DRILL - timedelta(seconds=30),
        )
    store_snapshots(services, week)
    if drill and alert_after_min >= 0:
        announce(
            services,
            clock,
            DRILL + timedelta(minutes=alert_after_min),
            "channel.poll_failing",
            after_s=alert_toast_s,
        )
    if blind:
        blind_run(
            services,
            clock,
            blind_at or NOW - timedelta(days=1),
            valid=blind_valid,
            correct=blind_correct,
        )
    if learning:
        finished_job(services, "learning_rules", NOW - timedelta(days=2))
    if imported:
        finished_import(services, NOW - timedelta(days=2, hours=1))
    clock.set_time(NOW)
