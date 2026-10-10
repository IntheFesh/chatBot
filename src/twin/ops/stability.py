"""The stability report: did the bot run for days without help? (R-EVAL-006).

``twin eval stability --days 7`` reads what the running bot left behind and answers with numbers:

* **availability** - the health snapshots come once a minute while the process lives
  (``health_snapshots``); a gap longer than 2.5 intervals is time the bot was **not available**
  (the process was down, restarting, or the machine slept), and the unavailable time is what the
  gaps exceed one interval by, plus the time since the last snapshot if that is a gap too;
* **continuity** - how long the process ran without a restart (the longest run), how many
  processes there were in the window (each new ``started_at`` after the first is a restart) and
  why each one came (the supervisor's restart history; a restart nobody recorded is "unknown");
* **how it was started** - the number of snapshots by launch kind: from the scheduled task, by
  hand, or without a supervisor;
* **channel outages** - a run of snapshots in which the chat channel was not working is one
  outage; it starts when the last long poll succeeded (``channel_last_ok_at``), ends at the first
  snapshot where it works again; for each outage the alert that announced it (``login_lost`` or
  ``channel_poll_stale``) and the **delay** from the start of the outage to the notification
  (``toast_at``; the first channel that delivered if there was no toast) - it must be at most
  ten minutes;
* the **drill**: an outage of at least ten minutes counts as the network drill the plan asks for
  (``twin ops drill network``).

The report is a plain dictionary (:class:`StabilityReport`) and is stored in
``eval_runs(kind=stability)``; the M4 judge (:mod:`twin.ops.gate_m4`) reads it from there.  Nothing
here is invented: a window without snapshots is reported as such, and the gate does not pass on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

from sqlalchemy import select

from twin.eval.store import EvalStore, RunView
from twin.ops.alerts import AlertView
from twin.ops.supervise import RestartLog
from twin.services import Services
from twin.storage.ops_models import HealthSnapshot

ALERT_LATENCY_LIMIT = timedelta(minutes=10)
UNAVAILABLE_LIMIT = timedelta(minutes=10)
DRILL_MIN = timedelta(minutes=10)
GAP_FACTOR = 2.5
CHANNEL_CATEGORIES = ("login_lost", "channel_poll_stale")
RESTART_MATCH = timedelta(minutes=8)  # a supervisor record this long before a new process


@dataclass(frozen=True)
class Snap:
    """The part of a ``health_snapshots`` row the report uses."""

    at: datetime
    started_at: datetime | None
    launch: str
    channel_ok: bool | None
    channel_last_ok_at: datetime | None


@dataclass(frozen=True)
class Outage:
    """A stretch in which the chat channel did not work."""

    start: datetime
    end: datetime
    ongoing: bool
    alert_category: str | None
    alert_at: datetime | None
    latency_s: float | None

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()

    def to_json(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_s": round(self.duration_s, 1),
            "ongoing": self.ongoing,
            "alert_category": self.alert_category,
            "alert_at": self.alert_at.isoformat() if self.alert_at else None,
            "latency_s": None if self.latency_s is None else round(self.latency_s, 1),
        }


@dataclass(frozen=True)
class Restart:
    at: datetime
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {"at": self.at.isoformat(), "reason": self.reason}


@dataclass(frozen=True)
class StabilityReport:
    """The numbers of one window (see the module description)."""

    window_start: datetime
    window_end: datetime
    days: float
    interval_s: float
    snapshots: int
    first_snapshot_at: datetime | None
    last_snapshot_at: datetime | None
    unavailable_s: float
    longest_gap_s: float
    processes: int
    longest_run_s: float
    restarts: tuple[Restart, ...]
    launches: Mapping[str, int]
    outages: tuple[Outage, ...]
    system_resumes: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- derived -------------------------------------------------------------------

    @property
    def covers_window(self) -> bool:
        """Snapshots exist from the start of the window (within the gap tolerance)."""
        first = self.first_snapshot_at
        return first is not None and first <= self.window_start + timedelta(
            seconds=GAP_FACTOR * self.interval_s
        )

    @property
    def max_latency_s(self) -> float | None:
        values = [o.latency_s for o in self.outages if o.latency_s is not None]
        return max(values) if values else None

    @property
    def missed_alerts(self) -> int:
        return sum(1 for o in self.outages if o.latency_s is None)

    @property
    def drill(self) -> Outage | None:
        """The longest outage of at least ten minutes (the network drill), if there was one."""
        long = [o for o in self.outages if o.duration_s >= DRILL_MIN.total_seconds()]
        return max(long, key=lambda o: o.duration_s) if long else None

    def to_json(self) -> dict[str, Any]:
        drill = self.drill
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "days": self.days,
            "interval_s": self.interval_s,
            "snapshots": self.snapshots,
            "first_snapshot_at": self.first_snapshot_at.isoformat()
            if self.first_snapshot_at
            else None,
            "last_snapshot_at": self.last_snapshot_at.isoformat()
            if self.last_snapshot_at
            else None,
            "covers_window": self.covers_window,
            "unavailable_s": round(self.unavailable_s, 1),
            "longest_gap_s": round(self.longest_gap_s, 1),
            "processes": self.processes,
            "longest_run_s": round(self.longest_run_s, 1),
            "restarts": [r.to_json() for r in self.restarts],
            "launches": dict(self.launches),
            "outages": [o.to_json() for o in self.outages],
            "max_alert_latency_s": None
            if self.max_latency_s is None
            else round(self.max_latency_s, 1),
            "missed_alerts": self.missed_alerts,
            "drill": drill.to_json() if drill else None,
            "system_resumes": self.system_resumes,
            "notes": list(self.notes),
        }


# ------------------------------------------------------------------ the computation


def _availability(snaps: Sequence[Snap], now: datetime, interval: float) -> tuple[float, float]:
    """``(unavailable seconds, longest gap in seconds)`` between the snapshots and up to now."""
    tolerance = GAP_FACTOR * interval
    unavailable = longest = 0.0
    for before, after in pairwise(snaps):
        gap = (after.at - before.at).total_seconds()
        longest = max(longest, gap)
        if gap > tolerance:
            unavailable += gap - interval
    if snaps:
        tail = (now - snaps[-1].at).total_seconds()
        longest = max(longest, tail)
        if tail > tolerance:
            unavailable += tail - interval
    return unavailable, longest


def _processes(
    snaps: Sequence[Snap],
) -> tuple[int, float, list[datetime]]:
    """``(processes, longest run in seconds, the start times of the processes)``."""
    runs: dict[datetime, tuple[datetime, datetime]] = {}
    for snap in snaps:
        key = snap.started_at or snap.at
        first, last = runs.get(key, (snap.at, snap.at))
        runs[key] = (min(first, snap.at), max(last, snap.at))
    longest = max(((last - first).total_seconds() for first, last in runs.values()), default=0.0)
    return len(runs), longest, sorted(runs)


def _restarts(starts: Sequence[datetime], history: Sequence[Mapping[str, Any]]) -> list[Restart]:
    """Each process after the first is a restart; the supervisor's record says why."""
    recorded = [
        (datetime.fromisoformat(str(item["at"])), str(item.get("reason", "")))
        for item in history
        if item.get("at")
    ]
    found: list[Restart] = []
    for begin in list(starts)[1:]:
        candidates = [
            (begin - at, text)
            for at, text in recorded
            if timedelta(0) <= begin - at <= RESTART_MATCH
        ]
        reason = (
            min(candidates)[1]
            if candidates
            else "unknown (no supervisor record: stopped by hand, or the machine restarted)"
        )
        found.append(Restart(begin, reason))
    return found


def _outages(snaps: Sequence[Snap], alerts: Sequence[AlertView], interval: float) -> list[Outage]:
    tolerance = timedelta(seconds=GAP_FACTOR * interval)
    runs: list[list[Snap]] = []
    current: list[Snap] = []
    previous: Snap | None = None
    for snap in snaps:
        if previous is not None and snap.at - previous.at > tolerance and current:
            runs.append(current)  # the application was down in between: the outage is cut
            current = []
        if snap.channel_ok is False:
            current.append(snap)
        elif current:
            runs.append(current)
            current = []
        previous = snap
    if current:
        runs.append(current)

    announced = sorted(
        (
            a
            for a in alerts
            if a.kind == "alert" and not a.suppressed and a.category in CHANNEL_CATEGORIES
        ),
        key=lambda a: a.created_at,
    )
    found: list[Outage] = []
    for run in runs:
        first, last = run[0], run[-1]
        start = first.channel_last_ok_at if first.channel_last_ok_at else first.at
        start = min(start, first.at)
        after = next((s for s in snaps if s.at > last.at and s.channel_ok is not False), None)
        end = after.at if after is not None and after.at - last.at <= tolerance else last.at
        ongoing = after is None
        alert = next(
            (
                a
                for a in announced
                if start - timedelta(minutes=1) <= a.created_at <= end + ALERT_LATENCY_LIMIT
            ),
            None,
        )
        delivered = None
        if alert is not None:
            delivered = alert.toast_at or alert.notified_at or alert.created_at
        latency = (delivered - start).total_seconds() if delivered is not None else None
        found.append(
            Outage(
                start=start,
                end=max(end, start),
                ongoing=ongoing,
                alert_category=alert.category if alert else None,
                alert_at=delivered,
                latency_s=None if latency is None else max(latency, 0.0),
            )
        )
    return found


def build_report(
    snaps: Sequence[Snap],
    alerts: Sequence[AlertView],
    restart_history: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    days: float,
    interval_s: float,
) -> StabilityReport:
    """The report of the window ``[now - days, now]`` from already loaded rows (pure)."""
    start = now - timedelta(days=days)
    inside = sorted((s for s in snaps if start <= s.at <= now), key=lambda s: s.at)
    unavailable, longest_gap = _availability(inside, now, interval_s)
    processes, longest_run, starts = _processes(inside)
    launches: dict[str, int] = {}
    for snap in inside:
        launches[snap.launch] = launches.get(snap.launch, 0) + 1
    in_window = [a for a in alerts if a.created_at >= start]
    notes: list[str] = []
    if not inside:
        notes.append("no health snapshots in the window: the application did not run")
    return StabilityReport(
        window_start=start,
        window_end=now,
        days=days,
        interval_s=interval_s,
        snapshots=len(inside),
        first_snapshot_at=inside[0].at if inside else None,
        last_snapshot_at=inside[-1].at if inside else None,
        unavailable_s=unavailable,
        longest_gap_s=longest_gap,
        processes=processes,
        longest_run_s=longest_run,
        restarts=tuple(_restarts(starts, restart_history)),
        launches=launches,
        outages=tuple(_outages(inside, in_window, interval_s)),
        system_resumes=sum(1 for a in in_window if a.category == "system_resumed"),
        notes=tuple(notes),
    )


# ------------------------------------------------------------------ loading and storing


def load_snapshots(services: Services, since: datetime) -> list[Snap]:
    with services.db.session() as session:
        rows = session.scalars(
            select(HealthSnapshot).where(HealthSnapshot.at >= since).order_by(HealthSnapshot.at)
        )
        return [
            Snap(r.at, r.started_at, r.launch, r.channel_ok, r.channel_last_ok_at) for r in rows
        ]


def run_stability(services: Services, days: float) -> tuple[StabilityReport, RunView]:
    """Compute the report for the last ``days`` days and keep it as ``eval_runs(stability)``."""
    now = services.clock.now_utc()
    since = now - timedelta(days=days)
    snaps = load_snapshots(services, since)
    alerts = services.alerts.recent(limit=5000, since=since - timedelta(hours=1))
    history = RestartLog(services.db, services.clock).entries()
    interval = services.settings.ops.health.interval_s
    report = build_report(snaps, alerts, history, now=now, days=days, interval_s=interval)
    run = EvalStore(services.db, services.clock).create_run(
        "stability",
        status="done",
        params={"days": days, "interval_s": interval},
        summary=report.to_json(),
    )
    return report, run


# ------------------------------------------------------------------------ the screen


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} 小时" if seconds >= 3600 else f"{seconds / 60:.1f} 分钟"


def render_lines(report: StabilityReport) -> list[str]:
    """The report for the terminal, one fact per line."""
    lines = [
        f"稳定性报告：{report.window_start:%Y-%m-%d %H:%M} 到 {report.window_end:%Y-%m-%d %H:%M}"
        f" UTC（{report.days:g} 天）",
        f"健康快照 {report.snapshots} 个"
        + (
            f"，首个 {report.first_snapshot_at:%Y-%m-%d %H:%M}，"
            f"最后 {report.last_snapshot_at:%Y-%m-%d %H:%M}"
            if report.first_snapshot_at and report.last_snapshot_at
            else ""
        )
        + ("" if report.covers_window else "（窗口开头没有快照：观察期还没满）"),
        f"累计不可用 {_hours(report.unavailable_s)}"
        f"（上限 {_hours(UNAVAILABLE_LIMIT.total_seconds())}），"
        f"最长一次间隔 {_hours(report.longest_gap_s)}",
        f"进程 {report.processes} 个，最长连续运行 {_hours(report.longest_run_s)}，"
        f"重启 {len(report.restarts)} 次，电脑睡眠恢复 {report.system_resumes} 次",
    ]
    lines += [f"  重启于 {r.at:%Y-%m-%d %H:%M}：{r.reason}" for r in report.restarts]
    lines.append(
        "启动方式（快照数）：" + (", ".join(f"{k} {v}" for k, v in report.launches.items()) or "无")
    )
    if not report.outages:
        lines.append("通道中断：没有")
    for outage in report.outages:
        latency = "没有告警" if outage.latency_s is None else f"告警延迟 {_hours(outage.latency_s)}"
        lines.append(
            f"通道中断：{outage.start:%m-%d %H:%M} 起 {_hours(outage.duration_s)}"
            f"{'（还在继续）' if outage.ongoing else ''}，{latency}"
            + (f"（{outage.alert_category}）" if outage.alert_category else "")
        )
    drill = report.drill
    lines.append(
        f"断网演练：有（{_hours(drill.duration_s)}）"
        if drill
        else "断网演练：还没有（twin ops drill network）"
    )
    lines.extend(report.notes)
    return lines
