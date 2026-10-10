"""The judge of milestone M4 "continuous growth" (R-EVAL-010, SPEC section 26, round 12).

``twin eval gate M4`` passes only if **all** of this is on record (nothing is estimated, nothing
is waived; the rules are fixed here and pinned by tests):

1. the latest stability report (``twin eval stability --days 7``, R-EVAL-006) covers **seven
   days in a row** - health snapshots from the first day to now - and is not older than two days;
2. in those seven days the application ran **unattended from the scheduled task**: every snapshot
   was taken by a process that ``twin supervise --from-task`` started, none by hand;
3. the application was **unavailable for ten minutes at most** in total, and it is running now
   (a restart in between is allowed: the supervisor brings it back by itself);
4. there was **a network drill**: a channel outage of at least ten minutes (``twin ops drill
   network``), and **every** channel outage - the drill included - was announced within **ten
   minutes** of its start; no outage went without an alert;
5. a **new blind test** of the default backend, started inside the observation window, on contexts
   no earlier blind test used, with at least 50 valid judgements and a guess rate (point estimate)
   of at most **60 %**;
6. **learning** (a ``learning_rules`` job of round 11) and an **incremental import** (an import
   run) each finished successfully inside the window.

A criterion that is measured and missed makes the verdict ``failed``; one for which the evidence
does not exist yet (no drill, no blind test, no report) makes it ``insufficient``.  The exit code
is 1 for both.  Do not change a threshold to make it pass: fix the cause and observe again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from twin.config.runtime import BACKEND_ACTIVE
from twin.eval.gates import (
    M4_MAX_GUESS_RATE,
    Check,
    GateContext,
    GateVerdict,
    Verdict,
    blind_checks,
    gate_judge,
)
from twin.ops.stability import ALERT_LATENCY_LIMIT, DRILL_MIN, UNAVAILABLE_LIMIT
from twin.storage.chat_models import ImportRun
from twin.storage.models import Job

OBSERVATION_DAYS = 7
REPORT_MAX_AGE = timedelta(days=2)
LEARNING_JOB = "learning_rules"


@dataclass(frozen=True)
class _Criterion:
    """A check and what a miss means: a measured miss (``failed``) or missing evidence."""

    check: Check
    measured: bool


def _stamp(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} 分钟"


def _stability_criteria(
    summary: dict[str, Any], run_age: timedelta, days_asked: float
) -> list[_Criterion]:
    unavailable = float(summary.get("unavailable_s", 0.0))
    launches = {str(k): int(v) for k, v in dict(summary.get("launches", {})).items()}
    snapshots = int(summary.get("snapshots", 0))
    task_snaps = launches.get("task", 0)
    window_end = _stamp(summary.get("window_end"))
    last = _stamp(summary.get("last_snapshot_at"))
    interval = float(summary.get("interval_s", 60.0))
    running_now = (
        last is not None
        and window_end is not None
        and (window_end - last).total_seconds() <= 2.5 * interval
    )
    drill = summary.get("drill")
    max_latency = summary.get("max_alert_latency_s")
    missed = int(summary.get("missed_alerts", 0))
    limit = ALERT_LATENCY_LIMIT.total_seconds()
    covered = bool(summary.get("covers_window")) and snapshots > 0
    complete = days_asked >= OBSERVATION_DAYS and covered
    return [
        _Criterion(
            Check(
                f"稳定性报告覆盖连续 {OBSERVATION_DAYS} 天，且不超过两天前",
                complete and run_age <= REPORT_MAX_AGE,
                f"报告窗口 {days_asked:g} 天，快照 {snapshots} 个，"
                + ("从窗口开头起都有" if covered else "窗口开头没有快照")
                + f"；报告生成于 {run_age.total_seconds() / 3600:.1f} 小时前",
                float(days_asked),
                float(OBSERVATION_DAYS),
                snapshots,
            ),
            measured=False,
        ),
        _Criterion(
            Check(
                "全部由计划任务启动（无人值守）",
                snapshots > 0 and task_snaps == snapshots,
                f"{task_snaps}/{snapshots} 个快照来自计划任务启动的进程；各启动方式：{launches}",
                float(task_snaps),
                float(snapshots),
                snapshots,
            ),
            measured=snapshots > 0,
        ),
        _Criterion(
            Check(
                f"应用累计不可用时间 ≤ {_minutes(UNAVAILABLE_LIMIT.total_seconds())}",
                snapshots > 0 and unavailable <= UNAVAILABLE_LIMIT.total_seconds(),
                f"累计不可用 {_minutes(unavailable)}，重启 {len(summary.get('restarts', []))} 次",
                unavailable,
                UNAVAILABLE_LIMIT.total_seconds(),
                snapshots,
            ),
            measured=snapshots > 0,
        ),
        _Criterion(
            Check(
                "报告时应用仍在运行（重启后已自动恢复）",
                running_now,
                "最后一个快照在报告时刻附近" if running_now else "最后一个快照太旧，应用没有在运行",
            ),
            measured=snapshots > 0,
        ),
        _Criterion(
            Check(
                f"做过一次断网演练（通道中断 ≥ {_minutes(DRILL_MIN.total_seconds())}）",
                drill is not None,
                "有一次中断 " + _minutes(float(drill["duration_s"]))
                if drill
                else "观察期内没有这样长的通道中断：运行 twin ops drill network 并照做",
            ),
            measured=False,
        ),
        _Criterion(
            Check(
                f"每次通道异常都在 {limit / 60:.0f} 分钟内发出告警，没有漏报",
                missed == 0 and max_latency is not None and float(max_latency) <= limit,
                (
                    f"最长延迟 {_minutes(float(max_latency))}，漏报 {missed} 次"
                    if max_latency is not None
                    else f"没有任何通道异常可核对，漏报 {missed} 次"
                ),
                None if max_latency is None else float(max_latency),
                limit,
                len(summary.get("outages", [])),
            ),
            measured=max_latency is not None or missed > 0,
        ),
    ]


def _blind_criteria(
    ctx: GateContext, window_start: datetime | None
) -> tuple[list[_Criterion], tuple[str, ...], dict[str, Any]]:
    backend = str(ctx.services.runtime.get(BACKEND_ACTIVE))
    verdict, checks, runs, values, _detail = blind_checks(ctx, backend, M4_MAX_GUESS_RATE)
    criteria = [_Criterion(c, measured=verdict != "insufficient") for c in checks]
    if not runs:
        return criteria, runs, values
    run = ctx.store.get_run(runs[0])
    fresh = window_start is not None and run.created_at >= window_start
    criteria.append(
        _Criterion(
            Check(
                "这次盲测是在观察期内做的",
                fresh,
                f"盲测开始于 {run.created_at:%Y-%m-%d %H:%M} UTC"
                + ("" if fresh else "，早于观察期的开始：观察期末要做新的盲测"),
            ),
            measured=True,
        )
    )
    earlier: set[str] = set()
    for other in ctx.store.list_runs("blind", limit=200):
        if other.id != run.id and other.created_at < run.created_at:
            earlier.update(i.sample_key for i in ctx.store.items(other.id, with_payload=False))
    reused = {i.sample_key for i in ctx.store.items(run.id, with_payload=False)} & earlier
    criteria.append(
        _Criterion(
            Check(
                "盲测用的是新的上下文（没有用过早先盲测的样本）",
                not reused,
                f"与早先盲测重复的样本 {len(reused)} 个",
                float(len(reused)),
                0.0,
            ),
            measured=True,
        )
    )
    return criteria, runs, values


def _activity_criteria(ctx: GateContext, since: datetime) -> list[_Criterion]:
    with ctx.services.db.session() as session:
        learning = int(
            session.scalar(
                select(func.count())
                .select_from(Job)
                .where(
                    Job.type == LEARNING_JOB,
                    Job.status == "done",
                    Job.finished_at >= since,
                )
            )
            or 0
        )
        imports = int(
            session.scalar(
                select(func.count())
                .select_from(ImportRun)
                .where(ImportRun.status == "done", ImportRun.finished_at >= since)
            )
            or 0
        )
    return [
        _Criterion(
            Check(
                "观察期内学习（每周整理纠正规则）至少成功一次",
                learning >= 1,
                f"成功 {learning} 次"
                + ("" if learning else "：给机器人一次 /不像 纠正，再整理规则"),
                float(learning),
                1.0,
            ),
            measured=False,
        ),
        _Criterion(
            Check(
                "观察期内增量导入至少成功一次",
                imports >= 1,
                f"成功 {imports} 次" + ("" if imports else "：用 /导入 或 twin import 导入新记录"),
                float(imports),
                1.0,
            ),
            measured=False,
        ),
    ]


@gate_judge("M4")
def judge_m4(ctx: GateContext) -> GateVerdict:
    """M4: seven unattended days, a drill alerted within ten minutes, a new blind test ≤ 60 %."""
    now = ctx.services.clock.now_utc()
    run = ctx.store.latest_run("stability", status="done")
    if run is None:
        check = Check("稳定性报告", False, "还没有稳定性报告：先运行 twin eval stability --days 7")
        return GateVerdict("M4", "insufficient", (check,), check.detail)
    summary = run.summary
    days_asked = float(run.params.get("days", summary.get("days", 0)))
    age = now - (run.finished_at or run.created_at)
    window_start = _stamp(summary.get("window_start"))
    criteria = _stability_criteria(summary, age, days_asked)
    blind, blind_runs, blind_values = _blind_criteria(ctx, window_start)
    criteria += blind
    criteria += _activity_criteria(ctx, window_start or now - timedelta(days=OBSERVATION_DAYS))

    checks = tuple(c.check for c in criteria)
    if all(c.passed for c in checks):
        verdict: Verdict = "passed"
    elif any(not c.check.passed and c.measured for c in criteria):
        verdict = "failed"
    else:
        verdict = "insufficient"
    failing = [c.check.name for c in criteria if not c.check.passed]
    text = "M4 通过" if verdict == "passed" else "未满足：" + "；".join(failing)
    values: dict[str, Any] = {
        "stability_run": run.id,
        "days": days_asked,
        "unavailable_s": summary.get("unavailable_s"),
        "max_alert_latency_s": summary.get("max_alert_latency_s"),
        "launches": summary.get("launches"),
        **blind_values,
    }
    return GateVerdict("M4", verdict, checks, text, (run.id, *blind_runs), values)
