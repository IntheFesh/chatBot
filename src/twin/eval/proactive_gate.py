"""The judge of the M3 gate: "有作息、会主动" (SPEC section 26, R-EVAL-010, R-EVAL-005).

The rule is fixed (and pinned by tests): in the **last seven completed local days** the
scheduler watched every day (it ran: the day has its mark), and the audit
(:mod:`twin.eval.proactive_audit`) finds

* no message in the core of her deep sleep,
* every day's count inside the range it was planned with,
* no message closer than ``min_spacing_min`` to the one before and no chase beyond ``max_chase``,
* at most ``edge_of_sleep_weekly_max`` messages at the edge of her sleep in any seven days,

and, in those days and after them up to now, **at least one ``/评分``** whose average is **4 or
more**.  With fewer than seven watched days in a row the verdict is "not enough yet" (the
observation period is not over; the command says how many days are missing); a rating period
without a rating is "not enough yet" too; a violation or an average below 4 is a failure.

Judging stores the audit it was made from as a run of kind ``proactive_audit`` and names it in the
evidence, so the verdict can be checked later (``twin eval gate M3 --check`` only reads it).
"""

from __future__ import annotations

from twin.eval.gates import Check, GateContext, GateVerdict, Verdict, gate_judge
from twin.eval.proactive_audit import Audit, audit_recent
from twin.schedule.proactive.store import ProactiveLogStore, RatingStore
from twin.schedule.service import time_service_for

M3_DAYS = 7
M3_MIN_RATING = 4.0


def judge_audit(audit: Audit, days: int = M3_DAYS) -> tuple[Verdict, list[Check], str]:
    """The verdict of M3 from an audit (pure: the tests feed it hand-made audits)."""
    problems = [
        f"{day.day:%m-%d} {'；'.join(day.problems)}" for day in audit.days if not day.compliant
    ]
    mean = audit.rating_mean
    rated = len(audit.ratings)
    checks = [
        Check(
            f"连续观察 {days} 个当地日",
            audit.complete,
            f"已连续观察 {audit.streak}/{len(audit.days)} 天"
            + ("" if audit.complete else f"，还差 {audit.missing_days} 天"),
            float(audit.streak),
            float(days),
            len(audit.days),
        ),
        Check(
            "深睡核心时段主动消息 0 次",
            audit.deep_sleep == 0,
            f"深睡时段发出 {audit.deep_sleep} 条",
            float(audit.deep_sleep),
            0.0,
            audit.sent,
        ),
        Check(
            "每天次数在设定范围内",
            audit.count_ok,
            "；".join(
                f"{day.day:%m-%d} 发 {day.sent} 条（{day.low}-{day.high}）"
                for day in audit.days
                if not day.observed or not day.count_ok
            )
            or f"{len(audit.days)} 天都在范围内",
            None,
            None,
            len(audit.days),
        ),
        Check(
            "间隔与追发零违规",
            audit.spacing_violations == 0 and audit.chase_violations == 0,
            f"间隔不足 {audit.spacing_violations} 次，追发超限 {audit.chase_violations} 次",
            float(audit.spacing_violations + audit.chase_violations),
            0.0,
            audit.sent,
        ),
        Check(
            f"边缘消息每周 ≤ {audit.edge_weekly_max}",
            audit.edge_ok,
            f"任意 7 天里最多 {audit.edge_max_week} 条",
            float(audit.edge_max_week),
            float(audit.edge_weekly_max),
        ),
        Check(
            "这段时间有 /评分",
            rated >= 1,
            f"{rated} 次评分" if rated else "还没有 /评分：在微信里给她打个分",
            float(rated),
            1.0,
            rated,
        ),
        Check(
            f"/评分平均 ≥ {M3_MIN_RATING:g}",
            mean is not None and mean >= M3_MIN_RATING,
            f"平均 {mean:.2f}" if mean is not None else "没有评分",
            mean,
            M3_MIN_RATING,
            rated,
        ),
    ]
    if not audit.complete:
        return (
            "insufficient",
            checks,
            f"观察期未满：已连续观察 {audit.streak} 天，还差 {audit.missing_days} 天",
        )
    if not audit.compliant:
        return "failed", checks, "审计不合规：" + "；".join(problems or ["边缘消息超过每周上限"])
    if rated == 0:
        return "insufficient", checks, "这段时间没有 /评分：先给她打个分"
    if mean is None or mean < M3_MIN_RATING:
        return "failed", checks, f"/评分平均 {mean:.2f}，低于 {M3_MIN_RATING:g}"
    return "passed", checks, f"M3 通过：{len(audit.days)} 天审计合规，/评分平均 {mean:.2f}"


@gate_judge("M3")
def judge_m3(ctx: GateContext) -> GateVerdict:
    """M3: seven watched days that pass the audit, and a rating of 4 or more (see above)."""
    services = ctx.services
    audit = audit_recent(
        ProactiveLogStore(services.db, services.clock),
        RatingStore(services.db, services.clock),
        services.settings.proactive,
        time_service_for(services),
        services.clock,
        days=M3_DAYS,
    )
    verdict, checks, summary = judge_audit(audit)
    run = ctx.store.create_run(
        "proactive_audit",
        status="done",
        verdict=verdict,
        params={
            "days": M3_DAYS,
            "first_day": str(audit.first_day),
            "last_day": str(audit.last_day),
        },
        summary=audit.to_json(),
    )
    values = {
        "days": M3_DAYS,
        "streak": audit.streak,
        "sent": audit.sent,
        "deep_sleep": audit.deep_sleep,
        "rating_count": len(audit.ratings),
        "rating_mean": audit.rating_mean,
        "audit_run": run.id,
    }
    return GateVerdict("M3", verdict, tuple(checks), summary, (run.id,), values)
