"""The judge of the M3 gate: "有作息、会主动" (SPEC section 26, R-EVAL-010, R-EVAL-005).

The rule is fixed (and pinned by tests).  Round 18 (user decision, D-850) made it this:

* the audit (:mod:`twin.eval.proactive_audit`) covers the **last ten completed local days**;
* among them at least **seven days of observation** - days she could write on (watched, the
  switch on, not paused, the range above ``0``) that are not excused - each **strictly** inside
  the range it was planned with; they need not be in a row;
* an **excused day** (her count below its minimum while the window, the count, the login or the
  budget refused something that day) is neither a violation nor a day of observation; **more than
  three** of them make the verdict "not enough yet", and the reasons of the refusals are listed;
* a day the user switched off or paused, a day planned for ``0-0`` and a day the scheduler did not
  run are not counted (and a message sent while off is still a violation, R17-14, D-663);
* in all ten days: no message in the core of her deep sleep, none above a day's maximum, no message
  closer than ``min_spacing_min`` to the one before and no chase beyond ``max_chase``, at most
  ``edge_of_sleep_weekly_max`` messages at the edge of her sleep in any seven days;
* and, in those days and after them up to now, **at least one ``/评分``** whose average is **4 or
  more**.

"Not enough yet" (fewer than seven days of observation, more than three excused days, no rating)
comes first and says what is missing; then a violation or an average below 4 is a failure.

Judging stores the audit it was made from as a run of kind ``proactive_audit`` and names it in the
evidence, so the verdict can be checked later (``twin eval gate M3 --check`` only reads it).
"""

from __future__ import annotations

from twin.eval.gates import Check, GateContext, GateVerdict, Verdict, gate_judge
from twin.eval.proactive_audit import Audit, DayAudit, audit_recent, describe_reasons
from twin.schedule.proactive.store import ProactiveLogStore, RatingStore
from twin.schedule.service import time_service_for

M3_DAYS = 10  # the window: the last ten completed local days (D-850)
M3_OBSERVATION_DAYS = 7  # days of observation the window must hold
M3_EXEMPT_MAX = 3  # excused days the window may hold
M3_MIN_RATING = 4.0


def _span(day: DayAudit) -> str:
    """The range a day was held to, as ``low-high`` (``0-0`` with the switch off)."""
    bounds = day.effective_range
    return f"{bounds[0]}-{bounds[1]}" if bounds is not None else "未知"


def _not_counted(audit: Audit) -> str:
    """The days that are no days of observation, by why (empty when there are none)."""
    parts = [
        f"免责日 {audit.exempt_days} 个" if audit.exempt_days else "",
        f"主动消息关闭、暂停或范围 0-0 的 {audit.inactive_days} 天" if audit.inactive_days else "",
        f"程序没有运行的 {audit.unobserved_days} 天" if audit.unobserved_days else "",
    ]
    found = "、".join(part for part in parts if part)
    return f"（{found}不计入观察）" if found else ""


def judge_days(audit: Audit) -> tuple[Verdict, str]:
    """The day rule of M3 alone (no rating): the audit command and the gate share it."""
    observed, exempt = audit.observation_days, audit.exempt_days
    span = len(audit.days)
    if exempt > M3_EXEMPT_MAX:
        return "insufficient", (
            f"样本不足：最近 {span} 天里免责日 {exempt} 个，超过 {M3_EXEMPT_MAX} 个"
            f"（拒绝原因：{describe_reasons(audit.exempt_reasons)}）；观察日只有 {observed} 个"
        )
    if observed < M3_OBSERVATION_DAYS:
        return "insufficient", (
            f"观察期未满：最近 {span} 天里有 {observed} 个观察日，"
            f"还差 {M3_OBSERVATION_DAYS - observed} 个{_not_counted(audit)}"
        )
    if not audit.compliant:
        problems = [
            f"{day.day:%m-%d} {'；'.join(day.problems)}" for day in audit.days if not day.compliant
        ]
        return "failed", "审计不合规：" + "；".join(problems or ["边缘消息超过每周上限"])
    return "passed", (f"最近 {span} 天里 {observed} 个观察日审计合规（免责日 {exempt} 个）")


def judge_audit(audit: Audit) -> tuple[Verdict, list[Check], str]:
    """The verdict of M3 from an audit (pure: the tests feed it hand-made audits)."""
    mean = audit.rating_mean
    rated = len(audit.ratings)
    span = len(audit.days)
    observed, exempt = audit.observation_days, audit.exempt_days
    missing = max(0, M3_OBSERVATION_DAYS - observed)
    checks = [
        Check(
            f"最近 {span} 个当地日里有 {M3_OBSERVATION_DAYS} 个观察日",
            observed >= M3_OBSERVATION_DAYS,
            f"观察日 {observed}/{span} 天"
            + (f"，还差 {missing} 个" if missing else "")
            + _not_counted(audit),
            float(observed),
            float(M3_OBSERVATION_DAYS),
            span,
        ),
        Check(
            f"免责日 ≤ {M3_EXEMPT_MAX} 个",
            exempt <= M3_EXEMPT_MAX,
            f"免责日 {exempt} 个"
            + (f"；拒绝原因：{describe_reasons(audit.exempt_reasons)}" if exempt else ""),
            float(exempt),
            float(M3_EXEMPT_MAX),
            span,
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
                f"{day.day:%m-%d} 发 {day.sent} 条（{_span(day)}）"
                for day in audit.days
                if not day.count_ok
            )
            or "观察日都在范围内，其余日子都没有超过上限",
            None,
            None,
            span,
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
    verdict, summary = judge_days(audit)
    if verdict != "passed":
        return verdict, checks, summary
    if rated == 0:
        return "insufficient", checks, "这段时间没有 /评分：先给她打个分"
    if mean is None or mean < M3_MIN_RATING:
        return "failed", checks, f"/评分平均 {mean:.2f}，低于 {M3_MIN_RATING:g}"
    return "passed", checks, f"M3 通过：{summary}，/评分平均 {mean:.2f}"


@gate_judge("M3")
def judge_m3(ctx: GateContext) -> GateVerdict:
    """M3: seven days of observation in the last ten, and a rating of 4 or more (see above)."""
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
            "observation_days": M3_OBSERVATION_DAYS,
            "exempt_max": M3_EXEMPT_MAX,
            "first_day": str(audit.first_day),
            "last_day": str(audit.last_day),
        },
        summary=audit.to_json(),
    )
    values = {
        "days": M3_DAYS,
        "observation_days": audit.observation_days,
        "required_observation_days": M3_OBSERVATION_DAYS,
        "exempt_days": audit.exempt_days,
        "exempt_max": M3_EXEMPT_MAX,
        "exempt_reasons": dict(audit.exempt_reasons),
        "sent": audit.sent,
        "deep_sleep": audit.deep_sleep,
        "rating_count": len(audit.ratings),
        "rating_mean": audit.rating_mean,
        "audit_run": run.id,
    }
    return GateVerdict("M3", verdict, tuple(checks), summary, (run.id,), values)
