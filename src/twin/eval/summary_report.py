"""``twin eval report``: every evaluation result in one Markdown file (R-EVAL-008).

The report is written to ``data/reports/eval-<local date>.md`` and has two parts.

**The milestones M0 to M5** (SPEC section 26): for each one its content and condition, the verdict
that ``twin eval gate`` last stored, when it was judged, the evidence (the command and the ids of
``eval_runs``: the gate's own and the runs it was decided from) and, criterion by criterion, the
threshold, the value, the number of samples and whether it was met.  Everything is read from the
stored gate runs by the code of ``twin eval gate --check`` - **nothing is judged again and no number
is made up**.  A milestone that was never judged says "未评估" and which command produces the
evidence.  M0 also points at ``docs/CHANNEL_REPORT.md`` and ``docs/LLM_REPORT.md`` and says honestly
whether they still are the "待实测" templates.

**The measures**: the blind tests by backend, the style metrics (the bot's real output and the
hold-out), the memory test, the consistency audit, the audit of the proactive messages, the
stability report and the cost (the daily account and the one-time batches apart) - each from the
latest stored run of its kind.

**No chat text.**  Only names, numbers, ids and the fixed wording of the gates go in; the sealed
payloads of the evaluation items and findings are never read here.  A test scans the written file
for the text of a synthetic conversation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from twin.channel.probe.report import MEASURED_MARKER as CHANNEL_MEASURED
from twin.channel.probe.report import PENDING_MARKER as CHANNEL_PENDING
from twin.eval.blind import MIN_VALID, blind_report
from twin.eval.gates import (
    M1_MAX_GUESS_RATE,
    M2_MIN_ACCURACY,
    M4_MAX_GUESS_RATE,
    GateOutcome,
    check_gate,
)
from twin.eval.memory_test import summarize
from twin.eval.report import interval_text, percent
from twin.eval.stats import two_proportion_test
from twin.eval.store import RESOLVE_LIMIT, EvalStore, RunView
from twin.eval.style_metrics import STYLE_METRICS
from twin.llm.probe_report import MEASURED_MARKER as LLM_MEASURED
from twin.llm.probe_report import PENDING_MARKER as LLM_PENDING
from twin.schedule.service import time_service_for
from twin.services import Services

NOT_EVALUATED = "未评估"
REPORT_PREFIX = "eval-"
BACKENDS = ("deepseek", "style", "hybrid")
VERDICT_WORDS = {
    "passed": "通过",
    "failed": "未通过",
    "insufficient": "未通过（样本不足）",
    "not_run": NOT_EVALUATED,
    "not_reached": "判定器还没有接入",
}
TOP_ROWS = 8


@dataclass(frozen=True)
class MilestoneSpec:
    """One row of the table of section 26 of the SPEC."""

    code: str
    title: str
    content: str
    condition: str
    how: str  # the command(s) that produce the evidence


MILESTONES: tuple[MilestoneSpec, ...] = (
    MilestoneSpec(
        "M0",
        "技术验证",
        "ClawBot 收发、主动窗口与条数、图片/GIF、引用、正在输入；DeepSeek 思考开关与缓存命中",
        "主动消息在窗口内稳定送达（R-CH-009/010 达标）；DeepSeek 探针按 R-LLM-013 的判定口径通过；"
        "否则停下评估企业微信通道",
        "`twin llm probe`、`twin channel probe start`",
    ),
    MilestoneSpec(
        "M1",
        "能聊、像她",
        "F1 导入、F2 画像、F3 生成与节奏、F4 表情代码",
        "盲测猜对率 ≤ 70%",
        "`twin eval blind`",
    ),
    MilestoneSpec(
        "M2",
        "记得住",
        "F4 表情包库、F5 记忆、F10 指令",
        "记忆测试 ≥ 80%",
        "`twin eval memory`",
    ),
    MilestoneSpec(
        "M3",
        "有作息、会主动",
        "F6 作息与真实模式、F7 主动、F9 思考开关",
        "连续 7 天真实运行：深睡时段主动 0 次、每天次数在范围内、间隔与追发零违规；"
        "该周 /评分 ≥ 4/5",
        "`twin eval proactive`（观察满 7 天后）",
    ),
    MilestoneSpec(
        "M4",
        "持续成长",
        "F8 学习、增量导入、F11 运维",
        "无人值守 7 天；盲测 ≤ 60%",
        "`twin eval stability --days 7`、`twin eval blind`",
    ),
    MilestoneSpec(
        "M5",
        "风格模型",
        "F12 训练、评估、量化部署",
        "R-SRV-005；不通过则保留 DeepSeek 后端（不阻塞第 15、16 轮）",
        "`twin model evaluate <模型>`",
    ),
)
# the kinds of run each gate is decided from: a newer one means the verdict may be out of date
GATE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "M0": (),
    "M1": ("blind",),
    "M2": ("memory",),
    "M3": ("proactive_audit",),
    "M4": ("stability", "blind"),
    "M5": ("blind",),
}


# ------------------------------------------------------------------------- formatting


def cell(value: object) -> str:
    """A table cell: no pipe, no line break, never empty."""
    text = " ".join(str(value).replace("|", "／").split())
    return text or "—"


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """A Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(item) for item in row) + " |" for row in rows]
    return lines


def stamp(moment: datetime | None, zone: ZoneInfo) -> str:
    """A moment on the bot's clock, with the name of the zone (``—`` when there is none)."""
    return "—" if moment is None else f"{moment.astimezone(zone):%Y-%m-%d %H:%M %Z}"


def number(value: object, digits: int = 2) -> str:
    return "—" if not isinstance(value, int | float) else f"{value:.{digits}f}"


def usd(value: object) -> str:
    return "—" if not isinstance(value, int | float) else f"${value:.2f}"


def hours(seconds: object) -> str:
    if not isinstance(seconds, int | float):
        return "—"
    return f"{seconds / 3600:.1f} 小时" if seconds >= 3600 else f"{seconds / 60:.1f} 分钟"


def latest_run(store: EvalStore, kind: str, *, backend: str | None = None) -> RunView | None:
    """The newest run of a kind that was not cancelled (optionally one that has the backend)."""
    for run in store.list_runs(kind, limit=RESOLVE_LIMIT):
        if run.status == "cancelled":
            continue
        if backend is not None and backend not in run.backends:
            continue
        return run
    return None


@dataclass
class Section:
    """A part of the report: its lines, the runs it was made from, what is not evaluated."""

    lines: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def absent(self, what: str, command: str) -> None:
        self.lines.append(f"{NOT_EVALUATED}：{what}。运行 `{command}`。")
        self.missing.append(f"{what}：`{command}`")


@dataclass(frozen=True)
class Context:
    services: Services
    store: EvalStore
    zone: ZoneInfo

    def stamp(self, moment: datetime | None) -> str:
        return stamp(moment, self.zone)


# ------------------------------------------------------------------------ milestones


def doc_state(path: Path, pending: str, measured: str) -> str:
    """Whether a probe report is still the template, has been written from a probe, or is gone."""
    if not path.is_file():
        return "文件不存在"
    text = path.read_text(encoding="utf-8")
    if pending in text:
        return "待实测（仍是模板，没有实测数字）"
    if measured in text:
        return "已由探针写成实测报告"
    return "无法识别（既不是模板也不是探针写的）"


def _m0_documents(ctx: Context) -> list[str]:
    docs = ctx.services.paths.root / "docs"
    return [
        "- 通道实测报告 `docs/CHANNEL_REPORT.md`："
        + doc_state(docs / "CHANNEL_REPORT.md", CHANNEL_PENDING, CHANNEL_MEASURED),
        "- DeepSeek 实测报告 `docs/LLM_REPORT.md`："
        + doc_state(docs / "LLM_REPORT.md", LLM_PENDING, LLM_MEASURED),
    ]


def _newer_evidence(ctx: Context, code: str, gate: RunView) -> list[str]:
    notes: list[str] = []
    for kind in GATE_EVIDENCE[code]:
        newest = latest_run(ctx.store, kind)
        if newest is not None and newest.created_at > gate.created_at:
            when = ctx.stamp(newest.created_at)
            notes.append(
                f"- 注意：判定之后又有新的 {kind} 评估记录 `{newest.id}`（{when}），"
                f"结论可能已过期：重新运行 `twin eval gate {code}`。"
            )
    return notes


def milestone_detail(ctx: Context, spec: MilestoneSpec, outcome: GateOutcome) -> list[str]:
    """The heading, the verdict with its evidence, and the criteria of one milestone."""
    lines = [f"### {spec.code} {spec.title}", "", f"- 过关条件：{spec.condition}"]
    run, verdict = outcome.run, outcome.verdict
    if run is None or verdict is None:
        lines.append(f"- 结论：**{VERDICT_WORDS[outcome.status]}**。{outcome.message}")
        lines.append(f"- 产生证据的命令：{spec.how}，然后运行 `twin eval gate {spec.code}`")
    else:
        evidence = "、".join(f"`{item}`" for item in verdict.runs) or "无（读取探针结果）"
        lines += [
            f"- 结论：**{VERDICT_WORDS[outcome.status]}**；判定时间：{ctx.stamp(run.created_at)}",
            f"- 证据：命令 `twin eval gate {spec.code}`，判定记录 `{run.id}`；"
            f"依据的评估记录：{evidence}",
            "",
        ]
        rows = [
            [
                check.name,
                check.detail,
                "—" if check.samples is None else check.samples,
                "达标" if check.passed else "未达标",
            ]
            for check in verdict.checks
        ]
        lines += table(["条件", "当前值 / 数据", "样本数", "结果"], rows)
        interval = verdict.values.get("interval")
        if isinstance(interval, list) and len(interval) == 2:
            lines += ["", f"- 95% 置信区间（Wilson）：{interval[0]:.1%}–{interval[1]:.1%}"]
        lines += ["", *_newer_evidence(ctx, spec.code, run)]
    if spec.code == "M0":
        lines += ["", *_m0_documents(ctx)]
    if spec.code == "M5":
        lines += ["", "- 说明：M5 未通过时保留 DeepSeek 后端，不阻塞后续轮次（SPEC 第 26 节）。"]
    lines.append("")
    return lines


def milestones_section(ctx: Context) -> tuple[Section, dict[str, dict[str, Any]]]:
    """The overview table and the detail of every milestone."""
    section = Section()
    outcomes = {spec.code: check_gate(ctx.services, spec.code) for spec in MILESTONES}
    rows: list[list[object]] = []
    stored: dict[str, dict[str, Any]] = {}
    for spec in MILESTONES:
        outcome = outcomes[spec.code]
        run = outcome.run
        if run is None:
            evidence = f"`twin eval gate {spec.code}`（尚无判定记录）"
            section.missing.append(f"{spec.code} 门槛：`twin eval gate {spec.code}`")
        else:
            section.runs.append(run.id)
            used = outcome.verdict.runs if outcome.verdict else ()
            evidence = f"`twin eval gate {spec.code}` · 记录 `{run.id}`" + (
                " · 依据 " + "、".join(f"`{item}`" for item in used) if used else ""
            )
        rows.append(
            [
                spec.code,
                spec.title,
                spec.condition,
                VERDICT_WORDS[outcome.status],
                ctx.stamp(run.created_at) if run else NOT_EVALUATED,
                evidence,
            ]
        )
        stored[spec.code] = {"status": outcome.status, "run": run.id if run else None}
    section.lines += table(["里程碑", "内容", "过关条件", "结论", "判定时间", "证据"], rows)
    section.lines += [""]
    for spec in MILESTONES:
        section.lines += milestone_detail(ctx, spec, outcomes[spec.code])
    return section, stored


# --------------------------------------------------------------------------- measures


def blind_section(ctx: Context) -> Section:
    section = Section(
        lines=[f"每个后端取包含它的最近一次盲测（有效判断至少 {MIN_VALID} 对才算数）。", ""]
    )
    rows: list[list[object]] = []
    for backend in BACKENDS:
        run = latest_run(ctx.store, "blind", backend=backend)
        if run is None:
            rows.append([backend, NOT_EVALUATED, "—", "—", "—", "—", "—"])
            section.missing.append(
                f"盲测（{backend} 后端）：`twin eval blind --backend {backend}`"
                if backend == "deepseek"
                else f"盲测（{backend} 后端）：`twin model evaluate <模型>`"
            )
            continue
        report = blind_report(ctx.store, run)
        entry = report.of(backend)
        if entry is None:  # cannot happen for a run that lists the backend; kept honest
            rows.append([backend, f"`{run.id}`", "—", "—", "—", "—", NOT_EVALUATED])
            continue
        section.runs.append(run.id)
        rate = entry.rate
        if entry.judged < MIN_VALID:
            verdict = f"样本不足（{entry.judged} < {MIN_VALID}）"
        elif backend == "deepseek":
            point = rate.point
            m1 = point is not None and point <= float(M1_MAX_GUESS_RATE)
            m4 = point is not None and point <= float(M4_MAX_GUESS_RATE)
            verdict = (
                f"M1 ≤ {float(M1_MAX_GUESS_RATE):.0%}：{'达标' if m1 else '未达标'}；"
                f"M4 ≤ {float(M4_MAX_GUESS_RATE):.0%}：{'达标' if m4 else '未达标'}"
            )
        else:
            baseline = report.of("deepseek")
            test = (
                two_proportion_test(entry.correct, entry.judged, baseline.correct, baseline.judged)
                if baseline is not None
                else None
            )
            verdict = (
                f"与 deepseek 同批对比：{backend} 更低的单侧 p = {number(test.p_less, 4)}"
                if test is not None
                else "没有与 deepseek 的同批对比"
            )
        rows.append(
            [
                backend,
                f"`{run.id}`",
                entry.judged,
                entry.correct,
                percent(rate.point),
                interval_text(rate),
                verdict,
            ]
        )
    section.lines += table(
        ["后端", "评估记录", "有效判断", "猜对", "猜对率", "95% 区间 (Wilson)", "对照门槛"], rows
    )
    return section


PROPORTION_METRICS = frozenset(spec.key for spec in STYLE_METRICS if spec.proportion)
STYLE_STATUS = {"pass": "通过", "fail": "不通过", "n/a": "不适用", "no_data": "无数据"}


def _style_value(key: str, value: object) -> str:
    """A style metric the way ``twin eval style`` shows it: a share in per cent, a size as is."""
    if not isinstance(value, int | float):
        return "—"
    return f"{value * 100:.2f}%" if key in PROPORTION_METRICS else f"{value:.2f}"


def _style_table(run: RunView) -> list[str]:
    rows: list[list[object]] = []
    for metric in run.summary.get("metrics", []):
        key = str(metric.get("key"))
        deviation = metric.get("deviation")
        rows.append(
            [
                metric.get("label"),
                _style_value(key, (metric.get("reference") or {}).get("value")),
                _style_value(key, (metric.get("measured") or {}).get("value")),
                "—" if deviation is None else f"{deviation * 100:+.1f}%",
                STYLE_STATUS.get(str(metric.get("status")), "—"),
            ]
        )
    return table(["指标", "她（画像）", "机器人", "相对偏差", "结论"], rows)


def style_section(ctx: Context) -> Section:
    """The latest ``style`` run of the bot's own output and of each backend on the hold-out."""
    section = Section()
    runs = [r for r in ctx.store.list_runs("style", limit=RESOLVE_LIMIT) if r.status == "done"]
    live = next((r for r in runs if r.params.get("source") == "live"), None)
    holdout = [
        (backend, run)
        for backend in BACKENDS
        if (
            run := next(
                (
                    r
                    for r in runs
                    if r.params.get("source") == "eval_items" and r.params.get("backend") == backend
                ),
                None,
            )
        )
        is not None
    ]
    parts: list[tuple[str, RunView]] = []
    if live is None:
        section.absent("风格指标（live）", "twin eval style --source live")
    else:
        parts.append(("live：机器人最近的真实输出对 live 画像", live))
    if not holdout:
        section.absent(
            "风格指标（留出集）",
            "twin eval style --source eval_items --run <盲测> --backend <后端>",
        )
    parts += [(f"留出集：{backend} 后端对 pre_holdout 画像", run) for backend, run in holdout]
    for title, run in parts:
        section.runs.append(run.id)
        verdict = (
            "每项偏差在 ±30% 内：通过" if run.summary.get("passed") else "有指标超出 ±30%：不通过"
        )
        section.lines += [
            "",
            f"**{title}**（记录 `{run.id}`，{ctx.stamp(run.created_at)}；"
            f"机器人消息 {run.summary.get('messages', 0)} 条）：{verdict}",
            "",
            *_style_table(run),
        ]
    return section


def memory_section(ctx: Context) -> Section:
    section = Section()
    run = latest_run(ctx.store, "memory")
    if run is None:
        section.absent("记忆测试", "twin eval memory")
        return section
    section.runs.append(run.id)
    score = summarize(ctx.store.items(run.id, with_payload=False))
    accuracy = score.accuracy
    section.lines += table(
        [
            "评估记录",
            "题数",
            "来源",
            "已复核",
            "正确 / 部分 / 错误",
            "得分",
            "正确率",
            "门槛",
            "结论",
        ],
        [
            [
                f"`{run.id}`",
                score.total,
                f"真实记录 {score.real_items} + 机器人对话 {score.bot_items}",
                f"{score.reviewed}/{score.total}",
                f"{score.correct} / {score.partial} / {score.wrong}",
                f"{score.points:g}",
                "—" if accuracy is None else f"{accuracy:.1%}",
                f"≥ {float(M2_MIN_ACCURACY):.0%}（partial 记 0.5）",
                VERDICT_WORDS[score.verdict],
            ]
        ],
    )
    return section


def consistency_section(ctx: Context) -> Section:
    section = Section()
    run = latest_run(ctx.store, "consistency")
    if run is None:
        section.absent("前后一致审计", "twin eval consistency --days 7")
        return section
    section.runs.append(run.id)
    summary = run.summary
    decisions = summary.get("decisions") or {}
    evidence = summary.get("evidence") or {}
    if run.status == "failed":
        state = "审计没有成功（见日志；重新运行 `twin eval consistency`）"
    elif run.verdict is None:
        state = (
            f"待确认：还有 {decisions.get('undecided', '?')} 条矛盾没有决定"
            "（`twin eval consistency --review`）"
        )
    else:
        state = VERDICT_WORDS[run.verdict]
    days = run.params.get("days", "—")
    section.lines += table(
        [
            "评估记录",
            "回看天数",
            "生活安排 / 她的话 / 事实",
            "确认的明显矛盾",
            "允许",
            "确认的不明显矛盾",
            "判为不是矛盾",
            "结论",
        ],
        [
            [
                f"`{run.id}`",
                days,
                " / ".join(str(evidence.get(key, "—")) for key in ("lifeline", "replies", "facts")),
                decisions.get("confirmed_obvious", "—"),
                f"≤ {summary.get('allowed_obvious', '—')}",
                decisions.get("confirmed_minor", "—"),
                decisions.get("rejected", "—"),
                state,
            ]
        ],
    )
    fixes = summary.get("fixes") or {}
    if fixes:
        section.lines += [
            "",
            f"记忆修正建议：已应用 {fixes.get('applied', 0)}，不改 {fixes.get('declined', 0)}，"
            f"待决定 {fixes.get('proposed', 0)}，已过期 {fixes.get('stale', 0)}。",
        ]
    return section


def proactive_section(ctx: Context) -> Section:
    section = Section()
    run = latest_run(ctx.store, "proactive_audit")
    if run is None:
        section.absent("主动消息审计", "twin eval proactive")
        return section
    section.runs.append(run.id)
    s = run.summary
    mean = s.get("rating_mean")
    section.lines += table(
        [
            "评估记录",
            "日期范围",
            "连续观察",
            "主动消息",
            "深睡时段",
            "间隔 / 追发违规",
            "边缘消息最多（7 天）",
            "/评分",
            "合规",
            "结论",
        ],
        [
            [
                f"`{run.id}`",
                f"{s.get('first_day', '—')} 至 {s.get('last_day', '—')}",
                f"{s.get('streak', '—')}/{len(s.get('days', []))} 天",
                s.get("sent", "—"),
                s.get("deep_sleep", "—"),
                f"{s.get('spacing_violations', '—')} / {s.get('chase_violations', '—')}",
                f"{s.get('edge_max_week', '—')}（上限 {s.get('edge_weekly_max', '—')}）",
                f"{s.get('rating_count', 0)} 次，平均 {number(mean)}",
                "是" if s.get("compliant") else "否",
                VERDICT_WORDS.get(run.verdict or "", NOT_EVALUATED),
            ]
        ],
    )
    return section


def stability_section(ctx: Context) -> Section:
    section = Section()
    run = latest_run(ctx.store, "stability")
    if run is None:
        section.absent("稳定性", "twin eval stability --days 7")
        return section
    section.runs.append(run.id)
    s = run.summary
    launches = ", ".join(f"{k} {v}" for k, v in (s.get("launches") or {}).items()) or "无"
    drill = s.get("drill")
    section.lines += table(
        [
            "评估记录",
            "窗口",
            "健康快照",
            "累计不可用",
            "进程 / 重启",
            "启动方式",
            "通道中断",
            "告警最长延迟",
            "漏报",
            "断网演练",
        ],
        [
            [
                f"`{run.id}`",
                f"{number(s.get('days'), 1)} 天",
                s.get("snapshots", "—"),
                hours(s.get("unavailable_s")),
                f"{s.get('processes', '—')} / {len(s.get('restarts', []))}",
                launches,
                len(s.get("outages", [])),
                hours(s.get("max_alert_latency_s")),
                s.get("missed_alerts", "—"),
                f"有（{hours(drill.get('duration_s'))}）" if isinstance(drill, dict) else "没有",
            ]
        ],
    )
    return section


def _spend_rows(items: Sequence[dict[str, Any]]) -> list[list[object]]:
    return [
        [
            row.get("key"),
            usd(row.get("cost_usd")),
            row.get("calls"),
            percent(row.get("cache_hit_ratio")),
        ]
        for row in items[:TOP_ROWS]
    ]


def cost_section(ctx: Context) -> Section:
    section = Section()
    run = latest_run(ctx.store, "cost")
    if run is None:
        section.absent("成本", "twin eval cost --month <YYYY-MM>")
        return section
    section.runs.append(run.id)
    s = run.summary
    section.lines += table(
        ["评估记录", "月份", "日常账目", "调用", "缓存命中率", "门槛", "月已结束", "结论"],
        [
            [
                f"`{run.id}`",
                run.params.get("month", "—"),
                usd(s.get("total_usd")),
                s.get("calls", "—"),
                percent(s.get("cache_hit_ratio")),
                f"≤ {usd(s.get('limit_usd'))}",
                "是"
                if s.get("complete")
                else f"否（已过 {s.get('days_elapsed')}/{s.get('days_in_month')} 天）",
                VERDICT_WORDS.get(run.verdict or "", NOT_EVALUATED),
            ]
        ],
    )
    headers = ["名称", "费用", "调用", "缓存命中率"]
    by_purpose = table(headers, _spend_rows(s.get("by_purpose", [])))
    by_model = table(headers, _spend_rows(s.get("by_model", [])))
    one_time = f"{usd(s.get('one_time_usd'))}，{s.get('one_time_calls', 0)} 次调用"
    section.lines += ["", "**日常账目 · 按用途**", "", *by_purpose]
    section.lines += ["", "**日常账目 · 按模型**", "", *by_model]
    section.lines += ["", f"**一次性批任务（单列，不计入门槛）**：{one_time}"]
    if s.get("one_time_by_purpose"):
        section.lines += ["", *table(headers, _spend_rows(s["one_time_by_purpose"]))]
    return section


# ------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class WrittenReport:
    """The file that was written and the run that records it."""

    path: Path
    run: RunView
    text: str
    day: date
    milestones: dict[str, dict[str, Any]]
    missing: list[str]


def report_path(services: Services, day: date) -> Path:
    return services.paths.reports_dir / f"{REPORT_PREFIX}{day:%Y-%m-%d}.md"


def build_text(
    services: Services,
) -> tuple[str, date, dict[str, dict[str, Any]], dict[str, list[str]], list[str]]:
    """The Markdown, the local date, the verdicts, the runs by section and what is missing."""
    time = time_service_for(services)
    zone = time.bot_timezone()
    now = services.clock.now_utc()
    day = time.local_date(now)
    ctx = Context(services, EvalStore(services.db, services.clock), zone)
    milestones, stored = milestones_section(ctx)
    measures: list[tuple[str, Section]] = [
        ("盲测（按后端）", blind_section(ctx)),
        ("风格指标（live 与留出集）", style_section(ctx)),
        ("记忆测试", memory_section(ctx)),
        ("前后一致", consistency_section(ctx)),
        ("主动消息审计", proactive_section(ctx)),
        ("稳定性", stability_section(ctx)),
        ("成本（日常与一次性分列）", cost_section(ctx)),
    ]
    passed = sum(1 for item in stored.values() if item["status"] == "passed")
    unrated = sum(1 for item in stored.values() if item["status"] in ("not_run", "not_reached"))
    lines = [
        "# 评估汇总报告",
        "",
        f"生成时间：{stamp(now, zone)}（{zone.key}）。",
        "本报告只读取已存的评估记录，不重新判定，也不含任何聊天内容。",
        "",
        f"里程碑：{len(stored)} 个，通过 {passed} 个，{NOT_EVALUATED} {unrated} 个，"
        f"其余 {len(stored) - passed - unrated} 个未通过。",
        "",
        "## 一、里程碑门槛（M0–M5）",
        "",
        *milestones.lines,
        "## 二、各项度量",
        "",
    ]
    runs: dict[str, list[str]] = {"milestones": milestones.runs}
    missing = list(milestones.missing)
    for number_, (title, section) in enumerate(measures, start=1):
        lines += [f"### 2.{number_} {title}", "", *section.lines, ""]
        runs[title] = section.runs
        missing += section.missing
    lines += ["## 三、尚未评估的项目", ""]
    lines += [f"- {item}" for item in missing] if missing else ["- 没有：以上各项都有评估记录。"]
    lines.append("")
    return "\n".join(lines), day, stored, runs, missing


def write_report(services: Services) -> WrittenReport:
    """Build the report, write ``data/reports/eval-<local date>.md`` and record the run."""
    text, day, stored, runs, missing = build_text(services)
    path = report_path(services, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    run = EvalStore(services.db, services.clock).create_run(
        "report",
        status="done",
        params={"date": day.isoformat(), "file": path.name},
        summary={
            "milestones": stored,
            "runs": runs,
            "missing": len(missing),
            "chars": len(text),
        },
    )
    return WrittenReport(path, run, text, day, stored, missing)
