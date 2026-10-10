"""R-EVAL-008: ``twin eval report`` - every stored result in one Markdown file.

The report reads what ``eval_runs`` holds and nothing else: the verdicts of the gates are the ones
``twin eval gate --check`` shows, a result that was never produced says "未评估", and the file
contains no chat text (a synthetic conversation with marker words is scanned for).
"""

from __future__ import annotations

import io
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.unit.test_eval_cost import call
from tests.unit.test_eval_gates import blind_run, channel_probe, llm_probe, memory_run
from twin.channel.probe.report import MEASURED_MARKER as CHANNEL_MEASURED
from twin.channel.probe.report import PENDING_MARKER as CHANNEL_PENDING
from twin.cli import app
from twin.eval.blind import MIN_VALID
from twin.eval.cli import use_interaction
from twin.eval.consistency_model import Finding, Statement, fingerprint_of
from twin.eval.consistency_store import ConsistencyStore, NewFinding
from twin.eval.cost_gate import evaluate_cost
from twin.eval.gates import MILESTONES as GATE_MILESTONES
from twin.eval.gates import check_gate, run_gate
from twin.eval.store import EvalStore, NewItem
from twin.eval.style_metrics import STYLE_METRICS, MetricReading, StyleReport, judge_metric
from twin.eval.summary_report import (
    MILESTONES,
    NOT_EVALUATED,
    VERDICT_WORDS,
    build_text,
    cell,
    doc_state,
    report_path,
    write_report,
)
from twin.eval.ui import LineKeys
from twin.llm.ledger import LedgerStore
from twin.llm.probe_report import PENDING_MARKER as LLM_PENDING
from twin.ops.stability import run_stability
from twin.schedule.service import time_service_for
from twin.services import Services

SPEC = Path(__file__).resolve().parents[2] / "docs" / "SPEC.md"
REPO_DOCS = Path(__file__).resolve().parents[2] / "docs"
runner = CliRunner()
SECRET = "绝密聊天暗号"


def store_of(services: Services) -> EvalStore:
    return EvalStore(services.db, services.clock)


def text_of(services: Services) -> str:
    return build_text(services)[0]


def section(text: str, heading: str) -> str:
    """The lines under a heading, up to the next heading of the same or a higher level."""
    lines = text.splitlines()
    level = len(heading) - len(heading.lstrip("#"))
    body: list[str] = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            break
        body.append(line)
    return "\n".join(body)


def table_rows(text: str) -> list[list[str]]:
    rows = [
        [part.strip() for part in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("| ") and "---" not in line
    ]
    return rows


# -------------------------------------------------------------------------- the SPEC


def test_the_milestones_are_the_rows_of_the_table_of_the_spec() -> None:
    text = SPEC.read_text(encoding="utf-8")
    rows = [
        [part.strip() for part in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if re.match(r"\| M\d ", line)
    ]
    assert [r[0].split(" ", 1)[0] for r in rows] == [m.code for m in MILESTONES]
    assert [m.code for m in MILESTONES] == list(GATE_MILESTONES)
    for spec, row in zip(MILESTONES, rows, strict=True):
        assert row[0] == f"{spec.code} {spec.title}", spec.code
        assert row[2].replace("`", "") == spec.condition, spec.code
        assert row[1] == spec.content, spec.code


# ----------------------------------------------------------------------------- empty


def test_a_database_without_any_evaluation_says_not_evaluated_everywhere(
    services: Services,
) -> None:
    text = text_of(services)
    overview = table_rows(section(text, "## 一、里程碑门槛（M0–M5）"))
    codes = [row[0] for row in overview[1:7]]
    assert codes == ["M0", "M1", "M2", "M3", "M4", "M5"]
    for row in overview[1:7]:
        assert row[3] == NOT_EVALUATED and row[4] == NOT_EVALUATED, row
        assert f"`twin eval gate {row[0]}`（尚无判定记录）" in row[5]
    for number, title in enumerate(
        (
            "盲测（按后端）",
            "风格指标（live 与留出集）",
            "记忆测试",
            "前后一致",
            "主动消息审计",
            "稳定性",
            "成本（日常与一次性分列）",
        ),
        start=1,
    ):
        assert NOT_EVALUATED in section(text, f"### 2.{number} {title}"), title
    tail = text.split("## 三、尚未评估的项目", 1)[1]
    for command in (
        "twin eval gate M0",
        "twin eval blind --backend deepseek",
        "twin model evaluate <模型>",
        "twin eval style --source live",
        "twin eval memory",
        "twin eval consistency --days 7",
        "twin eval proactive",
        "twin eval stability --days 7",
        "twin eval cost --month <YYYY-MM>",
    ):
        assert command in tail, command
    assert "里程碑：6 个，通过 0 个，未评估 6 个，其余 0 个未通过。" in text
    assert "没有：以上各项都有评估记录" not in text


def test_the_file_is_named_for_the_local_date_and_overwritten_the_same_day(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(datetime(2026, 10, 10, 2, tzinfo=UTC))  # 21:00 on the 9th in Chicago
    first = write_report(services)
    assert first.path == services.paths.reports_dir / "eval-2026-10-09.md"
    assert (
        first.path.read_text(encoding="utf-8") == first.text
        and first.day.isoformat() == "2026-10-09"
    )
    clock.tick(3 * 3600)  # the 10th, local time
    second = write_report(services)
    assert second.path.name == "eval-2026-10-10.md" and first.path.is_file()
    clock.tick(600)
    third = write_report(services)
    assert (
        third.path == second.path and len(list(services.paths.reports_dir.glob("eval-*.md"))) == 2
    )
    assert report_path(services, first.day) == first.path
    assert "CDT" in first.text and "America/Chicago" in first.text  # the zone is the bot's


# ----------------------------------------------------------------------- the milestones


def test_the_verdicts_are_the_stored_ones_with_their_evidence(services: Services) -> None:
    llm_probe(services)
    channel_probe(services)
    blind = blind_run(services, valid=50, correct=30)
    memory = memory_run(services, correct=14, partial=0, wrong=6)  # 70 %: not enough
    judged = {code: run_gate(services, code) for code in ("M0", "M1", "M2")}
    text = text_of(services)
    overview = {row[0]: row for row in table_rows(section(text, "## 一、里程碑门槛（M0–M5）"))[1:7]}
    for code in ("M0", "M1", "M2"):
        outcome = judged[code]
        assert outcome.run is not None
        assert overview[code][3] == VERDICT_WORDS[outcome.status]
        assert f"记录 `{outcome.run.id}`" in overview[code][5]
        assert f"twin eval gate {code}" in overview[code][5]
        stored = check_gate(services, code)
        assert stored.status == outcome.status  # what --check says is what the report says
    assert [overview[c][3] for c in ("M0", "M1", "M2")] == ["通过", "通过", "未通过"]
    assert [overview[c][3] for c in ("M3", "M4", "M5")] == [NOT_EVALUATED] * 3
    assert f"`{blind}`" in overview["M1"][5] and f"`{memory}`" in overview["M2"][5]
    assert "里程碑：6 个，通过 2 个，未评估 3 个，其余 1 个未通过。" in text
    assert "2026-10-09 07:00 CDT" in overview["M1"][4]  # when it was judged, on the bot's clock


def test_each_milestone_lists_threshold_value_samples_and_confidence(services: Services) -> None:
    blind_run(services, valid=50, correct=35)  # 70 % exactly
    run_gate(services, "M1")
    detail = section(text_of(services), "### M1 能聊、像她")
    rows = table_rows(detail)
    assert rows[0] == ["条件", "当前值 / 数据", "样本数", "结果"]
    by_name = {row[0]: row for row in rows[1:]}
    valid = by_name[f"有效判断 ≥ {MIN_VALID} 对"]
    assert (
        valid[1] == "deepseek 后端有 50 对有效判断（跳过不计）"
        and valid[2] == "50"
        and valid[3] == "达标"
    )
    rate = by_name["猜对率点估计 ≤ 70%"]
    assert "猜对 35/50 = 70.0%" in rate[1] and "95% 区间" in rate[1] and rate[3] == "达标"
    assert re.search(r"95% 置信区间（Wilson）：\d+\.\d%–\d+\.\d%", detail)
    assert "过关条件：盲测猜对率 ≤ 70%" in detail
    failing = section(text_of(services), "### M2 记得住")
    assert "未评估" in failing and "`twin eval memory`" in failing


def test_a_failed_gate_says_which_criterion_missed(services: Services) -> None:
    blind_run(services, valid=50, correct=40)  # 80 %
    run_gate(services, "M1")
    detail = section(text_of(services), "### M1 能聊、像她")
    assert "结论：**未通过**" in detail
    rows = {r[0]: r for r in table_rows(detail)[1:]}
    assert rows["猜对率点估计 ≤ 70%"][3] == "未达标"
    sparse = blind_run(services, valid=10, correct=2)
    services.clock.tick(60)  # type: ignore[attr-defined]
    run_gate(services, "M1")
    assert sparse
    assert "结论：**未通过（样本不足）**" in section(text_of(services), "### M1 能聊、像她")


def test_a_newer_test_than_the_verdict_is_pointed_out(
    services: Services, clock: ManualClock
) -> None:
    blind_run(services, valid=50, correct=20)
    run_gate(services, "M1")
    assert "判定之后又有新的" not in text_of(services)
    clock.tick(3600)
    newer = blind_run(services, valid=50, correct=45)
    text = text_of(services)
    assert f"判定之后又有新的 blind 评估记录 `{newer}`" in section(text, "### M1 能聊、像她")
    assert "重新运行 `twin eval gate M1`" in text


def test_m0_says_honestly_that_the_probe_reports_are_still_templates(
    services: Services,
) -> None:
    docs = services.paths.root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    detail = section(text_of(services), "### M0 技术验证")
    assert "`docs/CHANNEL_REPORT.md`：文件不存在" in detail
    for name in ("CHANNEL_REPORT.md", "LLM_REPORT.md"):
        (docs / name).write_text((REPO_DOCS / name).read_text(encoding="utf-8"), encoding="utf-8")
    detail = section(text_of(services), "### M0 技术验证")
    assert "`docs/CHANNEL_REPORT.md`：待实测（仍是模板，没有实测数字）" in detail
    assert "`docs/LLM_REPORT.md`：待实测（仍是模板，没有实测数字）" in detail
    (docs / "CHANNEL_REPORT.md").write_text("# x\n" + CHANNEL_MEASURED + "\n", encoding="utf-8")
    assert "`docs/CHANNEL_REPORT.md`：已由探针写成实测报告" in section(
        text_of(services), "### M0 技术验证"
    )
    (docs / "LLM_REPORT.md").write_text("something else", encoding="utf-8")
    assert "`docs/LLM_REPORT.md`：无法识别" in section(text_of(services), "### M0 技术验证")


def test_the_templates_in_the_repository_are_recognised_as_pending() -> None:
    assert "待实测" in doc_state(REPO_DOCS / "CHANNEL_REPORT.md", CHANNEL_PENDING, CHANNEL_MEASURED)
    assert "待实测" in doc_state(REPO_DOCS / "LLM_REPORT.md", LLM_PENDING, CHANNEL_MEASURED)


def test_m5_is_marked_as_not_blocking(services: Services) -> None:
    assert "不阻塞" in section(text_of(services), "### M5 风格模型")


# ------------------------------------------------------------------------- the measures


def test_the_blind_tests_are_listed_by_backend_with_their_intervals(services: Services) -> None:
    run_id = blind_run(services, backend="deepseek", valid=60, correct=33)
    text = text_of(services)
    blind = section(text, "### 2.1 盲测（按后端）")
    rows = {r[0]: r for r in table_rows(blind)[1:]}
    assert rows["deepseek"][1:5] == [f"`{run_id}`", "60", "33", "55.0%"]
    assert re.fullmatch(r"\d+\.\d%–\d+\.\d%", rows["deepseek"][5])
    assert "M1 ≤ 70%：达标；M4 ≤ 60%：达标" in rows["deepseek"][6]
    assert rows["style"][1] == NOT_EVALUATED and rows["hybrid"][1] == NOT_EVALUATED
    assert "twin model evaluate <模型>" in text
    blind_run(services, backend="style", valid=40, correct=10)
    services.clock.tick(60)  # type: ignore[attr-defined]
    rows2 = {r[0]: r for r in table_rows(section(text_of(services), "### 2.1 盲测（按后端）"))[1:]}
    assert rows2["style"][2] == "40" and f"样本不足（40 < {MIN_VALID}）" in rows2["style"][6]


def test_a_style_model_is_set_against_deepseek_when_both_were_in_one_test(
    services: Services,
) -> None:
    store = store_of(services)
    run = store.create_run("blind", mode="holdout", backends=["deepseek", "style"], status="done")
    items = [
        NewItem(f"{backend}{n}", backend, datetime(2026, 9, 1, tzinfo=UTC), {})
        for backend in ("deepseek", "style")
        for n in range(60)
    ]
    store.add_items(run.id, items)
    for item in store.items(run.id):
        wrong = 20 if item.backend == "deepseek" else 40
        store.save_generated(item.id, {"bot": {"lines": [], "quote": None}}, cost_usd=0.0)
        store.judge(item.id, "wrong" if item.seq % 60 < wrong else "correct", score=0.0)
    rows = {r[0]: r for r in table_rows(section(text_of(services), "### 2.1 盲测（按后端）"))[1:]}
    assert rows["deepseek"][1] == f"`{run.id}`" and rows["style"][1] == f"`{run.id}`"
    found = re.fullmatch(
        r"与 deepseek 同批对比：style 更低的单侧 p = (\d\.\d{4})", rows["style"][6]
    )
    assert found is not None
    # her real reply is found 40 times in 60 against deepseek and 20 times in 60 against the
    # style model: the model is the harder one to tell from her, and the test says so
    assert float(found.group(1)) < 0.1


def style_run(services: Services, source: str, backend: str | None, values: float) -> str:
    results = [
        judge_metric(spec, MetricReading(1.0, 100), MetricReading(values, 100))
        for spec in STYLE_METRICS
    ]
    report = StyleReport(
        source, "live" if source == "live" else "pre_holdout", results, backend=backend, days=7,
        messages=80,
    )  # fmt: skip
    return (
        store_of(services)
        .create_run(
            "style",
            mode="live" if source == "live" else "holdout",
            backends=[backend] if backend else [],
            status="done",
            verdict="passed" if report.passed else "failed",
            params={"source": source, "days": 7, "blind_run": None, "backend": backend},
            summary=report.to_json(),
        )
        .id
    )


def test_the_style_metrics_are_shown_for_the_live_output_and_the_holdout(
    services: Services,
) -> None:
    live = style_run(services, "live", None, 1.1)
    holdout = style_run(services, "eval_items", "style", 1.6)
    detail = section(text_of(services), "### 2.2 风格指标（live 与留出集）")
    assert f"记录 `{live}`" in detail and f"记录 `{holdout}`" in detail
    assert "live：机器人最近的真实输出对 live 画像" in detail and "留出集：style 后端" in detail
    assert "每项偏差在 ±30% 内：通过" in detail and "有指标超出 ±30%：不通过" in detail
    assert "文字长度中位数" in detail and "+10.0%" in detail and "+60.0%" in detail
    assert "机器人消息 80 条" in detail
    assert NOT_EVALUATED not in detail


def test_the_style_section_says_what_is_missing(services: Services) -> None:
    style_run(services, "live", None, 1.0)
    detail = section(text_of(services), "### 2.2 风格指标（live 与留出集）")
    assert "未评估：风格指标（留出集）" in detail and "未评估：风格指标（live）" not in detail


def test_the_memory_test_is_shown_with_its_score(services: Services) -> None:
    run_id = memory_run(services, correct=14, partial=4, wrong=2)  # 16/20 = 80 %
    row = table_rows(section(text_of(services), "### 2.3 记忆测试"))[1]
    assert row[0] == f"`{run_id}`" and row[1] == "20" and row[2] == "真实记录 10 + 机器人对话 10"
    assert row[3] == "20/20" and row[4] == "14 / 4 / 2" and row[5] == "16"
    assert row[6] == "80.0%" and row[7] == "≥ 80%（partial 记 0.5）" and row[8] == "通过"


def test_the_consistency_audit_is_shown_in_every_state(services: Services) -> None:
    store = store_of(services)
    waiting = store.create_run(
        "consistency",
        mode="live",
        status="running",
        params={"days": 7},
        summary={
            "evidence": {"lifeline": 12, "replies": 40, "facts": 9},
            "decisions": {
                "confirmed_obvious": 0,
                "confirmed_minor": 0,
                "rejected": 0,
                "undecided": 3,
            },
            "allowed_obvious": 1.0,
        },
    )
    row = table_rows(section(text_of(services), "### 2.4 前后一致"))[1]
    assert row[0] == f"`{waiting.id}`" and row[2] == "12 / 40 / 9"
    assert "待确认：还有 3 条矛盾没有决定" in row[7]
    services.clock.tick(60)  # type: ignore[attr-defined]
    done = store.create_run(
        "consistency",
        mode="live",
        status="done",
        verdict="failed",
        params={"days": 7},
        summary={
            "evidence": {"lifeline": 12, "replies": 40, "facts": 9},
            "decisions": {
                "confirmed_obvious": 2,
                "confirmed_minor": 1,
                "rejected": 4,
                "undecided": 0,
            },
            "allowed_obvious": 1.0,
            "fixes": {"applied": 1, "declined": 2, "proposed": 0, "stale": 0},
        },
    )
    detail = section(text_of(services), "### 2.4 前后一致")
    row = table_rows(detail)[1]
    assert row[0] == f"`{done.id}`" and row[3] == "2" and row[4] == "≤ 1.0" and row[5] == "1"
    assert row[6] == "4" and row[7] == "未通过"
    assert "已应用 1，不改 2，待决定 0，已过期 0" in detail
    services.clock.tick(60)  # type: ignore[attr-defined]
    store.create_run("consistency", mode="live", status="failed", params={"days": 7})
    assert "审计没有成功" in section(text_of(services), "### 2.4 前后一致")


def test_the_proactive_audit_and_the_stability_report_are_shown(services: Services) -> None:
    audit = store_of(services).create_run(
        "proactive_audit",
        status="done",
        verdict="passed",
        params={"days": 7},
        summary={
            "first_day": "2026-10-01",
            "last_day": "2026-10-07",
            "days": [{}] * 7,
            "streak": 7,
            "sent": 21,
            "deep_sleep": 0,
            "spacing_violations": 0,
            "chase_violations": 0,
            "edge_max_week": 1,
            "edge_weekly_max": 2,
            "rating_count": 3,
            "rating_mean": 4.33,
            "compliant": True,
        },
    )
    row = table_rows(section(text_of(services), "### 2.5 主动消息审计"))[1]
    assert row[0] == f"`{audit.id}`" and row[1] == "2026-10-01 至 2026-10-07" and row[2] == "7/7 天"
    assert row[3] == "21" and row[4] == "0" and row[5] == "0 / 0"
    assert (
        row[6] == "1（上限 2）"
        and row[7] == "3 次，平均 4.33"
        and row[8] == "是"
        and row[9] == "通过"
    )
    _, stability = run_stability(services, 7.0)
    row = table_rows(section(text_of(services), "### 2.6 稳定性"))[1]
    assert row[0] == f"`{stability.id}`" and row[1] == "7.0 天" and row[2] == "0"
    assert row[9] == "没有"  # no drill


def test_the_cost_is_shown_with_the_one_time_batches_apart(services: Services) -> None:
    from twin.llm.types import LedgerTag

    ledger = LedgerStore(services.db, services.clock, time_service_for(services))
    ledger.record(
        call(datetime(2026, 9, 3, 17, tzinfo=UTC), 6.0, purpose="reply", hit=900, miss=100)
    )
    ledger.record(
        call(datetime(2026, 9, 4, 17, tzinfo=UTC), 2.0, purpose="plan", model="deepseek-pro")
    )
    ledger.record(
        call(
            datetime(2026, 9, 5, 17, tzinfo=UTC),
            40.0,
            purpose="persona",
            tag=LedgerTag("one_time", "b1"),
        )
    )
    result = evaluate_cost(services, date(2026, 9, 1))
    detail = section(text_of(services), "### 2.7 成本（日常与一次性分列）")
    row = table_rows(detail)[1]
    assert result.run is not None
    assert row[0] == f"`{result.run.id}`" and row[1] == "2026-09" and row[2] == "$8.00"
    assert row[3] == "2" and row[5] == "≤ $15.00" and row[6] == "是" and row[7] == "通过"
    assert "**日常账目 · 按用途**" in detail and "**日常账目 · 按模型**" in detail
    assert "**一次性批任务（单列，不计入门槛）**：$40.00，1 次调用" in detail
    tables = detail.split("**一次性批任务")[1]
    assert "persona" in tables and "persona" not in detail.split("**一次性批任务")[0]


# ------------------------------------------------------------------- no chat in the file


def test_the_report_holds_none_of_the_conversation(services: Services) -> None:
    store = store_of(services)
    chat = {
        "shown": [{"who": "me", "lines": [f"{SECRET}用户说"]}],
        "real": {"lines": [{"text": f"{SECRET}她说"}]},
    }
    run = store.create_run("blind", mode="holdout", backends=["deepseek"], status="running")
    store.add_items(
        run.id,
        [NewItem(f"k{n}", "deepseek", datetime(2026, 9, 1, tzinfo=UTC), chat) for n in range(55)],
    )
    for item in store.items(run.id):
        store.save_generated(
            item.id,
            {"bot": {"lines": [{"text": f"{SECRET}机器人说"}], "quote": None}},
            cost_usd=0.0,
        )
        store.judge(item.id, "correct" if item.seq % 2 else "wrong", score=1.0)
    memory = store.create_run("memory", mode="live", status="running", backends=["deepseek"])
    store.add_items(
        memory.id,
        [
            NewItem(
                f"f{n}",
                "deepseek",
                datetime(2026, 9, 1, tzinfo=UTC),
                {"question": f"{SECRET}问题", "fact": f"{SECRET}事实"},
                "real_record" if n < 10 else "bot_invented",
            )
            for n in range(20)
        ],
    )
    for item in store.items(memory.id):
        store.save_auto(item.id, {"answer": f"{SECRET}回答"}, auto_outcome="correct", cost_usd=0.0)
        store.judge(item.id, "correct", score=1.0)
    consistency = store.create_run("consistency", mode="live", status="running", params={"days": 7})
    first = Statement(
        "L1", "lifeline", "a" * 26, f"{SECRET}生活安排", "plan", datetime(2026, 10, 1, tzinfo=UTC)
    )
    second = Statement(
        "R1", "reply", "b" * 26, f"{SECRET}她说的话", None, datetime(2026, 10, 1, tzinfo=UTC)
    )
    ConsistencyStore(services.db, services.clock).add_findings(
        consistency.id,
        [
            NewFinding(
                Finding(
                    f"{SECRET}时间",
                    first.at,
                    first,
                    second,
                    (),
                    "obvious",
                    f"{SECRET}理由",
                    None,
                    None,
                    fingerprint_of(first, second),
                )
            )
        ],
    )
    run_gate(services, "M1")
    run_gate(services, "M2")
    text = text_of(services)
    assert SECRET not in text
    written = write_report(services)
    assert SECRET not in written.path.read_text(encoding="utf-8")
    assert "M1" in text and "记忆测试" in text  # the sections are really there


# -------------------------------------------------------------------- the run and the CLI


def test_the_report_is_recorded_as_a_run_with_the_verdicts_and_the_runs_it_read(
    services: Services,
) -> None:
    blind = blind_run(services, valid=50, correct=30)
    gate = run_gate(services, "M1")
    written = write_report(services)
    run = store_of(services).get_run(written.run.id)
    assert run.kind == "report" and run.status == "done" and run.verdict is None
    assert run.params == {"date": "2026-10-09", "file": "eval-2026-10-09.md"}
    assert gate.run is not None
    assert run.summary["milestones"]["M1"] == {"status": "passed", "run": gate.run.id}
    assert run.summary["milestones"]["M3"] == {"status": "not_run", "run": None}
    assert gate.run.id in run.summary["runs"]["milestones"]
    assert blind in run.summary["runs"]["盲测（按后端）"]
    assert run.summary["missing"] == len(written.missing) and run.summary["chars"] == len(
        written.text
    )


def test_table_cells_cannot_break_the_table() -> None:
    assert cell("a|b\nc") == "a／b c" and cell("") == "—" and cell("  ") == "—"
    assert cell(12) == "12"


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


def test_the_command_writes_the_file_and_prints_the_milestones(data_dir: Path) -> None:
    screen = io.StringIO()
    console = Console(file=screen, width=140, color_system=None, highlight=False)
    with use_interaction(console, LineKeys(io.StringIO(""))):
        result = runner.invoke(app, ["eval", "report"])
    assert result.exit_code == 0, result.output
    shown = screen.getvalue()
    for code in ("M0", "M1", "M2", "M3", "M4", "M5"):
        assert code in shown
    assert shown.count(NOT_EVALUATED) >= 6 and "报告已写入" in shown and "评估记录：" in shown
    assert "尚未评估的项目" in shown
    written = list((data_dir / "reports").glob("eval-*.md"))
    assert len(written) == 1 and written[0].read_text(encoding="utf-8").startswith("# 评估汇总报告")
    listed = io.StringIO()
    with use_interaction(
        Console(file=listed, width=140, color_system=None, highlight=False),
        LineKeys(io.StringIO("")),
    ):
        runner.invoke(app, ["eval", "runs", "--kind", "report"])
    assert "report" in listed.getvalue()
