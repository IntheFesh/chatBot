"""Milestone gates: ``twin eval gate M0|M1|M2|M3|M4|M5`` (R-EVAL-010, SPEC section 26).

A gate is decided by a **judge**: a function ``(GateContext) -> GateVerdict`` registered for one
milestone.  The rounds that own a milestone register theirs (round 10 for M3, round 12 for M4,
round 14 for M5)::

    from twin.eval.gates import Check, GateContext, GateVerdict, gate_judge

    @gate_judge("M3")
    def judge_m3(ctx: GateContext) -> GateVerdict:
        ...
        return GateVerdict("M3", "passed", checks, runs=(...,), summary="...", values={...})

and list the module in :data:`JUDGE_MODULES` (the way job handlers are listed in
``twin.ops.jobs.HANDLER_MODULES``).  A milestone without a judge is not a failure but a stage
that has not been reached: the command says "the gate is wired in round NN" and exits with 2.

Exit codes: 0 passed, 1 not passed (the evidence is not good enough, or there is not enough of
it), 2 not reached yet.

Every judgement is written to ``eval_runs`` (``kind = gate``) with its verdict, the numbers and
the ids of the runs it was decided from; ``--check`` only reads the latest one, so the first
thing a round does is ``twin eval gate <previous milestone> --check`` (CLAUDE.md rule 12).

**The rules are in this file, fixed** (and pinned by tests): they are the SPEC's, and a gate that
is not passed is never made passable by changing a threshold or the way something is counted.

``M0``  the DeepSeek probe passed by the R-LLM-013 rule (thinking switch, cache hit, JPEG and PNG
        vision, JSON output all succeeded) **and** the channel probe is complete with the
        R-CH-009/010 verdict ``met``.
``M1``  in the most recent blind test that has the current default backend: at least 50 valid
        judgements and a guess rate (point estimate) of at most 70 %.
``M2``  in the most recent memory test: twenty reviewed questions, ten from real records and ten
        from the conversation with the bot, and ``correct + 0.5 * partial`` at least 80 %.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

from twin.channel.probe.summary import load_channel_probe_summary
from twin.config.runtime import BACKEND_ACTIVE
from twin.eval.blind import MIN_VALID, blind_report
from twin.eval.memory_test import PASS_RATIO_DEN, PASS_RATIO_NUM, TOTAL, summarize
from twin.eval.stats import wilson_interval
from twin.eval.store import EvalStore, RunView
from twin.llm.probe import load_probe_summary

if TYPE_CHECKING:
    from twin.services import Services

Verdict = Literal["passed", "failed", "insufficient"]
MILESTONES = ("M0", "M1", "M2", "M3", "M4", "M5")
GATE_ROUNDS = {"M3": 10, "M4": 12, "M5": 14}  # the round that registers the judge
EXIT_PASSED, EXIT_NOT_PASSED, EXIT_NOT_REACHED = 0, 1, 2

M1_MAX_GUESS_RATE = Fraction(7, 10)  # SPEC section 26 / R-EVAL-001
M4_MAX_GUESS_RATE = Fraction(6, 10)  # for the judge of round 12
M2_MIN_ACCURACY = Fraction(PASS_RATIO_NUM, PASS_RATIO_DEN)
LLM_GATE_CHECKS = ("thinking_toggle", "cache_hit", "vision", "json_output")  # R-LLM-013 1-4


class GateError(RuntimeError):
    """A gate cannot be judged or read (an unknown milestone, say)."""


@dataclass(frozen=True)
class Check:
    """One criterion of a gate, with the number it was judged by."""

    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None
    samples: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class GateVerdict:
    """What a judge concludes: the verdict, the criteria, the runs and the numbers."""

    milestone: str
    verdict: Verdict
    checks: tuple[Check, ...]
    summary: str
    runs: tuple[str, ...] = ()
    values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "passed"

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "checks": [c.to_json() for c in self.checks],
            "runs": list(self.runs),
            "values": dict(self.values),
        }


@dataclass(frozen=True)
class GateContext:
    """What a judge may read: the services and the evaluation store."""

    services: Services
    store: EvalStore


GateJudge = Callable[[GateContext], GateVerdict]


class GateRegistry:
    """The judges, one per milestone."""

    def __init__(self) -> None:
        self._judges: dict[str, GateJudge] = {}

    def register(self, milestone: str, judge: GateJudge, *, replace: bool = False) -> GateJudge:
        if milestone not in MILESTONES:
            raise GateError(f"unknown milestone {milestone!r}: use {', '.join(MILESTONES)}")
        if milestone in self._judges and not replace:
            raise GateError(f"the milestone {milestone} already has a judge")
        self._judges[milestone] = judge
        return judge

    def get(self, milestone: str) -> GateJudge | None:
        return self._judges.get(milestone)

    def milestones(self) -> frozenset[str]:
        return frozenset(self._judges)


default_registry = GateRegistry()

# Modules that register judges on import; each round that owns a milestone appends its module.
JUDGE_MODULES: tuple[str, ...] = ("twin.eval.gates", "twin.eval.proactive_gate")


def gate_judge(milestone: str) -> Callable[[GateJudge], GateJudge]:
    """Decorator registering ``judge(ctx) -> GateVerdict`` for ``milestone``."""

    def decorate(judge: GateJudge) -> GateJudge:
        return default_registry.register(milestone, judge)

    return decorate


def load_judges(modules: Iterable[str] | None = None) -> GateRegistry:
    """Import every judge module so that the default registry is complete."""
    for name in modules if modules is not None else JUDGE_MODULES:
        importlib.import_module(name)
    return default_registry


def not_reached_message(milestone: str) -> str:
    """What the command says for a milestone nobody has wired in yet."""
    number = GATE_ROUNDS.get(milestone)
    where = f"第 {number:02d} 轮" if number is not None else "后续轮次"
    return f"该门槛在{where}接入（{milestone} 的判定器还没有注册）：这是未到阶段，不是未通过"


# ---------------------------------------------------------------------- judges


def blind_checks(
    ctx: GateContext, backend: str, ceiling: Fraction
) -> tuple[Verdict, list[Check], tuple[str, ...], dict[str, Any], str]:
    """The blind-test criteria against ``ceiling`` for the latest run that has ``backend``."""
    run = ctx.store.latest_run("blind", backend=backend)
    if run is None or run.status == "cancelled":
        check = Check("盲测", False, f"还没有包含 {backend} 后端的盲测：先运行 twin eval blind")
        return "insufficient", [check], (), {"backend": backend}, check.detail
    report = blind_report(ctx.store, run)
    entry = report.of(backend)
    judged = entry.judged if entry else 0
    correct = entry.correct if entry else 0
    interval = wilson_interval(correct, judged)
    enough = judged >= MIN_VALID
    rate = Fraction(correct, judged) if judged else None
    low = rate is not None and rate <= ceiling
    checks = [
        Check(
            f"有效判断 ≥ {MIN_VALID} 对",
            enough,
            f"{backend} 后端有 {judged} 对有效判断（跳过不计）",
            float(judged),
            float(MIN_VALID),
            judged,
        ),
        Check(
            f"猜对率点估计 ≤ {float(ceiling):.0%}",
            enough and low,
            f"猜对 {correct}/{judged}"
            + (f" = {float(rate):.1%}" if rate is not None else "")
            + (f"，95% 区间 {interval[0]:.1%}–{interval[1]:.1%}" if interval else ""),
            float(rate) if rate is not None else None,
            float(ceiling),
            judged,
        ),
    ]
    values: dict[str, Any] = {
        "backend": backend,
        "valid": judged,
        "correct": correct,
        "rate": float(rate) if rate is not None else None,
        "interval": list(interval) if interval else None,
        "ceiling": float(ceiling),
        "minimum_valid": MIN_VALID,
    }
    verdict: Verdict = "insufficient" if not enough else ("passed" if low else "failed")
    return verdict, checks, (run.id,), values, checks[1].detail


@gate_judge("M1")
def judge_m1(ctx: GateContext) -> GateVerdict:
    """M1: the default backend's guess rate in the latest blind test is at most 70 %."""
    backend = str(ctx.services.runtime.get(BACKEND_ACTIVE))
    verdict, checks, runs, values, detail = blind_checks(ctx, backend, M1_MAX_GUESS_RATE)
    return GateVerdict("M1", verdict, tuple(checks), detail, runs, values)


@gate_judge("M2")
def judge_m2(ctx: GateContext) -> GateVerdict:
    """M2: the latest memory test has twenty reviewed questions of both sources and scores 80 %."""
    run = ctx.store.latest_run("memory")
    if run is None or run.status == "cancelled":
        check = Check("记忆测试", False, "还没有记忆测试：先运行 twin eval memory")
        return GateVerdict("M2", "insufficient", (check,), check.detail)
    items = ctx.store.items(run.id, with_payload=False)
    score = summarize(items)
    accuracy = score.accuracy
    checks = [
        Check(
            "来源配比完整：真实记录 10 题 + 机器人对话 10 题",
            score.composition_complete,
            f"真实记录 {score.real_items} 题，机器人对话 {score.bot_items} 题"
            + ("" if score.composition_complete else "（样本不足，未通过）"),
            float(score.real_items + score.bot_items),
            float(TOTAL),
            score.total,
        ),
        Check(
            "20 题都已复核",
            score.reviewed == TOTAL,
            f"已复核 {score.reviewed}/{TOTAL}",
            float(score.reviewed),
            float(TOTAL),
            score.total,
        ),
        Check(
            f"正确率 ≥ {float(M2_MIN_ACCURACY):.0%}（partial 记 0.5）",
            score.complete and score.meets_threshold,
            f"正确 {score.correct}、部分 {score.partial}、错误 {score.wrong}"
            + (
                f"，得分 {score.points:g}/{score.total} = {accuracy:.1%}"
                if accuracy is not None
                else ""
            ),
            accuracy,
            float(M2_MIN_ACCURACY),
            score.total,
        ),
    ]
    values: dict[str, Any] = {
        "total": score.total,
        "reviewed": score.reviewed,
        "correct": score.correct,
        "partial": score.partial,
        "wrong": score.wrong,
        "real_items": score.real_items,
        "bot_items": score.bot_items,
        "points": score.points,
        "accuracy": accuracy,
        "threshold": float(M2_MIN_ACCURACY),
    }
    return GateVerdict("M2", score.verdict, tuple(checks), checks[2].detail, (run.id,), values)


def llm_probe_checks(ctx: GateContext) -> list[Check]:
    """The DeepSeek part of M0: the R-LLM-013 rule on the stored probe result."""
    with ctx.services.db.session() as session:
        summary = load_probe_summary(session)
    if summary is None:
        return [Check("DeepSeek 探针（R-LLM-013）", False, "没有探针结果：先运行 twin llm probe")]
    checks = [
        Check(
            f"探针检查 {name}",
            summary.gate_checks.get(name) is True,
            "成功" if summary.gate_checks.get(name) is True else "失败或没有跑",
        )
        for name in LLM_GATE_CHECKS
    ]
    checks.append(
        Check(
            "探针跑完且没有致命错误",
            summary.fatal is None and summary.m0_passed,
            "通过" if summary.fatal is None and summary.m0_passed else f"致命错误：{summary.fatal}",
        )
    )
    return checks


def channel_probe_checks(ctx: GateContext) -> list[Check]:
    """The channel part of M0: a complete probe whose R-CH-010 verdict is ``met``."""
    with ctx.services.db.session() as session:
        summary = load_channel_probe_summary(session)
    if summary is None:
        return [Check("通道探针（R-CH-009/010）", False, "没有探针结果：先运行 twin channel probe")]
    ok = summary.complete and summary.meets_requirement is True
    detail = f"判定 {summary.verdict}，探针{'已完成' if summary.complete else '未完成'}"
    if summary.reasons and not ok:
        detail += "：" + "；".join(summary.reasons)
    return [Check("通道探针（R-CH-009/010）", ok, detail)]


@gate_judge("M0")
def judge_m0(ctx: GateContext) -> GateVerdict:
    """M0: the DeepSeek probe (R-LLM-013) and the channel probe (R-CH-009/010) both pass."""
    checks = [*llm_probe_checks(ctx), *channel_probe_checks(ctx)]
    passed = all(c.passed for c in checks)
    missing = any("没有探针结果" in c.detail for c in checks)
    verdict: Verdict = "passed" if passed else ("insufficient" if missing else "failed")
    summary = "M0 通过" if passed else "；".join(c.detail for c in checks if not c.passed)
    return GateVerdict("M0", verdict, tuple(checks), summary)


# ------------------------------------------------------------------- running


@dataclass(frozen=True)
class GateOutcome:
    """What ``twin eval gate`` reports and exits with."""

    milestone: str
    exit_code: int
    status: Literal["passed", "failed", "insufficient", "not_reached", "not_run"]
    message: str
    verdict: GateVerdict | None = None
    run: RunView | None = None
    checked: bool = False  # read from the store (``--check``), not judged now


def _as_verdict(value: str) -> Verdict:
    """A stored verdict (the table only holds these three)."""
    if value == "passed":
        return "passed"
    return "failed" if value == "failed" else "insufficient"


def _exit_of(verdict: str) -> int:
    return EXIT_PASSED if verdict == "passed" else EXIT_NOT_PASSED


def _normalise(milestone: str) -> str:
    name = milestone.strip().upper()
    if name not in MILESTONES:
        raise GateError(f"unknown milestone {milestone!r}: use {', '.join(MILESTONES)}")
    return name


def run_gate(
    services: Services, milestone: str, *, registry: GateRegistry | None = None
) -> GateOutcome:
    """Judge a milestone now and store the verdict (``eval_runs``, kind ``gate``)."""
    name = _normalise(milestone)
    judges = registry or load_judges()
    judge = judges.get(name)
    if judge is None:
        return GateOutcome(name, EXIT_NOT_REACHED, "not_reached", not_reached_message(name))
    store = EvalStore(services.db, services.clock)
    verdict = judge(GateContext(services, store))
    run = store.create_run(
        "gate",
        status="done",
        milestone=name,
        verdict=verdict.verdict,
        params={"milestone": name},
        summary=verdict.to_json(),
    )
    return GateOutcome(
        name, _exit_of(verdict.verdict), verdict.verdict, verdict.summary, verdict, run
    )


def check_gate(
    services: Services, milestone: str, *, registry: GateRegistry | None = None
) -> GateOutcome:
    """Read the latest stored verdict of a milestone; nothing is judged or written."""
    name = _normalise(milestone)
    judges = registry or load_judges()
    if judges.get(name) is None:
        return GateOutcome(
            name, EXIT_NOT_REACHED, "not_reached", not_reached_message(name), checked=True
        )
    store = EvalStore(services.db, services.clock)
    run = store.latest_run("gate", milestone=name)
    if run is None or run.verdict is None:
        return GateOutcome(
            name,
            EXIT_NOT_PASSED,
            "not_run",
            f"{name} 还没有判定记录：先运行 twin eval gate {name}",
            checked=True,
        )
    stored = run.summary
    checks = tuple(
        Check(
            str(c["name"]),
            bool(c["passed"]),
            str(c["detail"]),
            c.get("value"),
            c.get("threshold"),
            c.get("samples"),
        )
        for c in stored.get("checks", [])
    )
    stored_verdict = _as_verdict(run.verdict)
    verdict = GateVerdict(
        name,
        stored_verdict,
        checks,
        str(stored.get("summary", "")),
        tuple(str(r) for r in stored.get("runs", [])),
        dict(stored.get("values", {})),
    )
    return GateOutcome(
        name,
        _exit_of(stored_verdict),
        stored_verdict,
        verdict.summary,
        verdict,
        run,
        checked=True,
    )
