"""The judge of milestone M5 "style model" and the release gate of a model (R-SRV-005, R-EVAL-010).

``twin model evaluate <id>`` has the same new hold-out contexts answered by the DeepSeek backend
and by the model, as ``style`` (the model alone) and as ``hybrid`` (DeepSeek plans, the model
writes), and the user picks "which one is hers" in each pair.  The model may become the default
backend only if, **for at least one of the two ways**, all of this holds (the numbers are fixed
here and pinned by tests; nothing is waived and a gate that is not passed is never made passable
by moving a threshold):

1. at least **50 valid judgements** for that backend *and* for the DeepSeek backend (a skipped pair
   is not one) - with fewer the verdict is ``insufficient`` and says how many pairs are missing;
2. the way's guess rate is **lower** than the DeepSeek backend's, by a one-sided two-proportion
   test with ``p < 0.1`` (the user finds it harder to tell the model from her than DeepSeek);
3. the six core **style metrics** of the way's replies on those contexts (median text length,
   comma rate, median burst size, sticker share, emoji-code rate, quote rate) are each within
   +-30 % of her profile (``pre_holdout`` scope, R-EVAL-002).

If both ways pass, the one with the lower guess rate is the winner (a tie goes to ``style``, which
is cheaper); ``twin model activate`` then makes it ``backend.active``.  If neither passes the
DeepSeek backend stays the default: that does not block anything later.

The evidence is every blind run that was drawn for the model (``params.model_id``), pooled: more
contexts can be added with another ``twin model evaluate`` and never overlap (a context is used
once).  ``twin eval gate M5`` judges the model evaluated most recently; ``twin model activate <id>``
judges the model it is asked to activate, with :func:`judge_model`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from twin.eval.blind import MIN_VALID, blind_report
from twin.eval.gates import Check, GateContext, GateVerdict, Verdict, gate_judge
from twin.eval.stats import ProportionTest, two_proportion_test
from twin.eval.store import EvalStore, RunView
from twin.eval.style_metrics import StyleError, StyleReport, style_of_runs

if TYPE_CHECKING:
    from twin.services import Services

BASELINE: Final = "deepseek"
WAYS: Final = ("style", "hybrid")  # in the order a tie is decided
EVAL_BACKENDS: Final = (BASELINE, *WAYS)
P_THRESHOLD: Final = 0.1  # R-SRV-005: one-sided p of "the way guesses less often than DeepSeek"
MODEL_PARAM: Final = "model_id"

StyleOf = Callable[["Services", EvalStore, Sequence[RunView], str], StyleReport]


@dataclass(frozen=True)
class WayResult:
    """One way of using the model against the DeepSeek backend, with every number."""

    backend: str
    judged: int
    correct: int
    baseline_judged: int
    baseline_correct: int
    test: ProportionTest | None
    style: StyleReport | None
    style_error: str | None

    @property
    def rate(self) -> float | None:
        return self.correct / self.judged if self.judged else None

    @property
    def baseline_rate(self) -> float | None:
        return self.baseline_correct / self.baseline_judged if self.baseline_judged else None

    @property
    def missing(self) -> int:
        """The most pairs either side still lacks to reach the minimum."""
        return max(MIN_VALID - self.judged, MIN_VALID - self.baseline_judged, 0)

    @property
    def enough(self) -> bool:
        return self.judged >= MIN_VALID and self.baseline_judged >= MIN_VALID

    @property
    def lower(self) -> bool:
        """The way's guess rate is below DeepSeek's and the one-sided test says so (p < 0.1)."""
        test = self.test
        return test is not None and test.p1 < test.p2 and test.p_less < P_THRESHOLD

    @property
    def style_ok(self) -> bool:
        return self.style is not None and self.style.passed

    @property
    def verdict(self) -> Verdict:
        if not self.enough or self.style_error is not None:
            return "insufficient"
        return "passed" if self.lower and self.style_ok else "failed"


def evaluation_runs(store: EvalStore, model_id: str) -> list[RunView]:
    """The blind runs drawn for ``model_id``, newest first (a cancelled run counts for nothing)."""
    return [
        run
        for run in store.list_runs("blind", limit=200)
        if run.params.get(MODEL_PARAM) == model_id and run.status != "cancelled"
    ]


def latest_evaluated_model(store: EvalStore) -> str | None:
    """The model of the most recent blind run that was drawn for a model."""
    for run in store.list_runs("blind", limit=200):
        model = run.params.get(MODEL_PARAM)
        if model and run.status != "cancelled":
            return str(model)
    return None


def tally(store: EvalStore, runs: Sequence[RunView]) -> dict[str, tuple[int, int]]:
    """``{backend: (valid judgements, correct)}`` over the runs (a skip is not a judgement)."""
    totals = dict.fromkeys(EVAL_BACKENDS, (0, 0))
    for run in runs:
        report = blind_report(store, run)
        for name in EVAL_BACKENDS:
            entry = report.of(name)
            if entry is not None:
                judged, correct = totals[name]
                totals[name] = (judged + entry.judged, correct + entry.correct)
    return totals


def way_result(
    services: Services,
    store: EvalStore,
    runs: Sequence[RunView],
    totals: dict[str, tuple[int, int]],
    backend: str,
    style_of: StyleOf,
) -> WayResult:
    judged, correct = totals[backend]
    baseline_judged, baseline_correct = totals[BASELINE]
    test = two_proportion_test(correct, judged, baseline_correct, baseline_judged)
    style: StyleReport | None = None
    error: str | None = None
    if judged:
        try:
            style = style_of(services, store, runs, backend)
        except StyleError as exc:
            error = str(exc)
    else:
        error = f"no judged pair of the {backend} backend"
    return WayResult(
        backend, judged, correct, baseline_judged, baseline_correct, test, style, error
    )


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def way_checks(way: WayResult) -> list[Check]:
    """The three criteria of one way, as the gate shows them."""
    name = way.backend
    test = way.test
    enough = Check(
        f"{name}：{name} 与 {BASELINE} 各有效判断 ≥ {MIN_VALID} 对",
        way.enough,
        f"{name} {way.judged} 对，{BASELINE} {way.baseline_judged} 对（跳过不计）"
        + (f"；还需 {way.missing} 对" if not way.enough else ""),
        float(min(way.judged, way.baseline_judged)),
        float(MIN_VALID),
        way.judged,
    )
    if test is None:
        lower = Check(
            f"{name}：猜对率低于 {BASELINE}（单侧检验 p < {P_THRESHOLD:g}）",
            False,
            "没有可比较的判断",
        )
    else:
        lower = Check(
            f"{name}：猜对率低于 {BASELINE}（单侧检验 p < {P_THRESHOLD:g}）",
            way.enough and way.lower,
            f"{name} 猜对 {way.correct}/{way.judged} = {_percent(way.rate)}，"
            f"{BASELINE} 猜对 {way.baseline_correct}/{way.baseline_judged} = "
            f"{_percent(way.baseline_rate)}；p = {test.p_less:.3f}",
            test.p_less,
            P_THRESHOLD,
            way.judged,
        )
    report = way.style
    if report is None or way.style_error is not None:
        detail = way.style_error or "没有风格指标"
        style = Check(f"{name}：留出集风格指标每项偏差在 ±30% 内", False, detail)
    else:
        failing = [r.label for r in report.results if r.counts and r.status != "pass"]
        style = Check(
            f"{name}：留出集风格指标每项偏差在 ±30% 内",
            way.style_ok,
            "六项都在范围内" if way.style_ok else "不在范围内的指标：" + "、".join(failing),
            None,
            0.30,
            report.messages,
        )
    return [enough, lower, style]


def _way_values(way: WayResult) -> dict[str, Any]:
    worst = way.style.worst[0] if way.style is not None and way.style.worst else None
    return {
        "judged": way.judged,
        "correct": way.correct,
        "rate": way.rate,
        "missing": way.missing,
        "p_less": way.test.p_less if way.test is not None else None,
        "z": way.test.z if way.test is not None else None,
        "lower": way.lower,
        "style_passed": way.style_ok,
        "style_error": way.style_error,
        "worst_metric": worst.key if worst is not None else None,
        "worst_deviation": worst.deviation if worst is not None else None,
        "verdict": way.verdict,
    }


def winner_of(ways: Sequence[WayResult]) -> WayResult | None:
    """The passing way with the lower guess rate (a tie: the order of :data:`WAYS`)."""
    passing = [way for way in ways if way.verdict == "passed" and way.rate is not None]
    if not passing:
        return None
    return min(
        passing,
        key=lambda way: (way.rate if way.rate is not None else 1.0, WAYS.index(way.backend)),
    )


def judge_model(ctx: GateContext, model_id: str, *, style_of: StyleOf | None = None) -> GateVerdict:
    """M5 for one model, from every evaluation drawn for it (see the module description)."""
    store = ctx.store
    runs = evaluation_runs(store, model_id)
    if not runs:
        check = Check(
            "对这个模型的评估",
            False,
            f"还没有评估过这个模型：先运行 twin model evaluate {model_id}",
        )
        return GateVerdict(
            "M5", "insufficient", (check,), check.detail, (), {MODEL_PARAM: model_id}
        )
    totals = tally(store, runs)
    measure = style_of or style_of_runs
    ways = [way_result(ctx.services, store, runs, totals, backend, measure) for backend in WAYS]
    checks = tuple(check for way in ways for check in way_checks(way))
    winner = winner_of(ways)
    if winner is not None:
        verdict: Verdict = "passed"
        summary = (
            f"M5 通过：{winner.backend} 的猜对率 {_percent(winner.rate)} 低于 "
            f"{BASELINE} 的 {_percent(winner.baseline_rate)}（p = "
            f"{winner.test.p_less if winner.test else float('nan'):.3f}），风格指标都在范围内"
        )
    elif any(way.verdict == "failed" for way in ways):
        verdict = "failed"
        summary = "未通过：" + "；".join(
            f"{way.backend} {c.name.split('：', 1)[1]}（{c.detail}）"
            for way in ways
            for c in way_checks(way)
            if not c.passed
        )
    else:
        verdict = "insufficient"
        missing = max(way.missing for way in ways)
        summary = (
            f"样本不足：还需至少 {missing} 对有效判断；"
            f"再运行 twin model evaluate {model_id} --n {max(missing, 1)}"
            if missing
            else "证据不足：" + "；".join(c.detail for c in checks if not c.passed)
        )
    values: dict[str, Any] = {
        MODEL_PARAM: model_id,
        "baseline": {"judged": totals[BASELINE][0], "correct": totals[BASELINE][1]},
        "ways": {way.backend: _way_values(way) for way in ways},
        "winner": winner.backend if winner is not None else None,
        "p_threshold": P_THRESHOLD,
        "minimum_valid": MIN_VALID,
        "tolerance": 0.30,
    }
    return GateVerdict("M5", verdict, checks, summary, tuple(run.id for run in runs), values)


@gate_judge("M5")
def judge_m5(ctx: GateContext) -> GateVerdict:
    """M5: the model evaluated most recently may become the default backend (R-SRV-005)."""
    model_id = latest_evaluated_model(ctx.store)
    if model_id is None:
        check = Check(
            "对风格模型的评估", False, "还没有评估过风格模型：先运行 twin model evaluate <模型 id>"
        )
        return GateVerdict("M5", "insufficient", (check,), check.detail)
    return judge_model(ctx, model_id)
