"""Evaluation data for a model, made by hand, for the tests of the release gate (round 14).

``evaluation_run`` writes a blind run the way ``twin model evaluate`` leaves it: one run drawn for
a model (``params.model_id``), the same contexts answered by the DeepSeek backend and by the two
ways of using the model, every pair judged.  ``style_report`` is the style metrics of one way as a
:class:`~twin.eval.style_metrics.StyleReport` with the number of metrics outside the +-30 % chosen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from twin.eval.blind import MIN_VALID
from twin.eval.store import EvalStore, NewItem, RunView
from twin.eval.style_metrics import (
    STYLE_METRICS,
    MetricReading,
    StyleReport,
    judge_metric,
)
from twin.services import Services

WHEN = datetime(2026, 9, 1, 12, tzinfo=UTC)
BACKENDS = ("deepseek", "style", "hybrid")

__all__ = ["BACKENDS", "MIN_VALID", "evaluation_run", "style_report"]


def evaluation_run(
    services: Services,
    model_id: str,
    counts: Mapping[str, tuple[int, int]],
    *,
    skipped: int = 0,
    status: str = "done",
    tag: str = "a",
) -> RunView:
    """A blind run of ``model_id``; ``counts[backend] = (valid judgements, correct ones)``.

    Every pair is a generated reply that was judged; ``skipped`` more pairs per backend were
    skipped (they count for nothing).  ``tag`` keeps the contexts of two runs apart.
    """
    store = EvalStore(services.db, services.clock)
    run = store.create_run(
        "blind",
        mode="holdout",
        backends=list(counts),
        params={"model_id": model_id, "n": max(v for v, _ in counts.values())},
        status=status,
    )
    for backend, (valid, _correct) in counts.items():
        store.add_items(
            run.id,
            [
                NewItem(
                    f"{tag}{n}",
                    backend,
                    WHEN,
                    {"real": {"lines": [], "quote": None}},
                    None,
                    "night",
                    "short",
                    n % 2 == 0,
                )
                for n in range(valid + skipped)
            ],
            start=None,
        )
    for item in store.items(run.id):
        valid, correct = counts[item.backend]
        index = int(item.sample_key[len(tag) :])
        store.save_generated(item.id, {"bot": {"lines": [], "quote": None}}, cost_usd=0.0)
        if index < correct:
            store.judge(item.id, "correct", score=1.0)
        elif index < valid:
            store.judge(item.id, "wrong", score=0.0)
        else:
            store.judge(item.id, None)
    return store.get_run(run.id)


def style_report(backend: str, *, off: Sequence[str] = ()) -> StyleReport:
    """The six metrics of a way: all within the tolerance, except the keys named in ``off``."""
    results = []
    for spec in STYLE_METRICS:
        measured = 2.0 if spec.key in off else 1.0
        results.append(judge_metric(spec, MetricReading(1.0, 100), MetricReading(measured, 100)))
    return StyleReport(
        "eval_items", "pre_holdout", results, backend=backend, run_id="x", messages=100
    )
