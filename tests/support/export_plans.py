"""Plans for the samples the training export insists on (round 17, D-643).

A proactive sample of the training part always carries a plan, because the online proactive path
always hands the style model one; so an export of the synthetic conversation stops at its first
pass and hands the missing inputs back, whatever the plan ratio.  :func:`run_with_plans` is what a
test uses when the plans themselves are not the point: it writes a plan for each missing input
straight into the store (no DeepSeek call, no batch) and runs the same exporter again.
"""

from __future__ import annotations

from collections.abc import Mapping

from twin.services import Services
from twin.training.export import ExportOptions, ExportResult, TrainingSetExporter
from twin.training.plans import PlanInput, PlanStore, StoredPlan
from twin.training.tokenizer import TextTokenizer

TEST_MODEL = "test-planner"
INTENT = "回应对方的话，说说自己的事"


def plan_for(source: PlanInput) -> StoredPlan:
    """A plan for ``source``: it names the first fact of the block and none of her words."""
    return StoredPlan(
        intent=INTENT,
        facts_to_use=source.facts[:1],
        tone="随意",
        bubble_hint="两三条短句",
    )


def fill_plans(services: Services, missing: Mapping[str, PlanInput]) -> int:
    """Register the missing plans and mark each of them done; returns how many were written."""
    store = PlanStore(services.db)
    store.register(missing)
    for sample_id, source in missing.items():
        store.save_done(
            sample_id, plan_for(source), model=TEST_MODEL, template_ref="test", cost_usd=0.0
        )
    return len(missing)


def run_with_plans(services: Services, exporter: TrainingSetExporter) -> ExportResult:
    """``exporter.run()``; when it waits for plans they are written first and it runs again."""
    result = exporter.run()
    if result.state == "waiting_for_plans":
        fill_plans(services, result.missing_plans)
        result = exporter.run()
    return result


def fill_missing_plans(services: Services, tokenizer: TextTokenizer, *, ratio: float = 0.0) -> int:
    """Write the plans an export at this ``ratio`` would ask for; returns how many (a CLI test)."""
    exporter = TrainingSetExporter(services, tokenizer, ExportOptions(plan_ratio=ratio))
    missing, _ = exporter.scan()
    return fill_plans(services, missing)
