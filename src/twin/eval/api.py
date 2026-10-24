"""What later rounds import from the evaluation (round 09b).

::

    from twin.eval.api import (
        EvalSandbox, SandboxMode, SandboxRequest, build_sandbox,   # a reply for a past moment
        InMemoryChannel, isolated_writes,                          # the channel, the write fence
        EvalStore, RunView, ItemView, NewItem,                     # eval_runs, eval_items
        plan_blind, blind_report, generate_items,                  # the blind test, any backends
        Candidate, render_candidate,                               # how both sides are shown
        style_of_items, style_of_live,                             # the six style metrics
        gate_judge, GateContext, GateVerdict, Check, run_gate,     # the milestone gates
        two_proportion_test, wilson_interval,                      # the statistics
    )

**A judge for a milestone** (round 10: ``M3``, round 12: ``M4``, round 14: ``M5``)::

    @gate_judge("M3")
    def judge_m3(ctx: GateContext) -> GateVerdict: ...

and the module is appended to ``twin.eval.gates.JUDGE_MODULES``.  The blind-test criteria are
reusable: ``twin.eval.gates.blind_checks(ctx, backend, ceiling)`` counts the latest blind test of a
backend against a ceiling (``M1_MAX_GUESS_RATE`` 70 %, ``M4_MAX_GUESS_RATE`` 60 %) and returns
``(verdict, checks, run ids, values, detail)`` - the M4 judge of round 12 calls it and adds its
own criteria.

**Comparing backends** (round 14, ``twin model evaluate``): ``plan_blind(services, ("deepseek",
"style"), n)`` draws one set of contexts and prices the generation for every backend; after the
batch is approved and run, ``blind_report(store, run)`` gives each backend's guess rate with its
Wilson interval and ``Comparison.test`` (``p_less``: the first backend's rate is lower, the
one-sided p of R-SRV-005).  ``style_of_items(services, store, run, backend)`` gives the style
metrics of one backend on the same contexts.
"""

from __future__ import annotations

from twin.eval.blind import (
    BackendReport,
    BlindPlan,
    BlindReport,
    Comparison,
    EvalError,
    blind_report,
    generate_items,
    plan_blind,
)
from twin.eval.channel import EVAL_USER_ID, InMemoryChannel, SentMessage
from twin.eval.gates import (
    Check,
    GateContext,
    GateOutcome,
    GateRegistry,
    GateVerdict,
    check_gate,
    gate_judge,
    run_gate,
)
from twin.eval.isolation import (
    IsolationViolation,
    Snapshot,
    changes,
    isolated_writes,
    snapshot,
)
from twin.eval.memory_test import MemorySummary, plan_memory, summarize
from twin.eval.render import Candidate, CandidateLine, render_candidate
from twin.eval.sandbox import (
    EvalSandbox,
    SandboxKit,
    SandboxMode,
    SandboxReply,
    SandboxRequest,
    backend_status,
    build_sandbox,
)
from twin.eval.stats import ProportionTest, Rate, two_proportion_test, wilson_interval
from twin.eval.store import EvalStore, ItemView, NewItem, RunView
from twin.eval.style_metrics import StyleReport, style_of_items, style_of_live

__all__ = [
    "EVAL_USER_ID",
    "BackendReport",
    "BlindPlan",
    "BlindReport",
    "Candidate",
    "CandidateLine",
    "Check",
    "Comparison",
    "EvalError",
    "EvalSandbox",
    "EvalStore",
    "GateContext",
    "GateOutcome",
    "GateRegistry",
    "GateVerdict",
    "InMemoryChannel",
    "IsolationViolation",
    "ItemView",
    "MemorySummary",
    "NewItem",
    "ProportionTest",
    "Rate",
    "RunView",
    "SandboxKit",
    "SandboxMode",
    "SandboxReply",
    "SandboxRequest",
    "SentMessage",
    "Snapshot",
    "StyleReport",
    "backend_status",
    "blind_report",
    "build_sandbox",
    "changes",
    "check_gate",
    "gate_judge",
    "generate_items",
    "isolated_writes",
    "plan_blind",
    "plan_memory",
    "render_candidate",
    "run_gate",
    "snapshot",
    "style_of_items",
    "style_of_live",
    "summarize",
    "two_proportion_test",
    "wilson_interval",
]
