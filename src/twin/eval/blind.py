"""The blind test: planning, generating and counting (R-EVAL-001, R-LLM-014, R-SAFE-006).

A blind test has three steps, each of which can be stopped and continued:

1. **Plan** (:func:`plan_blind`).  The contexts are drawn from the hold-out window
   (:mod:`twin.eval.samples`), one (context, backend) pair becomes one ``eval_items`` row with her
   real reply, the conversation before it and a coin that decides which side the bot's reply is
   shown on.  The generation is priced from the prompts it will really send and queued as
   one-time batches that wait for ``twin jobs approve <batch>`` (R-LLM-014); nothing is spent
   before that.
2. **Generate** (:func:`generate_items`, the job ``eval_generate``).  For each pair the
   sandbox makes the bot's reply for the moment of the sample (:mod:`twin.eval.sandbox`) and the
   row is saved at once, so a stopped job goes on from the next pair.  A pair whose reply cannot
   be shown (the pipeline gave up, the bot chose silence, another backend than the asked one
   wrote it) is ``failed`` and never judged - the report counts those per backend.
3. **Judge** (:mod:`twin.eval.ui`).  The user picks the real reply; every decision is stored at
   once.

:func:`blind_report` counts what was judged: the guess rate (the point estimate the gates
judge), its Wilson interval, the number of valid judgements (a skip is not one), the rate per
period and per segment length, and the two-proportion test between the backends.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from twin.config.runtime import THINKING_CHAT
from twin.engine.style_models import StyleModels
from twin.engine.style_runtime import StyleRuntime
from twin.engine.types import InboundItem
from twin.eval.samples import LENGTH_BINS, PERIODS, DrawResult, SampleDrawer
from twin.eval.sandbox import (
    BACKENDS,
    STYLE_BACKENDS,
    EvalSandbox,
    SandboxKit,
    SandboxMode,
    SandboxRequest,
    backend_status,
    build_sandbox,
    stable_seed,
)
from twin.eval.stats import ProportionTest, Rate, group_rates, two_proportion_test
from twin.eval.store import EvalStore, ItemView, NewItem, RunView
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.onetime import BatchItem
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.memory.recent import Turn
from twin.ops.jobs import BatchTooLargeError, JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.profile.api import holdout_cutoff, load_activity_model, load_profile
from twin.profile.holdout import HoldoutError
from twin.profile.persona.api import render_compact, render_full

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.eval.blind")

EVAL_GENERATE_JOB = "eval_generate"
JOB_PRIORITY = 120
JOB_SIZE = 10  # pairs per job
REPLY_COMPLETION_TOKENS = 600  # the DeepSeek backend's cap on a reply (an upper bound)
PLAN_COMPLETION_TOKENS = 400  # the planner's cap of the hybrid backend
DEFAULT_N = 50
MIN_VALID = 50  # R-EVAL-001: a gate counts only with this many valid judgements

FAIL_FELL_BACK = "fell_back"
FAIL_NO_REPLY = "no_reply"
FAIL_EMPTY = "empty"
FAIL_NOT_DEPLOYED = "not_deployed"


class EvalError(RuntimeError):
    """The evaluation cannot go on; the message says what to do first."""


class EvaluationError(EvalError):
    """The model cannot be evaluated; the message says what to do first (R-SRV-005)."""


class ModelStyleSource(Protocol):
    """Whoever owns the server of the model under evaluation (``twin.serving.evaluation``)."""

    async def style_runtime(
        self, llm: LlmRuntime, model_id: str
    ) -> tuple[StyleRuntime, StyleModels]: ...


_model_styles: ModelStyleSource | None = None


def install_model_styles(source: ModelStyleSource | None) -> None:
    """Make ``source`` the owner the pairs of a model evaluation ask in this process.

    The running application and ``twin model evaluate --foreground`` install the pool of servers
    they started; this module only knows the protocol, so it does not depend on the serving
    package (which depends on this one).
    """
    global _model_styles
    _model_styles = source


def installed_model_styles() -> ModelStyleSource | None:
    return _model_styles


async def evaluation_style(
    services: Services, llm: LlmRuntime, model_id: str
) -> tuple[StyleRuntime, StyleModels]:
    """The style backends for the pairs of a run drawn for ``model_id`` (used by the job)."""
    source = _model_styles
    if source is None:
        raise EvaluationError(
            "the pairs of a model evaluation need the model's server, which only "
            f"`twin model evaluate {model_id} --resume <run> --foreground` or a running "
            "application starts"
        )
    return await source.style_runtime(llm, model_id)


# ------------------------------------------------------------------- readiness


def readiness_problems(services: Services, backends: Sequence[str]) -> list[str]:
    """What is missing before a hold-out test can run (empty when it can)."""
    problems: list[str] = []
    try:
        holdout_cutoff(services)
    except HoldoutError as exc:
        problems.append(f"the hold-out cannot be determined yet: {exc}")
    if load_profile(services, "pre_holdout") is None:
        problems.append("there is no pre-holdout profile; run `twin profile rebuild` first")
    if load_activity_model(services, "pre_holdout") is None:
        problems.append("there is no pre-holdout routine; run `twin profile rebuild` first")
    if any(b != "style" for b in backends) and render_full(services, "pre_holdout") is None:
        problems.append("there is no pre-holdout persona card; run `twin persona generate` first")
    if (
        any(b in STYLE_BACKENDS for b in backends)
        and render_compact(services, "pre_holdout") is None
    ):
        problems.append("there is no compact pre-holdout persona card for the style model")
    return problems


def check_backends(
    services: Services, backends: Sequence[str], models: StyleModels | None = None
) -> None:
    """Refuse a backend that is unknown or not deployed, naming it (R-EVAL-001)."""
    messages = [
        status.message
        for status in (backend_status(services, name, models) for name in backends)
        if not status.available
    ]
    if messages:
        raise EvalError("; ".join(messages))


def expand_backends(choice: str) -> tuple[str, ...]:
    """``deepseek``, ``style``, ``hybrid`` or ``all`` as a list of backend names."""
    if choice == "all":
        return BACKENDS
    if choice not in BACKENDS:
        raise EvalError(f"unknown backend {choice!r}: use deepseek, style, hybrid or all")
    return (choice,)


# ------------------------------------------------------------------------ plan


@dataclass
class BlindPlan:
    """The outcome of :func:`plan_blind`."""

    run: RunView
    samples: int
    pairs: int
    estimated_usd: float
    batch_ids: list[str] = field(default_factory=list)
    requested: int = 0
    excluded: Counter[str] = field(default_factory=Counter)
    strata: dict[str, int] = field(default_factory=dict)

    @property
    def short(self) -> bool:
        """Fewer contexts than asked for: the hold-out has no more usable ones."""
        return self.samples < self.requested


def pack(estimates: Sequence[float], limit: float) -> list[list[int]]:
    """Group the indices of jobs into batches that each estimate to at most ``limit``."""
    groups: list[list[int]] = [[]]
    running = 0.0
    for index, estimate in enumerate(estimates):
        if estimate > limit:
            raise BatchTooLargeError("a single evaluation job", estimate, limit)
        if groups[-1] and running + estimate > limit:
            groups.append([])
            running = 0.0
        groups[-1].append(index)
        running += estimate
    return [g for g in groups if g]


def new_seed() -> int:
    """A fresh seed for a draw (kept in the run so that the draw can be reproduced)."""
    return random.SystemRandom().randrange(2**31)


def assign_sides(count: int, seed: int) -> list[bool]:
    """``left_is_bot`` for ``count`` pairs: independent fair coins."""
    rng = random.Random(seed ^ 0x51DE)  # noqa: S311 - a fair coin, not security
    return [rng.random() < 0.5 for _ in range(count)]


def request_from(
    payload: Mapping[str, Any], at: datetime, backend: str, thinking: str
) -> SandboxRequest:
    """The round of a payload (a drawn sample) as the sandbox is asked to answer it."""
    inbound = tuple(
        InboundItem(
            id=str(entry["id"]),
            at=datetime.fromisoformat(str(entry["at"])),
            kind=str(entry["kind"]),
            text=str(entry["text"]),
        )
        for entry in payload["inbound"]
    )
    history = tuple(
        Turn(
            role="user" if entry["role"] == "user" else "bot",
            text=str(entry["text"]),
            at=datetime.fromisoformat(str(entry["at"])),
            last_at=datetime.fromisoformat(str(entry["last_at"])),
            message_ids=tuple(str(i) for i in entry["ids"]),
        )
        for entry in payload["history"]
    )
    return SandboxRequest(
        inbound=inbound,
        history=history,
        at=at,
        backend=backend,
        thinking_mode=thinking,  # type: ignore[arg-type]
        seed=int(payload.get("seed", stable_seed(str(payload.get("t", ""))))),
    )


def request_of(item: ItemView, thinking: str) -> SandboxRequest:
    """The round of an item as the sandbox is asked to answer it."""
    return request_from(item.payload, item.at, item.backend, thinking)


async def plan_blind(
    services: Services,
    backends: Sequence[str],
    n: int,
    *,
    seed: int | None = None,
    kit: SandboxKit | None = None,
    models: StyleModels | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> BlindPlan:
    """Draw the contexts, store the pairs and queue the generation as one-time batches.

    ``twin model evaluate`` compares the backends for one model: ``models`` pins that model for
    the readiness check and ``extra_params`` (its ``model_id``) go into the run, so that the
    generation and the release gate know which model the pairs belong to (R-SRV-005).
    """
    check_backends(services, backends, models)
    problems = readiness_problems(services, backends)
    if problems:
        raise EvalError("; ".join(problems))
    chosen_seed = new_seed() if seed is None else seed
    store = EvalStore(services.db, services.clock)
    exclude = store.used_sample_keys("blind")
    drawn: DrawResult = await asyncio.to_thread(
        lambda: SampleDrawer(services).draw(n, seed=chosen_seed, exclude=exclude)
    )
    if not drawn.samples:
        raise EvalError(
            f"the hold-out has no usable context left ({drawn.holdout_blocks} reply blocks; "
            f"left out: {dict(drawn.excluded)})"
        )
    own = kit is None
    sandbox_kit = kit or build_sandbox(services, mode=SandboxMode.HOLDOUT)
    try:
        thinking = str(services.runtime.get(THINKING_CHAT))
        prices = await price_pairs(services, sandbox_kit, drawn, backends, thinking)
        return _store_plan(
            services,
            sandbox_kit,
            store,
            drawn,
            backends,
            n,
            chosen_seed,
            thinking,
            prices,
            extra_params or {},
        )
    finally:
        if own:
            await sandbox_kit.aclose()


async def price_pairs(
    services: Services,
    kit: SandboxKit,
    drawn: DrawResult,
    backends: Sequence[str],
    thinking: str,
) -> dict[tuple[str, str], float]:
    """The estimated cost of every (sample, backend) pair, from the prompt it will send."""
    prices: dict[tuple[str, str], float] = {}
    paid = [b for b in backends if b != "style"]
    model = services.settings.deepseek.chat_model
    batches = kit.llm.batches
    for sample in drawn.samples:
        usd_reply = usd_plan = 0.0
        if paid:
            request = request_from(sample.payload(), sample.at, "deepseek", thinking)
            messages = await kit.sandbox.preview(request)
            reply_item = batches.item_from_messages(
                model, messages, completion_tokens=REPLY_COMPLETION_TOKENS
            )
            plan_item = BatchItem(model, reply_item.prompt_tokens, PLAN_COMPLETION_TOKENS)
            usd_reply = batches.estimate_item(reply_item)[0]
            usd_plan = batches.estimate_item(plan_item)[0]
        for backend in backends:
            prices[(sample.sample_key, backend)] = {
                "deepseek": usd_reply,
                "hybrid": usd_plan,
            }.get(backend, 0.0)
    return prices


def _store_plan(
    services: Services,
    kit: SandboxKit,
    store: EvalStore,
    drawn: DrawResult,
    backends: Sequence[str],
    requested: int,
    seed: int,
    thinking: str,
    prices: dict[tuple[str, str], float],
    extra_params: Mapping[str, Any],
) -> BlindPlan:
    cutoff = holdout_cutoff(services)
    run = store.create_run(
        "blind",
        mode="holdout",
        backends=backends,
        params={
            **extra_params,
            "n": requested,
            "seed": seed,
            "thinking_mode": thinking,
            "holdout_cutoff": cutoff.isoformat(),
            "context_turns": 8,
            "available": drawn.available,
            "holdout_blocks": drawn.holdout_blocks,
            "excluded": dict(drawn.excluded),
            "strata": drawn.strata,
        },
    )
    pairs = [(sample, backend) for sample in drawn.samples for backend in backends]
    sides = assign_sides(len(pairs), seed)
    items = [
        NewItem(
            sample_key=sample.sample_key,
            backend=backend,
            at=sample.at,
            payload={**sample.payload(), "seed": stable_seed(sample.sample_key, backend)},
            period=sample.period,
            length_bin=sample.length_bin,
            left_is_bot=side,
        )
        for (sample, backend), side in zip(pairs, sides, strict=True)
    ]
    store.add_items(run.id, items)
    stored = store.items(run.id, with_payload=False)
    estimate_of = {item.id: prices[(item.sample_key, item.backend)] for item in stored}
    jobs = [stored[i : i + JOB_SIZE] for i in range(0, len(stored), JOB_SIZE)]
    job_usd = [sum(estimate_of[item.id] for item in job) for job in jobs]
    batches = kit.llm.batches
    try:
        groups = pack(job_usd, batches.limit_usd)
        batch_ids: list[str] = []
        stamp = run.id[-8:].lower()
        for number, group in enumerate(groups, start=1):
            batch_id = f"eval-{stamp}-{number}"
            batches.enqueue(
                batch_id,
                EVAL_GENERATE_JOB,
                [
                    {
                        "run_id": run.id,
                        "batch_id": batch_id,
                        "item_ids": [item.id for item in jobs[i]],
                    }
                    for i in group
                ],
                [job_usd[i] for i in group],
                priority=JOB_PRIORITY,
                offpeak_only=False,  # the user is waiting for it: the estimate uses peak prices
            )
            batch_ids.append(batch_id)
    except BatchTooLargeError:
        store.update_run(run.id, status="cancelled")
        raise
    total = sum(job_usd)
    run = store.update_run(run.id, batch_ids=batch_ids, params={"estimated_usd": round(total, 6)})
    return BlindPlan(
        run=run,
        samples=len(drawn.samples),
        pairs=len(items),
        estimated_usd=total,
        batch_ids=batch_ids,
        requested=requested,
        excluded=drawn.excluded,
        strata=drawn.strata,
    )


# -------------------------------------------------------------------- generate


@dataclass
class GenerateSummary:
    generated: int = 0
    failed: int = 0
    skipped: int = 0
    cost_usd: float = 0.0


def draft_facts(
    backend: str,
    fell_back: bool,
    attempts: int,
    actions: Sequence[dict[str, Any]],
    kinds: Sequence[str],
) -> dict[str, Any]:
    """What is kept of how a reply came about: codes and numbers, never text."""
    return {
        "backend": backend,
        "fell_back": fell_back,
        "attempts": attempts,
        "actions": [a.get("step") for a in actions],
        "violations": list(kinds),
    }


async def generate_item(
    sandbox: EvalSandbox, store: EvalStore, item: ItemView, thinking: str
) -> str:
    """Make and save the bot's side of one pair; returns ``generated`` or ``failed``."""
    request = request_of(item, thinking)
    reply = await sandbox.reply(request)
    draft = reply.draft
    facts = draft_facts(
        draft.backend,
        reply.fell_back,
        draft.attempts,
        draft.actions_json(),
        [v.kind for v in draft.violations],
    )
    reason: str | None = None
    if draft.needs_fallback:
        reason = f"fallback:{draft.fallback_reason}"
    elif draft.no_reply:
        reason = FAIL_NO_REPLY
    elif reply.fell_back:
        reason = FAIL_FELL_BACK
    elif reply.candidate.empty:
        reason = FAIL_EMPTY
    if reason is not None:
        store.save_generated(
            item.id,
            {"failure": reason, "draft": facts},
            cost_usd=draft.cost_usd,
            status="failed",
        )
        return "failed"
    store.save_generated(
        item.id,
        {"bot": reply.candidate.to_json(), "draft": facts},
        cost_usd=draft.cost_usd,
    )
    return "generated"


async def generate_items(
    services: Services, run_id: str, item_ids: Sequence[str], batch_id: str | None
) -> GenerateSummary:
    """Generate the pairs of a job (the pairs that are not pending any more are left alone)."""
    store = EvalStore(services.db, services.clock)
    run = store.get_run(run_id)
    thinking = str(run.params.get("thinking_mode", "off"))
    summary = GenerateSummary()
    items = [store.item(item_id) for item_id in item_ids]
    wanted = any(item.status == "pending" and item.backend in STYLE_BACKENDS for item in items)
    model_id = run.params.get("model_id")
    models: StyleModels | None = None
    llm = build_llm_runtime(services)
    style: StyleRuntime | None = None
    if model_id and wanted:
        # a run drawn for a model (twin model evaluate) talks to the server of that model, which
        # is started for the evaluation and is not the active one (R-SRV-005)
        style, models = await evaluation_style(services, llm, str(model_id))
    kit = build_sandbox(services, mode=SandboxMode.HOLDOUT, batch_id=batch_id, llm=llm, style=style)
    try:
        for item in items:
            if item.status != "pending":
                summary.skipped += 1
                continue
            status = backend_status(services, item.backend, models)
            if not status.available:
                store.save_generated(
                    item.id, {"failure": FAIL_NOT_DEPLOYED}, cost_usd=0.0, status="failed"
                )
                summary.failed += 1
                continue
            outcome = await generate_item(kit.sandbox, store, item, thinking)
            after = store.item(item.id)
            summary.cost_usd += after.cost_usd
            if outcome == "generated":
                summary.generated += 1
            else:
                summary.failed += 1
    finally:
        await kit.aclose()
    return summary


@job_handler(EVAL_GENERATE_JOB)
async def handle_eval_generate(ctx: JobContext) -> None:
    """The job of a batch: generate the bot's replies for some pairs of a blind run."""
    services = ctx.services
    if services is None:
        raise RuntimeError("eval_generate needs the services container")
    payload = ctx.job.payload
    batch_id = payload.get("batch_id")
    try:
        summary = await generate_items(
            services,
            str(payload["run_id"]),
            [str(i) for i in payload.get("item_ids", [])],
            str(batch_id) if batch_id else None,
        )
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    log.info(
        "eval_pairs_generated",
        generated=summary.generated,
        failed=summary.failed,
        cost_usd=round(summary.cost_usd, 6),
    )


def queued_jobs(services: Services) -> int:
    """How many ``eval_generate`` jobs wait or run (any batch)."""
    queue = JobQueue(services.db, services.clock)
    return sum(
        len(queue.list_jobs(status=status, job_type=EVAL_GENERATE_JOB, limit=100_000))
        for status in ("pending", "running")
    )


# ---------------------------------------------------------------------- report


@dataclass(frozen=True)
class BackendReport:
    """What was judged for one backend of a run."""

    backend: str
    pairs: int
    generated: int  # pairs whose bot reply exists
    failed: int  # pairs the bot could not answer
    judged: int  # valid judgements (a skip is none)
    skipped: int
    correct: int  # the user picked her real reply
    by_period: dict[str, Rate]
    by_length: dict[str, Rate]

    @property
    def rate(self) -> Rate:
        return Rate(self.correct, self.judged)

    @property
    def waiting(self) -> int:
        """Generated pairs the user has not decided yet."""
        return self.generated - self.judged - self.skipped


@dataclass(frozen=True)
class Comparison:
    """The first backend's guess rate against the second's (``test.p_less``: first is lower)."""

    first: str
    second: str
    test: ProportionTest


@dataclass(frozen=True)
class BlindReport:
    run: RunView
    backends: list[BackendReport]
    comparisons: list[Comparison]

    def of(self, backend: str) -> BackendReport | None:
        return next((b for b in self.backends if b.backend == backend), None)


def blind_report(store: EvalStore, run: RunView) -> BlindReport:
    """Count the judgements of a blind run (see the module description)."""
    items = store.items(run.id, with_payload=False)
    names = list(run.backends) or sorted({item.backend for item in items})
    reports: list[BackendReport] = []
    for name in names:
        mine = [item for item in items if item.backend == name]
        valid = [item for item in mine if item.valid]
        reports.append(
            BackendReport(
                backend=name,
                pairs=len(mine),
                generated=sum(item.status in ("generated", "judged", "skipped") for item in mine),
                failed=sum(item.status == "failed" for item in mine),
                judged=len(valid),
                skipped=sum(item.status == "skipped" for item in mine),
                correct=sum(item.outcome == "correct" for item in valid),
                by_period=group_rates(valid, lambda i: i.period, lambda i: i.outcome == "correct"),
                by_length=group_rates(
                    valid, lambda i: i.length_bin, lambda i: i.outcome == "correct"
                ),
            )
        )
    comparisons: list[Comparison] = []
    for index, first in enumerate(reports):
        for second in reports[index + 1 :]:
            test = two_proportion_test(first.correct, first.judged, second.correct, second.judged)
            if test is not None:
                comparisons.append(Comparison(first.backend, second.backend, test))
    return BlindReport(run, reports, comparisons)


def ordered(rates: dict[str, Rate], order: Sequence[str]) -> list[tuple[str, Rate]]:
    """The groups in their natural order (periods of the day, lengths)."""
    return [(name, rates[name]) for name in order if name in rates]


PERIOD_ORDER = PERIODS
LENGTH_ORDER = LENGTH_BINS
