"""CLI: ``twin eval`` (R-EVAL-001, R-EVAL-002, R-EVAL-003, R-EVAL-010).

``blind``
    the blind test.  Without ``--resume`` it draws the contexts from the hold-out, stores the
    pairs, prices the generation and queues it as one-time batches, then ends - nothing is spent
    before ``twin jobs approve <batch>``.  With ``--resume <run>`` it goes on from wherever the run
    is: waiting for the approval, generating (``--foreground`` runs the jobs here when the
    application is not running), judging (the screen), reporting.
``style``
    the six core style metrics of what the bot wrote against her profile: the last days of the
    real conversation (``--source live``) or the replies generated for a blind run
    (``--source eval_items --run <id> --backend <name>``); the numbers are kept as a ``style`` run.
``memory``
    the memory test: twenty questions (ten from real records, ten from the conversation with the
    bot), asked in the sandbox, judged by DeepSeek and reviewed by you.  Same steps as ``blind``.
``gate``
    the milestone verdict ``M0`` to ``M5`` (exit code 0 passed, 1 not passed, 2 not reached yet);
    ``--check`` only reads the last stored verdict.
``stability``
    the stability report of the last days (round 12, R-EVAL-006), kept as a ``stability`` run.
``consistency``
    the audit of the bot against itself (round 15, R-EVAL-004): DeepSeek lists the contradictions
    between the life line, what she said about herself and the facts; you decide which are real
    and which corrections of the memory to apply.  ``--review`` goes on with an audit that waits
    for you (the weekly job leaves one), ``--queue`` queues the audit for the off-peak hours.
``cost``
    the cost of a month against the 15 US dollar ceiling (R-EVAL-007), kept as a ``cost`` run.
``report``
    every stored result in ``data/reports/eval-<date>.md`` (R-EVAL-008).
``runs``
    the recent evaluation runs.

All of them are LIGHT: they write short records to the evaluation tables (R-ARCH-006); the long
work is done by the queued jobs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from twin.eval.blind import (
    DEFAULT_N,
    EVAL_GENERATE_JOB,
    BlindPlan,
    EvalError,
    blind_report,
    expand_backends,
    handle_eval_generate,
    plan_blind,
)
from twin.eval.consistency_audit import AuditOutcome, refresh_run, run_audit
from twin.eval.consistency_fixes import FixApplier
from twin.eval.consistency_jobs import queue_consistency_job
from twin.eval.consistency_store import ConsistencyStore
from twin.eval.consistency_ui import ConsistencyReview, ReviewOutcome
from twin.eval.cost_gate import evaluate_cost, render_lines
from twin.eval.gates import GateError, check_gate, run_gate
from twin.eval.memory_test import (
    EVAL_MEMORY_JOB,
    MemoryPlan,
    MemorySession,
    handle_eval_memory,
    plan_memory,
    summarize,
)
from twin.eval.proactive_audit import audit_recent
from twin.eval.proactive_report import print_audit
from twin.eval.report import (
    print_blind_report,
    print_gate,
    print_memory_summary,
    print_style_report,
)
from twin.eval.samples import StickerDescriber
from twin.eval.store import EvalStore, EvalStoreError, RunView
from twin.eval.style_metrics import StyleError, style_of_items, style_of_live
from twin.eval.summary_report import VERDICT_WORDS, write_report
from twin.eval.ui import BlindSession, KeySource, TerminalKeys
from twin.llm.errors import LlmError
from twin.llm.onetime import BatchStatus
from twin.llm.runtime import build_llm_runtime
from twin.memory.api import Memory
from twin.ops.cost import CostReportError, parse_month
from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.schedule.proactive.store import ProactiveLogStore, RatingStore
from twin.schedule.service import time_service_for
from twin.services import Services, get_cli_context
from twin.stickers.catalog import StickerCatalog

eval_app = typer.Typer(
    help="Evaluation: blind test, style metrics, memory test and milestone gates.",
    no_args_is_help=True,
)

RESUME_OPTION = typer.Option("--resume", help="Continue the run with this id (or its start)")
FOREGROUND_OPTION = typer.Option(
    "--foreground", help="Run the approved generation jobs here when the application is stopped"
)
SEED_OPTION = typer.Option("--seed", help="Seed of the draw (default: a new one, kept in the run)")


@dataclass
class Interaction:
    """Where the screens write and where the keys come from (replaced by the tests)."""

    console: Callable[[], Console]
    keys: Callable[[], KeySource]


def _default_console() -> Console:
    return Console(highlight=False, soft_wrap=True)


_interaction = Interaction(_default_console, TerminalKeys)


def interaction() -> Interaction:
    """The console and the keys in use (the tests replace them with `use_interaction`)."""
    return _interaction


@contextmanager
def use_interaction(
    console: Console | None = None, keys: KeySource | None = None
) -> Iterator[Interaction]:
    """Use this console and these keys for the commands run inside the block."""
    global _interaction
    previous = _interaction
    _interaction = Interaction(
        (lambda: console) if console is not None else previous.console,
        (lambda: keys) if keys is not None else previous.keys,
    )
    try:
        yield _interaction
    finally:
        _interaction = previous


def _services() -> Services:
    return get_cli_context().services()


def _store(services: Services) -> EvalStore:
    return EvalStore(services.db, services.clock)


def _run_of(store: EvalStore, run_id: str, kind: str) -> RunView:
    try:
        run = store.get_run(run_id)
    except EvalStoreError as exc:
        raise CliError(str(exc), ExitCode.FAILURE) from exc
    if run.kind != kind:
        raise CliError(f"run {run.id} is a {run.kind} run, not a {kind} run", ExitCode.FAILURE)
    return run


def _batch_states(services: Services, run: RunView) -> list[BatchStatus]:
    batches = build_llm_runtime(services).batches
    return [batches.status(batch_id) for batch_id in run.batch_ids]


def _print_batches(console: Console, services: Services, run: RunView) -> None:
    for state in _batch_states(services, run):
        if state.paused:
            word = "已暂停（费用超出估算），再次 approve 才会继续"
        elif state.approved:
            word = "已批准"
        else:
            word = f"等待批准：twin jobs approve {state.batch_id}"
        console.print(
            Text(
                f"  批次 {state.batch_id}：估算 ${state.estimated_usd:.4f}，"
                f"已花 ${state.spent_usd:.4f}，{word}"
            )
        )


def _generate_in_foreground(
    services: Services, job_type: str, handler: Callable[..., object]
) -> None:
    registry = HandlerRegistry()
    registry.register(job_type, handler)  # type: ignore[arg-type]
    summary = asyncio.run(run_jobs_until_idle(services, registry))
    if summary.failed or summary.retried:
        typer.echo(f"some jobs did not finish: {'; '.join(summary.failures) or 'see the log'}")


# ------------------------------------------------------------------------ blind


def describe_plan(
    console: Console,
    services: Services,
    plan: BlindPlan,
    *,
    resume_command: str = "twin eval blind",
) -> None:
    """What was drawn, what it will cost and how to go on (a model evaluation says its own)."""
    run = plan.run
    console.print(
        Text(
            f"盲测 {run.id}：{plan.samples} 个上下文 × {len(run.backends)} 个后端 = {plan.pairs} 对"
        )
    )
    if plan.short:
        console.print(
            Text(
                f"注意：只有 {plan.samples} 个可用的上下文（要求 {plan.requested}）；"
                f"留出集里排除了 {dict(plan.excluded)}",
                style="yellow",
            )
        )
    if plan.strata:
        console.print(Text("分层：" + "、".join(f"{k} {v}" for k, v in plan.strata.items())))
    console.print(Text(f"生成费用估算 ${plan.estimated_usd:.4f}（一次性批任务，按峰值价估算）"))
    _print_batches(console, services, run)
    console.print(
        Text(
            f"批准并跑完后：{resume_command} --resume {run.id}"
            "（--foreground 可在这里执行已批准的任务）"
        )
    )


def _finish_blind(store: EvalStore, run: RunView) -> RunView:
    counts = store.counts(run.id)
    open_items = counts["pending"] + counts["generated"]
    if open_items == 0 and run.status != "done":
        return store.update_run(run.id, status="done")
    if open_items and run.status == "planned":
        return store.update_run(run.id, status="running")
    return run


@eval_app.command("blind")
@command(CommandKind.LIGHT)
def eval_blind(
    backend: Annotated[
        str, typer.Option("--backend", help="deepseek, style, hybrid or all")
    ] = "deepseek",
    n: Annotated[int, typer.Option("--n", min=1, help="Contexts to draw")] = DEFAULT_N,
    resume: Annotated[str | None, RESUME_OPTION] = None,
    seed: Annotated[int | None, SEED_OPTION] = None,
    foreground: Annotated[bool, FOREGROUND_OPTION] = False,
) -> None:
    """Blind test: pick which of two replies is hers (R-EVAL-001)."""
    services = _services()
    console = _interaction.console()
    store = _store(services)
    if resume is None:
        try:
            plan = asyncio.run(plan_blind(services, expand_backends(backend), n, seed=seed))
        except EvalError as exc:
            raise CliError(str(exc), ExitCode.FAILURE) from exc
        describe_plan(console, services, plan)
        return
    run = _run_of(store, resume, "blind")
    if run.params.get("model_id"):
        raise CliError(
            f"run {run.id} was drawn for the model {run.params['model_id']}: continue it with "
            f"`twin model evaluate {run.params['model_id']} --resume {run.id}`",
            ExitCode.FAILURE,
        )
    continue_blind(
        services, console, store, run, foreground=foreground, generate=_generate_blind_pairs
    )


async def run_blind_jobs(services: Services) -> None:
    """Run the approved generation jobs of a blind test in this process, until none is left."""
    registry = HandlerRegistry()
    registry.register(EVAL_GENERATE_JOB, handle_eval_generate)
    summary = await run_jobs_until_idle(services, registry)
    if summary.failed or summary.retried:
        typer.echo(f"some jobs did not finish: {'; '.join(summary.failures) or 'see the log'}")


def _generate_blind_pairs(services: Services) -> None:
    """Run the approved generation jobs of a blind test here (the application is stopped)."""
    asyncio.run(run_blind_jobs(services))


def continue_blind(
    services: Services,
    console: Console,
    store: EvalStore,
    run: RunView,
    *,
    foreground: bool,
    generate: Callable[[Services], None],
    resume_command: str = "twin eval blind",
) -> None:
    """Go on with a blind run from wherever it is: generate (``generate`` runs the jobs here),
    judge on the screen, report.  ``twin model evaluate --resume`` uses the same steps with
    a ``generate`` that starts the model's server first."""
    pending = store.counts(run.id)["pending"]
    if pending and foreground:
        if app_is_running(services):
            typer.echo("the application is running and will execute the approved jobs")
        else:
            generate(services)
            pending = store.counts(run.id)["pending"]
    if pending:
        console.print(Text(f"还有 {pending} 对没有生成（任务排队、等待批准或正在执行）："))
        _print_batches(console, services, run)
        console.print(
            Text("批准后用 twin jobs run --until-idle 或加 --foreground 执行，再回到这里。")
        )
        return
    run = _finish_blind(store, run)
    describe = StickerDescriber(StickerCatalog(services))
    session = BlindSession(store, run, describe, console, _interaction.keys())
    if session.pending():
        outcome = session.run()
        console.print(Text(f"本次判了 {outcome.judged} 对，跳过 {outcome.skipped} 对。"))
        run = _finish_blind(store, store.get_run(run.id))
    print_blind_report(console, blind_report(store, run))
    if run.status != "done":
        console.print(Text(f"没做完：{resume_command} --resume {run.id}"))


# ------------------------------------------------------------------------ style


@eval_app.command("style")
@command(CommandKind.LIGHT)
def eval_style(
    source: Annotated[str, typer.Option("--source", help="live or eval_items")] = "live",
    days: Annotated[int, typer.Option("--days", min=1, help="Days of the live conversation")] = 7,
    run: Annotated[str | None, typer.Option("--run", help="Blind run (eval_items)")] = None,
    backend: Annotated[str | None, typer.Option("--backend", help="Backend (eval_items)")] = None,
) -> None:
    """Style metrics of the bot against her profile, within +-30 % (R-EVAL-002)."""
    services = _services()
    console = _interaction.console()
    try:
        if source == "live":
            report = style_of_live(services, days)
        elif source == "eval_items":
            if run is None or backend is None:
                raise CliError(
                    "--source eval_items needs --run <id> and --backend <name>", ExitCode.USAGE
                )
            store = _store(services)
            report = style_of_items(services, store, _run_of(store, run, "blind"), backend)
        else:
            raise CliError("--source is live or eval_items", ExitCode.USAGE)
    except StyleError as exc:
        raise CliError(str(exc), ExitCode.FAILURE) from exc
    print_style_report(console, report)
    kept = _store(services).create_run(
        "style",
        mode="live" if report.source == "live" else "holdout",
        backends=[report.backend] if report.backend else [],
        status="done",
        verdict="passed" if report.passed else "failed",
        params={
            "source": report.source,
            "days": report.days,
            "blind_run": report.run_id,
            "backend": report.backend,
        },
        summary=report.to_json(),
    )
    console.print(Text(f"评估记录：{kept.id}"))
    if not report.passed:
        raise typer.Exit(1)


# ----------------------------------------------------------------------- memory


def _describe_memory_plan(console: Console, services: Services, plan: MemoryPlan) -> None:
    run = plan.run
    console.print(Text(f"记忆测试 {run.id}：20 题（真实记录 10 + 机器人对话 10）"))
    console.print(Text(f"生成与判分费用估算 ${plan.estimated_usd:.4f}（一次性批任务）"))
    _print_batches(console, services, run)
    console.print(
        Text(f"批准并跑完后：twin eval memory --resume {run.id}（--foreground 可在这里执行）")
    )


@eval_app.command("memory")
@command(CommandKind.LIGHT)
def eval_memory(
    resume: Annotated[str | None, RESUME_OPTION] = None,
    seed: Annotated[int | None, SEED_OPTION] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", help="Backend that answers (default: the active one)")
    ] = None,
    foreground: Annotated[bool, FOREGROUND_OPTION] = False,
) -> None:
    """Memory test: twenty questions from the fact store, judged and reviewed (R-EVAL-003)."""
    services = _services()
    console = _interaction.console()
    store = _store(services)
    if resume is None:
        try:
            plan = asyncio.run(plan_memory(services, seed=seed, backend=backend))
        except EvalError as exc:
            raise CliError(str(exc), ExitCode.FAILURE) from exc
        if plan.insufficient:
            print_memory_summary(console, summarize([]), list(plan.run.summary.get("reasons", [])))
            raise typer.Exit(1)
        _describe_memory_plan(console, services, plan)
        return
    run = _run_of(store, resume, "memory")
    pending = store.counts(run.id)["pending"]
    if pending and foreground and not app_is_running(services):
        _generate_in_foreground(services, EVAL_MEMORY_JOB, handle_eval_memory)
        pending = store.counts(run.id)["pending"]
    if pending:
        console.print(Text(f"还有 {pending} 题没有出完（任务排队、等待批准或正在执行）："))
        _print_batches(console, services, run)
        return
    outcome = MemorySession(store, run, console, _interaction.keys()).run()
    run = store.get_run(run.id)
    summary = summarize(store.items(run.id, with_payload=False))
    print_memory_summary(console, summary)
    if outcome.quit:
        console.print(Text(f"没复核完：twin eval memory --resume {run.id}"))
        return
    if summary.verdict != "passed":
        raise typer.Exit(1)


# ------------------------------------------------------------------------- gate


@eval_app.command("gate")
@command(CommandKind.LIGHT)
def eval_gate(
    milestone: Annotated[str, typer.Argument(help="M0, M1, M2, M3, M4 or M5")],
    check: Annotated[
        bool, typer.Option("--check", help="Only read the last stored verdict")
    ] = False,
) -> None:
    """Judge a milestone; exit code 0 passed, 1 not passed, 2 not reached yet (R-EVAL-010)."""
    services = _services()
    console = _interaction.console()
    try:
        outcome = check_gate(services, milestone) if check else run_gate(services, milestone)
    except GateError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    print_gate(console, outcome)
    raise typer.Exit(outcome.exit_code)


# --------------------------------------------------------------------- proactive


@eval_app.command("proactive")
@command(CommandKind.LIGHT)
def eval_proactive(
    days: Annotated[
        int, typer.Option("--days", min=1, max=60, help="How many completed local days")
    ] = 7,
    today: Annotated[
        bool, typer.Option("--today", help="Include today so far (a day that is not over)")
    ] = False,
) -> None:
    """Audit the proactive messages of the last days (R-EVAL-005); exit 1 if not compliant."""
    services = _services()
    console = _interaction.console()
    audit = audit_recent(
        ProactiveLogStore(services.db, services.clock),
        RatingStore(services.db, services.clock),
        services.settings.proactive,
        time_service_for(services),
        services.clock,
        days=days,
        include_today=today,
    )
    good = audit.compliant and audit.complete
    run = _store(services).create_run(
        "proactive_audit",
        status="done",
        verdict="passed" if good else ("insufficient" if not audit.complete else "failed"),
        params={"days": days, "first_day": str(audit.first_day), "last_day": str(audit.last_day)},
        summary=audit.to_json(),
    )
    print_audit(console, audit)
    console.print(Text(f"审计记录：{run.id}"))
    if not good:
        raise typer.Exit(1)


# ------------------------------------------------------------------- consistency


def _describe_audit(console: Console, outcome: AuditOutcome) -> None:
    run = outcome.run
    evidence = run.summary.get("evidence") or {}
    console.print(
        Text(
            f"一致性审计 {run.id}：回看 {run.params.get('days')} 天；生活安排 "
            f"{evidence.get('lifeline', 0)} 条、她说过的话 {evidence.get('replies', 0)} 段、"
            f"相关事实 {evidence.get('facts', 0)} 条"
        )
    )
    if not outcome.called:
        console.print(
            Text("这段时间既没有生活安排也没有她说过的话：没有可审阅的内容，没有调用 DeepSeek。")
        )
        return
    dropped = sum(outcome.dropped.values())
    console.print(
        Text(
            f"DeepSeek 报告了 {outcome.findings + outcome.inherited + dropped} 条；"
            f"等你决定 {outcome.findings} 条，沿用以前的决定 {outcome.inherited} 条，"
            f"没通过检查而丢弃 {dropped} 条；花费 ${outcome.cost_usd:.4f}"
        )
    )


def _review_outcome(console: Console, outcome: ReviewOutcome) -> None:
    console.print(
        Text(
            f"本次：明显矛盾 {outcome.obvious}，不明显的矛盾 {outcome.minor}，不是矛盾 "
            f"{outcome.rejected}，先跳过 {outcome.skipped}；修正：应用 {outcome.applied}，"
            f"不改 {outcome.declined}，已过期 {outcome.stale}"
        )
    )


def _print_audit_result(console: Console, run: RunView) -> None:
    summary = run.summary
    decisions = summary.get("decisions") or {}
    fixes = summary.get("fixes") or {}
    console.print(
        Text(
            f"确认的明显矛盾 {decisions.get('confirmed_obvious', 0)} 次"
            f"（{run.params.get('days')} 天内最多 {summary.get('allowed_obvious')} 次），"
            f"不明显的 {decisions.get('confirmed_minor', 0)} 次，"
            f"判为不是矛盾 {decisions.get('rejected', 0)} 次，"
            f"还没决定 {decisions.get('undecided', 0)} 次"
        )
    )
    if fixes.get("proposed"):
        console.print(
            Text(f"还有 {fixes['proposed']} 条记忆修正建议没有决定：twin eval consistency --review")
        )
    if run.verdict is None:
        console.print(
            Text(f"一致性：待确认（twin eval consistency --resume {run.id}）", style="yellow")
        )
        return
    console.print(
        Text(
            f"一致性：{VERDICT_WORDS[run.verdict]}。{summary.get('verdict_reason', '')}",
            style="green" if run.verdict == "passed" else "red",
        )
    )


async def _audit_now(services: Services, days: int | None) -> AuditOutcome:
    runtime = build_llm_runtime(services)
    try:
        return await run_audit(services, runtime.client, days=days)
    finally:
        await runtime.client.aclose()


def _run_to_review(store: EvalStore, cstore: ConsistencyStore) -> RunView | None:
    """The newest audit that still waits for a decision on a contradiction or a correction."""
    for run in store.list_runs("consistency", limit=50):
        if run.status in ("cancelled", "failed"):
            continue
        if cstore.findings(run.id, status="proposed") or cstore.run_fixes(
            run.id, status="proposed"
        ):
            return run
    return None


@eval_app.command("consistency")
@command(CommandKind.LIGHT)
def eval_consistency(
    days: Annotated[
        int | None,
        typer.Option(
            "--days",
            min=1,
            max=60,
            help="How many days to look back (default: eval.consistency_days)",
        ),
    ] = None,
    review: Annotated[
        bool,
        typer.Option(
            "--review", help="Go on with the newest audit that waits for you; DeepSeek is not asked"
        ),
    ] = False,
    resume: Annotated[str | None, RESUME_OPTION] = None,
    queue: Annotated[
        bool,
        typer.Option(
            "--queue", help="Queue the audit for the off-peak hours instead of running it now"
        ),
    ] = False,
) -> None:
    """Consistency audit: DeepSeek lists contradictions, you decide which are real (R-EVAL-004)."""
    services = _services()
    console = _interaction.console()
    store = _store(services)
    cstore = ConsistencyStore(services.db, services.clock)
    if queue:
        job_id = queue_consistency_job(services, days=days)
        if job_id is None:
            console.print(Text("已经有一次审计在排队或正在运行。"))
        elif app_is_running(services):
            console.print(Text(f"已排队（任务 {job_id}），应用会在非高峰时段执行。"))
        else:
            console.print(
                Text(f"已排队（任务 {job_id}）：twin jobs run --until-idle 可在这里执行。")
            )
        return
    if resume is not None:
        run = _run_of(store, resume, "consistency")
    elif review:
        found = _run_to_review(store, cstore)
        if found is None:
            console.print(Text("没有等你决定的审计。"))
            return
        run = found
    else:
        try:
            outcome = asyncio.run(_audit_now(services, days))
        except LlmError as exc:
            raise CliError(f"the audit could not be made: {exc}", ExitCode.FAILURE) from exc
        _describe_audit(console, outcome)
        run = outcome.run
    memory = Memory(services)
    session = ConsistencyReview(
        cstore, FixApplier(memory, cstore), memory, run, console, _interaction.keys()
    )
    undecided, proposed = session.waiting()
    left = False
    if undecided or proposed:
        result = session.run()
        _review_outcome(console, result)
        left = result.quit
    run = refresh_run(store, cstore, run.id)
    _print_audit_result(console, run)
    if left:
        console.print(Text(f"没做完：twin eval consistency --resume {run.id}"))
        return
    if run.verdict != "passed":
        raise typer.Exit(1)


# -------------------------------------------------------------------------- cost


@eval_app.command("cost")
@command(CommandKind.LIGHT)
def eval_cost(
    month: Annotated[
        str | None, typer.Option("--month", help="Month as YYYY-MM (default: this month)")
    ] = None,
) -> None:
    """The month's cost against 15 US dollars; one-time batches are shown apart (R-EVAL-007)."""
    services = _services()
    console = _interaction.console()
    time = time_service_for(services)
    try:
        start = parse_month(month) if month else time.local_date().replace(day=1)
    except CostReportError as exc:
        raise CliError(str(exc), ExitCode.USAGE) from exc
    evaluation = evaluate_cost(services, start)
    for line in render_lines(evaluation):
        console.print(Text(line))
    if not evaluation.passed:
        raise typer.Exit(1)


# ------------------------------------------------------------------------ report


@eval_app.command("report")
@command(CommandKind.LIGHT)
def eval_report() -> None:
    """Write every stored evaluation result to data/reports/eval-<date>.md (R-EVAL-008)."""
    services = _services()
    console = _interaction.console()
    written = write_report(services)
    table = Table(title="里程碑")
    for column in ("里程碑", "结论", "判定记录"):
        table.add_column(column)
    for code, item in written.milestones.items():
        table.add_row(Text(code), Text(VERDICT_WORDS[item["status"]]), Text(item["run"] or "—"))
    console.print(table)
    if written.missing:
        console.print(Text(f"尚未评估的项目 {len(written.missing)} 个，列在报告最后一节。"))
    console.print(Text(f"报告已写入 {written.path}"))
    console.print(Text(f"评估记录：{written.run.id}"))


# ------------------------------------------------------------------------- runs


@eval_app.command("stability")
@command(CommandKind.LIGHT)
def eval_stability(
    days: Annotated[float, typer.Option("--days", min=0.01, help="Window in days")] = 7.0,
) -> None:
    """The stability report of the last days, from the health snapshots (R-EVAL-006)."""
    from twin.ops.stability import render_lines, run_stability

    services = _services()
    report, run = run_stability(services, days)
    for line in render_lines(report):
        typer.echo(line)
    typer.echo(f"saved as evaluation run {run.id}; `twin eval gate M4` reads it")


@eval_app.command("runs")
@command(CommandKind.READ)
def eval_runs(
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help="blind, memory, style, gate, stability, proactive_audit, consistency, cost "
            "or report",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, help="How many to list")] = 20,
) -> None:
    """List the recent evaluation runs."""
    services = _services()
    console = _interaction.console()
    table = Table(title="评估运行")
    for column in ("id", "类型", "状态", "后端/门槛", "结论", "创建时间"):
        table.add_column(column)
    for run in _store(services).list_runs(kind, limit=limit):
        table.add_row(
            Text(run.id),
            Text(run.kind),
            Text(run.status),
            Text(run.milestone or ",".join(run.backends)),
            Text(run.verdict or "—"),
            Text(run.created_at.strftime("%Y-%m-%d %H:%M")),
        )
    console.print(table)
