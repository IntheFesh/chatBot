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
    (``--source eval_items --run <id> --backend <name>``).
``memory``
    the memory test: twenty questions (ten from real records, ten from the conversation with the
    bot), asked in the sandbox, judged by DeepSeek and reviewed by you.  Same steps as ``blind``.
``gate``
    the milestone verdict ``M0`` to ``M5`` (exit code 0 passed, 1 not passed, 2 not reached yet);
    ``--check`` only reads the last stored verdict.
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
from twin.eval.gates import GateError, check_gate, run_gate
from twin.eval.memory_test import (
    EVAL_MEMORY_JOB,
    MemoryPlan,
    MemorySession,
    handle_eval_memory,
    plan_memory,
    summarize,
)
from twin.eval.report import (
    print_blind_report,
    print_gate,
    print_memory_summary,
    print_style_report,
)
from twin.eval.samples import StickerDescriber
from twin.eval.store import EvalStore, EvalStoreError, RunView
from twin.eval.style_metrics import StyleError, style_of_items, style_of_live
from twin.eval.ui import BlindSession, KeySource, TerminalKeys
from twin.llm.onetime import BatchStatus
from twin.llm.runtime import build_llm_runtime
from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
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


def _describe_plan(console: Console, services: Services, plan: BlindPlan) -> None:
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
            f"批准并跑完后：twin eval blind --resume {run.id}"
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
        _describe_plan(console, services, plan)
        return
    run = _run_of(store, resume, "blind")
    pending = store.counts(run.id)["pending"]
    if pending and foreground:
        if app_is_running(services):
            typer.echo("the application is running and will execute the approved jobs")
        else:
            _generate_in_foreground(services, EVAL_GENERATE_JOB, handle_eval_generate)
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
        console.print(Text(f"没做完：twin eval blind --resume {run.id}"))


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


# ------------------------------------------------------------------------- runs


@eval_app.command("runs")
@command(CommandKind.READ)
def eval_runs(
    kind: Annotated[str | None, typer.Option("--kind", help="blind, memory, style or gate")] = None,
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
