"""``twin channel probe``: start, status, answer, stop and report (R-CH-009, R-CH-010).

``start`` only stores the plan; the running application (``twin run``) carries it out, because
it takes more than a day and needs the live channel.  Everything the probe needs from the
person at the keyboard (the fresh WeChat messages, what arrived on the phone, whether the GIF
moves) comes through ``answer``; ``status`` shows where things are; ``report`` writes
``docs/CHANNEL_REPORT.md`` and offers the measured values for the configuration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from twin.channel.base import AuthState
from twin.channel.console import Prompter, TyperPrompter
from twin.channel.ilink.store import IlinkStore
from twin.channel.probe.model import (
    PlanStatus,
    ProbeOptions,
    ProbePlan,
    Question,
    StepState,
    StepStatus,
)
from twin.channel.probe.report import render_report, write_report
from twin.channel.probe.store import AnswerRejected, ProbeAlreadyRunning, ProbeStore
from twin.channel.probe.summary import (
    VERDICT_NOT_MET,
    ChannelProbeSummary,
    load_channel_probe_summary,
    summarize,
)
from twin.channel.state import ChannelStateStore
from twin.config.loader import ConfigError, default_config_path
from twin.config.writer import set_section_values
from twin.ops.process_model import CliError, CommandKind, app_is_running, command
from twin.services import Services, get_cli_context

probe_app = typer.Typer(
    help="M0 probe: measure what the ClawBot channel allows (a day-long, restartable plan).",
    no_args_is_help=True,
)

REPORT_NAME = "CHANNEL_REPORT.md"
LATER_WORDS = {"q", "quit", "later", "skip"}


def _store(services: Services) -> ProbeStore:
    return ProbeStore(services.db, services.clock)


def _check_ready(services: Services) -> None:
    """The probe needs a WeChat login and a bound user."""
    if services.settings.channel.kind != "ilink":
        raise CliError("channel.kind is not 'ilink': the probe measures the WeChat channel")
    ilink = IlinkStore(ChannelStateStore(services.db), services.clock)
    if ilink.credentials() is None:
        raise CliError("not logged in: run `twin channel login` first")
    if ilink.auth_record().state is AuthState.NEEDS_RELOGIN:
        raise CliError("the WeChat login expired: run `twin channel login --force` first")
    if ilink.bound_user() is None:
        raise CliError("nobody is bound yet: run `twin channel login` and bind your account")


# ----------------------------------------------------------------- rendering


def _step_line(index: int, total: int, step: StepState) -> str:
    base = f"  {index}/{total} {step.id.value:<12} {step.status.value:<8}"
    data = step.data
    if step.status is StepStatus.SKIPPED:
        return f"{base} {step.skip_reason}"
    if step.status is not StepStatus.DONE:
        voided = f" ({step.voided_attempts} void)" if step.voided_attempts else ""
        return f"{base}{voided}"
    if step.id.value == "count":
        return (
            f"{base} N={data.get('n')}{' (at least)' if data.get('capped') else ''}, "
            f"server accepted {data.get('api_ok')}, phone received {data.get('phone_received')}"
        )
    if step.id.value == "window":
        return (
            f"{base} delivered up to {data.get('lower_bound_h')} h, "
            f"first miss at {data.get('upper_bound_h')} h"
        )
    return base


def status_lines(plan: ProbePlan | None, *, app_running: bool) -> list[str]:
    """The text of ``twin channel probe status``."""
    if plan is None:
        return ["no probe plan yet: `twin channel probe start`"]
    stalled = "NOT running - the probe only advances while `twin run` is up"
    lines = [
        f"probe {plan.run_id}: {plan.status.value}"
        f"{'' if plan.stop_reason is None else f' ({plan.stop_reason})'}",
        f"application: {'running' if app_running else stalled}",
        f"started {plan.created_at:%Y-%m-%d %H:%M:%S} UTC"
        + (f", ended {plan.finished_at:%Y-%m-%d %H:%M:%S} UTC" if plan.finished_at else ""),
    ]
    if plan.notice:
        lines.append(f">>> {plan.notice}")
        step = plan.active_step()
        attempt = step.current_attempt() if step else None
        if attempt is not None and attempt.announce_note not in (None, "sent"):
            lines.append(
                f"    (the WeChat reminder was {attempt.announce_note}; follow the line above here)"
            )
    pending = ProbeStore.pending_questions(plan)
    if pending:
        lines.append(f">>> {len(pending)} question(s) waiting: run `twin channel probe answer`")
    lines.append("steps:")
    total = len(plan.steps)
    lines += [_step_line(i, total, step) for i, step in enumerate(plan.steps, start=1)]
    if plan.events:
        lines.append("recent events:")
        lines += [f"  {event.at:%m-%d %H:%M:%S} {event.text}" for event in plan.events[-8:]]
    return lines


def ask_question(prompter: Prompter, store: ProbeStore, question: Question) -> bool:
    """Put one question to the user until they give a valid answer; ``False`` if they defer."""
    prompter.say("")
    prompter.say(question.prompt)
    while True:
        hint = f" [{'/'.join(question.choices)}]" if question.choices else ""
        text = prompter.ask(f"answer{hint}  (q = answer later)")
        if text.strip().lower() in LATER_WORDS:
            return False
        try:
            store.answer(question.id, text)
        except AnswerRejected as exc:
            prompter.say(f"  {exc}")
            continue
        return True


# ------------------------------------------------------------------ commands


@probe_app.command("start")
@command(CommandKind.LIGHT)
def probe_start(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
    empty_token_experiment: Annotated[
        bool,
        typer.Option(
            "--empty-token-experiment",
            help="Also test whether a text without context_token gets through (off by default)",
        ),
    ] = False,
) -> None:
    """Store a new probe plan; the running application carries it out (about 26 hours)."""
    services = get_cli_context().services()
    _check_ready(services)
    options = ProbeOptions(empty_token_experiment=empty_token_experiment)
    if not yes:
        typer.echo("This sends '[测试]' messages and synthetic test pictures to your ClawBot chat.")
        typer.echo(
            "Three measurements, each starting when YOU send the bot a fresh message: how many "
            "messages follow it, pictures/GIF/typing, and how long the bot can still write "
            f"(about {max(options.window_hours):g} hours of silence from you)."
        )
        if not typer.confirm("Start the probe?"):
            raise typer.Exit(1)
    try:
        plan = _store(services).create(options)
    except ProbeAlreadyRunning as exc:
        raise CliError(str(exc)) from None
    typer.echo(f"probe {plan.run_id} stored.")
    if app_is_running(services):
        typer.echo("The running application will begin within a few seconds.")
    else:
        typer.echo(
            "Start the application (`twin run`) and keep it running: it carries the probe out."
        )
    typer.echo("Then: `twin channel probe status`, and `twin channel probe answer` when it asks.")


@probe_app.command("status")
@command(CommandKind.READ)
def probe_status() -> None:
    """Where the probe is, what it waits for, and what it has measured so far."""
    services = get_cli_context().services()
    plan = _store(services).load()
    for line in status_lines(plan, app_running=app_is_running(services)):
        typer.echo(line)
    if plan is None:
        with services.db.session() as session:
            stored = load_channel_probe_summary(session)
        if stored is not None:
            typer.echo(f"last stored result: run {stored.run_id}, verdict {stored.verdict}")


@probe_app.command("answer")
@command(CommandKind.LIGHT)
def probe_answer(
    wait: Annotated[
        bool,
        typer.Option("--wait", help="Keep waiting for further questions until the probe ends"),
    ] = False,
) -> None:
    """Answer the probe's questions (what arrived on the phone, whether the GIF moves)."""
    services = get_cli_context().services()
    store = _store(services)
    prompter = TyperPrompter()
    answered = 0
    deferred: set[str] = set()
    while True:
        plan = store.load()
        if plan is None:
            raise CliError("there is no probe plan: `twin channel probe start`")
        pending = [q for q in ProbeStore.pending_questions(plan) if q.id not in deferred]
        for question in pending:
            if ask_question(prompter, store, question):
                answered += 1
            else:
                deferred.add(question.id)
        if not wait or plan.status is not PlanStatus.RUNNING:
            break
        if not pending:
            try:
                asyncio.run(services.clock.sleep(plan.options.poll_s))
            except KeyboardInterrupt:
                break
    if answered == 0 and not deferred:
        typer.echo("no question is waiting for an answer")
    else:
        typer.echo(f"answered {answered} question(s)")


@probe_app.command("stop")
@command(CommandKind.LIGHT)
def probe_stop(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
) -> None:
    """End the running probe now (what was measured so far is kept)."""
    services = get_cli_context().services()
    store = _store(services)
    plan = store.load()
    if plan is None or plan.status is not PlanStatus.RUNNING:
        typer.echo("no probe is running")
        return
    if not yes and not typer.confirm(f"Stop probe {plan.run_id}? Unfinished steps stay unmeasured"):
        raise typer.Exit(1)
    store.finish(PlanStatus.STOPPED, "stopped by the user")
    typer.echo("stopped; `twin channel probe report` shows what was measured")


def _current_summary(services: Services) -> ChannelProbeSummary | None:
    plan = _store(services).load()
    if plan is not None:
        return summarize(plan)
    with services.db.session() as session:
        return load_channel_probe_summary(session)


def _config_path(services: Services) -> Path:
    return get_cli_context().config_path or default_config_path(services.paths.root)


def _offer_suggestions(
    services: Services, summary: ChannelProbeSummary, apply: bool | None
) -> None:
    suggestions = {k: v for k, v in summary.suggestions.items() if v is not None}
    if not suggestions or apply is False:
        return
    config = services.settings.channel
    current = {
        "channel.proactive_window_safe_h": config.proactive_window_safe_h,
        "channel.outbound_quota_safe": config.outbound_quota_safe,
    }
    typer.echo("suggested configuration (measured value with 10% in reserve):")
    for key, value in suggestions.items():
        typer.echo(f"  {key}: {current[key]} -> {value}")
    path = _config_path(services)
    if apply is None and not typer.confirm(f"Write these values to {path}?", default=False):
        typer.echo("configuration left unchanged")
        return
    try:
        set_section_values(
            path, "channel", {key.split(".", 1)[1]: value for key, value in suggestions.items()}
        )
    except ConfigError as exc:
        raise CliError(f"could not update {path}: {exc}") from None
    typer.echo(f"updated {path}")


@probe_app.command("report")
@command(CommandKind.LIGHT)
def probe_report(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", help="Where to write the report (default: docs/CHANNEL_REPORT.md)"
        ),
    ] = None,
    apply: Annotated[
        bool | None,
        typer.Option(
            "--apply/--no-apply",
            help="Write the suggested channel.* values to the configuration without asking "
            "(--apply) or do not offer them (--no-apply); by default you are asked",
        ),
    ] = None,
) -> None:
    """Write docs/CHANNEL_REPORT.md from the probe result and offer the measured values."""
    services = get_cli_context().services()
    summary = _current_summary(services)
    if summary is None:
        raise CliError("no probe result yet: `twin channel probe start`")
    path = output or (services.paths.root / "docs" / REPORT_NAME)
    write_report(path, render_report(summary))
    typer.echo(f"report written to {path}")
    typer.echo(
        f"verdict: {summary.verdict}; N={summary.n_messages}; "
        f"window delivered up to {summary.window_lower_bound_h} h"
    )
    for reason in summary.reasons:
        typer.echo(f"  - {reason}")
    if summary.verdict == VERDICT_NOT_MET:
        raise CliError(
            "STOP: the channel does not meet R-CH-010 (window under 12 hours or fewer than 3 "
            "messages). Proactive messages depend on both; tell the maintainer before building "
            "further (the enterprise WeChat channel is outside this specification)."
        )
    _offer_suggestions(services, summary, apply)
