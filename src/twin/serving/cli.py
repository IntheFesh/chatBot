"""CLI: ``twin model activate|disable|serve|verify|recommend|evaluate|tunnel`` (round 14).

These join ``twin model register|list|show`` of round 13 (:mod:`twin.training.model_cli`).  Process
models (R-ARCH-006): ``activate``, ``disable``, ``evaluate`` and ``tunnel start|stop`` are LIGHT -
they write short records and the running application follows ``state_version``; ``serve``,
``verify`` and ``tunnel start`` (when no application is running) hold a server or a tunnel in this
window until Ctrl+C or the end of their work, and refuse to start a second owner of a port the
application already owns; ``recommend`` and ``tunnel status`` are READ.

A server started here is put in a Windows job object (:mod:`twin.ops.jobobject`) so that it
ends with this process, however the process ends.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from twin.app import ShutdownSignals
from twin.config.runtime import BACKEND_ACTIVE, TUNNEL_WANTED
from twin.engine.style_models import StyleModels
from twin.eval.blind import EvalError
from twin.eval.cli import continue_blind, describe_plan, interaction, run_blind_jobs
from twin.eval.gates import GateContext, GateOutcome
from twin.eval.report import print_gate, print_style_report
from twin.eval.store import EvalStore, EvalStoreError
from twin.eval.style_metrics import StyleError, style_of_runs
from twin.ops.jobobject import ProcessJob
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.ops.taskscheduler import SubprocessRunner
from twin.services import Services, get_cli_context
from twin.serving.activation import (
    ActivationRefused,
    TokenCheck,
    activate_model,
    disable_model,
    require_servable,
)
from twin.serving.evaluation import (
    EvaluationError,
    EvaluationServers,
    install_servers,
    plan_model_evaluation,
    tunnel_known_hosts,
)
from twin.serving.gate_m5 import MODEL_PARAM, WAYS, evaluation_runs, judge_model
from twin.serving.hardware import choose_build, detect_gpu
from twin.serving.llamacpp import ServeError, find_install, loopback_endpoint
from twin.serving.quant import (
    QuantOption,
    estimated_options,
    kv_bytes_per_token,
    recommend_quant,
)
from twin.serving.runtime import (
    TokenizerSource,
    build_local_server,
    client_for,
    model_path,
    parse_size,
    primary_port,
    resolve_program,
    verify_model_file,
)
from twin.serving.server import ServerSnapshot, ServerState
from twin.serving.state import ServingStateStore
from twin.serving.tokencheck import TokenizationReport
from twin.serving.tunnel import TunnelManager, TunnelSnapshot, TunnelState, TunnelTimings
from twin.training.model_cli import model_app
from twin.training.profiles import PROFILES
from twin.training.registry import (
    ModelView,
    RegistryError,
    find_model,
    get_model,
    list_models,
    record_eval,
)
from twin.training.remote.connection import RemoteError, target_from_settings

tunnel_app = typer.Typer(help="The SSH tunnel to the rented vLLM instance.", no_args_is_help=True)
model_app.add_typer(tunnel_app, name="tunnel")

GIB = 1024**3


def _services() -> Services:
    return get_cli_context().services()


def _console() -> Console:
    return interaction().console()


def _adopt_job() -> ProcessJob:
    """Children of this command end with it (a no-op off Windows)."""
    job = ProcessJob()
    job.adopt_current_process()
    return job


def _fail(message: str, *, lines: tuple[str, ...] = ()) -> CliError:
    for line in lines:
        typer.echo(line, err=True)
    return CliError(message, ExitCode.FAILURE)


def _model(services: Services, reference: str | None) -> ModelView:
    """The model named, or the active one when none is named."""
    try:
        if reference is not None:
            return find_model(services.db, reference)
        active = StyleModels(services.db, mode=services.settings.style_model.mode).active()
        if active is None:
            raise RegistryError("no model is active; name one: `twin model list`")
        return get_model(services.db, active.id)
    except RegistryError as exc:
        raise _fail(str(exc)) from exc


def _servers(services: Services, job: ProcessJob) -> EvaluationServers:
    return EvaluationServers(services, tokenizers=TokenizerSource(services), job=job)


# ------------------------------------------------------------------- activate


def _token_check(services: Services, job: ProcessJob) -> TokenCheck:
    async def check(model: ModelView) -> TokenizationReport:
        pool = _servers(services, job)
        try:
            return await pool.verify(model)
        except EvaluationError as exc:
            raise ActivationRefused("tokenizer_unavailable", str(exc)) from exc
        finally:
            await pool.aclose()

    return check


@model_app.command("activate")
@command(CommandKind.LIGHT)
def model_activate_command(
    model_id: Annotated[str, typer.Argument(help="Model id from `twin model list`")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Activate although the release gate (M5) is not passed"),
    ] = False,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="style or hybrid (the better one by default)"),
    ] = None,
) -> None:
    """Make a model the one in use, if it passed the release gate (R-SRV-005)."""
    services = _services()
    job = _adopt_job()
    try:
        result = asyncio.run(
            activate_model(
                services,
                model_id,
                token_check=_token_check(services, job),
                force=force,
                backend=backend,
            )
        )
    except ActivationRefused as exc:
        raise _fail(exc.message, lines=exc.lines) from exc
    model = result.model
    typer.echo(
        f"activated {model.id} ({model.kind} {model.quant}); backend.active = {result.backend}"
    )
    if result.previous is not None:
        typer.echo(f"{result.previous.id} is no longer active")
    if result.forced:
        typer.echo(
            "FORCED: the model did not pass the release gate. It is marked as not passed "
            "(/状态 says so), the budget's last level will not hand over to it, and the "
            "activation is on record."
        )
    elif result.verdict is not None:
        typer.echo(result.verdict.summary)
    typer.echo("A running application follows within seconds (it starts or stops the server).")


@model_app.command("disable")
@command(CommandKind.LIGHT)
def model_disable_command(
    model_id: Annotated[str, typer.Argument(help="Model id from `twin model list`")],
) -> None:
    """Stop using a model; with no other active model the default is DeepSeek again."""
    services = _services()
    try:
        result = disable_model(services, model_id)
    except ActivationRefused as exc:
        raise _fail(exc.message) from exc
    typer.echo(f"disabled {result.model.id}")
    if result.backend_reset:
        typer.echo("backend.active is deepseek again")


# ---------------------------------------------------------------------- serve


def _print_server(snapshot: ServerSnapshot) -> None:
    detail = f": {snapshot.detail}" if snapshot.detail else ""
    typer.echo(f"llama-server {snapshot.state.value}{detail}")


async def _serve(services: Services, model: ModelView, job: ProcessJob) -> int:
    config = services.settings.style_model
    program = resolve_program(services)
    path = model_path(services, model)
    typer.echo(f"checking {path.name} against its registered sha256 ...")
    await asyncio.to_thread(verify_model_file, model, path)
    local = build_local_server(
        services,
        model,
        program,
        port=primary_port(config),
        tokenizers=TokenizerSource(services),
        job=job,
        on_change=_print_server,
    )
    stop = asyncio.Event()
    signals = ShutdownSignals(asyncio.get_running_loop(), stop)
    signals.install()
    task = asyncio.create_task(local.manager.run())
    try:
        reported = False
        while not stop.is_set() and not task.done():
            state = local.manager.state
            if state is ServerState.READY and not reported:
                reported = True
                typer.echo(f"serving {model.id} on {local.endpoint} (Ctrl+C ends it)")
                if local.warmup is not None:
                    typer.echo(f"warm-up: {local.warmup.line()}")
                    await asyncio.to_thread(
                        record_eval, services.db, model.id, {"warmup": local.warmup.to_json()}
                    )
            elif state is ServerState.BLOCKED:
                blocked = local.manager.blocked
                typer.echo(
                    f"refused: {blocked.reason if blocked else ''} - "
                    f"{local.manager.snapshot().detail}",
                    err=True,
                )
                return 1
            await services.clock.sleep(0.2)
    finally:
        await local.manager.stop()
        await asyncio.gather(task, return_exceptions=True)
        await local.client.aclose()
        signals.notify_done()
        signals.uninstall()
    if task.done() and not task.cancelled() and task.exception() is not None:
        typer.echo(f"error: {task.exception()}", err=True)
        return 1
    return 0


@model_app.command("serve")
@command(CommandKind.LIGHT)
def model_serve_command(
    model_id: Annotated[
        str | None, typer.Argument(help="Model id (default: the active model)")
    ] = None,
) -> None:
    """Run llama-server for a model in this window (the application does it by itself)."""
    services = _services()
    if app_is_running(services):
        raise _fail(
            "the application is running and starts llama-server by itself while a style or "
            "hybrid backend is asked for; to look at it use `twin health` and `twin model show`"
        )
    model = _model(services, model_id)
    if services.settings.style_model.mode != "llamacpp_completion":
        raise _fail("style_model.mode is vllm_completion: use `twin model tunnel start`")
    try:
        require_servable(services, model)
        loopback_endpoint(services.settings.style_model.endpoint)
        job = _adopt_job()
        code = asyncio.run(_serve(services, model, job))
    except (ActivationRefused, ServeError, EvaluationError) as exc:
        raise _fail(getattr(exc, "message", str(exc))) from exc
    if code:
        raise typer.Exit(code)


# --------------------------------------------------------------------- verify


@model_app.command("verify")
@command(CommandKind.LIGHT)
def model_verify_command(
    model_id: Annotated[str, typer.Argument(help="Model id from `twin model list`")],
) -> None:
    """Start the model's server and compare its tokens with the training tokenizer (R-TRN-011)."""
    services = _services()
    model = _model(services, model_id)
    job = _adopt_job()

    async def verify() -> TokenizationReport:
        pool = _servers(services, job)
        try:
            return await pool.verify(model)
        finally:
            await pool.aclose()

    try:
        require_servable(services, model)
        report = asyncio.run(verify())
    except (ActivationRefused, EvaluationError) as exc:
        raise _fail(getattr(exc, "message", str(exc))) from exc
    for line in report.lines():
        typer.echo(line)
    if not report.ok:
        services.alerts.raise_alert(
            "style_tokenize_mismatch",
            f"the tokens of {model.id} differ from the training tokenizer",
            severity="warning",
            detail={"model": model.id, "differences": len(report.differences)},
            dedup_key=f"style_tokenize_mismatch:{model.id}",
        )
        raise typer.Exit(1)


# ------------------------------------------------------------------ recommend


@model_app.command("recommend")
@command(CommandKind.READ, consent=False)
def model_recommend_command(
    vram: Annotated[
        str | None,
        typer.Option("--vram", help="Memory of the card instead of asking nvidia-smi, e.g. 24GiB"),
    ] = None,
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Advise for a computer without a card")
    ] = False,
) -> None:
    """Which quantisation (Q8_0, Q5_K_M, Q4_K_M) the graphics card can hold (R-SRV-002)."""
    services = _services()
    config = services.settings.style_model
    gpu = None if (cpu or vram is not None) else detect_gpu(SubprocessRunner())
    memory: int | None
    if cpu:
        memory = None
        typer.echo("graphics card: none (--cpu)")
    elif vram is not None:
        memory = parse_size(vram)
        if memory is None:
            raise CliError("--vram looks like 24GiB, 24576MiB or 24576", ExitCode.USAGE)
        typer.echo(f"graphics card: {memory / GIB:.1f} GiB (--vram)")
    else:
        memory = gpu.memory_bytes if gpu is not None else None
        typer.echo(f"graphics card: {gpu.describe() if gpu else 'none found (nvidia-smi)'}")
        build = choose_build(gpu)
        typer.echo(f"llama.cpp build for it: {build.kind} ({build.reason})")
        install = find_install(services.paths.root, config.serve.binary)
        if install is not None:
            typer.echo(f"installed: {install.version} {install.kind}")
    context = config.serve.context
    registered = [m for m in list_models(services.db) if m.kind == "gguf"]
    groups: dict[str, list[ModelView]] = {}
    for model in registered:
        groups.setdefault(model.run_id, []).append(model)
    if not groups:
        typer.echo("no GGUF registered yet: sizes estimated from the training profiles")
        for name, profile in PROFILES.items():
            typer.echo(f"\n{name} ({profile.base_model})")
            advice = recommend_quant(
                estimated_options(profile.base_gb),
                vram_bytes=memory,
                context=context,
                kv_per_token=kv_bytes_per_token(profile.base_model),
            )
            for line in advice.lines():
                typer.echo(f"  {line}")
        return
    for run_id, models in groups.items():
        base = models[0].base_model
        options = [QuantOption(m.quant, m.size) for m in models]
        advice = recommend_quant(
            options,
            vram_bytes=memory,
            context=context,
            kv_per_token=kv_bytes_per_token(base),
        )
        typer.echo(f"\nrun {run_id} ({base})")
        for line in advice.lines():
            typer.echo(f"  {line}")
        if advice.quant is not None:
            chosen = next(m for m in models if m.quant == advice.quant)
            typer.echo(f"  -> twin model evaluate {chosen.id}")


# ------------------------------------------------------------------- evaluate


def _generate_with_servers(services: Services) -> Callable[[Services], None]:
    """Run the approved jobs here with the model's server (started on demand, ended after)."""

    def generate(_: Services) -> None:
        job = _adopt_job()

        async def run() -> None:
            pool = _servers(services, job)
            install_servers(pool)
            try:
                await run_blind_jobs(services)
            finally:
                install_servers(None)
                await pool.aclose()

        asyncio.run(run())

    return generate


def _print_results(console: Console, services: Services, model: ModelView) -> None:
    """After the judging: the style metrics of each way and what the gate would say."""
    store = EvalStore(services.db, services.clock)
    runs = evaluation_runs(store, model.id)
    for way in WAYS:
        try:
            print_style_report(console, style_of_runs(services, store, runs, way))
        except StyleError as exc:
            console.print(Text(f"{way}: {exc}"))
    verdict = judge_model(GateContext(services, store), model.id)
    outcome = GateOutcome(
        "M5", 0 if verdict.passed else 1, verdict.verdict, verdict.summary, verdict
    )
    print_gate(console, outcome)
    console.print(
        Text(
            "这只是预览；`twin eval gate M5` 会把判定记下来，"
            f"通过后 `twin model activate {model.id}` 才会生效。"
        )
    )


@model_app.command("evaluate")
@command(CommandKind.LIGHT)
def model_evaluate_command(
    model_id: Annotated[str, typer.Argument(help="Model id from `twin model list`")],
    n: Annotated[
        int | None, typer.Option("--n", min=1, help="Contexts to draw (default: eval.blind_n)")
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue the evaluation run with this id")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Seed of the draw")] = None,
    foreground: Annotated[
        bool,
        typer.Option("--foreground", help="Run the approved generation jobs here (no application)"),
    ] = False,
) -> None:
    """Blind test of DeepSeek against a model (as style and as hybrid) on new contexts (R-SRV-005).

    Without ``--resume`` it draws the contexts, prices the generation and queues it (approve the
    batches with ``twin jobs approve``); with ``--resume <run>`` it generates (``--foreground``:
    here, with the model's server), lets you judge and shows the results.
    """
    services = _services()
    console = _console()
    store = EvalStore(services.db, services.clock)
    contexts = n if n is not None else services.settings.eval.blind_n
    try:
        if resume is None:
            model, plan = asyncio.run(
                plan_model_evaluation(services, model_id, contexts, seed=seed)
            )
            describe_plan(console, services, plan, resume_command=f"twin model evaluate {model.id}")
            return
        model = _model(services, model_id)
        run = store.get_run(resume)
    except (EvaluationError, EvalError, EvalStoreError) as exc:
        raise _fail(str(exc)) from exc
    if run.kind != "blind" or run.params.get(MODEL_PARAM) != model.id:
        raise _fail(f"run {run.id} was not drawn for {model.id}", lines=())
    continue_blind(
        services,
        console,
        store,
        run,
        foreground=foreground,
        generate=_generate_with_servers(services),
        resume_command=f"twin model evaluate {model.id}",
    )
    if store.get_run(run.id).status == "done":
        _print_results(console, services, model)


# --------------------------------------------------------------------- tunnel


def _print_tunnel(snapshot: TunnelSnapshot) -> None:
    detail = f": {snapshot.detail}" if snapshot.detail else ""
    typer.echo(f"tunnel {snapshot.state.value}{detail}")


async def _hold_tunnel(services: Services) -> None:
    config = services.settings.style_model
    target = target_from_settings(
        services.settings.autodl, services.secrets, tunnel_known_hosts(services)
    )
    probe = client_for(config, services.clock)
    tunnel = TunnelManager(
        target,
        local_port=config.tunnel.local_port,
        remote_port=config.tunnel.remote_port,
        clock=services.clock,
        timings=TunnelTimings(config.tunnel.backoff_start_s, config.tunnel.backoff_max_s),
        local_health=probe.health,
        on_change=_print_tunnel,
    )
    stop = asyncio.Event()
    signals = ShutdownSignals(asyncio.get_running_loop(), stop)
    signals.install()
    task = asyncio.create_task(tunnel.run())

    async def watch_setting() -> None:
        """`twin model tunnel stop` in another window ends this one."""
        while not stop.is_set():
            wanted = await asyncio.to_thread(services.runtime.get, TUNNEL_WANTED)
            if not wanted:
                stop.set()
                return
            await services.clock.sleep(2.0)

    watcher = asyncio.create_task(watch_setting())
    try:
        announced = False
        while not stop.is_set() and not task.done():
            if tunnel.state is TunnelState.UP and not announced:
                announced = True
                typer.echo(
                    f"127.0.0.1:{tunnel.local_port} -> the instance's "
                    f"127.0.0.1:{config.tunnel.remote_port} "
                    "(Ctrl+C or `twin model tunnel stop` ends it)"
                )
            await services.clock.sleep(0.5)
    finally:
        await tunnel.stop()
        watcher.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)
        await probe.aclose()
        signals.notify_done()
        signals.uninstall()


@tunnel_app.command("start")
@command(CommandKind.LIGHT)
def tunnel_start_command() -> None:
    """Keep the SSH tunnel to the instance up (the application does it itself when it runs)."""
    services = _services()
    if services.settings.style_model.mode != "vllm_completion":
        raise _fail("style_model.mode is llamacpp_completion: there is no remote model to reach")
    try:
        target_from_settings(
            services.settings.autodl, services.secrets, tunnel_known_hosts(services)
        )
    except RemoteError as exc:
        raise _fail(str(exc)) from exc
    services.runtime.set(TUNNEL_WANTED, True, by="command")
    if app_is_running(services):
        typer.echo(
            "the application is running and brings the tunnel up (`twin model tunnel status`)"
        )
        return
    typer.echo("no application is running: holding the tunnel in this window")
    try:
        asyncio.run(_hold_tunnel(services))
    finally:
        services.runtime.set(TUNNEL_WANTED, False, by="command")
    typer.echo("tunnel closed")


@tunnel_app.command("stop")
@command(CommandKind.LIGHT)
def tunnel_stop_command() -> None:
    """Close the tunnel and stop asking for it; then shut the instance down in the console."""
    services = _services()
    changed = services.runtime.set(TUNNEL_WANTED, False, by="command")
    typer.echo("the tunnel is closed" if changed else "the tunnel was not wanted")
    requested = services.runtime.get(BACKEND_ACTIVE)
    if requested != "deepseek" and services.settings.style_model.mode == "vllm_completion":
        typer.echo(
            f"backend.active is {requested}: replies fall back to DeepSeek now; "
            "send /后端 deepseek to make that the choice"
        )
    typer.echo(
        "The rented instance is billed by the hour until it is shut down: stop it in the AutoDL "
        "console (shutting down keeps the disk; the instance is erased after 15 days)."
    )


@tunnel_app.command("status")
@command(CommandKind.READ)
def tunnel_status_command() -> None:
    """Whether the tunnel is wanted and up, how often it reconnected, how long the instance runs."""
    services = _services()
    wanted = bool(services.runtime.get(TUNNEL_WANTED))
    record = ServingStateStore(services.db, services.clock).read()
    tunnel = record.get("tunnel") if isinstance(record.get("tunnel"), dict) else None
    table = Table("field", "value", show_header=False)
    table.add_row("mode", services.settings.style_model.mode)
    table.add_row("wanted", "yes" if wanted else "no")
    if tunnel is not None:
        table.add_row("state", str(tunnel.get("state")))
        table.add_row("up since", str(tunnel.get("up_since") or "-"))
        table.add_row("reconnects", str(tunnel.get("reconnects")))
        uptime = tunnel.get("instance_uptime_s")
        table.add_row("instance up for", f"{float(uptime) / 3600:.1f} h" if uptime else "-")
        if tunnel.get("detail"):
            table.add_row("detail", str(tunnel["detail"]))
        if not app_is_running(services):
            table.add_row("note", "no application is running: this record may be old")
    else:
        table.add_row("state", "no tunnel is held by the application")

    async def answers() -> str:
        client = client_for(services.settings.style_model, services.clock)
        try:
            health = await client.health()
        except ServeError as exc:
            return str(exc)
        finally:
            await client.aclose()
        return "answers" if health.ok else f"does not answer ({health.detail})"

    if services.settings.style_model.mode == "vllm_completion":
        table.add_row(
            f"127.0.0.1:{services.settings.style_model.tunnel.local_port}", asyncio.run(answers())
        )
    Console(highlight=False, soft_wrap=True).print(table)
