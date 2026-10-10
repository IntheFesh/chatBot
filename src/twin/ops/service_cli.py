"""CLI: ``twin supervise`` and ``twin service`` (R-OPS-001, R-ARCH-006).

``supervise``
    EXCLUSIVE, holds the ``supervisor`` lock (``run`` is tolerated: the supervisor's child holds
    it).  Runs :class:`~twin.ops.supervise.Supervisor` with the alert delivery next to it - the
    supervisor must be able to say that ``twin run`` died while no application is there to say it.
``service install`` / ``uninstall``
    EXCLUSIVE: register or remove the scheduled task.  ``install --print-xml`` only prints the
    task definition (any platform).
``service start`` / ``stop``
    LIGHT: start the task now; stop the supervisor and ``twin run`` gracefully (and by force only
    if they do not stop in time).
``service status``
    READ: the task, the two processes, the restarts of the last day.

The scheduled task exists on Windows only; elsewhere the commands say so, and ``stop`` still works
for a ``twin supervise`` started by hand.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from twin.app import ShutdownSignals
from twin.clock import Clock, SystemClock
from twin.config.loader import ConfigError
from twin.config.secrets import SecretStoreError
from twin.ops.alert_delivery import AlertDelivery
from twin.ops.instance_lock import LOCK_RUN, LOCK_SUPERVISOR
from twin.ops.jobobject import ProcessJob
from twin.ops.logging import configure_logging, get_logger, shutdown_logging
from twin.ops.mail import SMTP_PASSWORD_SECRET, SmtpMailer
from twin.ops.notify import default_notifier
from twin.ops.process_model import CliError, CommandKind, ExitCode, command
from twin.ops.service import default_waiter, read_status, stop_file_path, stop_service
from twin.ops.supervise import (
    RestartLog,
    SubprocessLauncher,
    Supervisor,
    child_arguments,
)
from twin.ops.taskscheduler import (
    SUPERVISE_ARGUMENTS,
    CommandRunner,
    SubprocessRunner,
    TaskInfo,
    TaskScheduler,
    TaskSchedulerError,
    TaskSpec,
    build_task_xml,
)
from twin.ops.winapi import Win32
from twin.schedule.service import time_service_for
from twin.services import CliContext, Services, get_cli_context
from twin.storage.keystore import KeyStoreError
from twin.storage.migrate import SchemaOutdatedError

log = get_logger("twin.service_cli")

service_app = typer.Typer(
    help="The scheduled task that keeps the bot running (Windows).", no_args_is_help=True
)

AUTOLOGON_NOTE = (
    "说明：计划任务只在你登录 Windows 之后运行（它需要你的凭据管理器、通知和二维码窗口）。\n"
    "如果希望断电或系统更新重启后无人值守地恢复，需要开启 Windows 自动登录"
    "（netplwiz 或 Sysinternals Autologon）。风险：开机后任何接触这台电脑的人都直接进入你的账户，"
    "所以只在电脑放在可信的地方、并开了磁盘加密（BitLocker）时才这么做；"
    "否则重启后需要你手动登录一次。"
)


@dataclass
class ServiceEnv:
    """What the service commands talk to; the tests replace it."""

    runner: CommandRunner = field(default_factory=SubprocessRunner)
    platform: str = field(default_factory=lambda: sys.platform)
    waiter: Callable[[float], None] = default_waiter
    win32: Win32 | None = None  # the mutex API behind the instance locks (tests only)

    def scheduler(self) -> TaskScheduler | None:
        return TaskScheduler(self.runner) if self.platform == "win32" else None


_env = ServiceEnv()


@contextmanager
def use_service_env(env: ServiceEnv) -> Iterator[ServiceEnv]:
    """Use ``env`` for the service commands run inside the block."""
    global _env
    previous, _env = _env, env
    try:
        yield env
    finally:
        _env = previous


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


# ------------------------------------------------------------ the command line of the child


def global_options(context: CliContext) -> list[str]:
    """The options given before the command name, repeated for the child process."""
    options: list[str] = []
    if context.config_path is not None:
        options += ["--config", str(context.config_path)]

    def flatten(prefix: str, node: object) -> Iterator[tuple[str, object]]:
        if isinstance(node, dict):
            for key, value in node.items():
                yield from flatten(f"{prefix}.{key}" if prefix else str(key), value)
        else:
            yield prefix, node

    for key, value in flatten("", context.overrides):
        options += ["--set", f"{key}={json.dumps(value, ensure_ascii=False)}"]
    if context.log_level != "INFO":
        options += ["--log-level", context.log_level]
    return options


def task_spec(root: Path, options: list[str], *, python: Path | None = None) -> TaskSpec:
    """The task definition: ``twin.exe supervise --from-task`` of this environment, in ``root``."""
    interpreter = python or Path(sys.executable)
    twin_exe = interpreter.parent / "twin.exe"
    arguments = [*options, *SUPERVISE_ARGUMENTS]
    if twin_exe.is_file():
        command_path = str(twin_exe)
    else:
        command_path = str(interpreter)
        arguments = ["-X", "utf8", "-m", "twin", *arguments]
    domain, name = os.environ.get("USERDOMAIN", ""), os.environ.get("USERNAME", "")
    user = f"{domain}\\{name}" if domain and name else name
    return TaskSpec(
        command=command_path,
        arguments=subprocess.list2cmdline(arguments),
        working_dir=str(root),
        user_id=user,
    )


# ---------------------------------------------------------------------------- supervise


def _try_services(context: CliContext) -> Services | None:
    """The services container, or ``None`` when the database cannot be used yet.

    A supervisor that cannot start because the database needs a migration would never start
    ``twin run`` - which would say the same - so it works without alerts and restart records.
    """
    try:
        return context.services()
    except (SchemaOutdatedError, KeyStoreError, SecretStoreError, ConfigError, OSError) as exc:
        log.warning("supervisor_without_database", reason=type(exc).__name__)
        return None


async def _supervise(context: CliContext, services: Services | None, *, launch: str) -> int:
    paths = context.paths()
    settings = context.settings()
    stop = asyncio.Event()
    signals = ShutdownSignals(asyncio.get_running_loop(), stop)
    job = ProcessJob()
    job.open()
    delivery: AlertDelivery | None = None
    history: RestartLog | None = None
    if services is not None:
        time = time_service_for(services)
        mailer = SmtpMailer(settings.ops.smtp, lambda: services.secrets.get(SMTP_PASSWORD_SECRET))
        delivery = AlertDelivery(
            services.alerts,
            services.clock,
            notifier=default_notifier(),
            mailer=mailer,
            recipient=lambda: settings.ops.smtp.to,
            zone=time.bot_timezone,
        )
        history = RestartLog(services.db, services.clock)
        history.start_session(launch)
    supervisor = Supervisor(
        SubprocessLauncher(child_arguments(global_options(context)), cwd=paths.root),
        context_clock(services),
        settings.ops.supervise,
        env=dict(os.environ),
        alerts=services.alerts if services is not None else None,
        history=history,
        job=job,
        stop_file=stop_file_path(paths.locks_dir),
        launch=launch,
    )
    signals.install()
    try:
        if delivery is not None:
            await delivery.start()
        return await supervisor.run(stop)
    finally:
        if delivery is not None:
            await delivery.stop()
        signals.notify_done()
        signals.uninstall()


def context_clock(services: Services | None) -> Clock:
    return services.clock if services is not None else SystemClock()


@command(CommandKind.EXCLUSIVE, acquires=(LOCK_SUPERVISOR,), tolerates=(LOCK_RUN,))
def supervise_command(
    from_task: Annotated[
        bool, typer.Option("--from-task", hidden=True, help="Set by the scheduled task")
    ] = False,
) -> None:
    """Keep ``twin run`` running: restart it when it crashes (the scheduled task runs this)."""
    context = get_cli_context()
    paths = context.paths()
    configure_logging(paths.logs_dir, level=context.log_level, role="supervise")
    try:
        services = _try_services(context)
        launch = "task" if from_task else "manual"
        log.info("supervisor_starting", launch=launch, pid=os.getpid())
        code = asyncio.run(_supervise(context, services, launch=launch))
    finally:
        shutdown_logging()
    if code:
        raise typer.Exit(code)


# ------------------------------------------------------------------------------- service


def _need_windows(what: str) -> TaskScheduler:
    scheduler = _env.scheduler()
    if scheduler is None:
        raise CliError(
            f"{what}: the scheduled task exists on Windows only. Elsewhere run "
            "`twin supervise` yourself (or from your own init system).",
            ExitCode.FAILURE,
        )
    return scheduler


def describe_task(info: TaskInfo) -> list[str]:
    """The facts of a registered task, one per line, with what is wrong with them."""
    lines = [
        f"  runs as: {info.command or '?'} {info.arguments or ''}".rstrip(),
        f"  folder: {info.working_dir or '?'}",
        f"  logon type: {info.logon_type or '?'}"
        + ("" if info.interactive else "  <- must be InteractiveToken"),
        f"  time limit: {info.time_limit or '?'}"
        + ("" if info.time_limit == "PT0S" else "  <- must be none (PT0S)"),
        f"  second instance: {info.multiple_instances or '?'}",
        f"  runs on battery: {'yes' if info.on_battery_allowed else 'NO'}",
        f"  restart after failure every: {info.restart_interval or 'never'}",
    ]
    if info.runs_uv_sync:
        lines.append("  <- the action runs `uv sync`; it must not change the environment")
    if info.enabled is False:
        lines.append("  <- the task is disabled")
    return lines


@service_app.command("install")
@command(CommandKind.EXCLUSIVE)
def service_install(
    print_xml: Annotated[
        bool, typer.Option("--print-xml", help="Only print the task definition")
    ] = False,
) -> None:
    """Register the scheduled task that starts the bot when you log on."""
    context = get_cli_context()
    spec = task_spec(context.paths().root, global_options(context))
    if print_xml:
        typer.echo(build_task_xml(spec))
        return
    scheduler = _need_windows("install")
    try:
        scheduler.install(spec)
        info = scheduler.info()
    except TaskSchedulerError as exc:
        raise CliError(str(exc)) from exc
    if not info.interactive:
        raise CliError(
            f"the task was registered but its logon type is {info.logon_type}, "
            "not InteractiveToken; remove it with `twin service uninstall` and report this"
        )
    typer.echo(f"scheduled task '{scheduler.name}' registered for {spec.user_id}:")
    for line in describe_task(info):
        typer.echo(line)
    typer.echo("it starts when you log on; start it now with `twin service start`.")
    typer.echo(AUTOLOGON_NOTE)


@service_app.command("uninstall")
@command(CommandKind.EXCLUSIVE)
def service_uninstall() -> None:
    """Remove the scheduled task (the data stays; `twin purge` deletes data)."""
    scheduler = _need_windows("uninstall")
    try:
        removed = scheduler.uninstall()
    except TaskSchedulerError as exc:
        raise CliError(str(exc)) from exc
    typer.echo("scheduled task removed" if removed else "the scheduled task was not registered")


@service_app.command("start")
@command(CommandKind.LIGHT, consent=False)
def service_start() -> None:
    """Start the scheduled task now."""
    context = get_cli_context()
    scheduler = _need_windows("start")
    from twin.ops.service import running

    if running(context.paths().locks_dir, platform=_env.platform, win32=_env.win32)[0]:
        typer.echo("the supervisor is already running")
        return
    try:
        if not scheduler.info().registered:
            raise CliError("the scheduled task is not registered; run `twin service install`")
        scheduler.start()
    except TaskSchedulerError as exc:
        raise CliError(str(exc)) from exc
    typer.echo("started; watch it with `twin service status` and `twin health`")


@service_app.command("stop")
@command(CommandKind.LIGHT, consent=False)
def service_stop() -> None:
    """Stop the supervisor and `twin run`: the reply being sent is finished first."""
    context = get_cli_context()
    settings = context.settings()
    outcome = stop_service(
        context.paths().locks_dir,
        scheduler=_env.scheduler(),
        grace_s=settings.ops.supervise.stop_grace_s,
        waiter=_env.waiter,
        platform=_env.platform,
        win32=_env.win32,
    )
    if not outcome.was_running:
        typer.echo("nothing was running")
    elif outcome.manual_run:
        raise CliError(
            "a `twin run` is running without a supervisor (started by hand): "
            "press Ctrl+C in its window",
            ExitCode.BUSY,
        )
    elif outcome.still_running:
        raise CliError("it did not stop in time; end the processes in the task manager")
    else:
        how = "forced (the task was ended)" if outcome.forced else "gracefully"
        typer.echo(f"stopped {how}")


@service_app.command("status")
@command(CommandKind.READ, consent=False)
def service_status() -> None:
    """The scheduled task, the processes, the restarts of the last day."""
    context = get_cli_context()
    paths = context.paths()
    restart_log = None
    now = None
    try:
        services = context.services()
        restart_log = RestartLog(services.db, services.clock)
        now = services.clock.now_utc()
    except (SchemaOutdatedError, KeyStoreError, SecretStoreError, OSError):
        services = None
    if now is None:
        now = SystemClock().now_utc()
    status = read_status(
        paths.locks_dir,
        _env.scheduler(),
        restart_log,
        now,
        platform=_env.platform,
        win32=_env.win32,
    )
    if status.task is None:
        reason = status.task_error or "the scheduled task exists on Windows only"
        typer.echo(f"scheduled task: not available ({reason})")
    elif not status.task.registered:
        typer.echo("scheduled task: NOT REGISTERED (run `twin service install`)")
    else:
        typer.echo(f"scheduled task: registered, status {status.task_state or '?'}")
        for line in describe_task(status.task):
            typer.echo(line)
    typer.echo(
        f"supervisor (twin supervise): {'running' if status.supervisor_running else 'not running'}"
    )
    typer.echo(f"application (twin run):      {'running' if status.run_running else 'not running'}")
    if status.session:
        typer.echo(
            f"supervisor session: started {status.session.get('started_at')} "
            f"({status.session.get('launch')})"
        )
    if status.restarts is not None:
        r = status.restarts
        typer.echo(f"restarts: {r.last_24h} in the last 24 hours, {r.total} recorded in all")
        if r.last_at is not None:
            typer.echo(f"last restart: {r.last_at.isoformat()} ({r.last_reason})")
    if status.supervisor_running is False and status.run_running is False:
        typer.echo("nothing is running; `twin service start` starts it")
