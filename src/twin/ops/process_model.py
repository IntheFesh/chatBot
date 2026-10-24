"""CLI process model (R-ARCH-006).

Every CLI command is declared with :func:`command` and one of four kinds:

``READ``
    reads the database directly; any database write raises
    :class:`~twin.storage.db.ReadOnlyViolationError`.
``LIGHT``
    short writes; every write transaction also increments ``settings.state_version``
    so the running application (polling every two seconds) invalidates caches.
``HEAVY``
    long work.  The command only enqueues jobs (:func:`enqueue_heavy`); the
    running application executes them, or ``twin jobs run --until-idle`` does when
    the application is stopped (``--foreground`` does that immediately).  What the command
    says about it depends on whether the application runs (:func:`queued_hint`,
    :func:`rerun_hint`; D-826).
``EXCLUSIVE``
    restore, purge, key rotation, migrations: the wrapper *takes* the ``run`` and
    ``supervisor`` instance locks (the ones the command does not ``acquires`` or ``tolerates``
    itself) before the command body and holds them until it has finished, also while it asks
    a question and also when it fails; a lock that another process holds refuses the command
    (D-710).  A check that only looked at the locks would leave a gap in which ``twin run``
    starts between the look and the work.  A command may ``acquires`` locks for its own
    lifetime (``twin run`` takes ``run``) and ``tolerates`` others (``twin run`` is started by
    ``twin supervise``, which holds ``supervisor``).  Because the locks are held, nothing inside
    an exclusive command may look at them again (``app_is_running``, ``locks_held_elsewhere``
    would say "busy" for the command's own hold).

The wrapper also enforces the consent gate (R-SCOPE-003) and turns the known
operational errors into short messages with distinct exit codes.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, TypeVar

import typer

from twin.config.loader import ConfigError, ConsentError, ensure_consent
from twin.config.secrets import SecretStoreError
from twin.config.settings import ConfigFileError
from twin.ops.instance_lock import (
    ALL_LOCKS,
    InstanceAlreadyRunningError,
    InstanceLock,
    locks_held_elsewhere,
)
from twin.ops.logging import configure_logging
from twin.ops.winapi import Win32
from twin.services import CliContext, Services, get_cli_context
from twin.storage.crypto import CryptoError
from twin.storage.db import ReadOnlyViolationError, WritePolicy, use_write_policy
from twin.storage.keystore import KeyStoreError
from twin.storage.migrate import SchemaOutdatedError

F = TypeVar("F", bound=Callable[..., Any])


class CommandKind(StrEnum):
    READ = "read"
    LIGHT = "light"
    HEAVY = "heavy"
    EXCLUSIVE = "exclusive"


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    USAGE = 2
    CONSENT = 3
    BUSY = 4  # an instance lock refused the command
    SCHEMA = 5
    SECRETS = 6
    CONFIG = 7


class CliError(Exception):
    """A user-facing failure: message and exit code."""

    def __init__(self, message: str, code: ExitCode = ExitCode.FAILURE) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class CommandSpec:
    kind: CommandKind
    acquires: tuple[str, ...]
    tolerates: tuple[str, ...]
    consent: bool


SPEC_ATTR = "__twin_command__"


def get_spec(callback: Callable[..., Any]) -> CommandSpec | None:
    spec = getattr(callback, SPEC_ATTR, None)
    return spec if isinstance(spec, CommandSpec) else None


_ERRORS: tuple[tuple[type[Exception], ExitCode], ...] = (
    (ConsentError, ExitCode.CONSENT),
    (SchemaOutdatedError, ExitCode.SCHEMA),
    (SecretStoreError, ExitCode.SECRETS),
    (KeyStoreError, ExitCode.SECRETS),
    (CryptoError, ExitCode.SECRETS),
    (ConfigError, ExitCode.CONFIG),
    (ConfigFileError, ExitCode.CONFIG),
    (InstanceAlreadyRunningError, ExitCode.BUSY),
    (ReadOnlyViolationError, ExitCode.FAILURE),
)


def _fail(message: str, code: ExitCode) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(int(code))


STOP_HINT = (
    "stop the application first: run `twin service stop` "
    "(or press Ctrl+C in the window where `twin run` is open)"
)


def _guard_locks(
    stack: ExitStack,
    spec: CommandSpec,
    locks_dir: Path,
    *,
    platform: str | None = None,
    win32: Win32 | None = None,
) -> None:
    """Take the locks of the command and hand their release to ``stack`` (D-710).

    An EXCLUSIVE command takes every lock it neither acquires nor tolerates and keeps it until
    the stack is closed: the command body runs with ``twin run`` and ``twin supervise`` locked
    out.  If one of them is held elsewhere, the ones taken so far are released by the stack.
    """
    others = tuple(name for name in ALL_LOCKS if name not in spec.acquires + spec.tolerates)
    if spec.kind is CommandKind.EXCLUSIVE:
        for name in others:
            lock = InstanceLock(name, locks_dir=locks_dir, platform=platform, win32=win32)
            if not lock.acquire():
                raise CliError(
                    f"this command needs exclusive access, but the '{name}' instance is "
                    f"running; {STOP_HINT}",
                    ExitCode.BUSY,
                )
            stack.callback(lock.release)
    for name in spec.acquires:
        lock = InstanceLock(name, locks_dir=locks_dir, platform=platform, win32=win32)
        if not lock.acquire():
            raise CliError(
                f"another '{name}' instance is already running (lock '{name}' is held); "
                f"{STOP_HINT}",
                ExitCode.BUSY,
            )
        stack.callback(lock.release)


def _configure_cli_logging(context: CliContext) -> None:
    """File log for CLI invocations; the console shows warnings and errors only.

    A broken configuration or an unusable data directory must not stop diagnostics such as
    ``twin doctor`` from running, so in those cases only console logging is installed.
    """
    try:
        logs_dir: Path | None = context.paths().logs_dir
        configure_logging(
            logs_dir, level=context.log_level, role="cli", console_level=logging.WARNING
        )
    except (ConfigError, ConfigFileError, OSError):
        configure_logging(None, level=context.log_level, role="cli", console_level=logging.WARNING)


def command(
    kind: CommandKind,
    *,
    acquires: Sequence[str] = (),
    tolerates: Sequence[str] = (),
    consent: bool = True,
) -> Callable[[F], F]:
    """Declare the process-model class of a CLI command (apply under ``@app.command``)."""
    spec = CommandSpec(kind, tuple(acquires), tuple(tolerates), consent)

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            context = get_cli_context()
            try:
                with ExitStack() as stack:
                    if spec.consent:
                        ensure_consent(context.settings())
                    if not spec.acquires:  # `twin run` configures its own logging
                        _configure_cli_logging(context)
                    if spec.kind is CommandKind.EXCLUSIVE or spec.acquires:
                        _guard_locks(stack, spec, context.paths().locks_dir)
                    policy = WritePolicy(
                        read_only=spec.kind is CommandKind.READ,
                        bump_state=spec.kind is CommandKind.LIGHT,
                    )
                    stack.enter_context(use_write_policy(policy))
                    return func(*args, **kwargs)
            except CliError as exc:
                raise _fail(exc.message, exc.code) from None
            except tuple(exc_type for exc_type, _ in _ERRORS) as exc:
                code = next(c for t, c in _ERRORS if isinstance(exc, t))
                raise _fail(str(exc), code) from None

        setattr(wrapper, SPEC_ATTR, spec)
        return wrapper  # type: ignore[return-value]

    return decorate


# -------------------------------------------------------------- introspection


def iter_commands(app: typer.Typer, prefix: str = "") -> Iterator[tuple[str, Callable[..., Any]]]:
    """Yield ``(command path, callback)`` for every command of a Typer app tree."""
    for info in app.registered_commands:
        callback = info.callback
        if callback is None:
            continue
        name = info.name or callback.__name__.replace("_", "-")
        yield f"{prefix}{name}", callback
    for group in app.registered_groups:
        if group.typer_instance is None:
            continue
        yield from iter_commands(group.typer_instance, f"{prefix}{group.name or ''} ")


def undeclared_commands(app: typer.Typer) -> list[str]:
    """Names of commands without a process-model declaration (must be empty)."""
    return [name for name, callback in iter_commands(app) if get_spec(callback) is None]


# ------------------------------------------------------------ heavy commands


@dataclass(frozen=True)
class JobSpec:
    """A job to enqueue from a HEAVY command."""

    job_type: str
    payload: Any
    priority: int = 100
    max_attempts: int = 3
    offpeak_only: bool = False
    batch_id: str | None = None
    estimated_cost_usd: float | None = None
    requires_approval: bool = False


def app_is_running(services: Services) -> bool:
    return bool(locks_held_elsewhere(services.paths.locks_dir, ("run",)))


RUNNING_EXECUTES = "the running application executes it"
RUN_THE_QUEUE = "twin jobs run --until-idle"


def queued_hint(services: Services, *, foreground: str | None = None) -> str:
    """Who executes work a command has just queued - the one wording of every HEAVY command.

    With the application running: it does.  Without: the command's ``foreground`` form (when it
    has one) or ``twin jobs run --until-idle`` runs it now; otherwise it waits for the next start
    (R-ARCH-006, D-826).  Not for an EXCLUSIVE command, which holds the ``run`` lock itself.
    """
    if app_is_running(services):
        return RUNNING_EXECUTES
    here = f"`{foreground}` or " if foreground else ""
    return (
        f"the application is not running: {here}`{RUN_THE_QUEUE}` runs it now "
        "(otherwise it waits for the next start)"
    )


def rerun_hint(services: Services, command: str) -> str:
    """``command`` as the user should type it now: with ``--foreground`` while nothing runs it."""
    if app_is_running(services):
        return f"`{command}` ({RUNNING_EXECUTES})"
    return f"`{command} --foreground`"


def enqueue_heavy(
    services: Services, specs: Sequence[JobSpec], *, foreground: bool = False
) -> list[str]:
    """Enqueue HEAVY work; with ``foreground`` run it now if the application is stopped.

    The foreground run holds the ``run`` lock until it has finished (D-711); if the application
    holds it, nothing is run here and the queued jobs are left to the application.
    """
    from twin.llm.runtime import activate_offpeak_policy, close_llm_runtime
    from twin.ops.foreground import ApplicationRunningError, foreground_run_lock
    from twin.ops.jobs import FOREGROUND_WAIT_S, JobQueue, Worker, default_registry, load_handlers

    queue = JobQueue(services.db, services.clock)
    ids = [
        queue.enqueue(
            spec.job_type,
            spec.payload,
            priority=spec.priority,
            max_attempts=spec.max_attempts,
            offpeak_only=spec.offpeak_only,
            batch_id=spec.batch_id,
            estimated_cost_usd=spec.estimated_cost_usd,
            requires_approval=spec.requires_approval,
        )
        for spec in specs
    ]
    if not foreground:
        typer.echo(f"queued {len(ids)} job(s); {queued_hint(services)}")
        return ids
    try:
        with foreground_run_lock(services):
            load_handlers()
            activate_offpeak_policy(services)
            worker = Worker(
                queue,
                default_registry,
                services.clock,
                services=services,
                alerts=services.alerts,
                concurrency=services.settings.jobs.concurrency,
            )

            async def run_and_close() -> Any:
                try:
                    await worker.recover()
                    return await worker.run_until_idle(wait_s=FOREGROUND_WAIT_S)
                finally:
                    # the jobs shared the process's one client (D-701)
                    await close_llm_runtime(services)

            summary = asyncio.run(run_and_close())
    except ApplicationRunningError:
        typer.echo("the application is running and will execute the queued job(s)")
        return ids
    typer.echo(f"finished: done={summary.done} retried={summary.retried} failed={summary.failed}")
    return ids
