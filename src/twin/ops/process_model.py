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
    the application is stopped (``--foreground`` does that immediately).
``EXCLUSIVE``
    restore, purge, key rotation, migrations: refused while the ``run`` or
    ``supervisor`` instance lock is held by another process.  A command may
    ``acquires`` locks for its own lifetime (``twin run`` takes ``run``) and
    ``tolerates`` others (``twin run`` is started by ``twin supervise``, which holds
    ``supervisor``).

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


def stop_hint() -> str:
    return (
        "stop the application first: run `twin service stop` "
        "(or press Ctrl+C in the window where `twin run` is open)"
    )


def _guard_locks(
    stack: ExitStack, spec: CommandSpec, locks_dir: Path, *, platform: str | None = None
) -> None:
    others = tuple(name for name in ALL_LOCKS if name not in spec.acquires + spec.tolerates)
    if spec.kind is CommandKind.EXCLUSIVE:
        busy = locks_held_elsewhere(locks_dir, others, platform=platform)
        if busy:
            raise CliError(
                f"this command needs exclusive access, but the '{busy[0]}' instance is running; "
                + stop_hint(),
                ExitCode.BUSY,
            )
    for name in spec.acquires:
        lock = InstanceLock(name, locks_dir=locks_dir, platform=platform)
        if not lock.acquire():
            raise CliError(
                f"another '{name}' instance is already running (lock '{name}' is held)",
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


def enqueue_heavy(
    services: Services, specs: Sequence[JobSpec], *, foreground: bool = False
) -> list[str]:
    """Enqueue HEAVY work; with ``foreground`` run it now if the application is stopped."""
    from twin.llm.runtime import activate_offpeak_policy
    from twin.ops.jobs import JobQueue, Worker, default_registry, load_handlers

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
        typer.echo(f"queued {len(ids)} job(s); the running application will execute them")
        return ids
    if app_is_running(services):
        typer.echo("the application is running and will execute the queued job(s)")
        return ids
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
    summary = asyncio.run(_run_foreground(worker))
    typer.echo(f"finished: done={summary.done} retried={summary.retried} failed={summary.failed}")
    return ids


async def _run_foreground(worker: Any) -> Any:
    await worker.recover()
    return await worker.run_until_idle()
