"""Post-import hooks (R-IMP-011): what happens after chat records have been imported.

Every later round that derives data from the records (profile, routine, persona card,
sticker tagging, retrieval index, memory replay, retraining check, ...) registers one
hook here.  After an import, :func:`run_hooks` calls them in registration order and the
import report lists each result.

Registration requires the name of the **backfill command** (``backfill_command``): the
``twin`` sub-command a person can run to do the same work for data that is already
imported, for example ``"images caption-backfill"``.  The argument has no default, and a
test checks that every registered command exists in the CLI.

How a round registers its hook::

    from twin.ingest.hooks import HookContext, HookResult, post_import_hook

    @post_import_hook("profile", backfill_command="profile rebuild",
                      description="recompute the style profile")
    def queue_profile(context: HookContext) -> HookResult:
        ...enqueue a job...
        return HookResult("queued", "profile rebuild queued", jobs=1)

and adds the module's dotted name to :data:`HOOK_MODULES` so it is loaded before hooks run.
Hooks run in the order of that tuple, whatever the import order was.
A hook only *queues* work (jobs) or does small database updates; it must be safe to run
again, because ``--resume`` and re-imports call it again.  Its ``detail`` text is shown in
the report and must not contain message content.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from twin.ops.logging import get_logger

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.ingest.hooks")

HookStatus = Literal["queued", "done", "skipped", "failed"]

# Modules that register hooks on import; each later round appends its module here.
HOOK_MODULES: tuple[str, ...] = (
    "twin.ingest.builtin_hooks",
    "twin.profile.hook",
    "twin.retrieval.hook",
    "twin.profile.persona.hook",
    "twin.stickers.hook",
)


class HookRegistrationError(ValueError):
    """A hook was registered incorrectly (missing backfill command, duplicate name)."""


@dataclass(frozen=True)
class HookResult:
    status: HookStatus
    detail: str
    jobs: int = 0


@dataclass(frozen=True)
class HookContext:
    """What a hook may know about the import that just finished."""

    services: Services
    run_id: str
    conversation_id: str
    export_id: str | None
    inserted: int  # messages added by this run
    changed: int  # messages replaced by this run (conflicts)
    first_import: bool  # no earlier import of this conversation was finished


class PostImportHook(Protocol):
    def __call__(self, context: HookContext) -> HookResult: ...


@dataclass(frozen=True)
class RegisteredHook:
    name: str
    backfill_command: str
    run: PostImportHook
    description: str = ""


@dataclass
class HookOutcome:
    name: str
    backfill_command: str
    result: HookResult


@dataclass
class PostImportHooks:
    """Registry of hooks, in registration order."""

    _hooks: dict[str, RegisteredHook] = field(default_factory=dict)

    def register(
        self,
        name: str,
        run: PostImportHook,
        *,
        backfill_command: str,
        description: str = "",
    ) -> PostImportHook:
        """Add a hook; ``backfill_command`` (e.g. ``"stickers download"``) is required."""
        command = " ".join(backfill_command.split())
        if not name.strip():
            raise HookRegistrationError("a hook needs a name")
        if not command or command.startswith("twin "):
            raise HookRegistrationError(
                f"hook {name!r}: backfill_command must be the CLI path after 'twin', "
                "for example 'images caption-backfill'"
            )
        if name in self._hooks:
            raise HookRegistrationError(f"a hook named {name!r} is already registered")
        self._hooks[name] = RegisteredHook(name, command, run, description)
        return run

    def hooks(self) -> tuple[RegisteredHook, ...]:
        """The hooks in running order: by hook module (``HOOK_MODULES``), then registration.

        The order of registration alone would depend on which module happened to be imported
        first (a test importing ``twin.profile.hook`` before the others, for example); ranking
        by ``HOOK_MODULES`` makes the order the same in every process.  A hook registered from a
        module that is not listed there runs after the listed ones, in registration order.
        """
        unlisted = len(HOOK_MODULES)

        def rank(hook: RegisteredHook) -> int:
            module = getattr(hook.run, "__module__", "")
            return HOOK_MODULES.index(module) if module in HOOK_MODULES else unlisted

        return tuple(sorted(self._hooks.values(), key=rank))

    def names(self) -> tuple[str, ...]:
        return tuple(hook.name for hook in self.hooks())

    def missing_commands(self, available: Iterable[str]) -> list[str]:
        """Hooks whose backfill command is not in ``available`` (names of CLI commands)."""
        known = set(available)
        return [h.name for h in self._hooks.values() if h.backfill_command not in known]

    def clear(self) -> None:
        self._hooks.clear()


default_hooks = PostImportHooks()


def post_import_hook(
    name: str, *, backfill_command: str, description: str = ""
) -> Callable[[PostImportHook], PostImportHook]:
    """Decorator registering a function in :data:`default_hooks`."""

    def decorate(func: PostImportHook) -> PostImportHook:
        return default_hooks.register(
            name, func, backfill_command=backfill_command, description=description
        )

    return decorate


def load_hooks(modules: Iterable[str] | None = None) -> PostImportHooks:
    """Import every hook module so that :data:`default_hooks` is complete."""
    for name in modules if modules is not None else HOOK_MODULES:
        importlib.import_module(name)
    return default_hooks


def run_hooks(
    context: HookContext,
    registry: PostImportHooks | None = None,
    *,
    skip: Iterable[str] = (),
    on_result: Callable[[HookOutcome], None] | None = None,
) -> list[HookOutcome]:
    """Run the registered hooks in order; a failing hook never stops the others."""
    skipped = set(skip)
    outcomes: list[HookOutcome] = []
    for hook in (registry or default_hooks).hooks():
        if hook.name in skipped:
            continue
        try:
            result = hook.run(context)
        except Exception as exc:
            log.error("post_import_hook_failed", hook=hook.name, error=type(exc).__name__)
            result = HookResult("failed", f"{type(exc).__name__}; see the log")
        outcome = HookOutcome(hook.name, hook.backfill_command, result)
        outcomes.append(outcome)
        if on_result is not None:
            on_result(outcome)
    return outcomes
