"""Post-import hooks (R-IMP-011): registration, execution and the backfill commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import update

from tests.support.ingest import make_export, run_import
from twin.cli import app
from twin.ingest import hooks as hooks_module
from twin.ingest.hooks import (
    HOOK_MODULES,
    HookContext,
    HookOutcome,
    HookRegistrationError,
    HookResult,
    PostImportHooks,
    default_hooks,
    load_hooks,
    post_import_hook,
    run_hooks,
)
from twin.ingest.importer import ImportRunner
from twin.ingest.runs import get_run
from twin.ops.process_model import iter_commands
from twin.services import Services
from twin.storage.chat_models import ImportRun


def noop(context: HookContext) -> HookResult:
    return HookResult("done", "ok")


def context_for(services: Services) -> HookContext:
    return HookContext(
        services=services,
        run_id="r",
        conversation_id="c",
        export_id="e",
        inserted=1,
        changed=0,
        first_import=True,
    )


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> PostImportHooks:
    """The default registry (with the built-in hooks) that a test may add hooks to."""
    load_hooks()
    registry = PostImportHooks()
    for hook in default_hooks.hooks():
        registry.register(
            hook.name,
            hook.run,
            backfill_command=hook.backfill_command,
            description=hook.description,
        )
    monkeypatch.setattr(hooks_module, "default_hooks", registry)
    return registry


# ------------------------------------------------------------------ registration


def test_hooks_run_in_the_order_of_the_hook_modules_whatever_was_imported_first() -> None:
    def later(context: HookContext) -> HookResult:
        return HookResult("done", "later")

    def earlier(context: HookContext) -> HookResult:
        return HookResult("done", "earlier")

    def elsewhere(context: HookContext) -> HookResult:
        return HookResult("done", "elsewhere")

    later.__module__ = "twin.profile.hook"
    earlier.__module__ = "twin.ingest.builtin_hooks"
    registry = PostImportHooks()
    registry.register("elsewhere", elsewhere, backfill_command="x run")
    registry.register("later", later, backfill_command="p run")
    registry.register("earlier", earlier, backfill_command="b run")
    assert registry.names() == ("earlier", "later", "elsewhere")
    assert [h.name for h in registry.hooks()] == ["earlier", "later", "elsewhere"]


def test_the_backfill_command_is_a_required_argument() -> None:
    registry = PostImportHooks()
    with pytest.raises(TypeError, match="backfill_command"):
        registry.register("profile", noop)  # type: ignore[call-arg]
    registry.register("profile", noop, backfill_command="profile rebuild", description="d")
    assert registry.names() == ("profile",)
    hook = registry.hooks()[0]
    assert (hook.backfill_command, hook.description) == ("profile rebuild", "d")


@pytest.mark.parametrize("command", ["", "   ", "twin profile rebuild"])
def test_a_missing_or_malformed_backfill_command_is_refused(command: str) -> None:
    with pytest.raises(HookRegistrationError, match="backfill_command"):
        PostImportHooks().register("profile", noop, backfill_command=command)


def test_names_must_be_unique_and_not_blank() -> None:
    registry = PostImportHooks()
    registry.register("a", noop, backfill_command="x y")
    with pytest.raises(HookRegistrationError, match="already registered"):
        registry.register("a", noop, backfill_command="x y")
    with pytest.raises(HookRegistrationError, match="needs a name"):
        registry.register("  ", noop, backfill_command="x y")


def test_the_decorator_registers_in_the_default_registry(
    isolated_registry: PostImportHooks,
) -> None:
    @post_import_hook("late", backfill_command="late rebuild")
    def late(context: HookContext) -> HookResult:
        return HookResult("skipped", "nothing to do")

    assert isolated_registry.names()[-1] == "late"
    assert late(context_for(None)).status == "skipped"  # type: ignore[arg-type]


def test_hooks_run_in_registration_order() -> None:
    registry = PostImportHooks()
    calls: list[str] = []
    for name in ("first", "second", "third"):
        registry.register(
            name,
            lambda context, n=name: calls.append(n) or HookResult("done", n),  # type: ignore[misc,return-value]
            backfill_command=f"{name} run",
        )
    outcomes = run_hooks(context_for(None), registry)  # type: ignore[arg-type]
    assert calls == ["first", "second", "third"]
    assert [o.name for o in outcomes] == calls


def test_a_failing_hook_does_not_stop_the_others_and_is_reported() -> None:
    registry = PostImportHooks()

    def broken(context: HookContext) -> HookResult:
        raise RuntimeError("could not queue")

    registry.register("broken", broken, backfill_command="broken run")
    registry.register("fine", noop, backfill_command="fine run")
    seen: list[HookOutcome] = []
    outcomes = run_hooks(context_for(None), registry, on_result=seen.append)  # type: ignore[arg-type]
    assert [o.result.status for o in outcomes] == ["failed", "done"]
    assert (
        "RuntimeError" in outcomes[0].result.detail
        and "could not queue" not in outcomes[0].result.detail
    )
    assert seen == outcomes


def test_hooks_that_are_already_done_can_be_skipped() -> None:
    registry = PostImportHooks()
    registry.register("a", noop, backfill_command="a run")
    registry.register("b", noop, backfill_command="b run")
    assert [o.name for o in run_hooks(context_for(None), registry, skip=["a"])] == ["b"]  # type: ignore[arg-type]


def test_missing_backfill_commands_are_listed() -> None:
    registry = PostImportHooks()
    registry.register("a", noop, backfill_command="a run")
    registry.register("b", noop, backfill_command="b run")
    assert registry.missing_commands(["a run", "other"]) == ["b"]
    registry.clear()
    assert registry.names() == ()


# ---------------------------------------------------- this round's hooks and the CLI


def test_every_registered_hook_has_a_backfill_command_that_exists() -> None:
    registry = load_hooks()
    commands = {name for name, _ in iter_commands(app)}
    assert registry.missing_commands(commands) == []
    assert registry.names()[:3] == ("image_caption", "sticker_download", "profile")
    by_name = {h.name: h.backfill_command for h in registry.hooks()}
    assert by_name["image_caption"] == "images caption-backfill"
    assert by_name["sticker_download"] == "stickers download"
    assert by_name["profile"] == "profile rebuild"
    assert "twin.profile.hook" in HOOK_MODULES
    assert "twin.ingest.builtin_hooks" in HOOK_MODULES


# ------------------------------------------------------------- in an import run


def test_hooks_run_after_the_import_and_are_listed_with_their_results(
    services: Services, tmp_path: Path, isolated_registry: PostImportHooks
) -> None:
    contexts: list[HookContext] = []

    @post_import_hook("recorder", backfill_command="recorder run")
    def recorder(context: HookContext) -> HookResult:
        contexts.append(context)
        return HookResult("done", "recorded", jobs=0)

    export = make_export(tmp_path, target_messages=80)
    outcome = run_import(services, export)
    assert list(outcome.run.hooks) == [
        "image_caption",
        "sticker_download",
        "profile",
        "retrieval",
        "persona",
        "sticker_tag",
        "recorder",
    ]
    assert outcome.run.hooks["recorder"] == {
        "name": "recorder",
        "status": "done",
        "detail": "recorded",
        "jobs": 0,
        "backfill_command": "recorder run",
    }
    (context,) = contexts
    assert (context.inserted, context.changed, context.first_import) == (80, 0, True)
    assert context.run_id == outcome.run.id and context.export_id == "synthetic-export-0001"
    report = Path(outcome.report_path or "").read_text(encoding="utf-8")
    assert "recorder" in report and "`twin recorder run`" in report

    again = run_import(services, export)
    assert contexts[-1].first_import is False and contexts[-1].inserted == 0
    assert again.run.hooks["recorder"]["status"] == "done"


def test_a_failing_hook_does_not_fail_the_import(
    services: Services, tmp_path: Path, isolated_registry: PostImportHooks
) -> None:
    @post_import_hook("explodes", backfill_command="explodes run")
    def explodes(context: HookContext) -> HookResult:
        raise ValueError("secret detail")

    outcome = run_import(services, make_export(tmp_path, target_messages=20))
    assert outcome.status == "done"
    entry = outcome.run.hooks["explodes"]
    assert entry["status"] == "failed" and "secret detail" not in entry["detail"]


def test_a_resumed_run_skips_hooks_that_finished_and_repeats_the_others(
    services: Services, tmp_path: Path, isolated_registry: PostImportHooks
) -> None:
    calls: list[str] = []

    @post_import_hook("counted", backfill_command="counted run")
    def counted(context: HookContext) -> HookResult:
        calls.append("counted")
        return HookResult("done", "once")

    @post_import_hook("queues", backfill_command="queues run")
    def queues(context: HookContext) -> HookResult:
        calls.append("queues")
        return HookResult("queued", "again")

    export = make_export(tmp_path, target_messages=20)
    outcome = run_import(services, export)
    assert calls == ["counted", "queues"]
    with services.db.transaction() as session:  # the process died while the hooks were running
        session.execute(
            update(ImportRun)
            .where(ImportRun.id == outcome.run.id)
            .values(status="running", phase="hooks")
        )
    ImportRunner(services).run(outcome.run.id)
    assert calls == ["counted", "queues", "queues"]
    final = get_run(services.db, outcome.run.id)
    assert final is not None and final.status == "done"
