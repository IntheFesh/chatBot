"""CLI process model (R-ARCH-006): command classes, locks, state_version, heavy jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pytest
import typer
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from twin.cli import app
from twin.config.secrets import SecretStore
from twin.ops.instance_lock import LOCK_RUN, LOCK_SUPERVISOR, InstanceLock
from twin.ops.jobs import HandlerRegistry, JobContext, JobQueue
from twin.ops.process_model import (
    CliError,
    CommandKind,
    CommandSpec,
    ExitCode,
    JobSpec,
    _guard_locks,
    app_is_running,
    command,
    enqueue_heavy,
    get_spec,
    iter_commands,
    undeclared_commands,
)
from twin.services import CliContext, Services, set_cli_context
from twin.storage.state import read_state_version

runner = CliRunner()


def expected_names() -> list[str]:
    return [
        "run",
        "doctor",
        "config show",
        "settings list",
        "settings set",
        "settings history",
        "secrets set",
        "secrets delete",
        "secrets list",
        "secrets check",
        "secrets rotate-db-key",
        "db upgrade",
        "db status",
        "jobs list",
        "jobs show",
        "jobs retry",
        "jobs cancel",
        "jobs run",
        "jobs approve",
        "llm probe",
        "llm status",
        "channel login",
        "channel status",
        "channel send-test",
        "channel unbind",
        "channel listen",
        "channel echo-test",
        "channel probe start",
        "channel probe status",
        "channel probe answer",
        "channel probe stop",
        "channel probe report",
        "chat",
        "import start",
        "import status",
        "import inspect",
        "images caption-backfill",
        "stickers download",
        "profile rebuild",
        "profile history",
        "profile show",
        "profile phrases",
        "profile diff",
        "profile rollback",
        "routine list",
        "routine add sleep",
        "routine add busy",
        "routine add holiday",
        "routine remove",
        "routine enable",
        "routine disable",
        "retrieval rebuild",
        "retrieval stats",
        "retrieval resplit",
        "persona show",
        "persona history",
        "persona diff",
        "persona rollback",
        "persona edit",
        "persona regenerate",
        "persona status",
        "persona templates",
        "persona rules list",
        "persona rules delete",
        "persona rules consolidate",
        "stickers list",
        "stickers show",
        "stickers tag",
        "stickers untag",
        "stickers disable",
        "stickers enable",
        "stickers tag-all",
        "memory replay estimate",
        "memory replay start",
        "memory replay status",
        "memory list",
        "memory block",
        "memory remember",
        "memory forget",
        "memory summarize",
        "memory reindex",
        "timezone show",
        "timezone set",
        "timezone history",
        "plan show",
        "plan rebuild",
        "plan lifeline",
        "plan lifeline-generate",
        "train bundle",
        "train export",
        "train export-status",
        "train export-dpo",
        "train retrain-check",
        "train remote connect",
        "train remote upload",
        "train remote setup",
        "train remote train",
        "train remote dpo",
        "train remote eval",
        "train remote export",
        "train remote download",
        "train remote cleanup",
        "train remote status",
        "train remote all",
        "model register",
        "model list",
        "model show",
        "eval blind",
        "eval style",
        "eval memory",
        "eval gate",
        "eval runs",
        "supervise",
        "setup",
        "health",
        "purge",
        "eval stability",
        "service install",
        "service uninstall",
        "service start",
        "service stop",
        "service status",
        "cost report",
        "backup now",
        "backup list",
        "backup verify",
        "backup restore",
        "rollback profile",
        "rollback persona",
        "rollback prompt-template",
        "rollback style-model",
        "ops drill network",
        "eval proactive",
        "proactive log",
        # round 14: serving the style model
        "model activate",
        "model disable",
        "model serve",
        "model verify",
        "model recommend",
        "model evaluate",
        "model tunnel start",
        "model tunnel stop",
        "model tunnel status",
    ]


def test_every_cli_command_declares_its_process_model_class() -> None:
    assert undeclared_commands(app) == []
    names = dict(iter_commands(app))
    assert len(names) == 143 == len(expected_names())
    expected = {
        "run": CommandKind.EXCLUSIVE,
        "doctor": CommandKind.READ,
        "config show": CommandKind.READ,
        "settings list": CommandKind.READ,
        "settings set": CommandKind.LIGHT,
        "secrets set": CommandKind.LIGHT,
        "secrets delete": CommandKind.LIGHT,
        "secrets list": CommandKind.READ,
        "secrets check": CommandKind.READ,
        "secrets rotate-db-key": CommandKind.EXCLUSIVE,
        "db upgrade": CommandKind.EXCLUSIVE,
        "db status": CommandKind.READ,
        "jobs list": CommandKind.READ,
        "jobs show": CommandKind.READ,
        "jobs retry": CommandKind.LIGHT,
        "jobs cancel": CommandKind.LIGHT,
        "jobs approve": CommandKind.LIGHT,
        "jobs run": CommandKind.HEAVY,
        "llm probe": CommandKind.LIGHT,
        "llm status": CommandKind.READ,
        "channel login": CommandKind.LIGHT,
        "channel status": CommandKind.READ,
        "channel send-test": CommandKind.LIGHT,
        "channel unbind": CommandKind.LIGHT,
        "channel listen": CommandKind.EXCLUSIVE,
        "channel echo-test": CommandKind.EXCLUSIVE,
        "channel probe start": CommandKind.LIGHT,
        "channel probe status": CommandKind.READ,
        "channel probe answer": CommandKind.LIGHT,
        "channel probe stop": CommandKind.LIGHT,
        "channel probe report": CommandKind.LIGHT,
        "chat": CommandKind.EXCLUSIVE,
        "import start": CommandKind.HEAVY,
        "import status": CommandKind.READ,
        "import inspect": CommandKind.READ,
        "images caption-backfill": CommandKind.HEAVY,
        "stickers download": CommandKind.HEAVY,
        "profile rebuild": CommandKind.HEAVY,
        "profile history": CommandKind.READ,
        "profile show": CommandKind.READ,
        "profile phrases": CommandKind.READ,
        "profile diff": CommandKind.READ,
        "profile rollback": CommandKind.LIGHT,
        "routine list": CommandKind.READ,
        "routine add sleep": CommandKind.LIGHT,
        "routine add busy": CommandKind.LIGHT,
        "routine add holiday": CommandKind.LIGHT,
        "routine remove": CommandKind.LIGHT,
        "routine enable": CommandKind.LIGHT,
        "routine disable": CommandKind.LIGHT,
        "retrieval rebuild": CommandKind.HEAVY,
        "retrieval stats": CommandKind.READ,
        "retrieval resplit": CommandKind.HEAVY,
        "persona show": CommandKind.READ,
        "persona history": CommandKind.READ,
        "persona diff": CommandKind.READ,
        "persona rollback": CommandKind.LIGHT,
        "persona edit": CommandKind.LIGHT,
        "persona regenerate": CommandKind.HEAVY,
        "persona status": CommandKind.READ,
        "persona templates": CommandKind.READ,
        "persona rules list": CommandKind.READ,
        "persona rules delete": CommandKind.LIGHT,
        "persona rules consolidate": CommandKind.HEAVY,
        "stickers list": CommandKind.READ,
        "stickers show": CommandKind.READ,
        "stickers tag": CommandKind.LIGHT,
        "stickers untag": CommandKind.LIGHT,
        "stickers disable": CommandKind.LIGHT,
        "stickers enable": CommandKind.LIGHT,
        "stickers tag-all": CommandKind.HEAVY,
        "memory replay estimate": CommandKind.READ,
        "memory replay start": CommandKind.HEAVY,
        "memory replay status": CommandKind.READ,
        "memory list": CommandKind.READ,
        "memory block": CommandKind.READ,
        "memory remember": CommandKind.LIGHT,
        "memory forget": CommandKind.LIGHT,
        "memory summarize": CommandKind.HEAVY,
        "memory reindex": CommandKind.LIGHT,
        "timezone show": CommandKind.READ,
        "timezone set": CommandKind.LIGHT,
        "timezone history": CommandKind.READ,
        "plan show": CommandKind.READ,
        "plan rebuild": CommandKind.LIGHT,
        "plan lifeline": CommandKind.READ,
        "plan lifeline-generate": CommandKind.HEAVY,
        # the remote commands write short records only; the work happens on the instance
        "train bundle": CommandKind.LIGHT,
        "train export": CommandKind.HEAVY,
        "train export-status": CommandKind.READ,
        "train export-dpo": CommandKind.LIGHT,
        "train retrain-check": CommandKind.LIGHT,
        "train remote connect": CommandKind.LIGHT,
        "train remote upload": CommandKind.LIGHT,
        "train remote setup": CommandKind.LIGHT,
        "train remote train": CommandKind.LIGHT,
        "train remote dpo": CommandKind.LIGHT,
        "train remote eval": CommandKind.LIGHT,
        "train remote export": CommandKind.LIGHT,
        "train remote download": CommandKind.LIGHT,
        "train remote cleanup": CommandKind.LIGHT,
        "train remote status": CommandKind.READ,
        "train remote all": CommandKind.LIGHT,
        "model register": CommandKind.LIGHT,
        "model list": CommandKind.READ,
        "model show": CommandKind.READ,
        "eval blind": CommandKind.LIGHT,
        "eval style": CommandKind.LIGHT,
        "eval memory": CommandKind.LIGHT,
        "eval gate": CommandKind.LIGHT,
        "eval runs": CommandKind.READ,
        # round 12: residency, backups, deletion and rollbacks
        "supervise": CommandKind.EXCLUSIVE,
        "setup": CommandKind.LIGHT,
        "health": CommandKind.READ,
        "purge": CommandKind.EXCLUSIVE,
        "eval stability": CommandKind.LIGHT,
        "service install": CommandKind.EXCLUSIVE,
        "service uninstall": CommandKind.EXCLUSIVE,
        "service start": CommandKind.LIGHT,
        "service stop": CommandKind.LIGHT,
        "service status": CommandKind.READ,
        "cost report": CommandKind.READ,
        "backup now": CommandKind.LIGHT,
        "backup list": CommandKind.READ,
        "backup verify": CommandKind.READ,
        "backup restore": CommandKind.EXCLUSIVE,
        "rollback profile": CommandKind.LIGHT,
        "rollback persona": CommandKind.LIGHT,
        "rollback prompt-template": CommandKind.LIGHT,
        "rollback style-model": CommandKind.LIGHT,
        "ops drill network": CommandKind.READ,
        # round 10: the proactive messages
        "eval proactive": CommandKind.LIGHT,
        "proactive log": CommandKind.READ,
        # round 14: they write short records; serve, verify and a foreground tunnel hold a
        # server or a tunnel in their own window (serve refuses to run beside the application,
        # tunnel start leaves the tunnel to it, verify uses the port of the evaluation)
        "model activate": CommandKind.LIGHT,
        "model disable": CommandKind.LIGHT,
        "model serve": CommandKind.LIGHT,
        "model verify": CommandKind.LIGHT,
        "model recommend": CommandKind.READ,
        "model evaluate": CommandKind.LIGHT,
        "model tunnel start": CommandKind.LIGHT,
        "model tunnel stop": CommandKind.LIGHT,
        "model tunnel status": CommandKind.READ,
    }
    for name, kind in expected.items():
        spec = get_spec(names[name])
        assert spec is not None and spec.kind is kind, name
    run_spec = get_spec(names["run"])
    assert run_spec is not None
    assert run_spec.acquires == (LOCK_RUN,) and run_spec.tolerates == (LOCK_SUPERVISOR,)
    supervise_spec = get_spec(names["supervise"])  # the supervisor holds its own lock for good
    assert supervise_spec is not None and supervise_spec.kind is CommandKind.EXCLUSIVE
    assert supervise_spec.acquires == (LOCK_SUPERVISOR,) and supervise_spec.tolerates == (LOCK_RUN,)
    for polling in ("channel listen", "channel echo-test", "chat"):  # they must exclude `twin run`
        spec = get_spec(names[polling])
        assert spec is not None and spec.acquires == (LOCK_RUN,), polling


def test_the_check_detects_an_undeclared_command() -> None:
    sample = typer.Typer()

    @sample.command("declared")
    @command(CommandKind.READ, consent=False)
    def declared() -> None: ...

    @sample.command("forgotten")
    def forgotten() -> None: ...

    @sample.command()
    def implicit_name() -> None: ...

    inner = typer.Typer()
    sample.add_typer(inner, name="nested")

    @inner.command("deep")
    def deep() -> None: ...

    assert undeclared_commands(sample) == ["forgotten", "implicit-name", "nested deep"]


# ------------------------------------------------------------------- locks


@pytest.fixture
def home_services(services: Services, secret_store: SecretStore) -> Services:
    """CLI context pointing at the same data directory as ``services``."""
    context = CliContext(
        overrides={"paths": {"data_dir": str(services.paths.data_dir)}}, secrets=secret_store
    )
    set_cli_context(context)
    return services


def hold(services: Services, name: str) -> InstanceLock:
    lock = InstanceLock(name, locks_dir=services.paths.locks_dir)
    assert lock.acquire()
    return lock


@pytest.mark.parametrize("held", [LOCK_RUN, LOCK_SUPERVISOR])
def test_exclusive_commands_refuse_while_either_instance_lock_is_held(
    home_services: Services, held: str
) -> None:
    lock = hold(home_services, held)
    try:
        result = runner.invoke(
            app, ["--set", f"paths.data_dir={home_services.paths.data_dir}", "db", "upgrade"]
        )
    finally:
        lock.release()
    assert result.exit_code == ExitCode.BUSY
    assert f"'{held}' instance is running" in result.output
    assert "stop the application first" in result.output


def test_exclusive_commands_run_when_no_instance_is_running(home_services: Services) -> None:
    result = runner.invoke(
        app, ["--set", f"paths.data_dir={home_services.paths.data_dir}", "db", "upgrade"]
    )
    assert result.exit_code == 0, result.output
    assert not app_is_running(home_services)


def test_exclusive_command_releases_locks_it_acquired(home_services: Services) -> None:
    spec = CommandSpec(CommandKind.EXCLUSIVE, (LOCK_RUN,), (LOCK_SUPERVISOR,), True)
    from contextlib import ExitStack

    locks_dir = home_services.paths.locks_dir
    with ExitStack() as stack:
        _guard_locks(stack, spec, locks_dir)
        assert InstanceLock(LOCK_RUN, locks_dir=locks_dir).is_held_elsewhere()
    assert not InstanceLock(LOCK_RUN, locks_dir=locks_dir).is_held_elsewhere()


def test_twin_run_tolerates_the_supervisor_lock_but_not_a_second_run(
    home_services: Services,
) -> None:
    from contextlib import ExitStack

    spec = CommandSpec(CommandKind.EXCLUSIVE, (LOCK_RUN,), (LOCK_SUPERVISOR,), True)
    locks_dir = home_services.paths.locks_dir
    supervisor = hold(home_services, LOCK_SUPERVISOR)  # `twin supervise` started this `twin run`
    try:
        with ExitStack() as stack:
            _guard_locks(stack, spec, locks_dir)  # must not raise
    finally:
        supervisor.release()
    other_run = hold(home_services, LOCK_RUN)
    try:
        with ExitStack() as stack, pytest.raises(CliError, match="already running") as info:
            _guard_locks(stack, spec, locks_dir)
        assert info.value.code is ExitCode.BUSY
    finally:
        other_run.release()


def test_non_exclusive_commands_work_while_the_application_runs(home_services: Services) -> None:
    lock = hold(home_services, LOCK_RUN)
    try:
        read = runner.invoke(
            app, ["--set", f"paths.data_dir={home_services.paths.data_dir}", "jobs", "list"]
        )
        light = runner.invoke(
            app,
            [
                "--set",
                f"paths.data_dir={home_services.paths.data_dir}",
                "settings",
                "set",
                "paused",
                "true",
            ],
        )
    finally:
        lock.release()
    assert read.exit_code == 0, read.output
    assert light.exit_code == 0, light.output


# ----------------------------------------------------- READ / LIGHT policies


def custom_app(services: Services, secret_store: SecretStore) -> typer.Typer:
    sample = typer.Typer()

    @sample.command("read-that-writes")
    @command(CommandKind.READ, consent=False)
    def read_that_writes() -> None:
        with services.db.transaction():
            pass

    @sample.command("light-write")
    @command(CommandKind.LIGHT, consent=False)
    def light_write() -> None:
        with services.db.transaction():
            pass

    @sample.command("light-no-write")
    @command(CommandKind.LIGHT, consent=False)
    def light_no_write() -> None: ...

    @sample.command("heavy")
    @command(CommandKind.HEAVY, consent=False)
    def heavy() -> None:
        with services.db.transaction():
            pass

    @sample.command("explicit")
    @command(CommandKind.LIGHT, consent=False)
    def explicit(flag: Annotated[bool, typer.Option()] = False) -> None:
        with services.db.transaction(bump_state=flag):
            pass

    set_cli_context(
        CliContext(
            overrides={"paths": {"data_dir": str(services.paths.data_dir)}}, secrets=secret_store
        )
    )
    return sample


def version(services: Services) -> int:
    with services.db.session() as session:
        return read_state_version(session)


def test_a_read_command_cannot_write(services: Services, secret_store: SecretStore) -> None:
    sample = custom_app(services, secret_store)
    result = runner.invoke(sample, ["read-that-writes"])
    assert result.exit_code == 1
    assert "declared read-only" in result.output
    assert version(services) == 0


def test_light_commands_bump_state_version_with_their_writes(
    services: Services, secret_store: SecretStore
) -> None:
    sample = custom_app(services, secret_store)
    assert runner.invoke(sample, ["light-write"]).exit_code == 0
    assert version(services) == 1
    assert runner.invoke(sample, ["light-no-write"]).exit_code == 0
    assert version(services) == 1  # no write, no bump
    assert runner.invoke(sample, ["heavy"]).exit_code == 0
    assert version(services) == 1  # HEAVY writes do not bump
    assert runner.invoke(sample, ["explicit", "--flag"]).exit_code == 0
    assert version(services) == 2
    assert runner.invoke(sample, ["explicit"]).exit_code == 0
    assert version(services) == 2  # explicit bump_state=False overrides the LIGHT default


# ----------------------------------------------------- consent and errors


def test_commands_refuse_to_run_without_valid_consent(home_services: Services) -> None:
    base = ["--set", f"paths.data_dir={home_services.paths.data_dir}"]
    refused = runner.invoke(app, [*base, "--set", "consent.confirmed_at=null", "jobs", "list"])
    assert refused.exit_code == ExitCode.CONSENT
    assert "consent.confirmed_at is missing" in refused.output
    invalid = runner.invoke(app, [*base, "--set", "consent.confirmed_at=someday", "jobs", "list"])
    assert invalid.exit_code == ExitCode.CONSENT and "not a valid date" in invalid.output
    exempt = runner.invoke(app, [*base, "--set", "consent.confirmed_at=null", "db", "status"])
    assert exempt.exit_code == 0  # administrative commands that touch no chat data


def test_operational_errors_become_short_messages_with_distinct_exit_codes(
    home_services: Services,
) -> None:
    missing_db = runner.invoke(
        app,
        ["--set", "paths.data_dir=" + str(home_services.paths.data_dir / "other"), "jobs", "list"],
    )
    assert missing_db.exit_code == ExitCode.SCHEMA
    assert "twin db upgrade" in missing_db.output and "Traceback" not in missing_db.output
    bad_config = runner.invoke(
        app, ["--config", str(home_services.paths.data_dir / "no.yaml"), "jobs", "list"]
    )
    assert bad_config.exit_code == ExitCode.CONFIG
    bad_override = runner.invoke(app, ["--set", "nonsense", "jobs", "list"])
    assert bad_override.exit_code == ExitCode.CONFIG


# ---------------------------------------------------------- HEAVY commands


@pytest.fixture
def heavy_registry(monkeypatch: pytest.MonkeyPatch) -> HandlerRegistry:
    registry = HandlerRegistry()
    monkeypatch.setattr("twin.ops.jobs.default_registry", registry)
    monkeypatch.setattr("twin.ops.jobs.load_handlers", lambda modules=None: registry)
    return registry


def test_heavy_work_is_only_queued_by_default(
    services: Services, heavy_registry: HandlerRegistry, capsys: pytest.CaptureFixture[str]
) -> None:
    ran: list[int] = []

    async def handler(ctx: JobContext) -> None:
        ran.append(ctx.job.payload["n"])

    heavy_registry.register("heavy_task", handler)
    ids = enqueue_heavy(
        services, [JobSpec("heavy_task", {"n": 1}), JobSpec("heavy_task", {"n": 2})]
    )
    assert len(ids) == 2 and ran == []
    assert "queued 2 job(s)" in capsys.readouterr().out
    assert JobQueue(services.db, services.clock).counts()["pending"] == 2


def test_foreground_heavy_work_runs_with_the_same_worker_when_the_app_is_stopped(
    services: Services, heavy_registry: HandlerRegistry, capsys: pytest.CaptureFixture[str]
) -> None:
    ran: list[int] = []

    async def handler(ctx: JobContext) -> None:
        ran.append(ctx.job.payload["n"])

    heavy_registry.register("heavy_task", handler)
    enqueue_heavy(services, [JobSpec("heavy_task", {"n": 7}, priority=5)], foreground=True)
    assert ran == [7]
    assert "done=1" in capsys.readouterr().out


def test_foreground_heavy_work_leaves_execution_to_a_running_application(
    services: Services, heavy_registry: HandlerRegistry, capsys: pytest.CaptureFixture[str]
) -> None:
    async def handler(ctx: JobContext) -> None:
        raise AssertionError("must not run in the CLI process")

    heavy_registry.register("heavy_task", handler)
    lock = hold(services, LOCK_RUN)
    try:
        enqueue_heavy(services, [JobSpec("heavy_task", {"n": 1})], foreground=True)
    finally:
        lock.release()
    assert "will execute the queued job" in capsys.readouterr().out
    assert JobQueue(services.db, services.clock).counts()["pending"] == 1


def test_heavy_job_specs_carry_batch_information(
    services: Services, heavy_registry: HandlerRegistry
) -> None:
    enqueue_heavy(
        services,
        [
            JobSpec(
                "heavy_task",
                {},
                batch_id="b1",
                estimated_cost_usd=3.0,
                requires_approval=True,
                offpeak_only=True,
                max_attempts=5,
            )
        ],
    )
    job = JobQueue(services.db, services.clock).list_jobs(batch_id="b1")[0]
    assert (job.estimated_cost_usd, job.requires_approval, job.offpeak_only, job.max_attempts) == (
        3.0,
        True,
        True,
        5,
    )


def test_clock_fixture_is_the_services_clock(services: Services, clock: ManualClock) -> None:
    assert services.clock is clock
    assert Path(services.paths.db_path).exists()
