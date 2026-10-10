"""CLI: ``twin train`` (R-TRN-008, R-TRN-010, R-PRIV-003).

``bundle``
    builds the encrypted training package from an exported dataset directory (asks for a
    passphrase, which is never stored);
``remote ...``
    runs the training on the AutoDL instance from the ``autodl.*`` settings, one step per command
    (``connect``, ``upload``, ``setup``, ``train``, ``dpo``, ``eval``, ``export``, ``download``,
    ``cleanup``) or all of them (``all``); ``status`` lists the runs and reminds of the ones that
    still have data on an instance.

The commands are LIGHT: they write only short records to ``training_runs`` while the work happens
on the instance (R-ARCH-006).

``export``
    builds the training set from her real reply blocks (HEAVY: it queues the ``training_export``
    job; ``--foreground`` runs it here when the application is stopped).  When the plans of the
    hybrid share are missing it queues them as priced batches that wait for
    ``twin jobs approve <batch>`` and ends; run it again afterwards.  ``export-status`` shows how
    the last export ended;
``export-dpo``
    writes the preference pairs of ``/不像 <正确说法>`` (the only reader of ``preference_pairs``) as
    ``dpo_train.jsonl`` of a new dataset version made from the newest exported dataset (LIGHT:
    no model is called);
``retrain-check``
    compares her messages now with the ones the last training covered (R-TRN-012) and raises
    the ``retrain_suggested`` alert when it is time (the post-import hook calls the same code).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from twin.learning.pairs import PreferencePairStore, dpo_hint
from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.services import Services, get_cli_context
from twin.training import bundle_crypto
from twin.training.bundle import BundleError, build_bundle, verify_bundle
from twin.training.dataset_dir import DatasetError, file_sha256, load_dataset_dir
from twin.training.dpo_export import export_dpo
from twin.training.export import ExportError
from twin.training.export_job import (
    EXPORT_JOB,
    ExportRequest,
    handle_training_export,
    local_range,
    queue_export,
    read_state,
)
from twin.training.layout import RemoteLayout
from twin.training.plans import PlanStore
from twin.training.profiles import (
    ProfileError,
    TrainingProfile,
    get_profile,
    profile_names,
)
from twin.training.registry import RegistryError
from twin.training.remote.connection import RemoteError, target_from_settings
from twin.training.remote.session import RemoteSession
from twin.training.remote.steps import RemoteSteps, inspect_instance
from twin.training.retrain import check_retrain
from twin.training.runs import (
    RunError,
    RunStore,
    RunView,
    ensure_dataset_version,
    hyperparameters,
    new_run_id,
)
from twin.training.tokenizer import TokenizerError

train_app = typer.Typer(help="Train the style model on a rented GPU.", no_args_is_help=True)
remote_app = typer.Typer(help="Run the training on the AutoDL instance.", no_args_is_help=True)
train_app.add_typer(remote_app, name="remote")

SIDECAR_SUFFIX = ".json"
POLL_SECONDS = 2.0  # how often a running job is asked for news

PROFILE_OPTION = typer.Option("--profile", help="One of: " + ", ".join(profile_names()))
RUN_OPTION = typer.Option("--run", help="Run id (default: the latest run)")


def _console() -> Console:
    return Console(highlight=False, soft_wrap=True)


@contextmanager
def _failures() -> Iterator[None]:
    """Turn the expected errors of this package into short CLI errors."""
    try:
        yield
    except (
        BundleError,
        DatasetError,
        ExportError,
        ProfileError,
        RegistryError,
        RemoteError,
        RunError,
        TokenizerError,
        bundle_crypto.BundleDecryptionError,
    ) as exc:
        raise CliError(str(exc), ExitCode.FAILURE) from exc


def bundles_dir(services: Services) -> Path:
    return services.paths.data_dir / "training" / "bundles"


def known_hosts_path(services: Services) -> Path:
    return services.paths.data_dir / "training" / "known_hosts"


def sidecar_path(bundle: Path) -> Path:
    return bundle.with_name(bundle.name + SIDECAR_SUFFIX)


def _passphrase_for_new_bundle() -> str:
    while True:
        value = str(
            typer.prompt(
                "Passphrase for the training package (at least "
                f"{bundle_crypto.MIN_PASSPHRASE_CHARS} characters; it is not stored anywhere)",
                hide_input=True,
                confirmation_prompt=True,
            )
        )
        if len(value) >= bundle_crypto.MIN_PASSPHRASE_CHARS:
            return value
        typer.echo(f"the passphrase needs at least {bundle_crypto.MIN_PASSPHRASE_CHARS} characters")


def _passphrase_for_unpacking() -> str:
    return str(typer.prompt("Passphrase of the training package", hide_input=True))


@dataclass(frozen=True)
class BuiltBundle:
    path: Path
    sha256: str
    profile: str
    dataset_version: str
    passphrase: str


def _build(
    services: Services, profile: TrainingProfile, dataset_dir: Path, out: Path | None
) -> BuiltBundle:
    dataset = load_dataset_dir(dataset_dir)
    try:
        relative = dataset_dir.resolve().relative_to(services.paths.data_dir.resolve()).as_posix()
    except ValueError:
        relative = str(dataset_dir.resolve())
    ensure_dataset_version(services.db, dataset, relative)
    passphrase = _passphrase_for_new_bundle()
    result = build_bundle(
        dataset,
        profile,
        passphrase=passphrase,
        out_dir=out or bundles_dir(services),
        created_at=services.clock.now_utc(),
        layout=RemoteLayout(services.settings.autodl.workdir),
        dpo_min_pairs=services.settings.training.dpo_min_pairs,
    )
    verify_bundle(result.path, passphrase)
    sidecar_path(result.path).write_text(
        json.dumps(
            {
                "profile": profile.name,
                "dataset_version": dataset.meta.dataset_version,
                "sha256": result.sha256,
                "size": result.size,
                "created_at": services.clock.now_utc().isoformat(),
                "train_samples": dataset.meta.counts.train,
                "dpo_min_pairs": services.settings.training.dpo_min_pairs,
                "workdir": services.settings.autodl.workdir,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return BuiltBundle(
        result.path, result.sha256, profile.name, dataset.meta.dataset_version, passphrase
    )


@train_app.command("bundle")
@command(CommandKind.LIGHT)
def bundle_command(
    profile: Annotated[str, PROFILE_OPTION],
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Directory of an exported dataset (twin train export)")
    ],
    out: Annotated[Path | None, typer.Option("--out", help="Where to write the package")] = None,
) -> None:
    """Build the encrypted training package for a GPU profile (asks for a passphrase)."""
    services = get_cli_context().services()
    with _failures():
        chosen = get_profile(profile)
        built = _build(services, chosen, dataset, out)
    typer.echo(f"package:  {built.path}")
    typer.echo(f"sha256:   {built.sha256}")
    typer.echo(f"profile:  {chosen.name}  dataset: {built.dataset_version}")
    typer.echo(
        f"decrypt:  {built.path.parent / 'decrypt_bundle.py'} (standard library + cryptography)"
    )
    typer.echo("keep the passphrase: it is needed by `twin train remote setup` and is not stored")


# ----------------------------------------------------------------------------- export

FROM_OPTION = typer.Option("--from", help="First local day (YYYY-MM-DD); default: the first record")
TO_OPTION = typer.Option("--to", help="Last local day (YYYY-MM-DD); default: the last record")


def _day(value: str | None, option: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CliError(f"{option} must be a date like 2026-03-05", ExitCode.USAGE) from exc


def format_state(state: dict[str, Any]) -> list[str]:
    """The lines ``export`` and ``export-status`` print about the last export (no text)."""
    kind = state.get("state")
    lines = [f"export: {kind} ({state.get('at', '?')})"]
    if kind == "failed":
        lines.append(f"  {state.get('message', '')}")
    elif kind == "waiting_for_plans":
        lines.append(
            f"  {state.get('missing', 0)} of {state.get('selected', 0)} planned samples have no "
            "plan yet"
        )
        for batch in state.get("batches", []):
            lines.append(f"  approve with: twin jobs approve {batch}")
        if state.get("batches"):
            lines.append(f"  estimated ${state.get('estimated_usd', 0):.2f} (an upper bound)")
        lines.append(
            "  after the plans are written (`twin jobs run --until-idle` if the application is "
            "stopped) run `twin train export` again"
        )
    elif kind == "done":
        lines.append(
            f"  dataset {state.get('dataset_version')}: {state.get('train')} train, "
            f"{state.get('val')} validation, {state.get('test')} test samples"
        )
        lines.append(f"  in {state.get('directory')}")
        lines.extend(f"  {line}" for line in format_stats(state.get("stats", {})))
        lines.append("  next: twin train bundle --profile <profile> --dataset <that directory>")
    return lines


def format_stats(stats: dict[str, Any]) -> list[str]:
    """The numbers of an export report as short lines."""
    if not stats:
        return []
    sample = stats.get("samples", {})
    tokens = stats.get("tokens", {})
    plans = stats.get("plans", {})
    sticker = stats.get("sticker_share", {})
    codes = stats.get("emoji_code_rate", {})
    chars = stats.get("target_chars", {})
    hours = stats.get("training_hours", {})
    lines = [
        f"samples {sample.get('total')}; dropped {stats.get('dropped', {})}",
        f"turns per sample {stats.get('turns_per_sample', {}).get('mean')}; target length "
        f"p50 {chars.get('p50')} / p90 {chars.get('p90')} / max {chars.get('max')} characters",
        f"stickers {sticker.get('export')} of target lines (her profile {sticker.get('profile')}); "
        f"emoji codes {codes.get('export')} of text lines (profile {codes.get('profile')})",
        f"plans {plans.get('planned')}/{plans.get('selected')} chosen; "
        f"{plans.get('share_of_train_and_val')} of train+val",
        f"tokens {tokens.get('total')} ({tokens.get('train')} in train), "
        f"{stats.get('epochs')} epochs; about "
        + ", ".join(f"{name} {value} h" for name, value in hours.items())
        + " (planning figures)",
    ]
    return lines


async def _export_in_foreground(services: Services) -> None:
    registry = HandlerRegistry()
    registry.register(EXPORT_JOB, handle_training_export)
    summary = await run_jobs_until_idle(services, registry)
    if summary.failed or summary.retried:
        typer.echo(f"the export did not finish: {'; '.join(summary.failures) or 'see the log'}")


@train_app.command("export")
@command(CommandKind.HEAVY)
def export_command(
    first: Annotated[str | None, FROM_OPTION] = None,
    last: Annotated[str | None, TO_OPTION] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the dataset here (default: data/training)")
    ] = None,
    tokenizer: Annotated[
        Path | None,
        typer.Option("--tokenizer", help="tokenizer.json of Qwen3 (default: download once)"),
    ] = None,
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run here when the application is stopped")
    ] = False,
) -> None:
    """Export the training set from her real reply blocks (queued as a job, resumable)."""
    services = get_cli_context().services()
    start, end = _day(first, "--from"), _day(last, "--to")
    if start and end and end < start:
        raise CliError("--to is before --from", ExitCode.USAGE)
    since, until = local_range(services, start, end)
    request = ExportRequest(since, until, out, tokenizer)
    queued = queue_export(services, request)
    verb = "an export is already waiting" if queued.already_queued else "queued"
    typer.echo(f"{verb} (job {queued.job_id})")
    if not foreground:
        typer.echo("the running application executes it; see `twin train export-status`")
        return
    if app_is_running(services):
        typer.echo("the application is running and will execute the job")
        return
    asyncio.run(_export_in_foreground(services))
    state = read_state(services)
    if state is None:
        raise CliError("the export left no record; see `twin jobs list`", ExitCode.FAILURE)
    for line in format_state(state):
        typer.echo(line)
    if state.get("state") == "failed":
        raise CliError("the export failed", ExitCode.FAILURE)


@train_app.command("export-status")
@command(CommandKind.READ)
def export_status_command() -> None:
    """Show how the last export ended and how far the plans of the hybrid share are."""
    services = get_cli_context().services()
    state = read_state(services)
    if state is None:
        typer.echo("no export has run yet: `twin train export`")
    else:
        for line in format_state(state):
            typer.echo(line)
    counts = PlanStore(services.db).counts()
    typer.echo("plans: " + ", ".join(f"{name} {number}" for name, number in counts.items()))


@train_app.command("export-dpo")
@command(CommandKind.LIGHT)
def export_dpo_command(
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset", help="The exported dataset the SFT adapter is trained on (default: newest)"
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the new dataset here (default: next to it)")
    ] = None,
) -> None:
    """Write the preference pairs of /不像 as the DPO file of a new dataset version."""
    services = get_cli_context().services()
    with _failures():
        result = export_dpo(services, dataset=dataset, out_dir=out)
    meta = result.dataset.meta
    typer.echo(
        f"dataset {meta.dataset_version}: {result.pairs} preference pair(s) in dpo_train.jsonl"
    )
    typer.echo(f"in {result.dataset.path}")
    if result.regenerated:
        typer.echo(
            f"{result.regenerated} system segment(s) made again for the card "
            f"{meta.persona_version} and the template {meta.template_version} "
            "the adapter is locked to"
        )
    if result.skipped:
        typer.echo(
            "left out: " + ", ".join(f"{why} {n}" for why, n in sorted(result.skipped.items()))
        )
    if not result.enough:
        typer.echo(
            f"DPO needs at least {result.minimum} pairs; with fewer, dpo.sh skips the step "
            "(more come with /不像 <正确说法>)"
        )
    typer.echo("next: twin train bundle --profile <profile> --dataset <that directory>")


@train_app.command("retrain-check")
@command(CommandKind.LIGHT)
def retrain_check_command() -> None:
    """Compare her messages now with the last training; raise the alert when it is time."""
    services = get_cli_context().services()
    status = check_retrain(services)
    typer.echo(status.describe())
    if status.suggested:
        typer.echo("export the data again with `twin train export`, then train from the base model")
    pairs = PreferencePairStore(services.db, services.clock).count()
    typer.echo(f"preference pairs: {pairs}")
    hint = dpo_hint(pairs, services.settings.training.dpo_min_pairs)
    if hint:
        typer.echo(hint)


# ------------------------------------------------------------------------------ remote


def _latest_or(store: RunStore, run: str | None) -> RunView:
    chosen = store.get(run) if run else store.latest()
    if chosen is None:
        raise CliError(
            "there is no training run yet; start with `twin train remote upload`", ExitCode.USAGE
        )
    return chosen


def _steps_for(
    services: Services, run: RunView, session: RemoteSession, passphrase: Callable[[], str] | None
) -> RemoteSteps:
    return RemoteSteps(
        session=session,
        layout=RemoteLayout(services.settings.autodl.workdir),
        profile=get_profile(run.profile),
        store=RunStore(services.db),
        clock=services.clock,
        models_dir=services.paths.models_dir,
        echo=typer.echo,
        passphrase=passphrase,
        poll_seconds=POLL_SECONDS,
    )


def _trust(label: str, fingerprint: str) -> bool:
    typer.echo(f"host key of {label}: {fingerprint}")
    typer.echo("compare it with the SSH login shown in the AutoDL console if you can")
    return bool(typer.confirm("Trust this host key and remember it?", default=False))


def _session(services: Services) -> RemoteSession:
    target = target_from_settings(
        services.settings.autodl, services.secrets, known_hosts_path(services)
    )
    return RemoteSession(target, services.clock, trust=_trust)


def _run_remote[T](
    services: Services,
    run: RunView,
    action: Callable[[RemoteSteps], Awaitable[T]],
    *,
    passphrase: Callable[[], str] | None = None,
) -> T:
    async def go() -> T:
        async with _session(services) as session:
            return await action(_steps_for(services, run, session, passphrase))

    with _failures():
        return asyncio.run(go())


@remote_app.command("connect")
@command(CommandKind.LIGHT)
def remote_connect_command() -> None:
    """Log in to the instance, show the GPU and the data disk, and remember its host key."""
    services = get_cli_context().services()
    layout = RemoteLayout(services.settings.autodl.workdir)

    async def go() -> None:
        async with _session(services) as session:
            report = await inspect_instance(session, layout)
            typer.echo(f"connected to {report.target}")
            typer.echo(f"GPU:  {report.gpu}")
            typer.echo(f"disk: {report.disk}")

    with _failures():
        asyncio.run(go())


def _create_run(services: Services, bundle: Path, profile: str | None) -> RunView:
    sidecar = sidecar_path(bundle)
    if not bundle.is_file() or not sidecar.is_file():
        raise CliError(
            f"{bundle} or its {sidecar.name} is missing; build it with `twin train bundle`",
            ExitCode.USAGE,
        )
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if profile and profile != meta["profile"]:
        raise CliError(
            f"the package was built for {meta['profile']}, not {profile}", ExitCode.USAGE
        )
    if file_sha256(bundle) != meta["sha256"]:
        raise CliError(
            "the package file changed after it was built; build it again", ExitCode.FAILURE
        )
    chosen = get_profile(meta["profile"])
    store = RunStore(services.db)
    run_id = new_run_id(chosen.name, services.clock.now_utc())
    return store.create(
        run_id=run_id,
        profile=chosen.name,
        dataset_version=meta["dataset_version"],
        bundle_sha256=meta["sha256"],
        bundle_size=int(meta["size"]),
        parameters=hyperparameters(
            chosen, int(meta["train_samples"]), dpo_min_pairs=int(meta.get("dpo_min_pairs", 200))
        ),
    )


@remote_app.command("upload")
@command(CommandKind.LIGHT)
def remote_upload_command(
    bundle: Annotated[Path, typer.Option("--bundle", help="The package from `twin train bundle`")],
    run: Annotated[str | None, RUN_OPTION] = None,
) -> None:
    """Upload the scripts and the encrypted package (resumes after a broken connection)."""
    services = get_cli_context().services()
    with _failures():
        chosen = RunStore(services.db).get(run) if run else _create_run(services, bundle, None)
    typer.echo(f"run {chosen.id} ({chosen.profile})")
    report = _run_remote(services, chosen, lambda steps: steps.upload(chosen.id, bundle))
    typer.echo(f"upload finished: {report.detail}")


def _step_command(
    name: str,
    attribute: str,
    help_text: str,
    *,
    needs_passphrase: bool = False,
) -> Callable[..., None]:
    @remote_app.command(name)
    @command(CommandKind.LIGHT)
    def run_step(
        run: Annotated[str | None, RUN_OPTION] = None,
        again: Annotated[
            bool, typer.Option("--again", help="Run the step again even if it already finished")
        ] = False,
    ) -> None:
        services = get_cli_context().services()
        with _failures():
            chosen = _latest_or(RunStore(services.db), run)
        report = _run_remote(
            services,
            chosen,
            lambda steps: getattr(steps, attribute)(chosen.id, again=again),
            passphrase=_passphrase_for_unpacking if needs_passphrase else None,
        )
        typer.echo(f"{name}: {report.detail}")

    run_step.__doc__ = help_text
    run_step.__name__ = f"remote_{name}_command"
    return run_step


_step_command(
    "setup",
    "setup",
    "Install the packages, decrypt the package and verify the template.",
    needs_passphrase=True,
)
_step_command("train", "train", "Fine-tune the style model (continues from the last checkpoint).")
_step_command("dpo", "dpo", "Preference training on the SFT adapter (only with enough pairs).")
_step_command("eval", "evaluate", "Validation loss and one generated reply per test context.")
_step_command("export", "export", "Merge the adapter, convert to GGUF and quantise.")


@remote_app.command("download")
@command(CommandKind.LIGHT)
def remote_download_command(run: Annotated[str | None, RUN_OPTION] = None) -> None:
    """Download the model files to data/models/<run_id>/ and verify every sha256."""
    services = get_cli_context().services()
    with _failures():
        chosen = _latest_or(RunStore(services.db), run)
    report, target = _run_remote(services, chosen, lambda steps: steps.download(chosen.id))
    typer.echo(f"{report.detail} in {target}")
    typer.echo(f"register them with: twin model register {target}")


@remote_app.command("cleanup")
@command(CommandKind.LIGHT)
def remote_cleanup_command(
    run: Annotated[str | None, RUN_OPTION] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Clean up even though the artifacts were not downloaded")
    ] = False,
) -> None:
    """Erase the data on the instance (after the download); remember the time."""
    services = get_cli_context().services()
    with _failures():
        chosen = _latest_or(RunStore(services.db), run)
    report = _run_remote(services, chosen, lambda steps: steps.cleanup(chosen.id, force=force))
    typer.echo("cleanup finished: the data on the instance was overwritten and deleted")
    typer.echo(
        f"请到 AutoDL 控制台释放实例 ({report.detail}); the data disk is only reclaimed then"
    )


def _status_table(runs: list[RunView]) -> Table:
    table = Table("run", "profile", "dataset", "status", "best val loss", "GPU", "cleaned")
    for item in runs:
        table.add_row(
            item.id,
            item.profile,
            item.dataset_version,
            item.status,
            "-" if item.best_val_loss is None else f"{item.best_val_loss:.4f}",
            item.gpu_model or "-",
            item.cleaned_at.isoformat(timespec="seconds") if item.cleaned_at else "no",
        )
    return table


@remote_app.command("status")
@command(CommandKind.READ)
def remote_status_command(
    run: Annotated[str | None, RUN_OPTION] = None,
    remote: Annotated[
        bool, typer.Option("--remote", help="Also ask the instance for the state of its jobs")
    ] = False,
) -> None:
    """List the training runs and remind of the ones that still have data on an instance."""
    services = get_cli_context().services()
    store = RunStore(services.db)
    runs = store.all()
    if not runs:
        typer.echo("no training runs yet")
        return
    _console().print(_status_table(runs))
    for item in store.uncleaned():
        typer.echo(
            f"run {item.id} has data on the instance and was not cleaned up: "
            "`twin train remote cleanup`, then release the instance in the AutoDL console"
        )
    if remote:
        chosen = _latest_or(store, run)
        states = _run_remote(services, chosen, lambda steps: steps.remote_states(chosen))
        for step, state in states.items():
            typer.echo(f"{chosen.id} {step}: {state}")


@remote_app.command("all")
@command(CommandKind.LIGHT)
def remote_all_command(
    profile: Annotated[str | None, PROFILE_OPTION] = None,
    dataset: Annotated[
        Path | None, typer.Option("--dataset", help="Exported dataset (a package is built from it)")
    ] = None,
    bundle: Annotated[
        Path | None,
        typer.Option("--bundle", help="A package built earlier with `twin train bundle`"),
    ] = None,
    keep_instance_data: Annotated[
        bool, typer.Option("--no-cleanup", help="Stop after the download; clean up later")
    ] = False,
) -> None:
    """Upload, set up, train, evaluate, export, download and clean up in one go."""
    services = get_cli_context().services()
    if (dataset is None) == (bundle is None):
        raise CliError("give either --dataset (with --profile) or --bundle", ExitCode.USAGE)
    passphrase_holder: dict[str, str] = {}
    with _failures():
        if dataset is not None:
            if not profile:
                raise CliError("--dataset needs --profile", ExitCode.USAGE)
            built = _build(services, get_profile(profile), dataset, None)
            passphrase_holder["value"] = built.passphrase
            package = built.path
        else:
            package = bundle if bundle is not None else Path()
        chosen = _create_run(services, package, profile)

    def ask() -> str:
        return passphrase_holder.get("value") or _passphrase_for_unpacking()

    typer.echo(f"run {chosen.id} ({chosen.profile})")

    async def go() -> None:
        async with _session(services) as session:
            steps = _steps_for(services, chosen, session, ask)
            await steps.upload(chosen.id, package)
            await steps.setup(chosen.id)
            await steps.train(chosen.id)
            await steps.dpo(chosen.id)
            await steps.evaluate(chosen.id)
            await steps.export(chosen.id)
            _, target = await steps.download(chosen.id)
            typer.echo(
                f"model files are in {target}; register them with: twin model register {target}"
            )
            if not keep_instance_data:
                await steps.cleanup(chosen.id)
                typer.echo(
                    "请到 AutoDL 控制台释放实例 (release the instance in the AutoDL console)"
                )

    with _failures():
        asyncio.run(go())
