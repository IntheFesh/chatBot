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
on the instance (R-ARCH-006).  The dataset export itself (``twin train export``) is round 13b.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from twin.ops.process_model import CliError, CommandKind, ExitCode, command
from twin.services import Services, get_cli_context
from twin.training import bundle_crypto
from twin.training.bundle import BundleError, build_bundle, verify_bundle
from twin.training.dataset_dir import DatasetError, file_sha256, load_dataset_dir
from twin.training.layout import RemoteLayout
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
from twin.training.runs import (
    RunError,
    RunStore,
    RunView,
    ensure_dataset_version,
    hyperparameters,
    new_run_id,
)

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
        ProfileError,
        RegistryError,
        RemoteError,
        RunError,
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
