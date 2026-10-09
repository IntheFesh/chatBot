"""R-TRN-008, R-TRN-010, R-PRIV-003: ``twin train bundle`` and ``twin train remote ...``.

The commands talk to a real SSH server on localhost.  The scripts on that "instance" are small
bash programs that leave the files the real scripts leave (the real ones need a GPU), so the
steps, their records in ``training_runs`` and the transfers are the real ones.  Everything that
runs bash or needs absolute paths on both sides is POSIX-only.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.ssh_server import ThreadedSSHServer
from tests.support.training_artifacts import build_artifact_dir
from tests.support.training_data import write_synthetic_dataset
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.config.secrets import SecretStore
from twin.services import Services, build_services
from twin.storage.training_models import DatasetVersion
from twin.training import bundle as bundle_module
from twin.training.bundle import verify_bundle
from twin.training.runs import RunStore

runner = CliRunner()
PASSPHRASE = "correct horse battery staple"
INSTANCE_PASSWORD = "instance-password-1"

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the scripts on the instance are bash programs and the test instance serves absolute "
    "paths of the local disk; the transfers and jobs are covered on Windows by the transport tests",
)

STUBS = {
    "setup.sh": '#!/usr/bin/env bash\nset -eu\necho "setup $1 $2"\n',
    "decrypt.sh": (
        "#!/usr/bin/env bash\nset -eu\nread -r pass\n"
        f'if [ "$pass" != "{PASSPHRASE}" ]; then\n'
        '    echo "wrong passphrase, or the package is damaged" >&2\n    exit 1\nfi\n'
        'mkdir -p "$TWIN_HOME/data"\necho ok >"$TWIN_HOME/data/decrypted"\n'
        'echo "package decrypted for $1"\n'
    ),
    "train.sh": (
        "#!/usr/bin/env bash\nset -eu\n"
        'mkdir -p "$TWIN_HOME/artifacts/train"\n'
        'echo run >>"$TWIN_HOME/train.count"\n'
        'echo \'{"best_val_loss": 1.25, "best_checkpoint": "checkpoint-8", '
        '"peak_vram_mib": 21000, "gpu_name": "NVIDIA GeForce RTX 5090"}\' '
        '>"$TWIN_HOME/artifacts/train/metrics.json"\n'
        'echo "training $1"\n'
    ),
    "dpo.sh": '#!/usr/bin/env bash\nset -eu\necho "DPO skipped: 0 preference pairs"\n',
    "eval.sh": '#!/usr/bin/env bash\nset -eu\necho "evaluated $1 with $2"\n',
    "export.sh": '#!/usr/bin/env bash\nset -eu\necho "export $* run=$TWIN_RUN_ID"\n',
    "cleanup.sh": (
        '#!/usr/bin/env bash\nset -eu\nmkdir -p "$TWIN_HOME/state"\n'
        'date -u +%Y-%m-%dT%H:%M:%SZ >"$TWIN_HOME/state/cleanup.done"\n'
        'echo "请到 AutoDL 控制台释放实例"\n'
    ),
}


@dataclass
class Instance:
    server: ThreadedSSHServer
    workdir: Path
    data_dir: Path
    dataset: Path

    def cli(self, *args: str, input: str | None = None) -> Any:
        return runner.invoke(app, list(args), input=input)

    def services(self) -> Services:
        settings = load_settings()
        return build_services(settings, root=resolve_paths(settings).root)

    def with_store[T](self, work: Callable[[RunStore], T]) -> T:
        services = self.services()
        try:
            return work(RunStore(services.db))
        finally:
            services.close()

    def latest_run(self) -> Any:
        return self.with_store(lambda store: store.latest())

    def prepare_artifacts(self) -> None:
        build_artifact_dir(self.workdir / "artifacts", run_id=None)


@pytest.fixture
def patched_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    real = bundle_module.plain_upload_files

    def with_stubs(profile: Any, **kwargs: Any) -> dict[str, Any]:
        files = real(profile, **kwargs)
        for name, text in STUBS.items():
            files[f"autodl/{name}"] = text.encode("utf-8")
        return files

    monkeypatch.setattr(bundle_module, "plain_upload_files", with_stubs)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


@pytest.fixture
def instance(
    tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch, patched_scripts: None
) -> Iterator[Instance]:
    import twin.training.cli as training_cli

    monkeypatch.setattr(training_cli, "POLL_SECONDS", 0.05)
    with ThreadedSSHServer(tmp_path / "srv", chroot=False) as server:
        workdir = server.root / "work"
        monkeypatch.setenv("TWIN_AUTODL__HOST", "127.0.0.1")
        monkeypatch.setenv("TWIN_AUTODL__PORT", str(server.port))
        monkeypatch.setenv("TWIN_AUTODL__WORKDIR", str(workdir))
        SecretStore.default().set("autodl_password", INSTANCE_PASSWORD)
        write_synthetic_dataset(tmp_path / "ds")
        yield Instance(server, workdir, data_dir, tmp_path / "ds")


# ------------------------------------------------------------------------------- bundle


def test_the_bundle_command_builds_the_package_from_an_exported_dataset(
    tmp_path: Path, data_dir: Path
) -> None:
    dataset = write_synthetic_dataset(tmp_path / "ds", pairs=2)
    result = runner.invoke(
        app,
        ["train", "bundle", "--profile", "5090-8b", "--dataset", str(dataset.path)],
        input=f"{PASSPHRASE}\n{PASSPHRASE}\n",
    )
    assert result.exit_code == 0, result.output
    package = data_dir / "training" / "bundles" / "5090-8b-ds-test-01.bundle.enc"
    assert package.is_file() and (package.parent / "decrypt_bundle.py").is_file()
    assert PASSPHRASE not in result.output
    assert package.name in result.output and "sha256:" in result.output
    sidecar = json.loads(package.with_name(package.name + ".json").read_text("utf-8"))
    assert sidecar["profile"] == "5090-8b" and sidecar["dataset_version"] == "ds-test-01"
    assert sidecar["train_samples"] == 40 and sidecar["sha256"] in result.output
    manifest = verify_bundle(package, PASSPHRASE)
    assert manifest["counts"]["dpo"] == 2
    services = build_services(load_settings(), root=resolve_paths(load_settings()).root)
    try:
        with services.db.session() as session:
            row = session.get(DatasetVersion, "ds-test-01")
            assert row is not None and row.train_count == 40 and row.dpo_count == 2
    finally:
        services.close()
    for leftover in data_dir.rglob("*"):
        assert PASSPHRASE.encode() not in (leftover.read_bytes() if leftover.is_file() else b"")


def test_a_short_passphrase_is_asked_for_again(tmp_path: Path, data_dir: Path) -> None:
    dataset = write_synthetic_dataset(tmp_path / "ds")
    result = runner.invoke(
        app,
        ["train", "bundle", "--profile", "pro6000-14b", "--dataset", str(dataset.path)],
        input=f"short\nshort\n{PASSPHRASE}\n{PASSPHRASE}\n",
    )
    assert result.exit_code == 0, result.output
    assert "needs at least 12 characters" in result.output


def test_the_bundle_command_refuses_an_unknown_profile_and_a_missing_dataset(
    tmp_path: Path, data_dir: Path
) -> None:
    dataset = write_synthetic_dataset(tmp_path / "ds")
    bad_profile = runner.invoke(
        app, ["train", "bundle", "--profile", "4090-7b", "--dataset", str(dataset.path)]
    )
    assert bad_profile.exit_code == 1 and "unknown profile" in bad_profile.output
    nothing = runner.invoke(
        app, ["train", "bundle", "--profile", "5090-8b", "--dataset", str(tmp_path / "none")]
    )
    assert nothing.exit_code == 1 and "export the dataset first" in nothing.output


def test_a_dataset_that_is_not_desensitised_never_becomes_a_package(
    tmp_path: Path, data_dir: Path
) -> None:
    dataset = write_synthetic_dataset(tmp_path / "ds")
    meta = dataset.path / "dataset_meta.json"
    text = json.loads(meta.read_text("utf-8"))
    text["redacted"] = False
    meta.write_text(json.dumps(text), encoding="utf-8")
    result = runner.invoke(
        app, ["train", "bundle", "--profile", "5090-8b", "--dataset", str(dataset.path)]
    )
    assert result.exit_code == 1 and "desensitised" in result.output
    assert not (data_dir / "training" / "bundles").exists()


def test_the_remote_commands_need_the_instance_in_the_settings(data_dir: Path) -> None:
    no_host = runner.invoke(app, ["train", "remote", "connect"])
    assert no_host.exit_code == 1 and "autodl.host" in no_host.output
    none = runner.invoke(app, ["train", "remote", "train"])
    assert none.exit_code == 2 and "no training run yet" in none.output
    status = runner.invoke(app, ["train", "remote", "status"])
    assert status.exit_code == 0 and "no training runs yet" in status.output
    upload = runner.invoke(app, ["train", "remote", "upload", "--bundle", str(data_dir / "x.enc")])
    assert upload.exit_code == 2 and "twin train bundle" in upload.output


def test_all_needs_exactly_one_source_for_the_package(data_dir: Path, tmp_path: Path) -> None:
    both = runner.invoke(app, ["train", "remote", "all"])
    assert both.exit_code == 2 and "either --dataset" in both.output
    no_profile = runner.invoke(app, ["train", "remote", "all", "--dataset", str(tmp_path)])
    assert no_profile.exit_code == 2 and "--dataset needs --profile" in no_profile.output


def build_package(instance: Instance, profile: str = "5090-8b") -> Path:
    result = instance.cli(
        "train",
        "bundle",
        "--profile",
        profile,
        "--dataset",
        str(instance.dataset),
        input=f"{PASSPHRASE}\n{PASSPHRASE}\n",
    )
    assert result.exit_code == 0, result.output
    return instance.data_dir / "training" / "bundles" / f"{profile}-ds-test-01.bundle.enc"


# -------------------------------------------------------------------------- the procedure


@posix_only
def test_connect_shows_the_instance_and_remembers_the_host_key(instance: Instance) -> None:
    result = instance.cli("train", "remote", "connect", input="y\n")
    assert result.exit_code == 0, result.output
    assert f"connected to root@127.0.0.1:{instance.server.port}" in result.output
    assert instance.server.server.fingerprint() in result.output
    assert "GPU:" in result.output and "disk:" in result.output
    known = (instance.data_dir / "training" / "known_hosts").read_text("utf-8")
    assert known.startswith(f"[127.0.0.1]:{instance.server.port} ssh-ed25519 ")
    again = instance.cli("train", "remote", "connect")  # no question the second time
    assert again.exit_code == 0 and "Trust this host key" not in again.output


@posix_only
def test_an_untrusted_host_key_stops_the_command(instance: Instance) -> None:
    result = instance.cli("train", "remote", "connect", input="n\n")
    assert result.exit_code == 1 and "not trusted" in result.output
    assert not (instance.data_dir / "training" / "known_hosts").exists()


@posix_only
def test_a_wrong_instance_password_is_reported_without_retrying(
    instance: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    SecretStore.default().set("autodl_password", "not the password")
    result = instance.cli("train", "remote", "connect", input="y\n")
    assert result.exit_code == 1 and "refused the login" in result.output
    assert INSTANCE_PASSWORD not in result.output and "not the password" not in result.output


@posix_only
def test_the_whole_procedure_step_by_step(instance: Instance) -> None:
    package = build_package(instance)
    sidecar = json.loads(package.with_name(package.name + ".json").read_text("utf-8"))
    up = instance.cli("train", "remote", "upload", "--bundle", str(package), input="y\n")
    assert up.exit_code == 0, up.output
    run = instance.latest_run()
    assert (
        run.profile == "5090-8b"
        and run.status == "uploaded"
        and run.dataset_version == "ds-test-01"
    )
    assert (instance.workdir / "bundle.enc").read_bytes() == package.read_bytes()
    assert run.bundle_sha256 == sidecar["sha256"]
    assert (instance.workdir / "autodl" / "tools" / "remote_job.py").is_file()
    assert (
        (instance.workdir / "autodl" / "profile.env")
        .read_text("utf-8")
        .startswith("TWIN_PROFILE=5090-8b")
    )
    # the package holds data until the cleanup: the status says so
    status = instance.cli("train", "remote", "status")
    assert status.exit_code == 0 and run.id in status.output
    assert "was not cleaned up" in status.output and "twin train remote cleanup" in status.output

    setup = instance.cli("train", "remote", "setup", input=f"{PASSPHRASE}\n")
    assert setup.exit_code == 0, setup.output
    assert (
        "setup 5090-8b install" in setup.output and "package decrypted for 5090-8b" in setup.output
    )
    assert "setup 5090-8b verify" in setup.output and PASSPHRASE not in setup.output
    assert (instance.workdir / "data" / "decrypted").exists()

    for step, text in (
        ("train", "training 5090-8b"),
        ("dpo", "DPO skipped"),
        ("eval", "evaluated 5090-8b with auto"),
    ):
        done = instance.cli("train", "remote", step)
        assert done.exit_code == 0, done.output
        assert text in done.output
    run = instance.latest_run()
    assert run.best_val_loss == 1.25 and run.best_checkpoint == "checkpoint-8"
    assert run.gpu_model == "NVIDIA GeForce RTX 5090" and run.peak_vram_mib == 21000

    instance.prepare_artifacts()
    exported = instance.cli("train", "remote", "export")
    assert exported.exit_code == 0 and f"run={run.id}" in exported.output
    run = instance.latest_run()
    assert "gguf/5090-8b-Q4_K_M.gguf" in run.artifacts and run.status == "exported"

    early = instance.cli("train", "remote", "cleanup")
    assert early.exit_code == 1 and "have not been downloaded" in early.output
    assert not (instance.workdir / "state").exists()

    downloaded = instance.cli("train", "remote", "download")
    assert downloaded.exit_code == 0, downloaded.output
    target = instance.data_dir / "models" / run.id
    assert "twin model register" in downloaded.output
    assert (target / "gguf" / "5090-8b-Q4_K_M.gguf").read_bytes() == (
        instance.workdir / "artifacts" / "gguf" / "5090-8b-Q4_K_M.gguf"
    ).read_bytes()
    assert (target / "manifest.json").is_file()

    cleaned = instance.cli("train", "remote", "cleanup")
    assert cleaned.exit_code == 0, cleaned.output
    assert "请到 AutoDL 控制台释放实例" in cleaned.output
    run = instance.latest_run()
    assert run.status == "cleaned" and run.cleaned_at is not None and not run.needs_cleanup
    after = instance.cli("train", "remote", "status")
    assert "was not cleaned up" not in after.output
    registered = instance.cli("model", "register", str(target), "--run-id", run.id)
    assert registered.exit_code == 0, registered.output
    assert "4 registered" in registered.output


@posix_only
def test_a_wrong_passphrase_fails_the_setup_and_the_next_try_works(instance: Instance) -> None:
    package = build_package(instance)
    assert (
        instance.cli("train", "remote", "upload", "--bundle", str(package), input="y\n").exit_code
        == 0
    )
    bad = instance.cli("train", "remote", "setup", input="not the passphrase\n")
    assert bad.exit_code == 1 and "decrypting failed" in bad.output
    run = instance.latest_run()
    assert run.status == "failed" and "decrypt.sh exited with code 1" in run.last_error
    good = instance.cli("train", "remote", "setup", input=f"{PASSPHRASE}\n")
    assert good.exit_code == 0, good.output
    assert instance.latest_run().last_error is None


@posix_only
def test_a_failing_script_fails_the_step_and_a_finished_job_is_not_run_again(
    instance: Instance,
) -> None:
    package = build_package(instance)
    instance.cli("train", "remote", "upload", "--bundle", str(package), input="y\n")
    instance.cli("train", "remote", "setup", input=f"{PASSPHRASE}\n")
    train = instance.workdir / "autodl" / "train.sh"
    good = train.read_text("utf-8")
    train.write_text("#!/usr/bin/env bash\necho 'CUDA out of memory' >&2\nexit 3\n", "utf-8")
    failed = instance.cli("train", "remote", "train")
    assert failed.exit_code == 1 and "train exited with code 3" in failed.output
    assert "CUDA out of memory" in failed.output  # the log is shown
    run = instance.latest_run()
    assert run.status == "failed" and run.steps["train"]["exit_code"] == 3
    train.write_text(good, encoding="utf-8")
    fixed = instance.cli("train", "remote", "train")
    assert fixed.exit_code == 0, fixed.output
    assert instance.latest_run().status == "trained"
    count = instance.workdir / "train.count"
    assert count.read_text("utf-8").count("run") == 1
    again = instance.cli("train", "remote", "train")
    assert again.exit_code == 0 and "already finished on the instance" in again.output
    assert count.read_text("utf-8").count("run") == 1  # not run a second time
    forced = instance.cli("train", "remote", "train", "--again")
    assert forced.exit_code == 0 and count.read_text("utf-8").count("run") == 2


@posix_only
def test_the_cleanup_can_be_forced_before_the_download(instance: Instance) -> None:
    package = build_package(instance)
    instance.cli("train", "remote", "upload", "--bundle", str(package), input="y\n")
    forced = instance.cli("train", "remote", "cleanup", "--force")
    assert forced.exit_code == 0, forced.output
    assert instance.latest_run().cleaned_at is not None


@posix_only
def test_the_status_can_ask_the_instance_for_its_jobs(instance: Instance) -> None:
    package = build_package(instance)
    instance.cli("train", "remote", "upload", "--bundle", str(package), input="y\n")
    instance.cli("train", "remote", "setup", input=f"{PASSPHRASE}\n")
    status = instance.cli("train", "remote", "status", "--remote")
    run = instance.latest_run()
    assert status.exit_code == 0, status.output
    assert f"{run.id} setup: exited (exit 0)" in status.output
    assert f"{run.id} verify: exited (exit 0)" in status.output


@posix_only
def test_all_runs_every_step_and_leaves_the_instance_clean(instance: Instance) -> None:
    instance.prepare_artifacts()
    result = instance.cli(
        "train",
        "remote",
        "all",
        "--profile",
        "5090-8b",
        "--dataset",
        str(instance.dataset),
        input=f"{PASSPHRASE}\n{PASSPHRASE}\ny\n",
    )
    assert result.exit_code == 0, result.output
    run = instance.latest_run()
    assert run.status == "cleaned" and run.cleaned_at is not None
    assert all(
        run.steps[step]["ok"]
        for step in (
            "upload",
            "setup",
            "decrypt",
            "verify",
            "train",
            "dpo",
            "eval",
            "export",
            "download",
            "cleanup",
        )
    )
    assert (
        instance.data_dir / "models" / run.id / "adapter" / "adapter_model.safetensors"
    ).is_file()
    assert "twin model register" in result.output and "请到 AutoDL 控制台释放实例" in result.output
    assert PASSPHRASE not in result.output
    assert (instance.workdir / "state" / "cleanup.done").is_file()


@posix_only
def test_all_with_an_existing_package_can_stop_before_the_cleanup(instance: Instance) -> None:
    package = build_package(instance, "pro6000-14b")
    instance.prepare_artifacts()
    wrong = instance.cli(
        "train", "remote", "all", "--bundle", str(package), "--profile", "5090-8b", input="y\n"
    )
    assert wrong.exit_code == 2 and "built for pro6000-14b" in wrong.output
    result = instance.cli(
        "train",
        "remote",
        "all",
        "--bundle",
        str(package),
        "--no-cleanup",
        input=f"y\n{PASSPHRASE}\n",
    )
    assert result.exit_code == 0, result.output
    run = instance.latest_run()
    assert run.profile == "pro6000-14b" and run.status == "downloaded" and run.needs_cleanup
    assert not (instance.workdir / "state").exists()


@posix_only
def test_a_package_changed_after_it_was_built_is_not_uploaded(instance: Instance) -> None:
    package = build_package(instance)
    with package.open("ab") as stream:
        stream.write(b"x")
    result = instance.cli("train", "remote", "upload", "--bundle", str(package), input="y\n")
    assert result.exit_code == 1 and "changed after it was built" in result.output


def uploaded(instance: Instance) -> None:
    package = build_package(instance)
    assert (
        instance.cli("train", "remote", "upload", "--bundle", str(package), input="y\n").exit_code
        == 0
    )


@posix_only
def test_a_failing_install_stops_the_setup_before_the_passphrase_is_asked(
    instance: Instance,
) -> None:
    uploaded(instance)
    (instance.workdir / "autodl" / "setup.sh").write_text(
        "#!/usr/bin/env bash\necho 'no space left on device' >&2\nexit 1\n", encoding="utf-8"
    )
    result = instance.cli("train", "remote", "setup")  # no input: asking would abort
    assert result.exit_code == 1 and "setup exited with code 1" in result.output
    assert "no space left on device" in result.output
    run = instance.latest_run()
    assert run.status == "failed" and "decrypt" not in run.steps
    assert not (instance.workdir / "data").exists()


@posix_only
def test_the_download_needs_an_export_and_refuses_paths_outside_the_artifacts(
    instance: Instance,
) -> None:
    uploaded(instance)
    early = instance.cli("train", "remote", "download")
    assert early.exit_code == 1 and "run the export first" in early.output
    artifacts = instance.workdir / "artifacts"
    artifacts.mkdir()
    (artifacts / "manifest.json").write_text(
        json.dumps({"files": [{"path": "../secret", "sha256": "0" * 64, "size": 1}]}),
        encoding="utf-8",
    )
    escaping = instance.cli("train", "remote", "download")
    assert escaping.exit_code == 1 and "outside the artifacts" in escaping.output
    run = instance.latest_run()
    assert run.status == "failed" and not run.steps["download"]["ok"]


@posix_only
def test_a_file_that_changed_on_the_instance_after_the_export_is_not_accepted(
    instance: Instance,
) -> None:
    uploaded(instance)
    instance.prepare_artifacts()
    victim = instance.workdir / "artifacts" / "gguf" / "5090-8b-Q4_K_M.gguf"
    size = victim.stat().st_size
    victim.write_bytes(b"x" * size)  # same size, other bytes: the manifest hash no longer fits
    result = instance.cli("train", "remote", "download")
    assert result.exit_code == 1 and "does not match its sha256" in result.output
    target = instance.data_dir / "models" / instance.latest_run().id / "gguf"
    assert not (target / "5090-8b-Q4_K_M.gguf").exists()


@posix_only
def test_a_cleanup_that_does_not_finish_is_not_recorded_as_done(instance: Instance) -> None:
    uploaded(instance)
    (instance.workdir / "autodl" / "cleanup.sh").write_text(
        "#!/usr/bin/env bash\necho 'shred: permission denied' >&2\nexit 1\n", encoding="utf-8"
    )
    result = instance.cli("train", "remote", "cleanup", "--force")
    assert result.exit_code == 1 and "cleanup did not finish" in result.output
    run = instance.latest_run()
    assert run.cleaned_at is None and run.needs_cleanup and run.status == "failed"
    status = instance.cli("train", "remote", "status")
    assert "was not cleaned up" in status.output


@posix_only
def test_the_step_commands_ask_for_a_known_run(instance: Instance) -> None:
    uploaded(instance)
    unknown = instance.cli("train", "remote", "train", "--run", "r-unknown")
    assert unknown.exit_code == 1 and "no training run 'r-unknown'" in unknown.output


@posix_only
def test_the_scripts_are_uploaded_with_the_dpo_minimum_the_package_was_built_with(
    instance: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_TRAINING__DPO_MIN_PAIRS", "50")
    package = build_package(instance)
    monkeypatch.setenv("TWIN_TRAINING__DPO_MIN_PAIRS", "99")  # changed after the package was built
    assert (
        instance.cli("train", "remote", "upload", "--bundle", str(package), input="y\n").exit_code
        == 0
    )
    uploaded_env = (instance.workdir / "autodl" / "profile.env").read_text("utf-8")
    assert "TWIN_DPO_MIN_PAIRS=50\n" in uploaded_env
    assert instance.latest_run().hyperparameters["dpo_min_pairs"] == 50
