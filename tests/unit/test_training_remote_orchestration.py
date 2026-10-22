"""The steps of a remote run, driven against fakes of the instance (R-TRN-008, R-TRN-010).

``test_training_remote_cli.py`` runs the whole procedure against a real SSH server and the bash
scripts of the instance, which exists on POSIX only.  The orchestration itself - what each step
uploads, runs, reads back, records and refuses - needs neither, so it is tested here with a
session that answers from tables and jobs that end as the test says, on every platform.  (Round 17
(D-745): the Windows coverage job counts ``training/remote`` as a package of its own, and the
real-server tests are skipped there; this is what keeps the orchestration measured on both.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.clock import ManualClock
from tests.support.training_data import write_synthetic_dataset
from twin.services import Services
from twin.training.layout import ARTIFACT_MANIFEST, RemoteLayout
from twin.training.profiles import PROFILES
from twin.training.remote.connection import RemoteError
from twin.training.remote.jobs import JobOutcome, JobStatus
from twin.training.remote.steps import (
    DPO_REFERENCE_FILE,
    STACK_CHOICE_JSON,
    STACK_CHOICE_STATE,
    RemoteSteps,
    StepFailedError,
)
from twin.training.runs import RunStore, ensure_dataset_version, hyperparameters
from twin.training.stack_choice import StackFiles

RUN = "r20261101-073000-5090-8b-abcd"
LAYOUT = RemoteLayout()
ARTIFACTS = LAYOUT.artifacts


@dataclass
class Answer:
    ok: bool = True
    stdout: str = ""
    stderr: str = ""
    status: int = 0


class Files:
    """The file handle of the instance: ``texts`` are what it holds, ``written`` what it got."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts
        self.written: dict[str, tuple[bytes, bool]] = {}
        self.directories: list[str] = []
        self.uploaded: list[str] = []
        self.fail_on: str | None = None
        self.package = SimpleNamespace(
            sha256="a" * 64, size=10, skipped=False, bytes_sent=10, resumed_from=0
        )

    async def read_text(self, remote: str) -> str | None:
        return self.texts.get(remote)

    async def write_bytes(self, remote: str, data: bytes, *, executable: bool = False) -> None:
        if self.fail_on and remote.endswith(self.fail_on):
            raise RemoteError(f"the instance refused {remote}")
        self.written[remote] = (data, executable)
        self.texts[remote] = data.decode("utf-8", "replace")

    async def makedirs(self, directory: str) -> None:
        self.directories.append(directory)

    async def upload(self, path: Path, remote: str, *, progress: Any) -> Any:
        self.uploaded.append(remote)
        progress(3 << 20, 10 << 20)
        progress(7 << 19, 10 << 20)  # 35 %: the same step of ten percent, shown once
        progress(10 << 20, 10 << 20)
        return self.package


class Session:
    """A ``RemoteSession`` that answers from tables."""

    python = "python3"

    def __init__(self, texts: dict[str, str] | None = None) -> None:
        self.handle = Files(texts if texts is not None else {})
        self.commands: list[tuple[str, str | None]] = []
        self.answer = Answer()
        self.hash_tool: str | None = None
        self.target = SimpleNamespace(label="root@region-1.example.test:12345")

    async def files(self) -> Files:
        return self.handle

    async def retrying(self, action: Any, *, what: str) -> Any:
        return await action()

    def use_hash_tool(self, path: str) -> None:
        self.hash_tool = path

    async def run(
        self, command: str, *, stdin: str | None = None, limit_s: float | None = None
    ) -> Answer:
        self.commands.append((command, stdin))
        return self.answer


class Jobs:
    """The jobs of the instance: each ends as ``outcomes`` says (success by default)."""

    def __init__(self) -> None:
        self.states: dict[str, JobStatus] = {}
        self.outcomes: dict[str, JobOutcome] = {}
        self.runs: list[tuple[str, list[str]]] = []
        self.stopped: list[str] = []

    async def status(self, name: str) -> JobStatus:
        return self.states.get(name, JobStatus("none", None, 0))

    async def stop(self, name: str) -> None:
        self.stopped.append(name)

    async def run(
        self, name: str, command: list[str], *, env: dict[str, str], offset: int, emit: Any,
        on_offset: Any,
    ) -> JobOutcome:  # fmt: skip
        self.runs.append((name, command))
        on_offset(offset + 7)
        emit(f"log of {name}\n")
        return self.outcomes.get(name, JobOutcome(name, "exited", 0, offset + 7))


@dataclass
class Rig:
    steps: RemoteSteps
    session: Session
    jobs: Jobs
    store: RunStore
    echoed: list[str] = field(default_factory=list)
    judged: list[StackFiles] = field(default_factory=list)


def verdict(keep: str = "dpo") -> SimpleNamespace:
    return SimpleNamespace(
        keep=keep,
        reason="more of the style metrics within the tolerance",
        to_json=lambda: {"keep": keep},
        lines=lambda: [f"stack kept: {keep}", "  sft: 3 of 6", "  dpo: 5 of 6"],
    )


@pytest.fixture
def rig(services: Services, tmp_path: Path, clock: ManualClock) -> Rig:
    dataset = write_synthetic_dataset(tmp_path / "ds")
    ensure_dataset_version(services.db, dataset, "ds")
    store = RunStore(services.db)
    store.create(
        run_id=RUN,
        profile="5090-8b",
        dataset_version="ds-test-01",
        bundle_sha256="a" * 64,
        bundle_size=10,
        parameters=hyperparameters(PROFILES["5090-8b"], 40),
    )
    session = Session()
    echoed: list[str] = []
    held: list[StackFiles] = []

    def judge(files: StackFiles) -> Any:
        held.append(files)
        return verdict()

    steps = RemoteSteps(
        session=session,  # type: ignore[arg-type]
        layout=LAYOUT,
        profile=PROFILES["5090-8b"],
        store=store,
        clock=clock,
        models_dir=tmp_path / "models",
        echo=echoed.append,
        passphrase=lambda: "correct horse",
        stack_judge=judge,
    )
    jobs = Jobs()
    steps._jobs = jobs  # type: ignore[assignment]
    return Rig(steps, session, jobs, store, echoed, held)


def record(rig: Rig, step: str) -> dict[str, Any]:
    return dict(rig.store.get(RUN).steps[step])


# -------------------------------------------------------------------------------------- upload


async def test_the_upload_sends_the_scripts_then_the_package_and_records_it(
    rig: Rig, tmp_path: Path
) -> None:
    bundle = tmp_path / "bundle.enc"
    bundle.write_bytes(b"x" * 10)
    report = await rig.steps.upload(RUN, bundle)
    assert report.ok and report.detail == "10 bytes sent"
    written = rig.session.handle.written
    scripts = {path.rsplit("/", 1)[1]: flag for path, (_, flag) in written.items()}
    assert scripts["setup.sh"] is True and scripts["profile.env"] is False  # only programs run
    assert rig.session.hash_tool == f"{LAYOUT.tools}/file_hash.py"
    assert rig.session.handle.directories == [LAYOUT.jobs, LAYOUT.logs]
    assert rig.session.handle.uploaded == [LAYOUT.bundle]
    assert [line for line in rig.echoed if line.startswith("upload:")] == [
        "upload: 30% (3 of 10 MiB)",
        "upload: 100% (10 of 10 MiB)",
    ]
    assert record(rig, "upload")["ok"] is True


async def test_an_upload_says_when_the_package_was_already_there_or_resumed(
    rig: Rig, tmp_path: Path
) -> None:
    rig.session.handle.package = SimpleNamespace(
        sha256="a" * 64, size=10, skipped=True, bytes_sent=0, resumed_from=0
    )
    assert (await rig.steps.upload(RUN, tmp_path / "b")).detail == "already on the instance"
    rig.session.handle.package = SimpleNamespace(
        sha256="a" * 64, size=10, skipped=False, bytes_sent=6, resumed_from=4
    )
    assert (await rig.steps.upload(RUN, tmp_path / "b")).detail == "6 bytes sent, resumed from 4"


async def test_an_upload_of_another_package_than_the_run_was_made_with_is_refused(
    rig: Rig, tmp_path: Path
) -> None:
    rig.session.handle.package = SimpleNamespace(
        sha256="b" * 64, size=10, skipped=False, bytes_sent=10, resumed_from=0
    )
    with pytest.raises(RemoteError, match="not the one the run was created with"):
        await rig.steps.upload(RUN, tmp_path / "b")
    assert record(rig, "upload")["ok"] is False
    assert "not the one" in str(rig.store.get(RUN).last_error)


async def test_a_refusal_of_the_instance_during_the_upload_is_recorded_and_raised(
    rig: Rig, tmp_path: Path
) -> None:
    rig.session.handle.fail_on = "setup.sh"
    with pytest.raises(RemoteError, match="refused"):
        await rig.steps.upload(RUN, tmp_path / "b")
    assert "refused" in str(rig.store.get(RUN).last_error)


# --------------------------------------------------------------------------------------- setup


async def test_setup_installs_decrypts_with_the_passphrase_on_stdin_and_verifies(
    rig: Rig,
) -> None:
    report = await rig.steps.setup(RUN)
    assert report.step == "verify" and report.ok
    assert [name for name, _ in rig.jobs.runs] == [f"{RUN}-setup", f"{RUN}-verify"]
    [(command, stdin)] = rig.session.commands
    assert "decrypt.sh" in command and stdin == "correct horse\n"
    assert "correct horse" not in command  # never on the command line
    assert record(rig, "decrypt")["ok"] is True


async def test_a_failed_decryption_stops_the_setup_and_says_what_to_do(rig: Rig) -> None:
    rig.session.answer = Answer(False, "", "wrong passphrase, or the package is damaged", 2)
    with pytest.raises(StepFailedError, match="decrypting failed"):
        await rig.steps.setup(RUN)
    assert record(rig, "decrypt")["exit_code"] == 2
    assert f"{RUN}-verify" not in [name for name, _ in rig.jobs.runs]  # nothing after it
    assert "decrypt.sh exited with code 2" in rig.echoed


async def test_the_setup_needs_a_way_to_ask_for_the_passphrase(
    rig: Rig, services: Services, tmp_path: Path, clock: ManualClock
) -> None:
    rig.steps._passphrase = None
    with pytest.raises(RemoteError, match="passphrase"):
        await rig.steps.setup(RUN)


# ---------------------------------------------------------------- train, dpo, evaluate, export


async def test_train_keeps_what_the_metrics_of_the_instance_say(rig: Rig) -> None:
    metrics = {
        "best_val_loss": 1.5,
        "best_checkpoint": "checkpoint-300",
        "peak_vram_mib": 24000,
        "gpu_name": "NVIDIA GeForce RTX 5090",
    }
    rig.session.handle.texts[f"{ARTIFACTS}/train/metrics.json"] = json.dumps(metrics)
    assert (await rig.steps.train(RUN)).ok
    run = rig.store.get(RUN)
    assert run.best_val_loss == 1.5 and run.gpu_model == "NVIDIA GeForce RTX 5090"
    assert "log of " + f"{RUN}-train" in rig.echoed


async def test_a_train_job_that_fails_is_a_failed_step_and_reads_nothing(rig: Rig) -> None:
    rig.jobs.outcomes[f"{RUN}-train"] = JobOutcome(f"{RUN}-train", "exited", 1, 7)
    with pytest.raises(StepFailedError, match="train exited with code 1"):
        await rig.steps.train(RUN)


async def test_dpo_says_which_model_was_the_reference(rig: Rig) -> None:
    reference = {"reference": "sft", "fallback": False, "reference_quantization_bit": 4}
    rig.session.handle.texts[f"{ARTIFACTS}/{DPO_REFERENCE_FILE}"] = json.dumps(reference)
    assert (await rig.steps.dpo(RUN)).ok
    assert record(rig, "dpo")["reference"] == "sft"
    assert any("explicit, 4-bit" in line for line in rig.echoed)


async def test_dpo_without_a_reference_record_was_skipped_and_says_nothing(rig: Rig) -> None:
    assert (await rig.steps.dpo(RUN)).ok
    assert not any(line.startswith("dpo:") for line in rig.echoed)


async def test_an_evaluation_without_a_dpo_adapter_has_nothing_to_choose(rig: Rig) -> None:
    assert (await rig.steps.evaluate(RUN)).ok
    assert rig.judged == [] and "stack_kept" not in record(rig, "eval")
    assert f"{RUN}-eval" in [name for name, _ in rig.jobs.runs]


async def test_an_evaluation_with_both_stacks_compares_them_and_tells_the_instance(
    rig: Rig,
) -> None:
    base = f"{ARTIFACTS}/eval"
    texts = rig.session.handle.texts
    texts[f"{base}/dpo/eval_metrics.json"] = json.dumps({"eval_loss": 1.1})
    texts[f"{base}/sft/eval_metrics.json"] = json.dumps({"eval_loss": 1.2})
    texts[f"{base}/sft/eval_generations.jsonl"] = "sft-lines"
    texts[f"{base}/dpo/eval_generations.jsonl"] = "dpo-lines"
    assert (await rig.steps.evaluate(RUN)).ok
    [files] = rig.judged
    assert files.generations == {"sft": "sft-lines", "dpo": "dpo-lines"}
    assert files.metrics["dpo"] == {"eval_loss": 1.1} and files.metrics["sft"] == {"eval_loss": 1.2}
    written = rig.session.handle.written
    assert written[f"{LAYOUT.workdir}/{STACK_CHOICE_STATE}"][0] == b"dpo\n"
    assert json.loads(written[f"{ARTIFACTS}/{STACK_CHOICE_JSON}"][0])["keep"] == "dpo"
    assert record(rig, "eval")["stack_kept"] == "dpo"
    assert "stack kept: dpo" in rig.echoed


async def test_a_comparison_that_cannot_be_made_fails_the_evaluation_without_a_verdict(
    rig: Rig,
) -> None:
    base = f"{ARTIFACTS}/eval"
    rig.session.handle.texts[f"{base}/dpo/eval_metrics.json"] = "{}"
    with pytest.raises(StepFailedError, match="could not be compared"):  # no generations at all
        await rig.steps.evaluate(RUN)
    assert STACK_CHOICE_STATE not in " ".join(rig.session.handle.written)
    assert record(rig, "eval")["ok"] is False
    rig.steps._stack_judge = None
    with pytest.raises(StepFailedError, match="no judge"):
        await rig.steps.evaluate(RUN)


async def test_export_records_the_files_of_the_manifest(rig: Rig) -> None:
    manifest = {"files": [{"path": "gguf/m-Q4_K_M.gguf", "sha256": "c" * 64, "size": 5}]}
    rig.session.handle.texts[f"{ARTIFACTS}/{ARTIFACT_MANIFEST}"] = json.dumps(manifest)
    assert (await rig.steps.export(RUN)).ok
    assert rig.store.get(RUN).artifacts == {"gguf/m-Q4_K_M.gguf": {"sha256": "c" * 64, "size": 5}}
    assert rig.jobs.runs[-1][1][-1] != "--redo"
    await rig.steps.export(RUN, again=True)
    assert rig.jobs.runs[-1][1][-1] == "--redo"


async def test_a_step_run_again_stops_the_job_that_still_runs_first(rig: Rig) -> None:
    rig.jobs.states[f"{RUN}-train"] = JobStatus("running", None, 100)
    await rig.steps.train(RUN, again=True)
    assert rig.jobs.stopped == [f"{RUN}-train"]


# ------------------------------------------------------------------- download, cleanup, states


async def test_a_download_needs_the_manifest_of_the_instance(rig: Rig) -> None:
    with pytest.raises(RemoteError, match="run the export first"):
        await rig.steps.download(RUN)
    assert record(rig, "download")["ok"] is False


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside.gguf", "gguf/../../x"])
async def test_a_manifest_that_names_a_path_outside_the_artifacts_is_refused(
    rig: Rig, path: str
) -> None:
    manifest = {"files": [{"path": path, "sha256": "c" * 64, "size": 5}]}
    rig.session.handle.texts[f"{ARTIFACTS}/{ARTIFACT_MANIFEST}"] = json.dumps(manifest)
    with pytest.raises(RemoteError, match="outside the artifacts"):
        await rig.steps.download(RUN)
    assert "outside the artifacts" in str(rig.store.get(RUN).last_error)


async def test_a_cleanup_before_the_download_would_lose_the_artifacts(rig: Rig) -> None:
    with pytest.raises(RemoteError, match="not been downloaded yet"):
        await rig.steps.cleanup(RUN)
    assert rig.session.commands == []  # nothing was run


async def test_a_cleanup_after_the_download_erases_and_checks_the_marker(rig: Rig) -> None:
    rig.store.begin_step(RUN, "download", rig.steps._clock.now_utc())
    rig.store.end_step(RUN, "download", rig.steps._clock.now_utc(), exit_code=0)
    rig.session.handle.texts[f"{LAYOUT.workdir}/state/cleanup.done"] = "done"
    report = await rig.steps.cleanup(RUN)
    assert report.ok and "release the instance" in report.detail
    [(command, _)] = rig.session.commands
    assert "cleanup.sh" in command and "--yes" in command
    assert record(rig, "cleanup")["ok"] is True


async def test_a_cleanup_without_its_marker_did_not_finish(rig: Rig) -> None:
    rig.session.answer = Answer(True, "erasing\n", "", 0)
    with pytest.raises(StepFailedError, match="cleanup did not finish"):
        await rig.steps.cleanup(RUN, force=True)  # (force: the download was not made)
    assert record(rig, "cleanup")["ok"] is False
    assert "erasing" in rig.echoed


async def test_the_states_of_the_jobs_on_the_instance_are_listed_by_step(rig: Rig) -> None:
    rig.jobs.states[f"{RUN}-train"] = JobStatus("exited", 0, 10)
    rig.jobs.states[f"{RUN}-eval"] = JobStatus("running", None, 5)
    states = await rig.steps.remote_states(rig.store.get(RUN))
    assert states == {"train": "exited (exit 0)", "eval": "running"}
