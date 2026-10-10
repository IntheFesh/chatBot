"""The steps of a remote training run on the injected clock (R-TRN-008, R-TRN-010, R-NFR-005).

The whole procedure is exercised against a real SSH server by ``test_training_remote_cli.py``
(POSIX only).  Here the pieces that do not need a server run everywhere: what a finished or failed
step writes into the run record (and *when*, on the injected clock), what the instance's jobs are
reported as, and how the GPU and the disk of the instance are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.clock import ManualClock
from tests.support.training_data import write_synthetic_dataset
from twin.services import Services
from twin.training.layout import RemoteLayout
from twin.training.profiles import PROFILES
from twin.training.remote.jobs import JobOutcome, JobStatus
from twin.training.remote.steps import (
    ConnectionReport,
    RemoteSteps,
    StepFailedError,
    StepReport,
    inspect_instance,
)
from twin.training.runs import RunStore, ensure_dataset_version, hyperparameters

RUN = "r20261101-073000-5090-8b-abcd"


@dataclass
class Answer:
    ok: bool
    stdout: str = ""
    stderr: str = ""


class Instance:
    """The little of a ``RemoteSession`` these pieces use: commands answered from a table."""

    python = "python3"

    def __init__(self, answers: dict[str, Answer]) -> None:
        self.answers = answers
        self.target = SimpleNamespace(label="root@region-1.example.test:12345")
        self.commands: list[str] = []

    async def run(self, command: str) -> Answer:
        self.commands.append(command)
        for start, answer in self.answers.items():
            if command.startswith(start):
                return answer
        return Answer(False, "", "command not found")


class Jobs:
    """The jobs of the instance: ``{name: JobStatus}``."""

    def __init__(self, states: dict[str, JobStatus]) -> None:
        self.states = states

    async def status(self, name: str) -> JobStatus:
        return self.states.get(name, JobStatus("none", None, 0))


@pytest.fixture
def steps(services: Services, tmp_path: Path, clock: ManualClock) -> RemoteSteps:
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
    return RemoteSteps(
        session=Instance({}),  # type: ignore[arg-type]
        layout=RemoteLayout(),
        profile=PROFILES["5090-8b"],
        store=store,
        clock=clock,
        models_dir=tmp_path / "models",
        echo=lambda _text: None,
    )


# ---------------------------------------------------------------------------------- the record


def test_a_finished_step_is_recorded_with_the_moment_it_ended_on_the_injected_clock(
    steps: RemoteSteps, clock: ManualClock
) -> None:
    clock.set_time(datetime(2026, 11, 1, 7, 30, tzinfo=UTC))  # 01:30 CST: the repeated hour
    steps._store.begin_step(RUN, "train", clock.now_utc())
    clock.tick(3 * 3600 + 25 * 60)  # three hours and twenty-five minutes of training
    report = steps._finish(RUN, "train", JobOutcome("train", "exited", 0, 100))
    assert report == StepReport("train", True, "done")
    record = steps._store.get(RUN).steps["train"]
    assert record["ok"] is True and record["exit_code"] == 0
    started = datetime.fromisoformat(record["started_at"])
    finished = datetime.fromisoformat(record["finished_at"])
    assert started == datetime(2026, 11, 1, 7, 30, tzinfo=UTC)
    assert finished - started == timedelta(hours=3, minutes=25)
    assert started.utcoffset() == timedelta(0) and finished.utcoffset() == timedelta(0)  # UTC


def test_a_failed_step_stops_the_procedure_and_says_what_to_do(
    steps: RemoteSteps, clock: ManualClock
) -> None:
    steps._store.begin_step(RUN, "setup", clock.now_utc())
    clock.tick(90)
    with pytest.raises(StepFailedError, match=r"setup exited with code 3.*run the step again"):
        steps._finish(RUN, "setup", JobOutcome("setup", "exited", 3, 0))
    record = steps._store.get(RUN).steps["setup"]
    assert record["ok"] is False and record["exit_code"] == 3
    assert datetime.fromisoformat(record["finished_at"]) == clock.now_utc()
    steps._store.begin_step(RUN, "train", clock.now_utc())
    with pytest.raises(StepFailedError, match="train was lost on the instance"):
        steps._finish(RUN, "train", JobOutcome("train", "lost", None, 0))  # the instance restarted


def test_the_fields_learned_on_the_way_are_kept_with_the_step(
    steps: RemoteSteps, clock: ManualClock
) -> None:
    steps._store.begin_step(RUN, "train", clock.now_utc())
    fields = {"best_val_loss": 1.25, "gpu_model": "NVIDIA GeForce RTX 5090"}
    steps._finish(RUN, "train", JobOutcome("train", "exited", 0, 0), fields)
    run = steps._store.get(RUN)
    assert run.best_val_loss == 1.25 and run.gpu_model == "NVIDIA GeForce RTX 5090"


# ------------------------------------------------------------------------- the instance


async def test_the_jobs_of_a_run_are_reported_with_their_exit_codes(steps: RemoteSteps) -> None:
    steps._jobs = Jobs(  # type: ignore[assignment]
        {
            f"{RUN}-setup": JobStatus("exited", 0, 10),
            f"{RUN}-train": JobStatus("running", None, 20),
            f"{RUN}-eval": JobStatus("exited", 2, 5),
        }
    )
    states = await steps.remote_states(steps._store.get(RUN))
    assert states == {"setup": "exited (exit 0)", "train": "running", "eval": "exited (exit 2)"}


async def test_the_gpu_and_the_data_disk_of_the_instance_are_read_from_its_answers() -> None:
    gpu = "NVIDIA GeForce RTX 5090, 32607 MiB, 570.86, 12.0\n"
    df = (
        "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
        "/dev/vdb 52428800 0 52428800 0% /root/autodl-tmp\n"
    )
    session = Instance({"nvidia-smi": Answer(True, gpu), "df": Answer(True, df)})
    report = await inspect_instance(session, RemoteLayout("/root/autodl-tmp/twin"))  # type: ignore[arg-type]
    assert report == ConnectionReport(
        "root@region-1.example.test:12345",
        "NVIDIA GeForce RTX 5090, 32607 MiB, 570.86, 12.0",
        "/dev/vdb 52428800 0 52428800 0% /root/autodl-tmp",
    )
    assert "df -Pk /root/autodl-tmp" in session.commands[-1]


async def test_an_instance_without_a_gpu_or_a_disk_report_says_so() -> None:
    session = Instance({})
    report = await inspect_instance(session, RemoteLayout("/root/autodl-tmp/twin"))  # type: ignore[arg-type]
    assert (report.gpu, report.disk) == ("no GPU found", "unknown")
