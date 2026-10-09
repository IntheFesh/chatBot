"""R-TRN-009, R-TRN-010: the Python tools that ship with the AutoDL scripts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from tests.support.autodl_scripts import AUTODL
from tests.support.waiting import wait_until_sync as wait_until

TOOLS = AUTODL / "tools"


def load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tool_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


remote_job = load_tool("remote_job")
summarize_train = load_tool("summarize_train")
make_eval = load_tool("make_eval_generations")
artifact_manifest = load_tool("artifact_manifest")
verify_manifest = load_tool("verify_manifest")


def job(tmp_path: Path, *args: str) -> dict[str, object]:
    """Run remote_job.py the way the orchestrator does: a separate process, JSON on stdout."""
    done = subprocess.run(
        [sys.executable, str(TOOLS / "remote_job.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)  # type: ignore[no-any-return]


def finished(tmp_path: Path, name: str) -> Callable[[], bool]:
    return lambda: (
        job(tmp_path, "status", "--dir", str(tmp_path / "jobs"), "--name", name)["state"]
        in ("exited", "lost")
    )


def run_job(tmp_path: Path, name: str, *command: str, extra: tuple[str, ...] = ()) -> None:
    jobs = str(tmp_path / "jobs")
    job(tmp_path, "start", "--dir", jobs, "--name", name, *extra, "--", *command)


# --------------------------------------------------------------------------- remote_job


def test_a_job_runs_in_the_background_and_reports_its_exit_code_and_log(tmp_path: Path) -> None:
    jobs = str(tmp_path / "jobs")
    code = "import sys; print('hello'); print('oops', file=sys.stderr); sys.exit(3)"
    started = job(
        tmp_path, "start", "--dir", jobs, "--name", "t1", "--", sys.executable, "-c", code
    )
    assert started["state"] in ("started", "running", "exited")
    wait_until(lambda: job(tmp_path, "status", "--dir", jobs, "--name", "t1")["state"] == "exited")
    done = job(tmp_path, "status", "--dir", jobs, "--name", "t1")
    assert done["exit_code"] == 3 and done["log_size"] > 0
    tail = job(tmp_path, "tail", "--dir", jobs, "--name", "t1", "--offset", "0")
    assert "hello" in str(tail["data"]) and "oops" in str(tail["data"]) and tail["eof"] is True


def test_a_job_that_does_not_exist_reports_none(tmp_path: Path) -> None:
    assert job(tmp_path, "status", "--dir", str(tmp_path / "jobs"), "--name", "nope") == {
        "state": "none",
        "exit_code": None,
        "log_size": 0,
    }
    tail = job(tmp_path, "tail", "--dir", str(tmp_path / "jobs"), "--name", "nope", "--offset", "5")
    assert tail == {"offset": 5, "data": "", "eof": True}


def test_the_log_is_read_on_from_the_last_offset_and_survives_a_reconnect(tmp_path: Path) -> None:
    jobs = str(tmp_path / "jobs")
    marker = tmp_path / "go"
    code = (
        "import pathlib, sys, time\n"
        "print('first line', flush=True)\n"
        f"p = pathlib.Path({str(marker)!r})\n"
        "while not p.exists():\n    time.sleep(0.05)\n"
        "print('second line', flush=True)\n"
    )
    run_job(tmp_path, "t2", sys.executable, "-c", code)
    wait_until(
        lambda: (
            "first line"
            in str(job(tmp_path, "tail", "--dir", jobs, "--name", "t2", "--offset", "0")["data"])
        )
    )
    first = job(tmp_path, "tail", "--dir", jobs, "--name", "t2", "--offset", "0")
    assert str(first["data"]).strip() == "first line" and first["eof"] is True
    assert job(tmp_path, "status", "--dir", jobs, "--name", "t2")["state"] == "running"
    # "the connection dropped": a new process asks for the state and goes on from the offset
    again = job(
        tmp_path, "start", "--dir", jobs, "--name", "t2", "--", sys.executable, "-c", "pass"
    )
    assert again["state"] == "running"  # the same job, not a second one
    marker.write_text("go", encoding="utf-8")
    wait_until(finished(tmp_path, "t2"))
    rest = job(tmp_path, "tail", "--dir", jobs, "--name", "t2", "--offset", str(first["offset"]))
    assert str(rest["data"]).strip() == "second line"
    assert job(tmp_path, "status", "--dir", jobs, "--name", "t2")["exit_code"] == 0


def test_the_environment_and_working_directory_are_passed_to_the_command(tmp_path: Path) -> None:
    jobs = str(tmp_path / "jobs")
    work = tmp_path / "work"
    work.mkdir()
    code = "import os; print(os.environ['TWIN_X'], os.getcwd())"
    run_job(
        tmp_path,
        "t3",
        sys.executable,
        "-c",
        code,
        extra=("--cwd", str(work), "--env", "TWIN_X=a=b"),
    )
    wait_until(finished(tmp_path, "t3"))
    text = str(job(tmp_path, "tail", "--dir", jobs, "--name", "t3", "--offset", "0")["data"])
    assert text.split()[0] == "a=b" and Path(text.split()[1]).resolve() == work.resolve()


def test_a_finished_job_can_be_started_again_and_starts_with_an_empty_log(tmp_path: Path) -> None:
    jobs = str(tmp_path / "jobs")
    run_job(tmp_path, "t4", sys.executable, "-c", "print('one')")
    wait_until(finished(tmp_path, "t4"))
    run_job(tmp_path, "t4", sys.executable, "-c", "print('two')")
    wait_until(
        lambda: (
            "two"
            in str(job(tmp_path, "tail", "--dir", jobs, "--name", "t4", "--offset", "0")["data"])
        )
    )
    data = str(job(tmp_path, "tail", "--dir", jobs, "--name", "t4", "--offset", "0")["data"])
    assert "one" not in data


def test_a_missing_program_is_a_failed_job_with_a_reason_in_the_log(tmp_path: Path) -> None:
    jobs = str(tmp_path / "jobs")
    run_job(tmp_path, "t5", str(tmp_path / "no-such-program"))
    wait_until(finished(tmp_path, "t5"))
    done = job(tmp_path, "status", "--dir", jobs, "--name", "t5")
    assert done["exit_code"] == 127
    assert "could not start" in str(
        job(tmp_path, "tail", "--dir", jobs, "--name", "t5", "--offset", "0")["data"]
    )


def test_a_multibyte_character_is_never_cut_in_the_middle(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs" / "t6"
    jobs.mkdir(parents=True)
    (jobs / "log").write_bytes("ab我们".encode())  # 2 + 3 + 3 bytes
    tail = remote_job.tail(jobs, 0, 4)  # 4 bytes end inside the first Chinese character
    assert tail["data"] == "ab" and tail["offset"] == 2 and tail["eof"] is False
    rest = remote_job.tail(jobs, 2, 100)
    assert rest["data"] == "我们" and rest["offset"] == 8 and rest["eof"] is True


def test_a_job_whose_supervisor_vanished_is_reported_lost(tmp_path: Path) -> None:
    directory = tmp_path / "jobs" / "t7"
    directory.mkdir(parents=True)
    heartbeat = directory / "heartbeat"
    heartbeat.write_text("x", encoding="utf-8")
    old = time.time() - 600
    import os

    os.utime(heartbeat, (old, old))
    assert remote_job.status(directory)["state"] == "lost"
    heartbeat.write_text("y", encoding="utf-8")
    assert remote_job.status(directory)["state"] == "running"


def test_job_names_cannot_leave_the_jobs_directory(tmp_path: Path) -> None:
    for name in ("", "../x", "a/b", "a\\b"):
        with pytest.raises(ValueError, match="invalid job name"):
            remote_job.job_dir(tmp_path, name)


def test_stop_ends_a_running_job(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("SIGTERM semantics: the job runs on the Linux instance; the stop path is POSIX")
    jobs = str(tmp_path / "jobs")
    run_job(tmp_path, "t8", sys.executable, "-c", "import time; time.sleep(60)")
    wait_until(lambda: (tmp_path / "jobs" / "t8" / "child_pid").exists())
    job(tmp_path, "stop", "--dir", jobs, "--name", "t8")
    wait_until(finished(tmp_path, "t8"))
    assert job(tmp_path, "status", "--dir", jobs, "--name", "t8")["exit_code"] != 0


def test_start_without_a_command_is_a_usage_error(tmp_path: Path) -> None:
    done = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "remote_job.py"),
            "start",
            "--dir",
            str(tmp_path),
            "--name",
            "x",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert done.returncode == 2


# ------------------------------------------------------------------------- the others


def write_state(directory: Path, **extra: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "best_model_checkpoint": "/x/y/checkpoint-8",
        "best_metric": 1.25,
        "global_step": 12,
        "epoch": 3.0,
        "log_history": [
            {"step": 4, "eval_loss": 1.5},
            {"step": 8, "eval_loss": 1.25},
            {"step": 12, "loss": 1.1},
        ],
        **extra,
    }
    (directory / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")


def test_the_training_summary_has_the_best_checkpoint_loss_curve_and_peak_memory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    write_state(output)
    (output / "training_loss.png").write_bytes(b"png")
    gpu = tmp_path / "gpu.csv"
    gpu.write_text("1000\n21000\n19000\n\n", encoding="utf-8")
    code = summarize_train.main(
        [
            "--output-dir",
            str(output),
            "--out-dir",
            str(tmp_path / "art"),
            "--gpu-log",
            str(gpu),
            "--gpu-name",
            "NVIDIA GeForce RTX 5090",
        ]
    )
    assert code == 0
    metrics = json.loads((tmp_path / "art" / "metrics.json").read_text("utf-8"))
    assert metrics == {
        "best_checkpoint": "checkpoint-8",
        "best_val_loss": 1.25,
        "global_step": 12,
        "epochs_done": 3.0,
        "eval_history": [[4, 1.5], [8, 1.25]],
        "last_train_loss": 1.1,
        "peak_vram_mib": 21000,
        "gpu_name": "NVIDIA GeForce RTX 5090",
    }
    assert (tmp_path / "art" / "training_loss.png").read_bytes() == b"png"


def test_the_training_summary_fails_when_the_run_left_no_state(tmp_path: Path) -> None:
    assert (
        summarize_train.main(["--output-dir", str(tmp_path), "--out-dir", str(tmp_path / "a")]) == 1
    )
    assert summarize_train.peak_memory_mib(None) is None
    assert summarize_train.peak_memory_mib(tmp_path / "missing.csv") is None


def sft_row(sample_id: str, reply: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "system": "s",
        "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": reply}],
    }


def test_generations_get_their_ids_back_by_position_and_nothing_else_is_kept(
    tmp_path: Path,
) -> None:
    test = tmp_path / "test.jsonl"
    test.write_text(
        "\n".join(json.dumps(sft_row(f"te{i}", f"reply {i}")) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"prompt": "P", "predict": f" generated {i}\n", "label": f"reply {i}\n"})
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "eval_results.json"
    results.write_text(json.dumps({"eval_loss": 1.1}), encoding="utf-8")
    code = make_eval.main(
        [
            "--test",
            str(test),
            "--predictions",
            str(predictions),
            "--eval-results",
            str(results),
            "--adapter",
            "sft",
            "--out-dir",
            str(tmp_path / "eval"),
        ]
    )
    assert code == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "eval" / "eval_generations.jsonl").read_text("utf-8").splitlines()
    ]
    assert rows == [{"id": f"te{i}", "text": f"generated {i}"} for i in range(3)]
    metrics = json.loads((tmp_path / "eval" / "eval_metrics.json").read_text("utf-8"))
    assert metrics == {"val_loss": 1.1, "adapter": "sft", "generated": 3}


def test_generations_that_do_not_line_up_with_the_test_set_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps(sft_row("a", "secret reply")) + "\n", encoding="utf-8")
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps({"predict": "x", "label": "another reply"}) + "\n", encoding="utf-8")
    args = ["--test", str(test), "--predictions", str(pred), "--out-dir", str(tmp_path / "o")]
    assert make_eval.main(args) == 1
    error = capsys.readouterr().err
    assert "do not belong to the test sample" in error and "secret reply" not in error
    pred.write_text("", encoding="utf-8")
    assert make_eval.main(args) == 1
    assert "0 predictions for 1 test samples" in capsys.readouterr().err


def make_artifacts(root: Path) -> Path:
    for name, content in {
        "gguf/5090-8b-Q4_K_M.gguf": b"q4",
        "gguf/5090-8b-Q5_K_M.gguf": b"q5",
        "gguf/5090-8b-Q8_0.gguf": b"q8",
        "adapter/adapter_model.safetensors": b"adapter",
        "adapter/adapter_config.json": b"{}",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "train").mkdir()
    (root / "train" / "metrics.json").write_text(
        json.dumps(
            {"best_val_loss": 1.25, "best_checkpoint": "checkpoint-8", "peak_vram_mib": 21000}
        ),
        encoding="utf-8",
    )
    return root


BUNDLE_MANIFEST = {
    "profile": "5090-8b",
    "base_model": "Qwen/Qwen3-8B",
    "template": "qwen3_nothink",
    "template_version": "qwen3_nothink@llamafactory-0.9.5",
    "persona_version": "v3",
    "profile_version": "P1",
    "dataset_version": "ds-1",
    "llamafactory_version": "0.9.5",
    "llama_cpp_tag": "b11177",
}


def test_the_artifact_manifest_lists_files_models_versions_and_metrics(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path / "art")
    bundle = tmp_path / "manifest.json"
    bundle.write_text(json.dumps(BUNDLE_MANIFEST), encoding="utf-8")
    code = artifact_manifest.main(
        [
            "--artifacts",
            str(artifacts),
            "--bundle-manifest",
            str(bundle),
            "--quants",
            "Q4_K_M,Q5_K_M,Q8_0",
            "--adapter-kind",
            "dpo",
            "--run-id",
            "r1",
        ]
    )
    assert code == 0
    manifest = json.loads((artifacts / "manifest.json").read_text("utf-8"))
    assert manifest["run_id"] == "r1" and manifest["adapter_kind"] == "dpo"
    assert manifest["persona_version"] == "v3" and manifest["dataset_version"] == "ds-1"
    assert manifest["template_version"] == "qwen3_nothink@llamafactory-0.9.5"
    assert manifest["models"] == [
        {"quant": "Q4_K_M", "kind": "gguf", "path": "gguf/5090-8b-Q4_K_M.gguf"},
        {"quant": "Q5_K_M", "kind": "gguf", "path": "gguf/5090-8b-Q5_K_M.gguf"},
        {"quant": "Q8_0", "kind": "gguf", "path": "gguf/5090-8b-Q8_0.gguf"},
        {"quant": "lora", "kind": "adapter", "path": "adapter/adapter_model.safetensors"},
    ]
    files = {f["path"]: f for f in manifest["files"]}
    assert "manifest.json" not in files and "train/metrics.json" in files
    assert files["gguf/5090-8b-Q4_K_M.gguf"]["size"] == 2
    assert len(files["gguf/5090-8b-Q4_K_M.gguf"]["sha256"]) == 64
    assert (
        manifest["metrics"]["best_val_loss"] == 1.25
        and manifest["metrics"]["peak_vram_mib"] == 21000
    )
    # writing it again does not list the manifest itself
    assert (
        artifact_manifest.main(
            [
                "--artifacts",
                str(artifacts),
                "--bundle-manifest",
                str(bundle),
                "--quants",
                "Q4_K_M,Q5_K_M,Q8_0",
            ]
        )
        == 0
    )
    assert json.loads((artifacts / "manifest.json").read_text("utf-8"))["run_id"] is None


def test_the_artifact_manifest_needs_every_model_file_and_the_bundle_manifest(
    tmp_path: Path,
) -> None:
    artifacts = make_artifacts(tmp_path / "art")
    bundle = tmp_path / "manifest.json"
    args = ["--artifacts", str(artifacts), "--bundle-manifest", str(bundle), "--quants", "Q4_K_M"]
    assert artifact_manifest.main(args) == 1  # no bundle manifest yet
    bundle.write_text(json.dumps(BUNDLE_MANIFEST), encoding="utf-8")
    (artifacts / "gguf" / "5090-8b-Q4_K_M.gguf").unlink()
    assert artifact_manifest.main(args) == 1


def test_the_unpacked_package_is_checked_against_its_manifest_and_the_uploaded_scripts(
    tmp_path: Path,
) -> None:
    import hashlib

    root, scripts = tmp_path / "pkg", tmp_path / "up"
    (root / "data").mkdir(parents=True)
    (root / "autodl").mkdir()
    scripts.mkdir()
    entries = {"data/a.jsonl": b"aaa", "autodl/x.sh": b"echo\n"}
    for name, content in entries.items():
        (root / name).write_bytes(content)
    (scripts / "x.sh").write_bytes(b"echo\n")
    manifest = {
        "files": [
            {"path": n, "sha256": hashlib.sha256(c).hexdigest(), "size": len(c)}
            for n, c in entries.items()
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest.check(root, scripts) == []
    (scripts / "x.sh").write_bytes(b"echo changed\n")
    assert verify_manifest.check(root, scripts) == ["autodl/x.sh differs from the uploaded script"]
    (scripts / "x.sh").write_bytes(b"echo\n")
    (root / "data" / "extra.txt").write_text("x", encoding="utf-8")
    (root / "data" / "a.jsonl").write_bytes(b"bbb")
    problems = verify_manifest.check(root, None)
    assert "data/extra.txt is not listed in the manifest" in problems
    assert "data/a.jsonl does not match the manifest" in problems
    (root / "manifest.json").unlink()
    assert verify_manifest.check(root, None) == ["manifest.json is missing"]
    assert verify_manifest.main(["--root", str(root)]) == 1
