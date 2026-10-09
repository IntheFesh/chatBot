"""Helpers to run the AutoDL shell scripts (``training/autodl``) on a Linux machine in tests.

The scripts target the Linux instance, so the tests that execute them are POSIX-only
(:data:`posix_only`).  A fake ``llamafactory-cli`` replaces the real training program: it reads
the YAML it is given and leaves the files the real program would leave, so the scripts' own logic
(argument handling, checks, the order of the steps, what they delete) runs for real.
"""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from twin.training import bundle
from twin.training.profiles import PROFILES

ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "training" / "autodl"

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the AutoDL scripts are bash programs for the Linux instance; the static checks "
    "(shellcheck, structure, tool tests) run everywhere",
)


def install_scripts(workdir: Path, profile: str, *, dpo_min_pairs: int = 200) -> Path:
    """Write the plain upload set where ``twin train remote upload`` puts it (returns autodl/)."""
    for name, content in bundle.plain_upload_files(
        PROFILES[profile], dpo_min_pairs=dpo_min_pairs
    ).items():
        target = workdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.read_bytes())
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    return workdir / "autodl"


def script_env(workdir: Path, tmp_path: Path, **extra: str) -> dict[str, str]:
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    env = {
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "TWIN_HOME": str(workdir),
        "TWIN_MODELS": str(tmp_path / "models"),
        "TWIN_DISK_ROOT": str(tmp_path / "disk"),
        "TWIN_PYTHON": sys.executable,
        "TWIN_LOG_TEE": "0",
        "LC_ALL": "C.UTF-8",
    }
    env.update(extra)
    return env


def run_bash(
    script: str,
    env: dict[str, str],
    *,
    cwd: Path | None = None,
    stdin: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=timeout,
    )


def run_script(
    workdir: Path,
    env: dict[str, str],
    name: str,
    *args: str,
    stdin: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(workdir / "autodl" / name), *args],
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=timeout,
    )


def call_function(autodl: Path, env: dict[str, str], body: str) -> subprocess.CompletedProcess[str]:
    """Source the environment files and ``lib.sh`` of ``autodl``, then run ``body``."""
    prologue = (
        f"set -euo pipefail; cd {shlex.quote(str(autodl))}; "
        "set -a; . ./versions.env; . ./profile.env; set +a; . ./lib.sh; "
    )
    return run_bash(prologue + body, env)


# --------------------------------------------------------------------- fake LLaMA-Factory

FAKE_CLI = r'''
"""A stand-in for llamafactory-cli: leaves the files the real command leaves."""
import json
import sys
from pathlib import Path

import yaml

LOG = Path(sys.argv[0]).with_name("cli_calls.log")
mode, config_path = sys.argv[1], Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
with LOG.open("a", encoding="utf-8") as log:
    log.write(json.dumps({"mode": mode, "config": config_path.name,
                          "batch": config.get("per_device_train_batch_size"),
                          "adapter": config.get("adapter_name_or_path")}) + "\n")
limit_file = Path(sys.argv[0]).with_name("max_batch")
limit = int(limit_file.read_text()) if limit_file.exists() else 99
if config.get("per_device_train_batch_size", 1) > limit:
    sys.stderr.write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
    sys.exit(1)
if mode == "export":
    merged = Path(config["export_dir"])
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "model.safetensors").write_bytes(b"merged-weights")
    sys.exit(0)
output = Path(config["output_dir"])
output.mkdir(parents=True, exist_ok=True)
if config.get("do_train"):
    (output / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (output / "adapter_config.json").write_text("{}", encoding="utf-8")
    (output / "training_loss.png").write_bytes(b"png")
    state = {"best_model_checkpoint": str(output / "checkpoint-8"), "best_metric": 1.25,
             "global_step": 12, "epoch": 3.0,
             "log_history": [{"step": 4, "eval_loss": 1.5}, {"step": 8, "eval_loss": 1.25},
                             {"step": 12, "loss": 1.1}]}
    (output / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
elif config.get("do_predict"):
    dataset_dir = Path(config["dataset_dir"])
    lines = (dataset_dir / "sft_test.jsonl").read_text("utf-8").splitlines()
    test = [json.loads(line) for line in lines]
    with (output / "generated_predictions.jsonl").open("w", encoding="utf-8") as out:
        for row in test:
            label = row["conversations"][-1]["value"]
            out.write(json.dumps({"prompt": "p", "predict": "generated " + label[:10],
                                  "label": label + "\n"}) + "\n")
elif config.get("do_eval"):
    (output / "eval_results.json").write_text(json.dumps({"eval_loss": 1.2}), encoding="utf-8")
'''


def install_fake_cli(tmp_path: Path, *, max_batch: int | None = None) -> Path:
    """Put a ``llamafactory-cli`` on the PATH of :func:`script_env`; returns its call log path."""
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "fake_cli.py").write_text(FAKE_CLI, encoding="utf-8")
    wrapper = stubs / "llamafactory-cli"
    python, script = shlex.quote(sys.executable), shlex.quote(str(stubs / "fake_cli.py"))
    wrapper.write_text(f'#!/usr/bin/env bash\nexec {python} {script} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    # the script reads its neighbours through argv[0], which is the python file
    if max_batch is not None:
        (stubs / "max_batch").write_text(str(max_batch), encoding="utf-8")
    return stubs / "cli_calls.log"


def cli_calls(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text("utf-8").splitlines()]


def prepared_workdir(
    tmp_path: Path, profile: str = "5090-8b", *, pairs: int = 3, dpo_min_pairs: int = 200
) -> Path:
    """A work directory as it is after upload, setup and decrypt (data and config in place)."""
    from datetime import UTC, datetime

    from tests.support.training_data import write_synthetic_dataset
    from twin.training.layout import RemoteLayout

    workdir = tmp_path / "twin"
    install_scripts(workdir, profile, dpo_min_pairs=dpo_min_pairs)
    dataset = write_synthetic_dataset(tmp_path / "ds", pairs=pairs)
    layout_dir = str(workdir)

    result = bundle.build_bundle(
        dataset,
        PROFILES[profile],
        passphrase="correct horse battery staple",
        out_dir=tmp_path / "out",
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
        layout=RemoteLayout(layout_dir),
        dpo_min_pairs=dpo_min_pairs,
        kdf_log_n=10,
    )
    for name, content in bundle.iter_bundle(result.path, "correct horse battery staple"):
        if name.startswith(("data/", "config/", "pylib/")) or name == "manifest.json":
            target = workdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    (workdir / ".setup_done").write_text("done", encoding="utf-8")
    (workdir / "bundle.enc").write_bytes(result.path.read_bytes())
    return workdir


def yaml_value(path: Path, key: str) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))[key]
