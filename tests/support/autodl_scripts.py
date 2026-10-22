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
import shutil
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


def installed_pythons(minimum: tuple[int, int] = (3, 11)) -> list[tuple[str, str]]:
    """``(version, executable)`` of every different Python on this machine the instance could run.

    The AutoDL instance has whatever ``python3`` its image ships; LLaMA-Factory 0.9.5 needs 3.11 or
    newer, so the interpreters that matter are 3.11 and 3.12.0-3.12.6 next to the 3.12 the tests run
    under (R17-2, D-630).  The test interpreter comes first; each version appears once.
    """
    candidates = [sys.executable]
    for name in ("python3.11", "python3.12", "python3.13", "python3"):
        for directory in os.get_exec_path():  # every one on the PATH, not only the first
            found = shutil.which(name, path=directory)
            if found:
                candidates.append(found)
    seen: dict[str, str] = {}
    for executable in candidates:
        done = subprocess.run(
            [executable, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        version = done.stdout.strip()
        if done.returncode != 0 or not version:
            continue
        major, minor, _ = (int(part) for part in version.split("."))
        if (major, minor) >= minimum:
            seen.setdefault(version, executable)
    return list(seen.items())


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


DEFAULT_PROBES = """\
# the virtual environment of the tests has no jieba, nltk or rouge-chinese: the instance has them
probe_eval_packages() { :; }
# no GPU in the tests; the instance of the profiles has a Blackwell card with the right PyTorch
probe_gpu() { echo 'NVIDIA GeForce RTX 5090, 32607 MiB, 570.86.10, 12.0'; }
probe_torch() { echo '2.9.1+cu128 12.8 sm_80,sm_90,sm_120'; }
probe_torch_versions() {
    printf 'torch 2.9.1+cu128\\ntorchvision 0.24.1+cu128\\ntorchaudio 2.9.1+cu128\\n'
}
"""
"""Probes every script test replaces unless it brings its own (R17-1, R17-49b: D-631, D-632)."""


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
    defaults = tmp_path / "default_probes.sh"
    defaults.write_text(DEFAULT_PROBES, encoding="utf-8")
    env["TWIN_PROBE_OVERRIDES"] = str(defaults)
    env.update(extra)
    given = extra.get("TWIN_PROBE_OVERRIDES")
    if given:  # the probes a test replaces come on top of the defaults, not instead of them
        chained = tmp_path / "chained_probes.sh"
        chained.write_text(f". {shlex.quote(str(defaults))}\n. {shlex.quote(given)}\n", "utf-8")
        env["TWIN_PROBE_OVERRIDES"] = str(chained)
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


def write_distribution(site: Path, name: str, version: str) -> None:
    """An installed distribution for ``importlib.metadata`` (a ``.dist-info`` in ``site``)."""
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    (info / "METADATA").write_text(metadata, encoding="utf-8")


def call_setup(autodl: Path, env: dict[str, str], body: str) -> subprocess.CompletedProcess[str]:
    """Source ``setup.sh`` (it only defines its functions then), then run ``body``."""
    prologue = (
        f"set -euo pipefail; cd {shlex.quote(str(autodl))}; "
        "set -a; . ./versions.env; . ./profile.env; set +a; . ./setup.sh; "
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


def rows_of(name):
    """The rows of a dataset as LLaMA-Factory finds them: dataset_dir/dataset_info.json."""
    directory = Path(config["dataset_dir"])
    info = json.loads((directory / "dataset_info.json").read_text(encoding="utf-8"))
    if name not in info:
        sys.stderr.write(f"ValueError: Undefined dataset {name} in dataset_info.json.\n")
        sys.exit(1)
    lines = (directory / info[name]["file_name"]).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


samples = None
if config.get("dataset_dir"):
    if config.get("dataset"):
        rows = rows_of(config["dataset"])
        if config.get("max_samples"):  # dataset.select(range(max_samples)): the first rows
            rows = rows[: int(config["max_samples"])]
        samples = [row.get("id") for row in rows]
    if config.get("eval_dataset"):
        rows_of(config["eval_dataset"])
with LOG.open("a", encoding="utf-8") as log:
    log.write(json.dumps({"mode": mode, "config": config_path.name,
                          "batch": config.get("per_device_train_batch_size"),
                          "adapter": config.get("adapter_name_or_path"),
                          "dataset": config.get("dataset"),
                          "dataset_dir": config.get("dataset_dir"),
                          "eval_dataset": config.get("eval_dataset"),
                          "max_samples": config.get("max_samples"),
                          "export_device": config.get("export_device"),
                          "quantization_bit": config.get("quantization_bit"),
                          "samples": samples}) + "\n")
if config.get("predict_with_generate"):
    # LLaMA-Factory 0.9.5 (hparams/parser.py): these three are mandatory to generate replies
    installed_file = Path(sys.argv[0]).with_name("installed.txt")
    installed = set(installed_file.read_text().split()) if installed_file.exists() else set()
    for distribution in ("jieba", "nltk", "rouge-chinese"):
        if distribution not in installed:
            sys.stderr.write(f"ImportError: This program requires something that is not installed: "
                             f"{distribution}\nTo fix: run `pip install {distribution}`.\n")
            sys.exit(1)
limit_file = Path(sys.argv[0]).with_name("max_batch")
limit = int(limit_file.read_text()) if limit_file.exists() else 99
if config.get("per_device_train_batch_size", 1) > limit:
    sys.stderr.write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
    sys.exit(1)
if mode == "export":
    if config.get("adapter_name_or_path") and config.get("quantization_bit"):
        sys.stderr.write("ValueError: Cannot merge adapters to a quantized model.\n")
        sys.exit(1)
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


EVAL_PACKAGES = ("jieba", "nltk", "rouge-chinese")


def install_fake_cli(
    tmp_path: Path,
    *,
    max_batch: int | None = None,
    installed: tuple[str, ...] = EVAL_PACKAGES,
) -> Path:
    """Put a ``llamafactory-cli`` on the PATH of :func:`script_env`; returns its call log path.

    ``installed`` is what the instance's pip has installed (``stubs/installed.txt``, which
    :func:`install_stub_python` also writes): to generate replies the fake command insists on
    the three evaluation packages, as the real one does.
    """
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "installed.txt").write_text("".join(f"{name}\n" for name in installed), "utf-8")
    (stubs / "fake_cli.py").write_text(FAKE_CLI, encoding="utf-8")
    wrapper = stubs / "llamafactory-cli"
    python, script = shlex.quote(sys.executable), shlex.quote(str(stubs / "fake_cli.py"))
    wrapper.write_text(f'#!/usr/bin/env bash\nexec {python} {script} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    # the script reads its neighbours through argv[0], which is the python file
    if max_batch is not None:
        (stubs / "max_batch").write_text(str(max_batch), encoding="utf-8")
    return stubs / "cli_calls.log"


STUB_PIP = r'''
"""The pip of the instance in the tests: it records what is asked and "installs" the names."""
import json
import re
import sys
from pathlib import Path

stubs = Path(__file__).parent
arguments = sys.argv[1:]
with (stubs / "pip_calls.log").open("a", encoding="utf-8") as log:
    log.write(json.dumps(arguments) + "\n")
if arguments[:1] == ["install"]:
    takes_a_value = {"--index-url", "-i", "--extra-index-url", "-c", "--constraint", "-r"}
    names, skip = [], False
    for argument in arguments[1:]:
        if skip:
            skip = False
        elif argument in takes_a_value:
            skip = True
        elif not argument.startswith("-"):
            name = re.split(r"[<>=!~\[; ]", argument, maxsplit=1)[0]
            names.append(name.lower().replace("_", "-"))
    installed = stubs / "installed.txt"
    known = set(installed.read_text("utf-8").split()) if installed.exists() else set()
    installed.write_text("".join(f"{name}\n" for name in sorted(known | set(names))), "utf-8")
'''


def install_stub_python(tmp_path: Path) -> Path:
    """A ``TWIN_PYTHON`` whose ``-m pip`` is :data:`STUB_PIP`; everything else is this interpreter.

    Returns the path of the wrapper (give it to :func:`script_env` as ``TWIN_PYTHON``).  Calls are
    recorded in ``stubs/pip_calls.log`` (:func:`pip_calls`), installed names in ``installed.txt``.
    """
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "stub_pip.py").write_text(STUB_PIP, encoding="utf-8")
    python = shlex.quote(sys.executable)
    pip = shlex.quote(str(stubs / "stub_pip.py"))
    wrapper = stubs / "instance-python"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then\n'
        f'    exec {python} {pip} "${{@:3}}"\n'
        "fi\n"
        f'exec {python} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def pip_calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "stubs" / "pip_calls.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text("utf-8").splitlines()]


def cli_calls(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text("utf-8").splitlines()]


def prepared_workdir(
    tmp_path: Path,
    profile: str = "5090-8b",
    *,
    pairs: int = 3,
    dpo_min_pairs: int = 200,
    train: int = 40,
) -> Path:
    """A work directory as it is after upload, setup and decrypt (data and config in place)."""
    from datetime import UTC, datetime

    from tests.support.training_data import write_synthetic_dataset
    from twin.training.layout import RemoteLayout

    workdir = tmp_path / "twin"
    install_scripts(workdir, profile, dpo_min_pairs=dpo_min_pairs)
    dataset = write_synthetic_dataset(tmp_path / "ds", train=train, pairs=pairs)
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
