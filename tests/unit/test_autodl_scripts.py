"""R-TRN-009: the AutoDL shell scripts (``training/autodl``).

Static checks (shellcheck, structure) run everywhere.  The scripts themselves are bash programs
for the Linux instance, so the tests that run them are POSIX-only (``posix_only``); a fake
``llamafactory-cli`` and fake ``llama.cpp`` tools stand in for the GPU programs and leave the files
the real ones leave, so the scripts' own logic runs for real.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.support.autodl_scripts import (
    AUTODL,
    ROOT,
    call_function,
    call_setup,
    cli_calls,
    install_fake_cli,
    install_scripts,
    install_stub_python,
    pip_calls,
    posix_only,
    prepared_workdir,
    run_script,
    script_env,
    write_distribution,
    write_tiny_adapter,
    yaml_value,
)
from tests.support.training_data import write_synthetic_dataset
from twin.training import bundle
from twin.training.profiles import PROFILES, QUANT_FACTORS
from twin.training.versions import JIEBA, NLTK, ROUGE_CHINESE

PASSPHRASE = "correct horse battery staple"
SCRIPTS = sorted(path.name for path in AUTODL.glob("*.sh"))
EXECUTABLE_SCRIPTS = [name for name in SCRIPTS if name != "lib.sh"]
SNAPSHOT = ROOT / "tests" / "fixtures" / "training" / "export_plan.txt"


def shellcheck_command() -> list[str]:
    found = shutil.which("shellcheck")
    if found:
        return [found]
    for name in ("shellcheck", "shellcheck.exe"):
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return [str(candidate)]
    pytest.fail("shellcheck-py is a dev dependency but no shellcheck executable was found")


# ------------------------------------------------------------------------ static checks


def test_all_scripts_are_found() -> None:
    assert set(SCRIPTS) == {
        "lib.sh",
        "setup.sh",
        "decrypt.sh",
        "train.sh",
        "eval.sh",
        "dpo.sh",
        "export.sh",
        "serve_vllm.sh",
        "cleanup.sh",
    }


def test_every_script_passes_shellcheck() -> None:
    done = subprocess.run(
        [*shellcheck_command(), "--external-sources", "--shell=bash", *SCRIPTS],
        cwd=AUTODL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("name", EXECUTABLE_SCRIPTS)
def test_every_script_stops_on_errors_and_unset_variables(name: str) -> None:
    text = (AUTODL / name).read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "\nset -euo pipefail\n" in text
    assert "\r" not in text


def test_the_library_is_sourced_and_marks_itself_as_bash() -> None:
    text = (AUTODL / "lib.sh").read_text(encoding="utf-8")
    assert text.startswith("# shellcheck shell=bash\n") and "set -e" not in text


def test_scripts_never_carry_a_passphrase_or_password_argument() -> None:
    for name in SCRIPTS:
        text = (AUTODL / name).read_text(encoding="utf-8")
        assert "--passphrase" not in text and "--password" not in text
    decrypt = (AUTODL / "decrypt.sh").read_text(encoding="utf-8")
    assert "standard input" in decrypt


def test_every_step_of_the_spec_has_its_script() -> None:
    for name in ("setup", "decrypt", "train", "eval", "dpo", "export", "serve_vllm", "cleanup"):
        assert (AUTODL / f"{name}.sh").is_file()


# --------------------------------------------------------------- functions of lib.sh


@pytest.fixture
def env_for(tmp_path: Path):  # type: ignore[no-untyped-def]
    def make(profile: str):  # type: ignore[no-untyped-def]
        workdir = tmp_path / "twin"
        autodl = install_scripts(workdir, profile)
        return autodl, script_env(workdir, tmp_path)

    return make


@posix_only
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("2.9.1", "2.7", 0),
        ("2.7", "2.7", 0),
        ("2.7.0+cu128", "2.7", 0),
        ("2.6.0", "2.7", 1),
        ("2.10.0", "2.9.1", 0),
        ("2.9.1", "2.10.0", 1),
        ("12.8", "12.8", 0),
    ],
)
def test_version_comparison(env_for, a: str, b: str, expected: int) -> None:  # type: ignore[no-untyped-def]
    autodl, env = env_for("5090-8b")
    assert call_function(autodl, env, f"version_ge '{a}' '{b}'").returncode == expected


@posix_only
@pytest.mark.parametrize("name", list(PROFILES))
def test_the_disk_check_per_profile_and_free_space(env_for, name: str) -> None:  # type: ignore[no-untyped-def]
    profile = PROFILES[name]
    autodl, env = env_for(name)
    need, spec = profile.required_disk_gb, profile.spec_disk_gb
    enough = call_function(autodl, env, f"check_disk_space {need} {need} {spec}")
    assert enough.returncode == 0 and "ok" in enough.stdout
    for free in (need - 1, 50, 0):
        short = call_function(autodl, env, f"check_disk_space {free} {need} {spec}")
        assert short.returncode == 1
        assert f"only {free} GB" in short.stderr and f"{need} GB" in short.stderr
        assert "请到 AutoDL 控制台扩容数据盘/换实例" in short.stderr
        assert f"at least {spec} GB" in short.stderr
    assert call_function(autodl, env, f"check_disk_space {spec} {need} {spec}").returncode == 0


@posix_only
def test_the_documented_disk_sizes_are_enough_for_every_profile(env_for) -> None:  # type: ignore[no-untyped-def]
    for name, profile in PROFILES.items():
        autodl, env = env_for(name)
        done = call_function(
            autodl,
            env,
            f"check_disk_space {profile.spec_disk_gb} $TWIN_REQUIRED_DISK_GB $TWIN_SPEC_DISK_GB",
        )
        assert done.returncode == 0, name


def write_df_stub(tmp_path: Path, size_bytes: int, available_bytes: int) -> None:
    """A ``df -Pk`` that reports a data disk of ``size_bytes`` with ``available_bytes`` free."""
    stub = tmp_path / "stubs" / "df"
    stub.parent.mkdir(exist_ok=True)
    size, available = size_bytes // 1024, available_bytes // 1024
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
        f"echo '/dev/vdb {size} {size - available} {available} 2% /root/autodl-tmp'\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)


# What AutoDL calls a "GB" of the data disk is not documented: decimal or binary, the check must
# pass on a disk expanded to the number the user is told (R18-E4).
AUTODL_DISK_UNITS = (10**9, 2**30)


@posix_only
@pytest.mark.parametrize("name", list(PROFILES))
@pytest.mark.parametrize("unit", AUTODL_DISK_UNITS)
def test_a_disk_expanded_to_the_size_the_user_is_told_passes_the_check(
    env_for,  # type: ignore[no-untyped-def]
    tmp_path: Path,
    name: str,
    unit: int,
) -> None:
    """R18-E4 (D-811): the free space is measured in the unit of the requirement (10^9 bytes).

    ``probe_free_gb`` gave whole GiB, ``required_disk_gb`` is decimal GB, and the size the user
    was told to expand the disk to (``spec_disk_gb``) was the requirement itself: a fresh disk of
    exactly that size (98 % of it available) could never pass for the 14B profiles, and with a
    decimal "GB" not for 8B either.
    """
    profile = PROFILES[name]
    autodl, env = env_for(name)
    env["TWIN_PROBE_OVERRIDES"] = write_overrides(
        tmp_path, probe_present_gb="echo 0", probe_ram_gb="echo 999"
    )
    size = profile.spec_disk_gb * unit
    write_df_stub(tmp_path, size, int(size * 0.98))
    done = call_function(autodl, env, "run_resource_checks")
    assert done.returncode == 0, done.stdout + done.stderr


@posix_only
@pytest.mark.parametrize("name", list(PROFILES))
def test_a_disk_that_is_short_fails_and_the_message_names_the_unit(
    env_for,  # type: ignore[no-untyped-def]
    tmp_path: Path,
    name: str,
) -> None:
    """R18-E4 (D-811): one decimal GB short of the requirement is short, and the text says so."""
    profile = PROFILES[name]
    autodl, env = env_for(name)
    env["TWIN_PROBE_OVERRIDES"] = write_overrides(
        tmp_path, probe_present_gb="echo 0", probe_ram_gb="echo 999"
    )
    available = (profile.required_disk_gb - 1) * 10**9 + 5 * 10**8
    write_df_stub(tmp_path, profile.spec_disk_gb * 10**9, available)
    done = call_function(autodl, env, "run_resource_checks")
    assert done.returncode == 1
    assert f"only {profile.required_disk_gb - 1} GB" in done.stderr
    assert "1 GB = 10^9 bytes" in done.stderr
    assert f"at least {profile.spec_disk_gb} GB" in done.stderr


@posix_only
def test_the_free_space_and_the_project_size_are_decimal_gigabytes(  # type: ignore[no-untyped-def]
    env_for, tmp_path: Path
) -> None:
    """R18-E4 (D-811): ``df`` and ``du`` count KiB; the probes answer in 10^9 bytes."""
    autodl, env = env_for("5090-14b")
    write_df_stub(tmp_path, 120 * 10**9, 107_800_000_000)
    write_du_stub(tmp_path, 12_600_000_000)  # TWIN_HOME and TWIN_MODELS: 25.2e9 together
    (tmp_path / "models").mkdir()
    done = call_function(autodl, env, "probe_free_gb /root/autodl-tmp; probe_present_gb")
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["107", "25"]


@posix_only
@pytest.mark.parametrize("name", list(PROFILES))
def test_the_memory_check_per_profile(env_for, name: str) -> None:  # type: ignore[no-untyped-def]
    need = PROFILES[name].required_ram_gb
    autodl, env = env_for(name)
    assert call_function(autodl, env, f"check_memory {need} {need}").returncode == 0
    short = call_function(autodl, env, f"check_memory {need - 1} {need}")
    assert short.returncode == 1 and f"{need} GB" in short.stderr
    assert "请到 AutoDL 控制台" in short.stderr


def write_overrides(tmp_path: Path, **probes: str) -> str:
    path = tmp_path / "probes.sh"
    path.write_text("".join(f"{name}() {{ {body}; }}\n" for name, body in probes.items()), "utf-8")
    return str(path)


@posix_only
@pytest.mark.parametrize(
    ("name", "free", "present", "ram", "ok"),
    [
        ("5090-8b", 80, 0, 64, True),
        ("5090-8b", 67, 0, 64, False),
        ("5090-8b", 40, 30, 64, True),  # an interrupted setup: what is on the disk counts
        ("5090-8b", 80, 0, 10, False),
        ("5090-14b", 109, 0, 128, False),
        ("5090-14b", 110, 0, 42, False),
        ("5090-14b", 110, 0, 43, True),
        ("pro6000-32b", 224, 0, 200, False),
        ("pro6000-32b", 100, 130, 200, True),
    ],
)
def test_the_combined_resource_check(  # type: ignore[no-untyped-def]
    env_for, tmp_path: Path, name: str, free: int, present: int, ram: int, ok: bool
) -> None:
    autodl, env = env_for(name)
    env["TWIN_PROBE_OVERRIDES"] = write_overrides(
        tmp_path,
        probe_free_gb=f"echo {free}",
        probe_present_gb=f"echo {present}",
        probe_ram_gb=f"echo {ram}",
    )
    done = call_function(autodl, env, "run_resource_checks")
    assert (done.returncode == 0) is ok, done.stderr


@posix_only
@pytest.mark.parametrize(
    ("torch_version", "cuda", "archs", "capability", "expected"),
    [
        ("2.9.1+cu128", "12.8", "sm_80,sm_90,sm_120", "12.0", "ok"),
        ("2.7.0", "12.8", "sm_90,sm_120", "12.0", "ok"),
        ("2.6.0", "12.6", "sm_80,sm_90", "12.0", "reinstall: Blackwell needs torch >= 2.7"),
        (
            "2.8.0",
            "12.8",
            "sm_80,sm_90",
            "12.0",
            "reinstall: this PyTorch build does not list sm_120",
        ),
        ("2.8.0", "12.6", "sm_80,sm_120", "12.0", "reinstall: Blackwell needs a CUDA 12.8 build"),
        ("2.9.0", "12.9", "sm_80,sm_120", "12.0", "reinstall: Blackwell needs a CUDA 12.8 build"),
        ("2.11.0", "13.0", "sm_80,sm_120", "12.0", "reinstall: CUDA 13.0 builds are not used"),
        ("2.9.0", "13", "sm_120", "9.0", "reinstall: CUDA 13 builds are not used"),
        ("2.5.1", "12.4", "sm_80,sm_89", "8.9", "ok"),
        ("2.3.0", "12.1", "sm_80,sm_89", "8.9", "reinstall: LLaMA-Factory needs torch >= 2.4"),
        ("none", "none", "", "12.0", "reinstall: PyTorch is not installed"),
        ("2.9.1+cpu", "none", "", "12.0", "reinstall: PyTorch has no CUDA build"),
    ],
)
def test_the_torch_stack_verdict(  # type: ignore[no-untyped-def]
    env_for, torch_version: str, cuda: str, archs: str, capability: str, expected: str
) -> None:
    autodl, env = env_for("5090-8b")
    done = call_function(
        autodl,
        env,
        f"torch_stack_verdict '{torch_version}' '{cuda}' '{archs}' '{capability}'",
    )
    assert done.returncode == 0 and done.stdout.strip().startswith(expected), done.stdout


@posix_only
def test_the_gpu_and_torch_report_feeds_the_verdict(env_for, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    autodl, env = env_for("5090-8b")
    env["TWIN_PROBE_OVERRIDES"] = write_overrides(
        tmp_path,
        probe_gpu='echo "NVIDIA GeForce RTX 5090, 32607 MiB, 570.86.10, 12.0"',
        probe_torch='echo "2.8.0+cu128 12.8 sm_80,sm_90,sm_120"',
    )
    done = call_function(autodl, env, "run_torch_check")
    assert done.stdout.strip() == "ok"
    assert "RTX 5090" in done.stderr and "sm_120" in done.stderr
    env["TWIN_PROBE_OVERRIDES"] = write_overrides(
        tmp_path,
        probe_gpu='echo "NVIDIA GeForce RTX 5090, 32607 MiB, 570.86.10, 12.0"',
        probe_torch='echo "2.6.0 12.6 sm_80,sm_90"',
    )
    done = call_function(autodl, env, "run_torch_check")
    assert done.stdout.startswith("reinstall: Blackwell needs torch >= 2.7")


@posix_only
@pytest.mark.parametrize(
    ("batch", "accum", "expected"), [(4, 4, "2 8"), (2, 8, "1 16"), (8, 2, "4 4")]
)
def test_the_batch_is_halved_and_the_accumulation_doubled(  # type: ignore[no-untyped-def]
    env_for, batch: int, accum: int, expected: str
) -> None:
    autodl, env = env_for("5090-8b")
    done = call_function(autodl, env, f"next_smaller_batch {batch} {accum}")
    assert done.stdout.strip() == expected and done.returncode == 0
    assert call_function(autodl, env, "next_smaller_batch 1 16").returncode == 1


@posix_only
def test_yaml_keys_are_read_replaced_appended_and_dropped(env_for, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    autodl, env = env_for("5090-8b")
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\nab: 2\nb: two words\n# c: 3\n", encoding="utf-8")
    body = (
        f'f={path}; yaml_set "$f" a 10; yaml_set "$f" c \'"no"\'; yaml_drop "$f" b; '
        'yaml_get "$f" a; yaml_get "$f" ab'
    )
    done = call_function(autodl, env, body)
    assert done.stdout.split() == ["10", "2"]
    assert path.read_text("utf-8") == 'a: 10\nab: 2\n# c: 3\nc: "no"\n'


@posix_only
@pytest.mark.parametrize(
    ("profile", "dpo_stack", "layers"),
    [
        ("5090-8b", "{w}/output/sft,{w}/output/dpo", ["sft", "dpo"]),
        ("pro6000-14b", "{w}/output/sft,{w}/output/dpo", ["sft", "dpo"]),
        ("pro6000-32b", "{w}/output/sft,{w}/output/dpo", ["sft", "dpo"]),
        ("5090-14b", "{w}/output/dpo", ["dpo"]),
    ],
)
def test_the_adapter_stack_of_dpo_is_both_adapters_or_the_continued_one(  # type: ignore[no-untyped-def]
    env_for, profile: str, dpo_stack: str, layers: list[str]
) -> None:
    """R17-49d (D-650, D-652): ``sft,dpo`` where DPO is a delta over the merged SFT model.

    A bf16 profile trains a new adapter on the SFT model, so the model is both adapters, in this
    order (LLaMA-Factory merges ``adapter_name_or_path`` front to back); the 4-bit profile
    continues the SFT adapter, so ``dpo`` alone is the stack - listing ``sft`` first would count
    the SFT weights twice, and a quantised model only takes one adapter anyway.
    """
    autodl, env = env_for(profile)
    work = Path(env["TWIN_HOME"])
    assert call_function(autodl, env, "stack_adapters sft").stdout.strip() == f"{work}/output/sft"
    assert call_function(autodl, env, "stack_adapters dpo").stdout.strip() == dpo_stack.format(
        w=work
    )
    assert call_function(autodl, env, "stack_layers dpo").stdout.split() == layers
    assert call_function(autodl, env, "stack_layers sft").stdout.split() == ["sft"]
    assert call_function(autodl, env, "stack_adapters bogus").returncode == 1


@posix_only
def test_the_stack_that_was_kept_needs_a_verdict_once_there_is_a_dpo_adapter(  # type: ignore[no-untyped-def]
    env_for,
) -> None:
    """R17-49d (D-655): no DPO adapter - nothing to choose; with one, the evaluation decides."""
    autodl, env = env_for("5090-8b")
    work = Path(env["TWIN_HOME"])
    assert call_function(autodl, env, "stack_choice auto").stdout.strip() == "sft"
    (work / "output" / "dpo").mkdir(parents=True)
    (work / "output" / "dpo" / "adapter_model.safetensors").write_bytes(b"x")
    assert call_function(autodl, env, "stack_choice auto").returncode == 1  # no verdict yet
    (work / "state").mkdir()
    (work / "state" / "stack_choice").write_text("maybe\n", encoding="utf-8")
    assert call_function(autodl, env, "stack_choice auto").returncode == 1  # not a verdict
    for word in ("sft", "dpo"):
        (work / "state" / "stack_choice").write_text(f"{word}\n", encoding="utf-8")
        assert call_function(autodl, env, "stack_choice auto").stdout.strip() == word
    assert call_function(autodl, env, "stack_choice dpo").stdout.strip() == "dpo"
    assert call_function(autodl, env, "stack_choice sft").stdout.strip() == "sft"
    assert call_function(autodl, env, "stack_choice bogus").returncode == 1


@posix_only
def test_out_of_memory_is_recognised_in_a_log(env_for, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    autodl, env = env_for("5090-8b")
    oom = tmp_path / "oom.log"
    oom.write_text("...\ntorch.OutOfMemoryError: CUDA out of memory.\n", encoding="utf-8")
    fine = tmp_path / "fine.log"
    fine.write_text("loss 1.0\n", encoding="utf-8")
    assert call_function(autodl, env, f"oom_in_log {oom}").returncode == 0
    assert call_function(autodl, env, f"oom_in_log {fine}").returncode == 1


@posix_only
@pytest.mark.parametrize(
    ("found", "ok"),
    [
        ("3.11.0", True),
        ("3.11.17", True),
        ("3.12.3", True),
        ("3.13.1", True),
        ("3.10.12", False),
        ("3.9.18", False),
        ("3.8.10", False),
    ],
)
def test_the_python_of_the_instance_must_be_new_enough_for_llamafactory(  # type: ignore[no-untyped-def]
    env_for, found: str, ok: bool
) -> None:
    """R17-2 (D-630): LLaMA-Factory 0.9.5 needs Python >= 3.11 (``TWIN_PYTHON_MIN``)."""
    autodl, env = env_for("5090-8b")
    done = call_function(autodl, env, f"check_python_version {found} $TWIN_PYTHON_MIN")
    assert (done.returncode == 0) is ok, done.stderr
    if not ok:
        assert f"is {found}" in done.stderr and "3.11 or newer" in done.stderr
        assert "请在 AutoDL 控制台换带 Python 3.11 及以上的镜像" in done.stderr


@posix_only
def test_the_python_version_is_probed_and_recorded(env_for, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """R17-2 (D-630): ``state/python.version`` says which Python the run used."""
    autodl, env = env_for("5090-8b")
    real = call_function(autodl, env, "probe_python_version").stdout.strip()
    assert real == ".".join(str(part) for part in sys.version_info[:3])
    env["TWIN_PROBE_OVERRIDES"] = write_overrides(tmp_path, probe_python_version="echo 3.12.3")
    done = call_function(autodl, env, "run_python_check")
    assert done.returncode == 0, done.stderr
    assert (Path(env["TWIN_HOME"]) / "state" / "python.version").read_text("utf-8") == "3.12.3\n"
    env["TWIN_PROBE_OVERRIDES"] = write_overrides(tmp_path, probe_python_version="echo 3.10.4")
    old = call_function(autodl, env, "run_python_check")
    assert old.returncode == 1
    assert (Path(env["TWIN_HOME"]) / "state" / "python.version").read_text("utf-8") == "3.12.3\n"


@posix_only
def test_setup_install_stops_on_an_old_python_before_it_checks_or_installs_anything(
    tmp_path: Path,
) -> None:
    """R17-2 (D-630): the cheapest check is the first one; nothing is installed on a 3.10 image."""
    workdir = tmp_path / "twin"
    install_scripts(workdir, "5090-8b")
    env = script_env(
        workdir,
        tmp_path,
        TWIN_PROBE_OVERRIDES=write_overrides(
            tmp_path,
            probe_python_version="echo 3.10.12",
            # if the order ever changes, the resource check fails before anything is installed
            probe_free_gb="echo 0",
            probe_present_gb="echo 0",
            probe_ram_gb="echo 0",
        ),
    )
    done = run_script(workdir, env, "setup.sh", "5090-8b", "install")
    assert done.returncode == 1
    assert "needs Python 3.11 or newer" in done.stderr and "disk:" not in done.stderr
    assert not (workdir / "state" / "python.version").exists()


# ------------------------------------------------------------------------- export plan


@posix_only
def test_the_export_dry_run_prints_the_steps_in_the_deletion_safe_order(tmp_path: Path) -> None:
    workdir = tmp_path / "twin"
    install_scripts(workdir, "5090-14b")
    done = run_script(workdir, script_env(workdir, tmp_path), "export.sh", "5090-14b", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert done.stdout == SNAPSHOT.read_text(encoding="utf-8")
    steps = [line.split(": ", 1)[1].split(" - ")[0] for line in done.stdout.splitlines()]
    assert steps.index("delete_merged_hf") > steps.index("convert_f16")
    assert steps.index("delete_merged_hf") < steps.index("quantize")
    assert steps.index("delete_f16") > steps.index("quantize")
    assert not (workdir / "state").exists() and not (workdir / "output").exists()


@posix_only
def test_a_script_refuses_the_wrong_profile_and_unknown_arguments(tmp_path: Path) -> None:
    workdir = tmp_path / "twin"
    install_scripts(workdir, "5090-8b")
    env = script_env(workdir, tmp_path)
    wrong = run_script(workdir, env, "export.sh", "pro6000-32b", "--dry-run")
    assert wrong.returncode == 1 and "profile.env is for '5090-8b'" in wrong.stderr
    missing = run_script(workdir, env, "export.sh")
    assert missing.returncode == 1 and "usage" in missing.stderr
    odd = run_script(workdir, env, "export.sh", "5090-8b", "--bogus")
    assert odd.returncode == 1 and "unknown argument" in odd.stderr
    mode = run_script(workdir, env, "setup.sh", "5090-8b", "bogus")
    assert mode.returncode == 1 and "unknown mode" in mode.stderr


@posix_only
@pytest.mark.parametrize("name", list(PROFILES))
def test_the_vllm_command_listens_locally_and_serves_the_adapter(tmp_path: Path, name: str) -> None:
    profile = PROFILES[name]
    workdir = tmp_path / "twin"
    install_scripts(workdir, name)
    done = run_script(workdir, script_env(workdir, tmp_path), "serve_vllm.sh", name, "--dry-run")
    assert done.returncode == 0, done.stderr
    command = done.stdout.split()
    assert command[0] == "serve" and command[1].endswith(profile.base_model)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--enable-lora" in command
    assert command[command.index("--lora-modules") + 1].startswith("twin-style=")
    assert command[command.index("--max-lora-rank") + 1] == str(profile.lora_rank)
    assert ("--quantization" in command) is (profile.quantization_bit is not None)


@posix_only
@pytest.mark.parametrize(("name", "layers", "rank"), [("5090-8b", 2, 64), ("pro6000-32b", 2, 32)])
def test_the_vllm_rank_limit_is_the_rank_of_the_adapter_that_is_served(
    tmp_path: Path, name: str, layers: int, rank: int
) -> None:
    """R17-49d (D-652): the composed adapter has the sum of the ranks (the profile's: a layer's)."""
    profile = PROFILES[name]
    workdir = tmp_path / "twin"
    install_scripts(workdir, name)
    write_tiny_adapter(workdir / "artifacts" / "adapter", rank=profile.lora_rank * layers, alpha=1)
    done = run_script(workdir, script_env(workdir, tmp_path), "serve_vllm.sh", name, "--dry-run")
    assert done.returncode == 0, done.stderr
    command = done.stdout.split()
    assert command[command.index("--max-lora-rank") + 1] == str(rank) == str(profile.lora_rank * 2)


# ------------------------------------------------------------------------ setup verify


@posix_only
def test_the_verify_step_stops_with_an_error_when_the_template_check_is_missing(
    tmp_path: Path,
) -> None:
    workdir = prepared_workdir(tmp_path)
    shutil.rmtree(workdir / "pylib" / "twin" / "training" / "__pycache__", ignore_errors=True)
    # a package from before round 13b: the template check is not among its modules
    (workdir / "pylib" / "twin" / "training" / "parity_check.py").unlink()
    log = install_fake_cli(tmp_path)
    done = run_script(workdir, script_env(workdir, tmp_path), "setup.sh", "5090-8b", "verify")
    assert done.returncode == 1
    assert "twin.training.parity_check" in done.stderr and "not in this package" in done.stderr
    assert cli_calls(log) == []  # the trial step never ran: the check is not skipped


def add_template_check(workdir: Path, exit_code: int) -> None:
    module = workdir / "pylib" / "twin" / "training" / "parity_check.py"
    module.write_text(
        "import sys\n"
        "arguments = sys.argv[1:]\n"
        "assert '--model-dir' in arguments and '--cases' in arguments\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )


@posix_only
def test_a_failing_template_check_fails_the_verification(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path)
    add_template_check(workdir, 3)
    log = install_fake_cli(tmp_path)
    done = run_script(workdir, script_env(workdir, tmp_path), "setup.sh", "5090-8b", "verify")
    assert done.returncode != 0 and cli_calls(log) == []


@posix_only
def test_the_trial_step_lowers_the_batch_until_it_fits_and_keeps_the_layout(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path, "pro6000-14b")
    add_template_check(workdir, 0)
    log = install_fake_cli(tmp_path, max_batch=1)
    done = run_script(workdir, script_env(workdir, tmp_path), "setup.sh", "pro6000-14b", "verify")
    assert done.returncode == 0, done.stderr + done.stdout
    assert [call["batch"] for call in cli_calls(log)] == [4, 2, 1]
    config = workdir / "config" / "sft.yaml"
    text = config.read_text("utf-8")
    assert "per_device_train_batch_size: 1\n" in text
    assert "gradient_accumulation_steps: 16\n" in text and "per_device_eval_batch_size: 1\n" in text
    assert not (workdir / "config" / "trial.yaml").exists()
    assert not (workdir / "output" / "trial").exists()


@posix_only
def test_the_trial_step_gives_up_when_a_batch_of_one_does_not_fit(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path, "5090-8b")
    add_template_check(workdir, 0)
    log = install_fake_cli(tmp_path, max_batch=0)
    done = run_script(workdir, script_env(workdir, tmp_path), "setup.sh", "5090-8b", "verify")
    assert done.returncode == 1 and "even with a batch of 1" in done.stderr
    assert [call["batch"] for call in cli_calls(log)] == [2, 1]
    assert not (workdir / "trial_data").exists()  # no copy of her messages is left behind


def sample_length(row: dict[str, Any]) -> int:
    return len(row["system"]) + sum(len(turn["value"]) for turn in row["conversations"])


@posix_only
def test_the_trial_step_trains_on_the_longest_samples_not_the_earliest(tmp_path: Path) -> None:
    """R17-49c (D-633): the trial decides the batch size, and memory is decided by length.

    ``max_samples: 64`` makes LLaMA-Factory take the first 64 rows of the file - the earliest
    messages, whatever their length - so a batch that only fits short samples passed the trial and
    ran out of memory hours into the training.
    """
    workdir = prepared_workdir(tmp_path, "5090-8b", train=100)
    add_template_check(workdir, 0)
    log = install_fake_cli(tmp_path)
    done = run_script(workdir, script_env(workdir, tmp_path), "setup.sh", "5090-8b", "verify")
    assert done.returncode == 0, done.stderr + done.stdout
    trial = cli_calls(log)[0]
    assert trial["config"] == "trial.yaml" and trial["batch"] is not None
    train_lines = (workdir / "data" / "sft_train.jsonl").read_text("utf-8").splitlines()
    lengths = {row["id"]: sample_length(row) for row in map(json.loads, train_lines)}
    taken = trial["samples"]
    assert len(taken) == 64 and len(set(taken)) == 64 and set(taken) <= set(lengths)
    left = [name for name in lengths if name not in set(taken)]
    assert min(lengths[name] for name in taken) >= max(lengths[name] for name in left)
    assert trial["max_samples"] is None and trial["eval_dataset"] is None
    assert trial["dataset"] == "twin_sft_trial" and trial["dataset_dir"] != str(workdir / "data")
    # the real training set is what the layout of the batch is written back to, untouched
    assert yaml_value(workdir / "config" / "sft.yaml", "dataset") == "twin_sft_train"
    assert (
        not (workdir / "trial_data").exists() and not (workdir / "config" / "trial.yaml").exists()
    )


@posix_only
def test_a_training_set_smaller_than_the_trial_is_used_whole(tmp_path: Path) -> None:
    """R17-49c (D-633): 40 samples are all 40 samples (the default synthetic set)."""
    workdir = prepared_workdir(tmp_path, "5090-8b")
    add_template_check(workdir, 0)
    log = install_fake_cli(tmp_path)
    done = run_script(workdir, script_env(workdir, tmp_path), "setup.sh", "5090-8b", "verify")
    assert done.returncode == 0, done.stderr
    assert len(cli_calls(log)[0]["samples"]) == 40


@posix_only
def test_the_verify_step_needs_the_data_and_a_finished_setup(tmp_path: Path) -> None:
    workdir = tmp_path / "twin"
    install_scripts(workdir, "5090-8b")
    env = script_env(workdir, tmp_path)
    assert (
        "run decrypt.sh first" in run_script(workdir, env, "setup.sh", "5090-8b", "verify").stderr
    )
    (workdir / "data").mkdir()
    (workdir / "data" / "dataset_info.json").write_text("{}", encoding="utf-8")
    assert "install first" in run_script(workdir, env, "setup.sh", "5090-8b", "verify").stderr


# ------------------------------------------------- the packages of the evaluation (R17-1)

SETUP_WITHOUT_THE_DOWNLOAD = (
    # the instance's PyTorch is fine, the GPU is there, the disk and memory are plenty; the
    # download of the base model (tens of GB) and zstd's apt-get are the only things not run
    "probe_gpu() { echo 'NVIDIA GeForce RTX 5090, 32607 MiB, 570.86.10, 12.0'; }; "
    "probe_torch() { echo '2.9.1+cu128 12.8 sm_80,sm_90,sm_120'; }; "
    "probe_free_gb() { echo 900; }; probe_present_gb() { echo 0; }; probe_ram_gb() { echo 256; }; "
    "ensure_zstd() { :; }; download_base_model() { :; }; "
)


@posix_only
def test_install_packages_installs_what_llamafactory_needs_for_generation(tmp_path: Path) -> None:
    """R17-1 (D-631): jieba, nltk and rouge-chinese are not in the wheel's dependencies.

    LLaMA-Factory 0.9.5 checks them as mandatory when ``predict_with_generate`` is set (eval.sh),
    ``DISABLE_VERSION_CHECK`` does not skip the check, and ``install_packages`` installed none.
    """
    workdir = tmp_path / "twin"
    autodl = install_scripts(workdir, "5090-8b")
    env = script_env(workdir, tmp_path, TWIN_PYTHON=str(install_stub_python(tmp_path)))
    done = call_setup(autodl, env, "install_packages")
    assert done.returncode == 0, done.stderr
    requested = {
        argument for call in pip_calls(tmp_path) if call[:1] == ["install"] for argument in call[1:]
    }
    assert {f"jieba=={JIEBA}", f"nltk=={NLTK}", f"rouge-chinese=={ROUGE_CHINESE}"} <= requested
    assert "llamafactory==0.9.5" in requested and "zstandard==0.25.0" in requested


@posix_only
def test_eval_fails_without_the_generation_dependencies_and_passes_with_them(
    tmp_path: Path,
) -> None:
    """R17-1 (D-631): the setup of the instance leaves what the second step of eval.sh needs.

    The fake ``llamafactory-cli`` refuses ``predict_with_generate`` when one of the three
    packages is not installed (an ImportError with "To fix: run pip install ...", as the real one
    does); the pip of the instance is a recorder, so what ``setup.sh install`` asks pip for is
    what is "installed".
    """
    workdir = prepared_workdir(tmp_path)
    autodl = workdir / "autodl"
    install_fake_cli(tmp_path, installed=())
    env = script_env(workdir, tmp_path, TWIN_PYTHON=str(install_stub_python(tmp_path)))
    assert run_script(workdir, env, "train.sh", "5090-8b").returncode == 0
    # an instance set up by an earlier version of the scripts: eval.sh dies in its second step
    stale = run_script(workdir, env, "eval.sh", "5090-8b")
    assert stale.returncode != 0 and "pip install jieba" in stale.stderr
    done = call_setup(autodl, env, SETUP_WITHOUT_THE_DOWNLOAD + "run_install")
    assert done.returncode == 0, done.stderr + done.stdout
    assert (workdir / "state" / "python.version").is_file()
    fixed = run_script(workdir, env, "eval.sh", "5090-8b")
    assert fixed.returncode == 0, fixed.stderr
    assert (workdir / "artifacts" / "eval" / "sft" / "eval_generations.jsonl").is_file()


@posix_only
def test_the_probe_names_the_evaluation_packages_that_are_missing(tmp_path: Path) -> None:
    """R17-1 (D-631): the real probe, on a site directory with distributions (no overrides)."""
    workdir = tmp_path / "twin"
    autodl = install_scripts(workdir, "5090-8b")
    site = tmp_path / "site"
    env = script_env(workdir, tmp_path, PYTHONPATH=str(site))
    del env["TWIN_PROBE_OVERRIDES"]  # the tests' default stands in for the probe; this one is real

    def install(name: str) -> None:
        write_distribution(site, name, "1.0")

    site.mkdir()
    everything = call_function(autodl, env, "probe_eval_packages")
    assert everything.stdout.split() == ["jieba", "nltk", "rouge-chinese"], everything.stderr
    install("nltk")
    assert call_function(autodl, env, "probe_eval_packages").stdout.split() == [
        "jieba",
        "rouge-chinese",
    ]
    install("jieba")
    install("rouge-chinese")
    complete = call_function(autodl, env, "probe_eval_packages; run_eval_packages_check")
    assert complete.returncode == 0, complete.stderr
    assert complete.stdout.count("evaluation packages") == 1  # the probe itself printed nothing


@posix_only
def test_verify_names_the_missing_generation_packages(tmp_path: Path) -> None:
    """R17-1 (D-631): the missing package stops the verification, not hours of work later."""
    workdir = prepared_workdir(tmp_path)
    add_template_check(workdir, 0)
    log = install_fake_cli(tmp_path)
    env = script_env(
        workdir,
        tmp_path,
        TWIN_PROBE_OVERRIDES=write_overrides(
            tmp_path, probe_eval_packages="echo nltk; echo rouge-chinese"
        ),
    )
    done = run_script(workdir, env, "setup.sh", "5090-8b", "verify")
    assert done.returncode == 1
    assert "needs nltk, rouge-chinese to generate the replies" in done.stderr
    assert "run setup.sh 5090-8b install again" in done.stderr
    assert cli_calls(log) == []  # neither the template check nor the trial step ran


@posix_only
def test_sourcing_setup_defines_the_functions_and_runs_nothing(tmp_path: Path) -> None:
    """R17-1 (D-631): the guard of ``main``, which lets the tests call ``install_packages``."""
    workdir = tmp_path / "twin"
    autodl = install_scripts(workdir, "5090-8b")
    env = script_env(workdir, tmp_path)
    names = ["install_packages", "run_install", "run_verify", "main"]
    done = call_setup(autodl, env, "declare -F " + " ".join(names))
    assert done.returncode == 0, done.stderr
    assert [line.split()[-1] for line in done.stdout.splitlines()] == names


# Bytes of the bf16 safetensors of the base models (the file list of the Hugging Face repository;
# ModelScope has the same files).  ``base_gb`` of the profiles is this number in decimal gigabytes.
REAL_BASE_BYTES = {
    "Qwen/Qwen3-8B": 16_397_461_266,
    "Qwen/Qwen3-14B": 29_552_613_776,
    "Qwen/Qwen3-32B": 65_540_298_478,
}


def write_du_stub(tmp_path: Path, size_bytes: int) -> None:
    """A ``du`` that reports a directory of ``size_bytes`` (``-sk``: KiB, ``-sb``: bytes)."""
    stub = tmp_path / "stubs" / "du"
    stub.parent.mkdir(exist_ok=True)
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'case "$1" in -sb) echo "{size_bytes}\t$2" ;; '
        f'*) echo "{size_bytes // 1024}\t$2" ;; esac\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)


@posix_only
@pytest.mark.parametrize("name", list(PROFILES))
def test_a_complete_base_model_download_is_recognised_for_every_profile(
    env_for,  # type: ignore[no-untyped-def]
    tmp_path: Path,
    name: str,
) -> None:
    """R18-E1 (D-810): ``du`` counts binary units, ``base_gb`` is decimal - compare bytes.

    The check divided the KiB of ``du -sk`` by 1048576 (whole GiB) and compared that with
    ``base_gb * 0.95`` (decimal GB): a finished Qwen3-14B (29.55e9 bytes = 27 GiB) never reached
    28, a Qwen3-32B (61 GiB) never 62, so ``setup.sh`` stopped every 14B and 32B setup with
    "the download is incomplete".  A download that is 10 % short is still incomplete.
    """
    profile = PROFILES[name]
    real = REAL_BASE_BYTES[profile.base_model]
    assert abs(profile.base_gb * 1e9 - real) / real < 0.01  # base_gb is decimal gigabytes
    autodl, env = env_for(name)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    for size, complete in ((real, True), (int(real * 0.9), False)):
        write_du_stub(tmp_path, size)
        done = call_setup(autodl, env, f"base_model_complete '{model}'")
        assert (done.returncode == 0) is complete, (name, size, done.stdout, done.stderr)
    (model / "config.json").unlink()  # no config.json: never complete, whatever the size
    write_du_stub(tmp_path, real)
    assert call_setup(autodl, env, f"base_model_complete '{model}'").returncode == 1


# ------------------------------------------------------ PyTorch stays as it is (R17-49b)

COUNTED_TORCH = (
    # the first look at PyTorch (before the packages) is fine; every later one finds a CUDA 13 build
    'counter="${TWIN_TEST_COUNTER}"; '
    'probe_torch() { local n; n="$(cat "${counter}" 2>/dev/null || echo 0)"; '
    'echo $((n + 1)) >"${counter}"; '
    "if [ \"${n}\" -ge 1 ]; then echo '2.11.0 13.0 sm_80,sm_120'; "
    "else echo '2.9.1+cu128 12.8 sm_80,sm_90,sm_120'; fi; }; "
)


@posix_only
def test_the_torch_stack_is_frozen_while_the_packages_are_installed(tmp_path: Path) -> None:
    """R17-49b (D-632): pip must not swap PyTorch for another build as a side effect.

    ``install_packages`` passes a constraints file with the stack that was just accepted
    (``torch==2.9.1+cu128`` ...); before, nothing stopped the resolver from replacing it.
    """
    workdir = tmp_path / "twin"
    autodl = install_scripts(workdir, "5090-8b")
    env = script_env(workdir, tmp_path, TWIN_PYTHON=str(install_stub_python(tmp_path)))
    done = call_setup(autodl, env, "install_packages")
    assert done.returncode == 0, done.stderr
    install = next(call for call in pip_calls(tmp_path) if call[:1] == ["install"])
    assert "-c" in install, install
    constraints = Path(install[install.index("-c") + 1])
    assert constraints.read_text("utf-8").splitlines() == [
        "torch==2.9.1+cu128",
        "torchvision==0.24.1+cu128",
        "torchaudio==2.9.1+cu128",
    ]
    assert not any(argument.startswith("torch") for argument in install if "==" in argument[:-1])


@posix_only
def test_only_the_torch_packages_that_are_installed_are_frozen(tmp_path: Path) -> None:
    """R17-49b (D-632): the real probe reads ``importlib.metadata``; a missing one is not pinned."""
    workdir = tmp_path / "twin"
    autodl = install_scripts(workdir, "5090-8b")
    site = tmp_path / "site"
    write_distribution(site, "torch", "2.8.0+cu128")
    write_distribution(site, "torchaudio", "2.8.0+cu128")
    env = script_env(workdir, tmp_path, PYTHONPATH=str(site))
    del env["TWIN_PROBE_OVERRIDES"]  # the real probe, not the tests' default
    target = tmp_path / "constraints.txt"
    done = call_function(autodl, env, f"write_torch_constraints {target}")
    assert done.returncode == 0, done.stderr
    assert target.read_text("utf-8").splitlines() == [
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",
    ]


@posix_only
def test_a_torch_replaced_by_the_package_install_stops_the_setup(tmp_path: Path) -> None:
    """R17-49b (D-632): ``run_install`` asks again after ``install_packages``.

    The probe finds a good PyTorch the first time and a CUDA 13 build every time after: the
    setup stops with exit 1 before it downloads the base model or marks itself done.  Before,
    nothing looked at PyTorch again and the run went on to hours of work on a broken stack.
    """
    workdir = tmp_path / "twin"
    autodl = install_scripts(workdir, "5090-8b")
    env = script_env(
        workdir,
        tmp_path,
        TWIN_PYTHON=str(install_stub_python(tmp_path)),
        TWIN_TEST_COUNTER=str(tmp_path / "torch_looks"),
    )
    download = tmp_path / "download_started"
    body = (
        SETUP_WITHOUT_THE_DOWNLOAD
        + COUNTED_TORCH
        + f'download_base_model() {{ touch "{download}"; }}; run_install'
    )
    done = call_setup(autodl, env, body)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "PyTorch is not usable after the packages are installed" in done.stderr
    assert "CUDA 13.0 builds are not used" in done.stderr
    assert not download.exists() and not (workdir / ".setup_done").exists()
    assert int((tmp_path / "torch_looks").read_text("utf-8")) == 2  # before and after, no more


@posix_only
def test_verify_stops_when_the_torch_of_the_instance_is_not_usable(tmp_path: Path) -> None:
    """R17-49b (D-632): the cheap second look in ``setup.sh verify``."""
    workdir = prepared_workdir(tmp_path)
    add_template_check(workdir, 0)
    log = install_fake_cli(tmp_path)
    env = script_env(
        workdir,
        tmp_path,
        TWIN_PROBE_OVERRIDES=write_overrides(
            tmp_path, probe_torch="echo '2.11.0 13.0 sm_80,sm_120'"
        ),
    )
    done = run_script(workdir, env, "setup.sh", "5090-8b", "verify")
    assert done.returncode == 1
    assert "PyTorch is not usable on this instance" in done.stderr
    assert cli_calls(log) == []  # no template check, no trial step


# ---------------------------------------------------------------------------- decrypt


def staged(tmp_path: Path, profile: str = "5090-8b") -> tuple[Path, dict[str, str], Path]:
    dataset = write_synthetic_dataset(tmp_path / "ds", pairs=3)
    workdir = tmp_path / "twin"
    install_scripts(workdir, profile)
    result = bundle.build_bundle(
        dataset,
        PROFILES[profile],
        passphrase=PASSPHRASE,
        out_dir=tmp_path / "out",
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
        kdf_log_n=10,
    )
    shutil.copyfile(result.path, workdir / "bundle.enc")
    return workdir, script_env(workdir, tmp_path), dataset.path


@posix_only
def test_decrypt_unpacks_verifies_and_can_run_again(tmp_path: Path) -> None:
    workdir, env, dataset = staged(tmp_path)
    for _ in range(2):  # idempotent
        done = run_script(workdir, env, "decrypt.sh", "5090-8b", stdin=PASSPHRASE + "\n")
        assert done.returncode == 0, done.stderr
    assert (workdir / "data" / "sft_train.jsonl").read_bytes() == (
        dataset / "sft_train.jsonl"
    ).read_bytes()
    assert (workdir / "data" / "dataset_info.json").is_file()
    assert (workdir / "config" / "sft.yaml").is_file() and (workdir / "manifest.json").is_file()
    assert (workdir / "pylib" / "twin" / "training" / "lf_template.py").is_file()
    assert not (workdir / ".unpack").exists()
    assert json.loads((workdir / "manifest.json").read_text("utf-8"))["profile"] == "5090-8b"
    assert PASSPHRASE not in done.stdout + done.stderr


@posix_only
def test_a_wrong_passphrase_stops_decrypt_and_leaves_earlier_data_alone(tmp_path: Path) -> None:
    workdir, env, _ = staged(tmp_path)
    ok = run_script(workdir, env, "decrypt.sh", "5090-8b", stdin=PASSPHRASE + "\n")
    assert ok.returncode == 0
    before = (workdir / "data" / "sft_train.jsonl").read_bytes()
    bad = run_script(workdir, env, "decrypt.sh", "5090-8b", stdin="not the passphrase!!\n")
    assert bad.returncode != 0 and "wrong passphrase" in bad.stderr
    assert (workdir / "data" / "sft_train.jsonl").read_bytes() == before


@posix_only
def test_decrypt_refuses_scripts_that_differ_from_the_ones_in_the_package(tmp_path: Path) -> None:
    workdir, env, _ = staged(tmp_path)
    with (workdir / "autodl" / "train.sh").open("a", encoding="utf-8") as stream:
        stream.write("\necho tampered\n")
    done = run_script(workdir, env, "decrypt.sh", "5090-8b", stdin=PASSPHRASE + "\n")
    assert done.returncode != 0 and "differs from the uploaded script" in done.stderr
    assert not (workdir / "data").exists()


@posix_only
def test_decrypt_needs_the_package_and_the_right_profile(tmp_path: Path) -> None:
    workdir = tmp_path / "twin"
    install_scripts(workdir, "5090-8b")
    env = script_env(workdir, tmp_path)
    assert "upload the package first" in run_script(workdir, env, "decrypt.sh", "5090-8b").stderr
    assert "profile.env is for" in run_script(workdir, env, "decrypt.sh", "pro6000-14b").stderr


# ----------------------------------------------------------------------------- cleanup


def populate_for_cleanup(workdir: Path, tmp_path: Path) -> None:
    for relative in (
        "data/sft_train.jsonl",
        "logs/train.log",
        "jobs/r1-train/log",
        "output/sft/checkpoint-4/optimizer.pt",
        ".unpack/data/x",
        "gguf/model-F16.gguf",
        "trial_data/sft_trial.jsonl",
        "trial_data_dpo/dpo_trial.jsonl",  # R17-49d (D-651): the pairs of the DPO trial step
        "output/eval_generate_dpo/generated_predictions.jsonl",  # R17-49d: per-stack evaluation
        "artifacts/gguf/m.gguf",
        "artifacts/manifest.json",
    ):
        path = workdir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("her messages", encoding="utf-8")
    for relative in ("cache/huggingface/datasets/x/data.arrow", "tmp/scratch", "cache/pip/keep"):
        path = tmp_path / "disk" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")


@posix_only
def test_cleanup_refuses_without_confirmation(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path)
    populate_for_cleanup(workdir, tmp_path)
    done = run_script(workdir, script_env(workdir, tmp_path), "cleanup.sh")
    assert done.returncode == 1 and "--yes" in done.stderr
    assert (workdir / "data" / "sft_train.jsonl").exists()


@posix_only
def test_cleanup_erases_the_data_and_keeps_the_artifacts_until_asked(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path)
    populate_for_cleanup(workdir, tmp_path)
    env = script_env(workdir, tmp_path)
    done = run_script(workdir, env, "cleanup.sh", "--yes")
    assert done.returncode == 0, done.stderr
    for gone in (
        "data",
        "logs",
        "jobs",
        "output",
        ".unpack",
        "bundle.enc",
        "config",
        "pylib",
        "gguf",
        "trial_data",
        "trial_data_dpo",
    ):
        assert not (workdir / gone).exists(), gone
    assert not (tmp_path / "disk" / "cache" / "huggingface" / "datasets").exists()
    assert not (tmp_path / "disk" / "tmp").exists()
    assert (tmp_path / "disk" / "cache" / "pip" / "keep").exists()  # not a data cache
    assert (workdir / "artifacts" / "gguf" / "m.gguf").exists() and (workdir / "autodl").is_dir()
    assert (workdir / "state" / "cleanup.done").read_text("utf-8").endswith("Z\n")
    assert "请到 AutoDL 控制台释放实例" in done.stdout
    again = run_script(workdir, env, "cleanup.sh", "--yes")  # nothing left to do, still fine
    assert again.returncode == 0
    final = run_script(workdir, env, "cleanup.sh", "--yes", "--all")
    assert final.returncode == 0 and not (workdir / "artifacts").exists()


# ----------------------------------------------------------------- train / eval / dpo


@posix_only
def test_train_runs_the_config_and_writes_the_metrics_and_curves(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path)
    log = install_fake_cli(tmp_path)
    done = run_script(workdir, script_env(workdir, tmp_path), "train.sh", "5090-8b")
    assert done.returncode == 0, done.stderr
    assert [(c["mode"], c["config"]) for c in cli_calls(log)] == [("train", "sft.yaml")]
    metrics = json.loads((workdir / "artifacts" / "train" / "metrics.json").read_text("utf-8"))
    assert metrics["best_checkpoint"] == "checkpoint-8" and metrics["best_val_loss"] == 1.25
    assert metrics["eval_history"] == [[4, 1.5], [8, 1.25]]
    assert (workdir / "artifacts" / "train" / "training_loss.png").exists()
    assert "best checkpoint: checkpoint-8" in done.stdout


@posix_only
def test_train_needs_the_config_and_a_finished_setup(tmp_path: Path) -> None:
    workdir = tmp_path / "twin"
    install_scripts(workdir, "5090-8b")
    env = script_env(workdir, tmp_path)
    assert "run decrypt.sh first" in run_script(workdir, env, "train.sh", "5090-8b").stderr


def check_generations(workdir: Path, stack: str) -> None:
    """The generations of a stack are the test ids in order, with the texts, and nothing else."""
    rows = [
        json.loads(line)
        for line in (workdir / "artifacts" / "eval" / stack / "eval_generations.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    test_ids = [
        json.loads(line)["id"]
        for line in (workdir / "data" / "sft_test.jsonl").read_text("utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == test_ids
    assert all(set(row) == {"id", "text"} and row["text"].startswith("generated") for row in rows)
    metrics = json.loads(
        (workdir / "artifacts" / "eval" / stack / "eval_metrics.json").read_text("utf-8")
    )
    assert metrics == {"val_loss": 1.2, "adapter": stack, "generated": len(rows)}


@posix_only
def test_eval_without_a_dpo_adapter_evaluates_the_sft_stack(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path)
    log = install_fake_cli(tmp_path)
    env = script_env(workdir, tmp_path)
    assert "run train.sh first" in run_script(workdir, env, "eval.sh", "5090-8b").stderr
    assert run_script(workdir, env, "train.sh", "5090-8b").returncode == 0
    done = run_script(workdir, env, "eval.sh", "5090-8b")
    assert done.returncode == 0, done.stderr
    calls = cli_calls(log)[1:]
    assert [c["config"] for c in calls] == ["eval_loss.run.yaml", "eval_generate.run.yaml"]
    assert all(str(c["adapter"]).endswith("/output/sft") for c in calls)
    check_generations(workdir, "sft")
    assert not (workdir / "artifacts" / "eval" / "dpo").exists()
    assert run_script(workdir, env, "eval.sh", "5090-8b", "nope").returncode == 1
    missing = run_script(workdir, env, "eval.sh", "5090-8b", "dpo")
    assert (
        missing.returncode == 1 and "no adapter in" in missing.stderr and "dpo.sh" in missing.stderr
    )


def trained_with_dpo(
    tmp_path: Path, profile: str, **fake: int
) -> tuple[Path, dict[str, str], Path]:
    """A work directory with the SFT and the DPO adapter, as ``train.sh`` and ``dpo.sh`` leave."""
    workdir = prepared_workdir(tmp_path, profile, pairs=3, dpo_min_pairs=2)
    log = install_fake_cli(tmp_path, **fake)
    env = script_env(workdir, tmp_path)
    assert run_script(workdir, env, "train.sh", profile).returncode == 0
    done = run_script(workdir, env, "dpo.sh", profile)
    assert done.returncode == 0, done.stderr + done.stdout
    return workdir, env, log


@posix_only
@pytest.mark.parametrize(("profile", "layered"), [("5090-8b", True), ("5090-14b", False)])
def test_eval_compares_the_sft_stack_with_the_stack_that_has_dpo_on_the_same_samples(
    tmp_path: Path, profile: str, layered: bool
) -> None:
    """R17-49d (D-650, D-655): both stacks, the same test contexts and validation set.

    The bf16 profile evaluates ``sft,dpo`` (LLaMA-Factory merges both); the 4-bit profile
    evaluates the one continued adapter (a quantised model refuses a second one, the fake command
    knows the rule).  A verdict of an earlier evaluation is deleted: it described other texts.
    """
    workdir, env, log = trained_with_dpo(tmp_path, profile)
    (workdir / "state").mkdir(exist_ok=True)
    (workdir / "state" / "stack_choice").write_text("dpo\n", encoding="utf-8")
    (workdir / "artifacts" / "eval").mkdir(parents=True, exist_ok=True)
    (workdir / "artifacts" / "eval" / "stack_choice.json").write_text("{}", encoding="utf-8")
    before = len(cli_calls(log))
    done = run_script(workdir, env, "eval.sh", profile)
    assert done.returncode == 0, done.stderr + done.stdout
    calls = cli_calls(log)[before:]
    assert [c["config"] for c in calls] == ["eval_loss.run.yaml", "eval_generate.run.yaml"] * 2
    sft_dir = f"{workdir}/output/sft"
    dpo_stack = f"{workdir}/output/dpo" if not layered else f"{sft_dir},{workdir}/output/dpo"
    assert [str(c["adapter"]) for c in calls] == [sft_dir, sft_dir, dpo_stack, dpo_stack]
    for stack in ("sft", "dpo"):
        check_generations(workdir, stack)
    assert not (workdir / "state" / "stack_choice").exists()
    assert not (workdir / "artifacts" / "eval" / "stack_choice.json").exists()
    # the evaluation of one stack alone, on request
    run_script(workdir, env, "eval.sh", profile, "sft")
    assert str(cli_calls(log)[-1]["adapter"]) == sft_dir


@posix_only
def test_dpo_is_skipped_below_the_minimum_and_runs_above_it(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path, pairs=3, dpo_min_pairs=200)
    log = install_fake_cli(tmp_path)
    done = run_script(workdir, script_env(workdir, tmp_path), "dpo.sh", "5090-8b")
    assert done.returncode == 0 and "DPO skipped: 3 preference pairs, at least 200" in done.stdout
    assert cli_calls(log) == []
    other = tmp_path / "second"
    other.mkdir()
    workdir = prepared_workdir(other, pairs=3, dpo_min_pairs=2)
    log = install_fake_cli(other)
    env = script_env(workdir, other)
    assert "run train.sh first" in run_script(workdir, env, "dpo.sh", "5090-8b").stderr
    assert run_script(workdir, env, "train.sh", "5090-8b").returncode == 0
    done = run_script(workdir, env, "dpo.sh", "5090-8b")
    assert done.returncode == 0, done.stderr
    assert cli_calls(log)[-1]["config"] == "dpo.yaml"
    assert (workdir / "artifacts" / "dpo" / "metrics.json").exists()


def dpo_record(workdir: Path) -> dict[str, Any]:
    value = json.loads((workdir / "artifacts" / "dpo" / "dpo_reference.json").read_text("utf-8"))
    assert isinstance(value, dict)
    return value


@posix_only
@pytest.mark.parametrize("profile", ["5090-8b", "pro6000-14b", "pro6000-32b"])
def test_dpo_of_a_bf16_profile_trains_a_new_adapter_on_sft_without_a_trial(
    tmp_path: Path, profile: str
) -> None:
    """R17-49d (D-650): layered - ``create_new_adapter``, no ``ref_model``, nothing to try first.

    The reference is the merged SFT model (the new adapter switched off), which needs no memory
    beyond the policy's own, so there is no trial step; the record says so.
    """
    workdir, _, log = trained_with_dpo(tmp_path, profile)
    calls = [c for c in cli_calls(log) if c["stage"] == "dpo"]
    assert [c["config"] for c in calls] == ["dpo.yaml"]
    call = calls[0]
    assert call["adapter"] == f"{workdir}/output/sft" and call["create_new_adapter"] is True
    assert call["ref_model"] is None and call["max_steps"] is None
    assert dpo_record(workdir) == {
        "mode": "layered",
        "reference": "sft",
        "reference_quantization_bit": None,
        "fallback": False,
        "fallback_reason": None,
        "per_device_batch": 1,
        "accumulation": 16,
        "pairs": 3,
    }
    metrics = json.loads((workdir / "artifacts" / "dpo" / "metrics.json").read_text("utf-8"))
    assert metrics["dpo"]["reference"] == "sft"
    assert (workdir / "output" / "dpo" / "adapter_model.safetensors").is_file()
    assert (workdir / "output" / "sft" / "adapter_model.safetensors").is_file()  # untouched


@posix_only
def test_dpo_of_the_4bit_profile_tries_one_step_with_the_explicit_reference_first(
    tmp_path: Path,
) -> None:
    """R17-49d (D-651): the trial is the real config, one step, on the longest pairs."""
    workdir, _, log = trained_with_dpo(tmp_path, "5090-14b")
    calls = [c for c in cli_calls(log) if c["stage"] == "dpo"]
    assert [c["config"] for c in calls] == ["dpo_trial.yaml", "dpo.yaml"]
    trial, real = calls
    assert trial["max_steps"] == 1 and trial["dataset"] == "twin_dpo_trial"
    assert trial["dataset_dir"] == f"{workdir}/trial_data_dpo"
    assert trial["ref_model"] == real["ref_model"] == str(PROFILES["5090-14b"].model_dir)
    assert real["ref_model_adapters"] == f"{workdir}/output/sft"
    assert real["ref_model_quantization_bit"] == 4 and real["create_new_adapter"] is None
    assert real["max_steps"] is None and real["dataset"] == "twin_dpo_train"
    assert not (workdir / "trial_data_dpo").exists()  # copies of the pairs do not stay
    assert not (workdir / "output" / "dpo_trial").exists()
    assert dpo_record(workdir) == {
        "mode": "continued",
        "reference": "sft",
        "reference_quantization_bit": 4,
        "fallback": False,
        "fallback_reason": None,
        "per_device_batch": 1,
        "accumulation": 16,
        "pairs": 3,
    }


@posix_only
def test_the_dpo_trial_halves_the_batch_before_it_gives_up_the_reference(tmp_path: Path) -> None:
    """R17-49d (D-651): out of memory with the reference - a smaller batch first (same product)."""
    workdir = prepared_workdir(tmp_path, "5090-14b", pairs=3, dpo_min_pairs=2)
    log = install_fake_cli(tmp_path, max_batch_with_ref=2)
    env = script_env(workdir, tmp_path)
    assert run_script(workdir, env, "train.sh", "5090-14b").returncode == 0
    config = workdir / "config" / "dpo.yaml"
    text = config.read_text(encoding="utf-8")
    text = text.replace("per_device_train_batch_size: 1", "per_device_train_batch_size: 8")
    config.write_text(
        text.replace("gradient_accumulation_steps: 16", "gradient_accumulation_steps: 2"), "utf-8"
    )
    done = run_script(workdir, env, "dpo.sh", "5090-14b")
    assert done.returncode == 0, done.stderr + done.stdout
    calls = [c for c in cli_calls(log) if c["stage"] == "dpo"]
    assert [(c["batch"], c["accumulation"]) for c in calls] == [(8, 2), (4, 4), (2, 8), (2, 8)]
    assert [c["config"] for c in calls] == ["dpo_trial.yaml"] * 3 + ["dpo.yaml"]
    assert all(c["ref_model"] for c in calls)  # the reference was never given up
    assert yaml_value(config, "per_device_train_batch_size") == 2
    assert yaml_value(config, "gradient_accumulation_steps") == 8
    record = dpo_record(workdir)
    assert record["reference"] == "sft" and record["fallback"] is False
    assert (record["per_device_batch"], record["accumulation"]) == (2, 8)


@posix_only
def test_the_4bit_profile_falls_back_to_the_base_as_the_reference_when_nothing_else_fits(
    tmp_path: Path,
) -> None:
    """R17-49d (D-651): a batch of 1 with the explicit reference runs out of memory - only this
    profile drops ``ref_model`` (the base with the adapter switched off is the reference), the
    fact is in the record, in ``metrics.json`` and in the log, and a rerun keeps it."""
    workdir = prepared_workdir(tmp_path, "5090-14b", pairs=3, dpo_min_pairs=2)
    log = install_fake_cli(tmp_path, max_batch_with_ref=0)
    env = script_env(workdir, tmp_path)
    assert run_script(workdir, env, "train.sh", "5090-14b").returncode == 0
    done = run_script(workdir, env, "dpo.sh", "5090-14b")
    assert done.returncode == 0, done.stderr + done.stdout
    assert "WARNING: out of memory with the explicit 4-bit SFT reference" in done.stdout
    calls = [c for c in cli_calls(log) if c["stage"] == "dpo"]
    assert [c["config"] for c in calls] == ["dpo_trial.yaml", "dpo_trial.yaml", "dpo.yaml"]
    assert [bool(c["ref_model"]) for c in calls] == [True, False, False]
    assert (
        calls[-1]["ref_model_adapters"] is None and calls[-1]["ref_model_quantization_bit"] is None
    )
    config = workdir / "config" / "dpo.yaml"
    assert not any(line.startswith("ref_model") for line in config.read_text("utf-8").splitlines())
    assert yaml_value(config, "quantization_bit") == 4  # the policy is untouched
    assert "create_new_adapter" not in config.read_text("utf-8")  # a quantised base cannot
    record = dpo_record(workdir)
    assert record["reference"] == "base" and record["fallback"] is True
    assert (
        record["reference_quantization_bit"] is None
        and "out of memory" in record["fallback_reason"]
    )
    metrics = json.loads((workdir / "artifacts" / "dpo" / "metrics.json").read_text("utf-8"))
    assert metrics["dpo"] == record
    # a second run of the script does not forget it happened
    again = run_script(workdir, env, "dpo.sh", "5090-14b")
    assert again.returncode == 0, again.stderr
    assert dpo_record(workdir)["fallback"] is True and dpo_record(workdir)["reference"] == "base"


@posix_only
def test_the_dpo_trial_stops_for_another_reason_than_memory_and_when_the_base_does_not_fit_either(
    tmp_path: Path,
) -> None:
    workdir = prepared_workdir(tmp_path, "5090-14b", pairs=3, dpo_min_pairs=2)
    log = install_fake_cli(tmp_path)
    env = script_env(workdir, tmp_path)
    assert run_script(workdir, env, "train.sh", "5090-14b").returncode == 0
    # nothing fits now, with or without the reference
    (tmp_path / "stubs" / "max_batch").write_text("0", encoding="utf-8")
    done = run_script(workdir, env, "dpo.sh", "5090-14b")
    assert (
        done.returncode == 1
        and "even with a batch of 1 and the base as the reference" in done.stderr
    )
    assert not (workdir / "trial_data_dpo").exists()
    assert not (workdir / "artifacts" / "dpo" / "metrics.json").exists()  # no training happened
    assert [c["config"] for c in cli_calls(log) if c["stage"] == "dpo"] == ["dpo_trial.yaml"] * 2
    # an error that is not memory is not answered by a smaller batch or a different reference
    (tmp_path / "stubs" / "max_batch").unlink()
    (tmp_path / "stubs" / "fail_trial").write_text("1", encoding="utf-8")
    other = run_script(workdir, env, "dpo.sh", "5090-14b")
    assert other.returncode == 1 and "for a reason other than memory" in other.stderr
    assert not (workdir / "trial_data_dpo").exists()


# ---------------------------------------------------------------------------- export


EXPORT_STUBS = {
    "llama.cpp/convert_hf_to_gguf.py": (
        "import sys\n"
        "from pathlib import Path\n"
        "model, out = Path(sys.argv[1]), Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
        "assert sys.argv[sys.argv.index('--outtype') + 1] == 'f16'\n"
        "assert (model / 'config.json').exists()\n"
        "with open(Path(__file__).parents[2] / 'trace.log', 'a') as log:\n"
        "    log.write(f'convert merged_exists={int(model.exists())}\\n')\n"
        "out.write_bytes(b'F16' * 1000)\n"
    ),
}

QUANTIZE_STUB = """#!/usr/bin/env bash
set -eu
root="$(cd "$(dirname "$0")/../../.." && pwd)"
merged=0; [ -d "$root/output/merged" ] && merged=1
f16=0; [ -f "$1" ] && f16=1
printf 'quantize %s merged_exists=%s f16_exists=%s\\n' "$3" "$merged" "$f16" >>"$root/../trace.log"
printf '%s\\n' "$2" >>"$root/../quant_targets.log"
printf '%s' "$3-model" >"$2"
"""


def prepare_export(
    tmp_path: Path, profile: str = "5090-14b", *, free_gb: str = "echo 900"
) -> tuple[Path, dict[str, str], Path]:
    """A work directory ready for ``export.sh``: an SFT adapter, llama.cpp's products as stubs."""
    workdir = prepared_workdir(tmp_path, profile)
    install_fake_cli(tmp_path)
    env = script_env(
        workdir,
        tmp_path,
        TWIN_PROBE_OVERRIDES=write_overrides(tmp_path, probe_free_gb=free_gb),
        TWIN_RUN_ID="r-test-1",
    )
    write_tiny_adapter(
        workdir / "output" / "sft",
        rank=PROFILES[profile].lora_rank,
        alpha=PROFILES[profile].lora_alpha,
        seed=3,
    )
    # llama.cpp is cloned and built by the prepare step; here its products are in place already
    llama = workdir / "llama.cpp"
    for name, text in EXPORT_STUBS.items():
        (workdir / name).parent.mkdir(parents=True, exist_ok=True)
        (workdir / name).write_text(text, encoding="utf-8")
    quantize = llama / "build" / "bin" / "llama-quantize"
    quantize.parent.mkdir(parents=True)
    quantize.write_text(QUANTIZE_STUB, encoding="utf-8")
    quantize.chmod(0o755)
    venv_python = workdir / "venv-gguf" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n', "utf-8")
    venv_python.chmod(0o755)
    (workdir / "state").mkdir()
    (workdir / "state" / "export.prepare_llama_cpp").touch()
    # the trace file lives next to the work directory so the stubs can reach it
    return workdir, env, workdir.parent / "trace.log"


@posix_only
@pytest.mark.parametrize(
    ("profile", "export_device"), [("5090-14b", "cpu"), ("pro6000-14b", "auto")]
)
def test_the_export_runs_the_steps_in_order_and_deletes_only_what_is_replaced(
    tmp_path: Path, profile: str, export_device: str
) -> None:
    """R-TRN-009; R17-49f (D-635): for the QLoRA profile and the bf16 one, with their own settings.

    The 14B profile of the 5090 trains with 4-bit QLoRA and must merge on the CPU into the bf16
    base *without* ``quantization_bit`` (LLaMA-Factory refuses to merge into a quantised model);
    the PRO 6000 profile merges wherever LLaMA-Factory puts it.  The test used only one profile
    and looked at none of these settings.
    """
    workdir, env, trace = prepare_export(tmp_path, profile)

    done = run_script(workdir, env, "export.sh", profile)
    assert done.returncode == 0, done.stderr + done.stdout

    lines = trace.read_text("utf-8").splitlines()
    assert lines == [
        "convert merged_exists=1",
        "quantize Q4_K_M merged_exists=0 f16_exists=1",
        "quantize Q5_K_M merged_exists=0 f16_exists=1",
        "quantize Q8_0 merged_exists=0 f16_exists=1",
    ]
    assert not (workdir / "output" / "merged").exists()  # deleted after the F16 file existed
    assert not (workdir / "gguf" / "model-F16.gguf").exists()  # deleted after all three quants
    artifacts = workdir / "artifacts"
    # R17-8 (D-634): the quantizer writes where the files stay; there is no second copy on the disk
    targets = (tmp_path / "quant_targets.log").read_text("utf-8").splitlines()
    assert [Path(target).name for target in targets] == [
        f"{profile}-Q4_K_M.gguf.part",
        f"{profile}-Q5_K_M.gguf.part",
        f"{profile}-Q8_0.gguf.part",
    ]
    assert {Path(target).parent for target in targets} == {artifacts / "gguf"}
    assert list((workdir / "gguf").iterdir()) == []  # not even the F16 file is left
    assert sorted(p.name for p in (artifacts / "gguf").iterdir()) == [
        f"{profile}-Q4_K_M.gguf",
        f"{profile}-Q5_K_M.gguf",
        f"{profile}-Q8_0.gguf",
    ]
    # one adapter, the SFT one: copied as it is (the layers of a stack are in test_..._stack_...)
    assert (artifacts / "adapter" / "adapter_model.safetensors").read_bytes() == (
        workdir / "output" / "sft" / "adapter_model.safetensors"
    ).read_bytes()
    manifest = json.loads((artifacts / "manifest.json").read_text("utf-8"))
    assert manifest["run_id"] == "r-test-1" and manifest["profile"] == profile
    assert manifest["template_version"] == "qwen3_nothink@llamafactory-0.9.5"
    assert manifest["persona_version"] == "v3" and manifest["dataset_version"] == "ds-test-01"
    assert [m["quant"] for m in manifest["models"]] == ["Q4_K_M", "Q5_K_M", "Q8_0", "lora"]
    listed = {f["path"] for f in manifest["files"]}
    assert f"gguf/{profile}-Q4_K_M.gguf" in listed and "adapter/adapter_model.safetensors" in listed
    export_call = next(
        c for c in cli_calls(tmp_path / "stubs" / "cli_calls.log") if c["mode"] == "export"
    )
    assert str(export_call["adapter"]).endswith("/output/sft")
    # R17-49f (D-635): what is specific to the profile reached the merge
    assert export_call["export_device"] == export_device
    assert export_call["quantization_bit"] is None  # the merge loads the bf16 base, never 4-bit
    assert PROFILES[profile].export_device == export_device
    # a second run has nothing left to do
    again = run_script(workdir, env, "export.sh", profile)
    assert again.returncode == 0 and "already done" in again.stdout
    assert len(trace.read_text("utf-8").splitlines()) == 4


@posix_only
def test_a_merge_into_a_quantised_base_is_refused_like_llamafactory_does(tmp_path: Path) -> None:
    """R17-49f (D-635): the fake command knows the rule, so a wrong export.yaml cannot slip by."""
    workdir, env, _ = prepare_export(tmp_path, "5090-14b")
    config = workdir / "config" / "export.yaml"
    config.write_text(config.read_text("utf-8") + "quantization_bit: 4\n", encoding="utf-8")
    done = run_script(workdir, env, "export.sh", "5090-14b")
    assert done.returncode != 0
    assert "Cannot merge adapters to a quantized model" in done.stderr + done.stdout


def add_dpo_adapter(workdir: Path, profile: str, verdict: str | None) -> None:
    """The DPO adapter of a preference run, and the verdict `twin train remote eval` wrote."""
    write_tiny_adapter(
        workdir / "output" / "dpo",
        rank=PROFILES[profile].lora_rank,
        alpha=PROFILES[profile].lora_alpha,
        seed=7,
    )
    if verdict is not None:
        (workdir / "state").mkdir(exist_ok=True)
        (workdir / "state" / "stack_choice").write_text(f"{verdict}\n", encoding="utf-8")


def export_call(tmp_path: Path) -> dict[str, Any]:
    return next(c for c in cli_calls(tmp_path / "stubs" / "cli_calls.log") if c["mode"] == "export")


@posix_only
def test_the_export_waits_for_the_verdict_when_there_is_a_dpo_adapter(tmp_path: Path) -> None:
    """R17-49d (D-655): ``auto`` is the stack the evaluation kept; none yet: stop, touch nothing."""
    workdir, env, _ = prepare_export(tmp_path, "pro6000-14b")
    add_dpo_adapter(workdir, "pro6000-14b", None)
    done = run_script(workdir, env, "export.sh", "pro6000-14b")
    assert done.returncode == 1 and "no evaluation has kept a stack" in done.stderr
    assert cli_calls(tmp_path / "stubs" / "cli_calls.log") == []
    assert not (workdir / "output" / "merged").exists()
    # naming the stack is allowed (the operator decides), and so is the verdict
    named = run_script(workdir, env, "export.sh", "pro6000-14b", "sft")
    assert named.returncode == 0, named.stderr
    assert export_call(tmp_path)["adapter"] == f"{workdir}/output/sft"


def lora_delta(directory: Path, module: str) -> Any:
    """``scale * B @ A`` of one module of the adapter in ``directory`` (float64)."""
    from safetensors.numpy import load_file

    config = json.loads((directory / "adapter_config.json").read_text("utf-8"))
    tensors = load_file(str(directory / "adapter_model.safetensors"))
    a = tensors[f"base_model.model.model.layers.0.self_attn.{module}.lora_A.weight"].astype(
        "float64"
    )
    b = tensors[f"base_model.model.model.layers.0.self_attn.{module}.lora_B.weight"].astype(
        "float64"
    )
    return (config["lora_alpha"] / config["r"]) * (b @ a)


@posix_only
def test_the_export_merges_both_layers_and_composes_them_into_the_adapter_vllm_serves(
    tmp_path: Path,
) -> None:
    """R17-49d (D-650, D-652, D-653): ``sft,dpo`` into the base; one served adapter of rank 64.

    The served adapter is the rank concatenation of the two layers, the scales folded into B: its
    delta is the SFT delta plus the DPO delta, whatever alpha and rank each of them has.  The raw
    layers are kept next to it with their own hashes, in the manifest in order.
    """
    import hashlib

    import numpy as np

    workdir, env, _ = prepare_export(tmp_path, "pro6000-14b")
    add_dpo_adapter(workdir, "pro6000-14b", "dpo")
    done = run_script(workdir, env, "export.sh", "pro6000-14b")
    assert done.returncode == 0, done.stderr + done.stdout
    assert export_call(tmp_path)["adapter"] == f"{workdir}/output/sft,{workdir}/output/dpo"
    assert export_call(tmp_path)["quantization_bit"] is None
    artifacts = workdir / "artifacts"
    served = artifacts / "adapter"
    config = json.loads((served / "adapter_config.json").read_text("utf-8"))
    assert config["r"] == 64 and config["lora_alpha"] == 64  # scale 1: the scales are in B
    for module in ("q_proj", "v_proj"):
        expected = lora_delta(artifacts / "adapters" / "sft", module) + lora_delta(
            artifacts / "adapters" / "dpo", module
        )
        assert np.allclose(lora_delta(served, module), expected, rtol=1e-4, atol=1e-5), module
    for name in ("sft", "dpo"):  # the layers are the adapters the training left, byte for byte
        left = (workdir / "output" / name / "adapter_model.safetensors").read_bytes()
        assert (artifacts / "adapters" / name / "adapter_model.safetensors").read_bytes() == left
    manifest = json.loads((artifacts / "manifest.json").read_text("utf-8"))
    stack = manifest["adapter_stack"]
    assert manifest["adapter_kind"] == "dpo"
    assert (stack["kind"], stack["mode"], stack["choice"]) == ("sft+dpo", "layered", "dpo")
    assert [layer["name"] for layer in stack["layers"]] == ["sft", "dpo"]
    for layer in stack["layers"]:
        content = (artifacts / layer["path"]).read_bytes()
        assert layer["sha256"] == hashlib.sha256(content).hexdigest() and layer["rank"] == 32
    assert stack["served"]["composed"] is True and stack["served"]["rank"] == 64
    assert stack["served"]["path"] == "adapter/adapter_model.safetensors"
    listed = {entry["path"] for entry in manifest["files"]}
    assert {
        "adapters/sft/adapter_model.safetensors",
        "adapters/dpo/adapter_model.safetensors",
    } <= listed
    assert [m["path"] for m in manifest["models"] if m["kind"] == "adapter"] == [
        stack["served"]["path"]
    ]
    assert manifest["metrics"]["eval_val_loss"] is None  # no evaluation ran in this test


@posix_only
def test_the_export_keeps_the_sft_stack_when_the_evaluation_did(tmp_path: Path) -> None:
    """R17-49d (D-655): the DPO adapter exists but lost: SFT alone is merged and served."""
    workdir, env, _ = prepare_export(tmp_path, "5090-8b")
    add_dpo_adapter(workdir, "5090-8b", "sft")
    done = run_script(workdir, env, "export.sh", "5090-8b")
    assert done.returncode == 0, done.stderr
    assert export_call(tmp_path)["adapter"] == f"{workdir}/output/sft"
    artifacts = workdir / "artifacts"
    assert not (artifacts / "adapters").exists()
    manifest = json.loads((artifacts / "manifest.json").read_text("utf-8"))
    assert manifest["adapter_kind"] == "sft" and manifest["adapter_stack"]["kind"] == "sft"
    assert manifest["adapter_stack"]["served"]["composed"] is False
    assert (artifacts / "adapter" / "adapter_model.safetensors").read_bytes() == (
        workdir / "output" / "sft" / "adapter_model.safetensors"
    ).read_bytes()


@posix_only
def test_the_export_of_the_4bit_profile_uses_the_one_continued_adapter(tmp_path: Path) -> None:
    """R17-49d (D-650): the DPO adapter of the 4-bit profile already holds SFT: it is the stack."""
    workdir, env, _ = prepare_export(tmp_path, "5090-14b")
    add_dpo_adapter(workdir, "5090-14b", "dpo")
    done = run_script(workdir, env, "export.sh", "5090-14b")
    assert done.returncode == 0, done.stderr
    call = export_call(tmp_path)
    assert call["adapter"] == f"{workdir}/output/dpo"  # never sft,dpo: SFT would count twice
    assert call["export_device"] == "cpu" and call["quantization_bit"] is None
    artifacts = workdir / "artifacts"
    manifest = json.loads((artifacts / "manifest.json").read_text("utf-8"))
    stack = manifest["adapter_stack"]
    assert (stack["kind"], stack["mode"]) == ("dpo", "continued")
    assert [layer["name"] for layer in stack["layers"]] == ["dpo"]
    assert (
        stack["layers"][0]["path"] == stack["served"]["path"] == "adapter/adapter_model.safetensors"
    )
    assert (artifacts / "adapter" / "adapter_model.safetensors").read_bytes() == (
        workdir / "output" / "dpo" / "adapter_model.safetensors"
    ).read_bytes()
    assert not (artifacts / "adapters").exists()


@posix_only
def test_a_rerun_of_collect_uses_the_stack_the_merge_used(tmp_path: Path) -> None:
    """R17-49d: the step that collects the adapters must not pick another stack than the merge."""
    workdir, env, _ = prepare_export(tmp_path, "5090-8b")
    add_dpo_adapter(workdir, "5090-8b", "dpo")
    assert run_script(workdir, env, "export.sh", "5090-8b").returncode == 0
    (workdir / "state" / "stack_choice").write_text("sft\n", encoding="utf-8")
    (workdir / "state" / "export.collect_artifacts").unlink()
    done = run_script(workdir, env, "export.sh", "5090-8b")
    assert done.returncode == 0, done.stderr
    manifest = json.loads((workdir / "artifacts" / "manifest.json").read_text("utf-8"))
    assert manifest["adapter_stack"]["kind"] == "sft+dpo"  # what was merged, not the new verdict


@posix_only
def test_the_manifest_refuses_a_layer_that_changed_after_the_stack_was_recorded(
    tmp_path: Path,
) -> None:
    """R17-49d (D-653): every layer is hashed again when the manifest is written."""
    workdir, env, _ = prepare_export(tmp_path, "5090-8b")
    add_dpo_adapter(workdir, "5090-8b", "dpo")
    assert run_script(workdir, env, "export.sh", "5090-8b").returncode == 0
    layer = workdir / "artifacts" / "adapters" / "dpo" / "adapter_model.safetensors"
    layer.write_bytes(layer.read_bytes() + b"x")
    (workdir / "artifacts" / "manifest.json").unlink()
    manifest_tool = workdir / "autodl" / "tools" / "artifact_manifest.py"
    done = subprocess.run(
        [
            sys.executable,
            str(manifest_tool),
            "--artifacts",
            str(workdir / "artifacts"),
            "--bundle-manifest",
            str(workdir / "manifest.json"),
            "--quants",
            "Q4_K_M,Q5_K_M,Q8_0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1 and "changed after adapter_stack.json was written" in done.stderr
    assert not (workdir / "artifacts" / "manifest.json").exists()


@posix_only
def test_a_rerun_of_collect_keeps_the_quants(tmp_path: Path) -> None:
    """R17-8 (D-634): the last step used to ``rm -rf artifacts/gguf``; now the quants live there.

    Doing the last step again (the manifest, say) must not delete the finished quantizations or
    run the quantizer again.
    """
    workdir, env, trace = prepare_export(tmp_path)
    assert run_script(workdir, env, "export.sh", "5090-14b").returncode == 0
    gguf = workdir / "artifacts" / "gguf"
    before = {path.name: path.read_bytes() for path in gguf.iterdir()}
    assert len(before) == 3
    (workdir / "state" / "export.collect_artifacts").unlink()
    done = run_script(workdir, env, "export.sh", "5090-14b")
    assert done.returncode == 0, done.stderr
    assert {path.name: path.read_bytes() for path in gguf.iterdir()} == before
    assert len(trace.read_text("utf-8").splitlines()) == 4  # no quantizer run
    assert (workdir / "artifacts" / "manifest.json").is_file()


@posix_only
def test_collect_stops_when_a_quantization_is_missing(tmp_path: Path) -> None:
    """R17-8 (D-634): the manifest is only written for a complete set."""
    workdir, env, _ = prepare_export(tmp_path)
    assert run_script(workdir, env, "export.sh", "5090-14b").returncode == 0
    (workdir / "artifacts" / "gguf" / "5090-14b-Q5_K_M.gguf").unlink()
    (workdir / "artifacts" / "manifest.json").unlink()
    (workdir / "state" / "export.collect_artifacts").unlink()
    done = run_script(workdir, env, "export.sh", "5090-14b")
    assert done.returncode == 1 and "5090-14b-Q5_K_M.gguf is missing" in done.stderr
    assert not (workdir / "artifacts" / "manifest.json").exists()


@posix_only
def test_the_quantization_gate_is_the_size_of_the_file_it_writes(tmp_path: Path) -> None:
    """R17-8 (D-634): ceil(F16 size x 0.55) + 2 GB free, not half the base model + 2 GB.

    The F16 file of the test is 3000 bytes: 3 GB are needed.  The old gate asked for half of the
    14B base model (15 GB) plus 2, which is too much for a small model and for 32B (35 GB for a
    Q8_0 file of 34.8 GB) too little.
    """
    free = (
        'if [ -s "$TWIN_HOME/gguf/model-F16.gguf" ]; then echo ${TEST_FREE_GB}; else echo 900; fi'
    )
    workdir, env, _ = prepare_export(tmp_path, free_gb=free)
    short = run_script(workdir, {**env, "TEST_FREE_GB": "2"}, "export.sh", "5090-14b")
    assert short.returncode == 1, short.stdout
    assert (
        "not enough free space before 'quantising Q4_K_M': 2 GB free, 3 GB needed" in short.stderr
    )
    assert "请到 AutoDL 控制台扩容数据盘" in short.stderr
    assert not list((workdir / "artifacts" / "gguf").glob("*"))
    enough = run_script(workdir, {**env, "TEST_FREE_GB": "3"}, "export.sh", "5090-14b")
    assert enough.returncode == 0, enough.stderr


@posix_only
@pytest.mark.parametrize(
    ("f16_gb", "needed"), [(0, 2), (0.0000028, 3), (16.4, 12), (29.6, 19), (65.6, 39), (100.5, 58)]
)
def test_the_free_space_a_quantization_needs(  # type: ignore[no-untyped-def]
    env_for, f16_gb: float, needed: int
) -> None:
    """R17-8 (D-634): ceil(F16 size x 0.55) + 2, at least the largest quant (Q8_0, 0.53).

    R18-E4 (D-811): sizes are decimal GB (10^9 bytes) like ``base_gb`` and the free space.
    """
    autodl, env = env_for("5090-8b")
    size = int(f16_gb * 10**9)
    done = call_function(autodl, env, f"quant_space_needed_gb {size}")
    assert done.returncode == 0 and done.stdout.strip() == str(needed), done.stderr
    assert needed >= f16_gb * max(QUANT_FACTORS.values()) + 2


@posix_only
def test_the_gate_factor_of_the_script_covers_the_largest_quantization(env_for) -> None:  # type: ignore[no-untyped-def]
    autodl, env = env_for("5090-8b")
    done = call_function(autodl, env, 'echo "${QUANT_SPACE_FACTOR}"')
    assert float(done.stdout) >= max(QUANT_FACTORS.values())


@posix_only
def test_the_export_refuses_to_start_without_an_adapter_or_enough_space(tmp_path: Path) -> None:
    workdir = prepared_workdir(tmp_path, "5090-8b")
    install_fake_cli(tmp_path)
    env = script_env(
        workdir, tmp_path, TWIN_PROBE_OVERRIDES=write_overrides(tmp_path, probe_free_gb="echo 3")
    )
    short = run_script(workdir, env, "export.sh", "5090-8b")
    assert (
        short.returncode == 1
        and "not enough free space before 'merging the adapter'" in short.stderr
    )
    assert "请到 AutoDL 控制台扩容数据盘" in short.stderr
    env = script_env(
        workdir, tmp_path, TWIN_PROBE_OVERRIDES=write_overrides(tmp_path, probe_free_gb="echo 900")
    )
    none = run_script(workdir, env, "export.sh", "5090-8b")
    assert none.returncode == 1 and "no adapter" in none.stderr
