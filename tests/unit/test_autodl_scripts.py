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

import pytest

from tests.support.autodl_scripts import (
    AUTODL,
    ROOT,
    call_function,
    cli_calls,
    install_fake_cli,
    install_scripts,
    posix_only,
    prepared_workdir,
    run_script,
    script_env,
)
from tests.support.training_data import write_synthetic_dataset
from twin.training import bundle
from twin.training.profiles import PROFILES

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
def test_the_best_adapter_is_the_dpo_adapter_when_there_is_one(env_for, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    autodl, env = env_for("5090-8b")
    work = Path(env["TWIN_HOME"])
    auto = call_function(autodl, env, "best_adapter auto").stdout.strip()
    assert auto == f"{work}/output/sft"
    (work / "output" / "dpo").mkdir(parents=True)
    (work / "output" / "dpo" / "adapter_model.safetensors").write_bytes(b"x")
    assert call_function(autodl, env, "best_adapter").stdout.strip() == f"{work}/output/dpo"
    assert call_function(autodl, env, "best_adapter sft").stdout.strip() == f"{work}/output/sft"
    assert call_function(autodl, env, "best_adapter bogus").returncode == 1


@posix_only
def test_out_of_memory_is_recognised_in_a_log(env_for, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    autodl, env = env_for("5090-8b")
    oom = tmp_path / "oom.log"
    oom.write_text("...\ntorch.OutOfMemoryError: CUDA out of memory.\n", encoding="utf-8")
    fine = tmp_path / "fine.log"
    fine.write_text("loss 1.0\n", encoding="utf-8")
    assert call_function(autodl, env, f"oom_in_log {oom}").returncode == 0
    assert call_function(autodl, env, f"oom_in_log {fine}").returncode == 1


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


# ------------------------------------------------------------------------ setup verify


@posix_only
def test_the_verify_step_stops_with_an_error_when_the_template_check_is_missing(
    tmp_path: Path,
) -> None:
    workdir = prepared_workdir(tmp_path)
    shutil.rmtree(workdir / "pylib" / "twin" / "training" / "__pycache__", ignore_errors=True)
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


@posix_only
def test_eval_uses_the_best_adapter_and_leaves_only_ids_and_texts(tmp_path: Path) -> None:
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
    rows = [
        json.loads(line)
        for line in (workdir / "artifacts" / "eval" / "eval_generations.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    test_ids = [
        json.loads(line)["id"]
        for line in (workdir / "data" / "sft_test.jsonl").read_text("utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == test_ids
    assert all(set(row) == {"id", "text"} and row["text"].startswith("generated") for row in rows)
    metrics = json.loads((workdir / "artifacts" / "eval" / "eval_metrics.json").read_text("utf-8"))
    assert metrics == {"val_loss": 1.2, "adapter": "sft", "generated": len(rows)}
    # a DPO adapter is evaluated by default once it exists, the SFT one on request
    dpo = workdir / "output" / "dpo"
    dpo.mkdir(parents=True)
    (dpo / "adapter_model.safetensors").write_bytes(b"x")
    run_script(workdir, env, "eval.sh", "5090-8b")
    assert str(cli_calls(log)[-1]["adapter"]).endswith("/output/dpo")
    run_script(workdir, env, "eval.sh", "5090-8b", "sft")
    assert str(cli_calls(log)[-1]["adapter"]).endswith("/output/sft")
    assert run_script(workdir, env, "eval.sh", "5090-8b", "nope").returncode == 1


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
printf '%s' "$3-model" >"$2"
"""


@posix_only
def test_the_export_runs_the_steps_in_order_and_deletes_only_what_is_replaced(
    tmp_path: Path,
) -> None:
    workdir = prepared_workdir(tmp_path, "5090-14b")
    install_fake_cli(tmp_path)
    env = script_env(
        workdir,
        tmp_path,
        TWIN_PROBE_OVERRIDES=write_overrides(tmp_path, probe_free_gb="echo 900"),
        TWIN_RUN_ID="r-test-1",
    )
    sft = workdir / "output" / "sft"
    sft.mkdir(parents=True)
    (sft / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (sft / "adapter_config.json").write_text("{}", encoding="utf-8")
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
    trace = workdir.parent / "trace.log"

    done = run_script(workdir, env, "export.sh", "5090-14b")
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
    assert sorted(p.name for p in (artifacts / "gguf").iterdir()) == [
        "5090-14b-Q4_K_M.gguf",
        "5090-14b-Q5_K_M.gguf",
        "5090-14b-Q8_0.gguf",
    ]
    assert (artifacts / "adapter" / "adapter_model.safetensors").read_bytes() == b"adapter-weights"
    manifest = json.loads((artifacts / "manifest.json").read_text("utf-8"))
    assert manifest["run_id"] == "r-test-1" and manifest["profile"] == "5090-14b"
    assert manifest["template_version"] == "qwen3_nothink@llamafactory-0.9.5"
    assert manifest["persona_version"] == "v3" and manifest["dataset_version"] == "ds-test-01"
    assert [m["quant"] for m in manifest["models"]] == ["Q4_K_M", "Q5_K_M", "Q8_0", "lora"]
    listed = {f["path"] for f in manifest["files"]}
    assert "gguf/5090-14b-Q4_K_M.gguf" in listed and "adapter/adapter_model.safetensors" in listed
    export_call = next(
        c for c in cli_calls(tmp_path / "stubs" / "cli_calls.log") if c["mode"] == "export"
    )
    assert str(export_call["adapter"]).endswith("/output/sft")
    # a second run has nothing left to do
    again = run_script(workdir, env, "export.sh", "5090-14b")
    assert again.returncode == 0 and "already done" in again.stdout
    assert len(trace.read_text("utf-8").splitlines()) == 4


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
