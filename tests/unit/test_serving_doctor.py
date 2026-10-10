"""``twin doctor``: the llama.cpp installation fits the card and is complete (R-SRV-002, R-OPS-009).

The folders are written the way ``get_llamacpp.ps1`` unpacks a release; ``nvidia-smi`` is scripted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.ops import ScriptedRunner, failed, ok
from twin.config.settings import Settings
from twin.ops.doctor import CheckStatus, DoctorContext, check_llamacpp
from twin.ops.taskscheduler import CommandResult
from twin.serving.llamacpp import CUDA_RUNTIME, WINDOWS_BASE_FILES

TAG = "b11177"


def smi(cuda: str | None, *, name: str = "NVIDIA GeForce RTX 5090") -> ScriptedRunner:
    """A computer with one card whose driver supports ``cuda`` (``None``: no ``nvidia-smi``)."""
    if cuda is None:
        return ScriptedRunner({})

    def answer(args: list[str]) -> CommandResult:
        if any(part.startswith("--query-gpu") for part in args):
            return ok(f"{name}, 580.97, 32607, 12.0\n")
        return ok(f"| NVIDIA-SMI 580.97   Driver Version: 580.97   CUDA Version: {cuda} |\n")

    return ScriptedRunner({"nvidia-smi": answer})


def install(root: Path, kind: str, *, drop: tuple[str, ...] = (), tag: str = TAG) -> Path:
    """A Windows installation of ``kind`` (``cpu``, ``cuda-12.4``, ``cuda-13.4``)."""
    folder = root / "tools" / "llama.cpp" / tag
    folder.mkdir(parents=True, exist_ok=True)
    names = list(WINDOWS_BASE_FILES)
    if kind == "cpu":
        names.append("ggml-cpu-x64.dll")
    else:
        names += ["ggml-cuda.dll", *CUDA_RUNTIME[kind]]
    for name in names:
        if name not in drop:
            (folder / name).write_bytes(b"MZ")
    return folder


def context(root: Path, runner: ScriptedRunner, *, wanted: bool = True) -> DoctorContext:
    conf = Settings()
    conf.style_model.mode = "llamacpp_completion"
    conf.backend.active = "style" if wanted else "deepseek"
    return DoctorContext(conf, root=root, platform="win32", runner=runner)


def test_a_configuration_that_did_not_load_skips_the_check() -> None:
    result = check_llamacpp(DoctorContext(None, "bad config"))
    assert result.status is CheckStatus.WARN and "skipped" in result.detail


def test_nothing_installed_matters_only_when_the_local_model_is_wanted(tmp_path: Path) -> None:
    quiet = check_llamacpp(context(tmp_path, smi(None), wanted=False))
    assert quiet.status is CheckStatus.OK and "only the local style model needs it" in quiet.detail
    wanting = check_llamacpp(context(tmp_path, smi(None), wanted=True))
    assert wanting.status is CheckStatus.WARN and "get_llamacpp.ps1" in wanting.hint
    remote = Settings()
    remote.style_model.mode = "vllm_completion"
    remote.backend.active = "style"
    ctx = DoctorContext(remote, root=tmp_path, platform="win32", runner=smi(None))
    assert check_llamacpp(ctx).status is CheckStatus.OK  # the model runs on the rented machine


def test_a_cuda_build_without_its_runtime_dlls_is_a_failure_when_it_is_needed(
    tmp_path: Path,
) -> None:
    install(tmp_path, "cuda-13.4", drop=CUDA_RUNTIME["cuda-13.4"])
    needed = check_llamacpp(context(tmp_path, smi("13.0")))
    assert needed.status is CheckStatus.FAIL and "incomplete" in needed.detail
    assert "cudart" in needed.detail and "-Force" in needed.hint
    spare = check_llamacpp(context(tmp_path, smi("13.0"), wanted=False))
    assert spare.status is CheckStatus.WARN


def test_a_missing_base_file_is_named(tmp_path: Path) -> None:
    install(tmp_path, "cpu", drop=("llama.dll",))
    result = check_llamacpp(context(tmp_path, smi(None)))
    assert result.status is CheckStatus.FAIL and "llama.dll" in result.detail


def test_the_cpu_build_on_a_computer_with_a_card_is_a_warning(tmp_path: Path) -> None:
    install(tmp_path, "cpu")
    result = check_llamacpp(context(tmp_path, smi("13.0")))
    assert result.status is CheckStatus.WARN and "CPU build is installed" in result.detail
    assert "RTX 5090" in result.detail and "-Force" in result.hint


def test_a_cuda_build_that_this_computer_cannot_use_is_a_warning(tmp_path: Path) -> None:
    install(tmp_path, "cuda-12.4")
    result = check_llamacpp(context(tmp_path, smi(None)))
    assert result.status is CheckStatus.WARN and "cannot be used here" in result.detail
    assert "-Build cpu" in result.hint


def test_the_older_cuda_build_on_a_card_with_a_cuda_13_driver_is_suggested_to_change(
    tmp_path: Path,
) -> None:
    install(tmp_path, "cuda-12.4")
    result = check_llamacpp(context(tmp_path, smi("13.0")))
    assert result.status is CheckStatus.WARN
    assert "CUDA 13 build fits this card better" in result.detail


@pytest.mark.parametrize(
    ("kind", "cuda"), [("cuda-13.4", "13.0"), ("cuda-12.4", "12.6"), ("cpu", None)]
)
def test_a_complete_installation_that_fits_the_computer_is_fine(
    tmp_path: Path, kind: str, cuda: str | None
) -> None:
    install(tmp_path, kind)
    result = check_llamacpp(context(tmp_path, smi(cuda)))
    assert result.status is CheckStatus.OK and TAG in result.detail and "complete" in result.detail


def test_the_newest_release_folder_is_the_one_that_counts(tmp_path: Path) -> None:
    install(tmp_path, "cpu", tag="b10000", drop=("llama.dll",))  # an old, broken folder
    install(tmp_path, "cpu", tag="b11177")
    assert check_llamacpp(context(tmp_path, smi(None))).status is CheckStatus.OK


def test_a_program_that_does_not_answer_is_no_card(tmp_path: Path) -> None:
    install(tmp_path, "cpu")
    runner = ScriptedRunner({"nvidia-smi": failed()})
    assert check_llamacpp(context(tmp_path, runner)).status is CheckStatus.OK
