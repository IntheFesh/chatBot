"""The graphics card and the llama.cpp build chosen for it (R-SRV-002)."""

from __future__ import annotations

import pytest

from twin.ops.taskscheduler import CommandResult
from twin.serving.hardware import (
    GPU_QUERY_BASIC,
    GPU_QUERY_FULL,
    SMI_PLAIN,
    GpuInfo,
    choose_build,
    detect_gpu,
    parse_cuda_version,
    parse_gpu_query,
)

PLAIN_OUTPUT = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 576.80                 Driver Version: 576.80         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
"""


class ScriptedRunner:
    """Answers ``nvidia-smi`` calls from a table of ``{tuple(args): (code, text)}``."""

    def __init__(self, answers: dict[tuple[str, ...], tuple[int, str]]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def run(self, args: list[str], *, timeout_s: float = 30.0) -> CommandResult:
        self.calls.append(args)
        code, text = self.answers.get(tuple(args), (1, ""))
        return CommandResult(code, text.encode("utf-8"), b"")


def test_the_csv_of_nvidia_smi_is_read_with_and_without_the_compute_capability() -> None:
    full = parse_gpu_query("NVIDIA GeForce RTX 5090, 576.80, 32607, 12.0\n")
    assert full == [GpuInfo("NVIDIA GeForce RTX 5090", "576.80", 32607, (12, 0))]
    basic = parse_gpu_query("NVIDIA GeForce RTX 3060, 551.23, 12288\n")
    assert basic == [GpuInfo("NVIDIA GeForce RTX 3060", "551.23", 12288, None)]
    assert parse_gpu_query("") == [] and parse_gpu_query("not csv at all") == []


def test_the_cuda_version_of_the_driver_is_in_the_header() -> None:
    assert parse_cuda_version(PLAIN_OUTPUT) == (12, 9)
    assert parse_cuda_version("no header here") is None


def test_the_card_with_the_most_memory_is_taken_and_the_driver_cuda_added() -> None:
    runner = ScriptedRunner(
        {
            tuple(GPU_QUERY_FULL): (0, "Small, 551.23, 8192, 8.6\nBig, 576.80, 32607, 12.0\n"),
            tuple(SMI_PLAIN): (0, PLAIN_OUTPUT),
        }
    )
    card = detect_gpu(runner)
    assert card == GpuInfo("Big", "576.80", 32607, (12, 0), (12, 9))
    assert card is not None and card.blackwell and card.memory_bytes == 32607 * 1024 * 1024
    assert "compute capability 12.0" in card.describe() and "CUDA 12.9" in card.describe()


def test_an_old_driver_without_the_compute_capability_field_is_asked_again_without_it() -> None:
    runner = ScriptedRunner(
        {
            tuple(GPU_QUERY_BASIC): (0, "NVIDIA GeForce GTX 1080, 472.12, 8192\n"),
            tuple(SMI_PLAIN): (0, "CUDA Version: 11.4"),
        }
    )
    card = detect_gpu(runner)
    assert card == GpuInfo("NVIDIA GeForce GTX 1080", "472.12", 8192, None, (11, 4))
    assert runner.calls[0] == GPU_QUERY_FULL and runner.calls[1] == GPU_QUERY_BASIC


def test_no_nvidia_smi_means_no_card() -> None:
    assert detect_gpu(ScriptedRunner({})) is None

    class Missing:
        def run(self, args: list[str], *, timeout_s: float = 30.0) -> CommandResult:
            raise OSError("not installed")

    assert detect_gpu(Missing()) is None


@pytest.mark.parametrize(
    ("card", "kind"),
    [
        (GpuInfo("RTX 5090", "576", 32607, (12, 0), (13, 0)), "cuda-13.4"),
        (GpuInfo("RTX 4090", "580", 24564, (8, 9), (13, 0)), "cuda-13.4"),
        (GpuInfo("RTX 5090", "570", 32607, (12, 0), (12, 8)), "cuda-12.4"),
        (GpuInfo("RTX 3060", "551.61", 12288, (8, 6), (12, 4)), "cuda-12.4"),
        (GpuInfo("GTX 1080", "580", 8192, (6, 1), (13, 0)), "cuda-12.4"),
        (GpuInfo("RTX 2070", "580", 8192, (7, 5), (13, 0)), "cuda-13.4"),
        (GpuInfo("RTX 3060", "531", 12288, (8, 6), (12, 1)), "cpu"),
        (GpuInfo("RTX 3060", "531", 12288, (8, 6), None), "cpu"),
        (GpuInfo("GTX 1080", "472", 8192, (6, 1), (11, 4)), "cpu"),
        (None, "cpu"),
    ],
)
def test_the_build_follows_the_driver_and_the_architecture(card: GpuInfo | None, kind: str) -> None:
    choice = choose_build(card)
    assert choice.kind == kind and choice.reason
    assert choice.cuda == (kind != "cpu")


def test_a_blackwell_card_on_the_cuda_12_build_is_told_to_update_its_driver() -> None:
    choice = choose_build(GpuInfo("RTX 5090", "570", 32607, (12, 0), (12, 8)))
    assert "update the driver" in choice.reason


def test_the_script_and_this_module_use_the_same_numbers() -> None:
    """get_llamacpp.ps1 repeats the rule (it runs before Python exists): the numbers must match."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "windows" / "get_llamacpp.ps1"
    ).read_text(encoding="utf-8")
    for needle in ("13.4", "12.4", "$Cuda13", "$Cuda124", "7.5"):
        assert needle in script, needle
