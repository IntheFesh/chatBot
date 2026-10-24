"""The graphics card and the llama.cpp build that fits it (R-SRV-002).

``nvidia-smi`` is the one source: the card's name, the driver, the memory, the compute capability
and the newest CUDA version the *driver* supports (the header of the plain output).  From that
:func:`choose_build` picks one of the three Windows builds of the pinned llama.cpp release - the
same rule ``scripts/windows/get_llamacpp.ps1`` applies when it downloads (the script cannot call
Python before the environment exists, so the rule is written twice and a test compares the numbers):

``cuda-13.4``
    the driver supports CUDA 13 and the card is Turing or newer (compute capability 7.5+).  This
    build has native kernels for Blackwell (RTX 50 series, compute capability 12.0); the CUDA 12.4
    build only reaches such a card through PTX translation at load time.
``cuda-12.4``
    the driver supports CUDA 12.4 or newer and the CUDA 13 build is not an option (an older
    driver, or a Pascal/Volta card that CUDA 13 no longer supports).
``cpu``
    no NVIDIA card, or a driver too old for either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from twin.ops.taskscheduler import CommandRunner, decode_output

GPU_QUERY_FULL: Final = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total,compute_cap",
    "--format=csv,noheader,nounits",
]
GPU_QUERY_BASIC: Final = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader,nounits",
]
SMI_PLAIN: Final = ["nvidia-smi"]
CUDA_VERSION: Final = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")

BuildKind = Literal["cuda-13.4", "cuda-12.4", "cpu"]
CUDA_13: Final = (13, 0)  # the driver must support at least this for the "cuda-13.4" build
CUDA_12_4: Final = (12, 4)
TURING: Final = (7, 5)  # the oldest architecture the CUDA 13 build supports
BLACKWELL_MAJOR: Final = 12


@dataclass(frozen=True)
class GpuInfo:
    """One NVIDIA card."""

    name: str
    driver: str
    memory_mib: int
    compute_cap: tuple[int, int] | None = None
    cuda: tuple[int, int] | None = None  # the newest CUDA the driver supports

    @property
    def memory_bytes(self) -> int:
        return self.memory_mib * 1024 * 1024

    @property
    def blackwell(self) -> bool:
        return self.compute_cap is not None and self.compute_cap[0] >= BLACKWELL_MAJOR

    def describe(self) -> str:
        text = f"{self.name}, {self.memory_mib} MiB, driver {self.driver}"
        if self.compute_cap is not None:
            text += f", compute capability {self.compute_cap[0]}.{self.compute_cap[1]}"
        if self.cuda is not None:
            text += f", driver supports CUDA {self.cuda[0]}.{self.cuda[1]}"
        return text


@dataclass(frozen=True)
class BuildChoice:
    """The llama.cpp build for a card, and why."""

    kind: BuildKind
    reason: str

    @property
    def cuda(self) -> bool:
        return self.kind != "cpu"


def _capability(text: str) -> tuple[int, int] | None:
    found = re.fullmatch(r"(\d+)\.(\d+)", text.strip())
    return (int(found.group(1)), int(found.group(2))) if found else None


def parse_gpu_query(text: str) -> list[GpuInfo]:
    """The cards in the CSV of ``nvidia-smi --query-gpu`` (3 or 4 columns, no units)."""
    cards: list[GpuInfo] = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 3 or not cells[2].isdigit():
            continue
        capability = _capability(cells[3]) if len(cells) > 3 else None
        cards.append(GpuInfo(cells[0], cells[1], int(cells[2]), capability))
    return cards


def parse_cuda_version(text: str) -> tuple[int, int] | None:
    """The CUDA version in the header of the plain ``nvidia-smi`` output."""
    found = CUDA_VERSION.search(text)
    return (int(found.group(1)), int(found.group(2))) if found else None


def detect_gpu(runner: CommandRunner) -> GpuInfo | None:
    """The card with the most memory, or ``None`` when ``nvidia-smi`` is missing or finds none."""
    try:
        done = runner.run(GPU_QUERY_FULL, timeout_s=15.0)
        if done.returncode != 0:  # old drivers do not know the compute_cap field
            done = runner.run(GPU_QUERY_BASIC, timeout_s=15.0)
    except Exception:  # no nvidia-smi on this computer: there is no card to use
        return None
    if done.returncode != 0:
        return None
    cards = parse_gpu_query(decode_output(done.stdout))
    if not cards:
        return None
    best = max(cards, key=lambda card: card.memory_mib)
    try:
        plain = runner.run(SMI_PLAIN, timeout_s=15.0)
    except Exception:
        return best
    cuda = parse_cuda_version(decode_output(plain.stdout)) if plain.returncode == 0 else None
    return GpuInfo(best.name, best.driver, best.memory_mib, best.compute_cap, cuda)


def choose_build(gpu: GpuInfo | None) -> BuildChoice:
    """The Windows build of llama.cpp to use with ``gpu`` (see the module description)."""
    if gpu is None:
        return BuildChoice("cpu", "no NVIDIA card was found")
    if gpu.cuda is None:
        return BuildChoice("cpu", "the CUDA version of the driver is unknown")
    modern = gpu.compute_cap is None or gpu.compute_cap >= TURING
    if gpu.cuda >= CUDA_13 and modern:
        why = "native kernels for this card" if gpu.blackwell else "driver supports CUDA 13"
        return BuildChoice("cuda-13.4", why)
    if gpu.cuda >= CUDA_12_4:
        why = "driver supports CUDA 12.4"
        if gpu.blackwell:
            why += " (the card is translated at load time: update the driver for CUDA 13)"
        elif not modern:
            why += " (CUDA 13 no longer supports this card)"
        return BuildChoice("cuda-12.4", why)
    return BuildChoice("cpu", f"the driver supports CUDA {gpu.cuda[0]}.{gpu.cuda[1]} only")
