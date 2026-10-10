"""GPU profiles of the style model training (R-TRN-001) and what they need from the instance.

Four profiles, one per row of the table in R-TRN-001.  This module is the single source of every
number that differs between them: the base model, the LoRA shape, the batch layout, the
free disk space and memory the AutoDL scripts must find (R-TRN-009), the epoch rule and the
evaluation schedule.  The YAML templates (:mod:`twin.training.yaml_render`) and the environment
file the shell scripts source (:func:`profile_env`) are both generated from here, so the scripts
never repeat a number.

Disk space (``required_disk_gb``)
    the peak is reached while the F16 GGUF still exists: the base model, that F16 copy and all
    three quantised files (Q4_K_M, Q5_K_M and Q8_0 are together about 1.19 times the bf16 size)
    are on the data disk at the same moment, next to the Python environments, the caches and the
    checkpoints (about 15 GB).  The result is ``base_gb * 3.19 + 15``, rounded up: 68 GB for 8B,
    110 GB for 14B and 225 GB for 32B, which is what R-TRN-009 asks for ("about 70, 110 and
    230 GB").  The merged bf16 copy is deleted before the quantisation, so it adds nothing to it.
Memory (``required_ram_gb``)
    a QLoRA adapter is merged on the CPU into the bf16 base (``export_device: cpu``), which needs
    the whole bf16 model in RAM; the other profiles merge on the GPU and only map the weights.
Epochs
    3 below 20,000 training samples, 2 from there on (R-TRN-001).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

DISK_PEAK_FACTOR: Final = 3.19
DISK_OVERHEAD_GB: Final = 15
RAM_CPU_MERGE_FACTOR: Final = 1.3
RAM_CPU_MERGE_EXTRA_GB: Final = 4
RAM_GPU_MERGE_FACTOR: Final = 0.5
RAM_GPU_MERGE_EXTRA_GB: Final = 8

EPOCH_SAMPLE_LIMIT: Final = 20_000
CUTOFF_LEN: Final = 2048
LEARNING_RATE: Final = "1.0e-4"
WARMUP_RATIO: Final = 0.05
EARLY_STOPPING_PATIENCE: Final = 2
MODELS_ROOT: Final = "/root/autodl-tmp/models"
DEFAULT_WORKDIR: Final = "/root/autodl-tmp/twin"

DPO_LEARNING_RATE: Final = "5.0e-6"
DPO_BETA: Final = 0.1
DPO_EPOCHS: Final = 2

Method = Literal["lora", "qlora"]
ExportDevice = Literal["cpu", "auto"]


class ProfileError(ValueError):
    """An unknown profile name or an unusable profile setting."""


@dataclass(frozen=True)
class TrainingProfile:
    """One GPU profile of R-TRN-001."""

    name: str
    gpu: str
    vram_gb: int
    base_model: str  # Hugging Face and ModelScope id, e.g. Qwen/Qwen3-8B
    base_gb: float  # size of the bf16 weights in the ModelScope repository
    method: Method
    lora_rank: int
    lora_alpha: int
    quantization_bit: int | None
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    export_device: ExportDevice
    spec_disk_gb: int  # the round number of R-TRN-009 the user is told to expand the disk to

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def required_disk_gb(self) -> int:
        return math.ceil(self.base_gb * DISK_PEAK_FACTOR + DISK_OVERHEAD_GB)

    @property
    def required_ram_gb(self) -> int:
        if self.export_device == "cpu":
            return math.ceil(self.base_gb * RAM_CPU_MERGE_FACTOR + RAM_CPU_MERGE_EXTRA_GB)
        return math.ceil(self.base_gb * RAM_GPU_MERGE_FACTOR + RAM_GPU_MERGE_EXTRA_GB)

    @property
    def model_dir(self) -> str:
        """Where ``setup.sh`` downloads the bf16 base model on the instance."""
        return f"{MODELS_ROOT}/{self.base_model}"


PROFILES: Final[dict[str, TrainingProfile]] = {
    "5090-8b": TrainingProfile(
        name="5090-8b",
        gpu="RTX 5090 32GB",
        vram_gb=32,
        base_model="Qwen/Qwen3-8B",
        base_gb=16.4,
        method="lora",
        lora_rank=32,
        lora_alpha=64,
        quantization_bit=None,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        export_device="auto",
        spec_disk_gb=70,
    ),
    "5090-14b": TrainingProfile(
        name="5090-14b",
        gpu="RTX 5090 32GB",
        vram_gb=32,
        base_model="Qwen/Qwen3-14B",
        base_gb=29.6,
        method="qlora",
        lora_rank=32,
        lora_alpha=64,
        quantization_bit=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        export_device="cpu",
        spec_disk_gb=110,
    ),
    "pro6000-14b": TrainingProfile(
        name="pro6000-14b",
        gpu="RTX PRO 6000 96GB",
        vram_gb=96,
        base_model="Qwen/Qwen3-14B",
        base_gb=29.6,
        method="lora",
        lora_rank=32,
        lora_alpha=64,
        quantization_bit=None,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        export_device="auto",
        spec_disk_gb=110,
    ),
    "pro6000-32b": TrainingProfile(
        name="pro6000-32b",
        gpu="RTX PRO 6000 96GB",
        vram_gb=96,
        base_model="Qwen/Qwen3-32B",
        base_gb=65.6,
        method="lora",
        lora_rank=16,
        lora_alpha=32,
        quantization_bit=None,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        export_device="auto",
        spec_disk_gb=230,
    ),
}


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)


def get_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ProfileError(
            f"unknown profile {name!r}; choose one of {', '.join(PROFILES)}"
        ) from None


def epochs_for(train_samples: int) -> int:
    """Three epochs for fewer than 20,000 training samples, two from there on (R-TRN-001)."""
    if train_samples < 0:
        raise ProfileError("the number of samples cannot be negative")
    return 3 if train_samples < EPOCH_SAMPLE_LIMIT else 2


@dataclass(frozen=True)
class StepPlan:
    """Evaluation and checkpoint schedule derived from the size of the training set."""

    steps_per_epoch: int
    total_steps: int
    eval_steps: int
    logging_steps: int


def plan_steps(train_samples: int, profile: TrainingProfile, epochs: int) -> StepPlan:
    """About four evaluations (and checkpoints) per epoch, at least one step apart."""
    if train_samples < 1:
        raise ProfileError("a training run needs at least one training sample")
    steps_per_epoch = math.ceil(train_samples / profile.effective_batch_size)
    eval_steps = max(1, min(500, math.ceil(steps_per_epoch / 4)))
    return StepPlan(
        steps_per_epoch=steps_per_epoch,
        total_steps=steps_per_epoch * epochs,
        eval_steps=eval_steps,
        logging_steps=max(1, min(10, eval_steps)),
    )


def profile_env(profile: TrainingProfile, *, dpo_min_pairs: int = 200) -> str:
    """The ``KEY=value`` lines the AutoDL scripts source (``autodl/profile.env``).

    Every value is a plain word or number, so the file is safe to ``source``.
    """
    values = {
        "TWIN_PROFILE": profile.name,
        "TWIN_BASE_MODEL": profile.base_model,
        "TWIN_BASE_GB": f"{profile.base_gb:g}",
        "TWIN_BASE_GB_CEIL": str(math.ceil(profile.base_gb)),
        "TWIN_METHOD": profile.method,
        "TWIN_QUANT_BIT": str(profile.quantization_bit or 0),
        "TWIN_EXPORT_DEVICE": profile.export_device,
        "TWIN_VRAM_GB": str(profile.vram_gb),
        "TWIN_REQUIRED_DISK_GB": str(profile.required_disk_gb),
        "TWIN_REQUIRED_RAM_GB": str(profile.required_ram_gb),
        "TWIN_SPEC_DISK_GB": str(profile.spec_disk_gb),
        "TWIN_BATCH_SIZE": str(profile.per_device_train_batch_size),
        "TWIN_GRAD_ACCUM": str(profile.gradient_accumulation_steps),
        "TWIN_LORA_RANK": str(profile.lora_rank),
        "TWIN_DPO_MIN_PAIRS": str(dpo_min_pairs),
    }
    return "".join(f"{key}={value}\n" for key, value in values.items())
