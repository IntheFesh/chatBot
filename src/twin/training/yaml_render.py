"""LLaMA-Factory configuration files for a profile (R-TRN-001, R-TRN-009).

The Jinja2 templates live in ``training/llamafactory/*.yaml.j2``; this module renders one profile
into the five configuration files the AutoDL scripts use and into ``dataset_info.json``:

``sft``
    LoRA / QLoRA fine-tuning on the training split with the validation split as ``eval_dataset``
    (LLaMA-Factory's own random ``val_size`` split is not used), evaluation and checkpoints every
    quarter epoch, the best checkpoint kept, early stopping after two evaluations without
    improvement.  Training resumes by itself from the last checkpoint when ``output_dir`` already
    holds one.
``dpo``
    preference training that continues the SFT adapter.
``eval_generate`` / ``eval_loss``
    one generated reply per test context with fixed sampling, and the loss on the validation set.
``export``
    merges the adapter into the *unquantised* bf16 base; no profile sets ``quantization_bit``
    here, and the 4-bit profile merges on the CPU.

Templates and numbers come from :mod:`twin.training.profiles`; the format strings of the chat
template are not repeated here - only its name is, and LLaMA-Factory resolves it.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import jinja2

from twin.training import lf_template
from twin.training.layout import (
    DATASET_DPO,
    DATASET_TEST,
    DATASET_TRAIN,
    DATASET_VAL,
    FILE_DPO,
    FILE_TEST,
    FILE_TRAIN,
    FILE_VAL,
    RemoteLayout,
)
from twin.training.profiles import (
    CUTOFF_LEN,
    DPO_BETA,
    DPO_EPOCHS,
    DPO_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    WARMUP_RATIO,
    TrainingProfile,
    epochs_for,
    plan_steps,
)

GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.9
GENERATION_MAX_NEW_TOKENS = 256


class ConfigKind(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    EVAL_GENERATE = "eval_generate"
    EVAL_LOSS = "eval_loss"
    EXPORT = "export"


class AssetsNotFoundError(FileNotFoundError):
    """The ``training/`` directory of the repository cannot be found."""


def training_assets_dir() -> Path:
    """The ``training/`` directory next to ``src/`` (templates and AutoDL scripts)."""
    candidate = Path(__file__).resolve().parents[3] / "training"
    if not (candidate / "llamafactory").is_dir():
        raise AssetsNotFoundError(
            f"{candidate} does not hold the LLaMA-Factory templates; run twin from the repository"
        )
    return candidate


def _environment() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(training_assets_dir() / "llamafactory"),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,  # noqa: S701 - YAML, not HTML
    )


def render_config(
    kind: ConfigKind,
    profile: TrainingProfile,
    *,
    train_samples: int,
    layout: RemoteLayout | None = None,
) -> str:
    """The YAML text of ``kind`` for ``profile``; ``train_samples`` sets epochs and eval steps."""
    paths = layout or RemoteLayout()
    epochs = epochs_for(train_samples)
    context = {
        "profile": profile,
        "llamafactory_version": lf_template.LLAMAFACTORY_VERSION,
        "template": lf_template.LF_TEMPLATE_NAME,
        "model_path": profile.model_dir,
        "data_dir": paths.data,
        "output_dir": paths.output,
        "dataset_train": DATASET_TRAIN,
        "dataset_val": DATASET_VAL,
        "dataset_test": DATASET_TEST,
        "dataset_dpo": DATASET_DPO,
        "cutoff_len": CUTOFF_LEN,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "epochs": epochs,
        "plan": plan_steps(train_samples, profile, epochs),
        "patience": EARLY_STOPPING_PATIENCE,
        "dpo_beta": DPO_BETA,
        "dpo_epochs": DPO_EPOCHS,
        "dpo_learning_rate": DPO_LEARNING_RATE,
        "dpo_grad_accum": 16,
        "temperature": GENERATION_TEMPERATURE,
        "top_p": GENERATION_TOP_P,
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
    }
    return _environment().get_template(f"{kind.value}.yaml.j2").render(**context)


def render_all(
    profile: TrainingProfile,
    *,
    train_samples: int,
    layout: RemoteLayout | None = None,
    with_dpo: bool = False,
) -> dict[str, str]:
    """``{file name: text}`` of every configuration file a bundle carries."""
    kinds = [ConfigKind.SFT, ConfigKind.EVAL_GENERATE, ConfigKind.EVAL_LOSS, ConfigKind.EXPORT]
    if with_dpo:
        kinds.append(ConfigKind.DPO)
    return {
        f"{kind.value}.yaml": render_config(
            kind, profile, train_samples=train_samples, layout=layout
        )
        for kind in kinds
    }


def dataset_info(*, with_dpo: bool) -> str:
    """``dataset_info.json``: the three SFT splits and, when the bundle has pairs, the DPO set.

    All of them are ShareGPT files (``system`` + alternating ``human``/``gpt`` conversations, the
    preference file with ``chosen`` and ``rejected`` replies after an odd number of messages).
    """
    tags = {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": lf_template.ROLE_USER,
        "assistant_tag": lf_template.ROLE_ASSISTANT,
        "system_tag": lf_template.ROLE_SYSTEM,
    }
    columns = {"messages": "conversations", "system": "system"}
    entries: dict[str, dict[str, object]] = {
        name: {"file_name": file_name, "formatting": "sharegpt", "columns": columns, "tags": tags}
        for name, file_name in (
            (DATASET_TRAIN, FILE_TRAIN),
            (DATASET_VAL, FILE_VAL),
            (DATASET_TEST, FILE_TEST),
        )
    }
    if with_dpo:
        entries[DATASET_DPO] = {
            "file_name": FILE_DPO,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {**columns, "chosen": "chosen", "rejected": "rejected"},
            "tags": tags,
        }
    return json.dumps(entries, indent=2, ensure_ascii=False) + "\n"
