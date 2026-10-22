#!/usr/bin/env python3
"""Put the adapter stack that was kept into ``artifacts/`` (R-TRN-009, D-650, D-652, D-653).

    adapter_stack.py --artifacts DIR --choice sft|dpo --mode layered|continued \
        --layer sft=output/sft [--layer dpo=output/dpo]

What a stack is
    ``sft``                  one adapter, the SFT one.
    ``dpo`` of a bf16 profile
                             ``layered``: the DPO adapter was trained on the SFT model merged into
                             the base, so the model is ``W + dSFT + dDPO`` - two adapters, applied
                             one after the other (``adapter_name_or_path: sft,dpo``).
    ``dpo`` of the 4-bit profile
                             ``continued``: the SFT adapter was continued, ``dpo`` alone is the
                             whole adapter (a stack of one).

What this tool writes
    ``artifacts/adapter/``   the adapter a vLLM server loads (``--lora-modules``): the only adapter of
                             a one-layer stack, or the two layers **composed into one** by
                             concatenating them along the rank (below).
    ``artifacts/adapters/<name>/``
                             the layers themselves, for a stack of two or more, so that each file
                             can be checked against its own sha256 when the model is registered.
    ``artifacts/adapter_stack.json``
                             the stack in order, every layer and the served adapter with rank, alpha,
                             size and sha256; ``artifact_manifest.py`` copies it into the manifest.

Composing (the one-adapter form of ``W + dSFT + dDPO``)
    A LoRA layer adds ``s * B @ A`` to a weight, with ``A`` of shape (r, in), ``B`` of shape
    (out, r) and ``s = alpha / r`` (``alpha / sqrt(r)`` with rank-stabilised LoRA).  For two
    layers 1 and 2 of the same module

        A = [A1; A2]            (r1 + r2, in)    stacked along the rank
        B = [s1 * B1 | s2 * B2] (out, r1 + r2)   the scales folded into B, side by side
        B @ A = s1 * B1 @ A1 + s2 * B2 @ A2

    so the composed adapter, written with ``lora_alpha = r1 + r2`` (scale 1), adds exactly the sum
    of the two deltas.  Nothing is merged into a weight: no disk, no rounding of a bf16 base.  It
    only holds for plain LoRA on the same modules, so anything else (DoRA, ``modules_to_save``,
    rank or alpha patterns, a bias, a different module set) is refused instead of guessed at.

Needs ``torch`` and ``safetensors`` for composing (the instance has both once the packages are
installed); describing and copying use the standard library only.  Nothing about the data is
printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

SCHEMA = 1
BLOCK = 1 << 20
ADAPTER_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"
EXTRA_FILES = ("README.md",)
SERVED_DIR = "adapter"
LAYERS_DIR = "adapters"
STACK_FILE = "adapter_stack.json"
CHOICES = ("sft", "dpo")
MODES = ("layered", "continued")
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


class StackError(Exception):
    """The adapters cannot be put together; the message says why (never a sample)."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def read_config(directory: Path) -> dict[str, Any]:
    path = directory / CONFIG_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise StackError(f"{path} cannot be read") from None
    if not isinstance(value, dict):
        raise StackError(f"{path} is not a JSON object")
    return value


def rank_and_alpha(config: dict[str, Any]) -> tuple[int, float]:
    try:
        rank, alpha = int(config["r"]), float(config["lora_alpha"])
    except (KeyError, TypeError, ValueError):
        raise StackError("an adapter_config.json has no usable r and lora_alpha") from None
    if rank < 1 or alpha <= 0:
        raise StackError("an adapter has a rank or an alpha below 1")
    return rank, alpha


def scale_of(config: dict[str, Any]) -> float:
    """``alpha / r``, or ``alpha / sqrt(r)`` for rank-stabilised LoRA."""
    rank, alpha = rank_and_alpha(config)
    return alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank


def check_composable(name: str, config: dict[str, Any]) -> None:
    """Refuse every form of LoRA the rank concatenation does not describe exactly."""
    problems = []
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        problems.append(f"peft_type {config.get('peft_type')}")
    for key in ("use_dora", "use_qalora", "lora_bias"):
        if config.get(key):
            problems.append(key)
    for key in ("modules_to_save", "rank_pattern", "alpha_pattern", "trainable_token_indices"):
        if config.get(key):
            problems.append(key)
    if config.get("bias", "none") != "none":
        problems.append(f"bias {config.get('bias')}")
    if problems:
        raise StackError(f"the adapter {name} cannot be composed: {', '.join(problems)}")


def compose_config(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``adapter_config.json`` of the composed adapter: rank and alpha are the sums (scale 1)."""
    first = configs[0]
    total = sum(rank_and_alpha(config)[0] for config in configs)
    if any(bool(config.get("fan_in_fan_out")) != bool(first.get("fan_in_fan_out")) for config in configs):
        raise StackError("the adapters disagree on fan_in_fan_out")
    targets: list[str] = []
    for config in configs:
        raw = config.get("target_modules") or []
        for target in [raw] if isinstance(raw, str) else raw:
            if target not in targets:
                targets.append(target)
    composed = dict(first)
    composed.update(
        {"r": total, "lora_alpha": total, "use_rslora": False, "target_modules": targets,
         "rank_pattern": {}, "alpha_pattern": {}}
    )
    return composed


def compose_tensors(
    layers: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """The tensors of the composed adapter from ``(name, config, tensors)`` of each layer."""
    import torch  # on the instance since the packages are installed; see the module description

    first_name, _, first_tensors = layers[0]
    keys = sorted(first_tensors)
    for name, _, tensors in layers[1:]:
        if sorted(tensors) != keys:
            raise StackError(
                f"the adapters {first_name} and {name} do not cover the same weights, "
                "so they cannot be composed"
            )
    composed: dict[str, Any] = {}
    for key in keys:
        if key.endswith(A_SUFFIX):
            composed[key] = torch.cat(
                [tensors[key].to(torch.float32) for _, _, tensors in layers], dim=0
            )
        elif key.endswith(B_SUFFIX):
            composed[key] = torch.cat(
                [
                    tensors[key].to(torch.float32) * scale_of(config)
                    for _, config, tensors in layers
                ],
                dim=1,
            )
        else:
            raise StackError(f"the adapter has a weight that is not a LoRA matrix: {key}")
    return composed


def compose_files(sources: list[tuple[str, Path]], target: Path) -> dict[str, Any]:
    """Compose the adapters in ``sources`` (name, directory; in the order they are applied)."""
    from safetensors.torch import load_file, save_file

    loaded = []
    for name, directory in sources:
        config = read_config(directory)
        check_composable(name, config)
        loaded.append((name, config, load_file(str(directory / ADAPTER_FILE))))
    tensors = compose_tensors(loaded)
    config = compose_config([config for _, config, _ in loaded])
    target.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        str(target / ADAPTER_FILE),
        metadata={"format": "pt"},
    )
    (target / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def copy_adapter(source: Path, target: Path) -> None:
    if not (source / ADAPTER_FILE).is_file():
        raise StackError(f"{source / ADAPTER_FILE} is missing")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in (ADAPTER_FILE, CONFIG_FILE, *EXTRA_FILES):
        if (source / name).is_file():
            shutil.copyfile(source / name, target / name)


def describe(artifacts: Path, directory: Path, name: str | None) -> dict[str, Any]:
    """The entry of an adapter directory below ``artifacts``: path, size, sha256, rank, alpha."""
    file = directory / ADAPTER_FILE
    rank, alpha = rank_and_alpha(read_config(directory))
    entry: dict[str, Any] = {
        "path": file.relative_to(artifacts).as_posix(),
        "sha256": sha256_of(file),
        "size": file.stat().st_size,
        "rank": rank,
        "alpha": alpha,
    }
    return {"name": name, **entry} if name else entry


def kind_of(choice: str, mode: str) -> str:
    """``sft``; ``sft+dpo`` (two layers) for a layered profile; ``dpo`` (one adapter) otherwise."""
    if choice == "sft":
        return "sft"
    return "sft+dpo" if mode == "layered" else "dpo"


def collect(
    artifacts: Path, choice: str, mode: str, layers: list[tuple[str, Path]]
) -> dict[str, Any]:
    """Fill ``artifacts/adapter`` (and ``adapters/``) and write ``adapter_stack.json``."""
    if choice not in CHOICES or mode not in MODES:
        raise StackError(f"choice {choice!r} or mode {mode!r} is unknown")
    kind = kind_of(choice, mode)
    expected = ["sft", "dpo"] if kind == "sft+dpo" else [choice]
    if [name for name, _ in layers] != expected:
        raise StackError(
            f"the stack {kind} has the layers {', '.join(expected)}, "
            f"not {', '.join(name for name, _ in layers) or 'none'}"
        )
    served_dir = artifacts / SERVED_DIR
    layer_entries: list[dict[str, Any]]
    if len(layers) == 1:
        copy_adapter(layers[0][1], served_dir)
        layer_entries = [describe(artifacts, served_dir, layers[0][0])]
        composed = False
    else:
        for name, source in layers:
            copy_adapter(source, artifacts / LAYERS_DIR / name)
        layer_entries = [
            describe(artifacts, artifacts / LAYERS_DIR / name, name) for name, _ in layers
        ]
        if served_dir.exists():
            shutil.rmtree(served_dir)
        compose_files([(name, artifacts / LAYERS_DIR / name) for name, _ in layers], served_dir)
        composed = True
    served = describe(artifacts, served_dir, None)
    served["composed"] = composed
    stack = {
        "schema": SCHEMA,
        "choice": choice,
        "mode": mode,
        "kind": kind,
        "layers": layer_entries,
        "served": served,
    }
    (artifacts / STACK_FILE).write_text(json.dumps(stack, indent=2) + "\n", encoding="utf-8")
    return stack


def parse_layer(text: str) -> tuple[str, Path]:
    name, separator, directory = text.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError("--layer takes NAME=DIRECTORY")
    return name, Path(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--choice", required=True, choices=CHOICES)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--layer", action="append", required=True, type=parse_layer)
    args = parser.parse_args(argv)
    try:
        stack = collect(args.artifacts, args.choice, args.mode, args.layer)
    except (StackError, OSError, ImportError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    served = stack["served"]
    sys.stdout.write(
        f"adapter stack {stack['kind']}: {len(stack['layers'])} layer(s), served adapter of rank "
        f"{served['rank']}{' (composed)' if served['composed'] else ''}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
