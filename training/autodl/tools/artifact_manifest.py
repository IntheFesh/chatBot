#!/usr/bin/env python3
"""Write ``artifacts/manifest.json`` of a finished training run (standard library only).

    artifact_manifest.py --artifacts DIR --bundle-manifest FILE --quants Q4_K_M,Q5_K_M,Q8_0 \
        --adapter-kind sft --run-id ID

The manifest lists every file below the artifacts directory with its size and sha256 and names
the files a model registry should register (``models``).  The versions the model was trained
with - template, persona card, profile and dataset - are copied from the training package's
manifest, so ``twin model register`` can lock the model to them (R-SRV-001); so is the hold-out
cut-off of the dataset, which the registry locks too (round 18, D-885).

The adapter stack that was kept (``adapter_stack.json`` of ``adapter_stack.py``: ``sft``,
``sft+dpo`` or ``dpo``, every layer with its sha256, and the one adapter vLLM serves) goes into the
manifest as ``adapter_stack``; each layer is checked against the file on the disk first, so a
layer that changed after it was recorded stops the export (D-653).  How the preference run got its
reference model (``dpo/dpo_reference.json``, D-651) and which stack won the evaluation
(``eval/stack_choice.json``, D-655) are copied as ``dpo`` and ``stack_choice``.  The metrics of the
evaluation are those of the stack that was kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = 1
BLOCK = 1 << 20
STACK_FILE = "adapter_stack.json"
DPO_REFERENCE_FILE = "dpo/dpo_reference.json"
STACK_CHOICE_FILE = "eval/stack_choice.json"
DEFAULT_SERVED = "adapter/adapter_model.safetensors"
LOCKED_KEYS = (
    "profile",
    "base_model",
    "template",
    "template_version",
    "persona_version",
    "profile_version",
    "dataset_version",
    "holdout_cutoff",  # the cut-off of the training set: M5 draws after it (round 18, D-885)
    "llamafactory_version",
    "llama_cpp_tag",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def check_stack(stack: dict[str, object], files: dict[str, str]) -> None:
    """Every layer and the served adapter must be a listed file with the recorded sha256."""
    layers = stack.get("layers")
    served = stack.get("served")
    if not isinstance(layers, list) or not layers or not isinstance(served, dict):
        raise ValueError(f"{STACK_FILE} lists no layers or no served adapter")
    for entry in [*layers, served]:
        if not isinstance(entry, dict):
            raise ValueError(f"{STACK_FILE} has an entry that is not an object")
        path = str(entry.get("path", ""))
        if path not in files:
            raise ValueError(f"{path} of {STACK_FILE} is not on the disk")
        if files[path] != entry.get("sha256"):
            raise ValueError(f"{path} changed after {STACK_FILE} was written")


def build(
    artifacts: Path, bundle: dict[str, object], quants: list[str], adapter_kind: str, run_id: str
) -> dict[str, object]:
    profile = str(bundle["profile"])
    stack = read_json(artifacts / STACK_FILE)
    served_entry = stack.get("served")
    adapter = (
        str(served_entry["path"])
        if isinstance(served_entry, dict) and served_entry.get("path")
        else DEFAULT_SERVED
    )
    models: list[dict[str, str]] = []
    for quant in quants:
        path = f"gguf/{profile}-{quant}.gguf"
        if not (artifacts / path).is_file():
            raise FileNotFoundError(f"{path} is missing")
        models.append({"quant": quant, "kind": "gguf", "path": path})
    if not (artifacts / adapter).is_file():
        raise FileNotFoundError(f"{adapter} is missing")
    models.append({"quant": "lora", "kind": "adapter", "path": adapter})

    files = []
    for path in sorted(artifacts.rglob("*")):
        if not path.is_file() or path == artifacts / "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(artifacts).as_posix(),
                "sha256": sha256_of(path),
                "size": path.stat().st_size,
            }
        )
    if stack:
        check_stack(stack, {str(entry["path"]): str(entry["sha256"]) for entry in files})
    kept = str(stack.get("choice") or adapter_kind or "sft")
    train = read_json(artifacts / "train" / "metrics.json")
    evaluation = read_json(artifacts / "eval" / kept / "eval_metrics.json") or read_json(
        artifacts / "eval" / "eval_metrics.json"
    )
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "run_id": run_id or None,
        "created_at": datetime.now(UTC).isoformat(),
        "adapter_kind": adapter_kind,
        "adapter_stack": stack or None,
        "dpo": read_json(artifacts / DPO_REFERENCE_FILE) or None,
        "stack_choice": read_json(artifacts / STACK_CHOICE_FILE) or None,
        "quant_set": quants,
        "models": models,
        "files": files,
        "metrics": {
            "best_val_loss": train.get("best_val_loss"),
            "best_checkpoint": train.get("best_checkpoint"),
            "peak_vram_mib": train.get("peak_vram_mib"),
            "gpu_name": train.get("gpu_name"),
            "eval_val_loss": evaluation.get("val_loss"),
            "generated": evaluation.get("generated"),
        },
    }
    for key in LOCKED_KEYS:
        manifest[key] = bundle.get(key)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--quants", required=True)
    parser.add_argument("--adapter-kind", default="sft")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args(argv)
    try:
        bundle = read_json(args.bundle_manifest)
        if not bundle:
            raise FileNotFoundError(f"{args.bundle_manifest} is missing or empty")
        manifest = build(
            args.artifacts,
            bundle,
            [q for q in args.quants.split(",") if q],
            args.adapter_kind,
            args.run_id,
        )
    except (OSError, KeyError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    (args.artifacts / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"manifest written: {len(manifest['files'])} files\n")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
