#!/usr/bin/env python3
"""Write ``artifacts/manifest.json`` of a finished training run (standard library only).

    artifact_manifest.py --artifacts DIR --bundle-manifest FILE --quants Q4_K_M,Q5_K_M,Q8_0 \
        --adapter-kind sft --run-id ID

The manifest lists every file below the artifacts directory with its size and sha256 and names
the files a model registry should register (``models``).  The versions the model was trained
with - template, persona card, profile and dataset - are copied from the training package's
manifest, so ``twin model register`` can lock the model to them (R-SRV-001).
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
LOCKED_KEYS = (
    "profile",
    "base_model",
    "template",
    "template_version",
    "persona_version",
    "profile_version",
    "dataset_version",
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


def build(
    artifacts: Path, bundle: dict[str, object], quants: list[str], adapter_kind: str, run_id: str
) -> dict[str, object]:
    profile = str(bundle["profile"])
    models: list[dict[str, str]] = []
    for quant in quants:
        path = f"gguf/{profile}-{quant}.gguf"
        if not (artifacts / path).is_file():
            raise FileNotFoundError(f"{path} is missing")
        models.append({"quant": quant, "kind": "gguf", "path": path})
    adapter = "adapter/adapter_model.safetensors"
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
    train = read_json(artifacts / "train" / "metrics.json")
    evaluation = read_json(artifacts / "eval" / "eval_metrics.json")
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "run_id": run_id or None,
        "created_at": datetime.now(UTC).isoformat(),
        "adapter_kind": adapter_kind,
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
