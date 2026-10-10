"""Artifact directories as ``export.sh`` leaves them (manifest, GGUF files, adapter, metrics)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0")


def _tool() -> ModuleType:
    path = ROOT / "training" / "autodl" / "tools" / "artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("artifact_manifest_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bundle_manifest(
    profile: str = "5090-8b", dataset_version: str = "ds-test-01"
) -> dict[str, str]:
    return {
        "profile": profile,
        "base_model": "Qwen/Qwen3-8B",
        "template": "qwen3_nothink",
        "template_version": "qwen3_nothink@llamafactory-0.9.5",
        "persona_version": "v3",
        "profile_version": "01TESTPROFILEVERSION00000",
        "dataset_version": dataset_version,
        "llamafactory_version": "0.9.5",
        "llama_cpp_tag": "b11177",
    }


def build_artifact_dir(
    directory: Path,
    *,
    profile: str = "5090-8b",
    run_id: str | None = "r-test-1",
    dataset_version: str = "ds-test-01",
    sizes: int = 64,
    versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write model files and ``manifest.json`` (made by the very tool the export script runs)."""
    for quant in QUANTS:
        path = directory / "gguf" / f"{profile}-{quant}.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"gguf-{quant}".encode() * sizes)
    adapter = directory / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"lora-weights" * sizes)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    train = directory / "train"
    train.mkdir(exist_ok=True)
    (train / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_loss": 1.25,
                "best_checkpoint": "checkpoint-8",
                "peak_vram_mib": 21000,
                "gpu_name": "NVIDIA GeForce RTX 5090",
            }
        ),
        encoding="utf-8",
    )
    manifest = _tool().build(
        directory,
        {**bundle_manifest(profile, dataset_version), **(versions or {})},
        list(QUANTS),
        "sft",
        run_id or "",
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest  # type: ignore[no-any-return]
