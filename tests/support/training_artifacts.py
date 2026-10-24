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
        "holdout_cutoff": "2026-09-01T00:00:00+00:00",  # what the real package names (D-885)
        "llamafactory_version": "0.9.5",
        "llama_cpp_tag": "b11177",
    }


def _stack_tool() -> ModuleType:
    path = ROOT / "training" / "autodl" / "tools" / "adapter_stack.py"
    spec = importlib.util.spec_from_file_location("adapter_stack_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_artifact_dir(
    directory: Path,
    *,
    profile: str = "5090-8b",
    run_id: str | None = "r-test-1",
    dataset_version: str = "ds-test-01",
    sizes: int = 64,
    versions: dict[str, str] | None = None,
    stack: str | None = None,
    dpo_reference: dict[str, Any] | None = None,
    stack_choice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write model files and ``manifest.json`` (made by the very tool the export script runs).

    Without ``stack`` the directory is what an export before the adapter stacks left: one
    ``adapter/`` and no ``adapter_stack.json`` (a legacy manifest).  ``stack`` is ``sft``,
    ``sft+dpo`` (a layered profile: two real tiny adapters, composed by the tool) or ``dpo`` (the
    continued adapter of the 4-bit profile); ``dpo_reference`` and ``stack_choice`` are the
    records ``dpo.sh`` and ``twin train remote eval`` leave.
    """
    from tests.support.autodl_scripts import write_tiny_adapter

    for quant in QUANTS:
        path = directory / "gguf" / f"{profile}-{quant}.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"gguf-{quant}".encode() * sizes)
    adapter = directory / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    kept = "sft"
    if stack is None:
        (adapter / "adapter_model.safetensors").write_bytes(b"lora-weights" * sizes)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    else:
        names = {"sft": ["sft"], "sft+dpo": ["sft", "dpo"], "dpo": ["dpo"]}[stack]
        kept = "sft" if stack == "sft" else "dpo"
        mode = "continued" if stack == "dpo" else "layered"
        sources = {}
        for seed, name in enumerate(names):
            sources[name] = directory.parent / f"{directory.name}-training" / name
            write_tiny_adapter(sources[name], rank=4, alpha=8, seed=seed)
        _stack_tool().collect(directory, kept, mode, [(name, sources[name]) for name in names])
        for name in names:
            (directory / "eval" / name).mkdir(parents=True, exist_ok=True)
            (directory / "eval" / name / "eval_metrics.json").write_text(
                json.dumps({"val_loss": 1.2, "adapter": name, "generated": 3}), encoding="utf-8"
            )
    for relative, record in (
        ("dpo/dpo_reference.json", dpo_reference),
        ("eval/stack_choice.json", stack_choice),
    ):
        if record is not None:
            (directory / relative).parent.mkdir(parents=True, exist_ok=True)
            (directory / relative).write_text(json.dumps(record), encoding="utf-8")
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
        kept,
        run_id or "",
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest  # type: ignore[no-any-return]
