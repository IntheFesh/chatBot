#!/usr/bin/env python3
"""Summarise a finished LLaMA-Factory run (standard library only).

    summarize_train.py --output-dir output/sft --out-dir artifacts/train [--gpu-log FILE]

Reads ``trainer_state.json`` of the output directory and writes ``metrics.json`` (best
checkpoint, best validation loss, the evaluation history, peak GPU memory) next to copies of the
loss curve images LLaMA-Factory drew.  Prints the two numbers the operator wants to see.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CURVES = ("training_loss.png", "training_eval_loss.png")


def peak_memory_mib(gpu_log: Path | None) -> int | None:
    if gpu_log is None or not gpu_log.is_file():
        return None
    values = [int(line) for line in gpu_log.read_text(encoding="utf-8").split() if line.isdigit()]
    return max(values) if values else None


def summarise(state: dict[str, object], gpu_log: Path | None, gpu_name: str) -> dict[str, object]:
    history = state.get("log_history", [])
    if not isinstance(history, list):
        history = []
    evaluations = [
        [entry["step"], entry["eval_loss"]]
        for entry in history
        if isinstance(entry, dict) and "eval_loss" in entry and "step" in entry
    ]
    training_losses = [
        entry["loss"] for entry in history if isinstance(entry, dict) and "loss" in entry
    ]
    best_checkpoint = state.get("best_model_checkpoint")
    return {
        "best_checkpoint": Path(str(best_checkpoint)).name if best_checkpoint else None,
        "best_val_loss": state.get("best_metric"),
        "global_step": state.get("global_step"),
        "epochs_done": state.get("epoch"),
        "eval_history": evaluations,
        "last_train_loss": training_losses[-1] if training_losses else None,
        "peak_vram_mib": peak_memory_mib(gpu_log),
        "gpu_name": gpu_name or None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gpu-log", type=Path)
    parser.add_argument("--gpu-name", default="")
    args = parser.parse_args(argv)

    state_path = args.output_dir / "trainer_state.json"
    if not state_path.is_file():
        sys.stderr.write(f"error: {state_path} not found; did the run finish?\n")
        return 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    metrics = summarise(state, args.gpu_log, args.gpu_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    for name in CURVES:
        source = args.output_dir / name
        if source.is_file():
            shutil.copyfile(source, args.out_dir / name)
    sys.stdout.write(f"best checkpoint: {metrics['best_checkpoint']}\n")
    sys.stdout.write(f"best validation loss: {metrics['best_val_loss']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
