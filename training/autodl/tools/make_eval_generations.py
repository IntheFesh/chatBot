#!/usr/bin/env python3
"""Turn LLaMA-Factory's prediction file into ``eval_generations.jsonl`` (standard library only).

    make_eval_generations.py --test sft_test.jsonl --predictions generated_predictions.jsonl \
        --eval-results eval_results.json --adapter sft --out-dir artifacts/eval

LLaMA-Factory writes ``{"prompt", "predict", "label"}`` per line in the order of the test file,
without the sample ids.  The ids are put back by position, and the position is double-checked:
the decoded label of every prediction must equal the reply of the test sample at that position.
The output holds the sample id and the generated text only, and ``eval_metrics.json`` the
validation loss.  Error messages never contain any text of the data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_lines(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge(
    tests: list[dict[str, object]], predictions: list[dict[str, object]]
) -> list[dict[str, str]]:
    if len(tests) != len(predictions):
        raise ValueError(
            f"{len(predictions)} predictions for {len(tests)} test samples; "
            "they must match one to one"
        )
    merged: list[dict[str, str]] = []
    mismatches = 0
    for sample, prediction in zip(tests, predictions, strict=True):
        conversations = sample["conversations"]
        if not isinstance(conversations, list):
            raise ValueError("a test sample has no conversations")
        reply = str(conversations[-1]["value"]).strip()
        if str(prediction.get("label", "")).strip() != reply:
            mismatches += 1
        merged.append({"id": str(sample["id"]), "text": str(prediction.get("predict", "")).strip()})
    if mismatches:
        raise ValueError(
            f"{mismatches} predictions do not belong to the test sample at their position"
        )
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--eval-results", type=Path)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        merged = merge(read_lines(args.test), read_lines(args.predictions))
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "eval_generations.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for row in merged:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    val_loss = None
    if args.eval_results is not None and args.eval_results.is_file():
        val_loss = json.loads(args.eval_results.read_text(encoding="utf-8")).get("eval_loss")
    metrics = {"val_loss": val_loss, "adapter": args.adapter, "generated": len(merged)}
    (args.out_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"wrote {len(merged)} generations; validation loss {val_loss}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
