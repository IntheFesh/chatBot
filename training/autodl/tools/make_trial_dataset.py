#!/usr/bin/env python3
"""The data of the one-step trial run of ``setup.sh verify`` (standard library only).

The trial exists to find out whether a batch fits into the GPU memory.  Memory is decided by the
*longest* samples, but LLaMA-Factory's ``max_samples`` keeps the first N rows of the file, which
are the earliest and have nothing to do with length.  This tool writes a small dataset of the
longest training samples instead:

    make_trial_dataset.py --data-dir DATA --dataset NAME --out-dir TRIAL [--count 64]

``TRIAL/sft_trial.jsonl`` holds the ``--count`` longest rows of the file that ``DATA/dataset_info.json``
names for the dataset ``NAME`` (rows are copied unchanged; the length is the number of characters of
``system`` plus all turns, because no tokenizer is available here and the exporter has cut every
sample to the context length of the model already); ``TRIAL/dataset_info.json`` has one entry,
``twin_sft_trial``, a copy of the entry of ``NAME`` that points at that file, so the columns and tags
stay the ones the exporter wrote.  Nothing about the content is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRIAL_DATASET = "twin_sft_trial"
TRIAL_FILE = "sft_trial.jsonl"
DEFAULT_COUNT = 64


class TrialDataError(Exception):
    """The training data cannot be used for the trial (the message never contains a sample)."""


def sample_length(row: object) -> int:
    """Characters of the system prompt and of every turn of a ShareGPT row."""
    if not isinstance(row, dict):
        raise TrialDataError("a row is not a JSON object")
    system = row.get("system") or ""
    turns = row.get("conversations")
    if not isinstance(system, str) or not isinstance(turns, list):
        raise TrialDataError("a row has no system text or no conversations list")
    total = len(system)
    for turn in turns:
        value = turn.get("value") if isinstance(turn, dict) else None
        if not isinstance(value, str):
            raise TrialDataError("a turn has no text")
        total += len(value)
    return total


def select_longest(lines: list[str], count: int) -> list[tuple[int, str]]:
    """``(length, line)`` of the ``count`` longest rows, longest first (ties: the earlier row)."""
    measured: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            length = sample_length(json.loads(line))
        except json.JSONDecodeError:
            raise TrialDataError(f"line {index + 1} is not JSON") from None
        except TrialDataError as exc:
            raise TrialDataError(f"line {index + 1}: {exc}") from None
        measured.append((-length, index, line))
    measured.sort()
    return [(-negative, line) for negative, _, line in measured[: max(count, 0)]]


def build(data_dir: Path, dataset: str, out_dir: Path, count: int) -> tuple[int, int, int, int]:
    """Write the trial dataset; ``(selected, total, shortest, longest)`` (lengths of the selected)."""
    try:
        info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise TrialDataError(f"{data_dir / 'dataset_info.json'} cannot be read") from None
    entry = info.get(dataset) if isinstance(info, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("file_name"), str):
        raise TrialDataError(f"dataset_info.json has no dataset {dataset!r} with a file_name")
    source = data_dir / entry["file_name"]
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise TrialDataError(f"{source} cannot be read") from None
    chosen = select_longest(lines, count)
    if not chosen:
        raise TrialDataError(f"{source.name} has no samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / TRIAL_FILE).write_text("".join(f"{line}\n" for _, line in chosen), encoding="utf-8")
    trial_info = {TRIAL_DATASET: {**entry, "file_name": TRIAL_FILE}}
    (out_dir / "dataset_info.json").write_text(
        json.dumps(trial_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    total = sum(1 for line in lines if line.strip())
    return len(chosen), total, chosen[-1][0], chosen[0][0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args(argv)
    try:
        selected, total, shortest, longest = build(
            args.data_dir, args.dataset, args.out_dir, args.count
        )
    except TrialDataError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(
        f"trial dataset: the {selected} longest of {total} samples "
        f"({shortest} to {longest} characters)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
