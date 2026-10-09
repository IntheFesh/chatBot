"""Synthetic training data for the bundle, script and registry tests (no real chat content)."""

from __future__ import annotations

from pathlib import Path

from twin.training import lf_template
from twin.training.dataset_dir import DatasetDir, DpoSample, SftSample, write_dataset_dir
from twin.training.lf_template import Turn

SYSTEM = "She writes short lines, drops full stops and uses the smiley code [Smile] now and then."
WORDS = ("tea", "rain", "train", "lamp", "bridge", "kettle", "garden", "window", "notebook", "bus")


def sample(index: int, *, turns: int = 3, prefix: str = "s") -> SftSample:
    """One synthetic sample whose context has ``turns`` turns (user first)."""
    context = tuple(
        Turn(
            "user" if i % 2 == 0 else "assistant",
            f"line {i} about the {WORDS[(index + i) % len(WORDS)]}",
        )
        for i in range(turns if turns % 2 == 1 else turns + 1)
    )
    reply = f"ok the {WORDS[index % len(WORDS)]} is fine\n[表情包:笑]"
    return SftSample(f"{prefix}{index}", SYSTEM, context, reply)


def pair(index: int) -> DpoSample:
    context = (Turn("user", f"question {index} about the {WORDS[index % len(WORDS)]}"),)
    return DpoSample(
        f"p{index}", SYSTEM, context, f"short answer {index}", f"a long formal answer {index}"
    )


def write_synthetic_dataset(
    directory: Path,
    *,
    train: int = 40,
    val: int = 6,
    test: int = 6,
    pairs: int = 0,
    redacted: bool = True,
    version: str = "ds-test-01",
) -> DatasetDir:
    return write_dataset_dir(
        directory,
        dataset_version=version,
        created_at="2026-10-09T12:00:00+00:00",
        template_version=lf_template.TEMPLATE_VERSION,
        persona_version="v3",
        profile_version="01TESTPROFILEVERSION00000",
        holdout_cutoff="2026-09-01T00:00:00+00:00",
        train=[sample(i, turns=1 + 2 * (i % 4), prefix="tr") for i in range(train)],
        val=[sample(i, prefix="va") for i in range(val)],
        test=[sample(i, prefix="te") for i in range(test)],
        dpo=[pair(i) for i in range(pairs)],
        redacted=redacted,
        plan_ratio=0.3,
        stats={"note": "synthetic"},
    )
