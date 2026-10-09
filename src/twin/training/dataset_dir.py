"""The directory a dataset export leaves behind and the training bundle reads (R-TRN-002/008).

::

    <dataset>/dataset_meta.json   what the data is: version, cut-off, versions of the persona
                                  card and the profile, counts, file hashes
    <dataset>/sft_train.jsonl     ShareGPT samples (one JSON object per line)
    <dataset>/sft_val.jsonl
    <dataset>/sft_test.jsonl
    <dataset>/dpo_train.jsonl     optional: preference pairs (ShareGPT with chosen / rejected)

A sample line is ``{"id": ..., "system": ..., "conversations": [{"from": "human", "value": ...},
{"from": "gpt", "value": ...}, ...]}``: the system message first (the training prompt without the
conversation), then user and her turns alternating, ending with her reply, which is the only
part the model learns (``mask_history``).  A preference line has an odd number of
``conversations`` (ending with a human turn) and the two replies as ``chosen`` and ``rejected``.

:func:`write_dataset_dir` is what the exporter calls; :func:`load_dataset_dir` is what the bundle
builder calls.  The loader checks everything the training cannot afford to find out on the
instance: hashes, counts, line structure, role order, control tokens, and that the exporter
declared the data desensitised (R-TRN-007, R-PRIV-003).  Error messages carry line numbers and
file names but never message text.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from twin.training import lf_template
from twin.training.layout import (
    FILE_DATASET_META,
    FILE_DPO,
    FILE_TEST,
    FILE_TRAIN,
    FILE_VAL,
)

SCHEMA: Final = 1
HASH_BLOCK = 1 << 20


class DatasetError(ValueError):
    """The dataset directory cannot be used for training."""


class DatasetCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: int = Field(ge=0)
    val: int = Field(ge=0)
    test: int = Field(ge=0)
    dpo: int = Field(default=0, ge=0)


class DatasetMeta(BaseModel):
    """``dataset_meta.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA, alias="schema")
    dataset_version: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    created_at: str
    scope: str = "pre_holdout"
    template_version: str
    persona_version: str
    profile_version: str
    holdout_cutoff: str
    range_from: str | None = None
    range_to: str | None = None
    plan_ratio: float = Field(default=0.0, ge=0, le=1)
    redacted: bool
    counts: DatasetCounts
    files: dict[str, str]
    stats: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class SftSample:
    """One training sample: the context, her reply and the system message that frames them."""

    id: str
    system: str
    turns: tuple[lf_template.Turn, ...]
    reply: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "system": self.system,
            "conversations": lf_template.sharegpt_conversations(self.turns, self.reply),
        }


@dataclass(frozen=True)
class DpoSample:
    """One preference pair: the context and the preferred and rejected replies."""

    id: str
    system: str
    turns: tuple[lf_template.Turn, ...]
    chosen: str
    rejected: str

    def to_json(self) -> dict[str, Any]:
        lf_template.check_alternation(self.turns)
        conversations = [
            {
                "from": lf_template.ROLE_USER if t.role == "user" else lf_template.ROLE_ASSISTANT,
                "value": t.content,
            }
            for t in self.turns
        ]
        return {
            "id": self.id,
            "system": self.system,
            "conversations": conversations,
            "chosen": {"from": lf_template.ROLE_ASSISTANT, "value": self.chosen},
            "rejected": {"from": lf_template.ROLE_ASSISTANT, "value": self.rejected},
        }


@dataclass(frozen=True)
class DatasetDir:
    """A validated dataset directory."""

    path: Path
    meta: DatasetMeta

    @property
    def has_dpo(self) -> bool:
        return self.meta.counts.dpo > 0

    def file_path(self, name: str) -> Path:
        return self.path / name

    def data_files(self) -> list[str]:
        names = [FILE_TRAIN, FILE_VAL, FILE_TEST]
        if self.has_dpo:
            names.append(FILE_DPO)
        return names


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _write_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_dataset_dir(
    directory: Path,
    *,
    dataset_version: str,
    created_at: str,
    template_version: str,
    persona_version: str,
    profile_version: str,
    holdout_cutoff: str,
    train: Iterable[SftSample],
    val: Iterable[SftSample],
    test: Iterable[SftSample],
    dpo: Iterable[DpoSample] = (),
    redacted: bool,
    plan_ratio: float = 0.0,
    range_from: str | None = None,
    range_to: str | None = None,
    stats: Mapping[str, Any] | None = None,
) -> DatasetDir:
    """Write the JSONL files and ``dataset_meta.json`` of an export; returns the checked result."""
    directory.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for key, name, samples in (
        ("train", FILE_TRAIN, train),
        ("val", FILE_VAL, val),
        ("test", FILE_TEST, test),
    ):
        counts[key] = _write_lines(directory / name, (s.to_json() for s in samples))
    pairs = list(dpo)
    counts["dpo"] = _write_lines(directory / FILE_DPO, (s.to_json() for s in pairs)) if pairs else 0
    if not pairs:
        (directory / FILE_DPO).unlink(missing_ok=True)
    files = {
        name: file_sha256(directory / name)
        for name in (FILE_TRAIN, FILE_VAL, FILE_TEST, *([FILE_DPO] if pairs else []))
    }
    meta = DatasetMeta.model_validate(
        {
            "schema": SCHEMA,
            "dataset_version": dataset_version,
            "created_at": created_at,
            "template_version": template_version,
            "persona_version": persona_version,
            "profile_version": profile_version,
            "holdout_cutoff": holdout_cutoff,
            "range_from": range_from,
            "range_to": range_to,
            "plan_ratio": plan_ratio,
            "redacted": redacted,
            "counts": counts,
            "files": files,
            "stats": dict(stats or {}),
        }
    )
    (directory / FILE_DATASET_META).write_text(
        json.dumps(meta.model_dump(by_alias=True), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return load_dataset_dir(directory)


def _check_conversation(name: str, number: int, row: Mapping[str, Any], *, pairwise: bool) -> None:
    where = f"{name} line {number}"
    system = row.get("system", "")
    conversations = row.get("conversations")
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise DatasetError(f"{where}: every sample needs an id")
    if not isinstance(system, str):
        raise DatasetError(f"{where}: system must be text")
    if not isinstance(conversations, list) or not conversations:
        raise DatasetError(f"{where}: conversations must be a non-empty list")
    expected = (lf_template.ROLE_USER, lf_template.ROLE_ASSISTANT)
    texts = [system]
    for index, message in enumerate(conversations):
        if not isinstance(message, dict) or message.get("from") != expected[index % 2]:
            raise DatasetError(
                f"{where}: message {index} must come from {expected[index % 2]!r} "
                "(human first, alternating)"
            )
        value = message.get("value")
        if not isinstance(value, str) or not value:
            raise DatasetError(f"{where}: message {index} has no text")
        texts.append(value)
    if pairwise:
        if len(conversations) % 2 == 0:
            raise DatasetError(f"{where}: a preference prompt must end with a human message")
        for key in ("chosen", "rejected"):
            reply = row.get(key)
            if (
                not isinstance(reply, dict)
                or reply.get("from") != lf_template.ROLE_ASSISTANT
                or not isinstance(reply.get("value"), str)
                or not reply["value"]
            ):
                raise DatasetError(f"{where}: {key} must be a gpt message with text")
            texts.append(reply["value"])
    elif len(conversations) % 2 == 1:
        raise DatasetError(f"{where}: a sample must end with her reply (gpt)")
    for text in texts:
        try:
            lf_template.check_content(text)
        except lf_template.TemplateError as exc:
            raise DatasetError(f"{where}: {exc}") from None


def _check_file(path: Path, *, pairwise: bool) -> int:
    seen: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                raise DatasetError(f"{path.name} line {number}: empty line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise DatasetError(f"{path.name} line {number}: not valid JSON") from None
            if not isinstance(row, dict):
                raise DatasetError(f"{path.name} line {number}: not a JSON object")
            _check_conversation(path.name, number, row, pairwise=pairwise)
            if row["id"] in seen:
                raise DatasetError(f"{path.name} line {number}: duplicate sample id")
            seen.add(row["id"])
            count += 1
    return count


def load_dataset_dir(directory: Path) -> DatasetDir:
    """Read and validate a dataset directory (see the module docstring)."""
    meta_path = directory / FILE_DATASET_META
    if not meta_path.is_file():
        raise DatasetError(f"{directory} has no {FILE_DATASET_META}; export the dataset first")
    try:
        meta = DatasetMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise DatasetError(
            f"{FILE_DATASET_META} is not valid: {exc.error_count()} error(s)"
        ) from exc
    if not meta.redacted:
        raise DatasetError(
            "the dataset is not marked as desensitised; only redacted data may leave this machine"
        )
    if meta.counts.train < 1 or meta.counts.val < 1 or meta.counts.test < 1:
        raise DatasetError("the train, validation and test splits must each have a sample")
    result = DatasetDir(directory, meta)
    expected_files = set(result.data_files())
    if set(meta.files) != expected_files:
        raise DatasetError(
            f"{FILE_DATASET_META} lists {sorted(meta.files)}, expected {sorted(expected_files)}"
        )
    counts = {
        FILE_TRAIN: meta.counts.train,
        FILE_VAL: meta.counts.val,
        FILE_TEST: meta.counts.test,
        FILE_DPO: meta.counts.dpo,
    }
    for name in sorted(expected_files):
        path = directory / name
        if not path.is_file():
            raise DatasetError(f"{name} is missing")
        if file_sha256(path) != meta.files[name]:
            raise DatasetError(f"{name} does not match its recorded sha256")
        actual = _check_file(path, pairwise=name == FILE_DPO)
        if actual != counts[name]:
            raise DatasetError(f"{name} has {actual} samples but the metadata says {counts[name]}")
    return result
