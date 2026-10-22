"""The adapter stack of a trained model (R-SRV-001, R-TRN-009; D-650, D-652, D-653).

A model is not one adapter any more: the preference run (DPO) is trained on top of the SFT model,
so what the evaluation and the export use is a **stack**, an ordered list of adapters.

``sft``
    the SFT adapter alone.
``sft+dpo`` (a bf16 profile, ``layered``)
    the DPO adapter was trained on the SFT adapter merged into the base, so the model is
    ``W + dSFT + dDPO``: two adapters, applied one after the other (``adapter_name_or_path:
    sft,dpo``).  The remote vLLM serves them as **one** adapter, composed by concatenating the two
    along the rank with the scales folded into ``B`` (``training/autodl/tools/adapter_stack.py``);
    ``served.composed`` says so.
``dpo`` (the 4-bit profile, ``continued``)
    a quantised base cannot take a new adapter, so the SFT adapter was continued; the DPO adapter
    alone is the whole stack.

``export.sh`` writes the stack into ``artifacts/manifest.json`` (``adapter_stack``), every layer
with its own sha256; :func:`verify_stack` checks each of them against the listed files when the
model is registered, and the registry keeps the stack in the ``eval`` document of every row.  A
model registered before the stack existed has only ``adapter_kind``; :func:`legacy_stack` reads it
as the one adapter it was.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

KINDS: Final = ("sft", "sft+dpo", "dpo")
MODES: Final = ("layered", "continued")
LEGACY_ADAPTER_PATH: Final = "adapter/adapter_model.safetensors"


class StackError(ValueError):
    """A manifest's adapter stack is malformed or does not match the files it lists."""


@dataclass(frozen=True)
class AdapterLayer:
    """One adapter file of the stack: where it is below the artifact directory and what it is."""

    name: str
    path: str
    sha256: str | None
    size: int | None
    rank: int | None
    alpha: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "rank": self.rank,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class ServedAdapter:
    """The one adapter a vLLM server loads; composed from the layers when there are two."""

    path: str
    sha256: str | None
    size: int | None
    rank: int | None
    alpha: float | None
    composed: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "rank": self.rank,
            "alpha": self.alpha,
            "composed": self.composed,
        }


@dataclass(frozen=True)
class AdapterStack:
    """The ordered adapters a model was made from and the single adapter that is served."""

    choice: str  # the stack that was kept: sft | dpo
    mode: str  # layered | continued (how a DPO run of this profile stacks)
    kind: str  # sft | sft+dpo | dpo
    layers: tuple[AdapterLayer, ...]
    served: ServedAdapter
    legacy: bool = False  # read from an old manifest that had no stack

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    def describe(self) -> str:
        """``sft+dpo: sft (r32, 1a2b3c4d) -> dpo (r32, 5e6f7a8b); served: composed, r64``."""
        parts = []
        for layer in self.layers:
            facts = [f"r{layer.rank}"] if layer.rank is not None else []
            if layer.sha256:
                facts.append(layer.sha256[:8])
            parts.append(layer.name + (f" ({', '.join(facts)})" if facts else ""))
        served = "composed" if self.served.composed else "as it is"
        if self.served.rank is not None:
            served += f", r{self.served.rank}"
        note = " [registered before adapter stacks existed]" if self.legacy else ""
        return f"{self.kind}: {' -> '.join(parts)}; served: {served}{note}"

    def to_json(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "mode": self.mode,
            "kind": self.kind,
            "layers": [layer.to_json() for layer in self.layers],
            "served": self.served.to_json(),
            "legacy": self.legacy,
        }


def _int_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _layer(entry: object) -> AdapterLayer:
    if not isinstance(entry, Mapping) or not entry.get("name") or not entry.get("path"):
        raise StackError("an adapter layer of the manifest has no name or path")
    return AdapterLayer(
        name=str(entry["name"]),
        path=str(entry["path"]),
        sha256=str(entry["sha256"]) if entry.get("sha256") else None,
        size=_int_or_none(entry.get("size")),
        rank=_int_or_none(entry.get("rank")),
        alpha=_float_or_none(entry.get("alpha")),
    )


def stack_from_json(data: Mapping[str, Any]) -> AdapterStack:
    """The stack of a manifest's ``adapter_stack`` (or of a registry row's copy of it)."""
    layers = data.get("layers")
    served = data.get("served")
    if not isinstance(layers, list) or not layers or not isinstance(served, Mapping):
        raise StackError("the adapter stack lists no layers or no served adapter")
    kind, mode, choice = str(data.get("kind")), str(data.get("mode")), str(data.get("choice"))
    if kind not in KINDS or mode not in MODES or choice not in ("sft", "dpo"):
        raise StackError(f"the adapter stack has an unknown kind {kind!r} or mode {mode!r}")
    stack = AdapterStack(
        choice=choice,
        mode=mode,
        kind=kind,
        layers=tuple(_layer(entry) for entry in layers),
        served=ServedAdapter(
            path=str(served.get("path", "")),
            sha256=str(served["sha256"]) if served.get("sha256") else None,
            size=_int_or_none(served.get("size")),
            rank=_int_or_none(served.get("rank")),
            alpha=_float_or_none(served.get("alpha")),
            composed=bool(served.get("composed")),
        ),
        legacy=bool(data.get("legacy")),
    )
    expected = {"sft": ["sft"], "sft+dpo": ["sft", "dpo"], "dpo": ["dpo"]}[kind]
    if list(stack.names) != expected:
        raise StackError(
            f"the stack {kind} has the layers {', '.join(expected)}, not {', '.join(stack.names)}"
        )
    if not stack.served.path:
        raise StackError("the served adapter of the stack has no path")
    if stack.served.composed != (len(stack.layers) > 1):
        raise StackError("only a stack of more than one layer is composed into the served adapter")
    ranks = [layer.rank for layer in stack.layers if layer.rank is not None]
    if (
        stack.served.composed
        and len(ranks) == len(stack.layers)
        and stack.served.rank != sum(ranks)
    ):
        raise StackError(
            "the rank of the composed adapter is not the sum of the ranks of its layers"
        )
    return stack


def legacy_stack(
    adapter_kind: str | None, *, sha256: str | None = None, size: int | None = None
) -> AdapterStack:
    """The stack of a manifest that has none: the one adapter ``adapter_kind`` names.

    Before the stacks the DPO run continued the SFT adapter, so ``dpo`` there is one adapter that
    holds both - ``continued`` - and ``sft`` is the SFT adapter.
    """
    choice = "dpo" if adapter_kind == "dpo" else "sft"
    layer = AdapterLayer(choice, LEGACY_ADAPTER_PATH, sha256, size, None, None)
    served = ServedAdapter(LEGACY_ADAPTER_PATH, sha256, size, None, None, composed=False)
    return AdapterStack(choice, "continued", choice, (layer,), served, legacy=True)


def stack_of_manifest(
    manifest: Mapping[str, Any], files: Mapping[str, Mapping[str, Any]]
) -> AdapterStack:
    """The stack a manifest describes, verified against ``files`` (path -> manifest entry)."""
    raw = manifest.get("adapter_stack")
    if raw is None:
        entry = files.get(LEGACY_ADAPTER_PATH, {})
        return legacy_stack(
            str(manifest.get("adapter_kind") or "sft"),
            sha256=str(entry["sha256"]) if entry.get("sha256") else None,
            size=_int_or_none(entry.get("size")),
        )
    if not isinstance(raw, Mapping):
        raise StackError("the adapter stack of the manifest is not an object")
    stack = stack_from_json(raw)
    verify_stack(stack, files)
    return stack


def verify_stack(stack: AdapterStack, files: Mapping[str, Mapping[str, Any]]) -> None:
    """Each layer and the served adapter is a listed file with its own sha256 and size.

    The manifest's file list is checked against the disk by the registry; this ties every layer to
    its entry, so one that was swapped for another listed file (or never listed) is refused.
    """
    checked: list[tuple[str, str, str | None, int | None]] = [
        (layer.name, layer.path, layer.sha256, layer.size) for layer in stack.layers
    ]
    served = stack.served
    checked.append(("the served adapter", served.path, served.sha256, served.size))
    for name, path, sha256, size in checked:
        listed = files.get(path)
        if listed is None:
            raise StackError(f"{path} ({name}) is part of the adapter stack but not listed")
        if sha256 is None or listed.get("sha256") != sha256:
            raise StackError(f"{path} ({name}) does not match its sha256 in the adapter stack")
        if size is not None and listed.get("size") != size:
            raise StackError(f"{path} ({name}) has another size than the adapter stack says")
