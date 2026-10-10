"""Which style model is in use: the active row of ``model_registry`` (R-SRV-001, R-SRV-005).

Registering (round 13) fills the table, activating (round 14, ``twin model activate``) sets the
``active`` flag of one model - after the release gate, or with ``--force`` which leaves
``gate_passed`` as it was.  The engine only reads: :class:`StyleModels` answers "is there a model,
which one, which versions is it locked to, did it pass the gate", for the file kind the
configured server mode serves (a GGUF for llama.cpp, the LoRA adapter for vLLM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twin.storage.db import Database
from twin.training.registry import (
    EVAL_TOKENIZE_CHECK,
    LockedVersions,
    ModelView,
    get_model,
    list_models,
)

SERVED_KINDS = {"llamacpp_completion": "gguf", "vllm_completion": "adapter"}


@dataclass(frozen=True)
class ActiveStyleModel:
    """The model the style backends use, with the versions its prompts are rendered with."""

    id: str
    run_id: str
    quant: str
    kind: str
    versions: LockedVersions
    gate_passed: bool | None
    tokenizer_ok: bool | None = None
    """The last tokenizer comparison (R-TRN-011.4): ``False`` refuses the model, ``None`` is not
    checked yet (the serving component checks it each time it starts the server)."""

    @property
    def passed_gate(self) -> bool:
        """True only when the release gate said so; a forced activation does not count."""
        return self.gate_passed is True

    @property
    def label(self) -> str:
        return f"{self.run_id} {self.quant}"


def tokenizer_verdict(document: dict[str, Any]) -> bool | None:
    """``ok`` of the recorded tokenizer comparison of a registry ``eval`` document."""
    record = document.get(EVAL_TOKENIZE_CHECK)
    if isinstance(record, dict) and isinstance(record.get("ok"), bool):
        return bool(record["ok"])
    return None


def _active(view: ModelView) -> ActiveStyleModel:
    return ActiveStyleModel(
        view.id,
        view.run_id,
        view.quant,
        view.kind,
        view.versions,
        view.gate_passed,
        tokenizer_verdict(view.eval),
    )


class StyleModels:
    """Reads the registry for the engine (see the module description)."""

    def __init__(
        self, db: Database, *, mode: str = "llamacpp_completion", pinned: str | None = None
    ) -> None:
        self._db = db
        self._kind = SERVED_KINDS.get(mode)
        self._pinned = pinned

    def registered(self) -> int:
        """How many model files are registered (any kind)."""
        return len(list_models(self._db))

    def active(self) -> ActiveStyleModel | None:
        """The newest active model of the kind this server mode serves (else of any kind).

        A ``pinned`` instance (the evaluation of a model that is not active yet, R-SRV-005)
        always answers with that one model.
        """
        if self._pinned is not None:
            return _active(get_model(self._db, self._pinned))
        active = [view for view in list_models(self._db) if view.active]
        if not active:
            return None
        preferred = [view for view in active if view.kind == self._kind]
        return _active((preferred or active)[0])
