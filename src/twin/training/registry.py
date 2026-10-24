"""The model registry: register, list and show trained models (R-SRV-001).

``twin model register <artifact directory>`` takes what ``twin train remote download`` left in
``data/models/<run_id>/`` (or the same files copied from anywhere): the ``manifest.json`` that
``export.sh`` wrote and the files it lists.  Registering

* checks the manifest: every listed file must exist with the recorded size and sha256 and lie
  inside the directory;
* reads the four versions the model was trained with - prompt template, persona card (the
  pre-holdout compact card), statistical profile and dataset - from the manifest and **locks**
  them into the row.  The versions are required: a manifest without them is refused, because a
  model without them could be served with the wrong prompt (R-TRN-011);
* locks the **hold-out cut-off** of that dataset as well (round 18, user decision, D-885): the
  evaluation of the model draws only contexts after it (D-886), whatever the cut-off is later
  moved to.  The manifest names it (``holdout_cutoff`` of the training package); an older
  manifest does not, and the dataset version recorded at the export gives it; with neither the
  model is refused, and so is a manifest that contradicts the record.  Registering the files of a
  row that has no cut-off yet (registered before round 18) fills it in;
* writes one row per usable file: a GGUF per quantisation and the LoRA adapter (``quant`` is
  ``lora``).  Registering the same files again changes nothing; a different file under the same
  run and quantisation is refused.

Enabling, activating and the release gate belong to round 14 (:mod:`twin.serving.activation`);
this module owns the table and the few writes to it: :func:`activate_row`, :func:`deactivate_row`
and :func:`record_eval`.  The flags start false; ``gate_passed`` starts empty (never judged).
The ``eval`` document of a row keeps what was learnt about the model since it was registered:
``activation_log`` (every activation and deactivation: when, what, forced or not), ``gate`` (the
last verdict of the release gate) and ``tokenize_check`` (the last tokenizer comparison).

It also keeps how the model was made (D-650..D-655): ``adapter_stack`` (the ordered adapters of the
manifest, :mod:`twin.training.adapter_stack`; every layer is checked against its own sha256
before anything is written, and a manifest from before the stacks is read as the one adapter it
was), ``dpo_reference`` (which model the preference run used as its reference, and whether the
4-bit profile fell back to the base) and ``stack_choice`` (which stack the evaluation kept, with
the numbers).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.orm import Session

from twin.clock import ensure_aware
from twin.storage.db import Database
from twin.storage.training_models import DatasetVersion, ModelRegistryEntry, TrainingRun
from twin.training.adapter_stack import (
    AdapterStack,
    StackError,
    legacy_stack,
    stack_from_json,
    stack_of_manifest,
)
from twin.training.layout import ARTIFACT_MANIFEST
from twin.training.profiles import PROFILES

SCHEMA: Final = 1
HASH_BLOCK = 1 << 20
EVAL_ADAPTER_STACK: Final = "adapter_stack"
EVAL_DPO_REFERENCE: Final = "dpo_reference"
EVAL_STACK_CHOICE: Final = "stack_choice"
LOCKED_FIELDS: Final = ("template_version", "persona_version", "profile_version", "dataset_version")
EVAL_ACTIVATION_LOG: Final = "activation_log"
EVAL_GATE: Final = "gate"
EVAL_TOKENIZE_CHECK: Final = "tokenize_check"
ACTIVATION_LOG_KEEP: Final = 50


class RegistryError(ValueError):
    """An artifact directory cannot be registered, or a model does not exist."""


@dataclass(frozen=True)
class LockedVersions:
    """The versions a model is bound to; its prompts are rendered with exactly these."""

    template_version: str
    persona_version: str
    profile_version: str
    dataset_version: str


@dataclass(frozen=True)
class ModelView:
    """A row of the registry, detached from the database session."""

    id: str
    run_id: str
    kind: str
    profile: str
    base_model: str
    quant: str
    path: str
    sha256: str
    size: int
    versions: LockedVersions
    eval: dict[str, Any]
    enabled: bool
    active: bool
    gate_passed: bool | None
    created_at: str
    holdout_cutoff: datetime | None = None  # the cut-off of its training set (D-885)

    @property
    def adapter_stack(self) -> AdapterStack | None:
        """The adapters the model was made from; ``None`` when nothing is known about them.

        A row registered before the stacks has only ``adapter_kind`` in its ``eval`` document: it
        is read as the one adapter it was (D-653).
        """
        raw = self.eval.get(EVAL_ADAPTER_STACK)
        if isinstance(raw, dict):
            try:
                return stack_from_json(raw)
            except StackError:
                return None
        kind = self.eval.get("adapter_kind")
        if not kind:
            return None
        known = self.kind == "adapter"
        return legacy_stack(
            str(kind), sha256=self.sha256 if known else None, size=self.size if known else None
        )


@dataclass(frozen=True)
class RegistrationResult:
    run_id: str
    models: list[ModelView]
    created: int
    unchanged: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _view(row: ModelRegistryEntry) -> ModelView:
    return ModelView(
        id=row.id,
        run_id=row.run_id,
        kind=row.kind,
        profile=row.profile,
        base_model=row.base_model,
        quant=row.quant,
        path=row.path,
        sha256=row.sha256,
        size=row.size,
        versions=LockedVersions(
            row.template_version, row.persona_version, row.profile_version, row.dataset_version
        ),
        eval=dict(row.eval),
        enabled=row.enabled,
        active=row.active,
        gate_passed=row.gate_passed,
        created_at=row.created_at.isoformat(),
        holdout_cutoff=row.holdout_cutoff,
    )


def load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / ARTIFACT_MANIFEST
    if not path.is_file():
        raise RegistryError(
            f"{directory} has no {ARTIFACT_MANIFEST}; it is not an artifact directory"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise RegistryError(f"{path} is not valid JSON") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise RegistryError(f"{path} has an unsupported schema")
    return manifest


def _inside(directory: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not relative:
        raise RegistryError(f"manifest path {relative!r} leaves the artifact directory")
    return directory / candidate


def verify_artifacts(directory: Path, manifest: Mapping[str, Any]) -> None:
    """Every listed file must exist with the recorded size and sha256."""
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RegistryError("the manifest lists no files")
    for entry in files:
        path = _inside(directory, str(entry.get("path", "")))
        if not path.is_file():
            raise RegistryError(f"{entry['path']} is listed in the manifest but missing")
        if path.stat().st_size != entry.get("size"):
            raise RegistryError(f"{entry['path']} has a different size than the manifest says")
        if sha256_file(path) != entry.get("sha256"):
            raise RegistryError(f"{entry['path']} does not match its sha256 in the manifest")


def locked_versions(manifest: Mapping[str, Any]) -> LockedVersions:
    missing = [name for name in LOCKED_FIELDS if not str(manifest.get(name) or "").strip()]
    if missing:
        raise RegistryError(
            "the manifest does not name " + ", ".join(missing) + "; the model cannot be locked"
        )
    return LockedVersions(*(str(manifest[name]) for name in LOCKED_FIELDS))


def locked_cutoff(session: Session, manifest: Mapping[str, Any], dataset_version: str) -> datetime:
    """The hold-out cut-off of the dataset the model was trained on (see the module text)."""
    named: datetime | None = None
    raw = str(manifest.get("holdout_cutoff") or "").strip()
    if raw:
        try:
            named = ensure_aware(datetime.fromisoformat(raw))
        except ValueError:
            raise RegistryError(
                f"the manifest names a hold-out cut-off {raw!r} that is not a moment"
            ) from None
    row = session.get(DatasetVersion, dataset_version)
    recorded = row.holdout_cutoff if row is not None else None
    if named is not None and recorded is not None and named != recorded:
        raise RegistryError(
            f"the manifest says the hold-out cut-off was {named.isoformat()}, but dataset "
            f"{dataset_version} was exported with {recorded.isoformat()}"
        )
    found = named or recorded
    if found is None:
        raise RegistryError(
            f"the manifest names no hold-out cut-off and dataset {dataset_version} is not "
            "recorded here: the model cannot be locked to the cut-off it was trained with"
        )
    return found


def _stored_path(directory: Path, relative: str, models_dir: Path | None) -> str:
    file = (directory / relative).resolve()
    if models_dir is not None:
        try:
            return file.relative_to(models_dir.resolve()).as_posix()
        except ValueError:
            pass
    return str(file)


def register_artifacts(
    db: Database,
    directory: Path,
    *,
    models_dir: Path | None = None,
    run_id: str | None = None,
) -> RegistrationResult:
    """Verify ``directory`` and register its models; see the module docstring."""
    manifest = load_manifest(directory)
    verify_artifacts(directory, manifest)
    versions = locked_versions(manifest)
    profile = str(manifest.get("profile", ""))
    if profile not in PROFILES:
        raise RegistryError(f"the manifest names an unknown profile {profile!r}")
    chosen_run = run_id or manifest.get("run_id")
    if not chosen_run:
        raise RegistryError("the manifest has no run id; pass --run-id")
    chosen_run = str(chosen_run)
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise RegistryError("the manifest names no models")
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    try:
        stack = stack_of_manifest(manifest, by_path)
    except StackError as exc:
        raise RegistryError(str(exc)) from None
    metrics = dict(manifest.get("metrics") or {})
    metrics["adapter_kind"] = manifest.get("adapter_kind")
    metrics[EVAL_ADAPTER_STACK] = stack.to_json()
    if manifest.get("dpo"):
        metrics[EVAL_DPO_REFERENCE] = manifest["dpo"]
    if manifest.get("stack_choice"):
        metrics[EVAL_STACK_CHOICE] = manifest["stack_choice"]

    created = unchanged = 0
    views: list[ModelView] = []
    with db.transaction() as session:
        run = session.get(TrainingRun, chosen_run)
        if run is not None and run.dataset_version != versions.dataset_version:
            raise RegistryError(
                f"the run {chosen_run} used dataset {run.dataset_version}, "
                f"the manifest says {versions.dataset_version}"
            )
        cutoff = locked_cutoff(session, manifest, versions.dataset_version)
        for model in models:
            relative = str(model["path"])
            if relative not in by_path:
                raise RegistryError(f"{relative} is a model but not a listed file")
            quant, kind = str(model["quant"]), str(model["kind"])
            row_id = f"{chosen_run}-{quant}"
            sha, size = by_path[relative]["sha256"], int(by_path[relative]["size"])
            row = session.get(ModelRegistryEntry, row_id)
            if row is not None:
                if row.sha256 != sha:
                    raise RegistryError(
                        f"{row_id} is already registered with a different sha256; "
                        "a model file is never replaced"
                    )
                if row.holdout_cutoff is None:  # a row of an older registry gets its cut-off
                    row.holdout_cutoff = cutoff
                unchanged += 1
            else:
                row = ModelRegistryEntry(
                    id=row_id,
                    run_id=chosen_run,
                    kind=kind,
                    profile=profile,
                    base_model=str(manifest["base_model"]),
                    quant=quant,
                    path=_stored_path(directory, relative, models_dir),
                    sha256=sha,
                    size=size,
                    template_version=versions.template_version,
                    persona_version=versions.persona_version,
                    profile_version=versions.profile_version,
                    dataset_version=versions.dataset_version,
                    holdout_cutoff=cutoff,
                    eval=metrics,
                    enabled=False,
                    active=False,
                    gate_passed=None,
                )
                session.add(row)
                session.flush()
                created += 1
            views.append(_view(row))
    return RegistrationResult(chosen_run, views, created, unchanged)


def list_models(db: Database) -> list[ModelView]:
    with db.session() as session:
        rows = session.scalars(
            select(ModelRegistryEntry).order_by(
                ModelRegistryEntry.created_at.desc(), ModelRegistryEntry.id
            )
        ).all()
        return [_view(row) for row in rows]


def get_model(db: Database, model_id: str) -> ModelView:
    with db.session() as session:
        row = session.get(ModelRegistryEntry, model_id)
        if row is None:
            raise RegistryError(f"no model {model_id!r} in the registry; see `twin model list`")
        return _view(row)


def find_model(db: Database, reference: str) -> ModelView:
    """A model from its id, the start of its id, or its run id (when that names one model)."""
    models = list_models(db)
    exact = [m for m in models if m.id == reference]
    found = exact or [m for m in models if m.id.startswith(reference) or m.run_id == reference]
    if not found:
        raise RegistryError(f"no model {reference!r} in the registry; see `twin model list`")
    if len(found) > 1:
        names = ", ".join(m.id for m in found[:6])
        raise RegistryError(f"{reference!r} names {len(found)} models ({names}); use a full id")
    return found[0]


def rollback_model(db: Database, reference: str) -> tuple[ModelView | None, ModelView]:
    """Make an older registered model the one in use (``twin rollback style-model``, R-OPS-010).

    Of each file kind (a GGUF, a LoRA adapter) one model is active; the chosen one becomes the
    active one of its kind and is enabled.  ``gate_passed`` stays as it was: going back to a model
    never turns a model that did not pass the release gate into one that did (R-SRV-005).
    Returns ``(the model that was active, the model that is)``.
    """
    chosen = find_model(db, reference)
    with db.transaction() as session:
        rows = list(
            session.scalars(
                select(ModelRegistryEntry).where(ModelRegistryEntry.kind == chosen.kind)
            )
        )
        previous = next((_view(row) for row in rows if row.active), None)
        for row in rows:
            row.active = row.id == chosen.id
            if row.id == chosen.id:
                row.enabled = True
    return previous, get_model(db, chosen.id)


def resolve_model_path(models_dir: Path, view: ModelView) -> Path:
    """The file of a registered model (stored relative to the models directory when inside it)."""
    stored = Path(view.path)
    return stored if stored.is_absolute() else models_dir / stored


def _row(session: Session, model_id: str) -> ModelRegistryEntry:
    row = session.get(ModelRegistryEntry, model_id)
    if row is None:
        raise RegistryError(f"no model {model_id!r} in the registry; see `twin model list`")
    return row


def _logged(document: Mapping[str, Any], entry: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``eval`` document with ``entry`` appended to its activation log."""
    updated = dict(document)
    if entry is not None:
        history = [*list(updated.get(EVAL_ACTIVATION_LOG) or []), dict(entry)]
        updated[EVAL_ACTIVATION_LOG] = history[-ACTIVATION_LOG_KEEP:]
    return updated


def activate_row(
    db: Database,
    model_id: str,
    *,
    gate_passed: bool,
    entry: Mapping[str, Any],
    eval_updates: Mapping[str, Any] | None = None,
) -> tuple[ModelView | None, ModelView]:
    """Make a model the one in use for its kind of file (``twin model activate``).

    In one transaction: every other model of the same kind stops being active, this one becomes
    enabled and active, ``gate_passed`` is set to the verdict (``False`` for a forced activation:
    it never counts as passed), and ``entry`` is appended to the activation log.  Returns
    ``(the model that was active before, this model)``.
    """
    with db.transaction() as session:
        chosen = _row(session, model_id)
        previous: ModelView | None = None
        for row in session.scalars(
            select(ModelRegistryEntry).where(ModelRegistryEntry.kind == chosen.kind)
        ):
            if row.active and row.id != chosen.id:
                previous = _view(row)
            row.active = row.id == chosen.id
        chosen.enabled = True
        chosen.gate_passed = gate_passed
        chosen.eval = _logged({**chosen.eval, **dict(eval_updates or {})}, entry)
        session.flush()
        return previous, _view(chosen)


def deactivate_row(db: Database, model_id: str, *, entry: Mapping[str, Any]) -> ModelView:
    """Stop using a model (``twin model disable``): not active, not enabled; the verdict stays."""
    with db.transaction() as session:
        row = _row(session, model_id)
        row.active = False
        row.enabled = False
        row.eval = _logged(row.eval, entry)
        session.flush()
        return _view(row)


def record_eval(
    db: Database, model_id: str, updates: Mapping[str, Any], *, bump_state: bool | None = False
) -> ModelView:
    """Merge ``updates`` into the ``eval`` document of a model (a tokenizer check, a verdict).

    The running application records the comparison it makes itself and must not wake itself up
    with it (``bump_state=False``); a command that changes what the application should do passes
    ``None`` to follow the write policy of the command (a LIGHT command bumps ``state_version``).
    """
    with db.transaction(bump_state=bump_state) as session:
        row = _row(session, model_id)
        row.eval = {**row.eval, **dict(updates)}
        session.flush()
        return _view(row)
