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
* writes one row per usable file: a GGUF per quantisation and the LoRA adapter (``quant`` is
  ``lora``).  Registering the same files again changes nothing; a different file under the same
  run and quantisation is refused.

Enabling, activating and the release gate belong to round 14; the flags exist and start false.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from sqlalchemy import select

from twin.storage.db import Database
from twin.storage.training_models import ModelRegistryEntry, TrainingRun
from twin.training.layout import ARTIFACT_MANIFEST
from twin.training.profiles import PROFILES

SCHEMA: Final = 1
HASH_BLOCK = 1 << 20
LOCKED_FIELDS: Final = ("template_version", "persona_version", "profile_version", "dataset_version")


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
    metrics = dict(manifest.get("metrics") or {})
    metrics["adapter_kind"] = manifest.get("adapter_kind")

    created = unchanged = 0
    views: list[ModelView] = []
    with db.transaction() as session:
        run = session.get(TrainingRun, chosen_run)
        if run is not None and run.dataset_version != versions.dataset_version:
            raise RegistryError(
                f"the run {chosen_run} used dataset {run.dataset_version}, "
                f"the manifest says {versions.dataset_version}"
            )
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
