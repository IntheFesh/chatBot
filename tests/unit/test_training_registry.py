"""R-SRV-001: the model registry (register, list, show, locked versions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from tests.support.training_artifacts import QUANTS, build_artifact_dir
from tests.support.training_data import write_synthetic_dataset
from twin.cli import app
from twin.services import Services
from twin.storage.training_models import ModelRegistryEntry
from twin.training.registry import (
    LockedVersions,
    RegistryError,
    get_model,
    list_models,
    load_manifest,
    register_artifacts,
    resolve_model_path,
)
from twin.training.runs import RunStore, ensure_dataset_version

runner = CliRunner()


def test_registering_an_artifact_directory_locks_the_versions_of_the_manifest(
    services: Services, tmp_path: Path
) -> None:
    directory = tmp_path / "models" / "r-test-1"
    build_artifact_dir(directory)
    result = register_artifacts(services.db, directory, models_dir=tmp_path / "models")
    assert (result.run_id, result.created, result.unchanged) == ("r-test-1", 4, 0)
    assert [m.quant for m in result.models] == [*QUANTS, "lora"]
    assert [m.kind for m in result.models] == ["gguf", "gguf", "gguf", "adapter"]
    first = result.models[0]
    assert first.id == "r-test-1-Q4_K_M" and first.profile == "5090-8b"
    assert first.base_model == "Qwen/Qwen3-8B"
    assert first.versions == LockedVersions(
        template_version="qwen3_nothink@llamafactory-0.9.5",
        persona_version="v3",
        profile_version="01TESTPROFILEVERSION00000",
        dataset_version="ds-test-01",
    )
    assert first.path == "r-test-1/gguf/5090-8b-Q4_K_M.gguf"  # relative to the models directory
    assert len(first.sha256) == 64 and first.size > 0
    assert (first.enabled, first.active, first.gate_passed) == (False, False, None)
    assert first.eval["best_val_loss"] == 1.25 and first.eval["adapter_kind"] == "sft"
    assert resolve_model_path(tmp_path / "models", first).is_file()


def test_registering_again_changes_nothing_and_keeps_the_flags(
    services: Services, tmp_path: Path
) -> None:
    directory = tmp_path / "models" / "r-test-1"
    build_artifact_dir(directory)
    register_artifacts(services.db, directory, models_dir=tmp_path / "models")
    with services.db.transaction() as session:
        row = session.get(ModelRegistryEntry, "r-test-1-Q4_K_M")
        assert row is not None
        row.enabled = True
    again = register_artifacts(services.db, directory, models_dir=tmp_path / "models")
    assert (again.created, again.unchanged) == (0, 4)
    assert get_model(services.db, "r-test-1-Q4_K_M").enabled is True
    assert len(list_models(services.db)) == 4


def test_a_file_that_changed_since_the_export_is_refused(
    services: Services, tmp_path: Path
) -> None:
    directory = tmp_path / "r1"
    build_artifact_dir(directory)
    path = directory / "gguf" / "5090-8b-Q5_K_M.gguf"
    original_size = path.stat().st_size
    path.write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="different size"):
        register_artifacts(services.db, directory)
    assert list_models(services.db) == []  # nothing was registered
    path.write_bytes(b"x" * original_size)  # same size, other bytes
    with pytest.raises(RegistryError, match="does not match its sha256"):
        register_artifacts(services.db, directory)
    path.unlink()
    with pytest.raises(RegistryError, match="missing"):
        register_artifacts(services.db, directory)


def test_a_manifest_without_the_versions_cannot_lock_a_model(
    services: Services, tmp_path: Path
) -> None:
    directory = tmp_path / "r1"
    manifest = build_artifact_dir(directory)
    for missing in ("template_version", "persona_version", "profile_version", "dataset_version"):
        broken = {**manifest, missing: ""}
        (directory / "manifest.json").write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(RegistryError, match=f"does not name {missing}"):
            register_artifacts(services.db, directory)


def test_the_manifest_is_checked_for_its_schema_profile_run_and_paths(
    services: Services, tmp_path: Path
) -> None:
    directory = tmp_path / "r1"
    manifest = build_artifact_dir(directory, run_id=None)
    with pytest.raises(RegistryError, match="no run id"):
        register_artifacts(services.db, directory)
    assert register_artifacts(services.db, directory, run_id="r-chosen").run_id == "r-chosen"

    def write(change: dict[str, object]) -> None:
        (directory / "manifest.json").write_text(json.dumps({**manifest, **change}), "utf-8")

    write({"profile": "tpu-1b", "run_id": "x"})
    with pytest.raises(RegistryError, match="unknown profile"):
        register_artifacts(services.db, directory)
    write({"schema": 2})
    with pytest.raises(RegistryError, match="unsupported schema"):
        load_manifest(directory)
    write({"files": [{"path": "../outside", "sha256": "0" * 64, "size": 1}]})
    with pytest.raises(RegistryError, match="leaves the artifact directory"):
        register_artifacts(services.db, directory)
    write({"files": []})
    with pytest.raises(RegistryError, match="lists no files"):
        register_artifacts(services.db, directory)
    (directory / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RegistryError, match="not valid JSON"):
        load_manifest(directory)
    (directory / "manifest.json").unlink()
    with pytest.raises(RegistryError, match="not an artifact directory"):
        load_manifest(directory)


def test_a_different_file_cannot_take_the_place_of_a_registered_one(
    services: Services, tmp_path: Path
) -> None:
    first = tmp_path / "a"
    build_artifact_dir(first)
    register_artifacts(services.db, first)
    second = tmp_path / "b"
    build_artifact_dir(second, sizes=65)  # same run id and quantisations, other bytes
    with pytest.raises(RegistryError, match="different sha256"):
        register_artifacts(services.db, second)
    assert (
        get_model(services.db, "r-test-1-Q4_K_M").size
        == (first / "gguf" / "5090-8b-Q4_K_M.gguf").stat().st_size
    )


def test_a_known_run_must_have_used_the_dataset_of_the_manifest(
    services: Services, tmp_path: Path
) -> None:
    dataset = write_synthetic_dataset(tmp_path / "ds", version="ds-test-01")
    ensure_dataset_version(services.db, dataset, "ds")
    RunStore(services.db).create(
        run_id="r-test-1",
        profile="5090-8b",
        dataset_version="ds-test-01",
        bundle_sha256="a" * 64,
        bundle_size=1,
        parameters={},
    )
    directory = tmp_path / "r1"
    build_artifact_dir(directory, dataset_version="ds-other")
    with pytest.raises(RegistryError, match="used dataset ds-test-01"):
        register_artifacts(services.db, directory)
    ok = tmp_path / "r2"
    build_artifact_dir(ok)
    assert register_artifacts(services.db, ok).created == 4


def test_an_unknown_model_id_says_where_to_look(services: Services) -> None:
    with pytest.raises(RegistryError, match="twin model list"):
        get_model(services.db, "nope")


def test_models_outside_the_models_directory_keep_their_absolute_path(
    services: Services, tmp_path: Path
) -> None:
    directory = tmp_path / "elsewhere" / "r1"
    build_artifact_dir(directory)
    result = register_artifacts(services.db, directory, models_dir=tmp_path / "models")
    stored = Path(result.models[0].path)
    assert stored.is_absolute() and stored.is_file()
    assert resolve_model_path(tmp_path / "models", result.models[0]) == stored
    with services.db.session() as session:
        assert len(session.scalars(select(ModelRegistryEntry)).all()) == 4


# ------------------------------------------------------------------------------- CLI


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


def test_the_model_commands_register_list_and_show(data_dir: Path) -> None:
    empty = runner.invoke(app, ["model", "list"])
    assert empty.exit_code == 0 and "no models registered yet" in empty.output
    directory = data_dir / "models" / "r-test-1"
    build_artifact_dir(directory)
    done = runner.invoke(app, ["model", "register", str(directory)])
    assert done.exit_code == 0, done.output
    assert "4 registered, 0 already registered" in done.output
    assert "qwen3_nothink@llamafactory-0.9.5" in done.output and "ds-test-01" in done.output
    again = runner.invoke(app, ["model", "register", str(directory)])
    assert "0 registered, 4 already registered" in again.output
    listing = runner.invoke(app, ["model", "list"])
    assert (
        listing.exit_code == 0 and "r-test-1-Q4_K_M" in listing.output and "lora" in listing.output
    )
    shown = runner.invoke(app, ["model", "show", "r-test-1-lora"])
    assert shown.exit_code == 0, shown.output
    for text in (
        "persona card version",
        "v3",
        "01TESTPROFILEVERSION00000",
        "adapter",
        "best_val_loss",
    ):
        assert text in shown.output
    missing = runner.invoke(app, ["model", "show", "r-nope"])
    assert missing.exit_code == 1 and "no model 'r-nope'" in missing.output


def test_the_register_command_reports_a_damaged_directory(data_dir: Path) -> None:
    directory = data_dir / "models" / "r1"
    build_artifact_dir(directory)
    (directory / "gguf" / "5090-8b-Q8_0.gguf").write_bytes(b"short")
    done = runner.invoke(app, ["model", "register", str(directory)])
    assert done.exit_code == 1 and "Q8_0.gguf" in done.output
    other = runner.invoke(app, ["model", "register", str(data_dir)])
    assert other.exit_code == 1 and "not an artifact directory" in other.output
