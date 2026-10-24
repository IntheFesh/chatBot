"""R-SRV-001: the model registry (register, list, show, locked versions)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from tests.support.training_artifacts import QUANTS, build_artifact_dir
from tests.support.training_data import write_synthetic_dataset
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.services import Services, build_services
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


# ------------------------------------------------------------------- adapter stacks (R17-49d)


def test_the_registry_keeps_the_ordered_adapter_stack_with_every_layers_hash(
    services: Services, tmp_path: Path
) -> None:
    """R17-49d (D-653): ``sft`` then ``dpo``, each with its sha256, rank and alpha, in every row."""
    directory = tmp_path / "models" / "r-test-1"
    reference = {"mode": "layered", "reference": "sft", "fallback": False}
    choice = {"keep": "dpo", "reason": "more style metrics within +-30 % of hers: 5 against 4"}
    manifest = build_artifact_dir(
        directory, stack="sft+dpo", dpo_reference=reference, stack_choice=choice
    )
    result = register_artifacts(services.db, directory, models_dir=tmp_path / "models")
    assert result.created == 4
    listed = {entry["path"]: entry for entry in manifest["files"]}
    for model in result.models:  # the GGUF rows were merged from the stack: they say so too
        stack = model.adapter_stack
        assert stack is not None and not stack.legacy
        assert (stack.kind, stack.mode, stack.choice) == ("sft+dpo", "layered", "dpo")
        assert stack.names == ("sft", "dpo")
        for layer in stack.layers:
            assert layer.sha256 == listed[layer.path]["sha256"] and layer.rank == 4
        assert stack.served.composed and stack.served.rank == 8
        assert model.eval["dpo_reference"] == reference and model.eval["stack_choice"] == choice
    adapter = next(m for m in result.models if m.kind == "adapter")
    assert adapter.sha256 == listed["adapter/adapter_model.safetensors"]["sha256"]
    assert adapter.adapter_stack is not None
    assert adapter.adapter_stack.served.sha256 == adapter.sha256  # the served file is this row
    again = get_model(services.db, adapter.id)
    assert again.adapter_stack == adapter.adapter_stack  # survives the database
    assert "sft (r4," in again.adapter_stack.describe() and "composed, r8" in (
        again.adapter_stack.describe()
    )


def test_a_manifest_from_before_the_stacks_is_read_as_the_one_adapter_it_was(
    services: Services, tmp_path: Path
) -> None:
    """R17-49d (D-653): old registrations and old artifact directories keep working."""
    directory = tmp_path / "models" / "r-test-1"
    build_artifact_dir(directory)  # no adapter_stack.json: an export of the earlier scripts
    result = register_artifacts(services.db, directory, models_dir=tmp_path / "models")
    adapter = next(m for m in result.models if m.kind == "adapter")
    stack = adapter.adapter_stack
    assert stack is not None and stack.legacy and stack.kind == "sft" and stack.names == ("sft",)
    assert stack.layers[0].sha256 == adapter.sha256 and not stack.served.composed
    gguf = next(m for m in result.models if m.kind == "gguf")
    assert gguf.adapter_stack is not None  # the manifest names the hash of the one adapter file
    assert gguf.adapter_stack.layers[0].sha256 == adapter.sha256
    assert "registered before adapter stacks existed" in stack.describe()
    with services.db.transaction() as session:  # a GGUF row of the earlier registry
        row = session.get(ModelRegistryEntry, gguf.id)
        assert row is not None
        row.eval = {"adapter_kind": "sft"}
    old_gguf = get_model(services.db, gguf.id).adapter_stack
    assert old_gguf is not None and old_gguf.layers[0].sha256 is None  # not known for this row
    # a row written by the earlier registry has only adapter_kind in its eval document
    with services.db.transaction() as session:
        row = session.get(ModelRegistryEntry, adapter.id)
        assert row is not None
        row.eval = {"adapter_kind": "dpo", "best_val_loss": 1.0}
    old = get_model(services.db, adapter.id)
    assert old.adapter_stack is not None and old.adapter_stack.kind == "dpo"
    assert old.adapter_stack.mode == "continued"  # the DPO adapter of those runs continued SFT
    with services.db.transaction() as session:
        row = session.get(ModelRegistryEntry, adapter.id)
        assert row is not None
        row.eval = {}
    assert get_model(services.db, adapter.id).adapter_stack is None  # nothing is known


def test_a_stack_that_does_not_match_the_files_it_lists_is_refused(
    services: Services, tmp_path: Path
) -> None:
    """R17-49d (D-653): each layer is checked against its own hash before anything is written."""
    directory = tmp_path / "r1"
    manifest = build_artifact_dir(directory, stack="sft+dpo")

    def write(stack: dict[str, object]) -> None:
        (directory / "manifest.json").write_text(
            json.dumps({**manifest, "adapter_stack": stack}), encoding="utf-8"
        )

    original = manifest["adapter_stack"]
    layers = original["layers"]
    swapped = [{**layers[0], "sha256": layers[1]["sha256"]}, layers[1]]
    write({**original, "layers": swapped})
    with pytest.raises(RegistryError, match="does not match its sha256 in the adapter stack"):
        register_artifacts(services.db, directory)
    write(
        {
            **original,
            "layers": [
                {**layers[0], "path": "adapters/ghost/adapter_model.safetensors"},
                layers[1],
            ],
        }
    )
    with pytest.raises(RegistryError, match="not listed"):
        register_artifacts(services.db, directory)
    write({**original, "layers": list(reversed(layers))})
    with pytest.raises(RegistryError, match="has the layers sft, dpo"):
        register_artifacts(services.db, directory)
    write({**original, "served": {**original["served"], "composed": False}})
    with pytest.raises(RegistryError, match="only a stack of more than one layer is composed"):
        register_artifacts(services.db, directory)
    write({**original, "served": {**original["served"], "rank": 9}})
    with pytest.raises(RegistryError, match="not the sum of the ranks"):
        register_artifacts(services.db, directory)
    write({**original, "kind": "all"})
    with pytest.raises(RegistryError, match="unknown kind"):
        register_artifacts(services.db, directory)
    write({"layers": [], "served": {}})
    with pytest.raises(RegistryError, match="lists no layers"):
        register_artifacts(services.db, directory)
    (directory / "manifest.json").write_text(
        json.dumps({**manifest, "adapter_stack": "sft"}), encoding="utf-8"
    )
    with pytest.raises(RegistryError, match="not an object"):
        register_artifacts(services.db, directory)
    assert list_models(services.db) == []  # nothing was registered by any of them
    write(original)
    assert register_artifacts(services.db, directory).created == 4


def test_a_damaged_stack_in_a_row_reads_as_unknown_and_a_nameless_layer_is_refused(
    services: Services, tmp_path: Path
) -> None:
    """R17-49d (D-653): a row's stack is read, never trusted blindly and never a crash."""
    from twin.training.adapter_stack import StackError, stack_from_json

    directory = tmp_path / "r1"
    build_artifact_dir(directory, stack="sft")
    adapter = next(
        m for m in register_artifacts(services.db, directory).models if m.kind == "adapter"
    )
    assert adapter.adapter_stack is not None and adapter.adapter_stack.kind == "sft"
    with services.db.transaction() as session:
        row = session.get(ModelRegistryEntry, adapter.id)
        assert row is not None
        row.eval = {"adapter_stack": {"layers": [{"path": "x"}], "served": {}}}
    assert get_model(services.db, adapter.id).adapter_stack is None
    nameless = {
        "kind": "sft",
        "mode": "layered",
        "choice": "sft",
        "layers": [{"path": "adapter/adapter_model.safetensors"}],
        "served": {"path": "adapter/adapter_model.safetensors"},
    }
    with pytest.raises(StackError, match="no name or path"):
        stack_from_json(nameless)


def test_the_one_continued_adapter_of_the_4bit_profile_is_a_stack_of_one(
    services: Services, tmp_path: Path
) -> None:
    directory = tmp_path / "r1"
    fallback = {
        "mode": "continued",
        "reference": "base",
        "fallback": True,
        "fallback_reason": "oom",
    }
    build_artifact_dir(directory, profile="5090-14b", stack="dpo", dpo_reference=fallback)
    result = register_artifacts(services.db, directory)
    stack = result.models[0].adapter_stack
    assert stack is not None and (stack.kind, stack.mode) == ("dpo", "continued")
    assert stack.names == ("dpo",) and not stack.served.composed
    assert result.models[0].eval["dpo_reference"]["fallback"] is True


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


def test_model_show_tells_how_the_model_was_made(data_dir: Path) -> None:
    """R17-49d (D-651, D-653, D-655): the stack, the reference model and the stack that was kept."""
    directory = data_dir / "models" / "r-test-1"
    reference = {
        "mode": "continued",
        "reference": "base",
        "fallback": True,
        "fallback_reason": "out of memory",
    }
    choice = {"keep": "dpo", "reason": "more style metrics within +-30 % of hers"}
    build_artifact_dir(
        directory,
        profile="5090-14b",
        stack="dpo",
        dpo_reference=reference,
        stack_choice=choice,
    )
    assert runner.invoke(app, ["model", "register", str(directory)]).exit_code == 0
    shown = runner.invoke(app, ["model", "show", "r-test-1-lora"])
    assert shown.exit_code == 0, shown.output
    flat = " ".join(shown.output.split())
    assert "adapter stack" in flat and "dpo: dpo (r4," in flat
    assert "dpo reference model" in flat and "the BASE (adapter switched off), a fall-back" in flat
    assert "stack kept by the evaluation" in flat and "dpo: more style metrics" in flat
    assert "eval: adapter_stack" not in flat  # shown above, not as a raw document
    plain = runner.invoke(app, ["model", "show", "r-test-1-Q4_K_M"])
    assert "adapter stack" in " ".join(plain.output.split())


def test_the_register_command_reports_a_damaged_directory(data_dir: Path) -> None:
    directory = data_dir / "models" / "r1"
    build_artifact_dir(directory)
    (directory / "gguf" / "5090-8b-Q8_0.gguf").write_bytes(b"short")
    done = runner.invoke(app, ["model", "register", str(directory)])
    assert done.exit_code == 1 and "Q8_0.gguf" in done.output
    other = runner.invoke(app, ["model", "register", str(data_dir)])
    assert other.exit_code == 1 and "not an artifact directory" in other.output


# ------------------------------------------- the cut-off the model was trained with (R18-C3)

TRAINED_CUTOFF = "2026-08-14T02:46:19+00:00"


def test_r18_c3_registering_locks_the_holdout_cutoff_of_the_training_set(
    services: Services, tmp_path: Path
) -> None:
    """R18-C3 ③ (user decision, D-885): ``model_registry`` locks the hold-out cut-off of the
    dataset the model was trained on (the training package names it; the manifest copies it)."""
    directory = tmp_path / "models" / "r-test-1"
    manifest = build_artifact_dir(directory, versions={"holdout_cutoff": TRAINED_CUTOFF})
    assert manifest["holdout_cutoff"] == TRAINED_CUTOFF
    result = register_artifacts(services.db, directory, models_dir=tmp_path / "models")
    moment = datetime.fromisoformat(TRAINED_CUTOFF)
    assert {model.holdout_cutoff for model in result.models} == {moment}
    assert get_model(services.db, "r-test-1-lora").holdout_cutoff == moment


def test_r18_c3_the_local_dataset_record_gives_the_cutoff_a_manifest_does_not_name(
    services: Services, tmp_path: Path
) -> None:
    """R18-C3 ③ (D-885): an older manifest names no cut-off - the dataset version recorded at the
    export does; with neither the model is refused, and a manifest that contradicts the record
    too.  Registering again fills the cut-off of a row registered before round 18."""
    dataset = write_synthetic_dataset(tmp_path / "ds", version="ds-test-01")
    ensure_dataset_version(services.db, dataset, "ds")
    recorded = datetime.fromisoformat(dataset.meta.holdout_cutoff)
    directory = tmp_path / "r1"
    build_artifact_dir(directory, versions={"holdout_cutoff": ""})
    result = register_artifacts(services.db, directory)
    assert {model.holdout_cutoff for model in result.models} == {recorded}
    with services.db.transaction() as session:
        for row in session.scalars(select(ModelRegistryEntry)):
            row.holdout_cutoff = None  # what the rows of an older registry hold
    assert get_model(services.db, "r-test-1-Q4_K_M").holdout_cutoff is None
    again = register_artifacts(services.db, directory)
    assert again.created == 0 and {m.holdout_cutoff for m in again.models} == {recorded}
    unknown = tmp_path / "r2"
    build_artifact_dir(
        unknown, run_id="r-2", dataset_version="ds-x", versions={"holdout_cutoff": ""}
    )
    with pytest.raises(RegistryError, match="hold-out cut-off"):
        register_artifacts(services.db, unknown)
    other = tmp_path / "r3"
    build_artifact_dir(other, run_id="r-3", versions={"holdout_cutoff": TRAINED_CUTOFF})
    with pytest.raises(RegistryError, match="dataset ds-test-01 was exported with"):
        register_artifacts(services.db, other)
    garbled = tmp_path / "r4"
    build_artifact_dir(garbled, run_id="r-4", versions={"holdout_cutoff": "soon"})
    with pytest.raises(RegistryError, match="not a moment"):
        register_artifacts(services.db, garbled)


def test_r18_c3_model_show_names_the_locked_cutoff(data_dir: Path) -> None:
    """R18-C3 ③ (D-885): ``twin model show`` lists the cut-off with the other locked versions."""
    directory = data_dir / "models" / "r-test-1"
    build_artifact_dir(directory, versions={"holdout_cutoff": TRAINED_CUTOFF})
    assert runner.invoke(app, ["model", "register", str(directory)]).exit_code == 0
    shown = runner.invoke(app, ["model", "show", "r-test-1-Q5_K_M"])
    assert shown.exit_code == 0, shown.output
    assert "hold-out cut-off" in shown.output and "2026-08-14 02:46" in shown.output
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        with services.db.transaction() as session:
            row = session.get(ModelRegistryEntry, "r-test-1-Q5_K_M")
            assert row is not None
            row.holdout_cutoff = None  # a row of an older registry
    finally:
        services.close()
    older = runner.invoke(app, ["model", "show", "r-test-1-Q5_K_M"])
    assert older.exit_code == 0 and "unknown" in older.output
