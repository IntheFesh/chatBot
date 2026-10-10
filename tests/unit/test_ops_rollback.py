"""``twin rollback profile|persona|prompt-template|style-model`` and its audit log."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.support.synth_chat import ChatSpec, build_chat
from tests.support.training_artifacts import build_artifact_dir
from twin.cli import app
from twin.config.secrets import SecretStore
from twin.ops.rollback import (
    AUDIT_KEEP,
    RollbackError,
    RollbackResult,
    audit_entries,
    record_audit,
    rollback_persona,
    rollback_profile,
    rollback_prompt_template,
    rollback_style_model,
)
from twin.profile.builder import rebuild
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.profile.prompt_templates import TemplateStore
from twin.profile.store import VersionStore
from twin.services import CliContext, Services, set_cli_context
from twin.storage.state import read_state_version
from twin.training.registry import list_models, register_artifacts

runner = CliRunner()


def card(extra: str = "") -> str:
    return (
        compose.stats_block(["r"])
        + compose.empty_auto_block()
        + compose.manual_block(["- a"], ["- b"])
        + compose.dont_block()
        + extra
    )


@pytest.fixture
def versions(services: Services) -> Services:
    """Two versions of the profile in each scope, two persona cards, two models."""
    build_chat(services, ChatSpec(days=30))
    rebuild(services, "all")
    rebuild(services, "all", force=True)
    store = PersonaStore(services.db, services.clock)
    store.add_version("live", card(), reason="first")
    store.add_version("live", card("\n- later"), reason="second")
    for number in (1, 2):
        directory = services.paths.models_dir / f"r-test-{number}"
        build_artifact_dir(directory, run_id=f"r-test-{number}")
        register_artifacts(services.db, directory, models_dir=services.paths.models_dir)
    return services


# ----------------------------------------------------------------------------- profile


def test_the_profile_goes_back_to_an_older_version(versions: Services) -> None:
    store = VersionStore(versions.db, versions.clock)
    newer, older = (store.resolve("~0", "live"), store.resolve("~1", "live"))
    assert store.active_profile_id("live") == newer.id
    result = rollback_profile(versions, "~1", "live")
    assert result == RollbackResult("profile", newer.id, older.id, result.detail)
    assert result.detail.startswith("scope live")
    assert store.active_profile_id("live") == older.id
    again = rollback_profile(versions, older.id)  # a full id; the scope is read from the version
    assert again.previous == older.id and again.current == older.id


def test_a_profile_version_that_does_not_exist_is_refused(versions: Services) -> None:
    for bad in ("~9", "abc", "NOSUCHVERSION1234"):
        with pytest.raises(RollbackError):
            rollback_profile(versions, bad, "live")


# ----------------------------------------------------------------------------- persona


def test_the_persona_card_goes_back_to_an_older_version(versions: Services) -> None:
    store = PersonaStore(versions.db, versions.clock)
    first, second = store.resolve("v1", "live"), store.resolve("v2", "live")
    assert store.active_id("live") == second.id
    result = rollback_persona(versions, "v1")
    assert (result.subject, result.previous, result.current) == ("persona", second.id, first.id)
    assert "scope live" in result.detail and store.active_id("live") == first.id
    with pytest.raises(RollbackError):
        rollback_persona(versions, "v9")


# ----------------------------------------------------------------------------- templates


def write_template(directory: Path, name: str, version: int) -> None:
    (directory / f"{name}.v{version}.md").write_text(
        f"系统 {version}\n\n=== user ===\n请处理 $thing\n", encoding="utf-8"
    )


def test_a_prompt_template_goes_back_to_an_older_version(
    services: Services, tmp_path: Path
) -> None:
    write_template(tmp_path, "demo", 1)
    write_template(tmp_path, "demo", 2)
    result = rollback_prompt_template(services, "demo", 1, directory=tmp_path)
    assert (result.subject, result.previous, result.current) == (
        "prompt-template",
        "demo.v2",
        "demo.v1",
    )
    assert TemplateStore(services.db, services.clock, tmp_path).active("demo").version == 1
    with pytest.raises(RollbackError, match=r"demo\.v9"):
        rollback_prompt_template(services, "demo", 9, directory=tmp_path)
    with pytest.raises(RollbackError):
        rollback_prompt_template(services, "nosuchtemplate", 1, directory=tmp_path)


def test_the_shipped_templates_can_be_named(services: Services) -> None:
    result = rollback_prompt_template(services, "memory_extract", 1)
    assert result.current == "memory_extract.v1"


# ----------------------------------------------------------------------------- style model


def test_the_style_model_in_use_is_chosen_among_the_registered_ones(versions: Services) -> None:
    ids = [m.id for m in list_models(versions.db) if m.kind == "gguf" and m.id.endswith("Q4_K_M")]
    assert len(ids) == 2
    first = rollback_style_model(versions, ids[0])
    assert first.previous is None and first.current == ids[0]
    second = rollback_style_model(versions, ids[1])
    assert second.previous == ids[0] and second.current == ids[1]
    models = {m.id: m for m in list_models(versions.db)}
    assert models[ids[1]].active and models[ids[1]].enabled and not models[ids[0]].active
    assert "release gate" in second.detail  # it never passed the gate: said, not hidden
    assert models[ids[1]].gate_passed is None  # and not made to look as if it had
    adapters = [m for m in models.values() if m.kind == "adapter"]
    assert adapters and not any(m.active for m in adapters)  # the other kind is untouched
    with pytest.raises(RollbackError, match="no model"):
        rollback_style_model(versions, "no-such-model")
    with pytest.raises(RollbackError, match="names 8 models"):
        rollback_style_model(versions, "r-test")


# ----------------------------------------------------------------------------- audit


def test_every_rollback_leaves_an_audit_record(services: Services) -> None:
    assert audit_entries(services) == []
    record_audit(services, RollbackResult("persona", "A", "B", "scope live"))
    record_audit(services, RollbackResult("profile", None, "C"))
    entries = audit_entries(services)
    assert [e["subject"] for e in entries] == ["persona", "profile"]
    assert entries[0] == {
        "at": services.clock.now_utc().isoformat(),
        "subject": "persona",
        "from": "A",
        "to": "B",
        "detail": "scope live",
    }
    assert entries[1]["from"] is None


def test_the_audit_log_keeps_the_last_hundred(services: Services) -> None:
    for number in range(AUDIT_KEEP + 7):
        record_audit(services, RollbackResult("persona", str(number), str(number + 1)))
    entries = audit_entries(services)
    assert len(entries) == AUDIT_KEEP and entries[0]["from"] == "7"


# ----------------------------------------------------------------------------- the commands


@pytest.fixture
def cli(versions: Services, secret_store: SecretStore) -> Services:
    set_cli_context(CliContext(secrets=secret_store))
    return versions


def twin(services: Services, *args: str) -> tuple[int, str]:
    result = runner.invoke(
        app, ["--set", f"paths.data_dir={services.paths.data_dir}", "rollback", *args]
    )
    return result.exit_code, result.output


def test_the_commands_roll_back_and_write_the_audit_record(cli: Services) -> None:
    code, out = twin(cli, "profile", "~1", "--scope", "live")
    assert code == 0, out
    assert "profile:" in out and "-> " in out and "scope live" in out
    assert "rollback audit log" in out
    code, out = twin(cli, "persona", "v1")
    assert code == 0 and "persona:" in out
    ids = [m.id for m in list_models(cli.db) if m.kind == "gguf" and m.id.endswith("Q4_K_M")]
    code, out = twin(cli, "style-model", ids[0])
    assert code == 0 and f"(none) -> {ids[0]}" in out and "release gate" in out
    code, out = twin(cli, "prompt-template", "memory_extract", "1")
    assert code == 0 and "memory_extract.v1" in out
    assert [e["subject"] for e in audit_entries(cli)] == [
        "profile",
        "persona",
        "style-model",
        "prompt-template",
    ]


def test_the_commands_are_light_and_tell_a_running_application(cli: Services) -> None:
    with cli.db.session() as session:
        before = read_state_version(session)
    twin(cli, "persona", "v1")
    with cli.db.session() as session:
        assert read_state_version(session) > before  # the running application sees the change


def test_a_refused_rollback_is_an_error_and_leaves_no_record(cli: Services) -> None:
    for args in (
        ("profile", "~9"),
        ("persona", "v9"),
        ("style-model", "nope"),
        ("prompt-template", "memory_extract", "9"),
    ):
        code, out = twin(cli, *args)
        assert code != 0 and out.strip(), args
    assert audit_entries(cli) == []
    code, out = twin(cli, "persona", "v1", "--scope", "weekly")
    assert code == 2 and "live or pre_holdout" in out
