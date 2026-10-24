"""Versioned prompt templates (R-PERS-005, R-OPS-010)."""

from __future__ import annotations

from pathlib import Path

import pytest

from twin.profile.prompt_templates import (
    ACTIVE_KEY,
    PERSONA_MAP,
    PERSONA_REDUCE,
    STICKER_CONTEXT,
    STICKER_TAG,
    TEMPLATE_DIR,
    TemplateError,
    TemplateStore,
    newest_file_template,
    split_template,
    template_files,
)
from twin.services import Services
from twin.storage.persona_models import PromptTemplate
from twin.storage.settings_store import get_setting

SYSTEM = "你是助手。"
USER = '请处理 $thing，格式 {"a": 1}。'


def write(directory: Path, name: str, version: int, system: str = SYSTEM, user: str = USER) -> Path:
    path = directory / f"{name}.v{version}.md"
    path.write_text(f"{system}\n\n=== user ===\n{user}\n", encoding="utf-8")
    return path


def test_the_shipped_templates_are_all_loadable_and_versioned() -> None:
    found = template_files()
    assert {PERSONA_MAP, PERSONA_REDUCE, STICKER_TAG, STICKER_CONTEXT} <= set(found)
    for name, versions in found.items():
        assert 1 in versions, name
        text = versions[1].read_text(encoding="utf-8")
        template = split_template(name, 1, text)
        assert template.system and template.user and template.ref == f"{name}@1"
    assert TEMPLATE_DIR.is_dir()


def test_a_template_renders_its_two_messages_and_keeps_json_braces(tmp_path: Path) -> None:
    template = split_template("t", 2, f"{SYSTEM}\n\n=== user ===\n{USER}\n")
    messages = template.render(thing="猫")
    assert messages == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": '请处理 猫，格式 {"a": 1}。'},
    ]
    with pytest.raises(TemplateError, match="needs the field 'thing'"):
        template.render()


@pytest.mark.parametrize(
    "text", ["only a system part", "sys\n=== user ===\n", "\n=== user ===\nonly user"]
)
def test_a_template_needs_both_parts(text: str) -> None:
    with pytest.raises(TemplateError, match="user"):
        split_template("t", 1, text)


def test_files_are_loaded_into_the_table_and_the_newest_version_is_in_force(
    services: Services, tmp_path: Path
) -> None:
    write(tmp_path, "demo", 1)
    store = TemplateStore(services.db, services.clock, tmp_path)
    assert store.active("demo").version == 1
    write(tmp_path, "demo", 2, user="第二版 $thing")
    fresh = TemplateStore(services.db, services.clock, tmp_path)
    assert fresh.active("demo").version == 2 and fresh.versions("demo") == [1, 2]
    assert fresh.names() == ["demo"]
    assert fresh.get("demo", 1).user.startswith("请处理")
    with services.db.session() as session:
        assert get_setting(session, ACTIVE_KEY + "demo") == 2
        rows = session.query(PromptTemplate).count()
    assert rows == 2


def test_a_rollback_points_at_the_older_version_until_a_newer_file_appears(
    services: Services, tmp_path: Path
) -> None:
    write(tmp_path, "demo", 1)
    write(tmp_path, "demo", 2)
    store = TemplateStore(services.db, services.clock, tmp_path)
    assert store.active("demo").version == 2
    store.rollback("demo", 1)
    assert TemplateStore(services.db, services.clock, tmp_path).active("demo").version == 1
    write(tmp_path, "demo", 3)
    assert TemplateStore(services.db, services.clock, tmp_path).active("demo").version == 3
    with pytest.raises(TemplateError, match=r"no template demo\.v9"):
        store.rollback("demo", 9)


def test_a_file_edited_without_a_new_version_is_refused(services: Services, tmp_path: Path) -> None:
    path = write(tmp_path, "demo", 1)
    TemplateStore(services.db, services.clock, tmp_path).active("demo")
    path.write_text(path.read_text(encoding="utf-8") + "改了一句\n", encoding="utf-8")
    with pytest.raises(TemplateError, match="add a new version"):
        TemplateStore(services.db, services.clock, tmp_path).active("demo")


def test_a_malformed_file_and_a_missing_template_are_reported(
    services: Services, tmp_path: Path
) -> None:
    (tmp_path / "bad.v1.md").write_text("no marker here", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "other.v0.md").write_text("ignored: versions start at 1", encoding="utf-8")
    with pytest.raises(TemplateError, match=r"bad\.v1"):
        TemplateStore(services.db, services.clock, tmp_path).sync()
    (tmp_path / "bad.v1.md").unlink()
    store = TemplateStore(services.db, services.clock, tmp_path)
    store.sync()
    with pytest.raises(TemplateError, match="no active version"):
        store.active_version("demo")
    with pytest.raises(TemplateError, match=r"no template demo\.v1"):
        store.get("demo", 1)


def test_the_shipped_templates_are_stored_when_first_used(services: Services) -> None:
    store = TemplateStore(services.db, services.clock)
    template = store.active(PERSONA_MAP)
    assert template.version == 1 and "evidence" in template.system
    assert set(store.names()) >= {PERSONA_MAP, PERSONA_REDUCE, STICKER_TAG, STICKER_CONTEXT}


def test_the_newest_file_can_be_read_without_touching_the_database(
    services: Services, tmp_path: Path
) -> None:
    """A READ command sizes prompts from the files: the store would write them to the table."""
    write(tmp_path, "sizing", 1, system="第一版。")
    write(tmp_path, "sizing", 3, system="第三版。")
    newest = newest_file_template("sizing", tmp_path)
    assert (newest.name, newest.version, newest.system) == ("sizing", 3, "第三版。")
    with pytest.raises(TemplateError, match="no template file"):
        newest_file_template("absent", tmp_path)
    shipped = newest_file_template(PERSONA_MAP)
    assert shipped.version == max(template_files()[PERSONA_MAP])
    store = TemplateStore(services.db, services.clock)
    with services.db.session() as session:
        assert get_setting(session, ACTIVE_KEY + PERSONA_MAP) is None  # nothing was stored
    assert store.names() == []
