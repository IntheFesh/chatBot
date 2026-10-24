"""``twin import inspect``: structure only, never a value (R-IMP-014)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.fixtures.synth_export import SynthExport, SynthOptions, build_export
from twin.ingest.inspect import (
    OTHER_KEY,
    InspectResult,
    Structure,
    inspect_export,
    render_inspect,
    write_inspect_report,
)
from twin.ingest.layout import ExportLayoutError

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def leaked(export: SynthExport, text: str) -> list[str]:
    """Everything personal in ``text``: message texts, names, ids, file names of media."""
    found = [s for s in export.texts if len(s) >= 6 and s in text]  # short ones occur in prose
    names = [export.target_display_name, export.target_username, "wxid_" + "me0000"]
    names += export.other_usernames + ([export.group_username] if export.group_username else [])
    found += [n for n in names if n and n in text]
    found += [md5 for md5 in export.image_md5s + list(export.stickers) if md5 in text]
    found += [p for p in export.image_paths if p in text]
    found += [p for p in ("周末群聊", "小满") if p in text]
    return found


@pytest.fixture
def export(tmp_path: Path) -> SynthExport:
    return build_export(
        tmp_path / "e", SynthOptions(target_messages=400, integrity="json", other_messages=30)
    )


def test_the_report_has_structure_and_no_values(export: SynthExport) -> None:
    text = render_inspect(inspect_export(export.root), NOW)
    assert leaked(export, text) == []
    for expected in (
        "report.json",
        "conversations/",
        "`$.missingMedia[].kind`",
        "`$.schemaVersion`",
        "`$.messages[].renderType`",
        "`$.messages[].offlineMedia[].path`",
        "`$.messages[].offlineMedia[].kind`",
        "`$.conversation.displayName`",
        "manifest.json",
    ):
        assert expected in text, expected
    assert "string×" in text and "int×" in text and "bool×" in text and "null×" in text


def test_enumeration_fields_list_their_values_with_counts(export: SynthExport) -> None:
    result = inspect_export(export.root)
    values = result.messages.values["$.messages[].renderType"]
    assert values["text"] > 0 and values["emoji"] > 0 and values["holographic"] > 0
    assert sum(values.values()) == result.sampled
    assert set(result.messages.values["$.messages[].isSent"]) == {"true", "false"}
    assert result.messages.values["$.messages[].offlineMedia[].kind"]["emoji"] > 0
    assert set(result.header.values["$.schemaVersion"]) == {"1"}
    text = render_inspect(result, NOW)
    assert "| `$.messages[].renderType` | `text` |" in text


def test_free_text_fields_never_get_value_counts(export: SynthExport) -> None:
    result = inspect_export(export.root)
    for path in result.messages.values:
        leaf = path.rsplit(".", 1)[-1].removesuffix("[]")
        assert leaf in {
            "renderType",
            "type",
            "kind",
            "voipType",
            "linkType",
            "linkStyle",
            "quoteType",
            "paySubType",
            "transferStatus",
            "voiceTranscriptStatus",
            "schemaVersion",
            "isSent",
            "isGroup",
            "messageTypes",
        }, path
    assert "$.messages[].content" not in result.messages.values
    assert "$.messages[].emojiMd5" not in result.messages.values


def test_conversation_folders_are_shown_masked(export: SynthExport) -> None:
    result = inspect_export(export.root)
    folders = [line for line in result.tree if "[direct]" in line or "[group]" in line]
    assert len(folders) == 4
    assert all(line.split("/", 1)[1].split(" ", 1)[0].count("_") == 2 for line in folders)
    assert sum("[group]" in line for line in folders) == 1
    assert all("messages.json (" in line and "meta.json (" in line for line in folders)
    assert result.conversations == 4 and result.groups == 1
    assert any(line.startswith(("media/emojis:", "media/images:")) for line in result.tree)


def test_the_sample_limit_bounds_the_messages_looked_at(export: SynthExport) -> None:
    limited = inspect_export(export.root, sample=10)
    assert limited.sampled == 40  # ten per conversation file
    everything = inspect_export(export.root, sample=0)
    assert everything.sampled == 400 + 3 * 30


def test_keys_that_could_hold_personal_data_are_not_printed(tmp_path: Path) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=5, integrity="json"))
    secret_key = "wxid_" + "private12345"
    path_key = "conversations/小满/messages.json"
    meta_path = export.target_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta[secret_key] = 1
    meta[path_key] = "x"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    manifest = export.root / "_integrity" / "manifest.json"
    manifest.write_text(
        json.dumps({path_key: "ab" * 32, "ordinaryKey": {secret_key: 1}}, ensure_ascii=False),
        encoding="utf-8",
    )
    text = render_inspect(inspect_export(export.root), NOW)
    assert secret_key not in text and path_key not in text and "小满" not in text
    assert OTHER_KEY in text and "ordinaryKey" in text


def test_values_of_enumeration_fields_must_look_like_plain_words(tmp_path: Path) -> None:
    export = build_export(
        tmp_path / "e", SynthOptions(target_messages=3, other_conversations=0, include_group=False)
    )
    document = json.loads(export.messages_path.read_text(encoding="utf-8"))
    document["messages"][0]["renderType"] = "一段不应该出现的很长的文字" * 5
    document["messages"][1]["renderType"] = "wxid_" + "oddvalue99"
    document["messages"][2]["type"] = 3.5
    export.messages_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    result = inspect_export(export.root)
    text = render_inspect(result, NOW)
    assert "一段不应该" not in text and "oddvalue" not in text
    assert result.messages.values["$.messages[].renderType"]["<other>"] == 2
    assert result.messages.values["$.messages[].type"]["3.5"] == 1


def test_an_export_without_integrity_and_a_folder_of_text_checksums(tmp_path: Path) -> None:
    plain = build_export(tmp_path / "plain", SynthOptions(target_messages=3))
    assert "没有 _integrity 文件夹" in render_inspect(inspect_export(plain.root), NOW)
    sums = build_export(tmp_path / "sums", SynthOptions(target_messages=3, integrity="sha256sum"))
    result = inspect_export(sums.root)
    text = render_inspect(result, NOW)
    assert "SHA256SUMS.txt" in text and "of the form '<hex digest> <path>'" in text
    assert "conversations/" not in text.split("## _integrity/")[1].split("## ")[0]  # no paths


def test_unreadable_files_are_noted_not_fatal(tmp_path: Path) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=20, other_conversations=1))
    export.messages_path.write_bytes(export.messages_path.read_bytes()[:300])
    (export.root / "report.json").write_text("{broken", encoding="utf-8")
    (export.target_dir / "meta.json").write_text("{broken", encoding="utf-8")
    result = inspect_export(export.root)
    text = render_inspect(result, NOW)
    assert "report.json is not valid JSON" in text and "a meta.json is not valid JSON" in text
    assert "could not be read to the end" in text


def test_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ExportLayoutError, match="not found"):
        inspect_export(tmp_path / "nowhere")


def test_other_top_level_entries_are_not_named(tmp_path: Path) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=3))
    (export.root / "小满的备份.zip").write_bytes(b"x")
    text = render_inspect(inspect_export(export.root), NOW)
    assert "小满的备份" not in text and "<other entry>" in text


def test_the_report_is_written_with_a_utc_time_stamp(tmp_path: Path) -> None:
    result = InspectResult()
    path = write_inspect_report(tmp_path / "reports", render_inspect(result, NOW), NOW)
    assert path.name == "inspect-20261009T120000Z.md" and path.read_text(encoding="utf-8")
    assert "（没有数据）" in path.read_text(encoding="utf-8")


def test_structure_records_nested_types() -> None:
    structure = Structure()
    structure.observe("$", {"a": [1, 2.5, "x", None, True], "b": {"c": {}}})
    assert set(structure.types["$.a[]"]) == {"int", "float", "string", "null", "bool"}
    assert structure.types["$.b.c"]["object"] == 1
