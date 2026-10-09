"""Export files: models, layout, paths, integrity and streaming (R-IMP-001..003)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import pytest

from tests.fixtures.synth_export import SynthOptions, build_export, message
from twin.ingest.integrity import (
    FileVerdict,
    IntegrityEntry,
    IntegrityIndex,
    IntegrityState,
    StreamHasher,
    check_stream,
    digest_algorithm,
    hash_file,
    load_integrity,
    normalize_key,
    parse_manifest_text,
    stream_algorithms,
    verify_file,
)
from twin.ingest.jsonio import first_value, iter_messages, skip_bom
from twin.ingest.layout import (
    ExportLayout,
    ExportLayoutError,
    is_group_conversation,
    parse_header,
    parse_meta,
    parse_report,
    truthy,
)
from twin.ingest.paths import (
    extended_length_path,
    long_path,
    mask_dir_name,
    nfc,
    resolve_in_root,
    split_relative,
)
from twin.ingest.schema import (
    KNOWN_MESSAGE_KEYS,
    ExportMessage,
    UnsupportedSchema,
    check_schema_version,
)

# ----------------------------------------------------------------------- schema


def test_generated_export_validates_against_the_models(tmp_path: Path) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=200))
    layout = ExportLayout(export.root)
    info = layout.report()
    assert info.export_id == "synthetic-export-0001"
    assert all(item.messageId for item in info.report.missingMedia)
    entry = layout.find(export.target_username)
    assert entry is not None and entry.meta.messageCount == 200
    header = parse_header(
        {
            "schemaVersion": 1,
            "exportedAt": 1,
            "conversation": {"username": "x", "displayName": "y"},
            "filters": {"messageTypes": []},
        }
    )
    assert header.conversation is not None and header.conversation.username == "x"
    with entry.messages_path.open("rb") as handle:
        seen = 0
        for raw in iter_messages(handle):
            model = ExportMessage.model_validate(raw)
            assert model.id and model.renderType
            seen += 1
    assert seen == 200


def test_unknown_fields_are_kept_by_the_models() -> None:
    raw = message(id="a", novelField=[1, 2], renderType="text", **{"from": "someone"})
    model = ExportMessage.model_validate(raw)
    assert model.model_extra == {"novelField": [1, 2]}
    assert model.from_ == "someone"
    assert "novelField" not in KNOWN_MESSAGE_KEYS and "from" in KNOWN_MESSAGE_KEYS


SPEC_R_IMP_002_FIELDS = [
    "id",
    "localId",
    "serverId",
    "createTime",
    "createTimeText",
    "sortSeq",
    "type",
    "renderType",
    "isSent",
    "senderUsername",
    "conversationUsername",
    "isGroup",
    "content",
    "title",
    "url",
    "from",
    "fromUsername",
    "linkType",
    "linkStyle",
    "objectId",
    "objectNonceId",
    "recordItem",
    "thumbUrl",
    "imageMd5",
    "imageFileId",
    "imageMd5Candidates",
    "imageFileIdCandidates",
    "imageUrl",
    "emojiMd5",
    "emojiUrl",
    "videoMd5",
    "videoThumbMd5",
    "videoFileId",
    "videoThumbFileId",
    "videoUrl",
    "videoThumbUrl",
    "voiceLength",
    "voiceTranscript",
    "voiceTranscriptStatus",
    "voiceTranscriptError",
    "voiceTranscriptLanguage",
    "voiceTranscriptModel",
    "quoteUsername",
    "quoteServerId",
    "quoteType",
    "quoteThumbUrl",
    "quoteVoiceLength",
    "quoteTitle",
    "quoteContent",
    "amount",
    "coverUrl",
    "fileSize",
    "fileMd5",
    "paySubType",
    "transferStatus",
    "transferId",
    "voipType",
    "locationLat",
    "locationLng",
    "locationPoiname",
    "locationLabel",
    "senderDisplayName",
    "senderAvatarPath",
    "offlineMedia",
]


def test_the_message_model_lists_every_field_of_spec_r_imp_002() -> None:
    spec_fields = SPEC_R_IMP_002_FIELDS
    assert set(spec_fields) == set(KNOWN_MESSAGE_KEYS)


@pytest.mark.parametrize("version", [0, 2, "1", None, True, 1.5])
def test_other_schema_versions_are_refused(version: object) -> None:
    with pytest.raises(UnsupportedSchema) as info:
        check_schema_version("report.json", version)
    assert "schemaVersion" in str(info.value) and info.value.file_kind == "report.json"


def test_each_file_kind_checks_its_schema_version(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedSchema):
        parse_report({"schemaVersion": 2, "exportId": "x"})
    with pytest.raises(UnsupportedSchema):
        parse_meta({"schemaVersion": 2, "username": "u"})
    with pytest.raises(UnsupportedSchema):
        parse_header({"schemaVersion": 3})
    export = build_export(tmp_path / "e", SynthOptions(target_messages=5, schema_version=2))
    with pytest.raises(UnsupportedSchema):
        ExportLayout(export.root).report()


def test_missing_required_fields_are_reported_by_name() -> None:
    with pytest.raises(ExportLayoutError, match="username"):
        parse_meta({"schemaVersion": 1})


# ----------------------------------------------------------------------- layout


def test_only_the_meta_files_are_read_when_listing_conversations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=30, other_messages=5))
    opened: list[str] = []
    real_open = Path.open
    real_read = Path.read_bytes

    def spy_open(self: Path, *args: object, **kwargs: object) -> io.IOBase:
        opened.append(self.name)
        return real_open(self, *args, **kwargs)  # type: ignore[no-any-return,call-overload]

    def spy_read(self: Path) -> bytes:
        opened.append(self.name)
        return real_read(self)

    monkeypatch.setattr(Path, "open", spy_open)
    monkeypatch.setattr(Path, "read_bytes", spy_read)
    layout = ExportLayout(export.root)
    entries = layout.conversations()
    candidates = layout.direct_conversations()
    assert len(entries) == 4 and len(candidates) == 3  # target, two others, one group
    assert "messages.json" not in opened and "meta.json" in opened
    assert candidates[0].username == export.target_username


def test_groups_are_recognised_by_flag_and_by_name() -> None:
    assert is_group_conversation("123@chatroom", False)
    assert is_group_conversation("wxid_x", True)
    assert is_group_conversation("wxid_x", "1")
    assert not is_group_conversation("wxid_x", None)
    assert [truthy(v) for v in (True, 0, 2, "yes", "no", None)] == [
        True,
        False,
        True,
        True,
        False,
        False,
    ]


def test_directory_names_with_special_characters_are_handled(tmp_path: Path) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=3))
    layout = ExportLayout(export.root)
    entry = layout.find(export.target_username)
    assert entry is not None
    assert "🌸" in entry.dir_name and " (备份)" in entry.dir_name and "&co" in entry.dir_name
    assert entry.messages_path.is_file()
    assert layout.find("wxid_" + "unknown000") is None
    assert export.group_username is not None
    group = layout.find(export.group_username)
    assert group is not None and group.is_group


def test_a_directory_without_conversations_is_not_an_export(tmp_path: Path) -> None:
    with pytest.raises(ExportLayoutError, match="not found"):
        ExportLayout(tmp_path / "missing").validate()
    (tmp_path / "empty").mkdir()
    with pytest.raises(ExportLayoutError, match="conversations"):
        ExportLayout(tmp_path / "empty").validate()


def test_an_export_without_report_json_gets_a_derived_identity(tmp_path: Path) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=3))
    (export.root / "report.json").unlink()
    info = ExportLayout(export.root).report()
    assert info.export_id.startswith("derived-")
    (export.root / "report.json").write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")
    assert ExportLayout(export.root).report().export_id.startswith("derived-")


def test_invalid_json_is_reported_not_crashed(tmp_path: Path) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=3))
    (export.root / "report.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ExportLayoutError, match="not valid JSON"):
        ExportLayout(export.root).report()


def test_the_fingerprint_changes_with_the_files(tmp_path: Path) -> None:
    export = build_export(tmp_path / "e", SynthOptions(target_messages=5))
    layout = ExportLayout(export.root)
    entry = layout.find(export.target_username)
    assert entry is not None
    first = layout.fingerprint(entry, "x")
    assert first == layout.fingerprint(entry, "x") and first != layout.fingerprint(entry, "y")
    with entry.messages_path.open("ab") as handle:
        handle.write(b" ")
    assert layout.fingerprint(entry, "x") != first


# ------------------------------------------------------------------------ paths


def test_names_compare_in_unicode_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "é")
    assert decomposed != "é" and nfc(decomposed) == "é"


def test_extended_length_paths() -> None:
    assert extended_length_path("C:\\a\\b") == "\\\\?\\C:\\a\\b"
    assert extended_length_path("C:/a/b") == "\\\\?\\C:\\a\\b"
    assert extended_length_path("\\\\server\\share\\x") == "\\\\?\\UNC\\server\\share\\x"
    assert extended_length_path("\\\\?\\C:\\a") == "\\\\?\\C:\\a"
    assert extended_length_path("relative\\path") == "relative\\path"


def test_long_path_prefixes_only_long_windows_paths(tmp_path: Path) -> None:
    short = tmp_path / "a"
    assert long_path(short, win32=False) == short
    assert long_path(short, win32=True) == short  # below the threshold
    far = tmp_path.joinpath(*["segment"] * 40)
    result = long_path(far, win32=True)
    if sys.platform == "win32":
        assert str(result).startswith("\\\\?\\")
    else:
        assert result == Path(os.path.abspath(far))  # POSIX paths have no drive to prefix


@pytest.mark.parametrize(
    "relative",
    ["", "/etc/passwd", "C:\\x\\y", "..\\x", "a/../../b", "a\0b", "../x", "."],
)
def test_unsafe_relative_paths_are_refused(relative: str) -> None:
    assert split_relative(relative) is None


def test_relative_paths_accept_either_separator() -> None:
    assert split_relative("media\\images\\a.png") == ["media", "images", "a.png"]
    assert split_relative("./media//images/a.png") == ["media", "images", "a.png"]


def test_resolve_in_root_finds_files_and_refuses_escapes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "media" / "images").mkdir(parents=True)
    target = root / "media" / "images" / "图片 (1).png"
    target.write_bytes(b"x")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"y")
    assert resolve_in_root(root, "media/images/图片 (1).png") == target
    assert resolve_in_root(root, "media\\images\\图片 (1).png") == target
    assert resolve_in_root(root, "media/images/missing.png") is None
    assert resolve_in_root(root, "../outside.png") is None
    assert resolve_in_root(root, str(outside)) is None
    assert resolve_in_root(root, "media/images") is None  # a folder is not a file


def test_resolve_in_root_ignores_unicode_normalisation_differences(tmp_path: Path) -> None:
    root = tmp_path / "root"
    folder = root / "media"
    folder.mkdir(parents=True)
    stored = unicodedata.normalize("NFD", "café.png")
    (folder / stored).write_bytes(b"x")
    if not (folder / stored).exists():  # a file system that normalises names itself
        pytest.skip("the file system folded the name")
    found = resolve_in_root(root, "media/" + unicodedata.normalize("NFC", "café.png"))
    assert found is not None and found.read_bytes() == b"x"


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs privileges on Windows")
def test_a_link_leading_out_of_the_export_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    (root / "link.txt").symlink_to(secret)
    assert resolve_in_root(root, "link.txt") is None


def test_masked_directory_names_hide_nickname_and_id() -> None:
    masked = mask_dir_name("003_小满🌸_" + "wxid_" + "abc12345" + "_ffee0011", "小满🌸", 3)
    assert re.fullmatch(r"003_\d+_[0-9a-f]{4}", masked)
    assert masked.split("_")[1] == str(len("小满🌸"))
    assert "小满" not in masked and "wxid" not in masked
    assert mask_dir_name("noprefix", None, 7).startswith("007_")
    assert mask_dir_name("003_x_y", "x", 3) == mask_dir_name("003_x_y", "x", 99)


# -------------------------------------------------------------------- integrity


def digest(data: bytes, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, data).hexdigest()


def test_digest_algorithms_are_told_apart_by_length() -> None:
    assert digest_algorithm("a" * 32) == "md5"
    assert digest_algorithm("a" * 40) == "sha1"
    assert digest_algorithm("a" * 64) == "sha256"
    assert digest_algorithm("a" * 30) is None and digest_algorithm("z" * 32) is None


def test_manifest_shapes_are_parsed() -> None:
    sha = "ab" * 32
    shapes = {
        "m.json": {"files": [{"path": "a/b.png", "sha256": sha, "size": 5}]},
        "n.json": [{"file": "a\\b.png", "md5": "cd" * 16}],
        "o.json": {"entries": {"a/b.png": {"hash": sha, "bytes": 5}}},
        "p.json": {"a/b.png": sha, "a/c.png": 7},
        "q.json": {"manifest": [{"name": "./a/b.png", "checksum": sha}]},
    }
    for name, document in shapes.items():
        entries = parse_manifest_text(name, json.dumps(document).encode())
        assert entries and entries[0].path == "a/b.png", name
        assert entries[0].digest is not None or entries[0].size is not None
    text = f"{sha}  a/b.png\n{'ef' * 16} *c d.png\nnot a line\n".encode()
    entries = parse_manifest_text("SUMS.txt", text)
    assert [(e.path, e.algorithm) for e in entries] == [("a/b.png", "sha256"), ("c d.png", "md5")]
    assert parse_manifest_text("x.json", b"{}") == []


def test_load_integrity_states(tmp_path: Path) -> None:
    assert load_integrity(tmp_path).state is IntegrityState.ABSENT
    folder = tmp_path / "_integrity"
    folder.mkdir()
    (folder / "notes.txt").write_text("nothing useful here", encoding="utf-8")
    (folder / "broken.json").write_text("{", encoding="utf-8")
    index = load_integrity(tmp_path)
    assert index.state is IntegrityState.UNRECOGNIZED and index.problems
    (folder / "SUMS.txt").write_text(f"{'0' * 64}  a.bin\n", encoding="utf-8")
    index = load_integrity(tmp_path)
    assert index.state is IntegrityState.READ and index.lookup("a.bin") is not None


def test_verify_file_distinguishes_ok_failed_and_unlisted(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.bin", tmp_path / "bad.bin"
    good.write_bytes(b"hello")
    bad.write_bytes(b"hello")
    index = IntegrityIndex(IntegrityState.READ)
    index.entries = {
        "good.bin": IntegrityEntry("good.bin", "sha256", digest(b"hello"), 5),
        "bad.bin": IntegrityEntry("bad.bin", "sha256", "0" * 64, 5),
        "short.bin": IntegrityEntry("short.bin", None, None, 99),
        "md5.bin": IntegrityEntry("md5.bin", "md5", digest(b"hello", "md5"), None),
    }
    assert verify_file(index, "good.bin", good).verdict is FileVerdict.OK
    failed = verify_file(index, "bad.bin", bad)
    assert failed.verdict is FileVerdict.FAILED and "sha256" in failed.reason
    assert verify_file(index, "other.bin", good).verdict is FileVerdict.UNLISTED
    assert verify_file(index, "short.bin", good).verdict is FileVerdict.FAILED
    (tmp_path / "md5.bin").write_bytes(b"hello")
    assert verify_file(index, "md5.bin", tmp_path / "md5.bin").verdict is FileVerdict.OK
    assert verify_file(index, "good.bin", tmp_path / "gone.bin").verdict is FileVerdict.FAILED
    assert hash_file(good, "sha1") == digest(b"hello", "sha1")


def test_streams_are_checked_while_they_are_read() -> None:
    data = b"streamed bytes"
    entry = IntegrityEntry("x", "sha256", digest(data), len(data))
    hasher = StreamHasher(io.BytesIO(data), stream_algorithms(entry, "md5"))
    while hasher.read(4):
        pass
    assert check_stream(entry, hasher).verdict is FileVerdict.OK
    assert hasher.hexdigest("md5") == digest(data, "md5")
    wrong = IntegrityEntry("x", "sha256", "1" * 64, None)
    assert check_stream(wrong, hasher).verdict is FileVerdict.FAILED
    short = IntegrityEntry("x", None, None, 3)
    assert check_stream(short, hasher).verdict is FileVerdict.FAILED
    assert check_stream(None, hasher).verdict is FileVerdict.UNLISTED
    assert stream_algorithms(None, "md5") == ("md5",)


def test_manifest_paths_are_normalised() -> None:
    assert normalize_key(".\\media\\a.png") == "media/a.png"
    assert normalize_key("/media/a.png") == "media/a.png"
    assert normalize_key(unicodedata.normalize("NFD", "é")) == "é"


def test_generated_integrity_manifests_verify(tmp_path: Path) -> None:
    for style in ("json", "sha256sum"):
        export = build_export(tmp_path / style, SynthOptions(target_messages=20, integrity=style))
        index = load_integrity(export.root)
        assert index.state is IntegrityState.READ and len(index.entries) > 5
        relative = "conversations/" + export.target_dir.name + "/messages.json"
        assert verify_file(index, relative, export.messages_path).verdict is FileVerdict.OK


# ----------------------------------------------------------------------- jsonio


def test_a_byte_order_mark_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"schemaVersion": 1, "messages": [{"id": 1}]}).encode()
    )
    assert first_value(path, "schemaVersion") == 1
    with path.open("rb") as handle:
        skip_bom(handle)
        assert list(iter_messages(handle)) == [{"id": 1}]
    plain = tmp_path / "p.json"
    plain.write_bytes(b'{"messages": [1, 2, 3]}')
    with plain.open("rb") as handle:
        skip_bom(handle)
        assert list(iter_messages(handle, skip=1)) == [2, 3]
    assert first_value(plain, "missing") is None


def test_numbers_are_floats_not_decimals_and_big_ids_stay_exact(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_bytes(b'{"messages": [{"lat": 41.5, "serverId": 7000000000000007919}]}')
    with path.open("rb") as handle:
        (item,) = list(iter_messages(handle))
    assert item == {"lat": 41.5, "serverId": 7000000000000007919}
    assert isinstance(item["lat"], float)
