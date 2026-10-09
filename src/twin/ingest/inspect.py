"""Structure probe of an export directory (R-IMP-014): ``twin import inspect``.

The report describes the *shape* of an export and nothing else: the file tree (conversation
folders are shown as ``<sequence>_<nickname length>_<four hash characters>``), the key names
of ``report.json`` / ``meta.json`` / ``messages.json`` with the types of their values and how
often they occur, the value counts of the enumeration fields (``renderType``, ``type``,
``offlineMedia[].kind``, ...), and the files and keys of ``_integrity/``.  It never prints a
value of a text field, an id or a name.  Two rules make sure of that:

* only keys that look like identifiers are printed (``renderType``); any other key - a path,
  a name, a WeChat id used as a map key - is counted under ``<other key>``;
* values are printed only for a short list of enumeration fields, and only when they are
  short plain words or numbers.

The point of the report is to let the person who owns the export check the importer's idea
of the format (the synthetic generator in the tests, SPEC R-IMP-002, the ``_integrity``
reader) against the real files without sharing any of the content.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

import ijson
import orjson

from twin.ingest.integrity import INTEGRITY_DIR
from twin.ingest.jsonio import MESSAGES_PREFIX, skip_bom
from twin.ingest.layout import (
    CONVERSATIONS_DIR,
    MEDIA_DIR,
    MESSAGES_FILE,
    META_FILE,
    REPORT_FILE,
    ExportLayoutError,
)
from twin.ingest.paths import long_path, mask_dir_name, nfc

SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,47}$")
SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.\- ]{1,32}$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")
OTHER_KEY = "<other key>"
ENUM_KEYS = frozenset(
    {
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
    }
)
KNOWN_TOP_LEVEL = frozenset({REPORT_FILE, CONVERSATIONS_DIR, MEDIA_DIR, INTEGRITY_DIR})
ARRAY_LIMIT = 2000
DEFAULT_SAMPLE = 5000


def _is_safe_key(key: str) -> bool:
    lowered = key.lower()
    return (
        bool(SAFE_KEY.match(key)) and not lowered.startswith("wxid") and "chatroom" not in lowered
    )


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _safe_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str) and SAFE_VALUE.match(value) and not value.lower().startswith("wxid"):
        return value
    return None


@dataclass
class Structure:
    """Types, occurrences and enumeration values per path of one kind of JSON document."""

    documents: int = 0
    types: dict[str, Counter[str]] = field(default_factory=dict)
    values: dict[str, Counter[str]] = field(default_factory=dict)

    def observe(self, path: str, value: Any) -> None:
        kind = _type_name(value)
        self.types.setdefault(path, Counter())[kind] += 1
        leaf = path.rsplit(".", 1)[-1].removesuffix("[]")
        if leaf in ENUM_KEYS and kind not in ("object", "array"):
            shown = _safe_value(value)
            self.values.setdefault(path, Counter())[shown if shown is not None else "<other>"] += 1
        if isinstance(value, dict):
            for key, child in value.items():
                label = key if _is_safe_key(key) else OTHER_KEY
                self.observe(f"{path}.{label}", child)
        elif isinstance(value, list):
            for child in islice(value, ARRAY_LIMIT):
                self.observe(f"{path}[]", child)

    def observe_event(self, prefix: str, event: str, value: Any) -> None:
        """Record one low-level parser event (used for the part of a file before ``messages``)."""
        path = "$" + "".join(
            "[]" if part == "item" else "." + (part if _is_safe_key(part) else OTHER_KEY)
            for part in prefix.split(".")
            if part
        )
        if event == "start_map":
            self.types.setdefault(path, Counter())["object"] += 1
        elif event == "start_array":
            self.types.setdefault(path, Counter())["array"] += 1
        elif event in ("string", "number", "integer", "double", "boolean", "null"):
            self.observe(path, value)


@dataclass
class InspectResult:
    tree: list[str] = field(default_factory=list)
    report: Structure = field(default_factory=Structure)
    meta: Structure = field(default_factory=Structure)
    header: Structure = field(default_factory=Structure)
    messages: Structure = field(default_factory=Structure)
    integrity: dict[str, Structure] = field(default_factory=dict)
    integrity_files: list[str] = field(default_factory=list)
    conversations: int = 0
    groups: int = 0
    sampled: int = 0
    sample_limit: int = DEFAULT_SAMPLE
    notes: list[str] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    data = long_path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return orjson.loads(data)


def _count_files(folder: Path) -> tuple[int, Counter[str]]:
    total = 0
    suffixes: Counter[str] = Counter()
    for entry in folder.rglob("*"):
        if long_path(entry).is_file():
            total += 1
            suffix = entry.suffix.lower()
            suffixes[suffix if SAFE_FILE.match(suffix.lstrip(".") or "-") else "<other>"] += 1
    return total, suffixes


def _size_label(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{max(1, size // 1024)} KB" if size else "0 KB"


def _header_events(path: Path) -> Iterator[tuple[str, str, Any]]:
    """Parser events of ``messages.json`` up to the start of the ``messages`` array."""
    with long_path(path).open("rb") as handle:
        skip_bom(handle)
        for prefix, event, value in ijson.parse(handle, use_float=True):
            if prefix == "messages" and event == "start_array":
                return
            yield prefix, event, value


def _inspect_messages(path: Path, result: InspectResult) -> None:
    result.header.documents += 1
    try:
        for prefix, event, value in _header_events(path):
            result.header.observe_event(prefix, event, value)
        with long_path(path).open("rb") as handle:
            skip_bom(handle)
            stream = ijson.items(handle, MESSAGES_PREFIX, use_float=True)
            limit = result.sample_limit or None
            for message in islice(stream, limit):
                result.messages.observe("$.messages[]", message)
                result.sampled += 1
    except (ijson.JSONError, OSError, UnicodeDecodeError):
        result.notes.append("a messages.json could not be read to the end")


def _inspect_integrity(folder: Path, result: InspectResult) -> None:
    hex_line = re.compile(r"^[0-9a-fA-F]{32,64}\s+\S")
    for index, entry in enumerate(sorted(folder.rglob("*")), start=1):
        if not long_path(entry).is_file():
            continue
        name = entry.name if SAFE_FILE.match(entry.name) and "wxid" not in entry.name else None
        label = name or f"<file {index}>"
        size = long_path(entry).stat().st_size
        result.integrity_files.append(f"{label} ({_size_label(size)})")
        if entry.suffix.lower() == ".json" and size < 256 * 1024 * 1024:
            try:
                structure = result.integrity.setdefault(label, Structure())
                structure.documents += 1
                structure.observe("$", _read_json(entry))
            except (orjson.JSONDecodeError, OSError):
                result.notes.append(f"{label} is not valid JSON")
        elif size < 64 * 1024 * 1024:
            try:
                lines = long_path(entry).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            digests = sum(1 for line in lines if hex_line.match(line))
            result.notes.append(
                f"{label}: {len(lines)} text line(s), {digests} of the form '<hex digest> <path>'"
            )


def inspect_export(root: Path, *, sample: int = DEFAULT_SAMPLE) -> InspectResult:
    """Collect the structure of the export below ``root`` (see the module docstring)."""
    if not long_path(root).is_dir():
        raise ExportLayoutError(f"export directory not found: {root}")
    result = InspectResult(sample_limit=sample)
    for entry in sorted(root.iterdir(), key=lambda item: nfc(item.name)):
        name = nfc(entry.name)
        shown = name if name in KNOWN_TOP_LEVEL else "<other entry>"
        result.tree.append(f"{shown}{'/' if long_path(entry).is_dir() else ''}")

    report_path = root / REPORT_FILE
    if long_path(report_path).is_file():
        try:
            result.report.documents += 1
            result.report.observe("$", _read_json(report_path))
        except (orjson.JSONDecodeError, OSError):
            result.notes.append("report.json is not valid JSON")

    base = root / CONVERSATIONS_DIR
    folders = (
        sorted((p for p in base.iterdir() if long_path(p).is_dir()), key=lambda p: nfc(p.name))
        if long_path(base).is_dir()
        else []
    )
    for position, folder in enumerate(folders, start=1):
        meta_path = folder / META_FILE
        nickname: str | None = None
        is_group = False
        if long_path(meta_path).is_file():
            try:
                meta = _read_json(meta_path)
            except (orjson.JSONDecodeError, OSError):
                result.notes.append("a meta.json is not valid JSON")
            else:
                result.meta.documents += 1
                result.meta.observe("$", meta)
                if isinstance(meta, dict):
                    nickname = (
                        meta.get("displayName")
                        if isinstance(meta.get("displayName"), str)
                        else None
                    )
                    is_group = bool(meta.get("isGroup")) or str(meta.get("username", "")).endswith(
                        "@chatroom"
                    )
        label = mask_dir_name(folder.name, nickname, position)
        files = []
        for child in sorted(folder.iterdir(), key=lambda p: p.name):
            if child.name in (META_FILE, MESSAGES_FILE):
                size = long_path(child).stat().st_size if long_path(child).is_file() else 0
                files.append(f"{child.name} ({_size_label(size)})")
            else:
                files.append("<other file>" if long_path(child).is_file() else "<other folder>")
        kind = "group" if is_group else "direct"
        result.tree.append(f"{CONVERSATIONS_DIR}/{label} [{kind}]: {', '.join(files) or '-'}")
        result.conversations += 1
        result.groups += 1 if is_group else 0
        messages_path = folder / MESSAGES_FILE
        if long_path(messages_path).is_file():
            _inspect_messages(messages_path, result)

    media = root / MEDIA_DIR
    if long_path(media).is_dir():
        for folder in sorted(media.iterdir(), key=lambda p: p.name):
            if not long_path(folder).is_dir():
                continue
            count, suffixes = _count_files(folder)
            shown = folder.name if SAFE_FILE.match(folder.name) else "<other folder>"
            kinds = ", ".join(f"{k or '-'}×{v}" for k, v in sorted(suffixes.items()))
            result.tree.append(f"{MEDIA_DIR}/{shown}: {count} file(s) {kinds}")

    integrity = root / INTEGRITY_DIR
    if long_path(integrity).is_dir():
        _inspect_integrity(integrity, result)
    return result


# ------------------------------------------------------------------- rendering


def _structure_lines(title: str, structure: Structure) -> list[str]:
    lines = [f"## {title}", ""]
    if not structure.types:
        return [*lines, "（没有数据）", ""]
    lines += [
        f"文件数：{structure.documents}",
        "",
        "| 路径 | 值类型（出现次数） |",
        "| --- | --- |",
    ]
    for path in sorted(structure.types):
        kinds = ", ".join(f"{k}×{v}" for k, v in sorted(structure.types[path].items()))
        lines.append(f"| `{path}` | {kinds} |")
    if structure.values:
        lines += [
            "",
            "取值计数（只列枚举类字段）：",
            "",
            "| 路径 | 取值 | 次数 |",
            "| --- | --- | --- |",
        ]
        for path in sorted(structure.values):
            for value, count in sorted(structure.values[path].items(), key=lambda kv: -kv[1]):
                lines.append(f"| `{path}` | `{value}` | {count} |")
    lines.append("")
    return lines


def render_inspect(result: InspectResult, now: datetime) -> str:
    """Markdown for ``data/reports/inspect-<UTC>.md`` (no values, see the module docstring)."""
    lines = [
        "# 导出结构探查（只含结构，不含任何值）",
        "",
        f"- 生成时间（UTC）：{now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 会话文件夹：{result.conversations}（其中群聊 {result.groups}）",
        f"- 消息抽样：每个 messages.json 最多 {result.sample_limit or '全部'} 条，"
        f"共检视 {result.sampled} 条",
        "",
        "## 文件树",
        "",
        *[f"- {line}" for line in result.tree],
        "",
        *_structure_lines("report.json", result.report),
        *_structure_lines("meta.json（所有会话合并）", result.meta),
        *_structure_lines("messages.json 顶层（messages 数组之前的部分）", result.header),
        *_structure_lines("messages.json 的消息体（抽样）", result.messages),
        "## _integrity/",
        "",
    ]
    if result.integrity_files:
        lines += [f"- {entry}" for entry in result.integrity_files]
    else:
        lines.append("没有 _integrity 文件夹。")
    lines.append("")
    for name, structure in sorted(result.integrity.items()):
        lines += _structure_lines(f"_integrity/{name} 的 JSON 结构", structure)
    if result.notes:
        lines += ["## 备注", "", *[f"- {note}" for note in result.notes], ""]
    return "\n".join(lines)


def write_inspect_report(reports_dir: Path, text: str, now: datetime) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"inspect-{now.strftime('%Y%m%dT%H%M%SZ')}.md"
    path.write_text(text, encoding="utf-8")
    return path
