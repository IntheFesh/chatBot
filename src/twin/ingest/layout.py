"""Reading the structure of an export directory (R-IMP-001, R-IMP-003).

Only ``report.json`` and the small ``meta.json`` of every conversation are read here.
The ``messages.json`` of a conversation is opened by the importer for the *target*
conversation alone (privacy minimisation, R-IMP-003).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, ValidationError

from twin.ingest.paths import long_path, mask_dir_name, nfc
from twin.ingest.schema import (
    ConversationMeta,
    ExportReport,
    MessagesHeader,
    UnsupportedSchema,
    check_schema_version,
)

REPORT_FILE = "report.json"
CONVERSATIONS_DIR = "conversations"
META_FILE = "meta.json"
MESSAGES_FILE = "messages.json"
MEDIA_DIR = "media"
EMOJI_MEDIA_DIR = "media/emojis"


class ExportLayoutError(ValueError):
    """The directory does not look like an export, or a file in it cannot be read."""


@dataclass(frozen=True)
class ConversationEntry:
    """A conversation folder with its ``meta.json``."""

    dir_name: str
    path: Path
    position: int
    meta: ConversationMeta

    @property
    def username(self) -> str:
        return self.meta.username

    @property
    def is_group(self) -> bool:
        return is_group_conversation(self.meta.username, self.meta.isGroup)

    @property
    def messages_path(self) -> Path:
        return self.path / MESSAGES_FILE

    @property
    def masked_name(self) -> str:
        return mask_dir_name(self.dir_name, self.meta.displayName, self.position)


def truthy(value: object) -> bool:
    """Boolean from the many ways an exporter writes one (``true``, ``1``, ``"true"``)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def is_group_conversation(username: str | None, flag: object) -> bool:
    return truthy(flag) or bool(username and username.endswith("@chatroom"))


def read_json(path: Path) -> Any:
    """Parse a (small) JSON file; a UTF-8 byte order mark is tolerated."""
    try:
        data = long_path(path).read_bytes()
    except OSError as exc:
        raise ExportLayoutError(f"cannot read {path.name}: {exc.strerror or exc}") from exc
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError as exc:
        raise ExportLayoutError(f"{path.name} is not valid JSON") from exc


def _model[M: BaseModel](model: type[M], data: Any, file_kind: str) -> M:
    if isinstance(data, dict):
        check_schema_version(file_kind, data.get("schemaVersion"))
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()}))
        raise ExportLayoutError(f"{file_kind} has missing or invalid fields: {fields}") from exc


def parse_report(data: Any) -> ExportReport:
    return _model(ExportReport, data, REPORT_FILE)


def parse_meta(data: Any) -> ConversationMeta:
    return _model(ConversationMeta, data, META_FILE)


def parse_header(data: Any) -> MessagesHeader:
    return _model(MessagesHeader, data, MESSAGES_FILE)


@dataclass(frozen=True)
class ExportInfo:
    """``report.json`` of an export plus the export's identity."""

    report: ExportReport
    export_id: str


class ExportLayout:
    """An export root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # ------------------------------------------------------------------- checks

    def validate(self) -> None:
        if not long_path(self.root).is_dir():
            raise ExportLayoutError(f"export directory not found: {self.root}")
        if not long_path(self.root / CONVERSATIONS_DIR).is_dir():
            raise ExportLayoutError(
                f"{self.root} has no '{CONVERSATIONS_DIR}' folder; "
                "this does not look like a chat export"
            )

    # ------------------------------------------------------------------- report

    def report(self) -> ExportInfo:
        """``report.json`` (its schema version is checked).

        An export without ``report.json`` is accepted; its identity is then derived from the
        conversation folder names.
        """
        path = self.root / REPORT_FILE
        if not long_path(path).is_file():
            names = sorted(nfc(entry.name) for entry in self._conversation_dirs())
            digest = hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()[:16]
            return ExportInfo(ExportReport(schemaVersion=1, exportId=None), f"derived-{digest}")
        report = parse_report(read_json(path))
        if report.exportId:
            return ExportInfo(report, report.exportId)
        stat = long_path(path).stat()
        digest = hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
        return ExportInfo(report, f"derived-{digest}")

    # ----------------------------------------------------------- conversations

    def _conversation_dirs(self) -> list[Path]:
        base = self.root / CONVERSATIONS_DIR
        if not long_path(base).is_dir():
            return []
        return sorted(
            (entry for entry in base.iterdir() if long_path(entry).is_dir()),
            key=lambda entry: nfc(entry.name),
        )

    def conversations(self) -> list[ConversationEntry]:
        """Every conversation folder with a readable ``meta.json`` (messages are not opened)."""
        entries: list[ConversationEntry] = []
        for position, folder in enumerate(self._conversation_dirs(), start=1):
            meta_path = folder / META_FILE
            if not long_path(meta_path).is_file():
                continue
            meta = parse_meta(read_json(meta_path))
            entries.append(ConversationEntry(nfc(folder.name), folder, position, meta))
        return entries

    def direct_conversations(self) -> list[ConversationEntry]:
        """The non-group conversations (the candidates for the target, R-IMP-003)."""
        return [entry for entry in self.conversations() if not entry.is_group]

    def find(self, username: str) -> ConversationEntry | None:
        for entry in self.conversations():
            if entry.username == username:
                return entry
        return None

    # ------------------------------------------------------------------ messages

    def fingerprint(self, entry: ConversationEntry, export_id: str) -> str:
        """Identity of the export files an import run works from (R-IMP-006)."""
        parts = [export_id]
        for path in (self.root / REPORT_FILE, entry.messages_path, entry.path / META_FILE):
            try:
                stat = long_path(path).stat()
            except OSError:
                parts.append("-")
            else:
                parts.append(f"{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


__all__ = [
    "CONVERSATIONS_DIR",
    "EMOJI_MEDIA_DIR",
    "MEDIA_DIR",
    "MESSAGES_FILE",
    "META_FILE",
    "REPORT_FILE",
    "ConversationEntry",
    "ExportInfo",
    "ExportLayout",
    "ExportLayoutError",
    "UnsupportedSchema",
    "is_group_conversation",
    "parse_header",
    "parse_meta",
    "parse_report",
    "read_json",
    "truthy",
]
