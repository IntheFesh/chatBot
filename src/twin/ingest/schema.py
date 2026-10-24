"""Pydantic models of the export files (R-IMP-002).

The export format is ``schemaVersion`` 1.  Every model keeps unknown fields
(``extra="allow"``) so a newer exporter never loses data: the importer stores the
whole original message object in the encrypted ``raw`` column and only records the
*names* of fields it does not know (never their values).

Field names are the exporter's camelCase names on purpose, so that a model can be
compared with the file and with SPEC R-IMP-002 line by line.

Fields the importer relies on are typed; everything else is ``Any`` so that an odd
value in a field nobody reads cannot make a message unimportable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SUPPORTED_SCHEMA_VERSION = 1


class UnsupportedSchema(ValueError):
    """The export declares a ``schemaVersion`` this program does not understand."""

    def __init__(self, file_kind: str, version: object) -> None:
        self.file_kind = file_kind
        self.version = version
        super().__init__(
            f"{file_kind} has schemaVersion {version!r}; only schemaVersion "
            f"{SUPPORTED_SCHEMA_VERSION} is supported. Update wechat-twin or re-export."
        )


def check_schema_version(file_kind: str, version: object) -> None:
    """Raise :class:`UnsupportedSchema` unless ``version`` is exactly 1."""
    if isinstance(version, bool) or version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedSchema(file_kind, version)


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ----------------------------------------------------------------- report.json


class MissingMediaItem(_Open):
    kind: str | None = None
    id: str | int | None = None
    conversation: str | None = None
    messageId: str | int | None = None


class ExportReport(_Open):
    schemaVersion: int
    exportId: str | None = None
    account: Any = None
    createdAt: Any = None
    missingMedia: list[MissingMediaItem] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)


# ------------------------------------------------------------------- meta.json


class ConversationMeta(_Open):
    schemaVersion: int
    username: str
    displayName: str | None = None
    avatarPath: str | None = None
    isGroup: bool | int | str | None = None
    exportedAt: Any = None
    messageCount: int | None = None


# --------------------------------------------------------------- messages.json


class ConversationRef(_Open):
    username: str | None = None
    displayName: str | None = None
    avatarPath: str | None = None
    isGroup: bool | int | str | None = None


class MessageFilters(_Open):
    startTime: Any = None
    endTime: Any = None
    messageTypes: Any = None


class MessagesHeader(_Open):
    """Everything of ``messages.json`` except the ``messages`` array (which is streamed)."""

    schemaVersion: int
    exportedAt: Any = None
    account: Any = None
    conversation: ConversationRef | None = None
    filters: MessageFilters | None = None


class OfflineMedia(_Open):
    kind: str | None = None
    path: str | None = None
    md5: str | None = None
    fileId: str | None = None


class ExportMessage(_Open):
    """One message.  Only the fields the importer uses are typed."""

    id: str | int | None = None
    localId: int | str | None = None
    serverId: int | str | None = None
    createTime: int | float | str | None = None
    createTimeText: str | None = None
    sortSeq: int | float | str | None = None
    type: int | str | None = None
    renderType: str | None = None
    isSent: bool | int | str | None = None
    senderUsername: str | None = None
    conversationUsername: str | None = None
    isGroup: bool | int | str | None = None
    content: str | None = None
    title: str | None = None
    url: Any = None
    from_: Any = Field(default=None, alias="from")
    fromUsername: Any = None
    linkType: Any = None
    linkStyle: Any = None
    objectId: Any = None
    objectNonceId: Any = None
    recordItem: Any = None
    thumbUrl: Any = None
    imageMd5: Any = None
    imageFileId: Any = None
    imageMd5Candidates: Any = None
    imageFileIdCandidates: Any = None
    imageUrl: Any = None
    emojiMd5: str | None = None
    emojiUrl: str | None = None
    videoMd5: Any = None
    videoThumbMd5: Any = None
    videoFileId: Any = None
    videoThumbFileId: Any = None
    videoUrl: Any = None
    videoThumbUrl: Any = None
    voiceLength: Any = None
    voiceTranscript: str | None = None
    voiceTranscriptStatus: Any = None
    voiceTranscriptError: Any = None
    voiceTranscriptLanguage: Any = None
    voiceTranscriptModel: Any = None
    quoteUsername: Any = None
    quoteServerId: Any = None
    quoteType: Any = None
    quoteThumbUrl: Any = None
    quoteVoiceLength: Any = None
    quoteTitle: Any = None
    quoteContent: Any = None
    amount: Any = None
    coverUrl: Any = None
    fileSize: Any = None
    fileMd5: Any = None
    paySubType: Any = None
    transferStatus: Any = None
    transferId: Any = None
    voipType: Any = None
    locationLat: Any = None
    locationLng: Any = None
    locationPoiname: Any = None
    locationLabel: Any = None
    senderDisplayName: Any = None
    senderAvatarPath: str | None = None
    offlineMedia: list[OfflineMedia] | None = None


def _known_message_keys() -> frozenset[str]:
    keys: set[str] = set()
    for name, field in ExportMessage.model_fields.items():
        keys.add(field.alias or name)
    return frozenset(keys)


KNOWN_MESSAGE_KEYS: frozenset[str] = _known_message_keys()
QUOTE_KEYS: tuple[str, ...] = (
    "quoteUsername",
    "quoteServerId",
    "quoteType",
    "quoteThumbUrl",
    "quoteVoiceLength",
    "quoteTitle",
    "quoteContent",
)
