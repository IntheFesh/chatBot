"""iLink wire constants and tolerant models (``docs/ILINK_PROTOCOL.md`` sections 5 and 6).

The server may send fields this project does not know and omit fields it normally sends, so
every model ignores unknown keys and makes every field optional.  64-bit message ids arrive
as JSON integers; Python reads them exactly and :func:`as_text` turns them into strings.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict

DEFAULT_API_BASE = "https://ilinkai.weixin.qq.com"
CDN_BASE = "https://novac2c.cdn.weixin.qq.com/c2c"
APP_ID = "bot"
CLIENT_VERSION = "132105"  # (2 << 16) | (4 << 8) | 9, the official plugin 2.4.9 this follows
BOT_TYPE = 3

# error codes (section 3)
ERR_TOKEN_STALE = -14  # the bot token is no longer valid: log in again
ERR_SEND_REJECTED = -2  # reported by the community for window/quota refusals; meaning unconfirmed

# item types (section 6.2)
ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5
ITEM_TOOL_START = 11
ITEM_TOOL_RESULT = 12

MESSAGE_USER = 1
MESSAGE_BOT = 2
STATE_FINISH = 2

MEDIA_IMAGE = 1  # getuploadurl media_type (section 5.7)

TYPING_START = 1
TYPING_CANCEL = 2

TEXT_CHUNK_LIMIT = 4000  # the official client splits text at 4000 characters


def as_text(value: Any) -> str | None:
    """Ids and counters may arrive as numbers or strings; ``None`` stays ``None``."""
    if value is None or value == "":
        return None
    return str(value)


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CdnMedia(_Wire):
    encrypt_query_param: str | None = None
    aes_key: str | None = None
    encrypt_type: int | None = None
    full_url: str | None = None


class TextItem(_Wire):
    text: str | None = None


class ImageItem(_Wire):
    media: CdnMedia | None = None
    thumb_media: CdnMedia | None = None
    aeskey: str | None = None
    mid_size: int | None = None


class VoiceItem(_Wire):
    media: CdnMedia | None = None
    encode_type: int | None = None
    sample_rate: int | None = None
    playtime: int | None = None
    text: str | None = None


class FileItem(_Wire):
    media: CdnMedia | None = None
    file_name: str | None = None
    md5: str | None = None
    len: str | int | None = None


class VideoItem(_Wire):
    media: CdnMedia | None = None
    video_size: int | None = None
    play_length: int | None = None
    thumb_media: CdnMedia | None = None
    thumb_width: int | None = None
    thumb_height: int | None = None


class PartialText(_Wire):
    start: str | None = None
    end: str | None = None
    startindex: int | None = None
    endindex: int | None = None
    quotemd5: str | None = None


class RefMsg(_Wire):
    title: str | None = None
    svr_id: int | str | None = None
    message_item: WireItem | None = None
    partial_text: PartialText | None = None


class WireItem(_Wire):
    type: int | None = None
    msg_id: int | str | None = None
    text_item: TextItem | None = None
    image_item: ImageItem | None = None
    voice_item: VoiceItem | None = None
    file_item: FileItem | None = None
    video_item: VideoItem | None = None
    ref_msg: RefMsg | None = None


class WireMessage(_Wire):
    seq: int | None = None
    message_id: int | str | None = None
    from_user_id: str | None = None
    to_user_id: str | None = None
    client_id: str | None = None
    create_time_ms: float | None = None
    message_type: int | None = None
    message_state: int | None = None
    item_list: list[WireItem] | None = None
    context_token: str | None = None
    session_id: str | None = None
    group_id: str | None = None

    def items(self) -> list[WireItem]:
        return list(self.item_list or [])

    def dedup_id(self) -> str:
        """``message_id``, else the first item's ``msg_id``, else a hash of the message."""
        direct = as_text(self.message_id)
        if direct:
            return direct
        for item in self.items():
            from_item = as_text(item.msg_id)
            if from_item:
                return from_item
        body = self.model_copy(update={"context_token": None}).model_dump_json(exclude_none=True)
        return "h-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


RefMsg.model_rebuild()
WireItem.model_rebuild()
WireMessage.model_rebuild()
