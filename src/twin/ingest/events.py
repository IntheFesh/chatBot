"""Event text: how non-text messages appear in prompts and training data (R-IMP-007).

One template table (:data:`EVENT_TEMPLATES`) drives everything:

* :func:`render_event_text` turns a stored message into its bracketed event text
  (``[通话 37 分钟]``, ``[语音 5 秒，未转写]``, ...).  The persona prompts (round 09) and the
  training set (round 13) both call this single function;
* :class:`EventTextDetector` is derived *from the same table* and recognises a line that holds
  such a text anywhere (R-SAFE-006, D-670): a line the bot is about to send that carries an
  event text is removed, and training targets / retrieval examples are filtered with it.  The
  rule is the literal prefix of each template (``[图片`` / ``【语音`` ...), searched in folded
  text; because it is derived, adding or changing a template changes the detector too;
* :func:`is_reproducible` tells which kinds the bot could itself have sent.

Fields are written ``{name}``; the literal parts must not contain braces.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Protocol

MAX_TITLE_CHARS = 60


class Kind(StrEnum):
    """Normalised message kinds (R-IMP-007)."""

    TEXT = "text"
    STICKER = "sticker"
    IMAGE = "image"
    QUOTE = "quote"
    VOICE = "voice"
    VIDEO = "video"
    FILE = "file"
    LINK = "link"
    CALL = "call"
    TRANSFER = "transfer"
    REDPACKET = "redpacket"
    SYSTEM = "system"
    LOCATION = "location"
    CHATHISTORY = "chathistory"
    UNKNOWN = "unknown"


KINDS: tuple[str, ...] = tuple(kind.value for kind in Kind)

# kinds the bot could have produced itself: text (with emoji codes), stickers and quotes
REPRODUCIBLE_KINDS = frozenset({Kind.TEXT.value, Kind.STICKER.value, Kind.QUOTE.value})


class CallStatus(StrEnum):
    CONNECTED = "connected"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    MISSED = "missed"
    OTHER_DEVICE = "other_device"
    UNKNOWN = "unknown"


# ------------------------------------------------------------------ templates

EVENT_TEMPLATES: Mapping[str, str] = {
    "image": "[图片]",
    "image_caption": "[图片：{caption}]",
    "video": "[视频]",
    "video_caption": "[视频：{caption}]",
    "voice": "[语音 {seconds} 秒：{transcript}]",
    "voice_untranscribed": "[语音 {seconds} 秒，未转写]",
    "voice_no_length": "[语音：{transcript}]",
    "voice_untranscribed_no_length": "[语音，未转写]",
    "call": "[通话 {duration}]",
    "call_missed": "[未接通话]",
    "call_other_device": "[通话已在其它设备接听]",
    "call_unknown": "[通话]",
    "transfer": "[转账]",
    "redpacket": "[红包]",
    "link": "[链接：{title}]",
    "link_untitled": "[链接]",
    "file": "[文件：{title}]",
    "file_unnamed": "[文件]",
    "location": "[位置：{place}]",
    "location_unnamed": "[位置]",
    "chathistory": "[聊天记录]",
    "revoke_her": "[她撤回了一条消息]",
    "revoke_user": "[你撤回了一条消息]",
    "pat": "[拍一拍]",
    "system": "[系统消息]",
    "unknown": "[其他消息]",
}

_FIELD = re.compile(r"\{([a-z_]+)\}")


def format_call_duration(seconds: int) -> str:
    """``37 分钟`` / ``45 秒`` / ``1 小时 5 分钟`` (minutes are rounded down)."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} 秒"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} 分钟"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小时 {minutes} 分钟" if minutes else f"{hours} 小时"


def _one_line(text: str | None, limit: int | None = None) -> str:
    collapsed = " ".join((text or "").split())
    if limit is not None and len(collapsed) > limit:
        return collapsed[: limit - 1] + "…"
    return collapsed


class EventSource(Protocol):
    """What :func:`render_event_text` needs from a message (the ``messages`` row has it all)."""

    @property
    def kind(self) -> str: ...

    @property
    def is_sent(self) -> bool: ...

    @property
    def text(self) -> str | None: ...

    @property
    def call_status(self) -> str | None: ...

    @property
    def call_duration_s(self) -> int | None: ...

    @property
    def voice_seconds(self) -> int | None: ...

    @property
    def has_transcript(self) -> bool: ...


def _fill(name: str, templates: Mapping[str, str], **fields: object) -> str:
    return _FIELD.sub(lambda m: str(fields[m.group(1)]), templates[name])


def render_event_text(
    message: EventSource,
    *,
    caption: str | None = None,
    templates: Mapping[str, str] = EVENT_TEMPLATES,
) -> str | None:
    """The event text of ``message``, or ``None`` for ordinary chat (text, sticker, quote).

    ``caption`` is the image description (``media_assets.caption``) when there is one; the
    text then reads ``[图片：<描述>]``.
    """
    kind = message.kind
    if kind in (Kind.TEXT.value, Kind.STICKER.value, Kind.QUOTE.value):
        return None
    if kind in (Kind.IMAGE.value, Kind.VIDEO.value):
        described = _one_line(caption)
        if described:
            return _fill(f"{kind}_caption", templates, caption=described)
        return templates[kind]
    if kind == Kind.VOICE.value:
        return _render_voice(message, templates)
    if kind == Kind.CALL.value:
        return _render_call(message, templates)
    if kind == Kind.TRANSFER.value:
        return templates["transfer"]
    if kind == Kind.REDPACKET.value:
        return templates["redpacket"]
    if kind == Kind.CHATHISTORY.value:
        return templates["chathistory"]
    if kind == Kind.LINK.value:
        title = _one_line(message.text, MAX_TITLE_CHARS)
        return _fill("link", templates, title=title) if title else templates["link_untitled"]
    if kind == Kind.FILE.value:
        title = _one_line(message.text, MAX_TITLE_CHARS)
        return _fill("file", templates, title=title) if title else templates["file_unnamed"]
    if kind == Kind.LOCATION.value:
        place = _one_line(message.text, MAX_TITLE_CHARS)
        return _fill("location", templates, place=place) if place else templates["location_unnamed"]
    if kind == Kind.SYSTEM.value:
        return _render_system(message, templates)
    return templates["unknown"]


def _render_voice(message: EventSource, templates: Mapping[str, str]) -> str:
    transcript = _one_line(message.text) if message.has_transcript else ""
    seconds = message.voice_seconds
    if seconds is None or seconds <= 0:
        if transcript:
            return _fill("voice_no_length", templates, transcript=transcript)
        return templates["voice_untranscribed_no_length"]
    if transcript:
        return _fill("voice", templates, seconds=seconds, transcript=transcript)
    return _fill("voice_untranscribed", templates, seconds=seconds)


def _render_call(message: EventSource, templates: Mapping[str, str]) -> str:
    status = message.call_status
    if status == CallStatus.CONNECTED.value:
        duration = message.call_duration_s
        if duration is not None and duration > 0:
            return _fill("call", templates, duration=format_call_duration(duration))
        return templates["call_unknown"]
    if status in (
        CallStatus.CANCELLED.value,
        CallStatus.REJECTED.value,
        CallStatus.MISSED.value,
    ):
        return templates["call_missed"]
    if status == CallStatus.OTHER_DEVICE.value:
        return templates["call_other_device"]
    return templates["call_unknown"]


def _render_system(message: EventSource, templates: Mapping[str, str]) -> str:
    text = message.text or ""
    if "撤回" in text:
        own = message.is_sent or text.lstrip().startswith("你")
        return templates["revoke_user" if own else "revoke_her"]
    if "拍了拍" in text or "拍一拍" in text:
        return templates["pat"]
    return templates["system"]


def is_reproducible(message: object) -> bool:
    """True for kinds the bot could have sent itself (text, sticker, quote).

    Images, voice, video, calls, transfers, red packets, locations, files, links, chat
    histories and system events cannot be reproduced; they are excluded from training
    targets and from blind-test samples (R-SAFE-006).  Accepts a message or a kind name.
    """
    kind = message if isinstance(message, str) else getattr(message, "kind", None)
    return str(kind) in REPRODUCIBLE_KINDS


# ------------------------------------------------------------------- detector


#: Neutral wordings a model may write for a template whose own wording names a person
#: (``[她撤回了一条消息]`` / ``[你撤回了一条消息]``): extra literal prefix keys per template name.
#: They only apply to a detector whose table has that template.
EVENT_KEY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "revoke_her": ("撤回了一条消息", "对方撤回了一条消息"),
    "revoke_user": ("我撤回了一条消息",),
}

# After NFKC the full-width "［" is "[", but "【" is not folded, so both openings are listed.
_OPENING = r"[\[【]"
# where the literal prefix of a template ends: a field, a space, a separator or a closing bracket
_KEY_END = re.compile(r"[{\s:,\]】]")


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def fold_event_text(text: str) -> str:
    """``text`` as the detector reads it: NFKC, without any whitespace or format (Cf) character.

    ``［ 语​音　８ 秒］`` becomes ``[语音8秒]``, so a model (or a chat export) that spaces
    out or decorates a placeholder cannot slip it past the literal search.
    """
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text)
        if not (char.isspace() or unicodedata.category(char) == "Cf")
    )


def key_for_template(template: str) -> str | None:
    """The literal a rendering of ``template`` starts with after its opening bracket.

    ``[语音 {seconds} 秒：{transcript}]`` -> ``语音``; ``[通话已在其它设备接听]`` is its own key.
    ``None`` when the template does not open with ``[`` or ``【`` and a literal.
    """
    normalised = _normalise(template)
    if not normalised or normalised[0] not in "[【":
        return None
    literal = normalised[1:]
    end = _KEY_END.search(literal)
    key = fold_event_text(literal[: end.start()] if end else literal)
    return key or None


def pattern_for_template(template: str) -> re.Pattern[str]:
    """Regular expression that finds ``template`` in a folded line (see :func:`fold_event_text`).

    It looks for the bracketed literal prefix (:func:`key_for_template`) anywhere in the line;
    the closing bracket and the fields are not required, so a cut-off output is found too.
    Raises ``ValueError`` for a template without a bracketed literal prefix.
    """
    key = key_for_template(template)
    if key is None:
        raise ValueError(f"template has no bracketed literal prefix: {template!r}")
    return re.compile(_OPENING + re.escape(key))


class EventTextDetector:
    """Recognises lines that hold event text; the rules are derived from the template table.

    A line is event text when, once folded (:func:`fold_event_text`), it contains ``[`` or
    ``【`` right before the literal prefix of a template, anywhere in the line, with or without
    the closing bracket (R-SAFE-006, D-670).  Templates that share a prefix (``[图片]`` and
    ``[图片：…]``) are one family and are named by the first of them in table order.
    """

    def __init__(
        self,
        templates: Mapping[str, str] = EVENT_TEMPLATES,
        aliases: Mapping[str, tuple[str, ...]] = EVENT_KEY_ALIASES,
    ) -> None:
        self._names = tuple(templates)
        owners: dict[str, str] = {}
        for name, text in templates.items():
            keys = (key_for_template(text), *(fold_event_text(a) for a in aliases.get(name, ())))
            for key in keys:
                if key:
                    owners.setdefault(key, name)
        self._owners = owners
        longest_first = sorted(owners, key=lambda key: (-len(key), key))
        self._search = (
            re.compile(_OPENING + "(" + "|".join(re.escape(key) for key in longest_first) + ")")
            if longest_first
            else None
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def match(self, line: str) -> str | None:
        """Name of the template family whose placeholder ``line`` holds, else ``None``."""
        if self._search is None:
            return None
        found = self._search.search(fold_event_text(line))
        return self._owners[found.group(1)] if found else None

    def is_event_text(self, line: str) -> bool:
        return self.match(line) is not None

    def strip(self, text: str) -> tuple[str, int]:
        """``text`` without its event text lines, and the number of lines removed."""
        kept: list[str] = []
        removed = 0
        for line in text.split("\n"):
            if self.is_event_text(line):
                removed += 1
            else:
                kept.append(line)
        return "\n".join(kept), removed

    def lines_without_events(self, lines: Iterable[str]) -> list[str]:
        return [line for line in lines if not self.is_event_text(line)]


default_detector = EventTextDetector()
