"""Synthetic chat exports with the structure of the real one (R-IMP-001, R-IMP-002).

``build_export`` writes a directory that looks like an export of the chat application::

    report.json
    conversations/<sequence>_<nickname>_<wxid>_<hash>/{meta.json, messages.json}
    media/{images,emojis,avatars}/...
    _integrity/...                         (optional)

Everything is invented: the sentences are random strings of common characters, the
identifiers are generated at run time, the pictures are tiny generated images.  Nothing here
is, or is derived from, a real conversation.  The messages cover every ``renderType`` the
importer knows plus one it does not.

The writer streams ``messages.json`` message by message, so ``scripts/bench_import.py`` can
produce a million messages without holding them in memory.

Field names are the exporter's; they are written as keyword arguments (not quoted dict keys)
so that ``scripts/privacy_scan.py`` does not mistake this file for an export fragment.
"""

from __future__ import annotations

import hashlib
import io
import random
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import orjson
from PIL import Image

CHARS = (
    "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之"
    "过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如"
    "前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其"
    "种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便位因重被走电四第门相次东政"
    "海口使教西再平真听世气信北少关并内加化由却代军产入先山无记老吃饭睡觉今晚想念早安"
)
SPECIAL_NICKNAME_PARTS = ("小满", "🌸", " (备份)", "&co", "#1", "'q'", "％", "·")
DEFAULT_START = datetime(2025, 1, 1, 8, 0, tzinfo=UTC)
DEFAULT_EXPORTED_AT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
SOURCE_ZONE = ZoneInfo("America/Chicago")

# relative weights of the renderTypes in a generated conversation
DEFAULT_MIX: dict[str, float] = {
    "text": 62.0,
    "emoji": 12.0,
    "quote": 5.0,
    "image": 5.0,
    "voice": 4.0,
    "voip": 2.0,
    "system": 2.0,
    "transfer": 0.6,
    "redPacket": 0.6,
    "link": 2.0,
    "video": 1.5,
    "file": 1.0,
    "location": 1.0,
    "chathistory": 0.8,
    "holographic": 0.5,  # a renderType the importer has never heard of
}
TEXT_ONLY_MIX: dict[str, float] = {"text": 100.0}
VOIP_CONTENTS = (
    "通话时长 37:12",
    "通话时长 01:05",
    "通话时长 00:45",
    "通话时长 1:02:03",
    "对方已取消",
    "已拒绝",
    "未应答",
    "已在其它设备接听",
)


def message(**fields: Any) -> dict[str, Any]:
    """A message object; the keyword names are the exporter's field names."""
    return fields


def make_image_bytes(rng: random.Random, fmt: str = "PNG", size: int = 24) -> bytes:
    """A tiny picture with random colours (GIF, PNG or JPEG)."""
    image = Image.new("RGB", (size, size + rng.randint(0, 8)))
    pixels = image.load()
    assert pixels is not None
    base = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    for x in range(image.width):
        for y in range(image.height):
            pixels[x, y] = (
                (base[0] + x * 5) % 256,
                (base[1] + y * 5) % 256,
                (base[2] + x * y) % 256,
            )
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@dataclass
class SynthOptions:
    target_messages: int = 120
    other_conversations: int = 2
    other_messages: int = 12
    include_group: bool = True
    seed: int = 7
    start: datetime = DEFAULT_START
    average_gap_s: float = 900.0
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    media: bool = True  # write picture, sticker and avatar files
    local_sticker_ratio: float = 0.4  # stickers whose file is in media/emojis
    missing_image_ratio: float = 0.25  # images without an offlineMedia entry
    sticker_kinds: int = 6
    export_id: str = "synthetic-export-0001"
    exported_at: datetime = DEFAULT_EXPORTED_AT
    schema_version: int = 1
    integrity: str | None = None  # "json" or "sha256sum"
    corrupt: tuple[str, ...] = ()  # relative paths whose manifest digest is made wrong
    keep_texts: bool = True
    keep_ids: bool = True
    shift_text_hours: int = 0  # createTimeText written this many hours off (zone mistakes)
    target_username: str | None = None
    target_display_name: str | None = None


@dataclass
class StickerInfo:
    md5: str
    url: str
    data: bytes
    local: bool
    mime: str


@dataclass
class SynthExport:
    root: Path
    options: SynthOptions
    target_username: str
    target_display_name: str
    target_dir: Path
    other_usernames: list[str]
    group_username: str | None
    counts: Counter[tuple[str, bool]]  # (renderType, isSent) -> messages in the target
    message_ids: list[str]
    texts: set[str]
    stickers: dict[str, StickerInfo]
    image_md5s: list[str]
    image_paths: list[str]  # export-relative paths of the target's picture files
    missing_message_ids: list[str]
    target_messages: int
    first_time: datetime | None = None
    last_time: datetime | None = None

    @property
    def messages_path(self) -> Path:
        return self.target_dir / "messages.json"

    def total(self, render_type: str) -> int:
        return sum(n for (kind, _), n in self.counts.items() if kind == render_type)


def _other_nickname(number: int) -> str:
    return f"{SPECIAL_NICKNAME_PARTS[number % 4]}{number}{SPECIAL_NICKNAME_PARTS[-1]}"


def _wxid(rng: random.Random) -> str:
    return "wxid_" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(12))


def _sentence_pool(rng: random.Random, size: int = 4000) -> list[str]:
    pool: list[str] = []
    for _ in range(size):
        words = [
            "".join(rng.choice(CHARS) for _ in range(rng.randint(1, 3)))
            for _ in range(rng.randint(1, 5))
        ]
        pool.append("".join(words) + rng.choice(["", "", "", "啊", "呀", "吧", "！", "？"]))
    return pool


class _Conversation:
    """Generator state of one conversation."""

    def __init__(
        self,
        export: Path,
        index: int,
        rng: random.Random,
        options: SynthOptions,
        *,
        username: str,
        display_name: str,
        group: bool,
        count: int,
        is_target: bool,
        pool: list[str],
    ) -> None:
        self.export = export
        self.index = index
        self.rng = rng
        self.options = options
        self.username = username
        self.display_name = display_name
        self.group = group
        self.count = count
        self.is_target = is_target
        self.pool = pool
        self.dir_name = self._dir_name()
        self.dir = export / "conversations" / self.dir_name
        self.counts: Counter[tuple[str, bool]] = Counter()
        self.ids: list[str] = []
        self.texts: set[str] = set()
        self.stickers: dict[str, StickerInfo] = {}
        self.image_md5s: list[str] = []
        self.image_paths: list[str] = []
        self.missing_ids: list[str] = []
        self.first: datetime | None = None
        self.last: datetime | None = None
        self.files: list[Path] = []
        weights = options.mix
        self._kinds = list(weights)
        self._weights = [weights[k] for k in self._kinds]
        self._builders: dict[str, Callable[[int, bool], dict[str, Any]]] = {
            kind: getattr(self, f"_{kind.lower()}", self._holographic) for kind in self._kinds
        }

    def _dir_name(self) -> str:
        digest = hashlib.sha256(self.username.encode()).hexdigest()[:8]
        return f"{self.index:03d}_{self.display_name}_{self.username}_{digest}"

    # ----------------------------------------------------------------- files

    def _write_media(self, relative: str, data: bytes) -> str:
        path = self.export / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files.append(path)
        return relative

    def _make_stickers(self) -> None:
        for number in range(self.options.sticker_kinds):
            fmt = ("GIF", "PNG", "JPEG")[number % 3]
            data = make_image_bytes(self.rng, fmt)
            md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
            local = self.options.media and self.rng.random() < self.options.local_sticker_ratio
            info = StickerInfo(
                md5=md5,
                url=f"https://stickers.example.test/{md5[:10]}/{number}",
                data=data,
                local=local,
                mime=f"image/{fmt.lower()}",
            )
            self.stickers[md5] = info
            if local:
                suffix = {"GIF": "gif", "PNG": "png", "JPEG": "jpg"}[fmt]
                self._write_media(f"media/emojis/{md5}.{suffix}", data)

    # -------------------------------------------------------------- messages

    def _sentence(self) -> str:
        text = self.rng.choice(self.pool)
        if self.options.keep_texts:
            self.texts.add(text)
        return text

    def _build(self, number: int, when: datetime) -> dict[str, Any]:
        rng = self.rng
        kind = rng.choices(self._kinds, self._weights)[0]
        sent = rng.random() < 0.5
        local_id = 1000 + number
        server = 7_000_000_000_000_000_000 + number * 7919
        text_time = when.astimezone(SOURCE_ZONE) + timedelta(hours=self.options.shift_text_hours)
        base = message(
            id=f"{self.index:02d}-{number:09d}",
            localId=local_id,
            serverId=str(server),
            createTime=int(when.timestamp()),
            createTimeText=text_time.strftime("%Y-%m-%d %H:%M:%S"),
            sortSeq=number * 1000,
            isSent=sent,
            senderUsername="wxid_me" + "0000" if sent else self.username,
            conversationUsername=self.username,
            isGroup=self.group,
            renderType=kind,
            type={"text": 1, "emoji": 47, "image": 3, "voice": 34, "video": 43}.get(kind, 49),
        )
        if sent:
            base["senderAvatarPath"] = "media/avatars/me.png"
        base.update(self._builders[kind](number, sent))
        return base

    def _text(self, number: int, sent: bool) -> dict[str, Any]:
        return message(content=self._sentence())

    def _emoji(self, number: int, sent: bool) -> dict[str, Any]:
        info = self.rng.choice(list(self.stickers.values()))
        fields = message(emojiMd5=info.md5, emojiUrl=info.url, content="")
        if info.local:
            suffix = {"image/gif": "gif", "image/png": "png", "image/jpeg": "jpg"}[info.mime]
            fields["offlineMedia"] = [
                message(
                    kind="emoji",
                    path=f"media/emojis/{info.md5}.{suffix}",
                    md5=info.md5,
                    fileId=f"f{info.md5[:8]}",
                )
            ]
        return fields

    def _quote(self, number: int, sent: bool) -> dict[str, Any]:
        return message(
            content=self._sentence(),
            quoteUsername=self.username,
            quoteServerId=str(6_000_000_000_000_000_000 + number),
            quoteType=1,
            quoteTitle=self.display_name,
            quoteContent=self.rng.choice(self.pool),
            quoteThumbUrl="",
            quoteVoiceLength=0,
        )

    def _image(self, number: int, sent: bool) -> dict[str, Any]:
        rng = self.rng
        md5 = hashlib.md5(f"{self.index}-{number}".encode(), usedforsecurity=False).hexdigest()
        fields = message(
            imageMd5=md5,
            imageFileId=f"img{number}",
            imageUrl=f"https://images.example.test/{md5}",
            content="",
        )
        if self.options.media and rng.random() >= self.options.missing_image_ratio:
            fmt = rng.choice(["PNG", "JPEG"])
            data = make_image_bytes(rng, fmt, size=rng.choice([16, 32, 48]))
            real_md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
            suffix = "png" if fmt == "PNG" else "jpg"
            path = self._write_media(f"media/images/{real_md5}.{suffix}", data)
            fields["offlineMedia"] = [
                message(kind="image", path=path, md5=real_md5, fileId=f"img{number}")
            ]
            fields["imageMd5"] = real_md5
            self.image_md5s.append(real_md5)
            self.image_paths.append(path)
        else:
            self.missing_ids.append(f"{self.index:02d}-{number:09d}")
        return fields

    def _voice(self, number: int, sent: bool) -> dict[str, Any]:
        transcribed = self.rng.random() < 0.6
        fields = message(
            voiceLength=str(self.rng.choice([800, 3200, 4600, 12000, 59000])),
            content="",
        )
        if transcribed:
            fields.update(
                voiceTranscript=self._sentence(),
                voiceTranscriptStatus="done",
                voiceTranscriptLanguage="zh",
                voiceTranscriptModel="synthetic-asr",
            )
        else:
            fields.update(voiceTranscriptStatus="none")
        return fields

    def _voip(self, number: int, sent: bool) -> dict[str, Any]:
        return message(content=VOIP_CONTENTS[number % len(VOIP_CONTENTS)], voipType=number % 2)

    def _system(self, number: int, sent: bool) -> dict[str, Any]:
        options = ("你撤回了一条消息", "对方撤回了一条消息", f"“{self.display_name}”拍了拍你")
        return message(content=options[number % len(options)])

    def _transfer(self, number: int, sent: bool) -> dict[str, Any]:
        return message(
            title="转账",
            amount="88.00",
            paySubType=1,
            transferStatus="received",
            transferId=f"t{number}",
            content="",
        )

    def _redpacket(self, number: int, sent: bool) -> dict[str, Any]:
        return message(title="红包", amount="6.66", paySubType=1, content="")

    def _link(self, number: int, sent: bool) -> dict[str, Any]:
        return message(
            title=self._sentence(),
            url=f"https://links.example.test/{number}",
            linkType="article",
            linkStyle=1,
            thumbUrl="",
            content="",
        )

    def _video(self, number: int, sent: bool) -> dict[str, Any]:
        fields = message(videoMd5=f"v{number}", videoThumbMd5=f"t{number}", content="")
        if self.options.media and self.rng.random() < 0.7:
            data = make_image_bytes(self.rng, "JPEG", size=32)
            path = self._write_media(f"media/images/cover{self.index}_{number}.jpg", data)
            fields["offlineMedia"] = [message(kind="video_thumb", path=path, md5=None, fileId="c")]
        return fields

    def _file(self, number: int, sent: bool) -> dict[str, Any]:
        return message(
            title=f"文件{number}.pdf",
            fileSize=1024 * (number % 50 + 1),
            fileMd5=f"f{number}",
            content="",
        )

    def _location(self, number: int, sent: bool) -> dict[str, Any]:
        return message(
            locationLat=41.0 + (number % 10) / 100,
            locationLng=-87.0 - (number % 10) / 100,
            locationPoiname=self._sentence(),
            locationLabel=self.rng.choice(self.pool),
            content="",
        )

    def _chathistory(self, number: int, sent: bool) -> dict[str, Any]:
        return message(title="聊天记录", recordItem="<recorditem/>", content="")

    def _holographic(self, number: int, sent: bool) -> dict[str, Any]:
        return message(content="", novelField=f"n{number}")

    def iter_messages(self) -> Iterator[dict[str, Any]]:
        when = self.options.start
        for number in range(1, self.count + 1):
            when = when + timedelta(
                seconds=self.rng.expovariate(1.0 / self.options.average_gap_s) + 1
            )
            item = self._build(number, when)
            if self.first is None:
                self.first = when
            self.last = when
            if self.options.keep_ids:
                self.ids.append(item["id"])
            self.counts[(item["renderType"], bool(item["isSent"]))] += 1
            yield item

    # ----------------------------------------------------------------- output

    def write(self, report_missing: list[dict[str, Any]]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._make_stickers()  # emoji messages need stickers even when no file is written
        header = message(
            schemaVersion=self.options.schema_version,
            exportedAt=int(self.options.exported_at.timestamp()),
            account=message(username="wxid_me" + "0000", displayName="我"),
            conversation=message(
                username=self.username,
                displayName=self.display_name,
                avatarPath="media/avatars/her.png",
                isGroup=self.group,
            ),
            filters=message(startTime=None, endTime=None, messageTypes=[]),
        )
        path = self.dir / "messages.json"
        with path.open("wb") as handle:
            handle.write(orjson.dumps(header)[:-1] + b',"messages":[')
            for position, item in enumerate(self.iter_messages()):
                if position:
                    handle.write(b",")
                handle.write(orjson.dumps(item))
            handle.write(b"]}")
        self.files.append(path)
        for number_id in self.missing_ids:
            report_missing.append(
                message(kind="image", id=number_id, conversation=self.username, messageId=number_id)
            )
        meta = message(
            schemaVersion=self.options.schema_version,
            username=self.username,
            displayName=self.display_name,
            avatarPath="media/avatars/her.png",
            isGroup=self.group,
            exportedAt=int(self.options.exported_at.timestamp()),
            messageCount=self.count,
        )
        meta_path = self.dir / "meta.json"
        meta_path.write_bytes(orjson.dumps(meta))
        self.files.append(meta_path)


def write_manifest(root: Path, style: str = "json", corrupt: tuple[str, ...] = ()) -> Path:
    """Write ``_integrity/`` for the files below ``root``; ``corrupt`` paths get a wrong digest."""
    folder = root / "_integrity"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and folder not in p.parents):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative in corrupt:
            digest = "0" * 64
        entries.append((relative, digest, path.stat().st_size))
    if style == "sha256sum":
        target = folder / "SHA256SUMS.txt"
        target.write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest, _ in entries), encoding="utf-8"
        )
    else:
        target = folder / "manifest.json"
        target.write_bytes(
            orjson.dumps(
                message(
                    schemaVersion=1,
                    algorithm="sha256",
                    files=[
                        message(path=relative, sha256=digest, size=size)
                        for relative, digest, size in entries
                    ],
                )
            )
        )
    return target


def build_export(
    root: Path,
    options: SynthOptions | None = None,
    *,
    progress: Callable[[int], None] | None = None,
) -> SynthExport:
    """Write a synthetic export below ``root`` and describe what was written."""
    opts = options or SynthOptions()
    rng = random.Random(opts.seed)
    pool = _sentence_pool(rng)
    root.mkdir(parents=True, exist_ok=True)
    (root / "conversations").mkdir(exist_ok=True)

    nickname = opts.target_display_name or "".join(SPECIAL_NICKNAME_PARTS[:4])
    target_username = opts.target_username or _wxid(rng)
    conversations: list[_Conversation] = [
        _Conversation(
            root,
            1,
            rng,
            opts,
            username=target_username,
            display_name=nickname,
            group=False,
            count=opts.target_messages,
            is_target=True,
            pool=pool,
        )
    ]
    others: list[str] = []
    for number in range(opts.other_conversations):
        username = _wxid(rng)
        others.append(username)
        conversations.append(
            _Conversation(
                root,
                2 + number,
                rng,
                opts,
                username=username,
                display_name=_other_nickname(number),
                group=False,
                count=opts.other_messages,
                is_target=False,
                pool=pool,
            )
        )
    group_username: str | None = None
    if opts.include_group:
        group_username = f"{rng.randint(10**9, 10**10 - 1)}" + "@chatroom"
        conversations.append(
            _Conversation(
                root,
                2 + opts.other_conversations,
                rng,
                opts,
                username=group_username,
                display_name="周末群聊" + SPECIAL_NICKNAME_PARTS[1],
                group=True,
                count=opts.other_messages,
                is_target=False,
                pool=pool,
            )
        )

    missing: list[dict[str, Any]] = []
    for conversation in conversations:
        conversation.write(missing)
        if progress is not None:
            progress(conversation.count)

    all_files = [path for c in conversations for path in c.files]
    if opts.media:
        avatars = [("her.png", "PNG"), ("me.png", "PNG")]
        for name, fmt in avatars:
            path = root / "media" / "avatars" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(make_image_bytes(rng, fmt, size=20))
            all_files.append(path)

    report = message(
        schemaVersion=opts.schema_version,
        exportId=opts.export_id,
        account=message(username="wxid_me" + "0000"),
        createdAt=int(opts.exported_at.timestamp()),
        missingMedia=missing,
        errors=[],
    )
    report_path = root / "report.json"
    report_path.write_bytes(orjson.dumps(report))
    all_files.append(report_path)
    if opts.integrity:
        write_manifest(root, opts.integrity, opts.corrupt)

    target = conversations[0]
    return SynthExport(
        root=root,
        options=opts,
        target_username=target_username,
        target_display_name=nickname,
        target_dir=target.dir,
        other_usernames=others,
        group_username=group_username,
        counts=target.counts,
        message_ids=target.ids,
        texts=target.texts,
        stickers=target.stickers,
        image_md5s=target.image_md5s,
        image_paths=target.image_paths,
        missing_message_ids=target.missing_ids,
        target_messages=opts.target_messages,
        first_time=target.first,
        last_time=target.last,
    )
