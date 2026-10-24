"""Scenarios for the persona card and sticker tests (round 06).

Every scenario is written straight into the database with known regularities, so that a test can
say exactly which messages a sampler may read, which sticker uses fall before the hold-out
cutoff, and which sentences must never reach the model.  Nothing here is a real conversation.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from PIL import Image
from sqlalchemy import func, select

from tests.support.embedding import H, Msg, U, day, write_dialogue
from twin.services import Services
from twin.stickers.library import store_sticker_file
from twin.storage.chat_models import Message, Sticker, StickerUse

LATE_MARKER = "稍后才说的秘密句"  # only ever said after the cutoff
EARLY_MARKER = "很早就说过的句子"


def sticker_png(index: int) -> bytes:
    """A small distinct picture per index (the colour differs)."""
    image = Image.new("RGB", (64, 64), ((index * 53) % 256, (index * 97) % 256, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def sticker_files(count: int) -> dict[str, bytes]:
    """``md5 -> picture`` for ``count`` stickers; the md5 is the real md5 of the bytes."""
    files = {}
    for index in range(count):
        data = sticker_png(index + 1)
        files[hashlib.md5(data, usedforsecurity=False).hexdigest()] = data
    return files


def attach_files(services: Services, files: dict[str, bytes], *, strict: bool = True) -> None:
    """Store the pictures in the media store and mark the stickers available.

    A picture whose sticker no message uses has no row; that is an error unless ``strict`` is off.
    """
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=False) as session:
        for md5, data in files.items():
            row = session.get(Sticker, md5)
            if row is None:
                assert not strict, md5
                continue
            store_sticker_file(row, data, services.media, now)


def sync_counters(services: Services) -> None:
    """Set the use counters of the stickers from ``sticker_uses`` (the importer does this)."""
    with services.db.transaction(bump_state=False) as session:
        for sticker in session.scalars(select(Sticker)):
            for by_her in (True, False):
                count, first, last = session.execute(
                    select(
                        func.count(), func.min(StickerUse.used_at), func.max(StickerUse.used_at)
                    ).where(StickerUse.sticker_md5 == sticker.md5, StickerUse.by_her.is_(by_her))
                ).one()
                if by_her:
                    sticker.her_uses = int(count)
                    sticker.first_used_at = first
                    sticker.last_used_at = last
                else:
                    sticker.user_uses = int(count)


@dataclass
class Scenario:
    """A conversation with stickers; ``uses`` maps the sticker md5s to episode numbers."""

    files: dict[str, bytes]
    md5s: list[str]
    uses: dict[str, list[int]]
    episodes: int
    starts: list[datetime] = field(default_factory=list)

    def episode_time(self, number: int) -> datetime:
        return self.starts[number]


def episode_messages(number: int, stickers: dict[int, str]) -> list[Msg]:
    """One short exchange; episode ``number`` may contain her sticker."""
    messages: list[Msg] = [
        U(f"今天的第{number}件事你听说了吗"),
        H(f"听说了呀{number}"),
    ]
    if number in stickers:
        messages.append(U(f"那你开心吗{number}"))
        messages.append(H(kind="sticker", md5=stickers[number]))
    messages.append(H(f"好啦晚点聊{number}"))
    return messages


def sticker_scenario(
    services: Services, uses: dict[int, str] | None = None, *, episodes: int = 40, stickers: int = 4
) -> Scenario:
    """40 exchanges, one a day; her stickers appear in the episodes of ``uses``.

    Default: sticker 0 in episodes 2, 5, 9, 14 and 38 (four before the cutoff, one after),
    sticker 1 in 3, 7, 11 (three before) and 37, sticker 2 in 20 (one before), sticker 3 only in
    39 (first used after the cutoff).
    """
    files = sticker_files(stickers)
    md5s = list(files)
    plan = uses or {
        2: md5s[0], 5: md5s[0], 9: md5s[0], 14: md5s[0], 38: md5s[0],
        3: md5s[1], 7: md5s[1], 11: md5s[1], 37: md5s[1],
        20: md5s[2],
        39: md5s[3],
    }  # fmt: skip
    episodes_data = [(day(n, 12), episode_messages(n, plan)) for n in range(episodes)]
    starts = write_dialogue(services, episodes_data)
    attach_files(services, files, strict=False)
    sync_counters(services)
    by_sticker: dict[str, list[int]] = {}
    for number, md5 in sorted(plan.items()):
        by_sticker.setdefault(md5, []).append(number)
    return Scenario(files, md5s, by_sticker, episodes, starts)


def month_scenario(services: Services, months: int = 6, per_month: int = 6) -> list[datetime]:
    """``months`` x ``per_month`` exchanges spread over months, hours and lengths.

    Even hours of the day and several lengths and intensities, so that a sampler has something
    to stratify.  Returns the start time of every episode.
    """
    episodes: list[tuple[datetime, list[Msg]]] = []
    hours = (3, 9, 15, 21)
    base = datetime(2026, 1, 1, 12, tzinfo=day(0).tzinfo)
    for month in range(months):
        for index in range(per_month):
            start = base + timedelta(days=31 * month + 3 * index, hours=hours[index % 4] - 12)
            length = 2 + (index % 3) * 4
            messages: list[Msg] = [U(f"问题{month}-{index}")]
            for turn in range(length):
                loud = ("", "！", "！？")[(index // 2) % 3]
                messages.append(H(f"回答{month}-{index}-{turn}{loud}"))
                messages.append(U(f"嗯{turn}", gap=40))
            episodes.append((start, messages))
    return write_dialogue(services, episodes)


def message_ids_after(services: Services, moment: datetime) -> set[str]:
    with services.db.session() as session:
        return set(session.scalars(select(Message.id).where(Message.create_time_utc >= moment)))
