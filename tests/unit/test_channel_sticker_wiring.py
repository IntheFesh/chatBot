"""The channel only sends what the sticker library vouches for (R-SAFE-006, R-CH-006)."""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, datetime

import pytest

from tests.fixtures.synth_export import make_image_bytes
from twin.channel.base import MediaNotAllowed
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.policy import ProbeImageManifest, ensure_media_allowed
from twin.services import Services
from twin.stickers.library import StickerLibraryAllowList, store_sticker_file
from twin.storage.chat_models import Sticker
from twin.storage.media import MediaKind

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def picture(seed: int) -> bytes:
    return make_image_bytes(random.Random(seed), "PNG")


def add_available_sticker(services: Services, data: bytes) -> str:
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    with services.db.transaction() as session:
        row = Sticker(md5=md5, status="pending", attempts=0, her_uses=1, user_uses=0)
        session.add(row)
        session.flush()
        store_sticker_file(row, data, services.media, NOW)
        assert row.status == "available"
    return hashlib.sha256(data).hexdigest()


def test_the_allow_list_vouches_only_for_available_sticker_files(services: Services) -> None:
    good = picture(1)
    digest = add_available_sticker(services, good)
    allow = StickerLibraryAllowList(services.db)
    assert allow.allowed(digest)
    assert not allow.allowed(hashlib.sha256(b"a photo, not a sticker").hexdigest())
    assert not allow.allowed("not-a-digest")


def test_a_mismatching_sticker_file_is_kept_but_never_allowed(services: Services) -> None:
    data = picture(2)
    with services.db.transaction() as session:
        row = Sticker(md5="e" * 32, status="pending", attempts=0, her_uses=1, user_uses=0)
        session.add(row)
        session.flush()
        store_sticker_file(row, data, services.media, NOW)
        assert row.status == "md5_mismatch"
    assert not StickerLibraryAllowList(services.db).allowed(hashlib.sha256(data).hexdigest())


def test_the_default_channel_policy_is_library_stickers_plus_probe_pictures(
    services: Services,
) -> None:
    sticker = picture(3)
    photo = picture(4)
    add_available_sticker(services, sticker)
    # a photo stored as ordinary media is in the media store but not in the sticker library
    services.media.put(photo, MediaKind.IMAGE)

    channel = IlinkChannel.from_services(services, poll=False)
    assert (
        ensure_media_allowed(channel.media_policy, sticker) == hashlib.sha256(sticker).hexdigest()
    )
    with pytest.raises(MediaNotAllowed):
        ensure_media_allowed(channel.media_policy, photo)

    probe = picture(5)
    ProbeImageManifest(channel.state).register_bytes(probe)
    assert ensure_media_allowed(channel.media_policy, probe)
