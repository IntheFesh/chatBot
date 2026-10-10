"""The sticker library with tags, descriptions and the manual commands (R-STK-002, R-STK-003)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.support.persona import sticker_scenario
from twin.services import Services
from twin.stickers.catalog import StickerCatalog, UnknownStickerError
from twin.stickers.tags import TagError
from twin.storage.chat_models import Sticker

NOW = datetime(2026, 5, 1, 12, tzinfo=UTC)


@pytest.fixture
def catalog(services: Services) -> StickerCatalog:
    sticker_scenario(services)
    return StickerCatalog(services)


def md5s(catalog: StickerCatalog) -> list[str]:
    return [r.md5 for r in catalog.records()]


def test_the_library_lists_her_most_used_first_and_filters(catalog: StickerCatalog) -> None:
    records = catalog.records()
    assert [r.her_uses for r in records] == [5, 4, 1, 1]
    assert catalog.records(limit=2) == records[:2]
    assert len(catalog.records(status="available")) == 4 and catalog.records(status="pending") == []
    assert len(catalog.records(her_only=True)) == 4
    assert len(catalog.records(tagged=False)) == 4 and catalog.records(tagged=True) == []
    first = records[0]
    assert first.available and not first.tagged and not first.usable
    assert first.mime == "image/png" and first.width == first.height == 64


def test_a_sticker_is_found_by_its_md5_or_a_unique_prefix(catalog: StickerCatalog) -> None:
    target = md5s(catalog)[0]
    assert catalog.get(target) is not None and catalog.get("0" * 32) is None
    assert catalog.resolve(target).md5 == target
    assert catalog.resolve(target[:8].upper()).md5 == target
    assert catalog.require(target).md5 == target
    with pytest.raises(UnknownStickerError, match="at least 4"):
        catalog.resolve("ab")
    with pytest.raises(UnknownStickerError, match="no sticker MD5 starts"):
        catalog.resolve("ffff")
    with pytest.raises(UnknownStickerError, match="no sticker with MD5"):
        catalog.require("0" * 32)
    with pytest.raises(UnknownStickerError):
        catalog.resolve("0" * 32)


def test_two_stickers_with_the_same_prefix_are_refused(
    catalog: StickerCatalog, services: Services
) -> None:
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=False) as session:
        for suffix in ("a", "b"):
            session.add(
                Sticker(
                    md5="dead" + suffix * 28,
                    status="available",
                    attempts=0,
                    her_uses=0,
                    user_uses=0,
                    context_uses=0,
                    disabled=False,
                    origin="import",
                    created_at=now,
                    updated_at=now,
                )
            )
    with pytest.raises(UnknownStickerError, match="more than one"):
        catalog.resolve("dead")


def test_the_picture_decides_until_something_better_arrives(catalog: StickerCatalog) -> None:
    md5 = md5s(catalog)[0]
    saved = catalog.save_vision(md5, ["开心", "大笑"], "一只笑着的猫", "回应好消息", at=NOW)
    assert saved.tags == ("开心", "大笑") and saved.tag_source == "vision"
    assert saved.description == "一只笑着的猫" and saved.use_cases == "回应好消息"
    assert saved.vision_tags == ("开心", "大笑") and saved.tagged_at == NOW
    assert saved.usable and catalog.counts()["tagged"] == 1 and catalog.counts()["described"] == 1
    with pytest.raises(TagError):
        catalog.save_vision(md5, ["不存在"], "x", "y", at=NOW)
    assert catalog.get(md5).vision_tags == ("开心", "大笑")  # type: ignore[union-attr]


def test_her_use_outweighs_the_picture_and_hand_set_tags_outweigh_both(
    catalog: StickerCatalog,
) -> None:
    md5 = md5s(catalog)[0]
    catalog.save_vision(md5, ["开心", "晚安"], "一只猫", "场合", at=NOW)
    corrected = catalog.save_context(
        md5, ["委屈"], "她用它表示有点委屈", uses=4, cutoff=NOW - timedelta(days=1), at=NOW
    )
    assert corrected.tags == ("委屈",) and corrected.tag_source == "context"
    assert corrected.context_note == "她用它表示有点委屈" and corrected.context_uses == 4
    manual = catalog.set_manual(md5, ["晚安", "困"])
    assert manual.tags == ("晚安", "困") and manual.tag_source == "manual"
    assert manual.vision_tags == ("开心", "晚安") and manual.context_tags == ("委屈",)
    back = catalog.clear_manual(md5)
    assert back.tags == ("委屈",) and back.tag_source == "context" and back.manual_tags == ()
    with pytest.raises(TagError):
        catalog.set_manual(md5, ["x"])
    with pytest.raises(ValueError, match="at least one tag"):
        catalog.set_manual(md5, [])
    assert catalog.counts()["manual"] == 0 and catalog.counts()["context_corrected"] == 1


def test_a_sticker_can_be_switched_off_and_on(catalog: StickerCatalog) -> None:
    md5 = md5s(catalog)[0]
    catalog.save_vision(md5, ["开心"], "d", "u", at=NOW)
    assert catalog.set_disabled(md5, True).disabled and not catalog.require(md5).usable
    assert catalog.counts()["disabled"] == 1
    assert catalog.set_disabled(md5, False).usable


def test_a_change_of_tags_makes_the_description_vector_stale(
    catalog: StickerCatalog, services: Services
) -> None:
    md5 = md5s(catalog)[0]
    catalog.save_vision(md5, ["开心"], "d", "u", at=NOW)
    with services.db.transaction(bump_state=False) as session:
        row = session.get(Sticker, md5)
        assert row is not None
        row.desc_encoding = "model|3|abc|sticker-v1"
        row.desc_vector_id = md5
    catalog.set_manual(md5, ["开心"])  # the same tags: still current
    assert catalog.require(md5).desc_encoding == "model|3|abc|sticker-v1"
    catalog.set_manual(md5, ["晚安"])
    assert catalog.require(md5).desc_encoding is None


def test_the_tag_of_a_sticker_is_looked_up_by_its_md5(catalog: StickerCatalog) -> None:
    a, b = md5s(catalog)[:2]
    catalog.save_vision(a, ["开心", "大笑"], "d", "u", at=NOW)
    lookup = catalog.tag_lookup()
    assert lookup(a) == "开心" and lookup(b) is None and lookup("0" * 32) is None


def test_a_correction_made_for_another_cutoff_is_dropped(catalog: StickerCatalog) -> None:
    a, b = md5s(catalog)[:2]
    cutoff = NOW
    for md5 in (a, b):
        catalog.save_vision(md5, ["开心"], "d", "u", at=NOW)
    catalog.save_context(a, ["委屈"], "m", uses=3, cutoff=cutoff, at=NOW)
    catalog.save_context(b, ["难过"], "m", uses=3, cutoff=cutoff - timedelta(days=3), at=NOW)
    assert catalog.clear_stale_context(cutoff) == 1
    kept, dropped = catalog.require(a), catalog.require(b)
    assert kept.tags == ("委屈",) and kept.context_tagged_at == NOW
    assert dropped.tags == ("开心",) and dropped.tag_source == "vision"
    assert dropped.context_tags == () and dropped.context_note is None
    assert dropped.context_tagged_at is None and dropped.context_cutoff_at is None
    assert dropped.context_uses == 0
    assert catalog.clear_stale_context(cutoff) == 0


def test_the_counts_describe_the_library(catalog: StickerCatalog) -> None:
    assert catalog.counts() == {
        "total": 4, "available": 4, "hers": 4, "hers_available": 4, "tagged": 0,
        "described": 0, "context_corrected": 0, "manual": 0, "disabled": 0, "with_vector": 0,
    }  # fmt: skip


def test_each_use_of_hers_is_tied_to_the_example_window_it_happened_in(
    catalog: StickerCatalog, services: Services
) -> None:
    from sqlalchemy import select

    from twin.retrieval.indexer import sync_windows
    from twin.storage.chat_models import StickerUse
    from twin.storage.retrieval_models import ExampleWindow

    md5 = md5s(catalog)[0]
    assert catalog.use_windows(md5) == {}  # the library of windows is not built yet
    sync_windows(services)
    found = catalog.use_windows(md5)
    with services.db.session() as session:
        uses = set(
            session.scalars(
                select(StickerUse.message_id).where(
                    StickerUse.sticker_md5 == md5, StickerUse.by_her.is_(True)
                )
            )
        )
        windows = {w.id: set(w.reply_block_ids) for w in session.scalars(select(ExampleWindow))}
    assert set(found) == uses and len(uses) == 5
    for message_id, window_id in found.items():
        assert message_id in windows[window_id]
    assert len(set(found.values())) == 5  # five episodes, five windows
    assert catalog.use_windows("0" * 32) == {}
