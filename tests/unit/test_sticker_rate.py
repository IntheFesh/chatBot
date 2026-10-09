"""Keeping the share of stickers near hers (R-STK-005)."""

from __future__ import annotations

import pytest

from tests.support.synth_chat import ChatSpec, build_chat
from twin.profile.api import load_profile
from twin.profile.builder import rebuild
from twin.services import Services
from twin.stickers.rate import (
    HISTORY_KEY,
    MemoryBubbleHistory,
    StickerRateController,
    StoredBubbleHistory,
)
from twin.storage.settings_store import get_setting

HER_SHARE = 0.105  # 10.5 % of her messages are stickers (SPEC section 0)


def controller(flags: list[bool], *, share: float | None = HER_SHARE) -> StickerRateController:
    history = MemoryBubbleHistory(200, flags)
    return StickerRateController(share, history, window=200, tolerance=0.2)


def window(stickers: int, *, oldest_is_sticker: bool = False) -> list[bool]:
    """A full window of 200 bubbles with ``stickers`` of them stickers."""
    flags = [False] * 200
    placed = 0
    if oldest_is_sticker:
        flags[0] = True
        placed = 1
    for index in range(1, 200):
        if placed >= stickers:
            break
        flags[index] = True
        placed += 1
    assert sum(flags) == stickers
    return flags


def test_the_limits_are_her_share_plus_and_minus_the_tolerance() -> None:
    rate = controller([])
    assert rate.her_share == HER_SHARE
    assert rate.upper == pytest.approx(0.126) and rate.lower == pytest.approx(0.084)


def test_a_new_conversation_is_judged_as_if_the_window_were_at_her_share() -> None:
    rate = controller([])
    assert rate.share() == pytest.approx(HER_SHARE)
    assert rate.share_if_added() == pytest.approx((1 + HER_SHARE * 199) / 200)
    assert not rate.should_drop()  # the first sticker is allowed
    status = rate.status()
    assert status.state == "ok" and status.bubbles == 0


def test_one_more_sticker_is_dropped_when_it_would_pass_the_upper_limit() -> None:
    at_limit = controller(window(24))  # 24 of 200 = 12 %; one more is 12.5 %, below 12.6 %
    assert not at_limit.should_drop()
    over = controller(window(25))  # 25 of 200 = 12.5 %; one more is 13 %
    assert over.should_drop()
    assert over.status().state == "ok" and over.share_if_added() == pytest.approx(0.13)


def test_the_oldest_bubble_leaves_the_window_when_a_new_one_arrives() -> None:
    # the oldest bubble is a sticker: it makes room for the new one, the share does not grow
    assert not controller(window(25, oldest_is_sticker=True)).should_drop()
    assert controller(window(26, oldest_is_sticker=True)).should_drop()  # 26 of 200 is over 12.6 %
    assert controller(window(27)).should_drop()


def test_the_share_is_reported_as_low_ok_or_high() -> None:
    assert controller(window(0)).status().state == "low"
    assert controller(window(20)).status().state == "ok"
    assert controller(window(40)).status().state == "high"
    status = controller(window(40)).status()
    assert status.share == pytest.approx(0.2) and status.bubbles == 200
    assert status.lower == pytest.approx(0.084) and status.upper == pytest.approx(0.126)


def test_a_low_share_never_forces_a_sticker() -> None:
    rate = controller(window(0))
    assert rate.status().state == "low" and not rate.should_drop()
    assert not hasattr(rate, "should_insert") and not hasattr(rate, "must_add")


def test_a_woman_who_never_sends_stickers_gets_none() -> None:
    rate = controller([False] * 10, share=0.0)
    assert rate.should_drop() and rate.upper == 0.0
    assert controller([], share=0.0).should_drop()


def test_without_her_profile_there_is_nothing_to_compare_with() -> None:
    rate = controller(window(100), share=None)
    assert not rate.should_drop() and rate.upper is None and rate.lower is None
    assert rate.status().state == "unknown"


def test_sent_bubbles_are_recorded_in_the_window() -> None:
    rate = controller([])
    for _ in range(30):
        rate.record(True)
    assert rate.should_drop() and rate.status().bubbles == 30
    for _ in range(200):
        rate.record(False)
    assert not rate.should_drop() and rate.share() == 0.0
    memory = MemoryBubbleHistory(3)
    for flag in (True, False, True, True):
        memory.record(flag)
    assert memory.flags() == [False, True, True]


def test_the_window_and_tolerance_must_make_sense() -> None:
    history = MemoryBubbleHistory(5)
    with pytest.raises(ValueError, match="window"):
        StickerRateController(0.1, history, window=0, tolerance=0.2)
    with pytest.raises(ValueError, match="tolerance"):
        StickerRateController(0.1, history, window=5, tolerance=1.0)


def test_the_stored_window_survives_a_restart_and_keeps_only_the_last_bubbles(
    services: Services,
) -> None:
    first = StoredBubbleHistory(services.db, services.clock, 5)
    assert first.flags() == []
    for flag in (True, False, False, True, False, False, False):
        first.record(flag)
    again = StoredBubbleHistory(services.db, services.clock, 5)  # a new process
    assert again.flags() == [False, True, False, False, False]
    with services.db.session() as session:
        assert get_setting(session, HISTORY_KEY) == [0, 1, 0, 0, 0]
    assert StoredBubbleHistory(services.db, services.clock, 3).flags() == [False, False, False]


@pytest.fixture
def chat(services: Services) -> Services:
    build_chat(services, ChatSpec(days=40))
    rebuild(services, "all")
    return services


def test_her_share_comes_from_her_profile_of_the_scope(chat: Services) -> None:
    live = StickerRateController.from_services(chat)
    profile = load_profile(chat, "live")
    assert profile is not None
    assert live.her_share == profile.metrics.scalar("her", "sticker_share")
    assert live.her_share is not None and 0.05 < live.her_share < 0.2
    past = StickerRateController.from_services(chat, scope="pre_holdout")
    past_profile = load_profile(chat, "pre_holdout")
    assert past_profile is not None
    assert past.her_share == past_profile.metrics.scalar("her", "sticker_share")
    config = chat.settings.stickers
    assert live.upper == pytest.approx(live.her_share * (1 + config.rate_tolerance))


def test_the_default_window_is_stored_and_has_the_configured_size(chat: Services) -> None:
    rate = StickerRateController.from_services(chat)
    for _ in range(205):
        rate.record(True)
    assert rate.status().bubbles == chat.settings.stickers.rate_window == 200
    again = StickerRateController.from_services(chat)
    assert again.status().bubbles == 200 and again.should_drop()


def test_a_controller_without_a_profile_never_drops(services: Services) -> None:
    rate = StickerRateController.from_services(
        services, history=MemoryBubbleHistory(200, [True] * 200)
    )
    assert rate.her_share is None and not rate.should_drop()
