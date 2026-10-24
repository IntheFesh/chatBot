"""Keeping the share of stickers near hers (R-STK-005)."""

from __future__ import annotations

import pytest

from tests.support.synth_chat import ChatSpec, build_chat
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.profile.api import load_profile
from twin.profile.builder import rebuild
from twin.services import Services
from twin.stickers.rate import BotTurnBubbleHistory, MemoryBubbleHistory, StickerRateController

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


def send(
    store: BotTurnStore, services: Services, flags: list[bool], *, command: bool = False
) -> None:
    """Store one reply per flag: a sticker bubble (``True``) or a text bubble."""
    for flag in flags:
        services.clock.now_utc()  # the manual clock only moves when a test asks it to
        bubble = (
            OutboundBubble("[表情包:开心]", services.clock.now_utc(), "sticker", "a" * 32)
            if flag
            else OutboundBubble("好", services.clock.now_utc())
        )
        store.add_bubble(bubble, meta=ReplyMeta("deepseek"), is_command=command)
        services.clock.tick(1)  # type: ignore[attr-defined]


def test_the_window_is_read_from_bot_turns_and_keeps_only_the_last_bubbles(
    services: Services,
) -> None:
    store = BotTurnStore(services.db, services.clock)
    history = BotTurnBubbleHistory(services.db, 5)
    assert history.flags() == []
    send(store, services, [True, False, False, True, False, False, False])
    again = BotTurnBubbleHistory(services.db, 5)  # a new process sees the same bubbles
    assert again.flags() == [False, True, False, False, False]
    assert BotTurnBubbleHistory(services.db, 3).flags() == [False, False, False]
    with pytest.raises(ValueError, match="window"):
        BotTurnBubbleHistory(services.db, 0)


def test_commands_and_thrown_away_replies_are_not_in_the_window(services: Services) -> None:
    store = BotTurnStore(services.db, services.clock)
    send(store, services, [True, False])
    send(store, services, [True, True], command=True)
    send(store, services, [True])
    rejected = store.latest_reply()
    assert store.reject_reply(rejected[0].reply_id or "") == 1
    assert BotTurnBubbleHistory(services.db, 10).flags() == [True, False]


def test_a_history_that_fills_itself_cannot_be_recorded_into(services: Services) -> None:
    rate = StickerRateController(
        HER_SHARE, BotTurnBubbleHistory(services.db, 200), window=200, tolerance=0.2
    )
    with pytest.raises(TypeError, match="bot_turns"):
        rate.record(True)


def test_stickers_chosen_for_the_same_reply_count_towards_the_limit() -> None:
    # 24 of the last 200 bubbles are stickers (12 %); one more would make 12.5 %: allowed
    rate = controller([False] * 176 + [True] * 24)
    assert not rate.should_drop() and not rate.should_drop_after(())
    # a sticker already chosen for this reply makes the next one 13 %: over her share plus 20 %
    assert rate.should_drop_after([True])
    assert not rate.should_drop_after([False, False])  # text bubbles do not use up the room


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


def test_the_default_window_comes_from_bot_turns_and_has_the_configured_size(
    chat: Services,
) -> None:
    store = BotTurnStore(chat.db, chat.clock)
    size = chat.settings.stickers.rate_window
    send(store, chat, [True] * (size + 5))
    rate = StickerRateController.from_services(chat)
    assert rate.status().bubbles == size == 200
    assert rate.should_drop()
    assert StickerRateController.from_services(chat).status().bubbles == size  # another process


def test_a_controller_without_a_profile_never_drops(services: Services) -> None:
    rate = StickerRateController.from_services(
        services, history=MemoryBubbleHistory(200, [True] * 200)
    )
    assert rate.her_share is None and not rate.should_drop()
