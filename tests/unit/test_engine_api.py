"""What the engine package offers other rounds, and the small shapes it passes around."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from twin.engine import api
from twin.engine.types import (
    Bubble,
    InboundItem,
    PostAction,
    ReplyContext,
    ReplyDraft,
    UsageSummary,
    Violation,
)

AT = datetime(2026, 10, 9, 18, 30, tzinfo=UTC)


def test_everything_the_api_lists_exists() -> None:
    assert len(api.__all__) == len(set(api.__all__))
    for name in api.__all__:
        assert hasattr(api, name), name
    assert {"ReplyPipeline", "ReplyDataView", "BotTurnStore", "fit_bubbles_to_quota"} <= set(
        api.__all__
    )


def test_a_round_needs_something_to_answer() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        ReplyContext(inbound=())
    context = ReplyContext(
        inbound=(InboundItem("a", AT, "text", "你好"), InboundItem("b", AT, "image", "[图片]"))
    )
    assert context.user_text == "你好\n[图片]" and context.last_inbound.id == "b"
    assert context.last_inbound.turn_id is None  # set once the message is stored


def test_usage_adds_up_and_reports_the_cache_share() -> None:
    first = UsageSummary(1, 100, 10, 60, 40)
    second = UsageSummary(1, 200, 20, 0, 200)
    total = first.plus(second)
    assert (total.calls, total.prompt_tokens, total.completion_tokens) == (2, 300, 30)
    assert total.cache_hit_ratio == pytest.approx(60 / 300)
    assert UsageSummary().cache_hit_ratio == 0.0


def test_a_draft_exposes_what_the_engine_stores() -> None:
    draft = ReplyDraft(
        bubbles=(Bubble("text", "好呀"), Bubble("sticker", "[表情包:开心]", "a" * 32, "开心")),
        quote=None,
        no_reply=False,
        needs_fallback=False,
        fallback_reason=None,
        backend="deepseek",
        thinking=False,
        reasoning=None,
        plan=None,
        cost_usd=0.1,
        usage=UsageSummary(),
        timings_ms={"total": 5},
        actions=(PostAction("dedupe", 2), PostAction("quote_removed", detail="channel")),
        violations=(Violation("commitment", "pattern#3"),),
        attempts=1,
    )
    assert draft.usable and draft.texts == ("好呀", "[表情包:开心]")
    assert draft.actions_json() == (
        {"step": "dedupe", "count": 2},
        {"step": "quote_removed", "count": 1, "detail": "channel"},
    )
    assert draft.violations[0].to_json() == {"kind": "commitment", "detail": "pattern#3"}
    assert Violation("empty").to_json() == {"kind": "empty"}
    meta = draft.to_meta()
    assert (meta.backend, meta.thinking, meta.cost_usd) == ("deepseek", False, 0.1)
    assert meta.timings_ms == {"total": 5} and meta.actions == draft.actions_json()
    assert draft.to_meta("fallback").backend == "fallback"
