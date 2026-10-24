"""Token estimation (R-LLM-012) and the cache-friendly prompt layout (R-LLM-010)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.support.clock import ManualClock
from twin.llm.layout import CacheMonitor, PromptLayout
from twin.llm.tokens import Calibration, ImageTokenTable, TokenEstimator
from twin.llm.types import ChatMessage, Usage
from twin.storage.db import Database
from twin.storage.settings_store import get_setting, put_setting


def user(text: str) -> ChatMessage:
    return {"role": "user", "content": text}


def system(text: str) -> ChatMessage:
    return {"role": "system", "content": text}


def assistant(text: str) -> ChatMessage:
    return {"role": "assistant", "content": text}


# ------------------------------------------------------------------- estimation


def test_character_classes_are_weighted_differently() -> None:
    est = TokenEstimator()
    assert est.estimate_text("") == 0
    chinese = est.raw_text("你好世界今天吃什么")
    assert chinese == pytest.approx(9 * 0.6)
    latin = est.raw_text("hello world")
    assert latin == pytest.approx(2 * 5 * 0.3 + 0)  # two words of five letters
    assert est.raw_text("a") == 1.0  # a word is never below one token
    assert est.raw_text("12345678") == pytest.approx(4.0)
    assert est.raw_text("，。！") == pytest.approx(3.0)  # punctuation: one token each
    assert est.raw_text("   \n  ") == 0.0
    assert est.raw_text("😀") == pytest.approx(1.5)
    assert est.estimate_text("你") == 1  # at least one token for non-empty text


def test_message_estimates_add_overhead_and_images() -> None:
    est = TokenEstimator()
    plain = est.estimate_messages([user("你好")])
    assert plain == pytest.approx(3 + 1.2, abs=1)
    with_image = est.estimate_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看图"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                ],
            }
        ]
    )
    assert with_image >= 1024
    assert est.estimate_image() == 1024


def test_measured_image_sizes_replace_the_documented_cap() -> None:
    table = ImageTokenTable([(64 * 64, 16), (512 * 512, 260), (2048 * 2048, 1024)])
    est = TokenEstimator(image_tokens=table)
    assert est.estimate_image(60, 60) == 16
    assert est.estimate_image(500, 540) == 260
    assert est.estimate_image(4000, 4000) == 1024
    assert est.estimate_image(None, None) == 1024  # unknown size: the cap
    assert not ImageTokenTable() and bool(table)
    est.set_image_table(ImageTokenTable([(100, 5000)]))
    assert est.estimate_image(10, 10) == 1024  # never above the documented cap


def test_calibration_converges_to_the_real_ratio() -> None:
    est = TokenEstimator(alpha=0.2)
    messages = [user("你好世界今天吃什么" * 20)]
    raw = est.estimate_messages(messages)
    actual = int(raw * 1.5)
    for _ in range(40):
        est.observe(messages, actual)
    assert est.factor == pytest.approx(1.5, abs=0.05)
    assert est.calibration.samples == 40
    assert abs(est.estimate_messages(messages) - actual) <= 0.05 * actual


def test_the_first_observation_sets_the_factor_and_outliers_are_clamped() -> None:
    est = TokenEstimator()
    messages = [user("hello there my friend")]
    est.observe(messages, 100_000)
    assert est.factor == 3.0  # clamped to the maximum
    est2 = TokenEstimator()
    est2.observe(messages, 1)
    assert est2.factor >= 0.3


def test_image_tokens_are_taken_out_before_calibrating() -> None:
    est = TokenEstimator()
    messages: list[ChatMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello hello hello hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        }
    ]
    text_only = TokenEstimator().raw_text("hello hello hello hello") + 3
    est.observe(messages, 1024 + round(text_only))
    assert est.factor == pytest.approx(1.0, abs=0.05)


def test_observations_with_impossible_counts_are_ignored() -> None:
    est = TokenEstimator()
    assert est.observe([user("")], 0) == Calibration()
    assert est.observe([user("hello")], -5) == Calibration()
    assert est.calibration.samples == 0


def test_invalid_alpha_is_rejected() -> None:
    with pytest.raises(ValueError, match="alpha"):
        TokenEstimator(alpha=0)


def test_calibration_is_persisted_in_settings(db: Database, clock: ManualClock) -> None:
    est = TokenEstimator()
    est.observe([user("你好世界" * 10)], 100)
    with db.transaction(bump_state=False) as session:
        assert est.save(session, clock)
        assert not est.save(session, clock)  # unchanged
    fresh = TokenEstimator()
    with db.session() as session:
        loaded = fresh.load(session)
    assert loaded == est.calibration and fresh.factor == est.factor


def test_invalid_stored_calibration_is_ignored(db: Database, clock: ManualClock) -> None:
    fresh = TokenEstimator()
    with db.transaction(bump_state=False) as session:
        put_setting(session, "llm.token_calibration", {"factor": "x"}, clock=clock)
    with db.session() as session:
        assert fresh.load(session) == Calibration()
    with db.transaction(bump_state=False) as session:
        put_setting(session, "llm.token_calibration", {"factor": 99.0, "samples": 3}, clock=clock)
    with db.session() as session:
        assert fresh.load(session) == Calibration()
        assert get_setting(session, "llm.token_calibration")["factor"] == 99.0


@given(st.text(max_size=300))
def test_estimates_are_never_negative_and_never_crash(text: str) -> None:
    est = TokenEstimator()
    assert est.raw_text(text) >= 0
    assert est.estimate_text(text) >= (1 if text else 0)


# ----------------------------------------------------------------------- layout


def make_layout(history: list[ChatMessage], tail_text: str) -> PromptLayout:
    return PromptLayout.of([system("rules and persona"), *history], [user(tail_text)])


def test_layout_keeps_prefix_and_tail_apart_and_orders_them() -> None:
    layout = make_layout([user("a"), assistant("b")], "now")
    assert [m["content"] for m in layout.messages()] == ["rules and persona", "a", "b", "now"]
    assert len(layout.stable_prefix) == 3 and len(layout.variable_tail) == 1


def test_prefix_hash_ignores_the_tail_and_reflects_the_prefix() -> None:
    one = make_layout([user("a")], "first question")
    two = make_layout([user("a")], "a different question")
    three = make_layout([user("a2")], "first question")
    assert one.prefix_hash == two.prefix_hash
    assert one.prefix_hash != three.prefix_hash
    assert len(one.prefix_hash) == 64


def test_the_next_request_extends_the_previous_prefix_until_the_window_moves() -> None:
    history: list[ChatMessage] = [user("q1"), assistant("a1")]
    first = make_layout(history, "q2")
    history = [*history, user("q2"), assistant("a2")]
    second = make_layout(history, "q3")
    assert second.extends(first)
    shifted = make_layout(history[2:], "q4")  # the batch window moved forward
    assert not shifted.extends(second)
    changed = PromptLayout.of([system("rules and a NEW persona"), *history], [user("q3")])
    assert not changed.extends(second)
    assert first.extends(first)


def test_prefix_tokens_are_estimated_from_the_prefix_only() -> None:
    est = TokenEstimator()
    layout = make_layout([user("你好" * 50)], "x")
    assert layout.prefix_tokens(est) == est.estimate_messages(layout.stable_prefix)
    assert layout.prefix_tokens(est) > est.estimate_messages(layout.variable_tail)


def test_cache_monitor_reports_hit_ratios_and_prefix_changes() -> None:
    est = TokenEstimator()
    monitor = CacheMonitor(est)
    layout = make_layout([user("你好" * 100)], "q")
    prefix = layout.prefix_tokens(est)
    first = monitor.observe(
        layout, Usage(cache_hit_tokens=0, cache_miss_tokens=prefix + 5), purpose="reply"
    )
    assert not first.prefix_changed and first.hit_ratio == 0.0 and first.prefix_hit_ratio == 0.0
    warm = monitor.observe(
        layout, Usage(cache_hit_tokens=prefix, cache_miss_tokens=5), purpose="reply"
    )
    assert warm.prefix_hit_ratio == 1.0 and warm.hit_ratio == pytest.approx(prefix / (prefix + 5))
    other = make_layout([user("别的" * 100)], "q")
    changed = monitor.observe(
        other, Usage(cache_hit_tokens=0, cache_miss_tokens=100), purpose="reply"
    )
    assert changed.prefix_changed
    summary = monitor.summary("reply")
    assert summary.calls == 3 and summary.prefix_changes == 1
    assert summary.hit_tokens == prefix and summary.hit_ratio == pytest.approx(
        prefix / (prefix + prefix + 5 + 5 + 100)
    )
    assert 0 < summary.mean_prefix_hit_ratio < 1


def test_cache_monitor_separates_purposes_and_handles_no_data() -> None:
    monitor = CacheMonitor(TokenEstimator(), window=2)
    layout = make_layout([], "q")
    usage = Usage(cache_hit_tokens=10, cache_miss_tokens=10)
    for _ in range(5):
        monitor.observe(layout, usage, purpose="reply")
    monitor.observe(layout, usage, purpose="plan")
    assert monitor.summary("reply").calls == 2  # bounded window
    assert monitor.summary().calls == 3
    assert monitor.summary("summary").calls == 0 and monitor.summary("summary").hit_ratio == 0.0
    empty = monitor.observe(PromptLayout.of([], []), Usage(), purpose="x")
    assert empty.hit_ratio == 0.0 and empty.prefix_hit_ratio == 0.0
