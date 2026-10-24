"""A rare share is judged only on enough expected events (R-EVAL-002, R-SRV-005; R18-E5, D-852).

The user decided: a share whose expected events - her rate times the bot's denominator - are
fewer than 50 is "not enough", never a pass or a failure; the M5 style sample is sized so that
the rarest share reaches 50; the live report carries a share over to earlier weeks until it has
enough.  The +-30 % is unchanged.
"""

from __future__ import annotations

import io
import random
from datetime import timedelta
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rich.console import Console

from tests.support.export_world import World
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.eval.render import Candidate, CandidateLine
from twin.eval.report import print_style_report
from twin.eval.style_metrics import (
    MIN_EXPECTED_EVENTS,
    STYLE_METRICS,
    TOLERANCE,
    MetricReading,
    StyleReport,
    candidate_recs,
    compare,
    expected_per_reply,
    her_units_per_reply,
    judge_metric,
    measure,
    replies_needed,
    style_of_live,
)
from twin.eval.style_sample import PLAN_MARGIN
from twin.profile.distribution import EmpiricalDistribution
from twin.profile.snapshot import ProfileMetrics, assemble_metrics
from twin.profile.values import Dist, Leaf, Rates, Scalar

BURST_GAP, SEGMENT_GAP = 120.0, 60.0
SPEC = {spec.key: spec for spec in STYLE_METRICS}
NO_QUOTES = {"supports_quote": False}
# her reference rates (SPEC section 0): a comma in 2.9 % of her texts, an emoji code in 3.5 %, a
# sticker in 10.5 % of her messages, a quote in 6.4 % of her texts
HER = {"comma": 0.029, "code": 0.035, "sticker": 0.105, "quote": 0.064}
PLAIN, COMMA, CODE = "你好呀你好呀你好", "你好，呀你好呀你", "你好呀你好呀你好[拥抱]"
MD5 = "ab" * 16


def profile(*, burst: float = 3.0, length: float = 8.0, **rates: float) -> ProfileMetrics:
    """Her profile with exactly these values (the way the profile builder assembles it)."""
    found = {**HER, **rates}

    def dist(value: float) -> Dist:
        return Dist(EmpiricalDistribution.from_samples([value] * 50, discrete=True))

    leaves: dict[str, Leaf] = {
        "text_length": dist(length),
        "burst_size": dist(burst),
        "punct_rate": Rates({"comma": found["comma"], "question": 0.1}, 5000),
        "emoji_code_rate": Scalar(found["code"], 5000),
        "sticker_share": Scalar(found["sticker"], 6000),
        "quote_rate": Scalar(found["quote"], 5000),
    }
    return ProfileMetrics(
        assemble_metrics(
            scope="pre_holdout",
            config={},
            weight=0.0,
            full={"her": leaves, "user": {}},
            recent=None,
            full_info={},
            recent_info={},
        )
    )


def copy_of_her(rng: random.Random, count: int) -> list[Candidate]:
    """Replies drawn from her very rates: three messages each, any of them a sticker or a text."""
    replies = []
    for index in range(count):
        lines = []
        for position in range(3):
            if rng.random() < HER["sticker"]:
                lines.append(CandidateLine("sticker", sticker_md5=f"{index * 3 + position:032x}"))
            elif rng.random() < HER["comma"]:
                lines.append(CandidateLine("text", text=COMMA))
            elif rng.random() < HER["code"] / (1 - HER["comma"]):
                lines.append(CandidateLine("text", text=CODE))
            else:
                lines.append(CandidateLine("text", text=PLAIN))
        replies.append(Candidate(tuple(lines), None))
    return replies


# ------------------------------------------------------------------------------- the rule


def test_r18_e5_a_share_with_fewer_than_fifty_expected_events_is_insufficient_not_failed() -> None:
    """R18-E5 (D-852): "not enough" instead of a verdict on noise; the +-30 % is unchanged."""
    assert MIN_EXPECTED_EVENTS == 50 and TOLERANCE == 0.30
    comma, length = SPEC["comma_rate"], SPEC["text_length"]
    hers = MetricReading(0.029, 10_000)
    few = judge_metric(comma, hers, MetricReading(0.0, 100), min_expected=50)
    assert few.status == "insufficient" and few.expected == pytest.approx(2.9) and few.counts
    many = judge_metric(comma, hers, MetricReading(0.0, 2_000), min_expected=50)
    assert many.status == "fail" and many.expected == pytest.approx(58.0)
    assert judge_metric(comma, hers, MetricReading(0.029, 1_725), min_expected=50).status == "pass"
    edge = judge_metric(comma, hers, MetricReading(0.029, 1_724), min_expected=50)
    assert edge.status == "insufficient"  # 49.996 expected events: not 50
    never = judge_metric(comma, MetricReading(0.0, 900), MetricReading(0.0, 10), min_expected=50)
    assert never.status == "pass"  # she never writes a comma: nothing to expect, the bot must not
    assert judge_metric(comma, hers, MetricReading(0.0, 100)).status == "fail"  # no rule asked
    median = judge_metric(length, MetricReading(8.0, 900), MetricReading(8.0, 3), min_expected=50)
    assert median.status == "pass" and median.expected is None  # a median has no events
    report = StyleReport("eval_items", "pre_holdout", [few, median])
    assert report.verdict == "insufficient" and not report.passed
    assert report.insufficient == [few]
    off = judge_metric(length, MetricReading(8.0, 900), MetricReading(20.0, 900))
    assert StyleReport("live", "live", [few, off]).verdict == "insufficient"  # said first
    assert StyleReport("live", "live", [many, median]).verdict == "failed"
    assert StyleReport("live", "live", [median]).verdict == "passed"
    assert StyleReport("live", "live", []).verdict == "insufficient"  # nothing measured


@settings(max_examples=200, deadline=None)
@given(
    rate=st.floats(min_value=0.001, max_value=1.0),
    n=st.integers(min_value=0, max_value=5_000),
    measured=st.floats(min_value=0.0, max_value=1.0),
)
def test_r18_e5_a_share_is_insufficient_exactly_when_her_rate_times_the_bots_count_is_short(
    rate: float, n: int, measured: float
) -> None:
    """R18-E5 (D-852), any rate and count: below 50 expected events, never a pass or a fail."""
    result = judge_metric(
        SPEC["sticker_share"],
        MetricReading(rate, 10_000),
        MetricReading(measured if n else None, n),
        min_expected=MIN_EXPECTED_EVENTS,
    )
    assert result.expected == pytest.approx(rate * n)
    if rate * n < MIN_EXPECTED_EVENTS:
        assert result.status == "insufficient"
    else:
        within = abs(measured - rate) / rate <= TOLERANCE + 1e-9
        assert result.status == ("pass" if within else "fail")


# -------------------------------------------------------------------------- the sample size


def test_r18_e5_the_sample_is_sized_by_the_rarest_share_she_shows() -> None:
    """R18-E5 (D-852): about 700 replies at her rates (her comma rate decides)."""
    hers = profile(burst=2.5)
    units = her_units_per_reply(hers)
    assert units["sticker_share"] == pytest.approx(2.5)
    assert units["comma_rate"] == pytest.approx(2.5 * (1 - 0.105))
    per_reply = expected_per_reply(hers, NO_QUOTES, units)
    assert set(per_reply) == {"comma_rate", "sticker_share", "emoji_code_rate"}  # no quotes
    assert per_reply["comma_rate"] == pytest.approx(0.029 * 2.5 * 0.895)
    needed = replies_needed(hers, NO_QUOTES, units)
    assert needed == 771 and 0.029 * 2.5 * 0.895 * needed >= MIN_EXPECTED_EVENTS
    assert 0.029 * 2.5 * 0.895 * (needed - 1) < MIN_EXPECTED_EVENTS
    # a channel that quotes adds the quote rate (6.4 %, not the rarest); a share she never shows
    # expects nothing and the next rarest decides
    assert "quote_rate" in expected_per_reply(hers, {"supports_quote": True}, units)
    without_commas = profile(burst=2.5, comma=0.0)
    assert replies_needed(without_commas, NO_QUOTES, units) == pytest.approx(
        50 / (0.035 * 2.5 * 0.895), abs=1
    )
    assert replies_needed(profile(comma=0.0, code=0.0, sticker=0.0), NO_QUOTES, units) == 50


def test_r18_e5_a_copy_of_her_style_passes_on_the_sized_sample_far_more_often() -> None:
    """R18-E5 (D-852): 200 simulated samples of a model that writes exactly like her.

    On the 65 contexts of a blind test (``eval.blind_n``) the three rare shares together pass
    the +-30 % in a small minority of the samples (the audit found about a quarter); on the
    sample sized for 50 expected events of the rarest share, nearly always - and never one of
    them is "not enough".
    """
    hers = profile()
    needed = replies_needed(hers, NO_QUOTES, her_units_per_reply(hers))
    sized = int(needed * PLAN_MARGIN) + 1
    rng = random.Random(18)
    new_rule = old_rule = short = 0
    for _ in range(200):
        measured = measure(candidate_recs(copy_of_her(rng, sized)), BURST_GAP, SEGMENT_GAP)
        report = StyleReport(
            "eval_items",
            "pre_holdout",
            compare(hers, measured, channel=NO_QUOTES, min_expected=MIN_EXPECTED_EVENTS),
        )
        new_rule += report.verdict == "passed"
        short += report.verdict == "insufficient"
        small = measure(candidate_recs(copy_of_her(rng, 65)), BURST_GAP, SEGMENT_GAP)
        old_rule += StyleReport(
            "eval_items", "pre_holdout", compare(hers, small, channel=NO_QUOTES)
        ).passed
    assert new_rule >= 170, new_rule  # at least 85 %
    assert old_rule <= 100, old_rule  # at most half on a blind test's 65 contexts
    assert short == 0


# ------------------------------------------------------------------------------ the live weeks


def her_live_profile(monkeypatch: pytest.MonkeyPatch, **rates: float) -> ProfileMetrics:
    found = profile(burst=3.0, length=8.0, **rates)
    monkeypatch.setattr(
        "twin.eval.style_metrics.load_profile",
        lambda services, scope: SimpleNamespace(metrics=found) if scope == "live" else None,
    )
    return found


def write_week(world: World, weeks_ago: int, replies: int) -> None:
    """``replies`` replies of the bot in one week: a text with a comma, a plain text, a sticker."""
    services = world.services
    store = BotTurnStore(services.db, services.clock)
    start = services.clock.now_utc() - timedelta(days=7 * weeks_ago + 6, hours=1)
    for number in range(replies):
        at = start + timedelta(hours=8 * number)
        store.add_inbound(at=at, kind="text", text=f"问{number}")
        store.add_reply(
            [
                OutboundBubble(COMMA, at + timedelta(seconds=5)),
                OutboundBubble(PLAIN, at + timedelta(seconds=7)),
                OutboundBubble("", at + timedelta(seconds=9), kind="sticker", sticker_md5=MD5),
            ],
            ReplyMeta("deepseek"),
        )


async def test_r18_e5_the_live_report_carries_a_short_share_over_whole_weeks(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-E5 (D-852): a share short of events in its week is judged on the weeks before too.

    Each week the bot writes 15 replies (30 texts, half with a comma; 45 messages, a third
    stickers): 15 expected commas and 15 expected stickers a week - four weeks reach 60.
    """
    her_live_profile(monkeypatch, comma=0.5, sticker=1 / 3, code=0.0)
    for weeks_ago in range(6):
        write_week(world, weeks_ago, 15)
    report = style_of_live(world.services, 7)
    by_key = {result.key: result for result in report.results}
    for key in ("comma_rate", "sticker_share"):
        assert by_key[key].status == "pass" and by_key[key].days == 28, by_key[key]
        assert by_key[key].expected == pytest.approx(60.0, abs=0.5)
    assert by_key["emoji_code_rate"].days is None  # she shows none: nothing to carry over
    assert by_key["text_length"].days is None and report.days == 7
    assert report.verdict == "passed" and report.min_expected == MIN_EXPECTED_EVENTS
    assert any("judged on the last 28 days" in note for note in report.notes)
    data = report.to_json()
    assert data["verdict"] == "passed" and data["min_expected"] == 50
    assert {m["key"]: m["days"] for m in data["metrics"]}["comma_rate"] == 28


async def test_r18_e5_a_conversation_too_short_for_a_share_stays_not_enough(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-E5 (D-852): with no older week left the share is judged on all of it: not enough."""
    her_live_profile(monkeypatch, comma=0.5, sticker=1 / 3, code=0.0)
    write_week(world, 0, 15)
    write_week(world, 1, 15)
    report = style_of_live(world.services, 7)
    comma = next(result for result in report.results if result.key == "comma_rate")
    assert comma.status == "insufficient" and comma.days == 14
    assert comma.expected == pytest.approx(30.0, abs=0.5)
    assert report.verdict == "insufficient" and not report.passed
    assert any("still not enough" in note for note in report.notes)
    buffer = io.StringIO()
    print_style_report(Console(file=buffer, width=240, color_system=None), report)
    shown = buffer.getvalue()
    assert "期望事件数" in shown and "30.0（最近 14 天）" in shown
    assert "比例类指标要有 ≥ 50 次期望事件" in shown
    assert "样本不足：逗号率（期望事件 30.0（最近 14 天））" in shown
    assert "不通过" not in shown  # not enough is not a failure
