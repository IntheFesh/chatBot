"""Style metrics of the bot against her profile (R-EVAL-002, R-PROF-002)."""

from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

import pytest

from tests.support.export_world import World
from twin.engine.conversation_log import conversation_since
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.eval.render import Candidate, CandidateLine
from twin.eval.store import EvalStore, NewItem
from twin.eval.style_metrics import (
    STYLE_METRICS,
    TOLERANCE,
    MetricReading,
    StyleError,
    StyleReport,
    candidate_recs,
    compare,
    deviation_of,
    judge_metric,
    measure,
    read_metrics,
    style_of_items,
    style_of_live,
)
from twin.profile.distribution import EmpiricalDistribution
from twin.profile.snapshot import ProfileMetrics, assemble_metrics
from twin.profile.values import Dist, Leaf, Rates, Scalar
from twin.services import Services

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"
BURST_GAP, SEGMENT_GAP = 120.0, 60.0


def reply(index: int, *, comma: bool = False, code: bool = False, quote: bool = False,
          sticker: bool = False) -> Candidate:  # fmt: skip
    """Two text lines of eight characters (and a sticker): the lines carry the asked-for marks."""
    first = "你好呀你好呀你，好" if comma else "你好呀你好呀你好"
    second = "哈哈哈哈哈哈[拥抱]" if code else "哈哈哈哈哈哈哈哈"
    lines = [CandidateLine("text", text=first), CandidateLine("text", text=second)]
    if sticker:
        lines.append(CandidateLine("sticker", sticker_md5=f"{index:032x}"))
    return Candidate(tuple(lines), "被引用的话" if quote else None)


def known_replies() -> list[Candidate]:
    """20 replies: 12 with 2 messages and 8 with 3; the counts below are what the test expects.

    * text lines: 40, all eight characters long (the second line with a code is one longer
      than eight, so the code lines are nine: the median is still eight);
    * comma in 4 of 40 texts (0.10); emoji code in 8 of 40 texts (0.20);
    * stickers: 8 among 48 messages (1/6); quotes: 5 among 40 text replies (0.125).
    """
    return [
        reply(
            n,
            comma=n < 4,
            code=4 <= n < 12,
            quote=n % 4 == 0 and n < 20,
            sticker=n >= 12,
        )
        for n in range(20)
    ]


def profile_of(
    *,
    length: float = 8.0,
    burst: float = 2.0,
    comma: float = 0.10,
    sticker: float = 1 / 6,
    code: float = 0.20,
    quote: float = 0.125,
) -> ProfileMetrics:
    """A profile document with exactly these values for her (the way the builder assembles it)."""

    def dist(value: float) -> Dist:
        return Dist(EmpiricalDistribution.from_samples([value] * 50, discrete=True))

    leaves: dict[str, Leaf] = {
        "text_length": dist(length),
        "burst_size": dist(burst),
        "punct_rate": Rates({"comma": comma, "question": 0.1}, 200),
        "emoji_code_rate": Scalar(code, 200),
        "sticker_share": Scalar(sticker, 300),
        "quote_rate": Scalar(quote, 200),
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


def test_the_six_core_metrics_are_measured_by_the_profile_collector() -> None:
    measured = measure(candidate_recs(known_replies()), BURST_GAP, SEGMENT_GAP)
    found = read_metrics(measured)
    assert found["text_length"] == MetricReading(8.0, 40)
    assert found["burst_size"].value == 2.0 and found["burst_size"].n == 20
    assert found["comma_rate"].value == pytest.approx(0.10) and found["comma_rate"].n == 40
    assert found["emoji_code_rate"].value == pytest.approx(0.20)
    assert found["sticker_share"].value == pytest.approx(8 / 48, abs=1e-5)
    assert found["sticker_share"].n == 48
    assert found["quote_rate"].value == pytest.approx(5 / 40)
    assert [spec.key for spec in STYLE_METRICS] == [
        "text_length",
        "comma_rate",
        "burst_size",
        "sticker_share",
        "emoji_code_rate",
        "quote_rate",
    ]


def test_nothing_in_the_style_report_counts_a_comma_itself() -> None:
    """R-EVAL-002: the profile's own functions measure; this module only reads and compares."""
    tree = ast.parse((SRC / "eval" / "style_metrics.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("twin.profile")
        for alias in node.names
    }
    assert {"WindowCollector", "assemble_metrics", "ProfileMetrics"} <= imported
    assert not {"re", "regex"} & {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def test_a_profile_equal_to_the_output_passes_every_metric() -> None:
    measured = measure(candidate_recs(known_replies()), BURST_GAP, SEGMENT_GAP)
    results = compare(profile_of(), measured, live=False)
    assert [r.status for r in results] == ["pass"] * 6
    assert all(r.deviation == pytest.approx(0.0, abs=1e-6) for r in results)


@pytest.mark.parametrize(
    ("reference", "measured", "status"),
    [
        (1.0, 1.30, "pass"),  # exactly +30 % is inside
        (1.0, 1.31, "fail"),
        (1.0, 0.70, "pass"),  # exactly -30 % is inside
        (1.0, 0.69, "fail"),
        (0.0, 0.0, "pass"),  # hers is zero and so is the bot's
        (0.0, 0.01, "fail"),  # hers is zero and the bot's is not: undefined, never a pass
        (None, 0.5, "no_data"),
        (0.5, None, "no_data"),
    ],
)
def test_a_metric_passes_within_thirty_percent_and_the_boundary_is_inclusive(
    reference: float | None, measured: float | None, status: str
) -> None:
    spec = STYLE_METRICS[0]  # a median: not a proportion, so any positive value is possible
    result = judge_metric(spec, MetricReading(reference, 100), MetricReading(measured, 100))
    assert result.status == status
    assert TOLERANCE == 0.30


def test_the_deviation_is_relative_to_hers() -> None:
    assert deviation_of(0.2, 0.3) == pytest.approx(0.5)
    assert deviation_of(0.2, 0.1) == pytest.approx(-0.5)
    assert deviation_of(0.0, 0.0) == 0.0 and deviation_of(0.0, 0.1) is None
    assert deviation_of(None, 0.1) is None and deviation_of(0.1, None) is None


def test_a_known_bias_is_found_and_ranked_first() -> None:
    """Stickers at twice the share she uses them, everything else right."""
    measured = measure(candidate_recs(known_replies()), BURST_GAP, SEGMENT_GAP)
    results = compare(profile_of(sticker=1 / 12), measured, live=False)
    failing = [r.key for r in results if r.status == "fail"]
    assert failing == ["sticker_share"]
    sticker = next(r for r in results if r.key == "sticker_share")
    assert sticker.deviation == pytest.approx(1.0, abs=1e-4)  # +100 %
    report = StyleReport("eval_items", "pre_holdout", results)
    assert not report.passed and report.worst[0].key == "sticker_share"
    assert StyleReport("live", "live", [r for r in results if r.key != "sticker_share"]).passed
    assert not StyleReport("live", "live", []).passed  # nothing measured is not a pass


def test_the_quote_rate_is_not_measurable_where_the_channel_cannot_quote() -> None:
    measured = measure(candidate_recs([reply(n) for n in range(10)]), BURST_GAP, SEGMENT_GAP)
    results = compare(profile_of(), measured, live=True)
    quote = next(r for r in results if r.key == "quote_rate")
    assert quote.status == "n/a" and not quote.counts and quote.deviation is None
    assert all(r.counts for r in results if r.key != "quote_rate")
    other = compare(profile_of(), measured, live=False)
    assert next(r for r in other if r.key == "quote_rate").status == "fail"  # 0 against 12.5 %


def test_proportions_carry_a_wilson_interval_and_medians_do_not() -> None:
    measured = measure(candidate_recs(known_replies()), BURST_GAP, SEGMENT_GAP)
    by_key = {r.key: r for r in compare(profile_of(), measured, live=False)}
    low, high = by_key["comma_rate"].interval or (1.0, 0.0)
    assert low < 0.10 < high and by_key["text_length"].interval is None


async def test_the_items_source_reads_one_backends_replies_next_to_her_real_ones(
    world: World,
) -> None:
    services = world.services
    store = EvalStore(services.db, services.clock)
    run = store.create_run("blind", mode="holdout", backends=["deepseek", "style"])
    real = reply(1, comma=True)
    items = [
        NewItem(
            f"k{n}",
            backend,
            world.cutoff,
            {"real": real.to_json()},
            None,
            "night",
            "short",
            n % 2 == 0,
        )
        for backend in ("deepseek", "style")
        for n in range(6)
    ]
    store.add_items(run.id, items)
    for item in store.items(run.id):
        bot = known_replies()[item.seq] if item.backend == "deepseek" else reply(99)
        store.save_generated(item.id, {"bot": bot.to_json()}, cost_usd=0.0)
    report = style_of_items(services, store, run, "deepseek")
    assert report.source == "eval_items" and report.scope == "pre_holdout"
    assert report.backend == "deepseek" and report.run_id == run.id
    assert report.messages == sum(len(c.lines) for c in known_replies()[:6])
    assert len(report.results) == 6 and all(r.real is not None for r in report.results)
    comma = next(r for r in report.results if r.key == "comma_rate")
    assert comma.real is not None and comma.real.value == pytest.approx(
        0.5
    )  # her real: 1 of 2 lines
    with pytest.raises(StyleError, match="no generated replies"):
        style_of_items(services, store, run, "hybrid")
    gate = store.create_run("gate", status="done", milestone="M1", verdict="failed")
    with pytest.raises(StyleError, match="need a blind run"):
        style_of_items(services, store, gate, "deepseek")


async def test_the_live_source_measures_what_the_bot_really_sent_in_the_last_days(
    world: World,
) -> None:
    services = world.services
    now = services.clock.now_utc()
    bot = BotTurnStore(services.db, services.clock)
    meta = ReplyMeta("deepseek")
    for number in range(8):
        at = now - timedelta(days=1, minutes=number * 10)
        bot.add_inbound(at=at, kind="text", text=f"问题{number}")
        bot.add_reply(
            [
                OutboundBubble("你好呀你好呀你好", at + timedelta(seconds=5)),
                OutboundBubble("哈哈哈[拥抱]", at + timedelta(seconds=8)),
            ],
            meta,
        )
    old = now - timedelta(days=30)
    bot.add_reply([OutboundBubble("很久以前的话", old)], meta)
    bot.add_reply([OutboundBubble("/状态的回复", now - timedelta(hours=2))], meta, is_command=True)
    report = style_of_live(services, 7)
    assert report.source == "live" and report.scope == "live" and report.days == 7
    assert report.messages == 16  # sixteen bubbles: the old one and the command reply are out
    by_key = {r.key: r for r in report.results}
    assert by_key["quote_rate"].status == "n/a"
    assert by_key["text_length"].measured.value == pytest.approx(8.0, abs=2.0)
    assert by_key["emoji_code_rate"].measured.value == pytest.approx(0.5)
    wider = style_of_live(services, 60)
    assert wider.messages == 17
    with pytest.raises(StyleError, match="at least 1"):
        style_of_live(services, 0)


def test_the_conversation_log_gives_plain_values_in_time_order_without_commands(
    world: World,
) -> None:
    services = world.services
    now = services.clock.now_utc()
    bot = BotTurnStore(services.db, services.clock)
    asked = now - timedelta(hours=3)
    bot.add_inbound(at=asked, kind="text", text="问")
    bot.add_reply([OutboundBubble("答", asked + timedelta(seconds=4))], ReplyMeta("deepseek"))
    bot.add_reply(
        [OutboundBubble("/状态的回复", now - timedelta(hours=2))],
        ReplyMeta("deepseek"),
        is_command=True,
    )
    log = conversation_since(services, now - timedelta(hours=4))
    assert [(t.outbound, t.kind, t.text) for t in log] == [
        (False, "text", "问"),
        (True, "text", "答"),
    ]
    assert all(t.sticker_md5 is None for t in log)
    assert conversation_since(services, now) == []


def test_a_fresh_installation_has_no_profile_to_compare_with(services: Services) -> None:
    with pytest.raises(StyleError, match="no live profile"):
        style_of_live(services, 7)
    run = EvalStore(services.db, services.clock).create_run("blind", backends=["deepseek"])
    with pytest.raises(StyleError, match="no pre-holdout profile"):
        style_of_items(services, EvalStore(services.db, services.clock), run, "deepseek")
