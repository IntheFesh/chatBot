"""Metric values, blending, differences and the phrase counters (R-PROF-002 to R-PROF-004)."""

from __future__ import annotations

import pytest

from twin.profile.diffing import Change, diff_metrics, flatten, summarize
from twin.profile.distribution import BucketedDistribution, EmpiricalDistribution
from twin.profile.phrases import MAX_TRACKED, PhraseCollector, _prune, phrase_counts
from twin.profile.snapshot import PARTIES, ProfileMetrics, assemble_metrics
from twin.profile.values import (
    Dist,
    Hourly,
    Leaf,
    Rates,
    Scalar,
    Table,
    blend_leaf,
    leaf_from_json,
)


def metrics_of(her: dict[str, Leaf], recent: dict[str, Leaf] | None = None) -> ProfileMetrics:
    document = assemble_metrics(
        scope="live",
        config={"recency_weight": 0.6},
        weight=0.6,
        full={"her": her, "user": {}},
        recent=None if recent is None else {"her": recent, "user": {}},
        full_info={"days": 3},
        recent_info={"days": 1},
    )
    return ProfileMetrics(document)


def test_every_kind_of_value_survives_json() -> None:
    dist = EmpiricalDistribution.from_counter({1: 5, 9: 5}, discrete=True)
    leaves: list[Leaf] = [
        Scalar(0.25, 400, 7),
        Rates({"a": 0.5, "b": 0.25}, 40),
        Dist(dist),
        Hourly(BucketedDistribution.from_counters({3: {5.0: 40}}, sizes=(1,), min_samples=10)),
        Table({"md5": {"n": 3, "last": "2026-01-01T00:00:00Z"}}, 3),
    ]
    for leaf in leaves:
        again = leaf_from_json(leaf.to_json())
        assert type(again) is type(leaf) and again.to_json() == leaf.to_json()
    with pytest.raises(ValueError, match="unknown metric value tag"):
        leaf_from_json({"t": "zzz"})


def test_blending_follows_the_recency_weight_for_every_kind() -> None:
    full, recent = Scalar(0.10, 500), Scalar(0.20, 100)
    blended = blend_leaf(full, recent, 0.6)
    assert isinstance(blended, Scalar) and blended.value == pytest.approx(0.16)
    assert blend_leaf(full, Scalar(0.2, 5), 0.6) is full  # too thin to trust
    assert blend_leaf(full, recent, 0.0) is full and blend_leaf(full, None, 0.6) is full
    assert blend_leaf(full, Rates({"a": 1.0}, 50), 0.6) is full  # mismatched kinds
    rates = blend_leaf(Rates({"a": 0.8, "b": 0.2}, 100), Rates({"a": 0.2, "c": 0.8}, 100), 0.5)
    assert isinstance(rates, Rates)
    assert rates.values == pytest.approx({"a": 0.5, "b": 0.1, "c": 0.4})
    assert blend_leaf(Rates({"a": 1.0}, 100), Rates({"a": 0.0}, 3), 0.5).n == 100
    slow = Dist(EmpiricalDistribution.from_counter({100.0: 80}))
    fast = Dist(EmpiricalDistribution.from_counter({10.0: 80}))
    mixed = blend_leaf(slow, fast, 0.7)
    assert isinstance(mixed, Dist) and 10 <= mixed.dist.median() <= 100
    assert blend_leaf(slow, Dist(EmpiricalDistribution.empty()), 0.7) is slow
    layout = BucketedDistribution.from_counters({2: {1.0: 50}}, sizes=(1,), min_samples=10)
    hourly = blend_leaf(Hourly(layout), Hourly(layout), 0.5)
    assert isinstance(hourly, Hourly) and hourly.dist.get(2).median() == 1.0
    table = Table({"x": {"n": 1}}, 1)
    assert blend_leaf(table, Table({"y": {"n": 1}}, 1), 0.6) is table


def test_the_document_keeps_raw_and_blended_values_and_no_recent_table() -> None:
    document = metrics_of(
        {"rate": Scalar(0.1, 500), "stickers": Table({"m": {"n": 1}}, 1)},
        {"rate": Scalar(0.3, 200), "stickers": Table({"m": {"n": 9}}, 9)},
    )
    entry = document.data["parties"]["her"]["rate"]
    assert (entry["full"]["v"], entry["recent"]["v"]) == (0.1, 0.3)
    assert entry["blended"]["v"] == pytest.approx(0.22)
    assert set(document.data["parties"]["her"]["stickers"]) == {"full"}
    assert document.table("her", "stickers") == {"m": {"n": 1}}
    assert document.scalar("her", "rate", "recent") == 0.3
    assert document.scalar("her", "missing") is None and document.leaf("her", "missing") is None
    assert document.window_info("recent") == {"days": 1} and document.names("her") == [
        "rate",
        "stickers",
    ]
    assert document.hourly("her", "rate") is None and document.table("her", "rate") == {}
    assert document.distribution("her", "rate") is None and document.rates("her", "rate") == {}
    assert PARTIES == ("her", "user")
    with pytest.raises(ValueError, match="unsupported profile schema"):
        ProfileMetrics({"schema": 99})


def test_a_scalar_without_data_reads_as_missing() -> None:
    document = metrics_of({"rate": Scalar(0.0, 0)})
    assert document.scalar("her", "rate") is None


def test_differences_are_relative_and_have_a_floor() -> None:
    before = metrics_of(
        {
            "comma": Scalar(0.030, 500),
            "tiny": Scalar(0.001, 500),
            "length": Dist(EmpiricalDistribution.from_counter({5: 100}, discrete=True)),
            "codes": Rates({"[拥抱]": 0.5, "[亲亲]": 0.5}, 80),
        }
    )
    after = metrics_of(
        {
            "comma": Scalar(0.036, 500),  # +20 %
            "tiny": Scalar(0.002, 500),  # doubles, but half a point of a rate at most
            "length": Dist(EmpiricalDistribution.from_counter({5: 100}, discrete=True)),
            "codes": Rates({"[拥抱]": 0.5, "[亲亲]": 0.2, "[捂脸]": 0.3}, 80),
        }
    )
    names = {(c.party, c.metric): c for c in diff_metrics(before, after)}
    assert set(names) == {("her", "comma"), ("her", "codes.[亲亲]"), ("her", "codes.[捂脸]")}
    assert names[("her", "comma")].ratio == pytest.approx(0.2)
    assert names[("her", "codes.[捂脸]")].ratio == float("inf")
    assert diff_metrics(before, before) == []
    lines = summarize(list(names.values()), limit=2)
    assert len(lines) == 2 and lines[0].startswith("她 ")
    assert flatten(before)[("her", "length.median")] == 5
    assert "新出现" in names[("her", "codes.[捂脸]")].describe()


def test_a_change_names_and_describes_itself() -> None:
    change = Change("her", "punct_rate.comma", 0.029, 0.052)
    assert change.label() == "标点使用率·逗号"
    assert change.describe() == "她 标点使用率·逗号：2.9% → 5.2%（+79%）"
    assert Change.from_json(change.to_json()) == Change("her", "punct_rate.comma", 0.029, 0.052)
    length = Change("user", "text_length.median", 11.0, 14.0)
    assert length.label() == "文字长度·中位数" and "11.0 → 14.0" in length.describe()
    assert Change("her", "unknown_metric", 1.0, 3.0).label() == "unknown_metric"
    assert Change("her", "x", 0.0, 0.0).ratio == 0.0


# ------------------------------------------------------------------- phrases


def test_phrase_counters_keep_frequent_text_and_drop_rare_text() -> None:
    collector = PhraseCollector()
    for _ in range(12):
        collector.feed("晚安宝宝")
    for _ in range(6):
        collector.feed("宝宝，吃饭了吗")
    collector.feed("只说一次的话")
    result = collector.result()
    assert result["messages"] == 19
    assert ["晚安宝宝", 12] in result["sentences"]
    assert all(count >= 5 for _, count in result["sentences"])
    assert not any(text == "只说一次的话" for text, _ in result["sentences"])
    bigrams = dict(map(tuple, result["ngrams"]["2"]))  # type: ignore[arg-type]
    assert bigrams["晚安"] == 12 and bigrams["宝宝"] >= 18
    terms = {item["term"] for item in result["address_candidates"]}
    assert "宝宝" in terms
    counts = phrase_counts(result)
    assert counts["sentences"] == len(result["sentences"]) and "ngrams_2" in counts


def test_texts_are_split_into_sentences_and_emoji_are_not_part_of_grams() -> None:
    collector = PhraseCollector()
    for _ in range(6):
        collector.feed("好的！我马上来。[拥抱]😂")
    sentences = dict(map(tuple, collector.result()["sentences"]))  # type: ignore[arg-type]
    assert sentences.get("好的") == 6 and sentences.get("我马上来") == 6
    assert all(
        "[" not in gram for rows in collector.result()["ngrams"].values() for gram, _ in rows
    )


def test_counters_are_pruned_to_a_bound_without_losing_the_frequent_entries() -> None:
    from collections import Counter

    counter: Counter[str] = Counter({"常见": 50, "偶见": 2, "稀有一": 1, "稀有二": 1})
    _prune(counter, 2)
    assert set(counter) == {"常见", "偶见"}  # the singletons went first
    _prune(counter, 1)
    assert set(counter) == {"常见"}
    assert MAX_TRACKED > 1000
    small = PhraseCollector(limit=3)
    for index in range(40):
        small.feed(f"独特的句子{index}号")
    for _ in range(9):
        small.feed("反复出现的句子")
    assert ["反复出现的句子", 9] in small.result()["sentences"]
    assert len(small.sentences) <= 2 * 3
