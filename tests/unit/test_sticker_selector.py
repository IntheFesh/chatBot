"""Choosing a sticker for a tag (R-STK-004, R-TRN-013)."""

from __future__ import annotations

import random
from collections import Counter
from datetime import UTC, datetime

import pytest

from tests.support.embedding import H, HashingBackend, Msg, U, day, write_dialogue
from tests.support.persona import attach_files, sticker_files, sticker_scenario, sync_counters
from twin.ingest.corpus import her_bubble_skeleton
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.stickers.selector import (
    LIVE_VIEW,
    NEUTRAL_SIMILARITY,
    RECENCY_FLOOR,
    StickerSelector,
    StickerView,
    UsageTable,
    as_of_view,
    her_repeat_rate,
)
from twin.stickers.vectors import StickerVectors

NOW = datetime(2026, 5, 1, 12, tzinfo=UTC)
MD5S = list(sticker_files(6))  # the pictures of the scenarios, in a fixed order
A, B, C, D, E, F = MD5S

PLAN = {
    **dict.fromkeys((2, 4, 6, 8, 10, 12), A),  # six uses
    3: B,  # one use
    **dict.fromkeys((5, 7), C),  # two uses
    30: D,  # a late favourite of another kind
    38: E,  # first used after the hold-out cutoff
}
SPARSE = {2: A, 9: A, 16: A, 4: B, 11: B, 18: C}  # never twice within ten bubbles


def make(
    services: Services, plan: dict[int, str], tags: dict[str, tuple[str, str]]
) -> StickerCatalog:
    sticker_scenario(services, plan, episodes=40, stickers=6)
    catalog = StickerCatalog(services)
    for md5, (tag, description) in tags.items():
        catalog.save_vision(md5, [tag], description, "场合", at=NOW)
    return catalog


@pytest.fixture
def library(services: Services) -> StickerCatalog:
    return make(
        services,
        PLAN,
        {
            A: ("开心", "笑得很开心的黄猫"),
            B: ("开心", "月亮晚安睡觉的小熊"),
            C: ("开心", "狗狗在吃饭"),
            D: ("晚安", "星空下睡觉的兔子"),
            E: ("晚安", "夜空里的猫头鹰"),
        },
    )


def selector(services: Services, **kwargs: object) -> StickerSelector:
    return StickerSelector(services, rng=random.Random(7), **kwargs)  # type: ignore[arg-type]


def test_without_a_context_the_score_follows_her_use_with_add_one_smoothing(
    library: StickerCatalog, services: Services
) -> None:
    ranked = selector(services).rank("开心", "")
    assert [c.record.md5 for c in ranked] == [A, C, B]
    total, pool = 6 + 2 + 1, 3
    assert [c.uses for c in ranked] == [6, 2, 1]
    assert ranked[0].frequency == pytest.approx(7 / (total + pool))
    assert ranked[1].frequency == pytest.approx(3 / (total + pool))
    assert ranked[2].frequency == pytest.approx(2 / (total + pool))
    assert sum(c.frequency for c in ranked) == pytest.approx(1.0)  # a smoothed distribution
    assert all(c.similarity == NEUTRAL_SIMILARITY for c in ranked)


def test_frequency_and_similarity_work_together(
    library: StickerCatalog, services: Services, embedder: HashingBackend
) -> None:
    StickerVectors(services).sync()
    picker = selector(services)
    for md5, text in ((A, "笑得很开心的黄猫"), (B, "月亮晚安睡觉的小熊"), (C, "狗狗在吃饭")):
        ranked = picker.rank("开心", text)
        own = next(c for c in ranked if c.record.md5 == md5)
        others = [c for c in ranked if c.record.md5 != md5]
        assert all(own.similarity > c.similarity for c in others), md5
        assert [c.score for c in ranked] == sorted((c.score for c in ranked), reverse=True)
        for candidate in ranked:  # the three factors multiply
            assert candidate.score == pytest.approx(
                candidate.frequency * candidate.similarity * candidate.recency
            )
    # the often-used sticker wins when nothing in the context points elsewhere ...
    assert picker.rank("开心", "")[0].record.md5 == A
    # ... and a rarely used one wins when its description is exactly what is being talked about
    assert picker.rank("开心", "月亮晚安睡觉的小熊")[0].record.md5 == B
    # whereas the frequency still counts: with equal similarity the more used one is ahead
    flat = picker.rank("开心", "")
    assert flat[0].frequency > flat[1].frequency > flat[2].frequency


def test_a_sticker_without_a_vector_is_scored_with_the_neutral_similarity(
    library: StickerCatalog, services: Services, embedder: HashingBackend
) -> None:
    StickerVectors(services).sync([A])
    ranked = selector(services).rank("开心", "完全不相干的话题")
    by_md5 = {c.record.md5: c for c in ranked}
    assert by_md5[B].similarity == NEUTRAL_SIMILARITY and by_md5[A].similarity != NEUTRAL_SIMILARITY


def test_the_pick_is_a_draw_in_proportion_to_the_scores(
    library: StickerCatalog, services: Services
) -> None:
    picker = StickerSelector(services, rng=random.Random(3))
    scores = {c.record.md5: c.score for c in picker.rank("开心", "")}
    drawn = Counter(picker.choose("开心", "").md5 for _ in range(3000))  # type: ignore[union-attr]
    total = sum(scores.values())
    for md5, score in scores.items():
        assert drawn[md5] / 3000 == pytest.approx(score / total, abs=0.04)


def test_a_sticker_she_used_long_ago_scores_lower_but_never_zero(services: Services) -> None:
    catalog = make(
        services,
        {2: A, 4: A, 30: B, 32: B},
        {A: ("开心", "笑猫"), B: ("开心", "笑狗")},
    )
    services.settings.stickers.recency_half_life_days = 10
    ranked = selector(services).rank("开心", "")
    first, second = ranked
    assert first.record.md5 == B and second.record.md5 == A  # B was used about 26 days later
    assert first.frequency == pytest.approx(second.frequency)
    assert RECENCY_FLOOR < second.recency < first.recency <= 1.0
    last = UsageTable.load(services).as_of(None)[B].last
    age_days = (services.clock.now_utc() - last).total_seconds() / 86400
    assert first.recency == pytest.approx(
        RECENCY_FLOOR + (1 - RECENCY_FLOOR) * 0.5 ** (age_days / 10)
    )
    services.settings.stickers.recency_half_life_days = 1e-6  # ages count for almost nothing
    tiny = selector(services).rank("开心", "")
    assert all(c.recency == pytest.approx(RECENCY_FLOOR, abs=1e-6) for c in tiny)
    assert catalog.counts()["tagged"] == 2


# -------------------------------------------------------------------- repeats


def test_a_sticker_among_the_last_bubbles_is_not_sent_again(services: Services) -> None:
    sparse = make_sparse(services)
    assert sparse.repeat_rate() == 0.0 and not sparse.repeats_allowed()
    top = sparse.rank("开心", "")[0].record.md5
    recent = [None] * 5 + [top] + [None] * 3
    picked = {sparse.choose("开心", "", recent).md5 for _ in range(60)}  # type: ignore[union-attr]
    assert top not in picked and picked
    older = [top] + [None] * 10  # eleven bubbles ago: the window is ten
    assert top in {sparse.choose("开心", "", older).md5 for _ in range(60)}  # type: ignore[union-attr]


def make_sparse(services: Services) -> StickerSelector:
    make(services, SPARSE, {A: ("开心", "a猫"), B: ("开心", "b猫"), C: ("开心", "c猫")})
    return selector(services)


def test_when_every_candidate_was_just_used_the_nearby_tags_are_tried(services: Services) -> None:
    make(services, SPARSE, {A: ("开心", "a猫"), B: ("开心", "b猫"), C: ("大笑", "c猫")})
    picker = selector(services)
    recent = [A, B]
    assert {picker.choose("开心", "", recent).md5 for _ in range(30)} == {C}  # type: ignore[union-attr]
    assert picker.choose("开心", "", [A, B, C]) is None  # nothing else is near enough


def test_her_own_repeat_rate_relaxes_the_rule(library: StickerCatalog, services: Services) -> None:
    picker = selector(services)
    assert (
        picker.repeat_rate() > 0.30 and picker.repeats_allowed()
    )  # she sends A every other episode
    top = picker.rank("开心", "")[0].record.md5
    assert top in {picker.choose("开心", "", [top]).md5 for _ in range(60)}  # type: ignore[union-attr]
    services.settings.stickers.repeat_rate_threshold = 0.99
    strict = selector(services)
    assert not strict.repeats_allowed()
    assert top not in {strict.choose("开心", "", [top]).md5 for _ in range(60)}  # type: ignore[union-attr]
    services.settings.stickers.no_repeat_window = 0  # a zero window switches the rule off
    assert top in {selector(services).choose("开心", "", [top]).md5 for _ in range(60)}  # type: ignore[union-attr]


def test_the_repeat_rate_counts_stickers_within_the_window_of_bubbles(services: Services) -> None:
    files = sticker_files(2)
    one, two = list(files)
    # her bubbles: sticker, text, the same sticker again, another sticker -> 1 repeat in 3 stickers
    write_dialogue(
        services,
        [
            (
                day(0),
                [
                    U("hi"),
                    H(kind="sticker", md5=one),
                    H("嗯"),
                    H(kind="sticker", md5=one),
                    H(kind="sticker", md5=two),
                ],
            )
        ],
    )
    attach_files(services, files)
    assert her_repeat_rate(services, 10, before=None) == pytest.approx(1 / 3)
    assert her_repeat_rate(services, 1, before=None) == 0.0  # only the previous bubble counts
    assert her_repeat_rate(services, 2, before=None) == pytest.approx(1 / 3)
    assert her_repeat_rate(services, 10, before=day(0)) == 0.0  # nothing before the first message


def test_the_window_boundary_is_exactly_ten_bubbles(services: Services) -> None:
    files = sticker_files(1)
    only = next(iter(files))
    sticker = H(kind="sticker", md5=only)
    messages = [
        U("hi"),
        sticker,
        *[H("嗯")] * 9,
        sticker,
        *[H("嗯")] * 10,
        sticker,
        *[H("嗯")] * 10,
        sticker,
    ]
    write_dialogue(services, [(day(0), messages)])
    attach_files(services, files)
    # the 2nd sticker follows the 1st after nine bubbles (a repeat); the 3rd and 4th after ten
    assert her_repeat_rate(services, 10, before=None) == pytest.approx(1 / 4)
    assert her_repeat_rate(services, 9, before=None) == 0.0
    assert her_repeat_rate(services, 11, before=None) == pytest.approx(3 / 4)


# ------------------------------------------------------------------- fallbacks


def test_a_tag_without_candidates_falls_back_to_the_nearest_tags_then_to_none(
    services: Services,
) -> None:
    make(services, PLAN, {A: ("开心", "笑猫"), B: ("晚安", "睡熊"), C: ("撒娇", "撒娇猫")})
    picker = selector(services)
    assert picker.rank("大笑", "") == []  # no sticker carries it
    assert picker.choose("大笑", "").md5 == A  # type: ignore[union-attr]  # 开心 is its nearest tag
    assert picker.choose("委屈", "").md5 == C  # type: ignore[union-attr]  # nothing 委屈; 难过 none; 撒娇 yes
    assert picker.choose("饿", "").md5 == C  # type: ignore[union-attr]  # near 饿: 撒娇
    assert picker.choose("其他", "") is None  # no neighbours at all
    assert picker.choose("不存在的标签", "") is None


def test_stickers_that_may_not_be_chosen_are_never_chosen(
    library: StickerCatalog, services: Services
) -> None:
    picker = selector(services)
    assert {c.record.md5 for c in picker.rank("开心", "")} == {A, B, C}
    library.set_disabled(A, True)
    from twin.storage.chat_models import Sticker

    with services.db.transaction(bump_state=False) as session:
        row = session.get(Sticker, B)
        assert row is not None
        row.status = "pending"
    fresh = selector(services)
    assert {c.record.md5 for c in fresh.rank("开心", "")} == {C}
    library.clear_manual(C)
    library.set_manual(C, ["晚安"])
    again = selector(services)
    assert again.rank("开心", "") == [] and {c.record.md5 for c in again.rank("晚安", "")} == {
        C,
        D,
        E,
    }


def test_a_sticker_only_the_user_sent_is_not_a_candidate(
    library: StickerCatalog, services: Services
) -> None:
    write_dialogue(services, [(day(45), [Msg("u", None, "sticker", F), H("哈")])], append=True)
    attach_files(services, {F: sticker_files(6)[F]})
    sync_counters(services)
    library.save_vision(F, ["开心"], "用户发的", "场合", at=NOW)
    assert library.require(F).user_uses == 1 and library.require(F).her_uses == 0
    assert F not in {c.record.md5 for c in selector(services).rank("开心", "")}


# ---------------------------------------------------------------- the data views


def test_an_as_of_view_never_offers_a_sticker_first_used_after_that_moment(
    library: StickerCatalog, services: Services
) -> None:
    live = selector(services)
    assert {c.record.md5 for c in live.rank("晚安", "")} == {D, E}
    early = selector(services, view=as_of_view(day(20, 12)))
    assert early.rank("晚安", "") == [] and early.choose("晚安", "") is None
    late = selector(services, view=as_of_view(day(31, 12)))
    assert {c.record.md5 for c in late.rank("晚安", "")} == {D}
    assert as_of_view(day(20)).scope == "pre_holdout" and LIVE_VIEW.scope == "live"
    with pytest.raises(ValueError, match="time zone"):
        as_of_view(datetime(2026, 4, 1))  # noqa: DTZ001 - a naive time is the point


def test_an_as_of_view_counts_uses_and_recency_as_of_its_moment(
    library: StickerCatalog, services: Services
) -> None:
    moment = day(9, 12)  # before the uses of episodes 10 and 12 and of the later ones
    ranked = selector(services, view=as_of_view(moment)).rank("开心", "")
    uses = {c.record.md5: c.uses for c in ranked}
    assert uses == {A: 4, B: 1, C: 2}  # episodes 2, 4, 6, 8 / 3 / 5, 7
    table = UsageTable.load(services)
    assert table.as_of(moment)[A].uses == 4 and day(8, 12) < table.as_of(moment)[A].last < moment
    assert table.as_of(None)[A].uses == 6 and D not in table.as_of(day(29, 12))
    assert table.as_of(day(30, 13))[D].uses == 1
    # recency is measured from that moment, not from the present
    live = selector(services).rank("开心", "")
    assert max(c.recency for c in ranked) > max(c.recency for c in live)


def test_the_view_can_be_changed_without_loading_the_data_again(
    library: StickerCatalog, services: Services
) -> None:
    base = selector(services)
    base.rank("开心", "")
    past = base.with_view(StickerView("pre_holdout", day(9, 12)))
    assert past.view.as_of == day(9, 12)
    assert {c.record.md5: c.uses for c in past.rank("开心", "")} == {A: 4, B: 1, C: 2}
    assert {c.uses for c in base.rank("开心", "")} == {6, 2, 1}  # the base view is unchanged
    base.reload()
    assert {c.uses for c in base.rank("开心", "")} == {6, 2, 1}


def test_the_past_view_measures_her_repeat_rate_in_the_data_before_the_cutoff(
    library: StickerCatalog, services: Services
) -> None:
    from twin.profile.holdout import holdout_cutoff

    cutoff = holdout_cutoff(services)
    expected = her_repeat_rate(services, 10, before=cutoff)
    past = selector(services, view=StickerView("pre_holdout"))
    assert past.repeat_rate() == pytest.approx(expected)
    assert past.view.as_of is None and past._bound() == cutoff
    assert {c.record.md5 for c in past.rank("晚安", "")} == {D}  # E was first used after it
    with services.db.session() as session:
        times = [row.create_time_utc for row in session.scalars(her_bubble_skeleton(cutoff))]
    assert times and all(moment < cutoff for moment in times)
