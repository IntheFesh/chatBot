"""Drawing the contexts of a blind test from the hold-out (R-EVAL-001, R-SAFE-006, R-TRN-013)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import pytest

from tests.support.embedding import H, Msg, U, day, write_dialogue
from twin.eval.samples import (
    LENGTH_BINS,
    PERIODS,
    SampleDrawer,
    allocate,
    length_bin_of,
    period_of,
)
from twin.profile.holdout import holdout_cutoff
from twin.retrieval.windows import LocalPlace
from twin.services import Services

LATE_REPLY = "只在留出期里才出现的回复"


def episode(number: int) -> list[Msg]:
    """1 to 4 exchanges; the context before her reply has 1, 3, 5 or 7 turns."""
    exchanges = number % 4 + 1
    messages: list[Msg] = []
    for k in range(exchanges):
        messages.append(U(f"问题{number}-{k}"))
        if k < exchanges - 1:
            messages.append(H(f"回答{number}-{k}"))
    messages.append(H(f"最后的回复{number}"))
    return messages


def build(services: Services, count: int = 160, ratio: float = 0.6) -> list[datetime]:
    """``count`` episodes at hours that cycle through the day, two days apart in 5 h steps."""
    services.settings.retrieval.holdout_ratio = ratio
    episodes = [(day(n // 3, (n * 5) % 24, n % 3 * 7), episode(n)) for n in range(count)]
    return write_dialogue(services, episodes)


def keys(drawn: list[object]) -> list[str]:
    return [getattr(sample, "sample_key") for sample in drawn]  # noqa: B009


def test_periods_and_lengths_have_fixed_bounds() -> None:
    assert [period_of(h) for h in (0, 5, 6, 11, 12, 17, 18, 23)] == [
        "night",
        "night",
        "morning",
        "morning",
        "afternoon",
        "afternoon",
        "evening",
        "evening",
    ]
    assert [length_bin_of(n) for n in (1, 2, 3, 5, 6, 8)] == [
        "short",
        "short",
        "medium",
        "medium",
        "long",
        "long",
    ]
    with pytest.raises(ValueError, match="0 to 23"):
        period_of(24)


@pytest.mark.parametrize(
    ("sizes", "n", "expected"),
    [
        ({"a": 100, "b": 100}, 10, {"a": 5, "b": 5}),
        ({"a": 90, "b": 9, "c": 1}, 20, {"a": 18, "b": 1, "c": 1}),  # at least one from each
        ({"a": 3, "b": 50}, 20, {"a": 1, "b": 19}),  # proportional: 20 * 3 / 53 is about one
        ({"a": 2, "b": 3}, 5, {"a": 2, "b": 3}),  # everything there is
        ({"a": 5, "b": 5, "c": 5}, 2, {"a": 1, "b": 1}),  # fewer draws than groups: largest first
        ({"a": 4, "b": 0}, 10, {"a": 4}),  # never more than there is
        ({}, 5, {}),
    ],
)
def test_the_allocation_is_proportional_with_one_at_least_from_every_group(
    sizes: dict[str, int], n: int, expected: dict[str, int]
) -> None:
    assert allocate(sizes, n) == expected


def test_the_allocation_always_adds_up_and_respects_the_group_sizes() -> None:
    sizes = {("night", "short"): 13, ("night", "long"): 2, ("evening", "medium"): 41}
    for n in range(1, 60):
        found = allocate(sizes, n)
        assert sum(found.values()) == min(n, sum(sizes.values()))
        assert all(0 < count <= sizes[key] for key, count in found.items())
    assert allocate({"a": 5, "b": 5}, 4, at_least_one=False) == {"a": 2, "b": 2}


def test_only_hold_out_blocks_are_drawn_and_each_carries_what_the_sandbox_needs(
    services: Services,
) -> None:
    build(services)
    cutoff = holdout_cutoff(services)
    drawer = SampleDrawer(services)
    drawn = drawer.draw(20, seed=3)
    assert len(drawn.samples) == 20
    assert all(sample.at >= cutoff for sample in drawn.samples)
    assert len({s.sample_key for s in drawn.samples}) == 20
    for sample in drawn.samples:
        assert not sample.real.empty and sample.inbound and sample.shown
        assert sample.inbound[0]["text"].startswith("问题")
        assert len(sample.history) <= 7
        assert sample.shown[-1]["who"] == "me"  # the conversation ends with the user's turn
        assert all(turn["who"] in {"me", "her"} for turn in sample.shown)
        assert sample.payload()["t"] == sample.at.isoformat()
        # nothing of the hold-out reply is in its own context
        shown = " ".join(line for turn in sample.shown for line in turn["lines"])
        assert sample.real.lines[0].text not in shown
    assert drawer.render_real(drawn.samples[0]).startswith(("回答", "最后的回复"))


def test_the_draw_is_spread_over_the_four_periods_and_three_lengths(services: Services) -> None:
    build(services)
    drawer = SampleDrawer(services)
    groups, _ = drawer.candidates(set())
    drawn = drawer.draw(48, seed=11)
    assert len(drawn.samples) == 48
    covered = {(s.period, s.length_bin) for s in drawn.samples}
    assert covered == set(groups)  # every stratum that has a block is represented
    assert {p for p, _ in covered} == set(PERIODS) and {n for _, n in covered} == set(LENGTH_BINS)
    place = LocalPlace.create(services)
    for sample in drawn.samples:
        assert sample.period == period_of(int(place.stamp(sample.at).minute // 60))
    per_period = Counter(s.period for s in drawn.samples)
    assert min(per_period.values()) >= 6  # no period is neglected
    assert sum(drawn.strata.values()) == 48 and drawn.available == sum(map(len, groups.values()))


def test_contexts_that_an_earlier_test_used_are_not_drawn_again(services: Services) -> None:
    build(services)
    drawer = SampleDrawer(services)
    first = drawer.draw(25, seed=1)
    used = {s.sample_key for s in first.samples}
    second = drawer.draw(25, seed=2, exclude=used)
    assert not used & {s.sample_key for s in second.samples}
    assert second.excluded["context_used_before"] == len(used)
    everything = drawer.draw(500, seed=3)
    rest = drawer.draw(500, seed=3, exclude={s.sample_key for s in everything.samples})
    assert rest.samples == [] and rest.available == 0


def test_the_same_seed_gives_the_same_draw_and_another_seed_another(services: Services) -> None:
    build(services)
    drawer = SampleDrawer(services)
    a = [s.sample_key for s in drawer.draw(20, seed=5).samples]
    b = [s.sample_key for s in drawer.draw(20, seed=5).samples]
    c = [s.sample_key for s in drawer.draw(20, seed=6).samples]
    assert a == b and a != c


def test_a_reply_the_bot_could_not_have_written_is_never_drawn(services: Services) -> None:
    """R-SAFE-006 4: a picture, a voice message or a typed placeholder in her reply excludes it."""
    services.settings.retrieval.holdout_ratio = 0.9
    episodes = [(day(n, 3 + n % 20), episode(n)) for n in range(40)]
    picture = [U("看这个"), H(kind="image"), H("好看吧")]
    voice = [U("听听"), H(kind="voice", text="语音转写")]
    placeholder = [U("给你"), H("[链接]")]
    stray_event = [U("发给你了"), H("[图片]"), H("收到了吗")]
    for index, messages in enumerate((picture, voice, placeholder, stray_event)):
        episodes.append((day(40 + index, 12), messages))
    write_dialogue(services, episodes)
    drawn = SampleDrawer(services).draw(500, seed=1)
    texts = {line.text for s in drawn.samples for line in s.real.lines}
    assert "好看吧" not in texts and "收到了吗" not in texts and "[链接]" not in texts
    assert "语音转写" not in texts and "[图片]" not in texts
    assert drawn.excluded["reply_not_reproducible"] >= 2  # the picture and the voice message
    assert drawn.excluded["reply_has_event_text"] >= 2  # the typed placeholders
    assert len(drawn.samples) >= 20  # the ordinary ones are all there


def test_a_context_that_does_not_end_with_the_user_is_not_drawn(services: Services) -> None:
    """She wrote twice in a row (two bursts): the second reply answers nobody."""
    services.settings.retrieval.holdout_ratio = 0.9
    twice = [U("在吗"), H("在的"), Msg("h", "又想到一件事", gap=300.0)]
    episodes = [(day(n, 3 + n % 20), episode(n)) for n in range(30)]
    episodes.append((day(30, 12), twice))
    write_dialogue(services, episodes)
    drawn = SampleDrawer(services).draw(500, seed=1)
    assert "又想到一件事" not in {line.text for s in drawn.samples for line in s.real.lines}
    assert drawn.excluded["context_does_not_end_with_the_user"] == 1


def test_fewer_than_asked_are_returned_when_the_hold_out_has_no_more(services: Services) -> None:
    build(services, count=24, ratio=0.5)
    drawn = SampleDrawer(services).draw(200, seed=4)
    assert 0 < len(drawn.samples) < 200
    assert (
        len(drawn.samples) == drawn.available - drawn.excluded["context_does_not_end_with_the_user"]
    )


def test_the_history_is_at_most_seven_merged_turns_before_the_message_being_answered(
    services: Services,
) -> None:
    services.settings.retrieval.holdout_ratio = 0.9
    long = []
    for k in range(6):
        long += [U(f"问{k}"), H(f"答{k}")]
    long += [U("最后一问"), H("最后的回答")]
    write_dialogue(services, [(day(n, 4 + n % 18), episode(n)) for n in range(20)])
    write_dialogue(services, [(day(30, 12), long)], append=True)
    drawn = SampleDrawer(services).draw(500, seed=2)
    sample = next(s for s in drawn.samples if s.real.lines[0].text == "最后的回答")
    assert len(sample.history) == 7 and sample.history[-1]["role"] == "bot"
    assert sample.inbound[0]["text"] == "最后一问"
    assert [turn["role"] for turn in sample.history] == ["bot", "user"] * 3 + ["bot"]
