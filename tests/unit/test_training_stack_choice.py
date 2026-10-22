"""R-TRN-009, R17-49d (D-655): the SFT stack against the SFT+DPO stack, on the same samples.

The rule is an order over the project's own measures (the six style metrics within the +-30 % of
the M5 gate, the validation loss) - the tests pin the order, show that no threshold was moved or
added, and run the whole comparison on generated replies with a profile whose values are known.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.unit.test_eval_style import BURST_GAP, SEGMENT_GAP, known_replies, profile_of
from twin.eval.render import Candidate
from twin.eval.style_metrics import TOLERANCE, candidate_recs, measure, read_metrics
from twin.services import Services
from twin.training import stack_choice
from twin.training.stack_choice import (
    MetricLine,
    StackChoiceError,
    StackFiles,
    StackScore,
    candidate_of_generation,
    choose_stack,
    judge_stacks,
    read_generations,
)

KEYS = ("text_length", "comma_rate", "burst_size", "sticker_share", "emoji_code_rate", "quote_rate")


def score(
    stack: str,
    passes: int,
    *,
    deviation: float = 0.1,
    undefined: int = 0,
    val_loss: float | None = 1.2,
) -> StackScore:
    """A stack that passes the first ``passes`` of the six metrics, with this mean deviation."""
    lines = []
    for index, key in enumerate(KEYS):
        if index < undefined:
            lines.append(MetricLine(key, "fail", None, 1.0, 0.0))
        elif index < undefined + passes:
            lines.append(MetricLine(key, "pass", deviation, 1.0, 1.0))
        else:
            lines.append(MetricLine(key, "fail", deviation + 0.5, 1.0, 1.0))
    return StackScore(stack, tuple(lines), val_loss, 100)


# ------------------------------------------------------------------------------- the rule


@pytest.mark.parametrize(
    ("sft", "dpo", "keep", "why"),
    [
        (score("sft", 4), score("dpo", 5), "dpo", "dpo has more style metrics within"),
        (score("sft", 5), score("dpo", 4), "sft", "sft has more style metrics within"),
        # the number of metrics within the tolerance outranks everything after it
        (
            score("sft", 4, deviation=0.3, val_loss=2.0),
            score("dpo", 5, deviation=0.3, val_loss=3.0),
            "dpo",
            "5 for dpo against 4 for sft",
        ),
        (
            score("sft", 3, undefined=1),
            score("dpo", 3, undefined=0),
            "dpo",
            "dpo has fewer without a deviation",
        ),
        (
            score("sft", 3, undefined=0),
            score("dpo", 3, undefined=1),
            "sft",
            "sft has fewer without a deviation",
        ),
        (
            score("sft", 3, deviation=0.20),
            score("dpo", 3, deviation=0.10),
            "dpo",
            "dpo has the smaller mean deviation",
        ),
        (
            score("sft", 3, deviation=0.10),
            score("dpo", 3, deviation=0.20),
            "sft",
            "sft has the smaller mean deviation",
        ),
        (
            score("sft", 3, val_loss=1.3),
            score("dpo", 3, val_loss=1.2),
            "dpo",
            "dpo has the lower validation loss",
        ),
        (
            score("sft", 3, val_loss=1.2),
            score("dpo", 3, val_loss=1.3),
            "sft",
            "sft has the lower validation loss",
        ),
        (score("sft", 3), score("dpo", 3), "sft", "a tie on every measure"),
        (score("sft", 3, val_loss=None), score("dpo", 3, val_loss=1.0), "sft", "a tie"),
        (score("sft", 0), score("dpo", 0), "sft", "a tie"),
    ],
)
def test_the_stack_with_more_metrics_inside_the_tolerance_wins_and_a_tie_keeps_sft(
    sft: StackScore, dpo: StackScore, keep: str, why: str
) -> None:
    verdict = choose_stack(sft, dpo)
    assert verdict.keep == keep and why in verdict.reason


def test_dpo_has_to_be_better_to_be_kept_not_merely_not_worse() -> None:
    equal = choose_stack(score("sft", 5), score("dpo", 5))
    assert equal.keep == "sft"
    slightly_better = choose_stack(
        score("sft", 5, deviation=0.10000001), score("dpo", 5, deviation=0.1)
    )
    assert slightly_better.keep == "sft"  # below the 6 decimals the comparison looks at
    better = choose_stack(score("sft", 5, deviation=0.11), score("dpo", 5, deviation=0.10))
    assert better.keep == "dpo"


@given(
    sft_passes=st.integers(0, 6),
    dpo_passes=st.integers(0, 6),
    sft_deviation=st.floats(0.0, 0.3),
    dpo_deviation=st.floats(0.0, 0.3),
    sft_loss=st.floats(0.1, 5.0),
    dpo_loss=st.floats(0.1, 5.0),
)
def test_dpo_is_kept_exactly_when_it_comes_first_in_the_order_of_the_rule(
    sft_passes: int,
    dpo_passes: int,
    sft_deviation: float,
    dpo_deviation: float,
    sft_loss: float,
    dpo_loss: float,
) -> None:
    sft = score("sft", sft_passes, deviation=sft_deviation, val_loss=sft_loss)
    dpo = score("dpo", dpo_passes, deviation=dpo_deviation, val_loss=dpo_loss)

    def order(item: StackScore) -> tuple[float, float, float]:
        mean = item.mean_abs_deviation
        assert mean is not None
        return (-item.passed, round(mean, 6), round(item.val_loss or 0.0, 6))

    assert (choose_stack(sft, dpo).keep == "dpo") is (order(dpo) < order(sft))


@given(passes=st.integers(0, 5), deviation=st.floats(0.0, 0.3), extra=st.floats(0.01, 0.2))
def test_one_more_metric_inside_or_a_clearly_smaller_deviation_keeps_dpo(
    passes: int, deviation: float, extra: float
) -> None:
    """Monotonic: what makes DPO better by the rule makes it win, whatever the other numbers."""
    assert choose_stack(score("sft", passes), score("dpo", passes + 1)).keep == "dpo"
    sft = score("sft", passes, deviation=deviation + extra)
    assert choose_stack(sft, score("dpo", passes, deviation=deviation)).keep == "dpo"


def test_the_verdict_holds_numbers_and_words_and_the_unchanged_tolerance() -> None:
    verdict = choose_stack(score("sft", 4), score("dpo", 5))
    data = verdict.to_json()
    assert data["keep"] == "dpo" and data["schema"] == 1
    assert data["tolerance"] == TOLERANCE == 0.30  # the gate's own number, reused, not repeated
    assert data["stacks"]["dpo"]["passed"] == 5 and data["stacks"]["sft"]["counted"] == 6
    assert [m["key"] for m in data["stacks"]["sft"]["metrics"]] == list(KEYS)
    assert json.loads(json.dumps(data)) == data
    shown = "\n".join(verdict.lines())
    assert shown.startswith("stack kept: dpo (") and "5 of 6 style metrics within +-30%" in shown


def test_the_module_adds_and_moves_no_threshold() -> None:
    """R17-49d: the order uses the style metrics' tolerance and invents no number of its own."""
    source = Path(stack_choice.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "TOLERANCE" not in assigned  # imported from twin.eval.style_metrics, never redefined
    floats = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    ]
    assert floats == []  # no float constant at all: no threshold, no weight


def test_a_metric_the_profile_cannot_judge_does_not_count_as_inside_the_tolerance() -> None:
    lines = (
        MetricLine("text_length", "pass", 0.0, 8.0, 8.0),
        MetricLine("quote_rate", "n/a", None, None, None),
        MetricLine("comma_rate", "no_data", None, None, 0.1),
    )
    item = StackScore("sft", lines, None, 3)
    assert (item.passed, item.counted, item.undefined) == (1, 2, 1)
    assert item.mean_abs_deviation == 0.0


# ------------------------------------------------------------------ reading the generations


def raw_of(candidate: Candidate) -> str:
    """The generated text the model would write for a candidate (the output convention)."""
    lines = [f"[引用:{candidate.quote}]"] if candidate.quote else []
    lines.extend(
        "[表情包:开心]" if line.kind == "sticker" else line.text for line in candidate.lines
    )
    return "\n".join(lines)


def generations_of(replies: list[Candidate], *, prefix: str = "te") -> str:
    return "".join(
        json.dumps({"id": f"{prefix}{index}", "text": raw_of(reply)}, ensure_ascii=False) + "\n"
        for index, reply in enumerate(replies)
    )


def test_a_generated_reply_becomes_the_candidate_the_style_metrics_measure() -> None:
    replies = known_replies()
    rebuilt = [candidate_of_generation(raw_of(reply)) for reply in replies]
    for original, found in zip(replies, rebuilt, strict=True):
        assert [line.text for line in found.lines if line.kind == "text"] == [
            line.text for line in original.lines if line.kind == "text"
        ]
        assert len(found.stickers) == len(original.stickers) and found.quote == original.quote
    measured = read_metrics(measure(candidate_recs(rebuilt), BURST_GAP, SEGMENT_GAP))
    expected = read_metrics(measure(candidate_recs(replies), BURST_GAP, SEGMENT_GAP))
    assert measured == expected


def test_the_output_conventions_are_read_by_the_engines_parser() -> None:
    reply = candidate_of_generation("[引用：你好]\n她：哈哈 [表情包：大笑] 好的\n[不回]\n\n")
    assert reply.quote == "你好"
    assert [(line.kind, line.text) for line in reply.lines] == [
        ("text", "哈哈"),
        ("sticker", ""),
        ("text", "好的"),
    ]
    assert candidate_of_generation("").empty and candidate_of_generation("[不回]").empty
    same_tag = candidate_of_generation("[表情包:开心]").stickers
    assert same_tag == candidate_of_generation("[表情包：开心]").stickers  # one tag, one sticker
    assert same_tag != candidate_of_generation("[表情包:大笑]").stickers


def test_the_generations_file_is_read_in_order_and_a_broken_line_names_only_its_number() -> None:
    text = '{"id": "a", "text": "x"}\n\n{"id": "b", "text": "y"}\n'
    assert read_generations(text) == [("a", "x"), ("b", "y")]
    secret = "a sentence of hers"
    for broken in (secret, '{"id": "a"}', "[1, 2]"):
        with pytest.raises(StackChoiceError, match="line 1") as caught:
            read_generations(broken)
        assert secret not in str(caught.value)


# ------------------------------------------------------------------------ the whole judge


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-holdout profile whose values are exactly those of ``known_replies()``."""
    monkeypatch.setattr(
        stack_choice, "load_profile", lambda services, scope: SimpleNamespace(metrics=profile_of())
    )


def without_stickers(replies: list[Candidate]) -> list[Candidate]:
    return [
        Candidate(tuple(line for line in reply.lines if line.kind != "sticker"), reply.quote)
        for reply in replies
    ]


def files(sft: str, dpo: str, *, sft_loss: float = 1.2, dpo_loss: float = 1.3) -> StackFiles:
    return StackFiles(
        {"sft": sft, "dpo": dpo},
        {"sft": {"val_loss": sft_loss, "generated": 20}, "dpo": {"val_loss": dpo_loss}},
    )


@pytest.mark.usefixtures("profile")
def test_the_judge_keeps_the_stack_whose_replies_are_closer_to_her_style(
    services: Services,
) -> None:
    good, drifted = (
        generations_of(known_replies()),
        generations_of(without_stickers(known_replies())),
    )
    kept_sft = judge_stacks(services, files(good, drifted))
    assert kept_sft.keep == "sft"
    assert (kept_sft.sft.passed, kept_sft.dpo.passed) == (6, 5)  # no stickers: the share fails
    kept_dpo = judge_stacks(services, files(drifted, good))
    assert kept_dpo.keep == "dpo" and (kept_dpo.sft.passed, kept_dpo.dpo.passed) == (5, 6)
    assert "6 for dpo against 5 for sft" in kept_dpo.reason
    tie = judge_stacks(services, files(good, good, dpo_loss=1.2))
    assert tie.keep == "sft" and "a tie" in tie.reason
    assert tie.sft.val_loss == 1.2 and tie.dpo.val_loss == 1.2
    higher_loss = judge_stacks(services, files(good, good))  # the same replies, a worse loss
    assert higher_loss.keep == "sft" and "sft has the lower validation loss" in higher_loss.reason
    lower_loss = judge_stacks(services, files(good, good, dpo_loss=1.1))
    assert lower_loss.keep == "dpo" and "dpo has the lower validation loss" in lower_loss.reason
    sticker = next(m for m in kept_sft.dpo.metrics if m.key == "sticker_share")
    assert sticker.status == "fail" and sticker.deviation == pytest.approx(-1.0)
    # the verdict is numbers and words: nothing of a reply is in it
    assert "你好" not in json.dumps(kept_sft.to_json(), ensure_ascii=False)


@pytest.mark.usefixtures("profile")
def test_the_judge_needs_the_same_samples_in_the_same_order_and_something_to_judge(
    services: Services,
) -> None:
    good = generations_of(known_replies())
    other_ids = generations_of(known_replies(), prefix="x")
    with pytest.raises(StackChoiceError, match="not evaluated on the same test contexts"):
        judge_stacks(services, files(good, other_ids))
    reordered = "".join(reversed(good.splitlines(keepends=True)))
    with pytest.raises(StackChoiceError, match="same test contexts"):
        judge_stacks(services, files(good, reordered))
    with pytest.raises(StackChoiceError, match="generated no replies"):
        judge_stacks(services, files("", ""))
    with pytest.raises(StackChoiceError, match="dpo stack are missing"):
        judge_stacks(services, StackFiles({"sft": good}, {}))
    secret = "a sentence of hers"
    with pytest.raises(StackChoiceError, match="line 1") as caught:
        judge_stacks(services, files(good, secret))
    assert secret not in str(caught.value)


def test_the_judge_needs_the_pre_holdout_profile_to_measure_against(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stack_choice, "load_profile", lambda services, scope: None)
    good = generations_of(known_replies())
    with pytest.raises(StackChoiceError, match="no pre-holdout profile"):
        judge_stacks(services, files(good, good))


@pytest.mark.usefixtures("profile")
def test_a_missing_or_odd_validation_loss_is_not_a_number_to_compare(services: Services) -> None:
    good = generations_of(known_replies())
    odd = StackFiles(
        {"sft": good, "dpo": good},
        {"sft": {"val_loss": "high"}, "dpo": {"val_loss": float("nan")}},
    )
    verdict = judge_stacks(services, odd)
    assert verdict.sft.val_loss is None and verdict.dpo.val_loss is None
    assert verdict.keep == "sft"
    absent = judge_stacks(services, StackFiles({"sft": good, "dpo": good}, {}))
    assert absent.keep == "sft"
