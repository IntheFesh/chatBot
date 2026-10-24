"""R-EVAL-004: the contradiction list the audit asks DeepSeek for, and the checks it is held to.

The JSON is validated twice - by the schema of the reply and by the evidence the model was given -
and the rule of the requirement (at most one obvious contradiction a week) is pinned to the SPEC.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from twin.eval.consistency_model import (
    DROPPED_OVER_LIMIT,
    DROPPED_REPEAT,
    DROPPED_SAME_RECORD,
    DROPPED_UNKNOWN_REF,
    MAX_FINDINGS,
    MAX_OBVIOUS_PER_WEEK,
    WEEK_DAYS,
    ConsistencyOut,
    Evidence,
    EvidencePack,
    allowed_obvious,
    fingerprint_of,
    is_bot_made,
    judge_audit,
    losers_of,
    rank_of,
    ref_kind,
    statement_of,
    validate_output,
)
from twin.llm.deepseek import parse_json_reply

SPEC = Path(__file__).resolve().parents[2] / "docs" / "SPEC.md"
NOON = datetime(2026, 10, 5, 17, tzinfo=UTC)


def evidence(
    ref: str,
    text: str,
    *,
    item_id: str | None = None,
    source: str | None = None,
    minutes: int = 0,
) -> Evidence:
    kind = ref_kind(ref)
    assert kind is not None
    return Evidence(
        ref=ref,
        kind=kind,
        item_id=item_id or f"id-{ref}",
        at=NOON + timedelta(minutes=minutes),
        text=text,
        source=source,
        known_at=NOON,
        number=int(ref[1:]) if kind == "fact" else None,
        message_ids=(f"id-{ref}", f"{ref}-b") if kind == "reply" else (),
    )


def make_pack() -> EvidencePack:
    return EvidencePack(
        window_start=NOON - timedelta(days=7),
        window_end=NOON,
        zone="America/Chicago",
        days=7,
        items=(
            evidence("L1", "10-04（周日） 09:00-11:00 在图书馆看书"),
            evidence("L2", "10-04（周日） 13:00-15:00 在公司开会", source="plan"),
            evidence("R1", "今天整天都在家躺着，哪儿也没去", minutes=60),
            evidence("R2", "我周末一般会去爬山", minutes=90),
            evidence("F1", "她在北京读研", source="real_record"),
            evidence("F2", "她刚搬到上海工作", source="bot_invented"),
            evidence("F3", "她不吃香菜", source="user_command"),
        ),
    )


def contradiction(first: str, second: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "time": "10-04 下午",
        "first": {"ref": first, "quote": None},
        "second": {"ref": second, "quote": None},
        "related": [],
        "severity": "obvious",
        "reason": "两处说法不可能同时为真",
        "keep": None,
        "rewrite": None,
    }
    base.update(fields)
    return base


def answer(*items: dict[str, Any]) -> ConsistencyOut:
    return ConsistencyOut.model_validate({"contradictions": list(items)})


# ---------------------------------------------------------------------------- the schema


def test_a_well_formed_answer_is_parsed_with_every_field() -> None:
    reply = json.dumps(
        {"contradictions": [contradiction("L1", "R1", keep="L1", rewrite="在家休息")]},
        ensure_ascii=False,
    )
    out = parse_json_reply(reply, ConsistencyOut)
    item = out.contradictions[0]
    assert (item.first.ref, item.second.ref, item.severity) == ("L1", "R1", "obvious")
    assert item.keep == "L1" and item.rewrite == "在家休息" and item.time == "10-04 下午"


def test_an_empty_list_and_a_fenced_answer_are_fine() -> None:
    assert parse_json_reply('{"contradictions": []}', ConsistencyOut).contradictions == []
    fenced = '```json\n{"contradictions": []}\n```'
    assert parse_json_reply(fenced, ConsistencyOut).contradictions == []


@pytest.mark.parametrize(
    "broken",
    [
        "",
        "没有矛盾",
        '{"contradictions": [',
        '{"contradictions": "none"}',
        '{"contradictions": [{"time": "x"}]}',
        json.dumps({"contradictions": [contradiction("L1", "R1", severity="serious")]}),
        json.dumps({"contradictions": [contradiction("L1", "R1", reason="")]}),
        json.dumps({"contradictions": [contradiction("L1", "R1", time="")]}),
        json.dumps({"contradictions": [contradiction("X9", "R1")]}),
        json.dumps({"contradictions": [contradiction("L1", "R1", reason="字" * 201)]}),
        json.dumps({"contradictions": [contradiction("L1", "R1", rewrite="字" * 121)]}),
    ],
)
def test_a_malformed_answer_is_refused_with_the_reason(broken: str) -> None:
    with pytest.raises(ValueError, match=r"empty|invalid JSON"):
        parse_json_reply(broken, ConsistencyOut)


def test_unknown_keys_are_ignored_and_blank_optionals_become_none() -> None:
    item = contradiction("L1", "R1", keep="", rewrite="", extra="x")
    item["first"]["quote"] = "   "
    out = answer(item)
    assert out.contradictions[0].keep is None and out.contradictions[0].rewrite is None
    assert out.contradictions[0].first.quote is None


def test_ref_kinds_are_the_three_prefixes() -> None:
    assert [ref_kind(r) for r in ("L1", "R20", "F3")] == ["lifeline", "reply", "fact"]
    assert ref_kind("X1") is None and ref_kind("L") is None and ref_kind("l1") is None


# ---------------------------------------------------------------- held to the evidence


def test_a_valid_contradiction_becomes_a_finding_with_both_statements() -> None:
    pack = make_pack()
    checked = validate_output(answer(contradiction("L1", "R1")), pack)
    assert checked.dropped_total == 0 and len(checked.findings) == 1
    found = checked.findings[0]
    assert (found.first.ref, found.second.ref) == ("L1", "R1")
    assert found.first.kind == "lifeline" and found.second.kind == "reply"
    assert found.first.text == "10-04（周日） 09:00-11:00 在图书馆看书"
    assert found.at == NOON + timedelta(minutes=60)  # the later of the two
    assert found.severity == "obvious" and found.time_text == "10-04 下午"
    assert found.second.message_ids == ("id-R1", "R1-b")
    assert found.to_payload()["first"]["ref"] == "L1" and found.to_payload()["keep"] is None


def test_a_record_nobody_was_shown_drops_the_contradiction() -> None:
    checked = validate_output(
        answer(
            contradiction("L1", "R9"),
            contradiction("F7", "R1"),
            contradiction("L1", "R1"),
        ),
        make_pack(),
    )
    assert [f.second.ref for f in checked.findings] == ["R1"]
    assert checked.dropped[DROPPED_UNKNOWN_REF] == 2


def test_the_same_record_twice_is_not_a_contradiction() -> None:
    checked = validate_output(answer(contradiction("L1", "L1")), make_pack())
    assert checked.findings == [] and checked.dropped[DROPPED_SAME_RECORD] == 1


def test_a_pair_is_reported_once_whichever_side_is_named_first() -> None:
    checked = validate_output(
        answer(contradiction("L1", "R1"), contradiction("R1", "L1", severity="minor")),
        make_pack(),
    )
    assert len(checked.findings) == 1 and checked.dropped[DROPPED_REPEAT] == 1
    assert checked.findings[0].severity == "obvious"  # the first report stands


def test_no_more_than_the_limit_is_kept() -> None:
    pack = make_pack()
    refs = [item.ref for item in pack.items]
    pairs = [contradiction(a, b) for i, a in enumerate(refs) for b in refs[i + 1 :]]
    assert len(pairs) > MAX_FINDINGS
    checked = validate_output(answer(*pairs), pack)
    assert len(checked.findings) == MAX_FINDINGS
    assert checked.dropped[DROPPED_OVER_LIMIT] == len(pairs) - MAX_FINDINGS


def test_a_quotation_is_kept_only_when_it_is_in_the_record() -> None:
    pack = make_pack()
    real = statement_of(pack.items[2], "整天都在家躺着")
    assert real.text == "整天都在家躺着"
    invented = statement_of(pack.items[2], "我昨晚去看了电影")
    assert invented.text == pack.items[2].text  # the record's own text is shown instead
    spaced = statement_of(pack.items[2], "整天 都在家 躺着！")
    assert spaced.text == "整天 都在家 躺着！"  # punctuation and spaces do not matter
    assert statement_of(pack.items[2], None).text == pack.items[2].text


def test_unknown_related_records_are_dropped_and_known_ones_kept_once() -> None:
    checked = validate_output(
        answer(contradiction("L1", "R1", related=["F1", "F1", "F9", "L1", "R1"])), make_pack()
    )
    assert [s.ref for s in checked.findings[0].related] == ["F1"]
    assert [s.ref for s in checked.findings[0].statements] == ["L1", "R1", "F1"]


def test_a_keep_that_is_neither_side_is_ignored() -> None:
    pack = make_pack()
    ignored = validate_output(answer(contradiction("L1", "R1", keep="F1")), pack)
    assert ignored.findings[0].keep is None
    kept = validate_output(answer(contradiction("L1", "R1", keep="R1")), pack)
    assert kept.findings[0].keep == "R1"


def test_a_rewrite_is_kept_only_for_one_loser_the_audit_may_change() -> None:
    pack = make_pack()
    # the life line entry gives way: it can be rewritten
    ok = validate_output(answer(contradiction("L1", "R1", keep="R1", rewrite="在家休息")), pack)
    assert ok.findings[0].rewrite == "在家休息"
    # nobody is named and both are the bot's own words: no rewrite for an undecided loser
    undecided = validate_output(answer(contradiction("L1", "L2", rewrite="x")), pack)
    assert undecided.findings[0].rewrite is None
    # the loser is a real fact (the user's own command outranks it): never rewritten
    real = validate_output(answer(contradiction("F1", "F3", rewrite="x")), pack)
    assert real.findings[0].rewrite is None
    # the loser is a reply that was sent: it cannot be unsaid
    reply = validate_output(answer(contradiction("L1", "R1", keep="L1", rewrite="x")), pack)
    assert reply.findings[0].rewrite is None
    # the loser is a fact the bot invented: it can
    mine = validate_output(answer(contradiction("F1", "F2", rewrite="在北京工作")), pack)
    assert mine.findings[0].rewrite == "在北京工作"  # F2 gives way by the source rule


def test_the_fingerprint_names_the_pair_whichever_comes_first() -> None:
    pack = make_pack()
    a, b, c = pack.items[0], pack.items[2], pack.items[3]
    assert fingerprint_of(a, b) == fingerprint_of(b, a)
    assert fingerprint_of(a, b) != fingerprint_of(a, c)
    assert len(fingerprint_of(a, b)) == 64
    assert fingerprint_of(statement_of(a), statement_of(b)) == fingerprint_of(a, b)


# -------------------------------------------------------------------- who gives way


def test_the_lower_source_gives_way_whatever_the_model_keeps() -> None:
    real, mine, user = (
        evidence("F1", "x", source="real_record"),
        evidence("F2", "y", source="bot_invented"),
        evidence("F3", "z", source="user_said"),
    )
    assert losers_of(real, mine, None) == (mine,) and losers_of(mine, real, "F2") == (mine,)
    assert losers_of(user, mine, "F2") == (mine,)
    assert losers_of(real, user, None) == (user,)
    command = evidence("F4", "w", source="user_command")
    assert losers_of(command, real, None) == (real,)  # /记住 outranks everything


def test_between_equals_the_model_decides_then_what_was_said_stands() -> None:
    life, other, reply = evidence("L1", "a"), evidence("L2", "b"), evidence("R1", "c")
    assert losers_of(life, other, "L1") == (other,) and losers_of(life, other, "L2") == (life,)
    assert losers_of(life, other, None) == (life, other)  # nobody can say: the user chooses
    assert losers_of(life, reply, None) == (life,)  # the reply was sent
    assert losers_of(reply, life, None) == (life,)
    assert losers_of(reply, evidence("R2", "d"), None) == (reply, evidence("R2", "d"))


def test_only_what_the_bot_made_up_may_be_changed() -> None:
    assert is_bot_made(evidence("L1", "a")) and is_bot_made(evidence("R1", "a"))
    assert is_bot_made(evidence("F1", "a", source="bot_invented"))
    for source in ("real_record", "user_said", "user_command"):
        assert not is_bot_made(evidence("F1", "a", source=source))
    assert rank_of(evidence("L1", "a")) == rank_of(evidence("F1", "a", source="bot_invented"))
    assert rank_of(evidence("F1", "a", source="real_record")) > rank_of(evidence("L1", "a"))
    assert rank_of(evidence("F1", "a", source="unheard-of")) == rank_of(evidence("L1", "a"))


# --------------------------------------------------------------------------- the rule


def test_the_limit_is_the_one_in_the_spec() -> None:
    text = SPEC.read_text(encoding="utf-8")
    line = next(row for row in text.splitlines() if "**R-EVAL-004**" in row)
    found = re.search(r"明显矛盾 ≤ (\d+) 次/周", line)
    assert found is not None and int(found.group(1)) == MAX_OBVIOUS_PER_WEEK == 1
    assert WEEK_DAYS == 7


def test_one_obvious_contradiction_in_a_week_passes_and_two_do_not() -> None:
    assert judge_audit(days=7, confirmed_obvious=0, undecided=0)[0] == "passed"
    assert judge_audit(days=7, confirmed_obvious=1, undecided=0)[0] == "passed"
    verdict, why = judge_audit(days=7, confirmed_obvious=2, undecided=0)
    assert verdict == "failed" and "2" in why and "1" in why


def test_longer_windows_allow_one_a_week() -> None:
    assert allowed_obvious(7) == Fraction(1) and allowed_obvious(14) == Fraction(2)
    assert judge_audit(days=14, confirmed_obvious=2, undecided=0)[0] == "passed"
    assert judge_audit(days=14, confirmed_obvious=3, undecided=0)[0] == "failed"
    assert judge_audit(days=10, confirmed_obvious=1, undecided=0)[0] == "passed"
    assert judge_audit(days=10, confirmed_obvious=2, undecided=0)[0] == "failed"  # 10/7 = 1.43


def test_a_window_shorter_than_a_week_cannot_pass() -> None:
    verdict, why = judge_audit(days=3, confirmed_obvious=0, undecided=0)
    assert verdict == "insufficient" and "3" in why


def test_an_audit_with_contradictions_left_to_decide_is_not_finished() -> None:
    verdict, why = judge_audit(days=7, confirmed_obvious=0, undecided=2)
    assert verdict == "insufficient" and "2" in why
    assert judge_audit(days=7, confirmed_obvious=5, undecided=1)[0] == "insufficient"
