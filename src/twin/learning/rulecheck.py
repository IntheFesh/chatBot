"""Is a candidate rule only about the way she talks? (R-LRN-003, R-TRN-013)

A rule of ``[不要这样]`` goes into the persona card that the DeepSeek backend reads in every reply
and that the evaluation sandbox adds to its pre-holdout card, on the argument that it "only
describes a manner of speaking" (R-TRN-013).  This module is the local half of the check that keeps
that argument true; the other half is DeepSeek's second judgement (:mod:`twin.learning.rules`).

A rule is **refused** when it

* is empty, shorter than 4 or longer than ``learning.rule_max_chars`` characters, or has several
  lines;
* holds a digit, a date or a clock time (``3月5日``, ``下午三点``, ``周三``), a relative day
  (``明天``, ``上周``), a time of day (``早上``, ``晚上``) or an amount of money;
* names an event (a test, a trip, a birthday, an illness, ...);
* holds personal data the redaction finds (phone numbers, addresses, ...), or the words of a
  place or a person (``在…路``, ``…大学``, ``名叫…``, ``姓…``);
* is a question.

What it cannot tell - whether a rule that passes is about style at all - is left to the model.
Each refusal has a ``kind`` that the weekly job counts: ``fact`` (a time, an amount, an event),
``name`` (a person or a place) or ``other``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from twin.llm.redaction import redact

MIN_RULE_CHARS = 4

_NUMBER_WORDS = "零〇一二两三四五六七八九十百千万亿"
_TIME_MARKERS = (
    rf"[{_NUMBER_WORDS}]+(?:点钟|点半|点整|点多|号|日|月|年|块|元|刀|万|岁|天|周|个月|小时|分钟)"
    r"|周[一二三四五六日天末]|星期|礼拜|周末|节日|假期|假日|春节|国庆|中秋|元旦|圣诞"
    r"|昨天|今天|明天|后天|前天|昨晚|今晚|明早|昨日|今日|明日|上周|下周|这周|本周"
    r"|上个月|下个月|这个月|本月|去年|今年|明年|早上|早晨|晚上|中午|下午|凌晨|深夜|傍晚"
)
_MONEY = r"[¥￥$€£]|块钱|美元|美金|人民币|rmb|usd|多少钱|价格|工资|房租"
_EVENTS = (
    r"考试|面试|考研|答辩|论文|旅游|旅行|出差|出国|回国|生病|感冒|发烧|住院|搬家|毕业|结婚|分手"
    r"|生日|约会|聚餐|开会|加班|相亲|订婚|怀孕|手术|航班|机票|演唱会|放假"
)
_PLACES = (
    r"(?:在|去|到|住在|来自)[^，。,.\s]{1,8}(?:省|市|区|县|路|街|镇|村|楼|室|大学|学院|学校|公司"
    r"|医院|机场|车站|酒店|商场|公园|餐厅|店)"
    r"|[^，。,.\s]{1,6}(?:大学|学院|医院|公司|机场|酒店)"
)
_PEOPLE = r"(?:名叫|姓|外号|绰号)[^，。,.\s]{1,4}"

_DIGIT = re.compile(r"[0-9０-９]")
_TIME = re.compile(_TIME_MARKERS)
_MONEY_RE = re.compile(_MONEY, re.IGNORECASE)
_EVENT = re.compile(_EVENTS)
_PLACE = re.compile(_PLACES)
_PERSON = re.compile(_PEOPLE)
_QUESTION = re.compile(r"[?？]|吗$|呢$")
_LEAD = re.compile(r"^\s*(?:[-*•·]|\d+[.、)）])\s*")


@dataclass(frozen=True)
class RuleVerdict:
    """The local verdict on one rule: ``ok``, or why not (``kind`` and ``reason``)."""

    ok: bool
    kind: str = ""
    reason: str = ""


def clean_rule(text: str) -> str:
    """The rule as one tidy line: bullet and numbering removed, white space collapsed."""
    return " ".join(_LEAD.sub("", text).split())


def check_rule(text: str, *, max_chars: int) -> RuleVerdict:
    """The local verdict on ``text`` (see the module description)."""
    if "\n" in text.strip():
        return RuleVerdict(False, "other", "several lines")
    rule = clean_rule(text)
    if len(rule) < MIN_RULE_CHARS:
        return RuleVerdict(False, "other", "too short")
    if len(rule) > max_chars:
        return RuleVerdict(False, "other", "too long")
    if _QUESTION.search(rule):
        return RuleVerdict(False, "other", "a question")
    if _DIGIT.search(rule):
        return RuleVerdict(False, "fact", "a number")
    if _TIME.search(rule):
        return RuleVerdict(False, "fact", "a date or a time")
    if _MONEY_RE.search(rule):
        return RuleVerdict(False, "fact", "an amount of money")
    if _EVENT.search(rule):
        return RuleVerdict(False, "fact", "an event")
    if redact(rule).changed:
        return RuleVerdict(False, "name", "personal data")
    if _PLACE.search(rule):
        return RuleVerdict(False, "name", "a place")
    if _PERSON.search(rule):
        return RuleVerdict(False, "name", "a person")
    return RuleVerdict(True)
