"""Numeric style rules from the profile (R-PROF-005).

The statistics are turned into short Chinese rules ("几乎不用逗号，用分条代替", "单条通常
3–8 字") by a table of templates in ``config/lists/style_rules.yaml`` (setting
``profile.rules_file``).  A rule is ``id`` + ``when`` (a condition over named *facts*) +
``text`` (a format string over the same facts).  Facts are numbers or short strings computed
from the blended metrics of her messages (:func:`rule_facts`); the document of every fact is in
:data:`FACT_DOC`.  The condition language is the safe subset of Python expressions made of
comparisons, ``and``, ``or``, ``not``, numbers and fact names: it is parsed with ``ast`` and
walked by :func:`evaluate`, never executed.  A comparison that mentions a fact that cannot be
computed (no data for it) is false.

The generated text goes to the automatic block of the persona card (R-PERS-002).  It names
only closed vocabularies (emoji codes, emoji characters, particles), never message text.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from twin.profile.snapshot import ProfileMetrics
from twin.profile.values import Rates

FACT_DOC: Mapping[str, str] = {
    "comma_rate": "fraction of text messages with a comma",
    "comma_pct": "comma_rate in percent",
    "period_end_rate": "fraction of text messages ending with a full stop",
    "period_end_pct": "period_end_rate in percent",
    "question_rate": "fraction of text messages that ask something",
    "question_pct": "question_rate in percent",
    "exclaim_rate": "fraction of text messages with an exclamation mark",
    "exclaim_pct": "exclaim_rate in percent",
    "tilde_rate": "fraction of text messages with a tilde",
    "tilde_pct": "tilde_rate in percent",
    "ellipsis_rate": "fraction of text messages with an ellipsis",
    "ellipsis_pct": "ellipsis_rate in percent",
    "length_median": "median characters per text message",
    "length_p25": "25th percentile of characters per text message",
    "length_p75": "75th percentile of characters per text message",
    "length_p90": "90th percentile of characters per text message",
    "length_range": "'p25–p75' (or a single number when they are equal)",
    "burst_median": "median messages per burst",
    "burst_p75": "75th percentile of messages per burst",
    "burst_p90": "90th percentile of messages per burst",
    "burst_range": "'median–p75' (or a single number when they are equal)",
    "burst_gap_median": "median seconds between two messages of one burst",
    "emoji_code_rate": "fraction of text messages with a bracket emoji code",
    "emoji_code_pct": "emoji_code_rate in percent",
    "emoji_code_top": "her three most used emoji codes, written one after another",
    "emoji_code_top_count": "how many emoji codes emoji_code_top lists",
    "emoji_run_p90": "90th percentile of how many codes she writes in a row",
    "unicode_emoji_rate": "fraction of text messages with an emoji character",
    "unicode_emoji_pct": "unicode_emoji_rate in percent",
    "unicode_emoji_top": "her three most used emoji characters",
    "sticker_share": "stickers among all her messages",
    "sticker_pct": "sticker_share in percent",
    "quote_rate": "quotes among her text and quote messages",
    "quote_pct": "quote_rate in percent",
    "laugh_rate": "fraction of text messages with a run of 哈",
    "laugh_length_median": "median length of a run of 哈",
    "final_particle_rate": "fraction of text messages ending with a particle",
    "final_particle_pct": "final_particle_rate in percent",
    "final_particle_top": "her three most used sentence-final particles",
    "final_particle_count": "how many particles final_particle_top lists",
    "latency_median_s": "median seconds until she answers",
    "latency_p90_s": "90th percentile of seconds until she answers",
}
FACT_NAMES = frozenset(FACT_DOC)

Fact = float | str | None


class RuleError(ValueError):
    """The rule table is malformed."""


@dataclass(frozen=True)
class Rule:
    id: str
    when: str
    text: str


# ------------------------------------------------------------------ the facts


def _pct(rate: float | None) -> float | None:
    return None if rate is None else rate * 100.0


def _range(low: float | None, high: float | None) -> str | None:
    if low is None or high is None:
        return None
    a, b = round(low), round(max(high, low))
    return f"{a}" if a == b else f"{a}–{b}"


def rule_facts(metrics: ProfileMetrics, party: str = "her") -> dict[str, Fact]:
    """The facts of ``party`` from the blended metrics (``None`` where there is no data)."""
    facts: dict[str, Fact] = dict.fromkeys(FACT_NAMES)

    def rate(name: str, key: str | None = None) -> float | None:
        if key is None:
            return metrics.scalar(party, name)
        leaf = metrics.leaf(party, name)
        if not isinstance(leaf, Rates) or leaf.n == 0:
            return None
        return leaf.values.get(key, 0.0)

    def quantiles(name: str, *qs: float) -> list[float | None]:
        dist = metrics.distribution(party, name)
        return [dist.quantile(q) if dist is not None else None for q in qs]

    for key, source in (
        ("comma", "comma"),
        ("exclaim", "exclaim"),
        ("tilde", "tilde"),
        ("ellipsis", "ellipsis"),
    ):
        value = rate("punct_rate", source)
        facts[f"{key}_rate"] = value
        facts[f"{key}_pct"] = _pct(value)
    period = rate("end_rate", "period")
    facts["period_end_rate"] = period
    facts["period_end_pct"] = _pct(period)
    question = rate("question_rate")
    facts["question_rate"] = question
    facts["question_pct"] = _pct(question)

    median, p25, p75, p90 = quantiles("text_length", 0.5, 0.25, 0.75, 0.9)
    facts.update(
        length_median=median,
        length_p25=p25,
        length_p75=p75,
        length_p90=p90,
        length_range=_range(p25, p75),
    )
    b_median, b_p75, b_p90 = quantiles("burst_size", 0.5, 0.75, 0.9)
    facts.update(
        burst_median=b_median,
        burst_p75=b_p75,
        burst_p90=b_p90,
        burst_range=_range(b_median, b_p75),
    )
    (gap_median,) = quantiles("burst_gap_s", 0.5)
    facts["burst_gap_median"] = gap_median

    code_rate = rate("emoji_code_rate")
    facts["emoji_code_rate"] = code_rate
    facts["emoji_code_pct"] = _pct(code_rate)
    top_codes = [code for code, _ in _top(metrics.rates(party, "emoji_code_freq"), 3)]
    facts["emoji_code_top"] = "".join(top_codes) if top_codes else None
    facts["emoji_code_top_count"] = float(len(top_codes)) if code_rate is not None else None
    (run_p90,) = quantiles("emoji_code_run_length", 0.9)
    facts["emoji_run_p90"] = run_p90

    emoji_rate = rate("unicode_emoji_rate")
    facts["unicode_emoji_rate"] = emoji_rate
    facts["unicode_emoji_pct"] = _pct(emoji_rate)
    top_emoji = [e for e, _ in _top(metrics.rates(party, "unicode_emoji_freq"), 3)]
    facts["unicode_emoji_top"] = "".join(top_emoji) if top_emoji else None

    sticker = rate("sticker_share")
    facts["sticker_share"] = sticker
    facts["sticker_pct"] = _pct(sticker)
    quote = rate("quote_rate")
    facts["quote_rate"] = quote
    facts["quote_pct"] = _pct(quote)

    laugh = rate("laugh_rate")
    facts["laugh_rate"] = laugh
    (laugh_median,) = quantiles("laugh_length", 0.5)
    facts["laugh_length_median"] = laugh_median

    particle_rate = rate("final_particle_rate")
    facts["final_particle_rate"] = particle_rate
    facts["final_particle_pct"] = _pct(particle_rate)
    top_particles = [p for p, _ in _top(metrics.rates(party, "final_particle_freq"), 3)]
    facts["final_particle_top"] = "、".join(top_particles) if top_particles else None
    facts["final_particle_count"] = float(len(top_particles)) if particle_rate is not None else None

    latency_median, latency_p90 = quantiles("reply_latency_s", 0.5, 0.9)
    facts["latency_median_s"] = latency_median
    facts["latency_p90_s"] = latency_p90
    return facts


def _top(values: Mapping[str, float], count: int) -> list[tuple[str, float]]:
    ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return [item for item in ranked[:count] if item[1] > 0]


# ------------------------------------------------------- the condition language

_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def parse_condition(expression: str) -> ast.expr:
    """Parse ``expression`` and reject everything outside the condition language."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise RuleError(f"invalid condition {expression!r}: {exc.msg}") from exc
    _check(tree.body, expression)
    return tree.body


def _check(node: ast.expr, expression: str) -> None:
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _check(value, expression)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not | ast.USub):
        _check(node.operand, expression)
    elif isinstance(node, ast.Compare):
        if not all(type(op) in _COMPARE for op in node.ops):
            raise RuleError(f"unsupported comparison in {expression!r}")
        for part in (node.left, *node.comparators):
            _check(part, expression)
    elif isinstance(node, ast.Name):
        if node.id not in FACT_NAMES:
            raise RuleError(f"unknown fact {node.id!r} in {expression!r}")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, int | float | str):
            raise RuleError(f"unsupported constant in {expression!r}")
    else:
        raise RuleError(f"unsupported syntax {type(node).__name__} in {expression!r}")


def evaluate(node: ast.expr, facts: Mapping[str, Fact]) -> Any:
    """Value of a checked condition; comparisons with a missing fact are false."""
    if isinstance(node, ast.BoolOp):
        results = (evaluate(v, facts) for v in node.values)
        return all(results) if isinstance(node.op, ast.And) else any(results)
    if isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand, facts)
        if isinstance(node.op, ast.Not):
            return not value
        return None if value is None else -value
    if isinstance(node, ast.Compare):
        left = evaluate(node.left, facts)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = evaluate(comparator, facts)
            if left is None or right is None or not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        return facts.get(node.id)
    if isinstance(node, ast.Constant):
        return node.value
    raise RuleError(f"unsupported syntax {type(node).__name__}")  # unreachable after _check


# --------------------------------------------------------------- the rule table


def parse_rules(text: str, *, source: str = "rule table") -> list[Rule]:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuleError(f"{source}: not valid YAML: {exc}") from exc
    entries = document.get("rules") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RuleError(f"{source}: expected a non-empty 'rules' list")
    rules: list[Rule] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not {"id", "when", "text"} <= set(entry):
            raise RuleError(f"{source}: rule {position} needs id, when and text")
        rule = Rule(str(entry["id"]), str(entry["when"]), str(entry["text"]))
        if rule.id in seen:
            raise RuleError(f"{source}: rule id {rule.id!r} is used twice")
        seen.add(rule.id)
        parse_condition(rule.when)
        _check_template(rule)
        rules.append(rule)
    return rules


def _check_template(rule: Rule) -> None:
    probe = dict.fromkeys(FACT_NAMES, 1.0)
    try:
        rule.text.format_map(probe)
    except (KeyError, IndexError, ValueError) as exc:
        raise RuleError(f"rule {rule.id!r}: bad text template ({exc})") from exc


def load_rules(path: Path) -> list[Rule]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleError(f"cannot read the rule table {path}: {exc}") from exc
    return parse_rules(text, source=str(path))


def generate_rules(metrics: ProfileMetrics, rules: list[Rule], party: str = "her") -> list[str]:
    """The rule sentences whose condition holds, in table order."""
    facts = rule_facts(metrics, party)
    lines: list[str] = []
    for rule in rules:
        if not evaluate(parse_condition(rule.when), facts):
            continue
        try:
            lines.append(rule.text.format_map(facts))
        except (KeyError, ValueError, TypeError):
            continue  # a fact the text needs is missing: the rule says nothing rather than a gap
    return lines
