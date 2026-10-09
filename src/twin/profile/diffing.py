"""Differences between two profile versions (R-PROF-004).

Every metric is reduced to numbers (a scalar is itself; a distribution to its median, 90th
percentile and mean; a vocabulary to the share of each of its most frequent entries) and the
blended values of the two versions are compared.  A metric *changed* when it moved by more
than ``THRESHOLD`` (10 %) of its old value and by more than a small absolute amount (half a
percentage point for fractions, half a unit for everything else), so that a rate going from
0.1 % to 0.2 % does not count as a doubling of her style.  The result only contains names and
numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from twin.profile.snapshot import PARTIES, ProfileMetrics
from twin.profile.values import Dist, Rates, Scalar

THRESHOLD = 0.10
RATE_FLOOR = 0.005
UNIT_FLOOR = 0.5
TOP_ENTRIES = 20

PARTY_LABELS = {"her": "她", "user": "用户"}
METRIC_LABELS: Mapping[str, str] = {
    "kind_mix": "消息类型占比",
    "text_length": "文字长度",
    "punct_rate": "标点使用率",
    "end_rate": "句末方式占比",
    "emoji_code_rate": "带表情代码的文字比例",
    "emoji_code_freq": "表情代码占比",
    "emoji_code_known_share": "表情代码属于代码表的比例",
    "emoji_code_run_length": "表情代码连用长度",
    "unicode_emoji_rate": "带 emoji 字符的文字比例",
    "unicode_emoji_freq": "emoji 字符占比",
    "sticker_share": "表情包占全部消息",
    "sticker_distinct": "用过的表情包种数",
    "sticker_top3_share": "前 3 种表情包占比",
    "quote_rate": "引用回复占文字类消息",
    "final_particle_rate": "句末带语气词比例",
    "final_particle_freq": "句末语气词分布",
    "laugh_rate": "带笑声的文字比例",
    "laugh_length": "“哈”连写长度",
    "question_rate": "问句率",
    "burst_size": "连发条数",
    "burst_gap_s": "连发条间隔（秒）",
    "typing_s_per_char": "打字速度（每个字的秒数）",
    "reply_latency_s": "回复延迟（秒）",
    "delayed_reply_rate": "隔了一个会话段才回复的比例",
    "closing_no_reply_rate": "对方说完结束性短句后她不回的比例",
    "initiations_per_day": "每天先开口次数",
    "initiation_hour": "先开口时段分布",
    "initiation_silence_s": "先开口前的沉默时长（秒）",
    "messages_per_day": "每天消息数",
    "message_hour": "发消息时段分布",
}
SUFFIX_LABELS = {"median": "中位数", "p90": "90 分位", "mean": "均值"}
PUNCT_LABELS = {
    "comma": "逗号",
    "period": "句号",
    "question": "问号",
    "exclaim": "感叹号",
    "tilde": "波浪号",
    "ellipsis": "省略号",
    "pause": "顿号",
    "space": "空格",
}


@dataclass(frozen=True)
class Change:
    party: str
    metric: str
    before: float
    after: float

    @property
    def ratio(self) -> float:
        if self.before == 0:
            return math.inf if self.after != 0 else 0.0
        return (self.after - self.before) / abs(self.before)

    def to_json(self) -> dict[str, Any]:
        return {
            "party": self.party,
            "metric": self.metric,
            "before": round(self.before, 6),
            "after": round(self.after, 6),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Change:
        return cls(
            str(data["party"]), str(data["metric"]), float(data["before"]), float(data["after"])
        )

    def label(self) -> str:
        name, _, rest = self.metric.partition(".")
        base = METRIC_LABELS.get(name, name)
        if name == "punct_rate":
            rest = PUNCT_LABELS.get(rest, rest)
        elif name == "text_length" or rest in SUFFIX_LABELS:
            rest = SUFFIX_LABELS.get(rest, rest)
        return f"{base}·{rest}" if rest else base

    def describe(self) -> str:
        fractional = max(abs(self.before), abs(self.after)) <= 1.0
        if fractional:
            before, after = f"{self.before * 100:.1f}%", f"{self.after * 100:.1f}%"
        else:
            before, after = f"{self.before:.1f}", f"{self.after:.1f}"
        ratio = "新出现" if math.isinf(self.ratio) else f"{self.ratio * 100:+.0f}%"
        who = PARTY_LABELS.get(self.party, self.party)
        return f"{who} {self.label()}：{before} → {after}（{ratio}）"


def flatten(metrics: ProfileMetrics) -> dict[tuple[str, str], float]:
    """Numeric view of the blended values: ``{(party, metric path): number}``."""
    flat: dict[tuple[str, str], float] = {}
    for party in PARTIES:
        for name in metrics.names(party):
            leaf = metrics.leaf(party, name, "blended")
            if isinstance(leaf, Scalar):
                if leaf.n > 0:
                    flat[(party, name)] = leaf.value
            elif isinstance(leaf, Dist):
                if not leaf.dist.is_empty:
                    flat[(party, f"{name}.median")] = leaf.dist.median()
                    flat[(party, f"{name}.p90")] = leaf.dist.quantile(0.9)
                    flat[(party, f"{name}.mean")] = leaf.dist.mean()
            elif isinstance(leaf, Rates):
                for key, value in leaf.top(TOP_ENTRIES):
                    flat[(party, f"{name}.{key}")] = value
    return flat


def _changed(before: float, after: float) -> bool:
    delta = abs(after - before)
    floor = RATE_FLOOR if max(abs(before), abs(after)) <= 1.0 else UNIT_FLOOR
    if delta <= floor:
        return False
    return before == 0 or delta > THRESHOLD * abs(before)


def diff_metrics(before: ProfileMetrics, after: ProfileMetrics) -> list[Change]:
    """The metrics whose blended value moved by more than 10 % (biggest relative moves first)."""
    old, new = flatten(before), flatten(after)
    changes = [
        Change(party, metric, old.get(key, 0.0), new.get(key, 0.0))
        for key in sorted(set(old) | set(new))
        for party, metric in [key]
        if _changed(old.get(key, 0.0), new.get(key, 0.0))
    ]
    changes.sort(key=lambda c: (-(1e9 if math.isinf(c.ratio) else abs(c.ratio)), c.party, c.metric))
    return changes


def summarize(changes: Sequence[Change], limit: int = 10) -> list[str]:
    """One line per change, at most ``limit`` (her metrics first)."""
    ordered = sorted(changes, key=lambda c: c.party != "her")
    return [change.describe() for change in ordered[:limit]]
