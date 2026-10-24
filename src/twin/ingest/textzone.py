"""The zone the export's ``createTimeText`` is written in, inferred from the export (D-900, D-901).

Every message carries ``createTime`` (the moment) and ``createTimeText`` (that moment on the wall
clock of the *exporting device*).  Their difference is the UTC offset of the device's zone at that
moment, so the zone is inferred from the export itself (user decision of round 18, item C7): it is
the zone whose offset is the observed difference at the moment of *every* message, daylight saving
time included (``zoneinfo`` says what the offset was).  The zone has nothing to do with where she
was (``time.source_timezone`` and its ranges, R-ACT-001) and serves two things only: the
consistency check of the import report and of ``twin import inspect``, and reading the text of a
message that has no ``createTime``.  No configuration key: the export says it.

**Counting.**  An import streams the messages; the pairs are reduced as they pass to *spans*: per
observed offset (whole minutes, snapped to the nearest quarter hour when within a minute of it - a
text without seconds, a ``createTime`` in milliseconds) and UTC date, the first and the last moment
and a count (:class:`TextZoneTally`).  Years of chat are a few thousand spans; they are saved with
the run's counters, so a resumed run goes on counting.  More than :data:`MAX_SPANS` spans cannot be
one zone (that would be over fifty years of days), so the tally stops growing there and infers
nothing.

**Fitting.**  A zone fits a span if its offset at the span's first and its last moment is the
observed one; no zone changes its offset twice within one UTC day, so that is the same as fitting
every message of the span.  The candidates are :func:`candidate_zones` - the zones of the
geographic areas of the IANA database (about 450) and ``UTC``; each is tested with an early exit
at the first span it does not fit, which is where almost all of them fail, so inferring takes
milliseconds.  Zones with identical rules over the data (America/Chicago and America/Winnipeg) fit
alike and are one answer; it is named after the first fitting zone of ``preferred`` (the zones of
the configuration), then of :data:`COMMON_ZONES`, then alphabetically, and the report says how
many others fit as well.  When none fits, the offsets and their counts are what is shown.
"""

from __future__ import annotations

import functools
import zoneinfo
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from twin.config.settings import TimeConfig
from twin.ingest.times import wall_clock_as_utc

GEOGRAPHIC_AREAS = (
    "Africa/",
    "America/",
    "Antarctica/",
    "Asia/",
    "Atlantic/",
    "Australia/",
    "Europe/",
    "Indian/",
    "Pacific/",
)
COMMON_ZONES = (
    "America/Chicago",
    "Asia/Shanghai",
    "America/New_York",
    "America/Los_Angeles",
    "America/Denver",
    "America/Phoenix",
    "America/Anchorage",
    "Pacific/Honolulu",
    "Asia/Hong_Kong",
    "Asia/Taipei",
    "Asia/Tokyo",
    "Asia/Seoul",
    "Asia/Singapore",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Australia/Sydney",
    "UTC",
)
SNAP_S = 60.0  # within a minute of a quarter hour: a text without seconds or an epoch in ms
QUARTER_S = 900
DAY_S = 86_400
MAX_SPANS = 20_000
SHOWN_OFFSETS = 12  # the report lists this many offsets, the rest as one "other" line
FALLBACK_HER_ZONE = ""  # time_from_text key: read on her clock, no device zone known

ZoneSource = Callable[[], ZoneInfo | None]  # the device zone known so far (TextZoneReader.zone)


def observed_offset_minutes(moment: datetime, text: str | None) -> int | None:
    """The writer's UTC offset (minutes) of a text for ``moment``; ``None`` if unreadable.

    Within :data:`SNAP_S` of a quarter hour the offset is that quarter hour (every zone's offset
    in this century is one); anything else is kept to the minute, so it fits no zone.
    """
    reading = wall_clock_as_utc(text)
    if reading is None:
        return None
    seconds = (reading - moment.astimezone(UTC)).total_seconds()
    quarter = round(seconds / QUARTER_S) * QUARTER_S
    if abs(seconds - quarter) <= SNAP_S:
        return quarter // 60
    return round(seconds / 60)


def format_offset(minutes: int) -> str:
    """``+08:00`` / ``-05:00`` / ``+05:45``."""
    sign = "-" if minutes < 0 else "+"
    hours, rest = divmod(abs(minutes), 60)
    return f"{sign}{hours:02d}:{rest:02d}"


def configured_zones(time: TimeConfig) -> tuple[str, ...]:
    """The zones of the configuration: they name an inferred text zone when they fit."""
    ranges = (item.tz for item in time.source_timezone_ranges)
    return tuple(dict.fromkeys((time.source_timezone, *ranges)))


@functools.cache
def candidate_zones() -> tuple[str, ...]:
    """The zones a device can be set to: the geographic IANA zones and ``UTC`` (sorted)."""
    names = {name for name in zoneinfo.available_timezones() if name.startswith(GEOGRAPHIC_AREAS)}
    names.add("UTC")
    return tuple(sorted(names))


@functools.cache
def _zone(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        return None


@dataclass
class TextZoneTally:
    """The pairs of ``createTime`` and ``createTimeText`` seen so far, reduced to spans."""

    # (offset minutes, UTC day number) -> [first moment, last moment] (epoch seconds), count
    spans: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    offsets: Counter[int] = field(default_factory=Counter)  # offset minutes -> pairs
    pairs: int = 0
    unreadable: int = 0  # a createTime with a text that is missing or not a clock time
    overflow: int = 0  # pairs whose span no longer fitted under MAX_SPANS
    version: int = 0  # changes with every observation (for cached inferences)

    def observe(self, moment: datetime | None, text: object) -> None:
        """Count one message (``moment`` from ``createTime``; ``None`` if it had none)."""
        if moment is None:
            return
        offset = observed_offset_minutes(moment, text) if isinstance(text, str) else None
        if offset is None:
            self.unreadable += 1
            return
        stamp = moment.timestamp()
        key = (offset, int(stamp // DAY_S))
        self.pairs += 1
        self.offsets[offset] += 1
        self.version += 1
        span = self.spans.get(key)
        if span is None:
            if len(self.spans) >= MAX_SPANS:
                self.overflow += 1
                return
            self.spans[key] = [stamp, stamp, 1]
            return
        span[0] = min(span[0], stamp)
        span[1] = max(span[1], stamp)
        span[2] += 1

    def to_json(self) -> dict[str, Any]:
        return {
            "spans": [[o, d, a, b, int(n)] for (o, d), (a, b, n) in sorted(self.spans.items())],
            "offsets": {str(o): n for o, n in sorted(self.offsets.items())},
            "pairs": self.pairs,
            "unreadable": self.unreadable,
            "overflow": self.overflow,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> TextZoneTally:
        if not data:
            return cls()
        return cls(
            spans={
                (int(o), int(d)): [float(a), float(b), int(n)]
                for o, d, a, b, n in data.get("spans", [])
            },
            offsets=Counter({int(o): int(n) for o, n in data.get("offsets", {}).items()}),
            pairs=int(data.get("pairs", 0)),
            unreadable=int(data.get("unreadable", 0)),
            overflow=int(data.get("overflow", 0)),
        )


@dataclass(frozen=True)
class TextZoneInference:
    """What the export says about the zone of its texts."""

    zone: str | None  # the answer's name; None when no zone fits every message
    fitting: tuple[str, ...]  # every candidate that fits (zones with the same rules)
    pairs: int
    offsets: tuple[tuple[int, int], ...]  # (offset minutes, pairs), the most frequent first
    overflow: bool = False

    @property
    def equivalents(self) -> int:
        return max(0, len(self.fitting) - 1)

    def offset_rows(self) -> list[tuple[str, int]]:
        """``(+08:00, n)`` rows, the rarer ones beyond :data:`SHOWN_OFFSETS` as one ``其他`` row."""
        rows = [(format_offset(o), n) for o, n in self.offsets[:SHOWN_OFFSETS]]
        rest = sum(n for _, n in self.offsets[SHOWN_OFFSETS:])
        if rest:
            rows.append(("其他", rest))
        return rows

    def report_lines(self) -> list[str]:
        """Lines for the import report (counts, offsets and zone names only)."""
        if self.pairs == 0:
            return [
                "- 导出里没有同时带 createTime 与可读 createTimeText 的消息，推断不出导出文字时区"
            ]
        if self.zone is not None:
            shifts = "、".join(format_offset(o) for o, _ in self.offsets)
            also = (
                f"；另有 {self.equivalents} 个时区在这段时间里与它完全等价"
                if self.equivalents
                else ""
            )
            return [
                f"- 导出文字（createTimeText）所用时区：**{self.zone}**——{self.pairs:,} 条消息的 "
                f"createTime 与 createTimeText 全部吻合（时差 {shifts}，夏令时按 zoneinfo 判断）"
                f"{also}",
                "- 这只是导出设备的时钟，只用于这项一致性检查与缺 createTime 时的还原；"
                "她在哪由 time.source_timezone / source_timezone_ranges 表示，"
                "配错时由作息合理性检查发现（`twin profile show`）",
            ]
        cause = "（不同的时差太多，已停止统计）" if self.overflow else ""
        distribution = "、".join(f"{label} × {n:,}" for label, n in self.offset_rows())
        return [
            f"- 推断不出导出文字时区{cause}：没有一个时区能让全部 {self.pairs:,} 条消息的 "
            "createTime 与 createTimeText 吻合",
            f"- 时差分布（createTimeText − UTC）：{distribution}",
            "- 请确认：导出是否合并自不同时区的设备、或导出工具写错了文字时间。"
            "存储的时间以 createTime 为准，不受影响",
        ]


def _fits(zone: ZoneInfo, checks: Sequence[tuple[float, int]]) -> bool:
    for stamp, offset in checks:
        found = datetime.fromtimestamp(stamp, zone).utcoffset()
        if found is None or found.total_seconds() != offset * 60:
            return False
    return True


def _checks(spans: dict[tuple[int, int], list[float]]) -> list[tuple[float, int]]:
    """``(moment, offset)`` to test: one per distinct offset first (where most zones fail)."""
    first: list[tuple[float, int]] = []
    rest: list[tuple[float, int]] = []
    seen: set[int] = set()
    for (offset, _day), (start, end, _count) in sorted(spans.items(), key=lambda kv: -kv[1][2]):
        target = rest
        if offset not in seen:
            seen.add(offset)
            target = first
        target.append((start, offset))
        if end != start:
            rest.append((end, offset))
    return first + rest


def _representative(fitting: Sequence[str], preferred: Iterable[str]) -> str:
    chosen = set(fitting)
    for name in (*preferred, *COMMON_ZONES):
        if name in chosen:
            return name
    return fitting[0]


def infer_text_zone(tally: TextZoneTally, preferred: Iterable[str] = ()) -> TextZoneInference:
    """The zone every counted pair fits (see the module description)."""
    offsets = tuple(sorted(tally.offsets.items(), key=lambda kv: (-kv[1], kv[0])))
    if tally.pairs == 0:
        return TextZoneInference(None, (), 0, ())
    if tally.overflow:
        return TextZoneInference(None, (), tally.pairs, offsets, overflow=True)
    checks = _checks(tally.spans)
    preferred = tuple(preferred)
    fitting: list[str] = []
    for name in dict.fromkeys((*preferred, *candidate_zones())):
        zone = _zone(name)
        if zone is not None and _fits(zone, checks):
            fitting.append(name)
    if not fitting:
        return TextZoneInference(None, (), tally.pairs, offsets)
    ordered = tuple(sorted(fitting))
    return TextZoneInference(_representative(ordered, preferred), ordered, tally.pairs, offsets)


class TextZoneReader:
    """The zone inferred from a tally so far, recomputed only when the tally has changed."""

    def __init__(self, tally: TextZoneTally, preferred: Iterable[str] = ()) -> None:
        self._tally = tally
        self._preferred = tuple(preferred)
        self._version = -1
        self._zone: ZoneInfo | None = None

    def zone(self) -> ZoneInfo | None:
        if self._version != self._tally.version:
            self._version = self._tally.version
            name = infer_text_zone(self._tally, self._preferred).zone
            self._zone = _zone(name) if name is not None else None
        return self._zone
