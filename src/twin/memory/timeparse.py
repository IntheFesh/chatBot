"""Reading Chinese date and time phrases against the moment of the conversation (R-MEM-006/007).

"明天下午三点考试" means a different instant in Chicago and in Beijing: it depends on the local
date and the local clock **where the conversation took place**.  The extraction model is told
that moment and is asked to write the absolute time, but the arithmetic of "tomorrow" and "next
Wednesday" is done here as well, deterministically, from the time of the evidence message in the
zone of the conversation; when the phrase is one this module reads, its answer wins over the
model's, because a model gets calendar arithmetic wrong now and then and this code does not.

:func:`parse_when` reads one phrase against a reference moment (an aware ``datetime`` in the
conversation's own zone) and returns a :class:`When`: the calendar day, the clock time when the
phrase gives one, and the part of the day when it only says "evening".  Day phrases: 今天 明天
后天 大后天 昨天 前天, 周三 / 下周五 / 这周日 / 下下周一 / 周末, ``3月5日`` / ``2026年3月5号`` /
``3/5`` / ``2026-03-05``, 下个月5号, 三天后 / 两周后 / 一个月后.  Time phrases: 3点 / 三点半 /
下午3点10分 / 15:30 / 晚上8点 and, without an hour, 早上 上午 中午 下午 傍晚 晚上 夜里.

Reading an hour without a part of the day: 1-6 are taken as afternoon ("三点开会"), 7-11 as
morning, 12 as noon, 13-23 as written.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
PERIOD_DEFAULTS: dict[str, time] = {
    "凌晨": time(3, 0),
    "早上": time(8, 0),
    "上午": time(9, 0),
    "中午": time(12, 0),
    "下午": time(15, 0),
    "傍晚": time(18, 0),
    "晚上": time(20, 0),
    "夜里": time(23, 0),
}
_PERIOD_ALIASES = {
    "清晨": "早上",
    "早晨": "早上",
    "早": "早上",
    "午后": "下午",
    "晚": "晚上",
    "夜间": "夜里",
    "半夜": "夜里",
}
DAY_WORDS: dict[str, tuple[int, str | None]] = {
    "大前天": (-3, None),
    "前天": (-2, None),
    "昨天": (-1, None),
    "昨日": (-1, None),
    "昨儿": (-1, None),
    "昨晚": (-1, "晚上"),
    "今天": (0, None),
    "今日": (0, None),
    "今儿": (0, None),
    "今早": (0, "早上"),
    "今晨": (0, "早上"),
    "今晚": (0, "晚上"),
    "今夜": (0, "夜里"),
    "明天": (1, None),
    "明日": (1, None),
    "明儿": (1, None),
    "明早": (1, "早上"),
    "明晨": (1, "早上"),
    "明晚": (1, "晚上"),
    "后天": (2, None),
    "大后天": (3, None),
}
_NUM = r"(?:[0-9]{1,2}|[零〇一二两三四五六七八九十]{1,3})"
_PERIOD = r"(?:凌晨|清晨|早上|早晨|上午|中午|下午|午后|傍晚|晚上|夜里|夜间|半夜|早|晚)"

_FULL_DATE = re.compile(
    r"(?P<y>\d{4})\s*[年/.\-]\s*(?P<m>\d{1,2})\s*[月/.\-]\s*(?P<d>\d{1,2})\s*[日号]?"
)
_MONTH_DAY = re.compile(rf"(?P<m>{_NUM})\s*月\s*(?P<d>{_NUM})\s*(?:[日号]|(?![0-9]))")
_SLASH_DATE = re.compile(r"(?<![0-9/])(?P<m>\d{1,2})/(?P<d>\d{1,2})(?![0-9/])")
_MONTH_REL_DAY = re.compile(rf"(?P<which>下个?月|这个?月|本月|上个?月)\s*(?P<d>{_NUM})\s*[日号]")
_OFFSET = re.compile(
    rf"(?P<n>{_NUM})\s*(?P<unit>个?月|个?星期|个?礼拜|周|天|日)\s*(?:以后|之后|后|内)"
)
_WEEKDAY = re.compile(
    r"(?P<rel>下下|下|上|这|本)?\s*个?\s*(?:周|星期|礼拜)(?P<w>[一二三四五六日天])"
)
_WEEKEND = re.compile(r"(?P<rel>下|这|本)?\s*个?\s*周末")
_CLOCK = re.compile(
    rf"(?P<period>{_PERIOD})?\s*(?P<h>{_NUM})\s*(?:[点时]|[:：])\s*"
    rf"(?:(?P<half>半)|(?P<m>[0-9]{{1,2}}|[零〇一二两三四五六七八九十]{{1,3}})\s*分?|整)?"
)
_PERIOD_ONLY = re.compile(
    r"(?P<period>凌晨|清晨|早上|早晨|上午|中午|下午|午后|傍晚|晚上|夜里|夜间|半夜)"
)


def parse_number(text: str) -> int | None:
    """An integer from digits or Chinese numerals up to 99 (``十`` = 10, ``二十三`` = 23)."""
    text = text.strip()
    if not text:
        return None
    if text.isascii() and text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        head, _, tail = text.partition("十")
        tens = 1 if not head else CN_DIGITS.get(head)
        ones = 0 if not tail else CN_DIGITS.get(tail)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if len(text) == 1:
        return CN_DIGITS.get(text)
    return None


@dataclass(frozen=True)
class When:
    """What a phrase says about time, read against a reference moment."""

    day: date
    clock: time | None  # the time of day the phrase names, if it does
    period: str | None  # "晚上" and the like, when no exact clock time is given
    day_given: bool  # False when only a time was named and the day is the reference day

    def at(self, zone: ZoneInfo, *, default: time) -> datetime:
        """The instant on the wall clock of ``zone``.

        The phrase's own clock time is used first, then the usual time of its part of the day,
        then ``default``.
        """
        chosen = self.clock or (PERIOD_DEFAULTS.get(self.period) if self.period else None)
        wall = chosen or default
        return datetime.combine(self.day, wall, tzinfo=zone)

    @property
    def has_time(self) -> bool:
        return self.clock is not None or self.period is not None


def _add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _year_for(reference: date, month: int, day: int) -> date | None:
    """This year's month/day, or next year's when it is more than ~3 months behind."""
    found = _safe_date(reference.year, month, day)
    if found is None:
        return None
    if (found - reference).days < -90:
        return _safe_date(reference.year + 1, month, day)
    return found


def _read_day(text: str, reference: date) -> tuple[date, str | None] | None:
    """The calendar day a phrase names, and the part of the day the word itself implies."""
    match = _FULL_DATE.search(text)
    if match:
        found = _safe_date(int(match["y"]), int(match["m"]), int(match["d"]))
        if found:
            return found, None
    match = _MONTH_REL_DAY.search(text)
    if match:
        shift = {"下": 1, "这": 0, "本": 0, "上": -1}[match["which"][0]]
        number = parse_number(match["d"])
        base = _add_months(reference.replace(day=1), shift)
        found = _safe_date(base.year, base.month, number) if number else None
        if found:
            return found, None
    match = _MONTH_DAY.search(text)
    if match:
        month, number = parse_number(match["m"]), parse_number(match["d"])
        if month and number:
            found = _year_for(reference, month, number)
            if found:
                return found, None
    match = _SLASH_DATE.search(text)
    if match:
        found = _year_for(reference, int(match["m"]), int(match["d"]))
        if found:
            return found, None
    match = _OFFSET.search(text)
    if match:
        count = parse_number(match["n"])
        unit = match["unit"]
        if count is not None:
            if unit.endswith("月"):
                return _add_months(reference, count), None
            if "星期" in unit or "礼拜" in unit or unit == "周":
                return reference + timedelta(weeks=count), None
            return reference + timedelta(days=count), None
    for word in sorted(DAY_WORDS, key=len, reverse=True):
        if word in text:
            offset, period = DAY_WORDS[word]
            return reference + timedelta(days=offset), period
    match = _WEEKDAY.search(text)
    if match:
        target = WEEKDAYS[match["w"]]
        relation = match["rel"]
        monday = reference - timedelta(days=reference.weekday())
        if relation is None:
            ahead = (target - reference.weekday()) % 7 or 7
            return reference + timedelta(days=ahead), None
        weeks = {"下下": 2, "下": 1, "上": -1, "这": 0, "本": 0}[relation]
        return monday + timedelta(weeks=weeks, days=target), None
    match = _WEEKEND.search(text)
    if match:
        weeks = {"下": 1, None: 0, "这": 0, "本": 0}[match["rel"]]
        monday = reference - timedelta(days=reference.weekday())
        return monday + timedelta(weeks=weeks, days=5), None
    return None


def _hour_of_day(hour: int, period: str | None) -> int | None:
    """The 24-hour clock hour for ``hour`` said in ``period`` (None if it cannot be read)."""
    if hour > 24 or hour < 0:
        return None
    if period is None:
        if 1 <= hour <= 6:
            return hour + 12
        return hour % 24
    if period == "凌晨":
        return 0 if hour == 12 else hour % 24
    if period in ("早上", "上午"):
        return hour % 24
    if period == "中午":
        return hour + 12 if 1 <= hour <= 4 else hour
    if period in ("下午", "傍晚"):
        return hour + 12 if 1 <= hour < 12 else hour % 24
    if period == "晚上":
        if 1 <= hour < 12:
            return hour + 12
        return 0 if hour == 12 else hour % 24
    if period == "夜里":
        if 6 <= hour < 12:
            return hour + 12
        return 0 if hour == 12 else hour % 24
    return hour % 24


def _normalise_period(period: str | None) -> str | None:
    if period is None:
        return None
    return _PERIOD_ALIASES.get(period, period)


def _read_clock(text: str) -> tuple[time | None, str | None]:
    match = _CLOCK.search(text)
    if match:
        hour = parse_number(match["h"])
        period = _normalise_period(match["period"])
        if hour is not None:
            if match["half"]:
                minute: int | None = 30
            elif match["m"]:
                minute = parse_number(match["m"])
            else:
                minute = 0
            real_hour = _hour_of_day(hour, period)
            if real_hour is not None and minute is not None and 0 <= minute <= 59:
                return time(real_hour % 24, minute), period
    only = _PERIOD_ONLY.search(text)
    if only:
        return None, _normalise_period(only["period"])
    return None, None


def parse_when(text: str, reference: datetime) -> When | None:
    """Read ``text`` against ``reference`` (aware, in the zone of the conversation).

    Returns ``None`` when the phrase names neither a day nor a time.
    """
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ValueError("the reference moment needs a time zone")
    day_info = _read_day(text, reference.date())
    clock, period = _read_clock(text)
    if day_info is None and clock is None and period is None:
        return None
    if day_info is None:
        return When(reference.date(), clock, period, False)
    day, implied = day_info
    if clock is None and period is None:
        period = implied
    # an hour past midnight spoken as "晚上12点" belongs to the next calendar day
    if clock is not None and period == "晚上" and clock == time(0, 0):
        day += timedelta(days=1)
    return When(day, clock, period, True)


def parse_local_iso(text: str, zone: ZoneInfo) -> datetime | None:
    """``YYYY-MM-DD HH:MM`` (or with ``T``, or a bare date) as an aware datetime on ``zone``."""
    try:
        parsed = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(zone)
    return parsed.replace(tzinfo=zone)


def parse_iso_date(text: str) -> date | None:
    """``YYYY-MM-DD`` as a date, ``None`` if it is not one."""
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError:
        return None
