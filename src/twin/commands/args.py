"""Reading the arguments of the commands: durations, clock ranges, dates, zones (R-CMD-003).

Everything here is a pure function of its text (and of the moment or the day it is told), so the
rules of what a command accepts are tested without a chat.  Arguments are compared after
:func:`twin.commands.parse.fold` (full-width digits and colons become ASCII, case does not
matter), so ``２小时``, ``22：00`` and ``Asia/Shanghai`` in capitals all read.

``parse_pause``
    ``2小时``, ``半小时``, ``一个半小时``, ``2小时30分钟``, ``90min``, ``1天`` (a length of time)
    and ``到明早``, ``到22:00``, ``到下午3点`` (the next time the clock shows that).  "Morning"
    is the hour ``commands.morning_hour``; it is always the next such hour, so at 01:30 "到明早"
    ends at 08:00 of the same day - a person who says it then means the morning that is coming.
``parse_clock_range``
    ``01:00-09:30`` (also ``~``, ``至``, ``到``).
``parse_date_range``
    ``2026-10-01``, ``2026-10-01..2026-10-07``, ``10月1日``, ``10/1至10/7``; a date without a year
    is the next such date from today (at most a month in the past counts as this year's).
``resolve_zone``
    an IANA name in any case or an alias (``芝加哥``, ``北京``, ``上海``, ``中国``, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import available_timezones

from twin.commands.parse import fold
from twin.profile.localtime import format_minute, parse_clock

ZONE_ALIASES: dict[str, str] = {
    "芝加哥": "America/Chicago",
    "美中": "America/Chicago",
    "chicago": "America/Chicago",
    "纽约": "America/New_York",
    "洛杉矶": "America/Los_Angeles",
    "丹佛": "America/Denver",
    "北京": "Asia/Shanghai",
    "上海": "Asia/Shanghai",
    "中国": "Asia/Shanghai",
    "国内": "Asia/Shanghai",
    "北京时间": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "china": "Asia/Shanghai",
}
RANGE_SEPARATORS = "-–—~至到"
_DATE_SEPARATORS = ("..", "~", "至", "到", "—", "–")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6}
_CN_DIGITS |= {"七": 7, "八": 8, "九": 9}
_NUMBER = r"\d+(?:\.\d+)?|[零〇一二两三四五六七八九十]+"
_UNITS = {
    "小时": 60.0,
    "钟头": 60.0,
    "h": 60.0,
    "hr": 60.0,
    "hrs": 60.0,
    "hour": 60.0,
    "hours": 60.0,
    "分钟": 1.0,
    "分": 1.0,
    "min": 1.0,
    "mins": 1.0,
    "minute": 1.0,
    "minutes": 1.0,
    "m": 1.0,
    "天": 1440.0,
    "日": 1440.0,
    "d": 1440.0,
    "day": 1440.0,
    "days": 1440.0,
}
_PART = re.compile(rf"(?P<n>{_NUMBER})(?:个)?(?P<unit>小时|钟头|分钟|分|天|日|[a-z]+)")
_HALF_MORE = re.compile(rf"(?P<n>{_NUMBER})个半(?P<unit>小时|钟头|天)")
_MORNING_WORDS = frozenset(
    {"明早", "明天早上", "明天早晨", "明早上", "明晨", "明天", "早上", "早晨", "天亮", "早"}
)
_PERIODS = {"凌晨": "am", "早上": "am", "早晨": "am", "上午": "am", "中午": "noon", "下午": "pm"}
_PERIODS |= {"晚上": "pm", "傍晚": "pm", "今晚": "pm", "夜里": "pm"}
_CLOCK_WORDS = re.compile(
    r"^(?P<period>凌晨|早上|早晨|上午|中午|下午|晚上|傍晚|今晚|夜里)?"
    r"(?P<hour>\d{1,2})(?:(?::|点|时)(?P<minute>\d{1,2}|半)?分?)?钟?$"
)


class ArgumentError(ValueError):
    """An argument that a command cannot read; the message says why (it goes to the user)."""


def cn_number(text: str) -> float | None:
    """The value of ``12``, ``2.5``, ``两``, ``十二`` or ``三十五``; ``None`` if it is neither."""
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if not text or not all(char in _CN_DIGITS or char == "十" for char in text):
        return None
    if "十" not in text:
        if len(text) != 1:
            return None
        return float(_CN_DIGITS[text])
    head, _, tail = text.partition("十")
    if len(head) > 1 or len(tail) > 1 or "十" in tail:
        return None
    tens = _CN_DIGITS[head] if head else 1
    ones = _CN_DIGITS[tail] if tail else 0
    return float(tens * 10 + ones)


# ------------------------------------------------------------------------------- pausing


@dataclass(frozen=True)
class PauseTarget:
    """Until when the bot is paused (an aware UTC moment)."""

    until: datetime


def _length_minutes(text: str) -> float | None:
    """A length of time in minutes (``2小时30分钟``, ``半小时``), or ``None`` if it is none."""
    if text in {"半小时", "半个小时", "半钟头"}:
        return 30.0
    if text in {"半天"}:
        return 720.0
    half = _HALF_MORE.fullmatch(text)
    if half:
        number = cn_number(half["n"])
        if number is None:
            return None
        return (number + 0.5) * _UNITS[half["unit"]]
    total, position = 0.0, 0
    for part in _PART.finditer(text):
        if part.start() != position:
            return None
        number, unit = cn_number(part["n"]), _UNITS.get(part["unit"])
        if number is None or unit is None:
            return None
        total += number * unit
        position = part.end()
    return total if position == len(text) and position > 0 else None


def _next_clock(now: datetime, hour: int, minute: int) -> datetime:
    """The next moment after ``now`` at which the clock of ``now``'s zone shows ``hour:minute``."""
    candidate = datetime.combine(now.date(), time(hour, minute), tzinfo=now.tzinfo)
    if candidate <= now:
        candidate = datetime.combine(
            now.date() + timedelta(days=1), time(hour, minute), tzinfo=now.tzinfo
        )
    return candidate


def _spoken_clock(text: str) -> tuple[int, int] | None:
    """``22:00``, ``10点半``, ``下午3点`` as (hour, minute); ``None`` if it is not a clock time."""
    found = _CLOCK_WORDS.fullmatch(text)
    if found is None:
        return None
    hour = int(found["hour"])
    minute_text = found["minute"]
    minute = 30 if minute_text == "半" else int(minute_text) if minute_text else 0
    period = _PERIODS.get(found["period"] or "")
    if (period == "pm" and hour < 12) or (period == "noon" and hour < 11):
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def parse_pause(
    text: str, now: datetime, *, morning_hour: int = 8, max_hours: int = 168
) -> PauseTarget:
    """The end of a pause asked for with ``text`` (see the module description).

    ``now`` is an aware moment on the clock the words are meant on (the bot's local time).
    """
    spoken = fold(text).replace(" ", "").replace("　", "")
    if not spoken:
        raise ArgumentError("unreadable")
    until: datetime
    if spoken.startswith(("到", "直到", "至")):
        rest = spoken.removeprefix("直到").removeprefix("到").removeprefix("至")
        if rest in _MORNING_WORDS:
            until = _next_clock(now, morning_hour, 0)
        else:
            clock = _spoken_clock(rest)
            if clock is None:
                raise ArgumentError("unreadable")
            until = _next_clock(now, *clock)
    else:
        minutes = _length_minutes(spoken)
        if minutes is None:
            raise ArgumentError("unreadable")
        if minutes <= 0:
            raise ArgumentError("not_future")
        until = now.astimezone(UTC) + timedelta(minutes=minutes)  # absolute, DST-proof
    if until <= now:
        raise ArgumentError("not_future")
    if until - now > timedelta(hours=max_hours):
        raise ArgumentError("too_long")
    return PauseTarget(until.astimezone(UTC))


# ------------------------------------------------------------------- clock times and dates


def split_range(text: str) -> tuple[str, str]:
    """``a-b``, ``a~b``, ``a至b`` as (``a``, ``b``)."""
    spoken = fold(text).replace(" ", "")
    pieces = [p for p in re.split(f"[{RANGE_SEPARATORS}]", spoken) if p]
    if len(pieces) != 2:
        raise ArgumentError("a range needs a start and an end, like 01:00-09:30")
    return pieces[0], pieces[1]


def _clock_text(text: str) -> str:
    clock = _spoken_clock(text)
    if clock is None:
        raise ArgumentError(f"not a clock time: {text}")
    return format_minute(clock[0] * 60 + clock[1])


def parse_clock_range(text: str) -> tuple[str, str]:
    """``01:00-09:30`` as ``("01:00", "09:30")`` (24-hour clock, the bot's local time)."""
    first, second = split_range(text)
    start, end = _clock_text(first), _clock_text(second)
    parse_clock(start)
    parse_clock(end)
    return start, end


_FULL_DATE = re.compile(r"^(?P<y>\d{4})[-/.年](?P<m>\d{1,2})[-/.月](?P<d>\d{1,2})日?$")
_SHORT_DATE = re.compile(r"^(?P<m>\d{1,2})[-/月](?P<d>\d{1,2})日?$")


def parse_one_date(text: str, today: date) -> date:
    """One date (see the module description); a short date is the next one from ``today``."""
    spoken = fold(text).replace(" ", "")
    full = _FULL_DATE.fullmatch(spoken)
    try:
        if full:
            return date(int(full["y"]), int(full["m"]), int(full["d"]))
        short = _SHORT_DATE.fullmatch(spoken)
        if short is None:
            raise ArgumentError(f"not a date: {text}")
        month, day = int(short["m"]), int(short["d"])
        found = date(today.year, month, day)
        if found < today - timedelta(days=30):
            found = date(today.year + 1, month, day)
        return found
    except ValueError as exc:
        if isinstance(exc, ArgumentError):
            raise
        raise ArgumentError(f"not a date: {text}") from None


def parse_date_range(text: str, today: date) -> tuple[date, date]:
    """One date or a range of two (``a..b``, ``a至b``); the first is not after the last."""
    spoken = fold(text).replace(" ", "")
    if not spoken:
        raise ArgumentError("a date is needed")
    for separator in _DATE_SEPARATORS:
        if separator in spoken:
            first, _, last = spoken.partition(separator)
            if not first or not last:
                raise ArgumentError("a range needs two dates")
            begin, finish = parse_one_date(first, today), parse_one_date(last, today)
            if finish < begin:
                raise ArgumentError("the range ends before it starts")
            return begin, finish
    day = parse_one_date(spoken, today)
    return day, day


# ------------------------------------------------------------------------ zones, quotas


def resolve_zone(text: str) -> str:
    """The IANA name for ``text`` (an alias, or a name in any case); :class:`ArgumentError` else."""
    wanted = fold(text).replace(" ", "")
    if not wanted:
        raise ArgumentError("a time zone is needed")
    for alias, name in ZONE_ALIASES.items():
        if wanted == fold(alias):
            return name
    for name in available_timezones():
        if name.casefold() == wanted:
            return name
    raise ArgumentError(f"unknown time zone: {text}")


_QUOTA = re.compile(r"^(?P<low>\d+)[-~至到](?P<high>\d+)$")


def parse_quota(text: str, *, ceiling: int = 12) -> tuple[int, int]:
    """``2-5`` as (2, 5), checked: ``0 <= low <= high <= ceiling``."""
    spoken = fold(text).replace(" ", "")
    found = _QUOTA.fullmatch(spoken)
    if found is None:
        raise ArgumentError("unreadable")
    low, high = int(found["low"]), int(found["high"])
    if not 0 <= low <= high <= ceiling:
        raise ArgumentError("out_of_range")
    return low, high
