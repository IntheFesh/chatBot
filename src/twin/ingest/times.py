"""Time in an export: epoch seconds, local text and the zone the text was written in.

Messages carry ``createTime`` (epoch) and ``createTimeText`` (local wall-clock text).  The
epoch is what is stored (UTC, CLAUDE.md rule 5).  The text is interpreted in
``time.source_timezone`` (with ``time.source_timezone_ranges`` for periods in which the
exporting device was in another zone); it is used

* to recover a message whose epoch is missing, and
* as a cross-check: a text that disagrees with the epoch by more than an hour usually means
  the source time zone is configured wrongly, which the import report points out.

Report dates (R-IMP-010) are local dates in the same source zone.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from twin.config.settings import TimeConfig, TzRange

MILLISECOND_THRESHOLD = 100_000_000_000  # epoch values above this are milliseconds
MISMATCH_TOLERANCE_S = 3700.0  # one hour of daylight saving ambiguity plus slack


def epoch_to_utc(value: int | float | str | None) -> datetime | None:
    """An aware UTC datetime from epoch seconds or milliseconds; ``None`` if unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:  # NaN or not a real timestamp
        return None
    if number >= MILLISECOND_THRESHOLD:
        number /= 1000.0
    try:
        return datetime.fromtimestamp(number, UTC)
    except (OverflowError, OSError, ValueError):
        return None


class SourceTime:
    """Interprets the local texts of an export and dates messages in the source zone."""

    def __init__(self, default_zone: str, ranges: Sequence[TzRange] = ()) -> None:
        self._default = ZoneInfo(default_zone)
        self._ranges = [(item.from_, item.to, ZoneInfo(item.tz)) for item in ranges]

    @classmethod
    def from_config(cls, config: TimeConfig) -> SourceTime:
        return cls(config.source_timezone, config.source_timezone_ranges)

    def zone_on(self, day: date) -> ZoneInfo:
        for start, end, zone in self._ranges:
            if start <= day <= end:
                return zone
        return self._default

    def zone_for(self, moment: datetime) -> ZoneInfo:
        """The zone she was in at ``moment``: the first range containing its local date."""
        utc = moment.astimezone(UTC)
        for start, end, zone in self._ranges:
            if start <= utc.astimezone(zone).date() <= end:
                return zone
        return self._default

    def local_date(self, moment: datetime) -> date:
        """The local calendar date of ``moment`` in the source zone."""
        return moment.astimezone(self.zone_for(moment)).date()

    def parse_local_text(self, text: str | None) -> datetime | None:
        """``createTimeText`` as an aware UTC datetime; ``None`` if it cannot be read."""
        if not text:
            return None
        fields = _wall_clock_fields(text.strip())
        if fields is None:
            return None
        year, month, day, hour, minute, second = fields
        try:
            zone = self.zone_on(date(year, month, day))
            return datetime(year, month, day, hour, minute, second, tzinfo=zone).astimezone(UTC)
        except ValueError:
            return None

    def disagrees(self, moment: datetime, text: str | None) -> bool:
        """True if ``text`` is readable and differs from ``moment`` by more than an hour."""
        other = self.parse_local_text(text)
        if other is None:
            return False
        return abs((other - moment).total_seconds()) > MISMATCH_TOLERANCE_S


def _wall_clock_fields(text: str) -> tuple[int, int, int, int, int, int] | None:
    """``(Y, M, D, h, m, s)`` of ``YYYY-MM-DD HH:MM:SS`` (also ``/``, ``T``, no seconds)."""
    if len(text) < 16:
        return None
    sep = text[4]
    if sep not in "-/." or text[7] != sep or text[10] not in " T":
        return None
    try:
        second = int(text[17:19]) if len(text) >= 19 and text[16] == ":" else 0
        return (
            int(text[0:4]),
            int(text[5:7]),
            int(text[8:10]),
            int(text[11:13]),
            int(text[14:16]),
            second,
        )
    except ValueError:
        return None
