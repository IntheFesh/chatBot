"""Time in an export: epoch seconds, local text, and the days of the place she was in.

Messages carry ``createTime`` (epoch) and ``createTimeText`` (local wall-clock text).  The epoch is
what is stored (UTC, CLAUDE.md rule 5).  The text is the same moment on the clock of the
*exporting device*, in the device's zone - which has nothing to do with where she was.  That zone
is inferred from the export itself (:mod:`twin.ingest.textzone`, round 18 item C7, D-900); the text
is read in it only to recover a message whose epoch is missing, and the difference of the two is
the consistency check of the import report and of ``twin import inspect``.

``time.source_timezone`` (with ``time.source_timezone_ranges`` for the periods she was somewhere
else, R-ACT-001) means *where she was* and nothing else.  :class:`SourceTime` holds it.  Its
:meth:`SourceTime.parse_local_text` reads a text on her clock; that is only the last resort for a
text whose device zone cannot be inferred (the reading of the earlier rounds, D-137).

Report dates (R-IMP-010) are local dates in the zone she was in.

**Her local days** (D-782).  A day runs from its own midnight, in the zone of its date
(:meth:`SourceTime.zone_on`), to the midnight of the next date in the zone of *that* date, so the
days fit together without gap or overlap even where a range begins or ends: leaving Chicago for
Shanghai makes the last day at home 11 hours long, coming back makes the last day away 37 hours
long.  :meth:`SourceTime.day_bounds` gives those bounds and :meth:`SourceTime.local_date` the day
whose bounds hold a moment - the one definition the import report and the memory's replay, day
loader and as-of view share.  The *clock time* of a moment is read in the zone she was in at that
moment (:meth:`SourceTime.zone_for`, what the activity statistics use).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from twin.config.settings import TimeConfig, TzRange
from twin.schedule.wallclock import day_start_utc

ONE_DAY = timedelta(days=1)

MILLISECOND_THRESHOLD = 100_000_000_000  # epoch values above this are milliseconds


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
        self._starts: dict[date, datetime] = {}  # the replay dates every message: keep them

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

    def day_start(self, day: date) -> datetime:
        """The first instant of her local ``day`` (UTC; see the module description).

        It is the midnight of ``day`` in the zone of that date - unless the midnight of the date
        before, in *its* zone, comes later, which only a jump of the clock by more than a day can
        cause (UTC-11 to UTC+14): the day is then empty instead of overlapping the one before.
        """
        found = self._starts.get(day)
        if found is None:
            before = day - ONE_DAY
            found = max(
                day_start_utc(before, self.zone_on(before)), day_start_utc(day, self.zone_on(day))
            )
            self._starts[day] = found
        return found

    def day_bounds(self, day: date) -> tuple[datetime, datetime]:
        """``[start, end)`` of her local ``day``: from its first instant to the next day's."""
        return self.day_start(day), self.day_start(day + ONE_DAY)

    def local_date(self, moment: datetime) -> date:
        """Her local day that holds ``moment`` (the one whose :meth:`day_bounds` contain it)."""
        utc = moment.astimezone(UTC)
        today = utc.date()
        # every zone's midnight of a date lies between 14 hours before and 12 hours after the UTC
        # midnight of it, so the day of a moment is the UTC date or one of its neighbours
        for day in (today + ONE_DAY, today):
            if self.day_start(day) <= utc:
                return day
        return today - ONE_DAY

    def parse_local_text(self, text: str | None) -> datetime | None:
        """``createTimeText`` read on *her* clock (the zone of its date), as aware UTC.

        Only the last resort: the text is written in the exporting device's zone, which is read
        with :func:`read_wall_clock` once :mod:`twin.ingest.textzone` has inferred it.
        """
        fields = _fields_of(text)
        if fields is None:
            return None
        year, month, day, hour, minute, second = fields
        try:
            zone = self.zone_on(date(year, month, day))
            return datetime(year, month, day, hour, minute, second, tzinfo=zone).astimezone(UTC)
        except ValueError:
            return None


def read_wall_clock(text: str | None, zone: ZoneInfo) -> datetime | None:
    """A wall-clock text read in ``zone``, as aware UTC; ``None`` if it cannot be read.

    The earlier of two readings of a repeated autumn hour is taken (``fold=0``).
    """
    fields = _fields_of(text)
    if fields is None:
        return None
    try:
        return datetime(*fields, tzinfo=zone).astimezone(UTC)
    except ValueError:
        return None


def wall_clock_as_utc(text: str | None) -> datetime | None:
    """The clock reading of a text labelled UTC: minus a moment, it is the writer's UTC offset."""
    fields = _fields_of(text)
    if fields is None:
        return None
    try:
        return datetime(*fields, tzinfo=UTC)
    except ValueError:
        return None


def _fields_of(text: object) -> tuple[int, int, int, int, int, int] | None:
    if not isinstance(text, str) or not text:
        return None
    return _wall_clock_fields(text.strip())


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
