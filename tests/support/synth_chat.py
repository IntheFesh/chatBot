"""Synthetic conversations with *known* regularities, written straight into the database.

The profile and routine tests need data whose true answers are known: she is silent from 01:00
to 08:30, replies slowly on workdays between 13:00 and 17:00, opens exactly four conversations a
day, uses a comma in 3 % of her texts.  ``build_chat`` writes such a conversation as ``messages``
rows (no export files are needed), and ``build_hourly_chat`` writes messages whose hour-of-day
profile follows a given vector (the seven-day sample of SPEC section 0).

Nothing here is a real conversation: texts are random characters, drawn from a fixed pool.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from twin.schedule.daytype import DayTypeCalendar
from twin.services import Services
from twin.storage.chat_models import Conversation, Message, Sticker, StickerUse

POOL = (
    "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之"
    "过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如"
)
CODES = ("[拥抱]", "[亲亲]", "[捂脸]", "[抱抱]")
STICKER_COUNT = 6
EXPORT_ID = "synthetic-profile-data"


@dataclass(frozen=True)
class ChatSpec:
    """What the generated conversation looks like (all clock times are local hours)."""

    start: date = date(2026, 8, 3)  # a Monday
    days: int = 42
    zone: str = "America/Chicago"
    zone_by_day: dict[date, str] = field(default_factory=dict)  # overrides ``zone`` per day
    sleep: tuple[float, float] = (1.0, 8.5)
    weekend_sleep: tuple[float, float] | None = None
    busy: tuple[float, float] | None = (13.0, 17.0)  # slow replies on workdays
    busy_median_s: float = 1500.0
    night_message: bool = False  # the user writes at 03:10; she answers when she wakes
    her_comma_every: int = 33  # 1 in 33 of her texts has a comma (3.03 %)
    user_comma_every: int = 2  # 1 in 2 (50 %)
    seed: int = 11


@dataclass
class ChatTruth:
    """What a test may assert about the generated data."""

    messages: int = 0
    her_messages: int = 0
    her_texts: int = 0
    her_comma_texts: int = 0
    her_initiations: int = 0
    first_message_at: datetime | None = None
    last_message_at: datetime | None = None
    days: int = 0


def _local(day: date, hour: float, zone: ZoneInfo) -> datetime:
    """The instant at which the wall clock of ``zone`` shows ``hour`` (fractional) on ``day``."""
    wall = datetime.combine(day, time(0), tzinfo=zone) + timedelta(minutes=round(hour * 60))
    return wall.astimezone(UTC)


def _text(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(POOL) for _ in range(length))


class MessageWriter:
    """Collects messages, then stores them (as a new conversation or appended to the first)."""

    def __init__(self, services: Services) -> None:
        self.services = services
        self.rows: list[tuple[datetime, bool, str, str | None, str | None]] = []

    def add(
        self, at: datetime, her: bool, kind: str, text: str | None = None, md5: str | None = None
    ) -> None:
        self.rows.append((at, her, kind, text, md5))

    def store(self, *, append: bool = False) -> list[tuple[datetime, bool, str]]:
        self.rows.sort(key=lambda row: row[0])
        services = self.services
        now = services.clock.now_utc()
        with services.db.transaction(bump_state=False) as session:
            conversation = session.scalars(select(Conversation)).first() if append else None
            if conversation is None:
                conversation = Conversation(
                    username="synthetic-target",
                    display_name=None,
                    is_group=False,
                    message_count=0,
                    created_at=now,
                    updated_at=now,
                )
                session.add(conversation)
                session.flush()
            first_index = int(
                session.scalar(
                    select(func.count())
                    .select_from(Message)
                    .where(Message.conversation_id == conversation.id)
                )
                or 0
            )
            known = set(session.scalars(select(Sticker.md5)))
            for md5 in {row[4] for row in self.rows if row[4]} - known:
                session.add(
                    Sticker(
                        md5=md5,
                        status="available",
                        her_uses=0,
                        user_uses=0,
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.flush()
            for offset, (at, her, kind, text, md5) in enumerate(self.rows):
                index = first_index + offset
                message = Message(
                    id=f"synth-{index:07d}",
                    conversation_id=conversation.id,
                    create_time_utc=at,
                    sort_seq=index,
                    is_sent=not her,
                    kind=kind,
                    text=text,
                    raw={},
                    sticker_md5=md5,
                    source_export_id=EXPORT_ID,
                    created_at=now,
                    updated_at=now,
                )
                session.add(message)
                if kind == "sticker" and md5:
                    session.flush()
                    session.add(
                        StickerUse(
                            message_id=message.id,
                            sticker_md5=md5,
                            conversation_id=conversation.id,
                            by_her=her,
                            used_at=at,
                            created_at=now,
                            updated_at=now,
                        )
                    )
            conversation.message_count = first_index + len(self.rows)
        return [(r[0], r[1], r[2]) for r in self.rows]


def append_texts(
    services: Services, rows: Sequence[tuple[datetime, bool, str]]
) -> list[tuple[datetime, bool, str]]:
    """Add text messages ``(time, from her, text)`` to the conversation already stored."""
    writer = MessageWriter(services)
    for at, her, text in rows:
        writer.add(at, her, "text", text)
    return writer.store(append=True)


class _Chat:
    """The conversation simulator behind :func:`build_chat`."""

    def __init__(self, spec: ChatSpec, writer: MessageWriter) -> None:
        self.spec = spec
        self.writer = writer
        self.rng = random.Random(spec.seed)
        self.truth = ChatTruth()
        self._her_text_index = 0
        self._user_text_index = 0
        self.calendar = DayTypeCalendar()

    # ---------------------------------------------------------------- messages

    def _her_message(self, at: datetime) -> None:
        self.truth.her_messages += 1
        roll = self.rng.random()
        if roll < 0.105:  # stickers, a few favourites
            weights = [8, 5, 3, 1, 1, 1][:STICKER_COUNT]
            choice = self.rng.choices(range(STICKER_COUNT), weights=weights)[0]
            self.writer.add(at, True, "sticker", None, f"{choice:032x}")
            return
        index = self._her_text_index
        self._her_text_index += 1
        self.truth.her_texts += 1
        length = self.rng.choice((2, 3, 4, 4, 5, 5, 5, 6, 7, 8, 10, 12))
        comma = index % self.spec.her_comma_every == 0
        if comma:
            length = max(length, 6)
            self.truth.her_comma_texts += 1
        body = _text(self.rng, length)
        if comma:
            cut = self.rng.randint(2, length - 2)
            body = body[:cut] + "，" + body[cut:]
        if index % 20 == 7:
            body += "哈哈哈"
        if index % 25 == 3:
            body += self.rng.choice(CODES)
        kind = "quote" if index % 16 == 5 else "text"
        self.writer.add(at, True, kind, body)

    def _user_message(self, at: datetime) -> None:
        index = self._user_text_index
        self._user_text_index += 1
        length = self.rng.choice((4, 6, 8, 11, 11, 14, 20, 30, 45))
        body = _text(self.rng, length)
        if index % self.spec.user_comma_every == 0:
            cut = self.rng.randint(1, length - 1)
            body = body[:cut] + "，" + body[cut:]
        self.writer.add(at, False, "text", body)

    def _block(self, start: datetime, her: bool, busy: bool = False) -> datetime:
        """A burst starting at ``start``; returns the time of its last message."""
        size = 1 if busy else self.rng.choice((1, 1, 2, 2, 2, 3, 4, 6))
        moment = start
        for i in range(size):
            if i:
                moment += timedelta(
                    seconds=self.rng.randint(2, 8) if her else self.rng.randint(5, 20)
                )
            (self._her_message if her else self._user_message)(moment)
        return moment

    # ----------------------------------------------------------------- latency

    def _normal_latency(self) -> float:
        return min(600.0, max(2.0, self.rng.lognormvariate(math.log(18.0), 0.8)))

    def _busy_latency(self) -> float:
        value = self.rng.lognormvariate(math.log(self.spec.busy_median_s), 0.3)
        return min(2400.0, max(600.0, value))

    def _is_busy(self, local: datetime, day_is_workday: bool) -> bool:
        busy = self.spec.busy
        if busy is None or not day_is_workday:
            return False
        hour = local.hour + local.minute / 60.0
        return busy[0] <= hour < busy[1]

    # ------------------------------------------------------------------- days

    def _schedule(self, day: date) -> tuple[bool, tuple[float, float]]:
        zone_name = self.spec.zone_by_day.get(day, self.spec.zone)
        workday = self.calendar.day_type(day, zone_name) == "workday"
        if not workday and self.spec.weekend_sleep is not None:
            return workday, self.spec.weekend_sleep
        return workday, self.spec.sleep

    def _plan(self, wake: float) -> list[tuple[str, float]]:
        """The conversations of a day as ``(who starts, planned local hour)``."""
        jitter = self.rng.uniform
        plan: list[tuple[str, float]] = []
        if not self.spec.night_message:
            plan.append(("H", wake + 0.1 + jitter(0.0, 0.15)))
        plan += [
            ("U", 10.33 + jitter(-0.4, 0.4)),
            ("H", 11.92 + jitter(-0.25, 0.25)),
            ("U", 13.4 + jitter(-0.6, 0.6)),
            ("U", 15.0 + jitter(-0.6, 0.6)),
            ("U", 16.5 + jitter(-0.5, 0.5)),
            ("H", 18.5 + jitter(-0.25, 0.25)),
            ("U", 20.0 + jitter(-0.3, 0.3)),
            ("H", 21.5 + jitter(-0.25, 0.25)),
            ("U", 22.9 + jitter(-0.1, 0.1)),
        ]
        return [item for item in plan if item[1] >= wake + 0.1]

    def day(self, day: date) -> None:
        zone_name = self.spec.zone_by_day.get(day, self.spec.zone)
        zone = ZoneInfo(zone_name)
        workday, (onset_hour, wake_hour) = self._schedule(day)
        _, (previous_onset, _) = self._schedule(day - timedelta(days=1))
        evening_sleep = onset_hour >= 12  # falls asleep before midnight: the day ends with it
        if previous_onset < 12:
            # the end of last night's conversation, shortly before she falls asleep
            late = _local(day, previous_onset - 0.25 + self.rng.uniform(-0.1, 0.1), zone)
            self._episode("U", late, zone, workday)
        cursor = _local(day, wake_hour, zone) - timedelta(minutes=65)
        if self.spec.night_message:
            self._user_message(_local(day, 3.17, zone))
            reply = _local(day, wake_hour + 0.17, zone)
            cursor = self._block(reply, True) + timedelta(minutes=1)
            self.truth.her_initiations += 1
        limit = onset_hour - 1.3 if evening_sleep else 23.5
        for who, hour in self._plan(wake_hour):
            start = max(_local(day, hour, zone), cursor + timedelta(minutes=65))
            start += timedelta(seconds=self.rng.randint(0, 240))
            local_start = start.astimezone(zone)
            if local_start.date() != day or local_start.hour + local_start.minute / 60 > limit:
                continue
            cursor = self._episode(who, start, zone, workday)
            if who == "H":
                self.truth.her_initiations += 1
        if evening_sleep:
            last = _local(day, onset_hour - 0.25 + self.rng.uniform(-0.1, 0.1), zone)
            self._episode("U", last, zone, workday)

    def _episode(self, who: str, start: datetime, zone: ZoneInfo, workday: bool) -> datetime:
        her_first = who == "H"
        moment = start
        end = self._block(
            moment, her_first, busy=her_first and self._is_busy(moment.astimezone(zone), workday)
        )
        # the other side answers
        for turn in range(2):
            replier_is_her = her_first == (turn % 2 == 1)
            arrival_local = end.astimezone(zone)
            busy_reply = replier_is_her and self._is_busy(arrival_local, workday)
            latency = self._busy_latency() if busy_reply else self._normal_latency()
            moment = end + timedelta(seconds=latency)
            end = self._block(moment, replier_is_her, busy=busy_reply)
            if busy_reply:
                break
        return end


def build_chat(services: Services, spec: ChatSpec | None = None) -> ChatTruth:
    """Write the conversation of ``spec`` to the database and return what is true about it."""
    spec = spec or ChatSpec()
    writer = MessageWriter(services)
    chat = _Chat(spec, writer)
    for offset in range(spec.days):
        chat.day(spec.start + timedelta(days=offset))
    rows = writer.store()
    chat.truth.messages = len(rows)
    chat.truth.first_message_at = rows[0][0] if rows else None
    chat.truth.last_message_at = rows[-1][0] if rows else None
    chat.truth.days = spec.days
    # the very first message has no silence before it, so it does not count as an initiation
    if rows and rows[0][1] and chat.truth.her_initiations:
        chat.truth.her_initiations -= 1
    return chat.truth


def build_hourly_chat(
    services: Services,
    hourly_totals: Sequence[float],
    *,
    days: int = 28,
    totals_over_days: int = 7,
    zone: str = "America/Chicago",
    start: date = date(2026, 8, 3),
    seed: int = 5,
) -> ChatTruth:
    """Her messages with an hour-of-day profile proportional to ``hourly_totals``.

    ``hourly_totals`` are counts per local hour accumulated over ``totals_over_days`` days (the
    sample of SPEC section 0); each generated day gets ``total / totals_over_days`` messages in
    the hour, at random minutes.  Only her messages are written: the sleep search needs nothing
    else.
    """
    rng = random.Random(seed)
    writer = MessageWriter(services)
    truth = ChatTruth()
    tz = ZoneInfo(zone)
    for offset in range(days):
        day = start + timedelta(days=offset)
        for hour, total in enumerate(hourly_totals):
            expected = total / totals_over_days
            count = int(expected) + (1 if rng.random() < expected - int(expected) else 0)
            moments = sorted(rng.uniform(0.0, 3599.0) for _ in range(count))
            base = datetime.combine(day, time(hour), tzinfo=tz).astimezone(UTC)
            for seconds in moments:
                at = base + timedelta(seconds=seconds)
                writer.add(at, True, "text", _text(rng, rng.choice((3, 4, 5, 6, 8))))
                truth.her_messages += 1
    rows = writer.store()
    truth.messages = len(rows)
    truth.days = days
    truth.first_message_at = rows[0][0] if rows else None
    truth.last_message_at = rows[-1][0] if rows else None
    return truth
