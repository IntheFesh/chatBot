"""The zone of the export's ``createTimeText``, inferred from the export (round 18 item C7).

User decision (export text zone): the difference of ``createTimeText`` and ``createTime`` tells the
zone the exporting device wrote the text in; it is used for the consistency check and the import
report only, and ``time.source_timezone(_ranges)`` means nothing but "where she was" (D-900 to
D-903).
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.synth_export import SynthExport, SynthOptions, build_export
from tests.support.ingest import make_export, run_import, start_import
from twin.config.settings import TzRange
from twin.ingest.importer import ImportRunner
from twin.ingest.inspect import inspect_export, render_inspect
from twin.ingest.normalize import NormalizeContext, normalize_message
from twin.ingest.schema import ExportMessage
from twin.ingest.stats import RunStats
from twin.ingest.textzone import (
    TextZoneReader,
    TextZoneTally,
    candidate_zones,
    format_offset,
    infer_text_zone,
    observed_offset_minutes,
)
from twin.ingest.times import SourceTime
from twin.services import Services
from twin.storage.chat_models import Message

CHICAGO = ZoneInfo("America/Chicago")
SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
TEXT_FORMAT = "%Y-%m-%d %H:%M:%S"


def text_of(moment: datetime, zone: ZoneInfo, fmt: str = TEXT_FORMAT) -> str:
    return moment.astimezone(zone).strftime(fmt)


def tally_of(zone: ZoneInfo, moments: list[datetime], fmt: str = TEXT_FORMAT) -> TextZoneTally:
    tally = TextZoneTally()
    for moment in moments:
        tally.observe(moment, text_of(moment, zone, fmt))
    return tally


def every(hours: float, start: datetime, end: datetime) -> list[datetime]:
    found = []
    moment = start
    while moment < end:
        found.append(moment)
        moment += timedelta(hours=hours)
    return found


def her_range(first: date, last: date, zone: str = "Asia/Shanghai") -> TzRange:
    return TzRange.model_validate({"from": first, "to": last, "tz": zone})


def rewrite_messages(export: SynthExport, change: Any) -> None:
    """Apply ``change(position, message)`` to every message of the target's messages.json."""
    document = json.loads(export.messages_path.read_text(encoding="utf-8"))
    for position, item in enumerate(document["messages"]):
        change(position, item)
    export.messages_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def report_text(outcome: Any) -> str:
    return Path(outcome.report_path or "").read_text(encoding="utf-8")


def section(report: str, heading: str) -> str:
    return report.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]


# ---------------------------------------------------------------------- inference


def test_the_offset_of_a_text_is_its_wall_clock_minus_the_moment() -> None:
    """R18-C7: a text in Chicago in January is six hours behind UTC; seconds may be missing."""
    moment = datetime(2026, 1, 5, 12, 0, 30, tzinfo=UTC)
    assert observed_offset_minutes(moment, "2026-01-05 06:00:30") == -360
    assert observed_offset_minutes(moment, "2026-01-05 06:00") == -360  # no seconds
    assert observed_offset_minutes(moment, "2026/01/05 20:00:30") == 480
    assert observed_offset_minutes(moment, "2026-01-05 06:03:30") == -357  # not a zone's offset
    assert observed_offset_minutes(moment, "not a time") is None
    assert observed_offset_minutes(moment, None) is None
    assert [format_offset(m) for m in (-360, 480, 345, 0, -357)] == [
        "-06:00",
        "+08:00",
        "+05:45",
        "+00:00",
        "-05:57",
    ]


def test_chicago_text_across_both_clock_changes_is_chicago_and_not_a_zone_without_them() -> None:
    """R18-C7: data across 2026-03-08 and 2026-11-01 still has one answer (DST by zoneinfo)."""
    moments = every(5, datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 12, 20, tzinfo=UTC))
    inference = infer_text_zone(tally_of(CHICAGO, moments))
    assert inference.zone == "America/Chicago"
    assert inference.pairs == len(moments)
    fitting = set(inference.fitting)
    for other in ("America/Regina", "America/Mexico_City", "America/New_York", "Asia/Shanghai"):
        assert other not in fitting, other
    # the zones that fit as well follow the very same rules over the whole year
    for name in fitting:
        zone = ZoneInfo(name)
        assert all(
            m.astimezone(zone).utcoffset() == m.astimezone(CHICAGO).utcoffset() for m in moments
        )
    assert inference.equivalents == len(fitting) - 1
    assert {offset for offset, _ in inference.offsets} == {-360, -300}
    # a winter alone does not tell Chicago from Regina: the clock changes single it out
    winter = infer_text_zone(tally_of(CHICAGO, [m for m in moments if m.month in (1, 2)]))
    assert winter.zone == "America/Chicago" and "America/Regina" in winter.fitting


def test_the_clock_change_days_themselves_fit() -> None:
    """R18-C7: messages every 10 minutes through both changes of 2026 (the repeated hour too)."""
    spring = every(1 / 6, datetime(2026, 3, 8, 4, tzinfo=UTC), datetime(2026, 3, 8, 12, tzinfo=UTC))
    autumn = every(
        1 / 6, datetime(2026, 11, 1, 3, tzinfo=UTC), datetime(2026, 11, 1, 12, tzinfo=UTC)
    )
    inference = infer_text_zone(tally_of(CHICAGO, spring + autumn))
    assert inference.zone == "America/Chicago"
    assert "America/Regina" not in inference.fitting


def test_a_text_written_in_shanghai_is_shanghai() -> None:
    moments = every(7, datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 9, 1, tzinfo=UTC))
    inference = infer_text_zone(tally_of(SHANGHAI, moments))
    assert inference.zone == "Asia/Shanghai"
    assert inference.offsets == ((480, len(moments)),)


def test_the_configured_zone_names_the_answer_when_it_fits() -> None:
    """R18-C7: several zones fit the same data; a fitting configured zone is the one named."""
    moments = every(7, datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 9, 1, tzinfo=UTC))
    tally = tally_of(SHANGHAI, moments)
    assert infer_text_zone(tally, preferred=("Asia/Taipei",)).zone == "Asia/Taipei"
    # a preferred zone that does not fit is passed over
    assert infer_text_zone(tally, preferred=("America/Chicago",)).zone == "Asia/Shanghai"


def test_texts_from_two_zones_fit_no_zone_and_give_the_offsets() -> None:
    """R18-C7: a mixed export cannot be inferred; the offsets are listed for the user."""
    moments = every(3, datetime(2025, 1, 6, tzinfo=UTC), datetime(2025, 1, 20, tzinfo=UTC))
    tally = TextZoneTally()
    for position, moment in enumerate(moments):
        tally.observe(moment, text_of(moment, SHANGHAI if position % 3 == 0 else CHICAGO))
    inference = infer_text_zone(tally)
    assert inference.zone is None and inference.fitting == ()
    shanghai = sum(1 for position in range(len(moments)) if position % 3 == 0)
    assert dict(inference.offsets) == {-360: len(moments) - shanghai, 480: shanghai}
    assert inference.offsets[0][0] == -360  # the most frequent first
    lines = "\n".join(inference.report_lines())
    assert "-06:00" in lines and "+08:00" in lines and "推断不出" in lines


def test_one_odd_text_is_enough_to_refuse_a_zone() -> None:
    """R18-C7: every message has to fit ("全部吻合"); a single text an hour off breaks it."""
    moments = every(5, datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 5, tzinfo=UTC))
    tally = tally_of(CHICAGO, moments)
    odd = moments[10]
    tally.observe(odd, (odd.astimezone(CHICAGO) + timedelta(hours=1)).strftime(TEXT_FORMAT))
    inference = infer_text_zone(tally)
    assert inference.zone is None
    assert dict(inference.offsets)[-300] == 1


def test_nothing_to_compare_infers_nothing() -> None:
    tally = TextZoneTally()
    tally.observe(datetime(2026, 1, 5, tzinfo=UTC), "garbage")
    tally.observe(None, "2026-01-05 06:00:00")
    inference = infer_text_zone(tally)
    assert inference.zone is None and inference.pairs == 0 and inference.offsets == ()
    assert tally.unreadable == 1
    assert "没有" in "\n".join(inference.report_lines())


def test_the_tally_is_small_and_round_trips_through_json() -> None:
    """R18-C7: a year of messages every hour is one entry per day and offset, not per message."""
    moments = every(1, datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC))
    tally = tally_of(CHICAGO, moments, "%Y-%m-%d %H:%M")
    assert tally.pairs == len(moments) and len(tally.spans) <= 365 + 2
    again = TextZoneTally.from_json(json.loads(json.dumps(tally.to_json())))
    assert again.spans == tally.spans and again.offsets == tally.offsets
    assert again.pairs == tally.pairs and again.unreadable == tally.unreadable
    assert infer_text_zone(again).zone == "America/Chicago"
    assert TextZoneTally.from_json(None).pairs == 0
    stats = RunStats.from_json(RunStats(text_zone=tally).to_json())
    assert stats.text_zone.spans == tally.spans


def test_a_tally_with_too_many_spans_stops_growing_and_infers_nothing() -> None:
    tally = TextZoneTally()
    start = datetime(1990, 1, 1, tzinfo=UTC)
    for day in range(25_000):
        moment = start + timedelta(days=day)
        tally.observe(moment, text_of(moment, CHICAGO))
    assert len(tally.spans) <= 20_000 and tally.overflow > 0
    assert infer_text_zone(tally).zone is None


def test_the_candidates_are_the_geographic_zones() -> None:
    names = candidate_zones()
    assert "America/Chicago" in names and "Asia/Shanghai" in names and "UTC" in names
    assert not [n for n in names if n.startswith(("Etc/", "posix/", "right/", "US/"))]
    assert 300 < len(names) < 700


def test_the_reader_follows_the_tally() -> None:
    tally = TextZoneTally()
    reader = TextZoneReader(tally)
    assert reader.zone() is None
    moments = every(5, datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 7, 5, tzinfo=UTC))
    for moment in moments:
        tally.observe(moment, text_of(moment, CHICAGO))
    found = reader.zone()
    assert found is not None and found.key == "America/Chicago"
    assert reader.zone() is found  # cached until the tally changes
    tally.observe(moments[0], text_of(moments[0], SHANGHAI))
    assert reader.zone() is None


ZONES = (
    "America/Chicago",
    "America/New_York",
    "America/Phoenix",
    "Asia/Shanghai",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Europe/London",
    "Australia/Sydney",
    "Australia/Lord_Howe",
    "Pacific/Chatham",
    "UTC",
)


@settings(max_examples=40, deadline=None)
@given(
    zone_name=st.sampled_from(ZONES),
    seconds=st.lists(
        st.integers(min_value=1_577_836_800, max_value=1_893_456_000), min_size=1, max_size=40
    ),
    milliseconds=st.integers(min_value=0, max_value=999),
    with_seconds=st.booleans(),
)
def test_hypothesis_the_writing_zone_always_fits_and_every_fit_agrees(
    zone_name: str, seconds: list[int], milliseconds: int, with_seconds: bool
) -> None:
    """R18-C7: whatever the zone and the moments (DST, half-hour zones, Lord Howe's half hour),
    the zone that wrote the texts fits, and every zone named as fitting shows the same clock."""
    zone = ZoneInfo(zone_name)
    moments = [datetime.fromtimestamp(s + milliseconds / 1000, UTC) for s in seconds]
    fmt = TEXT_FORMAT if with_seconds else "%Y-%m-%d %H:%M"
    inference = infer_text_zone(tally_of(zone, moments, fmt))
    assert zone_name in inference.fitting
    assert inference.zone in inference.fitting
    for name in inference.fitting:
        other = ZoneInfo(name)
        for moment in moments:
            assert moment.astimezone(other).utcoffset() == moment.astimezone(zone).utcoffset()


# ---------------------------------------------------------------------- the import


def test_text_in_chicago_while_she_was_in_shanghai_is_no_mismatch(
    services: Services, tmp_path: Path
) -> None:
    """R18-C7: the export's text is Chicago, her range is Shanghai; the report infers Chicago and
    does not ask to change where she was (before: every message was a "mismatch")."""
    services.settings.time.source_timezone_ranges = [her_range(date(2024, 12, 1), date(2025, 3, 1))]
    export = make_export(tmp_path, target_messages=200)
    outcome = run_import(services, export)
    inference = infer_text_zone(outcome.run.stats.text_zone)
    assert inference.zone == "America/Chicago" and inference.pairs == 200
    report = report_text(outcome)
    quality = section(report, "数据质量提示")
    assert "source_timezone" not in quality and "createTimeText" not in quality
    zone_part = section(report, "导出文字时区")
    assert "America/Chicago" in zone_part and "200" in zone_part
    assert "请核对" not in zone_part and "删除" not in zone_part  # no advice about her zone


def test_a_mixed_export_lists_the_offsets_in_the_report(services: Services, tmp_path: Path) -> None:
    """R18-C7: no zone fits every message: the report lists the offsets for the user to confirm."""
    export = make_export(tmp_path, target_messages=90)

    def to_shanghai(position: int, item: dict[str, Any]) -> None:
        if position % 3 == 0:
            moment = datetime.fromtimestamp(item["createTime"], UTC)
            item["createTimeText"] = text_of(moment, SHANGHAI)

    rewrite_messages(export, to_shanghai)
    outcome = run_import(services, export, batch_size=1000)
    zone_part = section(report_text(outcome), "导出文字时区")
    assert "推断不出" in zone_part
    assert "-06:00 × 60" in zone_part and "+08:00 × 30" in zone_part
    assert "请核对 time.source_timezone" not in report_text(outcome)


def test_a_message_without_create_time_is_read_in_the_inferred_text_zone(
    services: Services, tmp_path: Path
) -> None:
    """R18-C7: her range says Shanghai, the device wrote Chicago: a lost createTime is recovered
    on the device's clock (before: on hers, fourteen hours off)."""
    services.settings.time.source_timezone_ranges = [her_range(date(2024, 12, 1), date(2025, 3, 1))]
    export = make_export(tmp_path, target_messages=120)
    lost: dict[str, int] = {}

    def drop(position: int, item: dict[str, Any]) -> None:
        if position in (40, 80):
            lost[str(item["id"])] = int(item.pop("createTime"))

    rewrite_messages(export, drop)
    outcome = run_import(services, export)
    with services.db.session() as session:
        for message_id, epoch in lost.items():
            row = session.get(Message, message_id)
            assert row is not None
            assert row.create_time_utc == datetime.fromtimestamp(epoch, UTC), message_id
    assert outcome.run.stats.time_from_text == {"America/Chicago": 2}
    zone_part = section(report_text(outcome), "导出文字时区")
    assert "2 条消息没有 createTime" in zone_part and "America/Chicago" in zone_part


def test_without_any_epoch_the_text_is_read_on_her_clock_and_the_report_says_so(
    services: Services, tmp_path: Path
) -> None:
    """R18-C7: nothing to infer from: the old reading (her zone) stays, and the report says so."""
    export = make_export(tmp_path, target_messages=30)
    rewrite_messages(export, lambda _position, item: item.pop("createTime"))
    outcome = run_import(services, export)
    assert outcome.run.invalid == 0
    assert outcome.run.stats.time_from_text == {"": 30}
    zone_part = section(report_text(outcome), "导出文字时区")
    assert "没有同时带" in zone_part
    assert "30 条消息没有 createTime" in zone_part and "time.source_timezone" in zone_part


def test_the_tally_goes_on_after_an_interrupted_run(services: Services, tmp_path: Path) -> None:
    """R18-C7: the counts are saved with each batch, so a resumed run infers from all messages."""
    export = make_export(tmp_path, target_messages=150)
    run_id = start_import(services, export)
    stop = threading.Event()
    stop_after_first = ImportRunner(services, batch_size=50, on_batch=lambda _e: stop.set())
    first = stop_after_first.run(run_id, stop)
    assert first.status == "interrupted"
    assert first.run.stats.text_zone.pairs == 50
    outcome = ImportRunner(services, batch_size=50).run(run_id)
    assert outcome.status == "done"
    assert outcome.run.stats.text_zone.pairs == 150
    assert infer_text_zone(outcome.run.stats.text_zone).zone == "America/Chicago"


# --------------------------------------------------------------------- normalising


def test_normalising_uses_the_reader_and_says_where_the_time_came_from() -> None:
    """R18-C7: with an inferred zone the text is read on it; without one, on her clock."""
    raw = {"id": "a", "isSent": False, "renderType": "text", "content": "x"}
    raw["createTimeText"] = "2025-01-01 06:00:00"
    source = SourceTime("Asia/Shanghai")
    known = NormalizeContext("u", source, text_zone=lambda: CHICAGO)
    item = normalize_message(raw, ExportMessage.model_validate(raw), known)
    assert item.create_time == datetime(2025, 1, 1, 12, tzinfo=UTC)
    assert item.time_from_text == "America/Chicago"
    unknown = NormalizeContext("u", source)
    item = normalize_message(raw, ExportMessage.model_validate(raw), unknown)
    assert item.create_time == datetime(2024, 12, 31, 22, tzinfo=UTC)  # Shanghai is UTC+8
    assert item.time_from_text == ""
    with_epoch = {**raw, "createTime": 1735732800}
    item = normalize_message(with_epoch, ExportMessage.model_validate(with_epoch), known)
    assert item.time_from_text is None


# ------------------------------------------------------------------------- inspect


def test_inspect_names_the_zone_of_the_texts(tmp_path: Path) -> None:
    """R18-C7: ``twin import inspect`` shows the inferred export text zone."""
    export = build_export(tmp_path / "e", SynthOptions(target_messages=200))
    result = inspect_export(export.root)
    text = render_inspect(result, NOW)
    part = section(text, "导出文字时区")
    assert "America/Chicago" in part
    assert infer_text_zone(result.text_zone).pairs == result.sampled


def test_inspect_lists_the_offsets_when_no_zone_fits(tmp_path: Path) -> None:
    """R18-C7: no zone fits: inspect lists the offset distribution for the user to confirm."""
    export = build_export(
        tmp_path / "e", SynthOptions(target_messages=90, other_conversations=0, include_group=False)
    )

    def to_shanghai(position: int, item: dict[str, Any]) -> None:
        if position % 3 == 0:
            moment = datetime.fromtimestamp(item["createTime"], UTC)
            item["createTimeText"] = text_of(moment, SHANGHAI)

    rewrite_messages(export, to_shanghai)
    part = section(render_inspect(inspect_export(export.root), NOW), "导出文字时区")
    assert "推断不出" in part
    assert "| -06:00 | 60 |" in part and "| +08:00 | 30 |" in part


def test_an_inferred_zone_is_never_a_message_value(tmp_path: Path) -> None:
    """R18-C7: the zone part holds zone names, offsets and counts only."""
    export = build_export(tmp_path / "e", SynthOptions(target_messages=50))
    part = section(render_inspect(inspect_export(export.root), NOW), "导出文字时区")
    assert not [t for t in export.texts if len(t) >= 4 and t in part]
    assert "2025-01" not in part  # no dates, no texts


@pytest.mark.parametrize("zone", ["Asia/Shanghai", "Europe/London"])
def test_the_synthetic_export_can_write_in_any_zone(tmp_path: Path, zone: str) -> None:
    export = build_export(
        tmp_path / zone.replace("/", "_"),
        SynthOptions(target_messages=40, text_zone=zone, other_conversations=0),
    )
    rows = json.loads(export.messages_path.read_text(encoding="utf-8"))["messages"]
    tally = TextZoneTally()
    for row in rows:
        tally.observe(datetime.fromtimestamp(row["createTime"], UTC), row["createTimeText"])
    assert zone in infer_text_zone(tally).fitting
