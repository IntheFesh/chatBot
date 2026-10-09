"""The chat-record import end to end (R-IMP-003 ... R-IMP-010, R-STO-006, R-SCOPE-002)."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import threading
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from tests.fixtures.synth_export import SynthExport, message, write_manifest
from tests.support.clock import ManualClock
from tests.support.ingest import (
    make_export,
    message_snapshot,
    run_import,
    second_services,
    start_import,
)
from twin.config.secrets import SecretStore
from twin.ingest.importer import (
    BatchEvent,
    ImportFailure,
    ImportRunner,
    IntegrityFailure,
    TargetNotFoundError,
    prepare_import,
    resumable_run,
)
from twin.ingest.normalize import RENDER_TYPE_KINDS
from twin.ingest.runs import get_run, latest_run
from twin.services import Services
from twin.stickers.library import sticker_file_allowed
from twin.storage.chat_models import (
    Conversation,
    MediaAsset,
    Message,
    Sticker,
    StickerUse,
)


class PowerCut(BaseException):
    """Stands for a process that dies between two batches (nothing catches it)."""


IMAGE_HEAVY_MIX = {"image": 40.0, "text": 50.0, "video": 10.0}


def picture_files(export: SynthExport) -> list[Path]:
    """The picture files of the target's messages, in a stable order."""
    return sorted(export.root / relative for relative in export.image_paths)


def kind_of_render_type(render_type: str) -> str:
    kind = RENDER_TYPE_KINDS.get(render_type.casefold())
    return kind.value if kind else "unknown"


def stored_counts(services: Services) -> Counter[tuple[str, bool]]:
    with services.db.session() as session:
        rows = session.execute(
            select(Message.kind, Message.is_sent, func.count()).group_by(
                Message.kind, Message.is_sent
            )
        ).all()
    return Counter({(kind, sent): n for kind, sent, n in rows})


def expected_counts(export: SynthExport) -> Counter[tuple[str, bool]]:
    expected: Counter[tuple[str, bool]] = Counter()
    for (render_type, sent), n in export.counts.items():
        expected[(kind_of_render_type(render_type), sent)] += n
    return expected


def row_count(services: Services, model: Any) -> int:
    with services.db.session() as session:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)


def rewrite_export(
    export: SynthExport,
    mutate: Callable[[list[dict[str, Any]]], None] | None = None,
    *,
    exported_at: datetime | None = None,
    export_id: str | None = None,
) -> None:
    """Edit the target's ``messages.json`` (and ``report.json``) of a generated export."""
    document = json.loads(export.messages_path.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(document["messages"])
    if exported_at is not None:
        document["exportedAt"] = int(exported_at.timestamp())
    export.messages_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    if export_id is not None:
        report_path = export.root / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["exportId"] = export_id
        report_path.write_text(json.dumps(report), encoding="utf-8")


# ---------------------------------------------------------------- the basics


def test_a_full_import_stores_every_message_and_who_spoke(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=400)
    outcome = run_import(services, export)
    assert outcome.status == "done"
    run = outcome.run
    assert (run.processed, run.inserted, run.duplicates, run.invalid) == (400, 400, 0, 0)
    assert run.total == 400 and run.phase == "done" and run.finished_at is not None
    assert stored_counts(services) == expected_counts(export)
    # isSent=false is her, isSent=true is the user (R-SCOPE-002)
    her = sum(n for (_, sent), n in stored_counts(services).items() if not sent)
    assert her == sum(n for (_, sent), n in export.counts.items() if not sent) > 0
    with services.db.session() as session:
        conversation = session.scalars(select(Conversation)).one()
        assert conversation.username == export.target_username
        assert conversation.display_name == export.target_display_name
        assert conversation.message_count == 400 and not conversation.is_group
        assert (
            conversation.first_message_at is not None and conversation.last_message_at is not None
        )
        assert conversation.last_export_id == "synthetic-export-0001"


def test_message_rows_hold_the_normalised_and_the_original_data(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=300)
    run_import(services, export)
    original = {
        item["id"]: item
        for item in json.loads(export.messages_path.read_text(encoding="utf-8"))["messages"]
    }
    seen_kinds: set[str] = set()
    with services.db.session() as session:
        for row in session.scalars(select(Message)):
            raw = original[row.id]
            assert row.raw == raw  # the original object, nothing lost
            assert row.is_sent is bool(raw["isSent"])
            assert row.render_type == raw["renderType"]
            assert row.sort_seq == raw["sortSeq"]
            assert row.local_id == str(raw["localId"]) and row.server_id == raw["serverId"]
            assert row.create_time_utc == datetime.fromtimestamp(raw["createTime"], UTC)
            assert row.source_export_id == "synthetic-export-0001"
            assert row.source_exported_at == datetime.fromtimestamp(
                json.loads(export.messages_path.read_text(encoding="utf-8"))["exportedAt"], UTC
            )
            seen_kinds.add(row.kind)
            if row.kind == "text":
                assert row.text == raw["content"] and row.media_sha256 is None
            if row.kind == "sticker":
                assert row.sticker_md5 == raw["emojiMd5"] and row.text is None
            if row.kind == "quote":
                assert row.quote is not None and row.quote["quoteContent"] == raw["quoteContent"]
                assert row.text == raw["content"]
            if row.kind == "voice":
                assert row.voice_seconds and row.has_transcript == ("voiceTranscript" in raw)
            if row.kind == "call":
                assert row.call_status in {
                    "connected",
                    "cancelled",
                    "rejected",
                    "missed",
                    "other_device",
                }
                assert (row.call_duration_s is not None) == (row.call_status == "connected")
    assert {"text", "sticker", "quote", "image", "voice", "call", "system", "unknown"} <= seen_kinds


def test_message_text_is_not_readable_in_the_database_file(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=200)
    run_import(services, export)
    services.db.engine.dispose()
    blob = b"".join(path.read_bytes() for path in tmp_path.glob("data/twin.db*") if path.is_file())
    assert len(export.texts) > 20
    for sentence in list(export.texts)[:50]:
        assert sentence.encode("utf-8") not in blob
    assert export.target_username.encode() not in blob
    connection = sqlite3.connect(tmp_path / "data" / "twin.db")
    try:
        text, raw = connection.execute(
            "SELECT text, raw FROM messages WHERE text IS NOT NULL LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    assert isinstance(text, bytes) and isinstance(raw, bytes) and text[0] == 1 and raw[0] == 1


def test_only_the_target_conversations_messages_json_is_opened(
    services: Services, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = make_export(tmp_path, target_messages=40, other_messages=10)
    opened: list[Path] = []
    real_open = Path.open
    real_read = Path.read_bytes

    def spy_open(self: Path, *args: object, **kwargs: object) -> io.IOBase:
        opened.append(self)
        return real_open(self, *args, **kwargs)  # type: ignore[no-any-return,call-overload]

    def spy_read(self: Path) -> bytes:
        opened.append(self)
        return real_read(self)

    monkeypatch.setattr(Path, "open", spy_open)
    monkeypatch.setattr(Path, "read_bytes", spy_read)
    run_import(services, export)
    messages_files = {p for p in opened if p.name == "messages.json"}
    assert messages_files, "the target's messages.json must have been read"
    assert {p.parent.name for p in messages_files} == {export.target_dir.name}


def test_group_chats_and_other_conversations_are_ignored(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=50, other_conversations=3, other_messages=9)
    run_import(services, export)
    assert row_count(services, Message) == 50
    assert row_count(services, Conversation) == 1
    assert export.group_username is not None
    with pytest.raises(ImportFailure, match="group"):
        prepare_import(services, export.root, target_username=export.group_username)
    with pytest.raises(TargetNotFoundError):
        prepare_import(services, export.root, target_username="wxid_" + "nobody0000")
    other = export.other_usernames[0]
    other_run = run_import(services, export, target=other)
    assert other_run.run.processed == 9  # a one-to-one chat can be chosen as the target
    assert row_count(services, Conversation) == 2


def test_the_import_run_records_progress_and_speed(
    services: Services, tmp_path: Path, clock: ManualClock
) -> None:
    export = make_export(tmp_path, target_messages=100)
    events: list[BatchEvent] = []

    def tick(event: BatchEvent) -> None:
        events.append(event)
        clock.tick(2.0)

    run_import(services, export, batch_size=25, on_batch=tick)
    assert [e.processed for e in events] == [25, 50, 75, 100]
    assert events[0].speed_per_s is None  # no time had passed yet
    assert events[1].speed_per_s == pytest.approx(50 / 2.0)
    assert events[-1].total == 100 and events[-1].batch_number == 4
    run = latest_run(services.db)
    assert run is not None and run.speed_per_s == pytest.approx(100 / 6.0)
    assert run.last_message_id == export.message_ids[-1] and run.last_sort_seq == 100_000
    assert run.stats.sessions == 1 and run.export_id == "synthetic-export-0001"


# ------------------------------------------------------------------ idempotence


def test_importing_the_same_export_twice_changes_nothing(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=150)
    first = run_import(services, export)
    before = message_snapshot(services)
    stickers_before = row_count(services, Sticker)
    uses_before = row_count(services, StickerUse)
    assets_before = row_count(services, MediaAsset)
    second = run_import(services, export)
    assert second.run.id != first.run.id
    assert (second.run.inserted, second.run.duplicates) == (0, 150)
    assert second.run.conflict_updated == second.run.conflict_kept == 0
    assert message_snapshot(services) == before
    assert row_count(services, Sticker) == stickers_before
    assert row_count(services, StickerUse) == uses_before
    assert row_count(services, MediaAsset) == assets_before
    assert row_count(services, Message) == 150


def test_a_newer_export_updates_changed_messages_and_adds_new_ones(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=60)
    run_import(services, export)
    texts = list(json.loads(export.messages_path.read_text("utf-8"))["messages"])
    target = next(m for m in texts if m["renderType"] == "text")

    def edit(items: list[dict[str, Any]]) -> None:
        for item in items:
            if item["id"] == target["id"]:
                item["content"] = "改过的一句话"
        items.append(
            message(
                id="01-000009999",
                localId=9999,
                serverId="1",
                createTime=target["createTime"] + 5,
                isSent=True,
                renderType="text",
                content="新增的一句",
                sortSeq=9_999_000,
            )
        )

    rewrite_export(
        export, edit, exported_at=datetime(2026, 11, 1, tzinfo=UTC), export_id="export-newer"
    )
    outcome = run_import(services, export)
    run = outcome.run
    assert (run.inserted, run.conflict_updated, run.conflict_kept) == (1, 1, 0)
    assert run.duplicates == 59
    with services.db.session() as session:
        changed = session.get(Message, target["id"])
        assert changed is not None and changed.text == "改过的一句话"
        assert changed.source_export_id == "export-newer"
        assert changed.created_at < changed.updated_at or changed.updated_at is not None
    assert row_count(services, Message) == 61
    report = Path(outcome.report_path or "").read_text(encoding="utf-8")
    assert "冲突并更新" in report


def test_an_older_export_does_not_overwrite_a_newer_version(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=40)
    rewrite_export(export, exported_at=datetime(2026, 11, 1, tzinfo=UTC), export_id="newer")
    run_import(services, export)
    first_text = next(
        m
        for m in json.loads(export.messages_path.read_text("utf-8"))["messages"]
        if m["renderType"] == "text"
    )

    def edit(items: list[dict[str, Any]]) -> None:
        for item in items:
            if item["id"] == first_text["id"]:
                item["content"] = "较早的导出里的改动"

    rewrite_export(export, edit, exported_at=datetime(2026, 9, 1, tzinfo=UTC), export_id="older")
    outcome = run_import(services, export)
    assert (outcome.run.conflict_updated, outcome.run.conflict_kept) == (0, 1)
    with services.db.session() as session:
        stored = session.get(Message, first_text["id"])
        assert stored is not None and stored.text == first_text["content"]
        assert stored.source_export_id == "newer"


def test_message_ids_of_another_conversation_are_refused(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=20, other_conversations=1)
    run_import(services, export)
    # the same ids now appear in another conversation's file
    other_dir = next(
        p for p in (export.root / "conversations").iterdir() if p.name.startswith("002_")
    )
    shutil_target = export.messages_path.read_bytes()
    (other_dir / "messages.json").write_bytes(shutil_target)
    with pytest.raises(ImportFailure, match="another conversation"):
        run_import(services, export, target=export.other_usernames[0])
    run = latest_run(services.db)
    assert (
        run is not None and run.status == "failed" and "another conversation" in (run.error or "")
    )


# --------------------------------------------------------------------- resuming


def test_a_run_killed_between_batches_resumes_to_the_same_result(
    services: Services, tmp_path: Path, clock: ManualClock, secret_store: SecretStore
) -> None:
    export = make_export(tmp_path, target_messages=520)

    def die_after_third_batch(event: BatchEvent) -> None:
        if event.batch_number == 3:
            raise PowerCut

    run_id = start_import(services, export)
    with pytest.raises(PowerCut):
        run_import(services, export, batch_size=50, on_batch=die_after_third_batch, run_id=run_id)
    interrupted = get_run(services.db, run_id)
    assert interrupted is not None
    assert (interrupted.status, interrupted.processed, interrupted.phase) == (
        "running",
        150,
        "messages",
    )
    assert row_count(services, Message) == 150  # exactly the committed batches

    resumed = prepare_import(services, export.root, target_username=export.target_username)
    assert resumed.resumed and resumed.run_id == run_id
    outcome = run_import(services, export, batch_size=50, run_id=run_id)
    assert outcome.status == "done" and outcome.run.processed == 520
    assert outcome.run.inserted == 520 and outcome.run.duplicates == 0
    assert outcome.run.stats.sessions == 2

    other = second_services(tmp_path, clock, secret_store)
    try:
        run_import(other, export, batch_size=50)
        assert message_snapshot(other) == message_snapshot(services)
        assert row_count(other, StickerUse) == row_count(services, StickerUse)
        assert row_count(other, MediaAsset) == row_count(services, MediaAsset)
    finally:
        other.close()


def test_a_failure_marks_the_run_and_resume_continues(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=120)

    def fail_once(event: BatchEvent) -> None:
        if event.batch_number == 2:
            raise RuntimeError("disk trouble")

    with pytest.raises(RuntimeError):
        run_import(services, export, batch_size=40, on_batch=fail_once)
    failed = latest_run(services.db)
    assert failed is not None and failed.status == "failed" and failed.processed == 80
    assert failed.error is not None and "RuntimeError" in failed.error
    unfinished = resumable_run(services)
    assert unfinished is not None and unfinished.id == failed.id
    outcome = run_import(services, export, batch_size=40, run_id=failed.id)
    assert outcome.status == "done" and row_count(services, Message) == 120
    assert outcome.run.error is None
    assert resumable_run(services) is None


def test_a_stop_request_ends_the_run_cleanly_and_resume_finishes_it(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=200)
    stop = threading.Event()

    def stop_after_first(event: BatchEvent) -> None:
        stop.set()

    first = run_import(services, export, batch_size=60, on_batch=stop_after_first, stop=stop)
    assert first.status == "interrupted" and first.run.status == "interrupted"
    assert first.run.processed == 60 and first.report_path is None
    final = run_import(services, export, batch_size=60, run_id=first.run.id)
    assert final.status == "done" and row_count(services, Message) == 200


def test_files_that_changed_since_the_run_was_prepared_restart_it(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=50)
    run_id = start_import(services, export)

    def append_message(items: list[dict[str, Any]]) -> None:
        items.append(
            message(
                id="x-1",
                localId=1,
                createTime=items[-1]["createTime"] + 1,
                isSent=False,
                renderType="text",
                content="后来加的",
                sortSeq=99_999_999,
            )
        )

    rewrite_export(export, append_message)
    outcome = run_import(services, export, run_id=run_id)
    assert outcome.run.processed == 51 and row_count(services, Message) == 51


def test_unfinished_runs_of_other_files_are_superseded(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=30)
    first = start_import(services, export)
    rewrite_export(export, lambda items: items.pop())
    second = start_import(services, export)
    assert second != first
    old = get_run(services.db, first)
    assert old is not None and old.status == "superseded"
    assert run_import(services, export, run_id=second).status == "done"
    assert resumable_run(services) is None


def test_running_a_finished_run_again_is_harmless(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=25)
    run_id = start_import(services, export)
    first = run_import(services, export, run_id=run_id)
    again = run_import(services, export, run_id=run_id)
    assert again.status == "done" and again.report_path == first.report_path
    assert row_count(services, Message) == 25


# ------------------------------------------------------------------------ media


def test_media_files_are_encrypted_into_the_store_and_linked(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=300, missing_image_ratio=0.3)
    run_import(services, export)
    with services.db.session() as session:
        available = session.scalars(
            select(MediaAsset).where(MediaAsset.status == "available", MediaAsset.kind == "image")
        ).all()
        assert available
        for asset in available:
            assert asset.sha256 and asset.mime in {"image/png", "image/jpeg"}
            assert services.media.exists(asset.sha256) and asset.width and asset.height
            stored = services.media.read_bytes(asset.sha256)
            assert hashlib.sha256(stored).hexdigest() == asset.sha256
            assert asset.size_bytes == len(stored)
            message_row = session.get(Message, asset.message_id)
            assert message_row is not None and message_row.media_sha256 == asset.sha256
    paths = sorted((services.paths.media_dir).glob("*.enc"))
    assert paths and b"PNG" not in paths[0].read_bytes()[:200]  # ciphertext, not the picture


def test_missing_media_is_recorded_with_its_reason(services: Services, tmp_path: Path) -> None:
    export = make_export(
        tmp_path, target_messages=300, missing_image_ratio=0.4, mix=IMAGE_HEAVY_MIX
    )
    # one file the manifest names but the disk lacks, one entry pointing out of the export
    victims = picture_files(export)
    victims[0].unlink()

    def point_outside(items: list[dict[str, Any]]) -> None:
        for item in items:
            media = item.get("offlineMedia")
            if media and item["renderType"] == "image" and victims[0].name not in media[0]["path"]:
                media[0]["path"] = "../outside.png"
                return

    rewrite_export(export, point_outside)
    run_import(services, export)
    with services.db.session() as session:
        reasons = Counter(
            (a.status, a.reason)
            for a in session.scalars(select(MediaAsset).where(MediaAsset.kind == "image"))
        )
    assert reasons[("missing", "file_not_found")] == 1, dict(reasons)
    assert reasons[("missing", "unsafe_path")] == 1
    assert reasons[("missing", "listed_missing")] == len(export.missing_message_ids)
    assert reasons[("available", None)] >= 1
    assert not any(status == "pending" for status, _ in reasons)


def test_a_video_without_a_cover_and_unknown_media_kinds_are_handled(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(
        tmp_path, target_messages=200, mix={"video": 50.0, "file": 20.0, "text": 30.0}, media=True
    )
    run_import(services, export)
    with services.db.session() as session:
        covers = Counter(
            (a.kind, a.status, a.reason)
            for a in session.scalars(select(MediaAsset))
            if a.kind != "avatar"
        )
    assert ("video_cover", "available", None) in covers
    assert ("video_cover", "missing", "not_exported") in covers
    assert not any(kind in ("image", "voice") for kind, _, _ in covers)  # files are not imported


def test_avatars_of_both_people_are_imported(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=60)
    run_import(services, export)
    with services.db.session() as session:
        conversation = session.scalars(select(Conversation)).one()
        her, mine = conversation.her_avatar_sha256, conversation.user_avatar_sha256
        assert her and mine and her != mine
        assert services.media.exists(her) and services.media.exists(mine)
        avatars = session.scalars(select(MediaAsset).where(MediaAsset.kind == "avatar")).all()
        assert len(avatars) == 2 and all(a.message_id is None for a in avatars)


# --------------------------------------------------------------------- stickers


def test_stickers_uses_and_local_files(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=400, sticker_kinds=8)
    outcome = run_import(services, export)
    emoji_total = export.total("emoji")
    assert emoji_total > 20
    with services.db.session() as session:
        stickers = {s.md5: s for s in session.scalars(select(Sticker))}
        assert set(stickers) <= set(export.stickers)
        local = {md5 for md5, info in export.stickers.items() if info.local}
        for md5, sticker in stickers.items():
            if md5 in local:
                assert sticker.status == "available" and sticker.sha256
                assert services.media.exists(sticker.sha256)
                assert sticker.mime == export.stickers[md5].mime
            else:
                assert sticker.status == "pending" and sticker.sha256 is None
                assert sticker.url == export.stickers[md5].url
        uses = session.scalars(select(StickerUse)).all()
        assert len(uses) == emoji_total
        by_her = sum(1 for u in uses if u.by_her)
        assert by_her == sum(
            n for (kind, sent), n in export.counts.items() if kind == "emoji" and not sent
        )
        for sticker in stickers.values():
            mine = [u for u in uses if u.sticker_md5 == sticker.md5]
            assert sticker.her_uses == sum(1 for u in mine if u.by_her)
            assert sticker.user_uses == sum(1 for u in mine if not u.by_her)
            assert sticker.first_used_at == min(u.used_at for u in mine)
            assert sticker.last_used_at == max(u.used_at for u in mine)
        message_row = session.get(Message, uses[0].message_id)
        assert message_row is not None and message_row.sticker_md5 == uses[0].sticker_md5
        assert uses[0].used_at == message_row.create_time_utc
    assert outcome.run.stats.stickers["local_files"] == len(local & set(stickers))


def test_only_available_stickers_may_be_sent(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=300, sticker_kinds=8, local_sticker_ratio=0.5)
    run_import(services, export)
    with services.db.session() as session:
        stickers = session.scalars(select(Sticker)).all()
        available = [s for s in stickers if s.status == "available"]
        pending = [s for s in stickers if s.status == "pending"]
        assert available and pending
        assert all(s.sha256 and sticker_file_allowed(session, s.sha256) for s in available)
        photo = session.scalars(
            select(MediaAsset).where(MediaAsset.kind == "image", MediaAsset.status == "available")
        ).first()
        assert photo is not None and photo.sha256
        assert not sticker_file_allowed(session, photo.sha256)  # a photo is never allowed
        assert not sticker_file_allowed(session, "0" * 64)
        assert not sticker_file_allowed(session, "not a hash")
        assert not sticker_file_allowed(session, available[0].sha256.upper())  # type: ignore[union-attr]


def test_a_sticker_file_with_the_wrong_md5_is_kept_but_not_sendable(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=200, local_sticker_ratio=1.0)
    victim_md5, info = next(iter(export.stickers.items()))
    path = next((export.root / "media" / "emojis").glob(f"{victim_md5}.*"))
    path.write_bytes(info.data + b"\x00\x00")  # still decodable, but no longer the same file
    run_import(services, export)
    with services.db.session() as session:
        sticker = session.get(Sticker, victim_md5)
        assert sticker is not None and sticker.status == "md5_mismatch"
        assert sticker.sha256 and services.media.exists(sticker.sha256)
        assert not sticker_file_allowed(session, sticker.sha256)


def test_a_local_file_that_is_not_a_picture_makes_the_sticker_unavailable(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=120, local_sticker_ratio=1.0)
    victim_md5 = next(iter(export.stickers))
    path = next((export.root / "media" / "emojis").glob(f"{victim_md5}.*"))
    path.write_bytes(b"this is not a picture")
    run_import(services, export)
    with services.db.session() as session:
        sticker = session.get(Sticker, victim_md5)
        assert sticker is not None
        assert (sticker.status, sticker.reason) == ("unavailable", "not_an_image")


# ------------------------------------------------------------------------ report


def test_the_report_contains_counts_but_no_message_text_or_ids(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=300)
    outcome = run_import(services, export)
    assert outcome.report_path is not None
    path = Path(outcome.report_path)
    assert path.parent == services.paths.reports_dir and path.name.startswith("import-")
    text = path.read_text(encoding="utf-8")
    for sentence in export.texts:
        assert sentence not in text
    assert export.target_username not in text and export.target_display_name not in text
    for other in export.other_usernames:
        assert other not in text
    assert "wxid_" not in text and "@chatroom" not in text
    for heading in (
        "## 本次导入",
        "## 按类型与发送方计数",
        "## 媒体文件",
        "## 表情包",
        "## 导入后钩子",
    ):
        assert heading in text
    assert "synthetic-export-0001" in text and "300" in text
    assert "image_caption" in text and "sticker_download" in text
    # the date range is in the source time zone
    first = export.first_time
    assert first is not None
    assert (
        first.astimezone(__import__("zoneinfo").ZoneInfo("America/Chicago")).date().isoformat()
        in text
    )


def test_the_report_counts_each_kind_by_sender(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=200, mix={"text": 70.0, "emoji": 30.0})
    outcome = run_import(services, export)
    text = Path(outcome.report_path or "").read_text(encoding="utf-8")
    her_text = export.counts[("text", False)]
    user_text = export.counts[("text", True)]
    assert f"| 文字 | {her_text} | {user_text} | {her_text + user_text} |" in text


# --------------------------------------------------------------------- integrity


def test_a_messages_file_that_fails_the_integrity_check_is_not_imported(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=40)
    relative = f"conversations/{export.target_dir.name}/messages.json"
    write_manifest(export.root, "json", corrupt=(relative,))
    with pytest.raises(IntegrityFailure, match="integrity"):
        run_import(services, export)
    assert row_count(services, Message) == 0
    run = latest_run(services.db)
    assert run is not None and run.status == "failed" and "integrity" in (run.error or "")


def test_media_that_fails_the_integrity_check_is_reported_and_not_stored(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=200, mix=IMAGE_HEAVY_MIX)
    victim = picture_files(export)[0]
    write_manifest(export.root, "sha256sum", corrupt=(victim.relative_to(export.root).as_posix(),))
    outcome = run_import(services, export)
    with services.db.session() as session:
        corrupt = session.scalars(select(MediaAsset).where(MediaAsset.status == "corrupt")).all()
        assert len(corrupt) == 1 and corrupt[0].reason == "integrity_failed"
        assert corrupt[0].sha256 is None
        good = session.scalars(
            select(MediaAsset).where(MediaAsset.status == "available", MediaAsset.kind == "image")
        ).all()
        assert good
    assert not services.media.exists(hashlib.sha256(victim.read_bytes()).hexdigest())
    assert outcome.run.stats.media["integrity_failed"] == 1
    assert outcome.run.stats.media["integrity_verified"] >= 1
    assert "完整性校验" in Path(outcome.report_path or "").read_text(encoding="utf-8")


def test_an_unrecognised_integrity_folder_is_reported_but_does_not_block(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=30)
    folder = export.root / "_integrity"
    folder.mkdir()
    (folder / "mystery.bin").write_bytes(b"\x00\x01\x02")
    outcome = run_import(services, export)
    assert outcome.status == "done"
    assert any("not recognised" in note for note in outcome.run.stats.notes)


# --------------------------------------------------------------------- data quality


def test_a_time_zone_mistake_shows_up_in_the_report(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=60, shift_text_hours=5)
    outcome = run_import(services, export)
    assert outcome.run.stats.time_mismatches == 60
    text = Path(outcome.report_path or "").read_text(encoding="utf-8")
    assert "createTimeText" in text and "time.source_timezone" in text


def test_messages_that_cannot_be_stored_are_counted_with_a_reason(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=30)

    def damage(items: list[dict[str, Any]]) -> None:
        items.append(message(isSent=True, createTime=items[-1]["createTime"], renderType="text"))
        items.append(message(id="no-time", isSent=True, renderType="text"))
        items.append(
            message(id="no-sender", createTime=items[-1].get("createTime") or 1_700_000_000)
        )
        items.append("not an object")  # type: ignore[arg-type]
        items.append(
            message(
                id="odd-type",
                createTime=1_735_732_800,
                isSent=False,
                renderType=["text"],
                content="类型不对的字段",
            )
        )

    rewrite_export(export, damage)
    outcome = run_import(services, export)
    run = outcome.run
    assert run.processed == 35 and run.invalid == 4 and run.inserted == 31
    assert run.stats.invalid_reasons == {
        "no_id": 1,
        "no_time": 1,
        "no_sender": 1,
        "not_an_object": 1,
    }
    assert run.stats.schema_errors["renderType"] == 1
    with services.db.session() as session:
        odd = session.get(Message, "odd-type")
        assert odd is not None and odd.kind == "unknown" and odd.render_type is None


def test_unknown_render_types_and_fields_are_noted_by_name_only(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=120, mix={"holographic": 40.0, "text": 60.0})
    outcome = run_import(services, export)
    assert outcome.run.stats.unknown_render_types["holographic"] > 0
    assert outcome.run.stats.unknown_fields["novelField"] > 0
    text = Path(outcome.report_path or "").read_text(encoding="utf-8")
    assert "holographic" in text and "novelField" in text
    with services.db.session() as session:
        row = session.scalars(select(Message).where(Message.kind == "unknown")).first()
        assert row is not None and row.raw["novelField"].startswith("n")


def test_a_file_that_is_cut_off_fails_with_a_safe_message_and_keeps_progress(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=100)
    data = export.messages_path.read_bytes()
    export.messages_path.write_bytes(data[: len(data) * 3 // 4])
    with pytest.raises(ImportFailure, match="cut off"):
        run_import(services, export, batch_size=10)
    run = latest_run(services.db)
    assert run is not None and run.status == "failed" and "valid JSON" in (run.error or "")
    assert 0 < run.processed < 100 and row_count(services, Message) == run.processed
    for sentence in export.texts:
        assert sentence not in (run.error or "")


def test_a_byte_order_mark_and_float_numbers_do_not_matter(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=30, mix={"location": 100.0})
    export.messages_path.write_bytes(b"\xef\xbb\xbf" + export.messages_path.read_bytes())
    outcome = run_import(services, export)
    assert outcome.run.processed == 30
    with services.db.session() as session:
        row = session.scalars(select(Message)).first()
        assert row is not None and isinstance(row.raw["locationLat"], float)


def test_the_run_stores_its_directory_and_target_encrypted(
    services: Services, tmp_path: Path
) -> None:
    export = make_export(tmp_path, target_messages=10)
    run_import(services, export)
    services.db.engine.dispose()
    connection = sqlite3.connect(tmp_path / "data" / "twin.db")
    try:
        blob = b"".join(
            bytes(value)
            for row in connection.execute("SELECT source_dir, target_username FROM import_runs")
            for value in row
        )
    finally:
        connection.close()
    assert str(export.root).encode() not in blob and export.target_username.encode() not in blob


def test_the_import_runs_against_a_tiny_batch_size_and_a_huge_one(
    services: Services, tmp_path: Path, clock: ManualClock, secret_store: SecretStore
) -> None:
    export = make_export(tmp_path, target_messages=90)
    run_import(services, export, batch_size=1)
    small = message_snapshot(services)
    other = second_services(tmp_path, clock, secret_store)
    try:
        run_import(other, export, batch_size=100_000)
        assert message_snapshot(other) == small
    finally:
        other.close()


def test_defaults_come_from_the_configuration(services: Services, tmp_path: Path) -> None:
    assert ImportRunner(services)._batch_size == services.settings.ingest.batch_size == 2000
    assert ImportRunner(services, batch_size=17)._batch_size == 17
