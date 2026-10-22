"""The label of a sticker: one table for the terminal, the examples and the training (D-691).

R17-12, R17-13: the label (first tag) of a library sticker shown in ``[表情包:<标签>]`` of an
example, and in ``[表情包：<标签>]`` of the terminal; the table is read once and read again when
``settings.state_version`` moves.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from tests.support.persona import Scenario, sticker_scenario
from twin.services import Services
from twin.stickers.catalog import StickerCatalog, StickerRecord, label_of
from twin.stickers.labels import StickerLabelBook, sticker_labels
from twin.storage.state import bump_state_version, current_state_version

NOW = datetime(2026, 5, 1, 12, tzinfo=UTC)


@pytest.fixture
def library(services: Services) -> Scenario:
    found = sticker_scenario(services)
    catalog = StickerCatalog(services)
    catalog.save_vision(found.md5s[0], ["开心", "大笑"], "笑猫", "好消息", at=NOW)
    catalog.save_vision(found.md5s[1], ["撒娇"], "歪头猫", "撒娇时", at=NOW)
    return found  # the other two stickers have no tags


def sha_of(services: Services, md5: str) -> str:
    record = StickerCatalog(services).require(md5)
    assert record.sha256 is not None
    return record.sha256


def counting_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    reads: list[int] = []
    original = StickerCatalog.records

    def counted(self: StickerCatalog, **kwargs: object) -> list[StickerRecord]:
        reads.append(1)
        return original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(StickerCatalog, "records", counted)
    return reads


def bump(services: Services) -> None:
    with services.db.transaction(bump_state=False) as session:
        bump_state_version(session, services.clock)


def test_a_label_is_the_first_tag_by_md5_and_by_the_hash_of_the_file(
    library: Scenario, services: Services
) -> None:
    book = StickerLabelBook(services)
    first, second = library.md5s[0], library.md5s[1]
    assert book.label_md5(first) == "开心" and book.label_md5(second) == "撒娇"
    assert book.label(sha_of(services, first)) == "开心"
    assert book.label(sha_of(services, second)) == "撒娇"
    record = StickerCatalog(services).require(first)
    assert (
        label_of(record) == "开心"
        and label_of(StickerCatalog(services).require(library.md5s[2])) is None
    )


def test_a_sticker_without_tags_or_outside_the_library_has_no_label(
    library: Scenario, services: Services
) -> None:
    book = StickerLabelBook(services)
    assert book.label_md5(library.md5s[2]) is None  # in the library, not tagged
    assert book.label(sha_of(services, library.md5s[3])) is None
    assert book.label_md5("0" * 32) is None and book.label("f" * 64) is None


def test_the_table_is_read_once_and_again_only_when_the_state_version_moves(
    library: Scenario, services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = counting_reads(monkeypatch)
    book = StickerLabelBook(services)
    for _ in range(25):
        assert book.label_md5(library.md5s[0]) == "开心"
        assert book.label(sha_of(services, library.md5s[1])) == "撒娇"
    assert len(reads) == 1  # fifty questions, one read of the library
    # a job tags a sticker with no bump of the state version: the book does not know yet
    StickerCatalog(services).save_vision(library.md5s[2], ["晚安"], "睡觉的兔子", "晚上", at=NOW)
    assert book.label_md5(library.md5s[2]) is None and len(reads) == 1
    bump(services)  # ... and is told, as every change the running application must hear about is
    assert book.label_md5(library.md5s[2]) == "晚安" and len(reads) == 2
    assert book.label_md5(library.md5s[0]) == "开心" and len(reads) == 2


def test_a_tag_set_by_hand_is_seen_at_once(library: Scenario, services: Services) -> None:
    book = StickerLabelBook(services)
    assert book.label_md5(library.md5s[0]) == "开心"
    StickerCatalog(services).set_manual(library.md5s[0], ["晚安"])  # bumps the state version
    assert book.label_md5(library.md5s[0]) == "晚安"
    assert book.label(sha_of(services, library.md5s[0])) == "晚安"
    StickerCatalog(services).clear_manual(library.md5s[0])
    assert book.label_md5(library.md5s[0]) == "开心"


def test_reload_forgets_the_table_without_a_new_version(
    library: Scenario, services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = counting_reads(monkeypatch)
    book = StickerLabelBook(services)
    assert book.label_md5(library.md5s[0]) == "开心"
    before = current_state_version(services.db)
    book.reload()
    assert book.label_md5(library.md5s[0]) == "开心" and len(reads) == 2
    assert current_state_version(services.db) == before


def test_the_book_gives_the_labels_the_training_export_and_the_evaluation_use(
    library: Scenario, services: Services
) -> None:
    """Iron rules 8 and 13: one definition of a sticker's label for the live bot and the data."""
    lookup = StickerCatalog(services).tag_lookup()
    book = StickerLabelBook(services)
    for md5 in library.md5s:
        assert book.label_md5(md5) == lookup(md5)


def test_one_book_serves_a_container(library: Scenario, services: Services) -> None:
    assert sticker_labels(services) is sticker_labels(services)
    assert isinstance(sticker_labels(services), StickerLabelBook)


def test_the_book_is_safe_to_ask_from_several_threads(
    library: Scenario, services: Services
) -> None:
    book = StickerLabelBook(services)
    answers: list[str | None] = []
    lock = threading.Lock()

    def ask() -> None:
        found = book.label_md5(library.md5s[0])
        with lock:
            answers.append(found)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert answers == ["开心"] * 8
