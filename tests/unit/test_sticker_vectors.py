"""Description vectors of the stickers, in a table of their own (R-STK-003, R-STO-005)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.support.embedding import HashingBackend
from tests.support.persona import sticker_scenario
from twin.retrieval import embedder as embedder_module
from twin.retrieval.embedder import reset_embedding_services
from twin.services import Services
from twin.stickers.catalog import StickerCatalog, StickerRecord
from twin.stickers.vectors import StickerVectors, description_text, encoding_of, sticker_table
from twin.storage.chat_models import Sticker
from twin.storage.vector_schema import STICKER_SCHEMA, VectorSchemaError

NOW = datetime(2026, 5, 1, 12, tzinfo=UTC)
DESCRIPTIONS = [
    ("开心", "一只笑得眯起眼睛的黄色小猫", "收到好消息时回应"),
    ("委屈", "一只低着头流泪的小狗", "被说了几句之后撒娇"),
    ("晚安", "月亮下睡着的小熊", "说晚安的时候"),
]


@pytest.fixture
def library(services: Services, embedder: HashingBackend) -> tuple[Services, StickerCatalog]:
    sticker_scenario(services)
    catalog = StickerCatalog(services)
    # the three most used stickers are described, the fourth stays without a description
    for md5, (tag, description, use) in zip(md5s_by_use(catalog), DESCRIPTIONS, strict=False):
        catalog.save_vision(md5, [tag], description, use, at=NOW)
    return services, catalog


def md5s_by_use(catalog: StickerCatalog) -> list[str]:
    return [r.md5 for r in catalog.records()]


def test_the_text_that_is_encoded_joins_tags_description_use_cases_and_her_use() -> None:
    record = StickerRecord(
        md5="a",
        status="available",
        mime=None,
        width=None,
        height=None,
        her_uses=1,
        user_uses=0,
        first_used_at=None,
        last_used_at=None,
        tags=("开心", "大笑"),
        vision_tags=("开心",),
        context_tags=(),
        manual_tags=(),
        tag_source="vision",
        description="笑猫。",
        use_cases="好消息",
        context_note="她用来回应夸奖",
        origin="import",
        disabled=False,
        tagged_at=None,
        context_tagged_at=None,
        context_cutoff_at=None,
        context_uses=0,
        desc_vector_id=None,
        desc_encoding=None,
        sha256=None,
    )
    assert description_text(record) == "开心、大笑。笑猫。好消息。她用来回应夸奖"


def test_each_described_sticker_of_hers_gets_a_vector_and_nothing_else_is_stored(
    library: tuple[Services, StickerCatalog], embedder: HashingBackend
) -> None:
    services, catalog = library
    sync = StickerVectors(services).sync()
    assert (sync.encoded, sync.removed, sync.reset) == (3, 0, False)
    table = sticker_table(services)
    assert table.count() == 3 and table.columns() == ["id", "vector", "at", "kind"]
    assert {row["id"] for row in table.rows()} == set(md5s_by_use(catalog)[:3])
    assert {row["kind"] for row in table.rows()} == {"sticker"}
    undescribed = md5s_by_use(catalog)[3]
    assert catalog.require(undescribed).desc_vector_id is None
    for record in catalog.records()[:3]:
        assert record.desc_vector_id == record.md5
        assert record.desc_encoding == encoding_of(embedder.info)
    assert table.read_meta() is not None and table.read_meta().model == embedder.info.model  # type: ignore[union-attr]
    # the table is a table of its own: the generic one does not exist
    assert STICKER_SCHEMA.name == "sticker_descriptions"
    with pytest.raises(VectorSchemaError, match="sticker record kind"):
        STICKER_SCHEMA.validate({"id": "a", "vector": [0.1], "at": 1, "kind": "message"})


def test_only_the_stickers_that_changed_are_encoded_again(
    library: tuple[Services, StickerCatalog], embedder: HashingBackend
) -> None:
    services, catalog = library
    vectors = StickerVectors(services)
    vectors.sync()
    runs = embedder.runs
    assert vectors.sync().encoded == 0 and embedder.runs == runs
    changed = md5s_by_use(catalog)[1]
    catalog.save_vision(changed, ["难过"], "一只哭泣的小猫", "难过的时候", at=NOW)
    again = vectors.sync()
    assert again.encoded == 1 and catalog.require(changed).desc_encoding == encoding_of(
        embedder.info
    )
    only = vectors.sync([md5s_by_use(catalog)[0]])
    assert only.encoded == 0


def test_a_description_that_is_gone_loses_its_vector(
    library: tuple[Services, StickerCatalog],
) -> None:
    services, catalog = library
    vectors = StickerVectors(services)
    vectors.sync()
    gone = md5s_by_use(catalog)[2]
    with services.db.transaction(bump_state=False) as session:
        row = session.get(Sticker, gone)
        assert row is not None
        row.description = None
    result = vectors.sync()
    assert result.removed == 1 and sticker_table(services).count() == 2


def test_another_model_rebuilds_the_table_instead_of_mixing_vectors(
    library: tuple[Services, StickerCatalog], monkeypatch: pytest.MonkeyPatch
) -> None:
    services, catalog = library
    StickerVectors(services).sync()
    other = HashingBackend(model="test/other-model", weights="other-weights")
    monkeypatch.setattr(embedder_module, "backend_factory", lambda config, paths: other)
    reset_embedding_services()
    result = StickerVectors(services).sync()
    assert result.reset and result.encoded == 3
    meta = sticker_table(services).read_meta()
    assert meta is not None and meta.model == "test/other-model"
    assert all(r.desc_encoding == encoding_of(other.info) for r in catalog.records()[:3])


def test_the_context_is_compared_with_the_descriptions(
    library: tuple[Services, StickerCatalog],
) -> None:
    services, catalog = library
    vectors = StickerVectors(services)
    assert vectors.similarities("好消息", md5s_by_use(catalog)) == {}  # no table yet
    vectors.sync()
    wanted = md5s_by_use(catalog)
    close = vectors.similarities("收到好消息时回应笑得眯起眼睛的小猫", wanted)
    assert set(close) == set(wanted[:3])
    assert max(close, key=close.__getitem__) == wanted[0]
    sleepy = vectors.similarities("说晚安月亮下睡着的小熊", wanted)
    assert max(sleepy, key=sleepy.__getitem__) == wanted[2]
    assert vectors.similarities("好消息", [wanted[1]]).keys() == {wanted[1]}
    assert vectors.similarities("好消息", []) == {}


def test_text_is_redacted_before_it_is_encoded(
    library: tuple[Services, StickerCatalog], embedder: HashingBackend
) -> None:
    services, _ = library
    vectors = StickerVectors(services)
    vectors.sync()
    phone = "1" + "38" + "12345678"
    vectors.similarities(f"我的电话是{phone}", [r.md5 for r in library[1].records()])
    assert any("[手机号]" in text for text in embedder.seen) and not any(
        phone in t for t in embedder.seen
    )


def test_a_table_made_with_another_model_gives_no_similarities_instead_of_an_error(
    library: tuple[Services, StickerCatalog], monkeypatch: pytest.MonkeyPatch
) -> None:
    services, catalog = library
    StickerVectors(services).sync()
    other = HashingBackend(dimension=64, model="test/other-model", weights="other-weights")
    monkeypatch.setattr(embedder_module, "backend_factory", lambda config, paths: other)
    reset_embedding_services()
    assert StickerVectors(services).similarities("好消息", md5s_by_use(catalog)) == {}
    assert StickerVectors(services).sync().reset  # and `tag-all` makes them again
    assert StickerVectors(services).similarities("好消息", md5s_by_use(catalog)) != {}


def test_the_model_is_not_loaded_when_every_vector_is_current(
    library: tuple[Services, StickerCatalog], embedder: HashingBackend
) -> None:
    services, _ = library
    services.settings.retrieval.model = embedder.info.model  # the configured model made the table
    vectors = StickerVectors(services)
    assert vectors.sync().encoded == 3
    reset_embedding_services()  # a new process: nothing is loaded yet
    fresh = StickerVectors(services)
    runs = embedder.runs
    assert fresh.sync().encoded == 0 and embedder.runs == runs
    assert fresh._embedder is None
