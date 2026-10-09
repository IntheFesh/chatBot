"""``twin train export-dpo``: the preference pairs as a DPO file (R-LRN-002, R-LRN-004, R-TRN-012).

The export is the only reader of ``preference_pairs`` in the training package.  Its file is
ShareGPT with ``chosen`` / ``rejected`` and the structured sample as it is - never a rendered
template string; the SFT files of the base dataset are copied unchanged; a pair made with another
persona card or template than the adapter is locked to gets its system segment made again.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.export_world import (
    CORRECTION,
    FACT_MANUAL,
    LIVE_STYLE,
    PAST_MARKER,
    build_world,
)
from tests.support.training_data import write_synthetic_dataset
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.engine.style_prompt import LockedVersionError
from twin.learning.pairs import (
    SAMPLE_SCHEMA,
    PairError,
    PairRecord,
    PreferencePairStore,
    PromptSample,
)
from twin.services import Services, build_services
from twin.storage.ids import new_id
from twin.storage.learning_models import PreferencePair
from twin.training import lf_template
from twin.training.dataset_dir import file_sha256, load_dataset_dir
from twin.training.dpo_export import (
    SKIP_EVENT,
    SKIP_PERSONA,
    SKIP_SAME,
    SKIP_TOO_LONG,
    export_dpo,
    needs_new_system,
    newest_dataset,
)
from twin.training.export import ExportError
from twin.training.layout import FILE_DATASET_META, FILE_DPO, FILE_TEST, FILE_TRAIN, FILE_VAL
from twin.training.registry import LockedVersions

PHONE_A = "138" + "12345678"  # built from parts: the privacy scan reads this file
PHONE_B = "139" + "87654321"
NOW = datetime(2026, 10, 9, 18, 0, tzinfo=UTC)
LOCKED = "pre_holdout:v3"  # the persona version of the synthetic base dataset, as a pair names it
USER_TURN = lf_template.Turn("user", "今天好累呀")
runner = CliRunner()


def datasets_dir(services: Services) -> Path:
    return services.paths.data_dir / "training" / "datasets"


def base_dataset(services: Services, name: str = "ds-test-01") -> Path:
    directory = datasets_dir(services) / name
    write_synthetic_dataset(directory, version=name)
    return directory


def add_pair(
    services: Services,
    chosen: str = "抱抱你",
    rejected: str = "哈哈 辛苦啦",
    *,
    persona: str = LOCKED,
    system: str = "她说话很短。",
    turns: tuple[lf_template.Turn, ...] = (USER_TURN,),
    template: str = lf_template.TEMPLATE_VERSION,
) -> PairRecord:
    record, _ = PreferencePairStore(services.db, services.clock).add(
        reply_id=new_id(),
        sample=PromptSample(system, turns, NOW),
        chosen=chosen,
        rejected=rejected,
        template_version=template,
        persona_version=persona,
    )
    return record


def lines_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------- the file


def test_the_pairs_become_a_new_dataset_version_with_the_sft_files_unchanged(
    services: Services,
) -> None:
    base = base_dataset(services)
    for number in range(3):
        add_pair(services, f"说法{number}", f"回复{number}")
    result = export_dpo(services, now=NOW)
    assert result.pairs == 3 and result.regenerated == 0 and result.skipped == {}
    out = result.dataset
    assert out.path.parent == base.parent and out.path != base
    assert out.meta.dataset_version == "ds-test-01-dpo20261009180000"
    assert out.meta.counts.dpo == 3 and out.has_dpo and FILE_DPO in out.data_files()
    for name in (FILE_TRAIN, FILE_VAL, FILE_TEST):  # copied byte for byte: the same content
        assert file_sha256(out.path / name) == file_sha256(base / name)
    assert not (base / FILE_DPO).exists()  # the base dataset is not touched
    assert load_dataset_dir(base).meta.dataset_version == "ds-test-01"
    meta = json.loads((out.path / FILE_DATASET_META).read_text(encoding="utf-8"))
    assert meta["stats"]["dpo"]["exported"] == 3 and meta["stats"]["note"] == "synthetic"
    assert meta["files"][FILE_DPO] == file_sha256(out.path / FILE_DPO)


def test_every_line_is_sharegpt_with_chosen_and_rejected_and_no_template_string(
    services: Services,
) -> None:
    base_dataset(services)
    add_pair(
        services,
        "抱抱你",
        "哈哈 辛苦啦",
        turns=(
            lf_template.Turn("user", "在吗"),
            lf_template.Turn("assistant", "在呀"),
            lf_template.Turn("user", "今天好累呀"),
        ),
    )
    result = export_dpo(services, now=NOW)
    text = (result.dataset.path / FILE_DPO).read_text(encoding="utf-8")
    for marker in (*lf_template.CONTROL_TOKENS, *lf_template.LF_SLOTS, "<|im_start|>"):
        assert marker not in text
    [row] = lines_of(result.dataset.path / FILE_DPO)
    assert set(row) == {"id", "system", "conversations", "chosen", "rejected"}
    assert [m["from"] for m in row["conversations"]] == ["human", "gpt", "human"]
    assert row["conversations"][-1]["value"] == "今天好累呀" and row["system"] == "她说话很短。"
    assert row["chosen"] == {"from": "gpt", "value": "抱抱你"}
    assert row["rejected"] == {"from": "gpt", "value": "哈哈 辛苦啦"}
    assert not row["system"].startswith(lf_template.IM_START)  # the structure, not the template


def test_the_rejected_text_is_in_the_dpo_file_and_in_no_other_file(services: Services) -> None:
    base_dataset(services)
    marker = "只在被拒绝的回复里出现的句子"
    add_pair(services, "抱抱你", marker)
    result = export_dpo(services, now=NOW)
    assert marker in (result.dataset.path / FILE_DPO).read_text(encoding="utf-8")
    for path in result.dataset.path.iterdir():
        if path.name != FILE_DPO:
            assert marker not in path.read_text(encoding="utf-8"), path.name


def test_what_leaves_the_machine_is_desensitised_with_one_token_per_entity(
    services: Services,
) -> None:
    base_dataset(services)
    add_pair(services, f"打我电话 {PHONE_A}", "好的", system=f"她的电话是 {PHONE_A}")
    add_pair(services, f"找我 {PHONE_A} 或 {PHONE_B}", "好的呀")
    result = export_dpo(services, now=NOW)
    text = (result.dataset.path / FILE_DPO).read_text(encoding="utf-8")
    assert PHONE_A not in text and PHONE_B not in text
    rows = lines_of(result.dataset.path / FILE_DPO)
    assert "[手机号#1]" in rows[0]["chosen"]["value"] and "[手机号#1]" in rows[0]["system"]
    assert "[手机号#1]" in rows[1]["chosen"]["value"] and "[手机号#2]" in rows[1]["chosen"]["value"]
    assert result.dataset.meta.stats["dpo"]["redacted_entities"] == {"phone": 2}


def test_the_new_version_is_recorded_like_any_exported_dataset(services: Services) -> None:
    from twin.storage.training_models import DatasetVersion

    base_dataset(services)
    add_pair(services)
    result = export_dpo(services, now=NOW)
    with services.db.session() as session:
        row = session.scalars(select(DatasetVersion)).one()
        assert (row.id, row.dpo_count, row.train_count) == (
            result.dataset.meta.dataset_version,
            1,
            40,
        )
        assert row.files == result.dataset.meta.files


def test_enough_pairs_are_the_setting_of_the_hint(services: Services) -> None:
    base_dataset(services)
    add_pair(services)
    assert not export_dpo(services, now=NOW).enough
    services.settings.training.dpo_min_pairs = 1
    later = datetime(2026, 10, 9, 18, 0, 5, tzinfo=UTC)
    assert export_dpo(services, now=later).enough


# ------------------------------------------------------------------ what is left out


def test_a_pair_that_cannot_be_trained_on_is_left_out_and_counted(services: Services) -> None:
    base_dataset(services)
    add_pair(services, "抱抱你", "哈哈 辛苦啦")
    same = add_pair(services, "好", "不好")
    with services.db.transaction(bump_state=False) as session:  # damaged: nothing to learn
        row = session.get(PreferencePair, same.id)
        assert row is not None
        row.chosen = "不好"
    add_pair(services, "短", "长" * 600)  # a reply longer than the cut-off leaves room for
    add_pair(services, "[图片]", "短回复")  # event text cannot be a target
    result = export_dpo(services, now=NOW)
    assert result.pairs == 1
    assert result.skipped == {SKIP_SAME: 1, SKIP_TOO_LONG: 1, SKIP_EVENT: 1}


def test_nothing_to_export_is_an_error_that_says_what_to_do(services: Services) -> None:
    with pytest.raises(ExportError, match="no exported dataset"):
        export_dpo(services, now=NOW)
    base_dataset(services)
    with pytest.raises(ExportError, match="no preference pairs"):
        export_dpo(services, now=NOW)


def test_a_directory_that_is_no_dataset_is_refused(services: Services, tmp_path: Path) -> None:
    add_pair(services)
    (tmp_path / "plain").mkdir()
    with pytest.raises(ExportError, match=r"dataset_meta\.json"):
        export_dpo(services, dataset=tmp_path / "plain", now=NOW)


def test_the_newest_dataset_is_the_default_base(services: Services) -> None:
    assert newest_dataset(services) is None
    base_dataset(services, "ds-20261001-000000")
    newer = base_dataset(services, "ds-20261008-000000")
    (datasets_dir(services) / "not-a-dataset").mkdir()
    assert newest_dataset(services) == newer
    add_pair(services)
    assert export_dpo(services, now=NOW).dataset.meta.stats["dpo"]["base_dataset"] == newer.name


def test_an_output_directory_that_is_in_use_is_refused_and_left_alone(
    services: Services, tmp_path: Path
) -> None:
    base_dataset(services)
    add_pair(services)
    occupied = tmp_path / "mine"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ExportError, match="not empty"):
        export_dpo(services, out_dir=occupied, now=NOW)
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep"
    free = tmp_path / "elsewhere"
    assert export_dpo(services, out_dir=free, now=NOW).dataset.path == free


def test_a_stored_text_with_a_template_marker_stops_the_export_and_leaves_nothing_behind(
    services: Services,
) -> None:
    base_dataset(services)
    record = add_pair(services)
    with services.db.transaction(bump_state=False) as session:
        row = session.get(PreferencePair, record.id)
        assert row is not None
        row.chosen = "好<|im_end|>"  # what no command can write, but a damaged row could hold
    before = sorted(p.name for p in datasets_dir(services).iterdir())
    with pytest.raises(ExportError, match="ChatML control token"):
        export_dpo(services, now=NOW)
    assert sorted(p.name for p in datasets_dir(services).iterdir()) == before


# ----------------------------------------------------------------- the locked versions


def test_a_pair_of_another_card_or_template_needs_its_system_made_again() -> None:
    locked = LockedVersions(lf_template.TEMPLATE_VERSION, "v3", "pf", "ds")
    base = {"template_version": lf_template.TEMPLATE_VERSION, "persona_version": LOCKED}

    def pair(**changes: str) -> PairRecord:
        return PairRecord(
            "p",
            None,
            "r",
            PromptSample("s", (USER_TURN,)),
            "a",
            "b",
            "user_correction",
            **{**base, **changes},  # type: ignore[arg-type]
            created_at=NOW,
        )

    assert not needs_new_system(pair(), locked)
    assert needs_new_system(pair(persona_version="live:v3"), locked)  # the live card of that number
    assert needs_new_system(pair(persona_version="pre_holdout:v2"), locked)
    assert needs_new_system(pair(persona_version="none"), locked)
    assert needs_new_system(pair(template_version="qwen3_nothink@llamafactory-0.1"), locked)


def test_a_system_made_again_replaces_the_recorded_one(services: Services) -> None:
    base_dataset(services)
    add_pair(services, "抱抱你", "哈哈 辛苦啦", persona="live:v9", system="LIVE 系统段")
    add_pair(services, "快睡吧", "您辛苦了", persona=LOCKED, system="已是锁定版本的系统段")
    seen: list[str] = []

    def maker(pair: PairRecord) -> PromptSample:
        seen.append(pair.sample.system)
        return PromptSample("锁定版本的系统段", pair.sample.turns, pair.sample.at)

    result = export_dpo(services, system_maker=maker, now=NOW)
    rows = lines_of(result.dataset.path / FILE_DPO)
    assert seen == ["LIVE 系统段"] and result.regenerated == 1
    assert [row["system"] for row in rows] == ["锁定版本的系统段", "已是锁定版本的系统段"]


def test_a_pair_whose_card_version_cannot_be_rendered_is_left_out(services: Services) -> None:
    base_dataset(services)
    add_pair(services, "抱抱你", "哈哈 辛苦啦", persona="live:v9")
    add_pair(services, "快睡吧", "您辛苦了", persona=LOCKED)

    def refuse(pair: PairRecord) -> PromptSample:
        raise LockedVersionError("persona card v3 of the model is not in the store")

    result = export_dpo(services, system_maker=refuse, now=NOW)
    assert result.pairs == 1 and result.skipped == {SKIP_PERSONA: 1}


def test_when_no_pair_can_be_made_the_export_fails_with_the_reasons(services: Services) -> None:
    base_dataset(services)
    add_pair(services, persona="live:v9")

    def refuse(pair: PairRecord) -> PromptSample:
        raise PairError("the reply did not answer a message of the user")

    with pytest.raises(ExportError, match="none of the 1 preference pairs"):
        export_dpo(services, system_maker=refuse, now=NOW)


def test_the_real_maker_renders_the_locked_pre_holdout_card_as_of_the_moment_of_the_reply(
    services: Services, embedder: HashingBackend
) -> None:
    world = build_world(services, embedder, days=20)
    card = world.store.active("pre_holdout")
    assert card is not None
    directory = datasets_dir(services) / "ds-world"
    write_synthetic_dataset(directory, version="ds-world")
    meta_path = directory / FILE_DATASET_META
    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    stored["persona_version"] = card.label
    meta_path.write_text(json.dumps(stored), encoding="utf-8")
    add_pair(
        services,
        "抱抱你",
        "哈哈 辛苦啦",
        persona=f"live:{card.label}",
        system=f"LIVE 系统段 {LIVE_STYLE} {CORRECTION} {FACT_MANUAL}",
    )
    result = export_dpo(services, dataset=directory, now=NOW)
    assert result.pairs == 1 and result.regenerated == 1
    [row] = lines_of(result.dataset.path / FILE_DPO)
    system = row["system"]
    assert PAST_MARKER in system  # the pre-holdout card the adapter was trained with
    for leaked in (LIVE_STYLE, CORRECTION, FACT_MANUAL, "LIVE 系统段"):
        assert (
            leaked not in system
        )  # nothing of the live card: no corrections, no hand-written facts
    assert "【此刻】" in system and row["conversations"][-1]["value"] == "今天好累呀"
    for marker in lf_template.CONTROL_TOKENS:
        assert marker not in system


# ------------------------------------------------------------------------ the command


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        return work(services)
    finally:
        services.close()


def test_the_command_writes_the_file_and_says_what_comes_next(
    data_dir: Path, clock: ManualClock
) -> None:
    def prepare(services: Services) -> None:
        base_dataset(services)
        add_pair(services, persona="live:v9")
        add_pair(services, "快睡吧", "您辛苦了")

    with_services(prepare)
    result = runner.invoke(app, ["train", "export-dpo"])
    assert result.exit_code == 0, result.output
    assert "1 preference pair(s) in dpo_train.jsonl" in result.output
    assert "left out: persona_version_missing 1" in result.output  # no card v9 in the store
    assert "DPO needs at least 200 pairs" in result.output
    assert "next: twin train bundle --profile <profile> --dataset" in result.output
    created = sorted(p.name for p in (data_dir / "training" / "datasets").iterdir())
    assert created[0] == "ds-test-01" and created[1].startswith("ds-test-01-dpo")


def test_the_command_fails_clearly_without_a_dataset_or_pairs(
    data_dir: Path, clock: ManualClock
) -> None:
    result = runner.invoke(app, ["train", "export-dpo"])
    assert result.exit_code != 0 and "twin train export" in result.output
    with_services(lambda services: base_dataset(services))
    result = runner.invoke(app, ["train", "export-dpo"])
    assert result.exit_code != 0 and "no preference pairs" in result.output


def test_the_retrain_check_reports_the_pairs_for_dpo(data_dir: Path, clock: ManualClock) -> None:
    def prepare(services: Services) -> None:
        services.settings.training.dpo_min_pairs = 1

    with_services(prepare)
    plain = runner.invoke(app, ["train", "retrain-check"])
    assert plain.exit_code == 0 and "preference pairs: 0" in plain.output

    def add(services: Services) -> None:
        add_pair(services)

    with_services(add)
    runner.invoke(app, ["--set", "training.dpo_min_pairs=1", "train", "retrain-check"])
    result = runner.invoke(app, ["--set", "training.dpo_min_pairs=1", "train", "retrain-check"])
    assert "preference pairs: 1" in result.output and "可以做 DPO" in result.output


def test_the_sample_schema_is_part_of_the_stored_form() -> None:
    sample = PromptSample("系统", (USER_TURN,), NOW, ("她开头的话",))
    data = sample.to_json()
    assert data["schema"] == SAMPLE_SCHEMA and data["prelude"] == ["她开头的话"]
    assert PromptSample.from_json(data) == sample
    with pytest.raises(PairError):
        PromptSample.from_json({**data, "schema": 99})
    with pytest.raises(PairError):
        PromptSample.from_json({**data, "turns": [{"role": "robot", "content": "x"}]})
    with pytest.raises(PairError):
        PromptSample.from_json({**data, "prelude": "not a list"})
    with pytest.raises(PairError):
        PromptSample.from_json({"schema": SAMPLE_SCHEMA})
