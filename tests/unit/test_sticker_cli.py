"""``twin stickers`` list, show, tag, untag, disable, enable, tag-all (R-STK-002, R-STK-003)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.support.persona import Scenario, sticker_scenario
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.services import Services, build_services
from twin.stickers.catalog import StickerCatalog

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def cli(*args: str) -> Any:
    return runner.invoke(app, list(args))


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        return work(services)
    finally:
        services.close()


@pytest.fixture
def library(data_dir: Path) -> Scenario:
    return with_services(sticker_scenario)


def test_the_list_starts_with_the_statistics_of_the_library(library: Scenario) -> None:
    result = cli("stickers", "list")
    assert result.exit_code == 0, result.output
    assert "4 sticker(s): 4 usable files, 4 used by her, 0 tagged" in result.output
    for md5 in library.md5s:
        assert md5[:8] in result.output
    first = result.output.index(library.md5s[0][:8])
    assert first < result.output.index(library.md5s[1][:8])  # the most used first
    limited = cli("stickers", "list", "--limit", "1")
    assert limited.output.count("available") == 1


def test_tags_are_set_by_hand_and_listed(library: Scenario) -> None:
    md5 = library.md5s[0]
    tagged = cli("stickers", "tag", md5[:10], "开心", "大笑")
    assert tagged.exit_code == 0, tagged.output
    assert f"{md5[:8]}: tags are now 开心、大笑 (by hand)" in tagged.output
    listing = cli("stickers", "list", "--tag", "开心")
    assert md5[:8] in listing.output and library.md5s[1][:8] not in listing.output
    assert "manual" in listing.output and "1 tagged (1 by hand" in listing.output
    untagged = cli("stickers", "list", "--untagged")
    assert md5[:8] not in untagged.output and library.md5s[1][:8] in untagged.output
    removed = cli("stickers", "untag", md5)
    assert removed.exit_code == 0 and "tags are now (none)" in removed.output


def test_a_tag_outside_the_vocabulary_or_an_unknown_sticker_is_refused(library: Scenario) -> None:
    bad = cli("stickers", "tag", library.md5s[0], "超级开心")
    assert bad.exit_code == 2 and "not in the tag vocabulary" in bad.output
    assert "allowed: 开心、大笑" in bad.output
    unknown = cli("stickers", "tag", "ffff", "开心")
    assert unknown.exit_code == 1 and "no sticker MD5 starts with 'ffff'" in unknown.output
    short = cli("stickers", "show", "ab")
    assert short.exit_code == 1 and "at least 4" in short.output
    assert cli("stickers", "disable", "eeee").exit_code == 1
    assert with_services(lambda s: StickerCatalog(s).counts()["tagged"]) == 0


def test_show_lists_everything_known_about_a_sticker(library: Scenario) -> None:
    md5 = library.md5s[0]

    def prepare(services: Services) -> None:
        catalog = StickerCatalog(services)
        catalog.save_vision(
            md5, ["开心", "晚安"], "一只笑着的猫", "回应好消息", at=services.clock.now_utc()
        )
        catalog.save_context(
            md5,
            ["委屈"],
            "她用它表示有点委屈",
            uses=4,
            cutoff=services.clock.now_utc(),
            at=services.clock.now_utc(),
        )

    with_services(prepare)
    shown = cli("stickers", "show", md5[:6])
    assert shown.exit_code == 0, shown.output
    for fragment in (
        f"md5          {md5}",
        "available, image/png, 64x64; origin import",
        "her 5, user 0",
        "tags         委屈 (decided by: context)",
        "picture    开心、晚安",
        "her use    委屈 (4 uses before the cutoff)",
        "by hand    -",
        "description  一只笑着的猫",
        "use cases    回应好消息",
        "her meaning  她用它表示有点委屈",
        "vector       no",
    ):
        assert fragment in shown.output, fragment


def test_a_sticker_can_be_switched_off_and_on(library: Scenario) -> None:
    md5 = library.md5s[0]
    off = cli("stickers", "disable", md5)
    assert off.exit_code == 0 and "disabled" in off.output
    assert (
        "(off)" in cli("stickers", "list").output and "1 disabled" in cli("stickers", "list").output
    )
    assert "switched off" in cli("stickers", "show", md5).output
    on = cli("stickers", "enable", md5)
    assert on.exit_code == 0 and "enabled" in on.output
    assert "(off)" not in cli("stickers", "list").output


def test_tag_all_prices_the_work_and_waits_for_the_approval(library: Scenario) -> None:
    result = cli("stickers", "tag-all")
    assert result.exit_code == 0, result.output
    assert "4 sticker(s) to tag (2 also to be corrected from her use) in 1 job(s)" in result.output
    assert (
        "upper bound" in result.output
        and "approve with: twin jobs approve stickers-" in result.output
    )
    jobs = cli("jobs", "list", "--type", "sticker_tag")
    assert jobs.exit_code == 0 and "sticker_tag" in jobs.output
    again = cli("stickers", "tag-all")
    assert (
        again.exit_code == 0
        and "every sticker is tagged and described (4 already queued)" in again.output
    )


def test_tag_all_on_an_empty_library_has_nothing_to_do(data_dir: Path) -> None:
    result = cli("stickers", "tag-all")
    assert result.exit_code == 0 and "every sticker is tagged and described" in result.output
    listing = cli("stickers", "list")
    assert listing.exit_code == 0 and "0 sticker(s)" in listing.output
