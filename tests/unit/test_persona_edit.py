"""Editing the hand-written part of the card (R-PERS-003)."""

from __future__ import annotations

import stat
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from twin.profile.persona import compose
from twin.profile.persona.edit import (
    CONFIRM_PROMPT,
    ConsoleEditor,
    EditOutcome,
    changed_sections,
    edit_manual,
    make_editor,
    shred_file,
    shred_siblings,
)
from twin.profile.persona.refresh import refresh_all, write_description
from twin.profile.persona.sections import AUTO, DONT, MANUAL, STATS, split_card
from twin.profile.persona.store import PersonaStore
from twin.services import Services

SENTINEL = "宝宝只在这里出现一次"


@pytest.fixture
def card(services: Services) -> Services:
    store = PersonaStore(services.db, services.clock)
    text = (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block("### 风格\n- 口头禅：哈哈\n\n### 基本情况\n- 事实：养猫\n\n")
        + compose.manual_block(["- 称呼：旧称呼"], ["- 生日在秋天"])
        + compose.dont_block(["不要用句号"])
    )
    store.add_version("live", text, reason="generate", described_her_messages=100)
    store.add_version(
        "pre_holdout",
        compose.stats_block(["几乎不用逗号"])
        + compose.empty_auto_block()
        + compose.manual_block((), None),
        reason="stats",
    )
    return services


def store_of(services: Services) -> PersonaStore:
    return PersonaStore(services.db, services.clock)


def replacing(old: str, new: str) -> Callable[[Path], None]:
    def launcher(path: Path) -> None:
        text = path.read_bytes().decode("utf-8")
        assert old in text
        path.write_bytes(text.replace(old, new).encode("utf-8"))

    return launcher


def leftovers(services: Services) -> list[str]:
    directory = services.paths.tmp_dir
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def test_a_change_in_the_manual_section_becomes_a_new_version_and_nothing_is_left_behind(
    card: Services,
) -> None:
    seen: list[Path] = []

    def launcher(path: Path) -> None:
        seen.append(path)
        text = path.read_text(encoding="utf-8")
        assert "旧称呼" in text and "几乎不用逗号" in text  # the decrypted card
        assert path.parent == card.paths.tmp_dir  # a controlled place below the data directory
        path.write_text(text.replace("旧称呼", SENTINEL), encoding="utf-8")

    before = store_of(card).active("live")
    assert before is not None
    outcome = edit_manual(card, launcher)
    assert outcome.status == "saved" and outcome.card is not None and outcome.card.number == 2
    assert "saved as version v2" in outcome.message
    after = store_of(card).active("live")
    assert after is not None and after.reason == "edit" and after.parent_id == before.id
    assert SENTINEL in after.sections.body(MANUAL) and "旧称呼" not in after.text
    for name in (STATS, AUTO, DONT):
        assert after.sections.block(name) == before.sections.block(name)
    assert after.described_her_messages == 100  # the description's bookkeeping is carried over
    assert leftovers(card) == [] and seen and not seen[0].exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX permission bits; Windows uses the ACL of the data folder"
)
def test_the_temporary_file_is_readable_by_its_owner_only(card: Services) -> None:
    modes: list[int] = []

    def launcher(path: Path) -> None:
        modes.append(stat.S_IMODE(path.stat().st_mode))

    edit_manual(card, launcher)
    assert modes == [0o600]


def test_the_style_lines_are_copied_to_the_past_card(card: Services) -> None:
    edit_manual(card, replacing("旧称呼", SENTINEL))
    past = store_of(card).active("pre_holdout")
    assert past is not None and past.reason == "manual_sync"
    assert SENTINEL in past.sections.body(MANUAL) and "生日" not in past.text


def test_a_change_in_any_other_section_is_refused_and_nothing_is_saved(card: Services) -> None:
    for old, new, section in (
        ("几乎不用逗号", "总是用逗号", STATS),
        ("口头禅：哈哈", "口头禅：嘿嘿", AUTO),
        ("不要用句号", "尽量用句号", DONT),
    ):
        outcome = edit_manual(card, replacing(old, new))
        assert outcome.status == "rejected" and section in outcome.message, section
        assert "nothing was saved" in outcome.message
    assert len(store_of(card).history("live")) == 1 and leftovers(card) == []


def test_a_change_in_the_manual_and_another_section_together_is_refused(card: Services) -> None:
    def launcher(path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("旧称呼", SENTINEL).replace("养猫", "养狗"), encoding="utf-8")

    outcome = edit_manual(card, launcher)
    assert outcome.status == "rejected" and AUTO in outcome.message
    assert len(store_of(card).history("live")) == 1


def test_an_editor_that_changes_every_line_ending_changes_nothing(card: Services) -> None:
    def launcher(path: Path) -> None:
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    outcome = edit_manual(card, launcher)
    assert outcome.status == "unchanged" and outcome.card is not None
    assert len(store_of(card).history("live")) == 1


def test_a_real_edit_by_such_an_editor_keeps_the_other_sections_as_stored(card: Services) -> None:
    def launcher(path: Path) -> None:
        text = path.read_bytes().decode("utf-8").replace("旧称呼", SENTINEL)
        path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    before = store_of(card).active("live")
    assert before is not None
    outcome = edit_manual(card, launcher)
    assert outcome.status == "saved"
    after = store_of(card).active("live")
    assert after is not None and SENTINEL in after.sections.body(MANUAL)
    for name in (STATS, AUTO, DONT):
        assert after.sections.block(name) == before.sections.block(name)
        assert "\r" not in (after.sections.block(name) or "")


def test_a_byte_order_mark_is_ignored(card: Services) -> None:
    def launcher(path: Path) -> None:
        path.write_bytes(
            b"\xef\xbb\xbf" + path.read_bytes().replace("旧称呼".encode(), SENTINEL.encode())
        )

    outcome = edit_manual(card, launcher)
    assert outcome.status == "saved"
    active = store_of(card).active("live")
    assert active is not None and not active.text.startswith("\ufeff")


@pytest.mark.parametrize(
    "mangle",
    [
        lambda text: "随手写的一行\n" + text,
        lambda text: text.replace("## [手动]\n", ""),
        lambda text: text.replace("## [不要这样]", "## [别的分区]"),
        lambda text: "完全不是人设卡",
        lambda text: text + "\n## [手动]\n又一个手动分区\n",
    ],
)
def test_a_file_that_is_no_longer_a_card_is_refused(
    card: Services, mangle: Callable[[str], str]
) -> None:
    def launcher(path: Path) -> None:
        path.write_text(mangle(path.read_text(encoding="utf-8")), encoding="utf-8")

    outcome = edit_manual(card, launcher)
    assert outcome.status == "rejected" and len(store_of(card).history("live")) == 1


def test_a_hand_edit_made_meanwhile_is_not_overwritten(card: Services) -> None:
    store = store_of(card)

    def launcher(path: Path) -> None:
        current = store.active("live")
        assert current is not None
        store.add_version(
            "live",
            current.sections.with_block(
                MANUAL, compose.manual_block(["- 称呼：别处改的"], [])
            ).assemble(),
            reason="edit",
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace("旧称呼", SENTINEL), encoding="utf-8"
        )

    outcome = edit_manual(card, launcher)
    assert outcome.status == "rejected" and "meanwhile" in outcome.message
    active = store.active("live")
    assert active is not None and "别处改的" in active.text and SENTINEL not in active.text


def test_the_new_description_that_arrives_meanwhile_is_kept(card: Services) -> None:
    def launcher(path: Path) -> None:
        write_description(
            card, "live", "### 风格\n- 口头禅：新生成\n\n", provenance={}, template_version="t@1",
            her_messages=200, at=card.clock.now_utc(),
        )  # fmt: skip
        path.write_text(
            path.read_text(encoding="utf-8").replace("旧称呼", SENTINEL), encoding="utf-8"
        )

    outcome = edit_manual(card, launcher)
    assert outcome.status == "saved" and outcome.card is not None
    assert "新生成" in outcome.card.text and SENTINEL in outcome.card.text
    assert "养猫" not in outcome.card.text  # the newer description, not the one that was opened


def test_without_a_card_there_is_nothing_to_edit(services: Services) -> None:
    outcome = edit_manual(services, lambda path: pytest.fail("the editor must not start"))
    assert outcome.status == "rejected" and "persona regenerate" in outcome.message


def test_the_temporary_file_is_removed_even_when_the_editor_fails(card: Services) -> None:
    def launcher(path: Path) -> None:
        raise RuntimeError("editor crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        edit_manual(card, launcher)
    assert leftovers(card) == []


def test_swap_and_backup_files_of_the_editor_are_removed_too(card: Services) -> None:
    def launcher(path: Path) -> None:
        for name in (f".{path.name}.swp", f"{path.name}~", f"{path.stem}.bak.md"):
            (path.parent / name).write_text("copy of the card", encoding="utf-8")
        (path.parent / "unrelated.txt").write_text("keep", encoding="utf-8")

    edit_manual(card, launcher)
    assert leftovers(card) == ["unrelated.txt"]


def test_sections_are_compared_without_their_line_endings() -> None:
    old = split_card("## [手动]\n### 风格\n- a\n\n## [不要这样]\n- b\n")
    new = split_card("## [手动]\r\n### 风格\r\n- changed\r\n\r\n## [不要这样]\r\n- b\r\n")
    assert changed_sections(old, new) == [MANUAL]


# ------------------------------------------------------------------- shredding


def test_a_file_is_overwritten_before_it_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "card.md"
    path.write_bytes("宝宝".encode() * 50)
    size = path.stat().st_size
    original_unlink = Path.unlink
    contents: list[bytes] = []

    def spying_unlink(self: Path, missing_ok: bool = False) -> None:
        contents.append(self.read_bytes())
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", spying_unlink)
    assert shred_file(path) is True and not path.exists()
    assert contents == [bytes(size)]  # zeros, not the card


def test_a_file_that_cannot_be_deleted_is_still_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "card.md"
    path.write_bytes(b"secret" * 10)

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("the editor still holds the file")

    monkeypatch.setattr(Path, "unlink", refuse)
    assert shred_file(path) is False
    assert path.read_bytes() == bytes(60)
    assert shred_file(tmp_path / "missing.md") is True


def test_siblings_with_the_stem_are_shredded_and_others_stay(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "persona-1.md.swp").write_text("x", encoding="utf-8")
    (work / "persona-1.md").write_text("x", encoding="utf-8")
    (work / "persona-2.md").write_text("x", encoding="utf-8")
    (work / "sub").mkdir()
    shred_siblings(work, "persona-1")
    assert sorted(p.name for p in work.iterdir()) == ["persona-2.md", "sub"]


# ------------------------------------------------------------ starting an editor


class Calls:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.commands: list[list[str]] = []
        self.prompts: list[str] = []

    def startfile(self, path: str) -> None:
        self.started.append(path)

    def run(self, command: Sequence[str]) -> int:
        self.commands.append(list(command))
        return 0

    def confirm(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return ""


def editor(
    calls: Calls, platform: str, env: dict[str, str], *, with_startfile: bool = True
) -> ConsoleEditor:
    return ConsoleEditor(
        platform=platform,
        env=env,
        startfile=calls.startfile if with_startfile else None,
        run=calls.run,
        confirm=calls.confirm,
    )


def test_on_windows_the_default_program_is_opened_and_the_person_confirms(tmp_path: Path) -> None:
    calls = Calls()
    editor(calls, "win32", {"EDITOR": "ignored"})(tmp_path / "a.md")
    assert calls.started == [str(tmp_path / "a.md")] and calls.commands == []
    assert calls.prompts == [CONFIRM_PROMPT]


def test_windows_without_startfile_is_an_error(tmp_path: Path) -> None:
    calls = Calls()
    with pytest.raises(OSError, match="startfile"):
        editor(calls, "win32", {}, with_startfile=False)(tmp_path / "a.md")


def test_elsewhere_the_editor_variable_is_run_and_waited_for(tmp_path: Path) -> None:
    calls = Calls()
    editor(calls, "linux", {"EDITOR": "nano", "VISUAL": "code --wait"})(tmp_path / "a.md")
    assert calls.commands == [["code", "--wait", str(tmp_path / "a.md")]]
    assert calls.prompts == [] and calls.started == []
    calls = Calls()
    editor(calls, "darwin", {"EDITOR": "vim -u NONE"})(tmp_path / "b.md")
    assert calls.commands == [["vim", "-u", "NONE", str(tmp_path / "b.md")]]


def test_without_an_editor_variable_the_system_opener_is_used(tmp_path: Path) -> None:
    calls = Calls()
    editor(calls, "linux", {})(tmp_path / "a.md")
    assert calls.commands == [["xdg-open", str(tmp_path / "a.md")]]
    assert calls.prompts == [CONFIRM_PROMPT]


def test_the_command_uses_the_console_editor_by_default() -> None:
    assert isinstance(make_editor(), ConsoleEditor)


def test_outcomes_are_plain_values() -> None:
    assert EditOutcome("unchanged", "nothing").card is None


def test_the_cards_of_a_refresh_can_be_edited_afterwards(services: Services) -> None:
    from tests.support.synth_chat import ChatSpec, build_chat
    from twin.profile.builder import rebuild

    build_chat(services, ChatSpec(days=20))
    rebuild(services, "all")
    refresh_all(services)
    outcome = edit_manual(services, replacing("### 风格\n", f"### 风格\n- 称呼：{SENTINEL}\n"))
    assert outcome.status == "saved"
    past = store_of(services).active("pre_holdout")
    assert past is not None and SENTINEL in past.text


@pytest.mark.windows
def test_on_windows_the_default_editor_is_the_real_startfile() -> None:
    import os

    assert make_editor().startfile is os.startfile  # type: ignore[attr-defined]


def test_the_default_runner_waits_for_the_program(tmp_path: Path) -> None:
    from twin.profile.persona.edit import default_run

    assert default_run([sys.executable, "-c", "raise SystemExit(0)"]) == 0
    assert default_run([sys.executable, "-c", "raise SystemExit(3)"]) == 3
