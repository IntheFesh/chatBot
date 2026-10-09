"""Editing the hand-written part of the card (R-PERS-003).

``twin persona edit`` decrypts the card in force into a controlled temporary file (below the
data directory, readable by the owner only), opens it in the system's editor, waits until the
person says they are done, and reads it back.  Only the ``[手动]`` section may change: the
other three sections are compared with the stored card and, if any of them differs, the edit is
refused with the name of the section.  An accepted edit becomes a new version; the other
sections are taken from the stored card, never from the edited file, so an editor that rewrites
line endings cannot alter them.  The temporary file - and the swap and backup files an editor
leaves next to it - are overwritten before they are deleted.

How the editor is started depends on the system (R-PERS-003): on Windows the default program of
a ``.md`` file is opened with ``os.startfile`` (which returns at once, so the person confirms in
the terminal with Enter); elsewhere ``$VISUAL`` / ``$EDITOR`` is run and waited for, else
``xdg-open`` is used and the person confirms with Enter.  Starting the editor and reading the
confirmation are both injectable, which is how the tests drive a session.
"""

from __future__ import annotations

import os
import secrets
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from twin.ops.logging import get_logger
from twin.profile.persona.refresh import persona_store, sync_manual_style
from twin.profile.persona.sections import (
    MANUAL,
    SECTION_NAMES,
    CardFormatError,
    CardText,
    split_card,
)
from twin.profile.persona.store import PersonaCardView
from twin.services import Services

log = get_logger("twin.profile.persona")

Launcher = Callable[[Path], None]
CONFIRM_PROMPT = "在编辑器里改好并保存、关闭文件后，按回车确认（Ctrl+C 放弃）："


def default_run(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode  # noqa: S603 - the user's editor


@dataclass
class ConsoleEditor:
    """Opens a file in the system editor and returns when the person is done with it."""

    platform: str = sys.platform
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    startfile: Callable[[str], None] | None = field(
        default_factory=lambda: getattr(os, "startfile", None)
    )
    run: Callable[[Sequence[str]], int] = default_run
    confirm: Callable[[str], str] = input

    def __call__(self, path: Path) -> None:
        if self.platform == "win32":
            if self.startfile is None:
                raise OSError("os.startfile is not available on this system")
            self.startfile(str(path))
            self.confirm(CONFIRM_PROMPT)
            return
        editor = self.env.get("VISUAL") or self.env.get("EDITOR")
        if editor:
            self.run([*shlex.split(editor), str(path)])  # returns when the editor is closed
            return
        self.run(["xdg-open", str(path)])  # returns at once
        self.confirm(CONFIRM_PROMPT)


def make_editor() -> Launcher:
    """The launcher used by the command (tests replace this function)."""
    return ConsoleEditor()


# ------------------------------------------------------------------ the temp file


def shred_file(path: Path) -> bool:
    """Overwrite a file with random bytes and zeros, then delete it; ``False`` if it is left."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return True
    try:
        with path.open("r+b") as handle:
            for chunk in (secrets.token_bytes(size), bytes(size)):
                handle.seek(0)
                handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        path.unlink()
    except OSError:
        log.warning("temporary_file_not_removed", name=path.name)
        return False
    return True


def shred_siblings(directory: Path, stem: str) -> None:
    """Remove the swap, backup and lock files an editor made for ``stem``."""
    for other in directory.iterdir():
        if stem in other.name and other.is_file():
            shred_file(other)


def _write_private(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


# ------------------------------------------------------------------ checking


def _same(first: str | None, second: str | None) -> bool:
    def normal(text: str | None) -> str:
        return (text or "").replace("\r\n", "\n").rstrip("\n")

    return normal(first) == normal(second)


def changed_sections(old: CardText, new: CardText) -> list[str]:
    """The sections whose text differs (line endings and trailing blank lines do not count)."""
    return [name for name in SECTION_NAMES if not _same(old.block(name), new.block(name))]


@dataclass(frozen=True)
class EditOutcome:
    status: Literal["saved", "unchanged", "rejected"]
    message: str
    card: PersonaCardView | None = None


def edit_manual(services: Services, launcher: Launcher) -> EditOutcome:
    """Run one editing session of the live card (see the module description)."""
    store = persona_store(services)
    card = store.active("live")
    if card is None:
        return EditOutcome(
            "rejected", "there is no persona card yet; run `twin persona regenerate --stats-only`"
        )
    directory = services.paths.tmp_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"persona-{card.id}"
    path = directory / f"{stem}.md"
    _write_private(path, card.text.encode("utf-8"))
    try:
        launcher(path)
        edited = path.read_bytes().decode("utf-8-sig")
    finally:
        shred_file(path)
        shred_siblings(directory, stem)
    try:
        new = split_card(edited)
    except CardFormatError as exc:
        return EditOutcome("rejected", f"the file is not a persona card any more: {exc}")
    old = card.sections
    if new.preamble.strip():
        return EditOutcome("rejected", "text was added before the first section heading")
    touched = [name for name in changed_sections(old, new) if name != MANUAL]
    if touched:
        return EditOutcome(
            "rejected",
            f"only the {MANUAL} section may be edited, but {', '.join(touched)} changed; "
            "nothing was saved",
        )
    if new.block(MANUAL) is None:
        return EditOutcome("rejected", f"the {MANUAL} section heading was removed; nothing saved")
    if _same(old.block(MANUAL), new.block(MANUAL)):
        return EditOutcome("unchanged", "nothing changed", card)
    latest = store.active("live")
    if latest is None or not _same(latest.sections.block(MANUAL), old.block(MANUAL)):
        return EditOutcome(
            "rejected", f"the {MANUAL} section was changed by something else meanwhile; try again"
        )
    block = new.block(MANUAL) or ""
    if not block.endswith("\n"):
        block += "\n"
    created = store.add_version(
        "live",
        latest.sections.with_block(MANUAL, block).assemble(),
        reason="edit",
        profile_version_id=latest.profile_version_id,
        template_version=latest.template_version,
        described_her_messages=latest.described_her_messages,
        described_at=latest.described_at,
        provenance=store.provenance(latest.id),
    )
    sync_manual_style(services)
    return EditOutcome("saved", f"saved as version v{created.number}", created)
