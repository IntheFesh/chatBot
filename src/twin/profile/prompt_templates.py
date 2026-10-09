"""Versioned prompt templates (R-PERS-005, R-OPS-010).

A template is a file in ``twin/profile/templates`` named ``<name>.v<version>.md``: the system
message, a line ``=== user ===``, then the user message.  Fields are written ``$field`` (Python
:class:`string.Template`), so the braces of the JSON examples inside a prompt need no escaping.

The files are the source; :class:`TemplateStore` copies every file it has not seen into the
``prompt_templates`` table and records the version in force in the ``settings`` table
(``template.active.<name>``).  A newer file version becomes the active one when it is first
loaded; ``twin rollback`` (round 12) points the setting back at an older row, which stays until
a still newer file appears.  A file whose text differs from the stored row of the same version
is refused: change a template by adding the next version, so that every card and every training
set can name the exact text it was made with.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from twin.clock import Clock
from twin.llm.types import ChatMessage
from twin.storage.db import Database
from twin.storage.persona_models import PromptTemplate
from twin.storage.settings_store import get_setting, put_setting

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
USER_MARKER = "=== user ==="
ACTIVE_KEY = "template.active."
_FILE = re.compile(r"^(?P<name>[a-z][a-z_]*)\.v(?P<version>[1-9]\d*)\.md$")

PERSONA_MAP = "persona_map"
PERSONA_REDUCE = "persona_reduce"
STICKER_TAG = "sticker_tag"
STICKER_CONTEXT = "sticker_context"
MEMORY_EXTRACT = "memory_extract"
MEMORY_EXTRACT_BOT = "memory_extract_bot"
MEMORY_CONFLICT = "memory_conflict"
MEMORY_SUMMARY = "memory_summary"
MEMORY_SUMMARY_MERGE = "memory_summary_merge"
LIFELINE_GENERATE = "lifeline_generate"
LIFELINE_CHECK = "lifeline_check"


class TemplateError(RuntimeError):
    """A template file is malformed, changed in place, or missing."""


@dataclass(frozen=True)
class PromptText:
    """A template: its identity and the text of its two messages."""

    name: str
    version: int
    system: str
    user: str
    sha256: str

    @property
    def ref(self) -> str:
        """``name@version``: how a card or a training set names the template it used."""
        return f"{self.name}@{self.version}"

    def render(self, **fields: object) -> list[ChatMessage]:
        """The system and user messages with ``$field`` replaced (a missing field is an error)."""
        values = {key: str(value) for key, value in fields.items()}
        try:
            system = Template(self.system).substitute(values)
            user = Template(self.user).substitute(values)
        except KeyError as exc:
            raise TemplateError(f"{self.ref} needs the field {exc.args[0]!r}") from exc
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def split_template(name: str, version: int, content: str) -> PromptText:
    """Split the text of a template file into its system and user parts."""
    parts = content.split(f"\n{USER_MARKER}\n", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise TemplateError(
            f"template {name}.v{version} needs a system part, a line {USER_MARKER!r} "
            "and a user part"
        )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return PromptText(name, version, parts[0].strip(), parts[1].strip(), digest)


def template_files(directory: Path = TEMPLATE_DIR) -> dict[str, dict[int, Path]]:
    """The template files by name and version."""
    found: dict[str, dict[int, Path]] = {}
    for path in sorted(directory.glob("*.md")):
        match = _FILE.match(path.name)
        if match:
            found.setdefault(match["name"], {})[int(match["version"])] = path
    return found


def newest_file_template(name: str, directory: Path = TEMPLATE_DIR) -> PromptText:
    """The newest file version of a template, read from the file alone (no database access).

    A READ command that has to size a prompt (the cost estimate of ``twin memory replay
    estimate``) cannot let :class:`TemplateStore` load the files into the table first: that is a
    write.  The newest file is the version that would be in force after the first load.
    """
    versions = template_files(directory).get(name)
    if not versions:
        raise TemplateError(f"there is no template file named {name!r}")
    version = max(versions)
    return split_template(name, version, versions[version].read_text(encoding="utf-8"))


class TemplateStore:
    """The templates in the database and which version of each is in force."""

    def __init__(self, db: Database, clock: Clock, directory: Path = TEMPLATE_DIR) -> None:
        self._db = db
        self._clock = clock
        self._directory = directory
        self._synced = False

    def sync(self, *, force: bool = False) -> None:
        """Load the files that are not in the table yet and move the active versions forward."""
        if self._synced and not force:
            return
        self._synced = True
        for name, versions in template_files(self._directory).items():
            newest = max(versions)
            for version, path in sorted(versions.items()):
                content = path.read_text(encoding="utf-8")
                split_template(name, version, content)  # refuses a malformed file
                self._store(name, version, content, advance=version == newest)

    def _store(self, name: str, version: int, content: str, *, advance: bool) -> None:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        now = self._clock.now_utc()
        for _ in range(2):  # the second round only happens when another process won the race
            try:
                with self._db.transaction(bump_state=False) as session:
                    row = session.scalars(
                        select(PromptTemplate).where(
                            PromptTemplate.name == name, PromptTemplate.version == version
                        )
                    ).first()
                    if row is None:
                        session.add(
                            PromptTemplate(
                                name=name,
                                version=version,
                                content=content,
                                content_sha256=digest,
                                source="file",
                                created_at=now,
                                updated_at=now,
                            )
                        )
                        if advance:
                            put_setting(
                                session,
                                ACTIVE_KEY + name,
                                version,
                                clock=self._clock,
                                by="template_sync",
                            )
                        return
                    if row.content_sha256 != digest:
                        raise TemplateError(
                            f"the file of template {name}.v{version} differs from the stored "
                            "version; add a new version file instead of editing this one"
                        )
                    return
            except IntegrityError:
                continue

    def active_version(self, name: str) -> int:
        with self._db.session() as session:
            value = get_setting(session, ACTIVE_KEY + name)
        if not isinstance(value, int):
            raise TemplateError(f"no active version of template {name!r}; is the file missing?")
        return value

    def get(self, name: str, version: int) -> PromptText:
        with self._db.session() as session:
            row = session.scalars(
                select(PromptTemplate).where(
                    PromptTemplate.name == name, PromptTemplate.version == version
                )
            ).first()
            if row is None:
                raise TemplateError(f"there is no template {name}.v{version}")
            content = row.content
        return split_template(name, version, content)

    def active(self, name: str) -> PromptText:
        """The template in force (loading the files first if the table does not know them)."""
        self.sync()
        return self.get(name, self.active_version(name))

    def versions(self, name: str) -> list[int]:
        with self._db.session() as session:
            return sorted(
                session.scalars(select(PromptTemplate.version).where(PromptTemplate.name == name))
            )

    def names(self) -> list[str]:
        with self._db.session() as session:
            return sorted(set(session.scalars(select(PromptTemplate.name))))

    def rollback(self, name: str, version: int) -> None:
        """Make an older (stored) version the one in force."""
        self.get(name, version)
        with self._db.transaction(bump_state=True) as session:
            put_setting(session, ACTIVE_KEY + name, version, clock=self._clock, by="rollback")
