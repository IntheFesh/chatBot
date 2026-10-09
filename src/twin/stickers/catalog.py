"""The sticker library as the rest of the program sees it (R-STK-002, R-STK-003).

:class:`StickerRecord` is a plain copy of one ``stickers`` row (decrypted while the session is
open); :class:`StickerCatalog` reads the library and makes the changes that belong to tagging:
saving the visual tags, saving the context correction, setting or clearing the hand tags,
disabling a sticker.  After every change the tags a sticker is *selected by* are recomputed
(:func:`twin.stickers.tags.effective_tags`): manual tags first, then the context evidence merged
with the picture, then the picture alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import bindparam, func, select, text
from sqlalchemy.orm import Session

from twin.services import Services
from twin.stickers.tags import TagVocabulary, effective_tags, load_vocabulary
from twin.storage.chat_models import Sticker, StickerUse

ORIGIN_IMPORT = "import"
ORIGIN_INCOMING = "incoming"


class UnknownStickerError(LookupError):
    """No sticker with that MD5 is in the library."""


@dataclass(frozen=True)
class StickerRecord:
    """One sticker of the library."""

    md5: str
    status: str
    mime: str | None
    width: int | None
    height: int | None
    her_uses: int
    user_uses: int
    first_used_at: datetime | None
    last_used_at: datetime | None
    tags: tuple[str, ...]
    vision_tags: tuple[str, ...]
    context_tags: tuple[str, ...]
    manual_tags: tuple[str, ...]
    tag_source: str | None
    description: str | None
    use_cases: str | None
    context_note: str | None
    origin: str
    disabled: bool
    tagged_at: datetime | None
    context_tagged_at: datetime | None
    context_cutoff_at: datetime | None
    context_uses: int
    desc_vector_id: str | None
    desc_encoding: str | None
    sha256: str | None

    @property
    def available(self) -> bool:
        """The file is stored and is the picture the export named (it can be sent)."""
        return self.status == "available"

    @property
    def tagged(self) -> bool:
        return bool(self.tags)

    @property
    def usable(self) -> bool:
        """May the selector choose it: available, tagged by someone, not switched off."""
        return self.available and self.tagged and not self.disabled


def record_of(row: Sticker) -> StickerRecord:
    """Copy a ``stickers`` row (call while its session is open)."""
    return StickerRecord(
        md5=row.md5,
        status=row.status,
        mime=row.mime,
        width=row.width,
        height=row.height,
        her_uses=row.her_uses,
        user_uses=row.user_uses,
        first_used_at=row.first_used_at,
        last_used_at=row.last_used_at,
        tags=tuple(row.tags or ()),
        vision_tags=tuple(row.vision_tags or ()),
        context_tags=tuple(row.context_tags or ()),
        manual_tags=tuple(row.manual_tags or ()),
        tag_source=row.tag_source,
        description=row.description,
        use_cases=row.use_cases,
        context_note=row.context_note,
        origin=row.origin,
        disabled=row.disabled,
        tagged_at=row.tagged_at,
        context_tagged_at=row.context_tagged_at,
        context_cutoff_at=row.context_cutoff_at,
        context_uses=row.context_uses,
        desc_vector_id=row.desc_vector_id,
        desc_encoding=row.desc_encoding,
        sha256=row.sha256,
    )


def refresh_effective(row: Sticker, vocabulary: TagVocabulary) -> None:
    """Recompute ``tags`` and ``tag_source`` of a row from its three sources.

    The description vector includes the tags, so a change of tags makes it stale.
    """
    tags, source = effective_tags(row.vision_tags, row.context_tags, row.manual_tags, vocabulary)
    if (tags or None) != (list(row.tags) if row.tags else None):
        row.desc_encoding = None
    row.tags = tags or None
    row.tag_source = source


class StickerCatalog:
    """Reads and tags the library."""

    def __init__(self, services: Services, vocabulary: TagVocabulary | None = None) -> None:
        self._services = services
        self._vocabulary = vocabulary

    @property
    def vocabulary(self) -> TagVocabulary:
        if self._vocabulary is None:
            self._vocabulary = load_vocabulary(self._services.settings, self._services.paths.root)
        return self._vocabulary

    # ------------------------------------------------------------------ reading

    def get(self, md5: str) -> StickerRecord | None:
        with self._services.db.session() as session:
            row = session.get(Sticker, md5)
            return record_of(row) if row is not None else None

    def require(self, md5: str) -> StickerRecord:
        found = self.get(md5)
        if found is None:
            raise UnknownStickerError(f"no sticker with MD5 {md5} in the library")
        return found

    def resolve(self, prefix: str) -> StickerRecord:
        """A sticker from its MD5 or a unique prefix of at least 4 characters."""
        text = prefix.strip().lower()
        if len(text) == 32:
            return self.require(text)
        if len(text) < 4:
            raise UnknownStickerError("give at least 4 characters of the MD5")
        with self._services.db.session() as session:
            rows = list(
                session.scalars(select(Sticker).where(Sticker.md5.like(text + "%")).limit(2))
            )
            if not rows:
                raise UnknownStickerError(f"no sticker MD5 starts with {text!r}")
            if len(rows) > 1:
                raise UnknownStickerError(f"{text!r} matches more than one sticker; use more")
            return record_of(rows[0])

    def records(
        self,
        *,
        status: str | None = None,
        tagged: bool | None = None,
        her_only: bool = False,
        limit: int | None = None,
    ) -> list[StickerRecord]:
        """Stickers ordered by her use count (most used first), then MD5."""
        stmt = select(Sticker).order_by(Sticker.her_uses.desc(), Sticker.md5)
        if status is not None:
            stmt = stmt.where(Sticker.status == status)
        if her_only:
            stmt = stmt.where(Sticker.her_uses > 0)
        if tagged is True:
            stmt = stmt.where(Sticker.tags_ct.is_not(None))
        elif tagged is False:
            stmt = stmt.where(Sticker.tags_ct.is_(None))
        if limit is not None:
            stmt = stmt.limit(limit)
        with self._services.db.session() as session:
            return [record_of(row) for row in session.scalars(stmt)]

    def use_windows(self, md5: str) -> dict[str, str]:
        """Where she used a sticker: ``{message id: id of the example window}`` (R-STK-002).

        The window is the one of the retrieval library (round 05) whose reply block holds the
        message, i.e. the conversation context in which she sent the sticker.  A use that is
        in no window (the library is not built yet, or the block has nothing the bot could
        have written) is left out.
        """
        with self._services.db.session() as session:
            ids = list(
                session.scalars(
                    select(StickerUse.message_id).where(
                        StickerUse.sticker_md5 == md5, StickerUse.by_her.is_(True)
                    )
                )
            )
            if not ids:
                return {}
            statement = text(
                "SELECT j.value, w.id FROM example_windows AS w, "
                "json_each(w.reply_block_ids) AS j WHERE j.value IN :ids"
            ).bindparams(bindparam("ids", expanding=True))
            return {str(row[0]): str(row[1]) for row in session.execute(statement, {"ids": ids})}

    def tag_lookup(self) -> Callable[[str], str | None]:
        """``md5 -> first tag`` for the stickers that have tags (loaded once, then cached)."""
        table = {record.md5: record.tags[0] for record in self.records(tagged=True) if record.tags}
        return table.get

    def counts(self) -> dict[str, int]:
        """Library statistics: total, available, tagged, with a description, with a vector."""
        with self._services.db.session() as session:

            def number(*conditions: object) -> int:
                stmt = select(func.count()).select_from(Sticker)
                for condition in conditions:
                    stmt = stmt.where(condition)  # type: ignore[arg-type]
                return int(session.scalar(stmt) or 0)

            return {
                "total": number(),
                "available": number(Sticker.status == "available"),
                "hers": number(Sticker.her_uses > 0),
                "hers_available": number(Sticker.her_uses > 0, Sticker.status == "available"),
                "tagged": number(Sticker.tags_ct.is_not(None)),
                "described": number(Sticker.description_ct.is_not(None)),
                "context_corrected": number(Sticker.context_tagged_at.is_not(None)),
                "manual": number(Sticker.manual_tags_ct.is_not(None)),
                "disabled": number(Sticker.disabled.is_(True)),
                "with_vector": number(Sticker.desc_vector_id.is_not(None)),
            }

    # ------------------------------------------------------------------ changes

    def save_vision(
        self,
        md5: str,
        tags: Sequence[str],
        description: str,
        use_cases: str,
        *,
        at: datetime,
    ) -> StickerRecord:
        """Store what the picture shows (tags are checked against the vocabulary)."""
        checked = self.vocabulary.check(tags)
        with self._services.db.transaction(bump_state=False) as session:
            row = self._row(session, md5)
            row.vision_tags = checked
            row.description = description
            row.use_cases = use_cases
            row.tagged_at = at
            row.desc_encoding = None  # the description changed: the vector must be made again
            refresh_effective(row, self.vocabulary)
            session.flush()
            return record_of(row)

    def save_context(
        self,
        md5: str,
        tags: Sequence[str],
        meaning: str,
        *,
        uses: int,
        cutoff: datetime,
        at: datetime,
    ) -> StickerRecord:
        """Store what her use of the sticker (before the cutoff) shows."""
        checked = self.vocabulary.check(tags)
        with self._services.db.transaction(bump_state=False) as session:
            row = self._row(session, md5)
            row.context_tags = checked
            row.context_note = meaning
            row.context_tagged_at = at
            row.context_cutoff_at = cutoff
            row.context_uses = uses
            row.desc_encoding = None
            refresh_effective(row, self.vocabulary)
            session.flush()
            return record_of(row)

    def set_manual(self, md5: str, tags: Sequence[str]) -> StickerRecord:
        """Set the tags by hand; they win over everything else (R-STK-003)."""
        checked = self.vocabulary.check(tags)
        if not checked:
            raise ValueError(
                "give at least one tag, or use `untag` to go back to the automatic ones"
            )
        with self._services.db.transaction(bump_state=True) as session:
            row = self._row(session, md5)
            row.manual_tags = checked[:3]
            refresh_effective(row, self.vocabulary)
            session.flush()
            return record_of(row)

    def clear_manual(self, md5: str) -> StickerRecord:
        with self._services.db.transaction(bump_state=True) as session:
            row = self._row(session, md5)
            row.manual_tags = None
            refresh_effective(row, self.vocabulary)
            session.flush()
            return record_of(row)

    def clear_stale_context(self, cutoff: datetime) -> int:
        """Forget the context correction made for another cutoff; returns how many stickers.

        After the hold-out is split again a correction may have used messages that are held out
        now (R-TRN-013); it is dropped at once and made again for the new cutoff.
        """
        cleared = 0
        with self._services.db.transaction(bump_state=True) as session:
            stmt = select(Sticker).where(
                Sticker.context_tagged_at.is_not(None),
                (Sticker.context_cutoff_at.is_(None)) | (Sticker.context_cutoff_at != cutoff),
            )
            for row in session.scalars(stmt):
                row.context_tags = None
                row.context_note = None
                row.context_tagged_at = None
                row.context_cutoff_at = None
                row.context_uses = 0
                refresh_effective(row, self.vocabulary)
                cleared += 1
        return cleared

    def set_disabled(self, md5: str, disabled: bool) -> StickerRecord:
        with self._services.db.transaction(bump_state=True) as session:
            row = self._row(session, md5)
            row.disabled = disabled
            session.flush()
            return record_of(row)

    @staticmethod
    def _row(session: Session, md5: str) -> Sticker:
        row = session.get(Sticker, md5)
        if row is None:
            raise UnknownStickerError(f"no sticker with MD5 {md5} in the library")
        return row
