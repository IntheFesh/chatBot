"""Counters an import run keeps (and saves with each batch so a resumed run continues them).

Everything here is a count or a short code: no message text, no ids, no names.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

MAX_NOTES = 20


@dataclass
class RunStats:
    """Counters of one import run (stored sealed in ``import_runs.stats``)."""

    kinds: Counter[str] = field(default_factory=Counter)  # "<kind>:<her|user>" -> messages seen
    unknown_render_types: Counter[str] = field(default_factory=Counter)
    unknown_fields: Counter[str] = field(default_factory=Counter)
    invalid_reasons: Counter[str] = field(default_factory=Counter)
    schema_errors: Counter[str] = field(default_factory=Counter)
    time_mismatches: int = 0
    user_avatar_path: str | None = None
    target_label: str | None = None
    integrity: dict[str, Any] = field(default_factory=dict)
    media: Counter[str] = field(default_factory=Counter)
    stickers: Counter[str] = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)
    sessions: int = 0

    def note(self, text: str) -> None:
        if text not in self.notes and len(self.notes) < MAX_NOTES:
            self.notes.append(text)

    def to_json(self) -> dict[str, Any]:
        return {
            "kinds": dict(self.kinds),
            "unknown_render_types": dict(self.unknown_render_types),
            "unknown_fields": dict(self.unknown_fields),
            "invalid_reasons": dict(self.invalid_reasons),
            "schema_errors": dict(self.schema_errors),
            "time_mismatches": self.time_mismatches,
            "user_avatar_path": self.user_avatar_path,
            "target_label": self.target_label,
            "integrity": self.integrity,
            "media": dict(self.media),
            "stickers": dict(self.stickers),
            "notes": list(self.notes),
            "sessions": self.sessions,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> RunStats:
        if not data:
            return cls()
        return cls(
            kinds=Counter(data.get("kinds", {})),
            unknown_render_types=Counter(data.get("unknown_render_types", {})),
            unknown_fields=Counter(data.get("unknown_fields", {})),
            invalid_reasons=Counter(data.get("invalid_reasons", {})),
            schema_errors=Counter(data.get("schema_errors", {})),
            time_mismatches=int(data.get("time_mismatches", 0)),
            user_avatar_path=data.get("user_avatar_path"),
            target_label=data.get("target_label"),
            integrity=dict(data.get("integrity", {})),
            media=Counter(data.get("media", {})),
            stickers=Counter(data.get("stickers", {})),
            notes=list(data.get("notes", [])),
            sessions=int(data.get("sessions", 0)),
        )
