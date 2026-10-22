"""What the last sticker download did, kept for the import report (R-IMP-010, R17-39).

The download is a job that runs long after the import wrote its report, and the numbers of a job
(how many pictures were tried, how many arrived, how many had another MD5, how many could not be
fetched and why) used to be in the output of ``twin stickers download`` only.  The job now keeps
them under one key of ``settings`` - counts and short reason codes, never an URL, a name or a
picture - and the report section reads them from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from twin.storage.settings_store import get_setting, put_setting

if TYPE_CHECKING:
    from twin.services import Services

SUMMARY_KEY = "stickers.download.summary"


@dataclass(frozen=True)
class DownloadSummary:
    """The result of one run of the download job."""

    at: datetime
    attempted: int
    available: int
    md5_mismatch: int
    unavailable: int
    reasons: dict[str, int] = field(default_factory=dict)
    stopped: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "at": self.at.astimezone(UTC).isoformat(),
            "attempted": self.attempted,
            "available": self.available,
            "md5_mismatch": self.md5_mismatch,
            "unavailable": self.unavailable,
            "reasons": dict(sorted(self.reasons.items())),
            "stopped": self.stopped,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DownloadSummary:
        reasons = data.get("reasons") or {}
        at = datetime.fromisoformat(str(data["at"]))
        return cls(
            at=at if at.tzinfo is not None else at.replace(tzinfo=UTC),
            attempted=int(data.get("attempted", 0)),
            available=int(data.get("available", 0)),
            md5_mismatch=int(data.get("md5_mismatch", 0)),
            unavailable=int(data.get("unavailable", 0)),
            reasons={str(k): int(v) for k, v in reasons.items()},
            stopped=bool(data.get("stopped", False)),
        )

    def describe(self) -> str:
        """One line of the report: when, the four counts, the reasons (no text, no URL)."""
        parts = [
            f"尝试 {self.attempted}",
            f"可用 {self.available}",
            f"MD5 不符 {self.md5_mismatch}",
            f"不可用 {self.unavailable}",
        ]
        line = f"最近一次下载（{self.at:%Y-%m-%d %H:%M} UTC）：" + "，".join(parts)
        if self.reasons:
            why = "，".join(f"{reason} × {count}" for reason, count in sorted(self.reasons.items()))
            line += f"（原因：{why}）"
        if self.stopped:
            line += "；被中途停止，其余的下次继续"
        return line


def save_download_summary(services: Services, summary: DownloadSummary) -> None:
    """Keep ``summary`` as the last download (the one before it is replaced)."""
    with services.db.transaction(bump_state=False) as session:
        put_setting(
            session,
            SUMMARY_KEY,
            summary.to_json(),
            clock=services.clock,
            by="sticker_download",
            record_history=False,
        )


def load_download_summary(services: Services) -> DownloadSummary | None:
    """The summary of the last download, or ``None`` if no download has finished yet."""
    with services.db.session() as session:
        value = get_setting(session, SUMMARY_KEY, None)
    if not isinstance(value, dict):
        return None
    try:
        return DownloadSummary.from_json(value)
    except (KeyError, TypeError, ValueError):
        return None
