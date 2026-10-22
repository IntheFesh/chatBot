"""The "风格变化" section of the import report (R-IMP-010) and how a job rewrites a section.

The import queues the profile recomputation (a job), so when the report is written the new
profile may not exist yet.  The section therefore says what is known at that moment — the
changes of the newest ``live`` version over its parent, or a note that the recomputation is
queued — and the job rewrites the section of the newest import report when it finishes
(:func:`refresh_latest_import_report`).  The section lists metric names and numbers only.

The same two functions serve every section that a later job completes: a ``heading`` says
which one, and the lines come from the caller (the sticker download rewrites "## 表情包", R17-39).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from twin.profile.diffing import summarize
from twin.profile.store import VersionStore
from twin.storage.chat_models import ImportRun

if TYPE_CHECKING:
    from twin.services import Services

HEADING = "## 风格变化"
MAX_LINES = 12


def style_change_lines(services: Services) -> list[str]:
    """The bullet lines of the section (without the heading)."""
    version = VersionStore(services.db, services.clock).active_profile("live")
    if version is None:
        return [
            "- 画像还没有计算：导入后会自动排队重算，也可以手动运行 `twin profile rebuild`；"
            "完成后这里会列出变化超过 10% 的指标"
        ]
    stamp = f"{version.created_at:%Y-%m-%d %H:%M} UTC"
    if version.parent_id is None:
        return [
            f"- 首次画像（版本 {version.id[-8:]}，{stamp}，她的 {version.her_messages:,} 条消息）"
        ]
    if not version.changes:
        return [f"- 与上一版（{stamp} 的版本 {version.id[-8:]}）相比没有变化超过 10% 的指标"]
    lines = [f"- 版本 {version.id[-8:]}（{stamp}）相对上一版变化超过 10% 的指标："]
    lines += [f"  - {line}" for line in summarize(version.changes, MAX_LINES)]
    if len(version.changes) > MAX_LINES:
        lines.append(f"  - 其余 {len(version.changes) - MAX_LINES} 项见 `twin profile show`")
    return lines


def replace_section(text: str, lines: list[str], heading: str = HEADING) -> str:
    """``text`` with the body of the section ``heading`` replaced (appended if it is missing)."""
    body = "\n".join(lines)
    start = text.find(heading)
    if start < 0:
        return text.rstrip("\n") + f"\n\n{heading}\n\n{body}\n"
    after = text.find("\n## ", start + len(heading))
    tail = text[after:] if after >= 0 else "\n"
    return f"{text[:start]}{heading}\n\n{body}\n{tail}"


def refresh_latest_import_report(
    services: Services,
    *,
    heading: str = HEADING,
    lines: Callable[[dict[str, Any]], list[str]] | None = None,
) -> Path | None:
    """Rewrite one section of the newest finished import's report.

    By default the style-change section; with ``heading`` and ``lines`` another one - ``lines``
    receives the stored counters of that import run (the run the report belongs to) and returns
    the bullet lines.  "The newest finished import" is the report a later job completes: a job
    that finishes after a newer import was made fills in that one's report, which is the one the
    user is reading.  Nothing is written if there is no report file any more.
    """
    with services.db.session() as session:
        stmt = (
            select(ImportRun)
            .where(ImportRun.status == "done", ImportRun.report_path.is_not(None))
            .order_by(ImportRun.finished_at.desc())
            .limit(1)
        )
        run = session.scalars(stmt).first()
        found = run.report_path if run is not None else None
        stats = dict(run.stats or {}) if run is not None else {}
    if not found:
        return None
    path = Path(found)
    if not path.is_file():
        return None
    original = path.read_text(encoding="utf-8")
    body = lines(stats) if lines is not None else style_change_lines(services)
    updated = replace_section(original, body, heading)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
    return path
