"""The import report (R-IMP-010).

Counts and codes only: no message text, no ids, no names.  The target conversation is
referred to by its masked folder label (sequence, nickname length, four hash characters).
The report is written to ``data/reports/import-<UTC time>.md`` and shown in the terminal.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from twin.ingest.events import KINDS
from twin.ingest.runs import RunView
from twin.ingest.times import SourceTime
from twin.profile.report_section import style_change_lines
from twin.storage.chat_models import MediaAsset, Message, Sticker, StickerUse

if TYPE_CHECKING:
    from twin.services import Services

REPORT_PREFIX = "import-"
KIND_LABELS = {
    "text": "文字",
    "sticker": "表情包",
    "image": "图片",
    "quote": "引用回复",
    "voice": "语音",
    "video": "视频",
    "file": "文件",
    "link": "链接",
    "call": "通话",
    "transfer": "转账",
    "redpacket": "红包",
    "system": "系统消息",
    "location": "位置",
    "chathistory": "聊天记录",
    "unknown": "未识别",
}
STATUS_LABELS = {"queued": "已排队", "done": "完成", "skipped": "跳过", "failed": "失败"}


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _count(value: int) -> str:
    return f"{value:,}"


def build_report(
    services: Services,
    run: RunView,
    conversation_id: str,
    source_time: SourceTime,
    now: datetime,
    status: str | None = None,
) -> str:
    """Markdown text of the report for ``run`` (``status`` overrides the stored status)."""
    status = status or run.status
    db = services.db
    lines: list[str] = ["# 导入报告", ""]
    with db.session() as session:
        label = run.stats.target_label or "-"
        by_kind = session.execute(
            select(Message.kind, Message.is_sent, func.count())
            .where(Message.conversation_id == conversation_id)
            .group_by(Message.kind, Message.is_sent)
        ).all()
        total, first, last = session.execute(
            select(
                func.count(), func.min(Message.create_time_utc), func.max(Message.create_time_utc)
            ).where(Message.conversation_id == conversation_id)
        ).one()
        assets = session.execute(
            select(MediaAsset.kind, MediaAsset.status, MediaAsset.reason, func.count())
            .where(MediaAsset.conversation_id == conversation_id)
            .group_by(MediaAsset.kind, MediaAsset.status, MediaAsset.reason)
        ).all()
        captioned = session.scalar(
            select(func.count())
            .select_from(MediaAsset)
            .where(
                MediaAsset.conversation_id == conversation_id, MediaAsset.caption_ct.is_not(None)
            )
        )
        sticker_status = session.execute(
            select(Sticker.status, func.count()).group_by(Sticker.status)
        ).all()
        usage = session.execute(
            select(
                func.count().filter(Sticker.her_uses > 0),
                func.count().filter(Sticker.user_uses > 0),
                func.count().filter((Sticker.her_uses > 0) & (Sticker.user_uses > 0)),
                func.count(),
            )
        ).one()
        uses = session.execute(
            select(StickerUse.by_her, func.count()).group_by(StickerUse.by_her)
        ).all()

    lines += [
        f"- 生成时间（UTC）：{now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 导出标识：{run.export_id or '-'}",
        f"- 目标会话（打码）：{label}",
        f"- 导入状态：{status}（第 {run.stats.sessions} 次运行）",
        "",
        "## 本次导入",
        "",
        *_table(
            ["项目", "条数"],
            [
                ["读取", _count(run.processed)],
                ["新增", _count(run.inserted)],
                ["重复（已存在且内容相同）", _count(run.duplicates)],
                ["冲突并更新（新导出较新）", _count(run.conflict_updated)],
                ["冲突并保留（已存版本较新）", _count(run.conflict_kept)],
                ["无法导入", _count(run.invalid)],
            ],
        ),
    ]
    if run.stats.invalid_reasons:
        reasons = "，".join(f"{k} × {v}" for k, v in sorted(run.stats.invalid_reasons.items()))
        lines += ["", f"无法导入的原因：{reasons}"]

    lines += ["", "## 会话总览（数据库中）", ""]
    if total and first is not None and last is not None:
        first_day, last_day = source_time.local_date(first), source_time.local_date(last)
        lines += [
            f"- 消息总数：{_count(total)}",
            f"- 日期范围（按 time.source_timezone 的当地日期）：{first_day} 至 {last_day}"
            f"（共 {(last_day - first_day).days + 1} 天）",
        ]
    else:
        lines.append("- 数据库中还没有这个会话的消息")

    counts: dict[tuple[str, bool], int] = {(kind, sent): n for kind, sent, n in by_kind}
    rows = []
    for kind in KINDS:
        her, user = counts.get((kind, False), 0), counts.get((kind, True), 0)
        if her or user:
            rows.append([KIND_LABELS[kind], _count(her), _count(user), _count(her + user)])
    lines += ["", "## 按类型与发送方计数", ""]
    lines += _table(["类型", "她", "用户", "合计"], rows) if rows else ["（没有消息）"]

    lines += ["", "## 媒体文件", ""]
    if assets:
        lines += _table(
            ["种类", "状态", "原因", "数量"],
            [[kind, status, reason or "-", _count(n)] for kind, status, reason, n in assets],
        )
    else:
        lines.append("没有媒体记录。")
    lines += ["", f"已有图片描述：{_count(captioned or 0)}"]

    lines += ["", "## 表情包", ""]
    if sticker_status:
        lines += _table(["状态", "种类数"], [[s, _count(n)] for s, n in sticker_status])
        her_n, user_n, both_n, kinds_n = (int(x or 0) for x in usage)
        use_by = {bool(by_her): n for by_her, n in uses}
        lines += [
            "",
            f"- 共 {_count(kinds_n)} 种；她用过 {_count(her_n)} 种，用户用过 {_count(user_n)} 种，"
            f"两者都用过 {_count(both_n)} 种",
            f"- 使用记录：她 {_count(use_by.get(True, 0))} 次，"
            f"用户 {_count(use_by.get(False, 0))} 次",
            "- 直接取自本地 media/emojis 的文件："
            f"{_count(run.stats.stickers.get('local_files', 0))}",
            "- 其余等待下载（`twin stickers download`），下载统计见该命令的输出",
        ]
    else:
        lines.append("没有表情包。")

    lines += ["", "## 数据质量提示", ""]
    notes = _quality_notes(run)
    lines += [f"- {note}" for note in notes] if notes else ["- 无"]

    lines += ["", "## 风格变化", ""]
    lines += style_change_lines(services)

    lines += ["", "## 导入后钩子", ""]
    if run.hooks:
        lines += _table(
            ["钩子", "状态", "说明", "回填命令"],
            [
                [
                    name,
                    STATUS_LABELS.get(str(entry.get("status")), str(entry.get("status"))),
                    entry.get("detail") or "-",
                    f"`twin {entry.get('backfill_command')}`",
                ]
                for name, entry in run.hooks.items()
            ],
        )
    else:
        lines.append("没有已注册的钩子。")
    lines.append("")
    return "\n".join(lines)


def _quality_notes(run: RunView) -> list[str]:
    stats = run.stats
    notes: list[str] = []
    if stats.unknown_render_types:
        found = "，".join(f"{k} × {v}" for k, v in sorted(stats.unknown_render_types.items()))
        notes.append(f"未识别的 renderType（已按“未识别”类型保存原始记录）：{found}")
    if stats.unknown_fields:
        names = "，".join(sorted(stats.unknown_fields))
        notes.append(f"消息里有 SPEC 未列出的字段（原样保存在原始记录中，只列字段名）：{names}")
    if stats.schema_errors:
        names = "，".join(f"{k} × {v}" for k, v in sorted(stats.schema_errors.items()))
        notes.append(f"这些字段的类型与预期不符，已当作缺失处理：{names}")
    if stats.time_mismatches:
        notes.append(
            f"{stats.time_mismatches} 条消息的 createTime 与 createTimeText 相差超过一小时："
            "请核对 time.source_timezone / time.source_timezone_ranges（SPEC R-CFG-004）"
        )
    integrity = stats.integrity
    if integrity.get("state") == "read":
        notes.append(
            f"完整性校验（_integrity）：读到 {integrity.get('entries', 0)} 条记录；"
            f"messages.json {integrity.get('messages_json', '未校验')}；"
            f"媒体通过 {stats.media.get('integrity_verified', 0)}，"
            f"失败 {stats.media.get('integrity_failed', 0)}，"
            f"未列入清单 {stats.media.get('integrity_unlisted', 0)}"
        )
    for note in stats.notes:
        notes.append(note)
    if stats.media.get("md5_mismatch"):
        notes.append(
            f"{stats.media['md5_mismatch']} 个媒体文件的 MD5 与导出记录不一致（文件仍已保存）"
        )
    return notes


def write_report(reports_dir: Path, text: str, now: datetime) -> Path:
    """Save the report as ``import-<UTC time>.md`` and return its path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{REPORT_PREFIX}{now.strftime('%Y%m%dT%H%M%SZ')}.md"
    counter = 1
    while path.exists():
        counter += 1
        path = reports_dir / f"{REPORT_PREFIX}{now.strftime('%Y%m%dT%H%M%SZ')}-{counter}.md"
    path.write_text(text, encoding="utf-8")
    return path
