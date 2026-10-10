"""The words of an alert notice (Windows notification and e-mail), R-OPS-004, R-SAFE-004.

What leaves the machine is built from three things only: the fixed wording of the category
(:data:`~twin.ops.alerts.SPECS`), the time, and numbers or short codes taken from the alert's
``detail``.  A text value of ``detail`` is shown only if it is a single lowercase token such as
``getupdates`` or ``http_503`` - a sentence, any Chinese, a key named like content
(:data:`~twin.ops.logging.CONTENT_FIELDS`) never is - and the finished text passes
:func:`~twin.llm.redaction.redact_text` once more.

The one-line ``title`` that the caller gives an alert is for the ``alerts`` table, the log and
``twin health``; it is **not** part of any notice.  A free-text field that a notice copies is a
field through which chat content could leave the machine one day, so there is none: the facts a
caller wants the user to see go into ``detail`` as numbers and codes (``tests/unit/
test_outbound_audit.py`` puts a chat sentence into every field and asserts it never comes out).

The QR code of a lost login is never part of a notice: the category's advice says where it is.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from twin.llm.redaction import redact_text
from twin.ops.alerts import AlertView, spec_of
from twin.ops.logging import CONTENT_FIELDS

SUBJECT_PREFIX = "[wechat-twin]"
SEVERITY_WORDS = {"info": "提示", "warning": "警告", "critical": "严重"}
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_TOKEN = re.compile(r"^[a-z0-9_.:\-]{1,32}$")
MAX_DETAIL_LINES = 8


@dataclass(frozen=True)
class RenderedAlert:
    """One notice in the forms the channels need."""

    subject: str
    text: str
    html: str
    toast_title: str
    toast_body: str


def safe_detail(detail: dict[str, Any] | None) -> list[tuple[str, str]]:
    """The ``key = value`` pairs of ``detail`` that may be shown (see the module text)."""
    pairs: list[tuple[str, str]] = []
    for key, value in (detail or {}).items():
        if not isinstance(key, str) or not _KEY.fullmatch(key) or key in CONTENT_FIELDS:
            continue
        if isinstance(value, bool):
            pairs.append((key, "是" if value else "否"))
        elif isinstance(value, int | float):
            pairs.append((key, f"{value:g}" if isinstance(value, float) else str(value)))
        elif isinstance(value, str) and _TOKEN.fullmatch(value):
            pairs.append((key, value))
        if len(pairs) >= MAX_DETAIL_LINES:
            break
    return pairs


def render_alert(view: AlertView, zone: ZoneInfo) -> RenderedAlert:
    """The notice for one alert (or the "recovered" notice that closes one)."""
    spec = spec_of(view.category)
    recovered = view.kind == "recovery"
    label = f"已恢复：{spec.label}" if recovered else spec.label
    local = view.created_at.astimezone(zone)
    when = f"{local:%Y-%m-%d %H:%M}（{zone.key}）"
    lines = [f"{SEVERITY_WORDS.get(view.severity, view.severity)}：{label}", f"时间：{when}"]
    if not recovered:
        lines.append(f"怎么办：{spec.advice}")
    for key, value in safe_detail(view.detail):
        lines.append(f"{key} = {value}")
    lines.append("（这封通知不包含任何聊天内容。）")
    text = redact_text("\n".join(lines))
    body = "".join(f"<p>{html.escape(line)}</p>" for line in text.splitlines())
    page = (
        '<!doctype html><html lang="zh"><head><meta charset="utf-8"></head>'
        f"<body>{body}</body></html>"
    )
    return RenderedAlert(
        subject=f"{SUBJECT_PREFIX} {label}",
        text=text,
        html=page,
        toast_title=f"wechat-twin：{label}",
        toast_body=redact_text(label if recovered else spec.advice)[:200],
    )
