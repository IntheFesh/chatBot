"""The message to the emergency contact: a time and nothing else (R-SAFE-001, R-SAFE-004).

Only when ``safety.emergency_contact.enabled`` is on *and* an address is filled in, a crisis sends
one e-mail to that address.  The e-mail never carries chat content, and the code makes that a
property of the types, not a promise: an :class:`EmergencyNotice` has no field for text - only the
moment and the recipient - and :func:`render_notice` fills the one fixed template from the moment
alone.  The SMTP delivery is round 12's (:class:`EmergencyNotifier` is the interface it
implements); the recipient is always the address from the configuration (R-SAFE-004).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from twin.clock import ensure_aware

SUBJECT = "提醒：他可能需要关心"
BODY_TEMPLATE = (
    "这是一封自动发送的提醒。\n"
    "时间：{time}。\n"
    "他可能需要关心，方便的话请联系一下他。\n"
    "（这封邮件不包含任何聊天内容。）"
)


@dataclass(frozen=True)
class EmergencyNotice:
    """Who is told, and when it happened.  There is deliberately no field for anything else."""

    at: datetime
    recipient: str

    def __post_init__(self) -> None:
        ensure_aware(self.at)
        if not self.recipient.strip():
            raise ValueError("an emergency notice needs a recipient")


@dataclass(frozen=True)
class EmergencyEmail:
    """The rendered message: a fixed subject and a body made from the time alone."""

    to: str
    subject: str
    body: str


def render_notice(notice: EmergencyNotice, zone: ZoneInfo) -> EmergencyEmail:
    """The e-mail for ``notice``, with the time written on the clock of ``zone``."""
    local = ensure_aware(notice.at).astimezone(zone)
    return EmergencyEmail(
        to=notice.recipient,
        subject=SUBJECT,
        body=BODY_TEMPLATE.format(time=f"{local:%Y-%m-%d %H:%M}（{zone.key}）"),
    )


class EmergencyNotifier(Protocol):
    """Delivers an :class:`EmergencyNotice` (the SMTP implementation is round 12's)."""

    async def notify(self, notice: EmergencyNotice) -> bool:
        """Send it; ``True`` if the message was handed to the mail server."""
        ...
