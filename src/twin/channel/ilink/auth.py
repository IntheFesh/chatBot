"""What happens when the iLink login stops working (R-CH-003, R-OPS-004, protocol section 8).

The server signals a dead bot token with error code -14.  The channel then switches to
"needs re-login": nothing is sent or polled, an alert is written to the ``alerts`` table
(critical, one per episode) and the console of the running application shows a conspicuous
message telling the user to run ``twin channel login``.  The QR code itself is only ever
shown on this machine.
"""

from __future__ import annotations

import asyncio

from twin.channel.console import AlertBanner
from twin.channel.ilink.store import IlinkStore
from twin.llm.redaction import redact_text
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger

log = get_logger("twin.channel.ilink.auth")

ALERT_CATEGORY = "channel.auth_expired"
RECOVERED_CATEGORY = "channel.auth_recovered"
LOGIN_COMMAND = "twin channel login"


class AuthGuard:
    """Records an expired login once and tells the user how to fix it."""

    def __init__(self, store: IlinkStore, alerts: AlertSink, banner: AlertBanner) -> None:
        self._store = store
        self._alerts = alerts
        self._banner = banner

    async def on_expired(self, *, source: str, code: int | None, errmsg: str | None) -> bool:
        """Mark the login expired.  Returns ``True`` only for the first report of an episode."""
        clean = redact_text(errmsg)[:200] if errmsg else None
        changed = await asyncio.to_thread(self._store.mark_needs_relogin, code, clean)
        if not changed:
            return False
        log.error("login_expired", source=source, code=code)
        await asyncio.to_thread(self._raise_alert, source, code, clean)
        self._banner.show(
            "WeChat login expired",
            [
                "The bot can no longer receive or send messages.",
                f"Run `{LOGIN_COMMAND}` and scan the new QR code with your phone.",
            ],
        )
        return True

    def _raise_alert(self, source: str, code: int | None, errmsg: str | None) -> None:
        self._alerts.raise_alert(
            ALERT_CATEGORY,
            f"WeChat login expired: run `{LOGIN_COMMAND}` and scan the QR code",
            severity="critical",
            detail={"source": source, "code": code, "errmsg": errmsg},
            dedup_key=ALERT_CATEGORY,
        )

    async def on_recovered(self) -> bool:
        """The old token works again (hourly probe).  Returns ``True`` if the state changed."""
        changed = await asyncio.to_thread(self._store.mark_auth_ok)
        if changed:
            log.info("login_recovered")
            await asyncio.to_thread(
                self._alerts.raise_alert,
                RECOVERED_CATEGORY,
                "WeChat login works again",
                severity="info",
                dedup_key=RECOVERED_CATEGORY,
            )
        return changed
