"""The WeChat platform and the user's phone, made up: the other end of the iLink channel.

``tests/support/ilink.py`` builds single wire messages for the channel's unit tests.  The
end-to-end scenarios need the **whole other side**: a server that keeps the messages the user types
until the bot asks for them (``getupdates`` with a cursor, so that a restarted bot gets the ones it
did not commit again), takes the bot's ``sendmessage`` requests and shows them on a phone, and says
no the way the platform does when the session is over - ``ret: -2`` - or when the count of
messages since the user's last one is used up.  Nothing here is imported by ``src``; the channel
under test is the production :class:`~twin.channel.ilink.channel.IlinkChannel`, talking HTTP that a
``respx`` router answers from this object.

* :attr:`WeChatDouble.phone` is what the user sees (:meth:`Phone.messages` has the same shape as
  the terminal screen of the console scenarios);
* :meth:`WeChatDouble.user_types` is the user typing;
* ``window_h`` / ``quota`` are the platform's own rules (the bot's safe thresholds are lower);
* ``after_delivery`` runs at the instant a message has been delivered and the answer is not yet
  on its way back - the instant at which a process can die with the message already shown.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import httpx
import respx

from tests.support.ilink import API, CTX, USER, message, now_ms, text_item, updates
from tests.support.life_clock import LifeClock
from tests.support.life_screen import PREFIX, Said
from twin.channel.ilink.wire import ERR_SEND_REJECTED, ITEM_TEXT

TYPING_TICKET = "TYPING-TICKET-SYNTHETIC"


@dataclass(frozen=True)
class Delivery:
    """One thing that reached the user's phone."""

    at: datetime
    kind: str  # "text" | "image" | "typing"
    text: str | None = None


@dataclass
class Phone:
    """The user's phone: everything the platform showed him."""

    delivered: list[Delivery] = field(default_factory=list)

    def messages(self) -> list[Said]:
        found: list[Said] = []
        for item in self.delivered:
            if item.kind == "text" and item.text is not None:
                kind = "system" if item.text.startswith(PREFIX) else "text"
                found.append(Said(item.at, kind, item.text))
            elif item.kind == "image":
                found.append(Said(item.at, "image", "[图片]"))
        return found

    def typing_times(self) -> list[datetime]:
        return [item.at for item in self.delivered if item.kind == "typing"]


@dataclass
class WeChatDouble:
    """The platform (see the module description)."""

    clock: LifeClock
    window_h: float = 24.0
    quota: int = 10
    phone: Phone = field(default_factory=Phone)
    typed: list[tuple[int, dict[str, Any]]] = field(default_factory=list)  # (seq, wire message)
    last_user_at: datetime | None = None
    sent_since_user: int = 0
    refused: int = 0
    lose_response: bool = False  # the next delivery is shown, its answer never arrives
    after_delivery: Callable[[Delivery], None] | None = None
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    _seq: int = 0

    # ---- the user ------------------------------------------------------------------------

    def user_types(self, text: str) -> None:
        """The user sends ``text`` now: the window restarts, the count of messages starts afresh."""
        self._seq += 1
        wire = message(
            text_item(text),
            mid=10_000 + self._seq,
            created_ms=now_ms(self.clock),
            context_token=CTX,
            seq=self._seq,
        )
        self.typed.append((self._seq, wire))
        self.last_user_at = self.clock.now_utc()
        self.sent_since_user = 0

    @property
    def unread(self) -> int:
        return len(self.typed)

    # ---- the transport ----------------------------------------------------------------------

    def mount(self, router: respx.MockRouter) -> None:
        router.route(method="POST", url__startswith=f"{API}/ilink/bot/").mock(
            side_effect=self.handle
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.removeprefix("/ilink/bot/")
        body = json.loads(request.content) if request.content else {}
        self.requests.append((endpoint, body))
        if endpoint == "getupdates":
            return self._updates(body)
        if endpoint == "sendmessage":
            return self._send(body, request)
        if endpoint == "getconfig":
            return httpx.Response(200, json={"ret": 0, "typing_ticket": TYPING_TICKET})
        if endpoint == "sendtyping":
            self.phone.delivered.append(
                Delivery(self.clock.now_utc(), "typing", str(body.get("status")))
            )
            return httpx.Response(200, json={"ret": 0})
        return httpx.Response(200, json={"ret": 0})  # notifystart, notifystop ...

    def _updates(self, body: dict[str, Any]) -> httpx.Response:
        cursor = str(body.get("get_updates_buf") or "0")
        seen = int(cursor) if cursor.isdigit() else 0
        fresh = [wire for seq, wire in self.typed if seq > seen]
        newest = max((seq for seq, _ in self.typed), default=seen)
        return httpx.Response(200, json=updates(fresh, cursor=str(newest)))

    def _send(self, body: dict[str, Any], request: httpx.Request) -> httpx.Response:
        sent = body["msg"]
        assert sent["to_user_id"] == USER and sent["from_user_id"] == ""  # only ever to him
        now = self.clock.now_utc()
        expired = self.last_user_at is None or now - self.last_user_at > timedelta(
            hours=self.window_h
        )
        if expired or self.sent_since_user >= self.quota:
            self.refused += 1
            return httpx.Response(
                200, json={"ret": ERR_SEND_REJECTED, "errmsg": "session expired or count used"}
            )
        item = sent["item_list"][0]
        if item["type"] == ITEM_TEXT:
            shown = Delivery(now, "text", item["text_item"]["text"])
        else:
            shown = Delivery(now, "image")
        self.phone.delivered.append(shown)
        self.sent_since_user += 1
        if self.after_delivery is not None:
            self.after_delivery(shown)
        if self.lose_response:
            self.lose_response = False
            raise httpx.ReadTimeout("no answer came back", request=request)
        return httpx.Response(
            200, json={"ret": 0, "message_id": f"out-{len(self.phone.delivered)}"}
        )

    # ---- what the tests read ------------------------------------------------------------------

    def texts(self) -> list[str]:
        return [m.text for m in self.phone.messages() if m.kind == "text"]


__all__ = ["Delivery", "Phone", "WeChatDouble"]
