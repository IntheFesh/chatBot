"""Persistent state of the iLink channel, kept in the encrypted ``channel_state`` table.

Keys (see ``docs/ILINK_PROTOCOL.md`` section 10.3):

``ilink.credentials``   bot token, bot id, expected user id, API host, save time
``ilink.auth_state``    ``ok`` or ``needs_relogin`` (+ since, code)
``ilink.bound_user``    the one user the bot talks to (R-CH-007)
``ilink.pending_bind``  the first sender seen while nobody is bound, awaiting confirmation
``ilink.cursor``        the ``get_updates_buf`` poll cursor
``ilink.seen_ids``      the most recent processed message ids (de-duplication)
``ilink.inbox``         accepted messages not yet handed to the consumer
``ilink.context_token`` the newest ``context_token`` of the bound user
``ilink.window``        the conversation window (:class:`~twin.channel.window.WindowState`)
``ilink.quote_index``   recent message texts, to resolve quotes that carry only a server id
``ilink.item_stats``    item type numbers of recent inbound messages and parse failures
``ilink.poll_status``   consecutive poll failures and the last poll error

The cursor, the seen ids and the inbox change together in one transaction
(:meth:`IlinkStore.commit_batch`): after a crash a message is either still in the inbox (and
delivered again) or already handled, never lost and never repeated.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TypeVar

from twin.channel.base import AuthState, InboundMessage
from twin.channel.state import ChannelStateStore, StateTx
from twin.channel.window import SessionWindow, WindowState
from twin.clock import Clock, ensure_aware

T = TypeVar("T")

KEY_CREDENTIALS = "ilink.credentials"
KEY_AUTH = "ilink.auth_state"
KEY_BOUND = "ilink.bound_user"
KEY_PENDING_BIND = "ilink.pending_bind"
KEY_CURSOR = "ilink.cursor"
KEY_SEEN = "ilink.seen_ids"
KEY_INBOX = "ilink.inbox"
KEY_CONTEXT = "ilink.context_token"
KEY_WINDOW = "ilink.window"
KEY_QUOTES = "ilink.quote_index"
KEY_STATS = "ilink.item_stats"
KEY_POLL = "ilink.poll_status"

SEEN_IDS_MAX = 500
INBOX_MAX = 500
QUOTE_INDEX_MAX = 300
QUOTE_TEXT_MAX = 200
QUOTE_KEEP = timedelta(days=30)
POLL_OK_WRITE_INTERVAL = timedelta(seconds=60)
STATS_RECENT_MAX = 100


@dataclass(frozen=True)
class Credentials:
    bot_token: str
    ilink_bot_id: str
    ilink_user_id: str | None
    api_base_url: str
    saved_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_token": self.bot_token,
            "ilink_bot_id": self.ilink_bot_id,
            "ilink_user_id": self.ilink_user_id,
            "api_base_url": self.api_base_url,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Credentials:
        return cls(
            bot_token=str(data["bot_token"]),
            ilink_bot_id=str(data["ilink_bot_id"]),
            ilink_user_id=data.get("ilink_user_id"),
            api_base_url=str(data["api_base_url"]),
            saved_at=str(data["saved_at"]),
        )


@dataclass(frozen=True)
class AuthRecord:
    state: AuthState
    since: datetime | None = None
    code: int | None = None
    errmsg: str | None = None


@dataclass(frozen=True)
class BoundUser:
    user_id: str
    bound_at: datetime


@dataclass(frozen=True)
class PendingBinding:
    """The first sender seen while nobody is bound (their message is not processed)."""

    user_id: str
    seen_at: datetime
    context_token: str | None
    matches_expected: bool | None  # compared with the user id the login returned


@dataclass(frozen=True)
class ContextToken:
    token: str
    received_at: datetime


@dataclass
class BatchCommit:
    """Everything one ``getupdates`` batch changes, committed together."""

    new_cursor: str | None = None
    seen_ids: list[str] = field(default_factory=list)
    deliver: list[InboundMessage] = field(default_factory=list)
    context_token: ContextToken | None = None
    last_inbound_at: datetime | None = None
    item_types: list[int] = field(default_factory=list)
    failures: Counter[str] = field(default_factory=Counter)
    quote_entries: list[tuple[str, str]] = field(default_factory=list)
    candidate: PendingBinding | None = None


@dataclass(frozen=True)
class ItemStats:
    recent_types: list[int]
    failures: dict[str, int]

    def counts(self) -> dict[int, int]:
        return dict(sorted(Counter(self.recent_types).items()))


@dataclass(frozen=True)
class PollStatus:
    consecutive_failures: int = 0
    last_error: dict[str, Any] | None = None
    last_ok_at: datetime | None = None


def _parse(value: str | None) -> datetime | None:
    return None if not value else ensure_aware(datetime.fromisoformat(value))


class IlinkStore:
    """Typed access to the iLink keys of ``channel_state``."""

    def __init__(self, state: ChannelStateStore, clock: Clock) -> None:
        self._state = state
        self._clock = clock

    @property
    def state(self) -> ChannelStateStore:
        return self._state

    # --------------------------------------------------------- credentials

    def credentials(self) -> Credentials | None:
        data = self._state.get(KEY_CREDENTIALS)
        return Credentials.from_dict(data) if data else None

    def save_credentials(self, credentials: Credentials) -> None:
        """Store a fresh login: auth ok, cursor and context token reset (section 8.3)."""
        with self._state.transaction() as tx:
            tx.put(KEY_CREDENTIALS, credentials.to_dict())
            tx.put(KEY_AUTH, {"state": "ok", "since": credentials.saved_at})
            tx.put(KEY_CURSOR, "")
            tx.delete(KEY_CONTEXT)
            tx.delete(KEY_WINDOW)
            tx.delete(KEY_POLL)

    def auth_record(self) -> AuthRecord:
        creds = self._state.get(KEY_CREDENTIALS)
        if not creds:
            return AuthRecord(AuthState.NOT_LOGGED_IN)
        data = self._state.get(KEY_AUTH) or {}
        state = AuthState.NEEDS_RELOGIN if data.get("state") == "needs_relogin" else AuthState.OK
        return AuthRecord(state, _parse(data.get("since")), data.get("code"), data.get("errmsg"))

    def mark_needs_relogin(self, code: int | None, errmsg: str | None) -> bool:
        """Switch to "needs re-login"; ``False`` if it already was (no second alert)."""
        with self._state.transaction() as tx:
            current = tx.get(KEY_AUTH) or {}
            if current.get("state") == "needs_relogin":
                return False
            tx.put(
                KEY_AUTH,
                {
                    "state": "needs_relogin",
                    "since": self._clock.now_utc().isoformat(),
                    "code": code,
                    "errmsg": errmsg,
                },
            )
            return True

    def mark_auth_ok(self) -> bool:
        """The old token works again (hourly recovery probe); ``False`` if nothing changed."""
        with self._state.transaction() as tx:
            current = tx.get(KEY_AUTH) or {}
            if current.get("state") != "needs_relogin":
                return False
            tx.put(KEY_AUTH, {"state": "ok", "since": self._clock.now_utc().isoformat()})
            return True

    # ------------------------------------------------------------ binding

    def bound_user(self) -> BoundUser | None:
        data = self._state.get(KEY_BOUND)
        if not data:
            return None
        return BoundUser(
            str(data["user_id"]), ensure_aware(datetime.fromisoformat(data["bound_at"]))
        )

    def pending_binding(self) -> PendingBinding | None:
        data = self._state.get(KEY_PENDING_BIND)
        if not data:
            return None
        return PendingBinding(
            user_id=str(data["user_id"]),
            seen_at=ensure_aware(datetime.fromisoformat(data["seen_at"])),
            context_token=data.get("context_token"),
            matches_expected=data.get("matches_expected"),
        )

    def bind(self, user_id: str, *, context_token: str | None = None) -> BoundUser:
        """Bind ``user_id`` as the only user.  Keeps the candidate's context token."""
        now = self._clock.now_utc()
        with self._state.transaction() as tx:
            tx.put(KEY_BOUND, {"user_id": user_id, "bound_at": now.isoformat()})
            tx.delete(KEY_PENDING_BIND)
            if context_token:
                tx.put(KEY_CONTEXT, {"token": context_token, "received_at": now.isoformat()})
                tx.put(KEY_WINDOW, WindowState().after_inbound(now).to_dict())
        return BoundUser(user_id, now)

    def unbind(self) -> bool:
        """Forget the bound user and everything tied to the conversation (R-CH-007)."""
        with self._state.transaction() as tx:
            had = tx.get(KEY_BOUND) is not None
            for key in (
                KEY_BOUND,
                KEY_PENDING_BIND,
                KEY_CONTEXT,
                KEY_WINDOW,
                KEY_INBOX,
                KEY_QUOTES,
            ):
                tx.delete(key)
            return had

    def clear_pending_binding(self) -> None:
        self._state.delete(KEY_PENDING_BIND)

    # ------------------------------------------------------- cursor, ids

    def cursor(self) -> str:
        return str(self._state.get(KEY_CURSOR, "") or "")

    def seen_ids(self) -> list[str]:
        return [str(item) for item in self._state.get(KEY_SEEN, [])]

    def context_token(self) -> ContextToken | None:
        data = self._state.get(KEY_CONTEXT)
        if not data:
            return None
        return ContextToken(
            str(data["token"]), ensure_aware(datetime.fromisoformat(data["received_at"]))
        )

    # -------------------------------------------------------------- inbox

    def inbox(self) -> list[InboundMessage]:
        return [InboundMessage.from_dict(item) for item in self._state.get(KEY_INBOX, [])]

    def ack(self, message_id: str) -> bool:
        """The consumer is done with ``message_id``: it will not be delivered again."""
        with self._state.transaction() as tx:
            items = list(tx.get(KEY_INBOX, []))
            kept = [item for item in items if item.get("id") != message_id]
            if len(kept) == len(items):
                return False
            tx.put(KEY_INBOX, kept)
            return True

    # ------------------------------------------------------------- window

    def window_state(self) -> WindowState:
        return WindowState.from_dict(self._state.get(KEY_WINDOW))

    def update_window(self, window_h: float, quota: int, change: Callable[[SessionWindow], T]) -> T:
        """Load the stored window, apply ``change`` and store the result atomically."""
        with self._state.transaction() as tx:
            window = SessionWindow(
                window_h=window_h, quota=quota, state=WindowState.from_dict(tx.get(KEY_WINDOW))
            )
            result = change(window)
            tx.put(KEY_WINDOW, window.state.to_dict())
            return result

    # -------------------------------------------------------- quote index

    def quote_text(self, message_id: str) -> str | None:
        for entry_id, text, _at in self._state.get(KEY_QUOTES, []):
            if entry_id == message_id:
                return str(text)
        return None

    def add_quote(self, message_id: str, text: str) -> None:
        """Remember what a message said (for quotes that carry only the server id)."""
        with self._state.transaction() as tx:
            self._add_quotes(tx, [(message_id, text)])

    def _add_quotes(self, tx: StateTx, entries: list[tuple[str, str]]) -> None:
        if not entries:
            return
        now = self._clock.now_utc()
        horizon = now - QUOTE_KEEP
        index: list[list[str]] = [
            list(item)
            for item in tx.get(KEY_QUOTES, [])
            if datetime.fromisoformat(item[2]) >= horizon
        ]
        new_ids = {entry_id for entry_id, _ in entries}
        index = [item for item in index if item[0] not in new_ids]
        for entry_id, text in entries:
            index.append([entry_id, text[:QUOTE_TEXT_MAX], now.isoformat()])
        tx.put(KEY_QUOTES, index[-QUOTE_INDEX_MAX:])

    # -------------------------------------------------------------- stats

    def item_stats(self) -> ItemStats:
        data = self._state.get(KEY_STATS) or {}
        return ItemStats(
            [int(value) for value in data.get("recent", [])],
            {str(k): int(v) for k, v in data.get("failures", {}).items()},
        )

    def poll_status(self) -> PollStatus:
        data = self._state.get(KEY_POLL) or {}
        return PollStatus(
            int(data.get("consecutive_failures", 0)),
            data.get("last_error"),
            _parse(data.get("last_ok_at")),
        )

    def record_poll_failure(self, kind: str, code: int | None, errmsg: str | None) -> int:
        """Count one failed poll; returns the number of failures in a row."""
        now = self._clock.now_utc()
        with self._state.transaction() as tx:
            data = dict(tx.get(KEY_POLL) or {})
            failures = int(data.get("consecutive_failures", 0)) + 1
            data["consecutive_failures"] = failures
            data["last_error"] = {
                "at": now.isoformat(),
                "kind": kind,
                "code": code,
                "errmsg": errmsg,
            }
            tx.put(KEY_POLL, data)
            return failures

    def record_poll_success(self) -> None:
        """A poll worked.

        Writes when recovering from failures, when never recorded and otherwise at most once a
        minute: the health check (R-OPS-003) treats a last success older than five minutes as a
        broken channel, so the stored time must stay fresher than that while polling works.
        """
        now = self._clock.now_utc()
        with self._state.transaction() as tx:
            data = dict(tx.get(KEY_POLL) or {})
            last_ok = _parse(data.get("last_ok_at"))
            recovering = int(data.get("consecutive_failures", 0)) > 0
            stale = last_ok is None or now - last_ok >= POLL_OK_WRITE_INTERVAL
            if not (recovering or stale):
                return
            data["consecutive_failures"] = 0
            data["last_ok_at"] = now.isoformat()
            tx.put(KEY_POLL, data)

    # ------------------------------------------------------------- commit

    def commit_batch(self, batch: BatchCommit) -> None:
        """Apply one poll batch atomically (section 10.5 of the protocol document)."""
        with self._state.transaction() as tx:
            if batch.seen_ids:
                seen = [str(item) for item in tx.get(KEY_SEEN, [])]
                known = set(seen)
                seen.extend(item for item in batch.seen_ids if item not in known)
                tx.put(KEY_SEEN, seen[-SEEN_IDS_MAX:])
            if batch.deliver:
                inbox = list(tx.get(KEY_INBOX, []))
                have = {item.get("id") for item in inbox}
                inbox.extend(m.to_dict() for m in batch.deliver if m.id not in have)
                tx.put(KEY_INBOX, inbox[-INBOX_MAX:])
            if batch.context_token is not None:
                tx.put(
                    KEY_CONTEXT,
                    {
                        "token": batch.context_token.token,
                        "received_at": batch.context_token.received_at.isoformat(),
                    },
                )
            if batch.last_inbound_at is not None:
                state = WindowState.from_dict(tx.get(KEY_WINDOW))
                tx.put(KEY_WINDOW, state.after_inbound(batch.last_inbound_at).to_dict())
            if batch.item_types or batch.failures:
                data = dict(tx.get(KEY_STATS) or {})
                recent = [int(v) for v in data.get("recent", [])] + batch.item_types
                data["recent"] = recent[-STATS_RECENT_MAX:]
                failures = {str(k): int(v) for k, v in data.get("failures", {}).items()}
                for reason, count in batch.failures.items():
                    failures[reason] = failures.get(reason, 0) + count
                data["failures"] = failures
                tx.put(KEY_STATS, data)
            self._add_quotes(tx, batch.quote_entries)
            if batch.candidate is not None and tx.get(KEY_PENDING_BIND) is None:
                tx.put(
                    KEY_PENDING_BIND,
                    {
                        "user_id": batch.candidate.user_id,
                        "seen_at": batch.candidate.seen_at.isoformat(),
                        "context_token": batch.candidate.context_token,
                        "matches_expected": batch.candidate.matches_expected,
                    },
                )
            if batch.new_cursor:
                tx.put(KEY_CURSOR, batch.new_cursor)
