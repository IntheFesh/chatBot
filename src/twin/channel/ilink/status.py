"""The text of ``twin channel status`` (R-CH-007, R-CH-008, protocol document section 10).

Only identifiers, counts, numbers and times are shown: never message content.  Item types
appear as their protocol numbers (1 text, 2 image, 3 voice, 4 file, 5 video; anything else is
a form the parser does not know yet).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from twin.channel.base import AuthState
from twin.channel.binding import mask_user_id
from twin.channel.ilink.store import IlinkStore
from twin.channel.window import SessionWindow
from twin.clock import Clock

KNOWN_ITEM_TYPES = {1: "text", 2: "image", 3: "voice", 4: "file", 5: "video"}


def _age(now: datetime, then: datetime | None) -> str:
    if then is None:
        return "never"
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 90:
        return f"{seconds} s ago"
    if seconds < 5400:
        return f"{seconds // 60} min ago"
    return f"{seconds / 3600:.1f} h ago"


def _hours(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    sign = "-" if total < 0 else ""
    total = abs(total)
    return f"{sign}{total // 3600}h{(total % 3600) // 60:02d}m"


def status_lines(store: IlinkStore, clock: Clock, *, window_h: float, quota: int) -> list[str]:
    """One line per fact, in the order a person debugging the channel wants them."""
    now = clock.now_utc()
    lines: list[str] = []
    credentials = store.credentials()
    auth = store.auth_record()
    if credentials is None:
        lines.append("login: NOT LOGGED IN (run `twin channel login`)")
    elif auth.state is AuthState.NEEDS_RELOGIN:
        since = auth.since.isoformat() if auth.since else "?"
        lines.append(
            f"login: NEEDS RE-LOGIN since {since} (code {auth.code}); "
            "run `twin channel login --force`"
        )
    else:
        host = credentials.api_base_url.removeprefix("https://")
        bot = mask_user_id(credentials.ilink_bot_id)
        lines.append(f"login: logged in as {bot} (API host {host})")
        saved = datetime.fromisoformat(credentials.saved_at)
        lines.append(f"login saved: {_age(now, saved)}")

    bound = store.bound_user()
    if bound is not None:
        lines.append(
            f"bound user: {mask_user_id(bound.user_id)} (since {bound.bound_at.isoformat()})"
        )
    else:
        pending = store.pending_binding()
        if pending is None:
            lines.append("bound user: NOBODY (log in, then send the bot a message)")
        else:
            verdict = {True: "matches", False: "DOES NOT MATCH", None: "not comparable to"}[
                pending.matches_expected
            ]
            lines.append(
                f"bound user: NOBODY; waiting for confirmation of {mask_user_id(pending.user_id)} "
                f"({verdict} the account that scanned the code)"
            )

    window = SessionWindow(window_h=window_h, quota=quota, state=store.window_state())
    context = store.context_token()
    lines.append(
        "context token: "
        + (f"present (received {_age(now, context.received_at)})" if context else "MISSING")
    )
    lines.append(f"last inbound message: {_age(now, window.last_inbound_at)}")
    remaining = window.window_remaining(now)
    lines.append(
        f"window: safe limit {window_h:g} h, "
        + ("no inbound yet" if remaining is None else f"{_hours(remaining)} left")
        + f"; expired by the platform: {'YES' if window.expired else 'no'}"
    )
    lines.append(
        f"quota: sent {window.outbound_since_inbound} since the last inbound, "
        f"{window.remaining_quota()} of {quota} left (safe limit)"
    )
    error = window.state.last_error
    if error:
        lines.append(
            f"last send problem: {error.get('kind')} code={error.get('code')} "
            f"at {error.get('at')} msg={error.get('errmsg') or '-'}"
        )

    poll = store.poll_status()
    lines.append(
        f"polling: {poll.consecutive_failures} failure(s) in a row; "
        f"last success {_age(now, poll.last_ok_at)}"
    )
    if poll.last_error:
        err = poll.last_error
        lines.append(
            f"last poll problem: {err.get('kind')} code={err.get('code')} at {err.get('at')}"
        )
    lines.append(f"unread inbox: {len(store.inbox())} message(s)")

    stats = store.item_stats()
    counts = stats.counts()
    if counts:
        shown = ", ".join(
            f"type {number} ({KNOWN_ITEM_TYPES.get(number, 'UNKNOWN')}): {count}"
            for number, count in counts.items()
        )
        lines.append(f"recent inbound item types ({len(stats.recent_types)} items): {shown}")
    else:
        lines.append("recent inbound item types: none yet")
    if stats.failures:
        shown = ", ".join(f"{name}: {count}" for name, count in sorted(stats.failures.items()))
        lines.append(f"parse and handling counters: {shown}")
    return lines
