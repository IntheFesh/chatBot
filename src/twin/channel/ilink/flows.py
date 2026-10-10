"""The interactive login and binding flows behind ``twin channel login`` (R-CH-003, R-CH-007).

Both are plain async functions that talk to the person through a :class:`Prompter`, so tests
drive them with scripted answers.  The flow:

1. QR login (skipped when a working login exists and ``force`` is not given);
2. the account that scanned must be the bound account if one exists (otherwise the login is
   refused and nothing is saved);
3. if nobody is bound yet: wait for the first message, show the masked sender id and bind
   after explicit confirmation.  While the application runs, its poller records the sender and
   this command only watches the database; otherwise this command polls itself.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from twin.channel.base import AuthState
from twin.channel.binding import confirm_binding, mask_user_id
from twin.channel.console import Prompter
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.ilink.http import IlinkHttp
from twin.channel.ilink.login import IlinkLogin, LoginError, LoginResult, LoginUI
from twin.channel.ilink.qr import open_in_viewer, remove_old_pngs, render_terminal, save_png
from twin.channel.ilink.store import Credentials, IlinkStore
from twin.channel.state import ChannelStateStore

if TYPE_CHECKING:
    from twin.services import Services

BIND_WAIT_S = 600.0
BIND_POLL_S = 2.0


class ConsoleLoginUI:
    """Shows the QR code as a PNG in the viewer and as characters in the terminal."""

    def __init__(self, prompter: Prompter, tmp_dir: Path, *, open_viewer: bool = True) -> None:
        self._prompter = prompter
        self._tmp_dir = tmp_dir
        self._open_viewer = open_viewer

    def show_qr(self, content: str, *, number: int, total: int) -> None:
        remove_old_pngs(self._tmp_dir)
        say = self._prompter.say
        say(f"\nScan this code with WeChat (code {number} of {total}):\n")
        say(render_terminal(content))
        try:
            png = save_png(content, self._tmp_dir)
        except OSError as exc:
            say(f"(could not save the picture: {exc})")
        else:
            opened = self._open_viewer and open_in_viewer(png)
            say(f"\nThe same code is saved as {png}" + (" and was opened." if opened else "."))
        say(f"If the code does not scan, open this link in WeChat: {content}")

    def ask_verify_code(self, *, previous_was_wrong: bool) -> str:
        if previous_was_wrong:
            self._prompter.say("That number was not accepted.")
        return self._prompter.ask("Enter the number shown on your phone")

    def info(self, message: str) -> None:
        self._prompter.say(message)

    def cleanup(self) -> None:
        remove_old_pngs(self._tmp_dir)


@dataclass(frozen=True)
class LoginOutcome:
    logged_in_now: bool
    bound: bool


async def login_with_ui(
    services: Services, ui: LoginUI, *, http_client: httpx.AsyncClient | None = None
) -> LoginResult:
    """Scan-to-login with ``ui`` and store the credentials (the shared core of every login).

    The account that scanned must be the bound one, if there is one: otherwise nothing is saved
    and :class:`LoginError` says so.  ``twin channel login`` and the recovery window of the
    running application (:mod:`twin.ops.login_recovery`) both come through here.
    """
    store = IlinkStore(ChannelStateStore(services.db), services.clock)
    http = IlinkHttp(http_client)
    try:
        result = await IlinkLogin(http, services.clock, ui).run()
    finally:
        await http.aclose()
    bound = await asyncio.to_thread(store.bound_user)
    if bound and result.ilink_user_id and result.ilink_user_id != bound.user_id:
        raise LoginError(
            f"the account that scanned ({mask_user_id(result.ilink_user_id)}) is not the "
            f"bound account ({mask_user_id(bound.user_id)}); nothing was changed. "
            "Run `twin channel unbind` first if you really want to switch accounts."
        )
    await asyncio.to_thread(
        store.save_credentials,
        Credentials(
            bot_token=result.bot_token,
            ilink_bot_id=result.ilink_bot_id,
            ilink_user_id=result.ilink_user_id,
            api_base_url=result.api_base_url,
            saved_at=services.clock.now_utc().isoformat(),
        ),
    )
    return result


async def run_login(
    services: Services,
    prompter: Prompter,
    *,
    force: bool = False,
    open_viewer: bool = True,
    app_running: bool = False,
    bind_wait_s: float = BIND_WAIT_S,
    http_client: httpx.AsyncClient | None = None,
) -> LoginOutcome:
    store = IlinkStore(ChannelStateStore(services.db), services.clock)
    credentials = await asyncio.to_thread(store.credentials)
    auth = await asyncio.to_thread(store.auth_record)
    logged_in_now = False
    if credentials is None or auth.state is not AuthState.OK or force:
        ui = ConsoleLoginUI(prompter, services.paths.tmp_dir, open_viewer=open_viewer)
        try:
            await login_with_ui(services, ui, http_client=http_client)
        finally:
            ui.cleanup()
        logged_in_now = True
        prompter.say("Logged in. The credentials are stored encrypted.")
    else:
        prompter.say("Already logged in (use --force to scan again).")
    bound_user = await asyncio.to_thread(store.bound_user)
    if bound_user is not None:
        prompter.say(f"Bound to {mask_user_id(bound_user.user_id)}.")
        return LoginOutcome(logged_in_now, True)
    bound_now = await run_binding(
        services,
        prompter,
        app_running=app_running,
        wait_s=bind_wait_s,
        http_client=http_client,
    )
    return LoginOutcome(logged_in_now, bound_now)


async def run_binding(
    services: Services,
    prompter: Prompter,
    *,
    app_running: bool,
    wait_s: float = BIND_WAIT_S,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Wait for the first message, confirm the sender and bind it.  ``True`` if bound."""
    clock = services.clock
    store = IlinkStore(ChannelStateStore(services.db), clock)
    candidate = await asyncio.to_thread(store.pending_binding)
    if candidate is None:
        prompter.say(
            "Now send any message to the ClawBot chat on your phone. "
            f"Waiting up to {int(wait_s // 60)} minutes..."
        )
        channel: IlinkChannel | None = None
        if not app_running:
            channel = IlinkChannel.from_services(services, poll=True, http_client=http_client)
            await channel.start()
        try:
            deadline = clock.monotonic() + wait_s
            while candidate is None:
                auth = await asyncio.to_thread(store.auth_record)
                if auth.state is AuthState.NEEDS_RELOGIN:
                    prompter.say("The login stopped working; run `twin channel login --force`.")
                    return False
                if clock.monotonic() >= deadline:
                    prompter.say("No message arrived. Run `twin channel login` again later.")
                    return False
                await clock.sleep(BIND_POLL_S)
                candidate = await asyncio.to_thread(store.pending_binding)
        finally:
            if channel is not None:
                await channel.stop()
    agreed = confirm_binding(
        prompter, candidate.user_id, matches_expected=candidate.matches_expected
    )
    if not agreed:
        await asyncio.to_thread(store.clear_pending_binding)
        prompter.say("Not bound. Run `twin channel login` to try again.")
        return False
    await asyncio.to_thread(store.bind, candidate.user_id, context_token=candidate.context_token)
    prompter.say(f"Bound to {mask_user_id(candidate.user_id)}. The bot talks only to this account.")
    return True
