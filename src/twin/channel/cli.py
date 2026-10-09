"""``twin channel``: login, status, send-test, unbind, listen, echo-test and probe.

(R-CH-003, R-CH-007, R-CH-009; round 02 prompt sections C, D and E.)  ``twin chat --local`` is
here too: :func:`chat_command` is registered on the root command.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Annotated

import typer

from twin.channel.base import TEST_PREFIX, InboundMessage, OutboundResult, RecipientNotAllowed
from twin.channel.binding import confirm_unbind, mask_user_id
from twin.channel.chat import run_local_chat
from twin.channel.console import TyperPrompter
from twin.channel.echo import EchoHandler
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.ilink.flows import BIND_WAIT_S, run_login
from twin.channel.ilink.login import LoginError
from twin.channel.ilink.status import status_lines
from twin.channel.ilink.store import IlinkStore
from twin.channel.local import StreamInput, StreamOutput
from twin.channel.probe.cli import probe_app
from twin.channel.probe.store import ProbeStore
from twin.channel.state import ChannelStateStore
from twin.ops.instance_lock import LOCK_RUN, LOCK_SUPERVISOR
from twin.ops.logging import configure_logging, shutdown_logging
from twin.ops.process_model import CliError, CommandKind, ExitCode, app_is_running, command
from twin.services import Services, get_cli_context

channel_app = typer.Typer(
    help="WeChat channel: login, status, send-test, unbind, listen, echo-test, probe.",
    no_args_is_help=True,
)
channel_app.add_typer(probe_app, name="probe")

_HINTS = {
    "no_context_token": "send the bot a message from your phone first",
    "no_inbound_yet": "send the bot a message from your phone first",
    "window_elapsed": "the safe window has passed; send the bot a message from your phone first",
    "quota_exhausted": "the safe message count is used up; send the bot a message first",
    "session_expired": "the platform refused earlier; send the bot a message from your phone first",
    "needs_relogin": "run `twin channel login --force`",
    "not_logged_in": "run `twin channel login`",
}


def _store(services: Services) -> IlinkStore:
    return IlinkStore(ChannelStateStore(services.db), services.clock)


@channel_app.command("login")
@command(CommandKind.LIGHT)
def channel_login(
    force: Annotated[
        bool, typer.Option("--force", help="Scan a new code even if the login still works")
    ] = False,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Do not open the QR picture in the image viewer")
    ] = False,
    wait_minutes: Annotated[
        int, typer.Option(help="How long to wait for your first message when binding")
    ] = int(BIND_WAIT_S // 60),
) -> None:
    """Scan the QR code with WeChat, then bind the account that writes to the bot first."""
    services = get_cli_context().services()
    try:
        outcome = asyncio.run(
            run_login(
                services,
                TyperPrompter(),
                force=force,
                open_viewer=not no_open,
                app_running=app_is_running(services),
                bind_wait_s=float(wait_minutes * 60),
            )
        )
    except LoginError as exc:
        raise CliError(str(exc)) from None
    if not outcome.bound:
        raise CliError("logged in, but no user is bound yet")


@channel_app.command("status")
@command(CommandKind.READ)
def channel_status() -> None:
    """Login, binding, window and quota, and the item types seen recently (no content)."""
    services = get_cli_context().services()
    config = services.settings.channel
    for line in status_lines(
        _store(services),
        services.clock,
        window_h=config.proactive_window_safe_h,
        quota=config.outbound_quota_safe,
    ):
        typer.echo(line)
    typer.echo(f"application: {'running' if app_is_running(services) else 'not running'}")
    plan = ProbeStore(services.db, services.clock).load()
    if plan is not None:
        notice = f" - {plan.notice}" if plan.notice else ""
        typer.echo(f"probe: {plan.run_id} {plan.status.value}{notice} (twin channel probe status)")


def describe_failure(result: OutboundResult) -> str:
    parts = [f"{result.kind.value}: {result.reason}"]
    if result.code is not None:
        parts.append(f"code {result.code}")
    if result.errmsg:
        parts.append(f"message {result.errmsg!r}")
    hint = _HINTS.get(result.reason)
    return ", ".join(parts) + (f" ({hint})" if hint else "")


@channel_app.command("send-test")
@command(CommandKind.LIGHT)
def channel_send_test(
    text: Annotated[str, typer.Argument(help="Text to send; '[测试]' is put in front")],
) -> None:
    """Send one test message to the bound user (and nobody else)."""
    services = get_cli_context().services()
    message = text if text.startswith(TEST_PREFIX) else TEST_PREFIX + text

    async def go() -> OutboundResult:
        channel = IlinkChannel.from_services(services, poll=False)
        await channel.start()
        try:
            return await channel.send_text(message)
        finally:
            await channel.stop()

    try:
        result = asyncio.run(go())
    except RecipientNotAllowed:
        raise CliError("no user is bound yet: run `twin channel login`") from None
    if not result.ok:
        raise CliError("not sent - " + describe_failure(result))
    typer.echo(
        "sent to the bound user"
        + (f" (message id {result.message_id})" if result.message_id else "")
    )
    typer.echo("Check your phone: the message should read " + message)


@channel_app.command("unbind")
@command(CommandKind.LIGHT)
def channel_unbind() -> None:
    """Stop talking to the bound user (asks twice)."""
    services = get_cli_context().services()
    store = _store(services)
    bound = store.bound_user()
    if bound is None:
        typer.echo("nobody is bound")
        return
    if not confirm_unbind(TyperPrompter(), bound.user_id):
        typer.echo("nothing changed")
        raise typer.Exit(1)
    store.unbind()
    typer.echo(f"unbound {mask_user_id(bound.user_id)}; run `twin channel login` to bind again")


def describe_inbound(message: InboundMessage) -> str:
    """One line about a received message: its form, never its content."""
    parts = [
        message.at.strftime("%H:%M:%SZ"),
        f"kind={message.kind.value}",
        f"item_type={message.item_type}",
    ]
    if message.text is not None:
        parts.append(f"text_chars={len(message.text)}")
    if message.media_ref is not None:
        parts.append(f"media={message.media_ref.kind.value}:{message.media_ref.size}B")
    if message.quote is not None:
        parts.append("quote=resolved" if message.quote.resolved else "quote=unresolved")
    if message.flags:
        parts.append("flags=" + ",".join(sorted(message.flags)))
    return "  ".join(parts)


@channel_app.command("listen")
@command(CommandKind.EXCLUSIVE, acquires=(LOCK_RUN,), tolerates=(LOCK_SUPERVISOR,))
def channel_listen(
    count: Annotated[
        int, typer.Option(help="Stop after this many messages (0 = until Ctrl+C)")
    ] = 0,
) -> None:
    """Receive the bound user's messages and print what kind they are (never what they say).

    A diagnostic for the first real-device run: send the bot text, a picture, a voice message,
    a video, a file, a quote and a sticker, and see how each arrives.  Nothing is sent back.
    It takes the same lock as `twin run`, so only one of them polls at a time.
    """
    context = get_cli_context()
    services = context.services()
    configure_logging(
        services.paths.logs_dir, level=context.log_level, role="cli", console_level=logging.WARNING
    )

    async def go() -> int:
        channel = IlinkChannel.from_services(services, poll=True)
        await channel.start()
        seen = 0
        try:
            async for message in channel.incoming():
                seen += 1
                typer.echo(f"#{seen}  {describe_inbound(message)}")
                if count and seen >= count:
                    break
        finally:
            await channel.stop()
        return seen

    try:
        if services.settings.channel.kind != "ilink":
            raise CliError("channel.kind is not 'ilink'")
        if _store(services).bound_user() is None:
            raise CliError("nobody is bound yet: run `twin channel login`")
        typer.echo("Listening for messages (Ctrl+C to stop). Contents are never shown.")
        asyncio.run(go())
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        shutdown_logging()


class TyperOutput:
    """Lines to the terminal through ``typer`` (so tests can capture them)."""

    def write_line(self, text: str) -> None:
        typer.echo(text)


@channel_app.command("echo-test")
@command(CommandKind.EXCLUSIVE, acquires=(LOCK_RUN,), tolerates=(LOCK_SUPERVISOR,))
def channel_echo_test(
    local: Annotated[
        bool,
        typer.Option("--local", help="Use the terminal channel instead of WeChat"),
    ] = False,
    count: Annotated[
        int, typer.Option(help="Stop after this many messages (0 = until Ctrl+C)")
    ] = 0,
) -> None:
    """Send back what you write, prefixed '[测试]' - a check of the channel itself.

    Not a reply path of the product.  Over WeChat it answers only the bound user, inside the
    safe window and count; with --local it uses the terminal.
    """
    context = get_cli_context()
    services = context.services()
    configure_logging(
        services.paths.logs_dir, level=context.log_level, role="cli", console_level=logging.WARNING
    )
    try:
        if local:
            output = StreamOutput(sys.stdout)
            asyncio.run(
                run_local_chat(
                    services,
                    input=StreamInput(sys.stdin),
                    output=output,
                    handler=EchoHandler(output),
                    limit=count,
                )
            )
            return
        if services.settings.channel.kind != "ilink":
            raise CliError("channel.kind is not 'ilink'; use --local for the terminal channel")
        if _store(services).bound_user() is None:
            raise CliError("nobody is bound yet: run `twin channel login`")
        typer.echo("Echoing your messages back (Ctrl+C to stop). Contents are never logged.")
        asyncio.run(_echo_over_wechat(services, count))
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        shutdown_logging()


async def _echo_over_wechat(services: Services, count: int) -> None:
    channel = IlinkChannel.from_services(services, poll=True)
    handler = EchoHandler(TyperOutput())
    await channel.start()
    seen = 0
    try:
        async for message in channel.incoming():
            seen += 1
            typer.echo(f"#{seen}  {describe_inbound(message)}")
            await handler(message, channel)
            if count and seen >= count:
                break
    finally:
        await channel.stop()


@command(CommandKind.EXCLUSIVE, acquires=(LOCK_RUN,), tolerates=(LOCK_SUPERVISOR,))
def chat_command(
    local: Annotated[
        bool,
        typer.Option("--local", help="Talk through the terminal (the only mode so far)"),
    ] = False,
    window_h: Annotated[
        float | None,
        typer.Option("--window-h", help="Simulated window in hours (default: channel config)"),
    ] = None,
    quota: Annotated[
        int | None,
        typer.Option("--quota", help="Simulated message count (default: channel config)"),
    ] = None,
) -> None:
    """Chat in the terminal with the application running (the engine joins in round 09).

    Starts the application with only the local console channel.  Until the persona engine is
    connected nothing answers; the command says so.
    """
    if not local:
        raise CliError(
            "choose --local: the WeChat conversation is served by `twin run`",
            ExitCode.USAGE,
        )
    context = get_cli_context()
    services = context.services()
    configure_logging(
        services.paths.logs_dir, level=context.log_level, role="cli", console_level=logging.WARNING
    )
    try:
        asyncio.run(
            run_local_chat(
                services,
                input=StreamInput(sys.stdin),
                output=StreamOutput(sys.stdout),
                window_h=window_h,
                quota=quota,
            )
        )
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        shutdown_logging()
