"""CLI: ``twin setup`` - the first-run wizard (R-OPS-001, R-SAFE-001, R-SCOPE-003).

Asks, one thing at a time, for what a new installation needs and writes it where it belongs:

========================  =====================================================================
consent                   she knows and agrees; the date goes to ``consent.confirmed_at``
DeepSeek API key          credential store (``deepseek_api_key``), never a file
SMTP account              host, port, address and recipient to ``ops.smtp``; the password to the
                          credential store (``smtp_password``); a test mail if wanted
target conversation       the WeChat id to imitate (``target.username``); empty: chosen at the
                          first import
time zone                 ``time.bot_timezone``
emergency contact         ``safety.emergency_contact``; the wizard says what the mail contains
========================  =====================================================================

Answers that are already in place are shown and kept unless the person changes them, so the wizard
can be run again.  The configuration file is changed with
:func:`~twin.config.writer.set_section_values` (comments and layout kept, validated before it is
kept).  The file only seeds the runtime settings of a new database (R-CFG-003), so once the
database exists the target and the time zone the wizard shows are the ones in force and a changed
answer is written there too (D-830): the time zone through the planner's ``switch_timezone`` - a
switch with its ``timezone_history`` row, as ``twin timezone set`` and ``/时区`` make it - and the
target as the runtime setting, unless the database already holds another conversation (one target
per database, R-IMP-003, D-695).  LIGHT.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import typer

from twin.clock import get_clock
from twin.config.loader import ConfigError, default_config_path
from twin.config.runtime import BOT_TIMEZONE, TARGET_USERNAME
from twin.config.secrets import SecretStore
from twin.config.settings import Settings
from twin.config.writer import set_section_values
from twin.engine.safety.notifier import BODY_TEMPLATE, SUBJECT
from twin.ingest.conversations import holds_another_conversation
from twin.ops.mail import SMTP_PASSWORD_SECRET, MailError, OutgoingMail, SmtpMailer
from twin.ops.process_model import CliError, CommandKind, command
from twin.schedule.service import schedule_kit
from twin.services import CliContext, Services, get_cli_context
from twin.storage.migrate import SchemaState, schema_status

DEEPSEEK_SECRET = "deepseek_api_key"  # noqa: S105 - the credential store entry name, not a secret


def _config_file(context: CliContext) -> Path:
    return context.config_path or default_config_path(context.paths().root)


def _ask(text: str, default: str | None = None) -> str:
    return str(
        typer.prompt(
            text, default=default if default is not None else "", show_default=bool(default)
        )
    ).strip()


def _consent(settings: Settings, config: Path) -> None:
    typer.echo("1/6 她的同意")
    typer.echo("这个机器人模仿你女朋友的说话方式，只有在她知情并同意的情况下才能使用。")
    if not typer.confirm("她已经知道并同意了吗？", default=False):
        raise CliError("没有她的同意不能继续。")
    # without an earlier answer the proposal is today's date where the user lives (not in UTC)
    today = get_clock().now_utc().astimezone(ZoneInfo(settings.time.bot_timezone)).date()
    current = settings.consent.confirmed_at or today.isoformat()
    answer = _ask("她同意的日期（YYYY-MM-DD）", current)
    try:
        date.fromisoformat(answer)
    except ValueError:
        raise CliError(f"{answer!r} 不是日期，请用 YYYY-MM-DD") from None
    set_section_values(config, "consent", {"confirmed_at": answer})


def _deepseek_key(store: SecretStore) -> None:
    typer.echo("2/6 DeepSeek API Key")
    if store.exists(DEEPSEEK_SECRET) and not typer.confirm(
        "已经有 Key 了，要换成新的吗？", default=False
    ):
        return
    key = typer.prompt(
        "粘贴 DeepSeek API Key（输入时不显示）", hide_input=True, default="", show_default=False
    )
    if key.strip():
        store.set(DEEPSEEK_SECRET, key.strip())
        typer.echo("已存入凭据管理器。")
    else:
        typer.echo("先跳过；之后用 twin secrets set deepseek_api_key。")


def _smtp(settings: Settings, config: Path, store: SecretStore) -> None:
    typer.echo("3/6 告警邮件（SMTP）")
    typer.echo("登录失效、费用超限、备份失败等会发邮件给你；邮件里没有任何聊天内容。")
    if not typer.confirm("现在设置发件邮箱吗？", default=True):
        return
    current = settings.ops.smtp
    host = _ask("SMTP 服务器（例如 smtp.gmail.com）", current.host)
    port = int(_ask("端口（465 用 SSL，587 用 STARTTLS）", str(current.port)))
    user = _ask("发件邮箱地址（也是登录名）", current.user)
    recipient = _ask("收件地址", current.to or user)
    password = typer.prompt(
        "邮箱的应用专用密码（输入时不显示；留空保持不变）",
        hide_input=True,
        default="",
        show_default=False,
    )
    if password.strip():
        store.set(SMTP_PASSWORD_SECRET, password.strip())
    smtp = {**current.model_dump(), "host": host, "port": port, "user": user, "to": recipient}
    set_section_values(config, "ops", {"smtp": smtp})
    mailer = SmtpMailer(
        settings.ops.smtp.model_copy(update=smtp), lambda: store.get(SMTP_PASSWORD_SECRET)
    )
    if mailer.configured and typer.confirm("发一封测试邮件吗？", default=True):
        try:
            mailer.send(
                OutgoingMail(
                    recipient, "[wechat-twin] 测试邮件", "能收到这封邮件，说明告警邮件设置好了。"
                )
            )
            typer.echo("测试邮件已发出，请查收。")
        except MailError as exc:
            typer.echo(
                f"测试邮件没发出去（{exc.code}）；检查服务器、端口、密码后再运行 twin setup。"
            )


def _database(context: CliContext) -> Services | None:
    """The services of a database at the current schema, else ``None`` (nothing in force yet)."""
    if schema_status(context.paths().db_path).state is not SchemaState.CURRENT:
        return None
    return context.services()


def _target(settings: Settings, config: Path, services: Services | None) -> None:
    typer.echo("4/6 要模仿的会话")
    typer.echo("她在微信里的 wxid；留空则在第一次导入时从会话列表里选。")
    current = services.runtime.get(TARGET_USERNAME) if services else settings.target.username
    answer = _ask("wxid", current)
    if not answer:
        return
    if services is not None and answer != current:
        with services.db.session() as session:
            other = holds_another_conversation(session, answer)
        if other:
            typer.echo(
                "数据库里已经导入了另一个会话的聊天记录，而一个数据库只放一个目标会话"
                "（换人会把别人的话混进她的画像、例子和训练集）。目标会话保持不变；"
                "要换成别人，先用 `twin purge` 删掉已导入的数据，再导入她的导出。"
            )
            return
        services.runtime.set(TARGET_USERNAME, answer, by="setup")
    set_section_values(config, "target", {"username": answer})


def _timezone(settings: Settings, config: Path, services: Services | None) -> None:
    typer.echo("5/6 机器人所在的时区")
    current = services.runtime.get(BOT_TIMEZONE) if services else settings.time.bot_timezone
    while True:
        answer = _ask("IANA 时区（美国中部 America/Chicago，中国 Asia/Shanghai）", current)
        try:
            ZoneInfo(answer)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            typer.echo(f"不认识时区 {answer!r}，再试一次。")
            continue
        break
    if services is not None and answer != current:
        # a switch, written to the history: the bot's past stays dated in the old zone (D-830)
        outcome = schedule_kit(services).planner.switch_timezone(answer, source="cli")
        typer.echo(f"已切换时区：{outcome.old_timezone} -> {outcome.new_timezone}（记入时区历史）")
    set_section_values(config, "time", {"bot_timezone": answer})


def _emergency(settings: Settings, config: Path) -> None:
    typer.echo("6/6 紧急联系人提醒（默认关闭）")
    typer.echo(
        "如果机器人判断你可能处在危机中，它会跳出角色给你求助渠道。你也可以让它另外通知一个人。"
    )
    typer.echo("发出去的只是一封固定模板的邮件，里面只有时间，没有任何聊天内容：")
    typer.echo(f"  主题：{SUBJECT}")
    for line in BODY_TEMPLATE.splitlines():
        typer.echo(f"  {line}")
    enabled = typer.confirm("开启吗？", default=settings.safety.emergency_contact.enabled)
    email = ""
    if enabled:
        email = _ask("联系人的邮箱", settings.safety.emergency_contact.email)
        if not email:
            typer.echo("没有填邮箱，保持关闭。")
            enabled = False
    set_section_values(
        config,
        "safety",
        {"emergency_contact": {"enabled": enabled, "email": email or None}},
    )


@command(CommandKind.LIGHT, consent=False)
def setup_command() -> None:
    """First-run wizard: consent, DeepSeek key, e-mail alerts, conversation, time zone."""
    context = get_cli_context()
    config = _config_file(context)
    store = context.secret_store()
    try:
        settings = context.settings()
        _consent(settings, config)
        context.reset()
        settings = context.settings()
        _deepseek_key(store)
        _smtp(settings, config, store)
        context.reset()
        settings = context.settings()
        services = _database(context)
        _target(settings, config, services)
        _timezone(settings, config, services)
        context.reset()
        settings = context.settings()
        _emergency(settings, config)
    except ConfigError as exc:
        raise CliError(str(exc)) from exc
    context.reset()
    typer.echo("")
    typer.echo(f"设置写在 {config}；密钥在凭据管理器里。接下来：")
    typer.echo("  twin db upgrade        建立数据库")
    typer.echo("  twin channel login     扫码登录微信并绑定你的账号")
    typer.echo("  twin doctor            检查环境")
    typer.echo("  twin service install   注册开机自启的计划任务（Windows）")
