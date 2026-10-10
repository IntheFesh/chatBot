"""CLI: ``twin ops drill network`` (R-EVAL-006, M4).  READ.

The network drill proves the alarm works: the cable is pulled for fifteen minutes during the
seven days of observation and the notification must appear within ten minutes of the connection
being lost.  The command changes nothing and breaks nothing; it prints the steps, and the state
of the bot right now so that the drill starts from a healthy one.
"""

from __future__ import annotations

import typer

from twin.clock import SystemClock
from twin.ops.process_model import CommandKind, command
from twin.ops.service import running

ops_app = typer.Typer(help="Operations rehearsals.", no_args_is_help=True)
drill_app = typer.Typer(help="Rehearse a failure to prove the alarm works.", no_args_is_help=True)
ops_app.add_typer(drill_app, name="drill")

STEPS = (
    "1. 确认机器人正在运行：twin service status 里 application (twin run) 是 running，"
    "twin health 里 channel 是 ok。",
    "2. 记下现在的时间，然后拔掉网线或关闭 Wi-Fi，让电脑完全断网 15 分钟（不要少于 12 分钟）。",
    "3. 断网后大约 5 到 6 分钟，电脑上应弹出 Windows 通知「微信长轮询中断」。"
    "邮件会在网络恢复后补发（断网时发不出去，程序会重试）。",
    "4. 15 分钟后恢复网络。几分钟内应收到「已恢复」通知，twin health 的 channel 回到 ok。",
    "5. 运行 twin eval stability --days 7：报告里要看到这次中断，以及它的告警延迟"
    "（从断网到通知，必须 ≤ 10 分钟）。观察期内至少做一次。",
)


@drill_app.command("network")
@command(CommandKind.READ, consent=False)
def drill_network() -> None:
    """Print the steps of the network drill."""
    typer.echo("断网演练（观察期内做一次）")
    typer.echo(f"现在是 {SystemClock().now_utc():%Y-%m-%d %H:%M} UTC。")
    from twin.services import get_cli_context

    supervisor, run = running(get_cli_context().paths().locks_dir)
    state = "在运行" if run else "没有在运行（先 twin service start）"
    typer.echo(f"twin run {state}；监督进程 {'在运行' if supervisor else '没有在运行'}。")
    typer.echo("")
    for step in STEPS:
        typer.echo(step)
