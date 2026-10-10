"""CLI: ``twin purge --all`` and ``twin purge --training-only`` (R-OPS-008, R-PRIV-005).

EXCLUSIVE.  The command

1. refuses to run while the application is (the lock check of every exclusive command; the hint
   says ``twin service stop``);
2. lists what it would delete, with numbers;
3. asks for the confirmation phrase **typed in**: ``删除她的全部数据`` for ``--all``,
   ``删除训练数据`` for ``--training-only``.  There is no option that gives the phrase or skips the
   question, and anything else than the exact phrase deletes nothing;
4. checks once more that the application is not running, deletes, and prints a report of counts.

See :mod:`twin.ops.purge` for what each scope removes and what it leaves alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from twin.ops.instance_lock import ALL_LOCKS, locks_held_elsewhere
from twin.ops.process_model import CliError, CommandKind, ExitCode, command, stop_hint
from twin.ops.purge import (
    CONFIRM_ALL,
    CONFIRM_TRAINING,
    PurgePlan,
    PurgeReport,
    plan_all,
    plan_training,
    purge_all,
    purge_training,
)
from twin.services import get_cli_context
from twin.storage.keystore import KeyStore, KeyStoreError
from twin.training.runs import uncleaned_runs


def _confirm(plan: PurgePlan, phrase: str) -> None:
    typer.echo("将要删除：")
    for line in plan.lines():
        typer.echo(f"  - {line}")
    typer.echo("这些数据删除后无法恢复。")
    answer = typer.prompt(f"确认请输入「{phrase}」", default="", show_default=False)
    if answer.strip() != phrase:
        typer.echo("确认短语不符，什么都没有删除。")
        raise typer.Exit(1)


def _still_stopped(locks_dir: Path) -> None:
    busy = locks_held_elsewhere(locks_dir, ALL_LOCKS)
    if busy:
        raise CliError(f"the '{busy[0]}' instance is running now; {stop_hint()}", ExitCode.BUSY)


def _show(report: PurgeReport) -> None:
    typer.echo("已删除（只列数量）：")
    for line in report.lines():
        typer.echo(f"  - {line}")


def _fail_if_left_over(report: PurgeReport) -> None:
    """A file that another program holds open cannot be deleted (Windows): say so, exit 1."""
    if report.failed:
        raise CliError(
            "上面列出的文件没能删除，很可能被别的程序占用（数据库查看器、杀毒软件、资源管理器预览）；"
            "关掉它们之后再运行一次同一条命令。密钥已经删除，剩下的文件已无法解密。"
            if report.scope == "all"
            else "上面列出的文件没能删除，很可能被别的程序占用；关掉它们之后再运行一次同一条命令。"
        )


@command(CommandKind.EXCLUSIVE, consent=False)
def purge_command(
    all_data: Annotated[
        bool, typer.Option("--all", help="Delete everything about her (asks for a phrase)")
    ] = False,
    training_only: Annotated[
        bool,
        typer.Option("--training-only", help="Delete only training sets, bundles and models"),
    ] = False,
) -> None:
    """Delete her data for good: database, media, vectors, backups, training, models, keys."""
    if all_data == training_only:
        raise CliError("use exactly one of --all or --training-only", ExitCode.USAGE)
    context = get_cli_context()
    settings = context.settings()
    paths = context.paths()
    mirror = (
        Path(settings.ops.backup_mirror_dir).expanduser()
        if settings.ops.backup_mirror_dir
        else None
    )
    keystore = KeyStore(context.secret_store())
    if all_data:
        try:
            keys = len(keystore.load().key_ids) if keystore.exists() else 0
        except KeyStoreError:
            keys = 0  # a damaged key store is deleted all the same
        _confirm(plan_all(paths, mirror, keys), CONFIRM_ALL)
        _still_stopped(paths.locks_dir)
        context.reset()  # nothing of this process may keep the database or the log files open
        report = purge_all(paths, mirror, keystore)
        _show(report)
        _fail_if_left_over(report)
        typer.echo("备份和任何残留的副本现在都无法解密。外部的导出目录（paths.export_dir）没有动，")
        typer.echo("需要的话请自己删除；AutoDL 上如果还有实例，请到控制台确认已释放。")
        return
    services = context.services()
    runs = len(uncleaned_runs(services.db))
    _confirm(plan_training(paths, runs), CONFIRM_TRAINING)
    _still_stopped(paths.locks_dir)
    report = purge_training(paths, runs, services.db)
    _show(report)
    _fail_if_left_over(report)
    if runs:
        typer.echo(f"有 {runs} 次远程训练没有记录清理完成：请到 AutoDL 控制台确认实例已释放。")
    else:
        typer.echo("提醒：到 AutoDL 控制台确认实例已经释放。")
