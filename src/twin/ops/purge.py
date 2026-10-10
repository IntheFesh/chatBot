"""Deleting everything about her (``twin purge``, R-OPS-008, R-PRIV-005).

``--all`` removes, in this order, and reports only **counts**:

========================  =========================================================
database                  ``twin.db`` and its journal files (the rows are counted first)
media                     the encrypted files in ``data/media``
vector index              ``data/vectors``
backups                   every archive, index file, media-pool file and pre-restore
                          backup in ``data/backups`` - and in ``ops.backup_mirror_dir``
training                  ``data/training`` (training sets, bundles, tokenizer cache)
local models              ``data/models`` except the embedding model (a public download)
reports and logs          ``data/reports``, ``data/logs`` (closed first), ``data/tmp``
credential store          every database key - the retired ones too - so that any copy of
                          the data or of a backup that is left anywhere cannot be decrypted:
                          the backup keys are derived from these keys, which is the
                          "crypto-shredding" of R-OPS-008
========================  =========================================================

What it does **not** touch: the export folder the data was imported from (``paths.export_dir`` -
the user's own files; delete it yourself), the configuration file and the secrets that are not
about her (the DeepSeek key, the SMTP password).

``--training-only`` removes the training sets and bundles, the local model files and the registry
rows that point at them, and lists the runs whose data may still be on a rented machine so that the
instance can be checked and released in the AutoDL console (R-PRIV-003).

The confirmation phrase is asked by the command and cannot be given on the command line.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import delete

from twin.config.loader import DataPaths
from twin.ops.backup.layout import (
    ARCHIVE_SUFFIX,
    POOL_DIRNAME,
    PRE_RESTORE_PREFIX,
    SIDECAR_SUFFIX,
)
from twin.ops.logging import shutdown_logging
from twin.storage.db import Database
from twin.storage.keystore import KeyStore
from twin.storage.training_models import ModelRegistryEntry

CONFIRM_ALL = "删除她的全部数据"
CONFIRM_TRAINING = "删除训练数据"
TRAINING_DIRNAME = "training"
EMBEDDINGS_DIRNAME = "embeddings"


@dataclass(frozen=True)
class PurgeItem:
    """One category: what it is called and how many of what."""

    category: str
    count: int
    unit: str
    note: str = ""

    def line(self) -> str:
        text = f"{self.category}：{self.count} {self.unit}"
        return f"{text}（{self.note}）" if self.note else text


@dataclass
class PurgePlan:
    """What a purge would delete."""

    scope: str  # "all" or "training"
    items: list[PurgeItem] = field(default_factory=list)

    def lines(self) -> list[str]:
        return [item.line() for item in self.items]


@dataclass
class PurgeReport:
    """What a purge deleted: counts only.

    ``failed`` names (file or folder names, never contents) what could not be deleted - on
    Windows a file that another program has open cannot be removed.  The purge carries on with
    everything else; the command reports them and ends with exit code 1.
    """

    scope: str
    items: list[PurgeItem] = field(default_factory=list)
    keys_deleted: int = 0
    uncleaned_runs: int = 0
    failed: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = [item.line() for item in self.items]
        if self.scope == "all":
            lines.append(f"凭据管理器里的数据库密钥：删除 {self.keys_deleted} 个")
        if self.failed:
            lines.append(f"没能删除：{len(self.failed)} 项（{'、'.join(self.failed)}）")
        return lines


# ---------------------------------------------------------------------- counting


def files_in(directory: Path) -> tuple[int, int]:
    """``(files, bytes)`` below ``directory`` (0, 0 if it is not there)."""
    if not directory.is_dir():
        return 0, 0
    count = size = 0
    for path in directory.rglob("*"):
        if path.is_file():
            count += 1
            try:
                size += path.stat().st_size
            except OSError:
                continue
    return count, size


def database_counts(db_path: Path) -> tuple[int, int]:
    """``(tables, rows)`` of a database file, counted without any key."""
    if not db_path.is_file():
        return 0, 0
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        names = [
            str(name)
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
            )
        ]
        rows = sum(
            int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in names
        )
        return len(names), rows
    except sqlite3.DatabaseError:
        return 0, 0
    finally:
        connection.close()


def _megabytes(size: int) -> str:
    return f"{size / 1e6:.1f} MB"


def backup_files(directory: Path) -> list[Path]:
    """The files of the backup scheme in ``directory`` (archives, indexes, pool, pre-restore)."""
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for entry in directory.iterdir():
        if entry.is_file() and (
            entry.name.endswith((ARCHIVE_SUFFIX, SIDECAR_SUFFIX, ".part"))
            or entry.name.startswith(PRE_RESTORE_PREFIX)
        ):
            found.append(entry)
    pool = directory / POOL_DIRNAME
    if pool.is_dir():
        found.extend(path for path in pool.rglob("*") if path.is_file())
    return found


def model_entries(models_dir: Path) -> list[Path]:
    """The local model files and folders (the embedding model is a public download)."""
    if not models_dir.is_dir():
        return []
    return sorted(entry for entry in models_dir.iterdir() if entry.name != EMBEDDINGS_DIRNAME)


def plan_all(paths: DataPaths, mirror_dir: Path | None, keys: int) -> PurgePlan:
    """The list shown before ``--all`` asks for the confirmation phrase."""
    plan = PurgePlan("all")
    tables, rows = database_counts(paths.db_path)
    plan.items.append(PurgeItem("数据库", rows, "行", f"{tables} 张表，{paths.db_path.name}"))
    media_files, media_bytes = files_in(paths.media_dir)
    plan.items.append(PurgeItem("媒体文件", media_files, "个", _megabytes(media_bytes)))
    vector_files, _ = files_in(paths.vectors_dir)
    tables_dirs = (
        sum(1 for p in paths.vectors_dir.iterdir() if p.is_dir())
        if paths.vectors_dir.is_dir()
        else 0
    )
    plan.items.append(PurgeItem("向量库", tables_dirs, "张表", f"{vector_files} 个文件"))
    local = backup_files(paths.backups_dir)
    plan.items.append(PurgeItem("本地备份（含媒体池、恢复前备份）", len(local), "个文件"))
    mirrored = backup_files(mirror_dir) if mirror_dir is not None else []
    plan.items.append(
        PurgeItem("异地备份", len(mirrored), "个文件", str(mirror_dir) if mirror_dir else "未配置")
    )
    training_files, _ = files_in(paths.data_dir / TRAINING_DIRNAME)
    plan.items.append(PurgeItem("训练集与训练包", training_files, "个文件"))
    models = model_entries(paths.models_dir)
    plan.items.append(PurgeItem("本地风格模型与适配器", len(models), "项"))
    reports, _ = files_in(paths.reports_dir)
    logs, _ = files_in(paths.logs_dir)
    plan.items.append(PurgeItem("报告", reports, "个文件"))
    plan.items.append(PurgeItem("日志", logs, "个文件"))
    plan.items.append(PurgeItem("凭据管理器里的数据库密钥（含已退役的）", keys, "个"))
    return plan


def plan_training(paths: DataPaths, uncleaned: int) -> PurgePlan:
    plan = PurgePlan("training")
    training_files, size = files_in(paths.data_dir / TRAINING_DIRNAME)
    plan.items.append(PurgeItem("训练集与训练包", training_files, "个文件", _megabytes(size)))
    plan.items.append(PurgeItem("本地风格模型与适配器", len(model_entries(paths.models_dir)), "项"))
    plan.items.append(
        PurgeItem("还没确认清理的远程训练记录", uncleaned, "条", "需要去 AutoDL 控制台核对")
    )
    return plan


# --------------------------------------------------------------------- deleting


def _has_entries(folder: Path) -> bool:
    try:
        return any(folder.iterdir())
    except OSError:
        return False


class Eraser:
    """Deletes files and folders and remembers what refused to go (see :class:`PurgeReport`).

    A read-only file is made writable and tried again (Windows refuses to delete it otherwise).
    Any other :class:`OSError` - typically a sharing violation, a file some program still has
    open - is noted by name and the deletion goes on with the rest.
    """

    def __init__(self) -> None:
        self.failed: list[str] = []

    def _note(self, path: Path) -> None:
        if path.name not in self.failed:
            self.failed.append(path.name)

    def _retry_writable(self, action: Callable[[], object], path: Path) -> bool:
        """Make a read-only file writable and do ``action`` again; whether that worked."""
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            action()
        except OSError:
            return False
        return True

    def _on_rmtree_error(
        self, function: Callable[..., Any], name: str, error: BaseException
    ) -> None:
        path = Path(name)
        if isinstance(error, FileNotFoundError):
            return  # already gone
        if function is os.rmdir and _has_entries(path):
            return  # a folder that cannot go because a file in it could not: that file is listed
        if (
            function in (os.unlink, os.rmdir)
            and isinstance(error, PermissionError)
            and self._retry_writable(lambda: function(name), path)
        ):
            return
        self._note(path)

    def file(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            if not self._retry_writable(lambda: path.unlink(missing_ok=True), path):
                self._note(path)
        except OSError:
            self._note(path)

    def files(self, files: list[Path]) -> int:
        for path in files:
            self.file(path)
        return len(files)

    def tree(self, path: Path) -> int:
        """Delete a file or a folder with everything in it; returns how many files it held."""
        if path.is_dir():
            count, _ = files_in(path)
            shutil.rmtree(path, onexc=self._on_rmtree_error)
            return count
        if path.exists():
            self.file(path)
            return 1
        return 0


def purge_all(
    paths: DataPaths, mirror_dir: Path | None, keystore: KeyStore, *, close_logs: bool = True
) -> PurgeReport:
    """Delete everything of ``--all`` (the caller has the confirmation and the stopped app).

    Everything that can be deleted is; what cannot (a file another program holds open) is listed
    in ``failed``.  The keys are deleted in any case: the data that is left cannot be read
    without them, and running the purge again removes the files once they are free.
    """
    plan = plan_all(paths, mirror_dir, 0)
    report = PurgeReport("all", [item for item in plan.items if "密钥" not in item.category])
    eraser = Eraser()
    eraser.files([paths.db_path, Path(f"{paths.db_path}-wal"), Path(f"{paths.db_path}-shm")])
    eraser.tree(paths.media_dir)
    eraser.tree(paths.vectors_dir)
    eraser.files(backup_files(paths.backups_dir))
    if mirror_dir is not None:
        eraser.files(backup_files(mirror_dir))
    eraser.tree(paths.backups_dir / POOL_DIRNAME)
    if mirror_dir is not None:
        eraser.tree(mirror_dir / POOL_DIRNAME)
    eraser.tree(paths.data_dir / TRAINING_DIRNAME)
    for entry in model_entries(paths.models_dir):
        eraser.tree(entry)
    eraser.tree(paths.reports_dir)
    eraser.tree(paths.tmp_dir)
    if close_logs:
        shutdown_logging()  # the log files must be closed before they can be deleted (Windows)
    eraser.tree(paths.logs_dir)
    report.keys_deleted = keystore.delete_all()
    report.failed = eraser.failed
    return report


def delete_registry_rows(db: Database) -> int:
    """Forget the model files that ``--training-only`` deleted (the registry would point at
    nothing); the run history stays - it holds no content, and the cleanup record is the point."""
    with db.transaction() as session:
        result = session.execute(delete(ModelRegistryEntry))
        return int(getattr(result, "rowcount", 0) or 0)


def purge_training(paths: DataPaths, runs: int, db: Database | None = None) -> PurgeReport:
    """Delete the training data and local models of ``--training-only``."""
    plan = plan_training(paths, runs)
    report = PurgeReport("training", plan.items, uncleaned_runs=runs)
    eraser = Eraser()
    eraser.tree(paths.data_dir / TRAINING_DIRNAME)
    for entry in model_entries(paths.models_dir):
        eraser.tree(entry)
    report.failed = eraser.failed
    if db is not None:
        removed = delete_registry_rows(db)
        report.items.append(PurgeItem("模型登记", removed, "条"))
    return report
