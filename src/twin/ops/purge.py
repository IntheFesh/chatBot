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

import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

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
    """What a purge deleted: counts only."""

    scope: str
    items: list[PurgeItem] = field(default_factory=list)
    keys_deleted: int = 0
    uncleaned_runs: int = 0

    def lines(self) -> list[str]:
        lines = [item.line() for item in self.items]
        if self.scope == "all":
            lines.append(f"凭据管理器里的数据库密钥：删除 {self.keys_deleted} 个")
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


def _remove(path: Path) -> int:
    """Delete a file or folder; returns how many files went."""
    if path.is_dir():
        count, _ = files_in(path)
        shutil.rmtree(path, ignore_errors=False)
        return count
    if path.exists():
        path.unlink()
        return 1
    return 0


def _remove_files(files: list[Path]) -> int:
    removed = 0
    for path in files:
        path.unlink(missing_ok=True)
        removed += 1
    pool_dirs = {path.parent for path in files if path.parent.name == POOL_DIRNAME}
    for folder in pool_dirs:
        shutil.rmtree(folder, ignore_errors=True)
    return removed


def purge_all(
    paths: DataPaths, mirror_dir: Path | None, keystore: KeyStore, *, close_logs: bool = True
) -> PurgeReport:
    """Delete everything of ``--all`` (the caller has the confirmation and the stopped app)."""
    plan = plan_all(paths, mirror_dir, 0)
    report = PurgeReport("all", [item for item in plan.items if "密钥" not in item.category])
    db_files = [paths.db_path, Path(f"{paths.db_path}-wal"), Path(f"{paths.db_path}-shm")]
    for path in db_files:
        path.unlink(missing_ok=True)
    _remove(paths.media_dir)
    _remove(paths.vectors_dir)
    _remove_files(backup_files(paths.backups_dir))
    if mirror_dir is not None:
        _remove_files(backup_files(mirror_dir))
    shutil.rmtree(paths.backups_dir / POOL_DIRNAME, ignore_errors=True)
    _remove(paths.data_dir / TRAINING_DIRNAME)
    for entry in model_entries(paths.models_dir):
        _remove(entry)
    _remove(paths.reports_dir)
    _remove(paths.tmp_dir)
    if close_logs:
        shutdown_logging()  # the log files must be closed before they can be deleted (Windows)
    _remove(paths.logs_dir)
    report.keys_deleted = keystore.delete_all()
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
    _remove(paths.data_dir / TRAINING_DIRNAME)
    for entry in model_entries(paths.models_dir):
        _remove(entry)
    if db is not None:
        removed = delete_registry_rows(db)
        report.items.append(PurgeItem("模型登记", removed, "条"))
    return report
