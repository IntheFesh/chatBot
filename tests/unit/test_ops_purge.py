"""``twin purge``: complete deletion, shredded backups, typed confirmation (R-OPS-008)."""

from __future__ import annotations

import inspect
import os
import shutil
import stat
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from tests.support.backup_world import MARKER, BackupWorld, build_backup_world, table_counts
from tests.support.cli_runner import invoke
from tests.support.embedding import HashingBackend
from tests.support.training_history import MOMENT, record_training
from twin.cli import app
from twin.config.secrets import SecretStore
from twin.ops.backup.archive import read_manifest
from twin.ops.backup.layout import POOL_DIRNAME
from twin.ops.backup.sealed import BackupDecryptionError
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.process_model import ExitCode
from twin.ops.purge import (
    CONFIRM_ALL,
    CONFIRM_TRAINING,
    Eraser,
    PurgeItem,
    backup_files,
    database_counts,
    files_in,
    model_entries,
    plan_all,
    plan_training,
    purge_all,
    purge_training,
)
from twin.ops.purge_cli import purge_command
from twin.services import CliContext, Services, set_cli_context
from twin.storage.keystore import INDEX_NAME, KeyStore, key_secret_name
from twin.storage.rotate import rotate_db_key
from twin.storage.training_models import ModelRegistryEntry
from twin.training.runs import RunStore

runner = CliRunner()
DEEPSEEK_KEY = "deepseek_api_key"


class Installation:
    """A backed-up installation with everything ``purge`` is meant to delete, and some it is not."""

    def __init__(self, world: BackupWorld, root: Path) -> None:
        self.world = world
        self.services = world.services
        self.paths = world.services.paths
        self.root = root
        self.mirror = root / "usb"
        self.mirror.mkdir()
        self.export = root / "export"
        self.export.mkdir()
        (self.export / "chat.json").write_text("{}", encoding="utf-8")
        self.services.settings.ops.backup_mirror_dir = str(self.mirror)
        self.services.secrets.set(DEEPSEEK_KEY, "synthetic-key-keep-me")
        data = self.paths.data_dir
        (data / "training" / "datasets" / "v1").mkdir(parents=True)
        (data / "training" / "datasets" / "v1" / "sft_train.jsonl").write_text(
            f"{MARKER}-训练样本\n", encoding="utf-8"
        )
        (data / "training" / "bundles").mkdir(parents=True)
        (data / "training" / "bundles" / "bundle.tar.zst").write_bytes(b"bundle")
        (data / "models" / "style-lora" / "adapter").mkdir(parents=True)
        (data / "models" / "style-lora" / "adapter" / "weights.bin").write_bytes(b"weights")
        (data / "models" / "embeddings" / "bge").mkdir(parents=True)
        (data / "models" / "embeddings" / "bge" / "model.bin").write_bytes(b"public")
        (data / "reports").mkdir(exist_ok=True)
        (data / "reports" / "import-report.md").write_text(f"{MARKER}-报告", encoding="utf-8")
        self.paths.logs_dir.mkdir(exist_ok=True)
        (self.paths.logs_dir / "twin.log").write_text("log line", encoding="utf-8")
        self.paths.tmp_dir.mkdir(exist_ok=True)
        (self.paths.tmp_dir / "leftover").write_bytes(b"x")
        self.archive = self.world.backup.backups_dir / str(
            self.world.backup.create("manual").file_name
        )
        self.stolen = root / "copy-somewhere-else.bak.enc"
        shutil.copyfile(self.archive, self.stolen)  # a copy that purge cannot reach


@pytest.fixture
def installed(services: Services, embedder: HashingBackend, tmp_path: Path) -> Installation:
    return Installation(build_backup_world(services), tmp_path)


def tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*")) if path.exists() else []


# ----------------------------------------------------------------------------- the plan


def test_the_plan_counts_every_category_and_shows_no_content(installed: Installation) -> None:
    paths = installed.paths
    plan = plan_all(paths, installed.mirror, keys=1)
    by_name = {item.category: item for item in plan.items}
    assert by_name["数据库"].count == database_counts(paths.db_path)[1] > 10
    assert by_name["媒体文件"].count == 2 and by_name["向量库"].count == 1
    assert by_name["本地备份（含媒体池、恢复前备份）"].count == 4  # archive, index, 2 pool files
    assert by_name["异地备份"].count == 4 and str(installed.mirror) in by_name["异地备份"].note
    assert by_name["训练集与训练包"].count == 2 and by_name["本地风格模型与适配器"].count == 1
    assert by_name["报告"].count == 1 and by_name["日志"].count >= 1
    assert by_name["凭据管理器里的数据库密钥（含已退役的）"].count == 1
    assert plan.scope == "all" and MARKER not in "\n".join(plan.lines())
    assert "未配置" in {i.category: i for i in plan_all(paths, None, 0).items}["异地备份"].note


def test_the_training_plan_lists_the_runs_whose_data_may_be_on_a_rented_machine(
    installed: Installation,
) -> None:
    record_training(installed.services)
    plan = plan_training(installed.paths, 2)
    names = {item.category: item for item in plan.items}
    assert names["训练集与训练包"].count == 2 and names["本地风格模型与适配器"].count == 1
    assert names["还没确认清理的远程训练记录"].count == 2
    assert "AutoDL" in names["还没确认清理的远程训练记录"].note


def test_the_embedding_model_is_a_public_download_and_is_not_a_style_model(
    installed: Installation,
) -> None:
    assert [p.name for p in model_entries(installed.paths.models_dir)] == ["style-lora"]
    assert model_entries(installed.root / "nowhere") == []
    assert files_in(installed.root / "nowhere") == (0, 0)
    assert database_counts(installed.root / "nowhere.db") == (0, 0)
    junk = installed.root / "junk.db"
    junk.write_bytes(b"not a database" * 100)
    assert database_counts(junk) == (0, 0)
    assert PurgeItem("数据库", 3, "行").line() == "数据库：3 行"


# ----------------------------------------------------------------------------- --all


def test_everything_about_her_is_gone_and_only_counts_are_reported(
    installed: Installation,
) -> None:
    paths = installed.paths
    keystore = installed.services.keystore
    installed.services.db.dispose()
    report = purge_all(paths, installed.mirror, keystore, close_logs=False)
    assert not paths.db_path.exists() and not Path(f"{paths.db_path}-wal").exists()
    assert tree(paths.media_dir) == [] and tree(paths.vectors_dir) == []
    assert backup_files(paths.backups_dir) == [] and not (paths.backups_dir / POOL_DIRNAME).exists()
    assert backup_files(installed.mirror) == [] and tree(installed.mirror) == []
    assert tree(paths.data_dir / "training") == [] and not (paths.data_dir / "training").exists()
    assert [p.name for p in paths.models_dir.iterdir()] == ["embeddings"]  # the public download
    assert not paths.reports_dir.exists() and not paths.logs_dir.exists()
    assert not paths.tmp_dir.exists()
    # what is not hers stays: the export she sent, the DeepSeek key
    assert (installed.export / "chat.json").is_file()
    assert installed.services.secrets.get(DEEPSEEK_KEY) == "synthetic-key-keep-me"
    # the report holds numbers and names of categories, never content
    text = "\n".join(report.lines())
    assert MARKER not in text and report.keys_deleted == 1
    assert "凭据管理器里的数据库密钥：删除 1 个" in text and report.scope == "all"


def test_every_key_goes_with_it_the_retired_ones_too(installed: Installation) -> None:
    services = installed.services
    rotate_db_key(services.db, services.keystore, services.keyring, services.clock, services.media)
    assert services.secrets.exists(key_secret_name(1)) and services.secrets.exists(
        key_secret_name(2)
    )
    services.db.dispose()
    report = purge_all(installed.paths, installed.mirror, services.keystore, close_logs=False)
    assert report.keys_deleted == 2
    assert not services.secrets.exists(key_secret_name(1))
    assert not services.secrets.exists(key_secret_name(2))
    assert not services.secrets.exists(INDEX_NAME) and not services.keystore.exists()


def test_a_copy_of_a_backup_cannot_be_read_after_the_purge_even_with_a_new_key_1(
    installed: Installation,
) -> None:
    services = installed.services
    ring_before = services.keyring
    assert read_manifest(installed.stolen, ring_before).key_id == 1  # readable now
    services.db.dispose()
    purge_all(installed.paths, installed.mirror, services.keystore, close_logs=False)
    fresh = KeyStore(services.secrets).load_or_create(allow_create=True)  # a new installation
    assert fresh.current_id == 1  # the same id as the old key, other bytes
    assert fresh.key_bytes(1) != ring_before.key_bytes(1)
    with pytest.raises(BackupDecryptionError):
        read_manifest(installed.stolen, fresh)


def test_a_damaged_key_index_does_not_keep_a_key_alive(installed: Installation) -> None:
    services = installed.services
    rotate_db_key(services.db, services.keystore, services.keyring, services.clock, services.media)
    services.secrets.set(INDEX_NAME, "{broken")
    services.db.dispose()
    purge_all(installed.paths, installed.mirror, services.keystore, close_logs=False)
    assert not services.secrets.exists(key_secret_name(1))
    assert not services.secrets.exists(key_secret_name(2))
    assert not services.secrets.exists(INDEX_NAME)
    assert services.secrets.get(DEEPSEEK_KEY) == "synthetic-key-keep-me"


def test_a_purge_of_an_installation_without_a_mirror_or_backups_works(
    services: Services,
) -> None:
    services.db.dispose()
    report = purge_all(services.paths, None, services.keystore, close_logs=False)
    assert report.keys_deleted == 1 and not services.paths.db_path.exists()


def hold_files_open(monkeypatch: pytest.MonkeyPatch, *names: str) -> list[str]:
    """Make deleting files with these names fail like a sharing violation on Windows."""
    real_unlink = os.unlink
    refused: list[str] = []

    def unlink(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        if Path(os.fsdecode(path)).name in names:
            refused.append(Path(os.fsdecode(path)).name)
            raise PermissionError(13, "The process cannot access the file: it is in use")
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "unlink", unlink)
    return refused


def test_a_file_another_program_holds_open_is_reported_and_the_rest_still_goes(
    installed: Installation, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = installed.paths
    installed.services.db.dispose()
    refused = hold_files_open(monkeypatch, "twin.db", "bundle.tar.zst")
    report = purge_all(paths, installed.mirror, installed.services.keystore, close_logs=False)
    assert set(report.failed) == {"twin.db", "bundle.tar.zst"} and refused
    assert paths.db_path.exists() and (paths.data_dir / "training" / "bundles").exists()
    assert tree(paths.media_dir) == [] and tree(paths.vectors_dir) == []  # everything else went
    assert backup_files(paths.backups_dir) == [] and backup_files(installed.mirror) == []
    assert [p.name for p in paths.models_dir.iterdir()] == [
        "embeddings"
    ] and not paths.logs_dir.exists()
    assert report.keys_deleted == 1 and not installed.services.keystore.exists()  # shredded anyway
    assert "没能删除：2 项" in "\n".join(report.lines()) and MARKER not in "\n".join(report.lines())


def test_the_command_says_what_could_not_be_deleted_and_ends_with_an_error(
    cli: Installation, monkeypatch: pytest.MonkeyPatch
) -> None:
    hold_files_open(monkeypatch, "twin.db")
    code, out = run_purge(cli, "--all", phrase=CONFIRM_ALL)
    assert code == 1 and "没能删除：1 项（twin.db）" in out and "再运行一次" in out
    assert "已删除（只列数量）" in out and "备份和任何残留的副本" not in out  # no all-clear
    assert cli.paths.db_path.exists() and not cli.services.keystore.exists()
    monkeypatch.undo()
    cli.services.secrets.set(DEEPSEEK_KEY, "synthetic-key-keep-me")
    code, out = run_purge(cli, "--all", phrase=CONFIRM_ALL)  # the second run finishes the job
    assert code == 0 and not cli.paths.db_path.exists()


def test_the_eraser_takes_files_folders_and_things_that_are_not_there(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "folder" / "inner").mkdir(parents=True)
    (work / "folder" / "inner" / "a.bin").write_bytes(b"a")
    (work / "folder" / "b.bin").write_bytes(b"b")
    (work / "single.bin").write_bytes(b"c")
    eraser = Eraser()
    assert eraser.tree(work / "folder") == 2
    assert eraser.tree(work / "single.bin") == 1
    assert eraser.tree(work / "never-there") == 0
    assert eraser.failed == [] and list(work.iterdir()) == []


def test_a_read_only_file_is_made_writable_and_deleted(
    installed: Installation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows refuses to delete a read-only file; the purge clears the flag and goes on."""
    weights = installed.paths.models_dir / "style-lora" / "adapter" / "weights.bin"
    weights.chmod(stat.S_IREAD)
    real_unlink = os.unlink

    def windows_like(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> None:
        if not stat.S_IMODE(os.stat(path, dir_fd=dir_fd).st_mode) & stat.S_IWRITE:
            raise PermissionError(13, "Access is denied")
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", windows_like)
    report = purge_training(installed.paths, 0)
    assert report.failed == [] and not weights.exists()


# ----------------------------------------------------------------------------- --training-only


def add_registry_entry(services: Services) -> None:
    with services.db.transaction() as session:
        session.add(
            ModelRegistryEntry(
                id="m1",
                run_id="r1",
                kind="gguf",
                profile="5090-8b",
                base_model="qwen3-8b",
                quant="q4_k_m",
                path="models/style-lora/model.gguf",
                sha256="0" * 64,
                size=1,
                template_version="t",
                persona_version="p",
                profile_version="pr",
                dataset_version="ds-1",
            )
        )


def test_training_only_leaves_the_rest_alone(installed: Installation) -> None:
    services = installed.services
    record_training(services)
    add_registry_entry(services)
    counts = table_counts(installed.paths.db_path)
    report = purge_training(installed.paths, 1, services.db)
    assert not (installed.paths.data_dir / "training").exists()
    assert [p.name for p in installed.paths.models_dir.iterdir()] == ["embeddings"]
    with services.db.session() as session:
        assert session.scalars(select(ModelRegistryEntry)).all() == []
    after = table_counts(installed.paths.db_path)
    assert after["model_registry"] == 0 and counts["model_registry"] == 1
    assert {k: v for k, v in after.items() if k != "model_registry"} == {
        k: v for k, v in counts.items() if k != "model_registry"
    }  # the run history stays: it holds no content
    assert installed.paths.db_path.is_file() and tree(installed.paths.media_dir)
    assert installed.archive.is_file() and services.keystore.exists()
    assert report.scope == "training" and report.uncleaned_runs == 1
    assert any(i.category == "模型登记" and i.count == 1 for i in report.items)


# ----------------------------------------------------------------------------- the command


@pytest.fixture
def cli(installed: Installation, secret_store: SecretStore) -> Installation:
    set_cli_context(CliContext(secrets=secret_store))  # the root callback keeps the secret store
    return installed


def run_purge(installed: Installation, *args: str, phrase: str | None = None) -> tuple[int, str]:
    options = [
        "--set",
        f"paths.data_dir={installed.paths.data_dir}",
        "--set",
        f"ops.backup_mirror_dir={installed.mirror}",
    ]
    if "--all" in args:
        # The application is stopped when a purge runs; the fixture's own container stands in
        # for it, and Windows refuses to delete a database file that a connection of this
        # process still has open.
        installed.services.db.dispose()
    return invoke(
        runner, [*options, "purge", *args], answer=None if phrase is None else phrase + "\n"
    )


def test_the_command_needs_exactly_one_scope(cli: Installation) -> None:
    code, out = run_purge(cli)
    assert code == ExitCode.USAGE and "exactly one" in out
    code, out = run_purge(cli, "--all", "--training-only")
    assert code == ExitCode.USAGE and "exactly one" in out
    assert cli.paths.db_path.is_file()


def test_a_wrong_phrase_deletes_nothing(cli: Installation) -> None:
    before = table_counts(cli.paths.db_path)
    for phrase in ("删除", "yes", "", "删除她的全部数据吗", "删除训练数据"):
        code, out = run_purge(cli, "--all", phrase=phrase)
        assert code == 1 and "什么都没有删除" in out, phrase
    assert table_counts(cli.paths.db_path) == before
    assert cli.archive.is_file() and cli.services.keystore.exists()
    assert tree(cli.paths.media_dir)


def test_the_typed_phrase_deletes_everything_and_says_what_was_removed(cli: Installation) -> None:
    code, out = run_purge(cli, "--all", phrase=CONFIRM_ALL)
    assert code == 0, out
    assert "将要删除" in out and "数据库" in out and "已删除（只列数量）" in out
    assert "无法解密" in out and MARKER not in out
    assert not cli.paths.db_path.exists() and tree(cli.paths.media_dir) == []
    assert not cli.archive.exists() and tree(cli.mirror) == []
    assert not cli.services.keystore.exists()
    assert (cli.export / "chat.json").is_file()


def test_there_is_no_way_to_skip_the_question() -> None:
    parameters = set(inspect.signature(purge_command).parameters)
    assert parameters == {"all_data", "training_only"}  # no --yes, no --confirm, no phrase option
    for flag in ("--yes", "--confirm", "--force", "-y"):
        result = runner.invoke(app, ["purge", "--all", flag])
        assert result.exit_code == 2, flag  # an unknown option


def test_the_running_application_blocks_the_purge(cli: Installation) -> None:
    lock = InstanceLock(LOCK_RUN, locks_dir=cli.paths.locks_dir)
    assert lock.acquire()
    try:
        code, out = run_purge(cli, "--all", phrase=CONFIRM_ALL)
    finally:
        lock.release()
    assert code == ExitCode.BUSY and "twin service stop" in out
    assert cli.paths.db_path.is_file() and cli.services.keystore.exists()


def upload_to_the_rented_machine(services: Services) -> None:
    """A run whose training set went to the rented machine and was not cleaned up yet."""
    store = RunStore(services.db)
    store.begin_step("r1", "upload", MOMENT)
    store.end_step("r1", "upload", MOMENT, exit_code=0)


def test_the_training_scope_has_its_own_phrase(cli: Installation) -> None:
    record_training(cli.services)
    upload_to_the_rented_machine(cli.services)
    code, out = run_purge(
        cli, "--training-only", phrase=CONFIRM_ALL
    )  # the other phrase does not do
    assert code == 1 and (cli.paths.data_dir / "training").exists()
    code, out = run_purge(cli, "--training-only", phrase=CONFIRM_TRAINING)
    assert code == 0, out
    assert not (cli.paths.data_dir / "training").exists()
    assert "AutoDL 控制台" in out and "有 1 次远程训练没有记录清理完成" in out
    assert cli.paths.db_path.is_file() and cli.services.keystore.exists()
