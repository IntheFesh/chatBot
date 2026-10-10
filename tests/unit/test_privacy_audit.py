"""R-PRIV-001 and the privacy rules of CLAUDE.md: the total audit (round 16).

The promises, and what holds each of them:

1. **Nothing real is in the repository.**  ``.gitignore`` covers every place real data lives,
   git tracks none of them, ``scripts/privacy_scan.py`` finds the shapes of real data in every
   tracked file and runs in three places (pre-commit, ``scripts/check.ps1``, CI), and the test
   fixtures are generators and templates, never recorded data.
2. **Logs carry no words of the conversation** above DEBUG, and redacted ones at DEBUG.
3. **What leaves the machine is redacted or encrypted.**  The modules that can reach the network
   are listed below with the reason each is harmless; a new one fails this test until it is
   reviewed and added.
4. **Every privacy requirement (R-PRIV-001..006) has tests**, cited below and looked up so that a
   renamed test cannot silently drop one.

The detailed behaviour is tested where it lives; this file is the part that fails when the
*arrangement* changes.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.support.nodes import missing_nodes
from tests.support.scripts import load_script
from tests.support.synthetic import (
    card_with_luhn,
    export_fragment,
    id_card_with_check,
    mobile,
    wxid,
)
from twin.ops.logging import configure_logging, get_logger, shutdown_logging

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "twin"
scan = load_script("privacy_scan")
BODY = "这是一段绝对不能出现在日志里的聊天正文"

# ------------------------------------------------------------------ 1. the repository


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False, encoding="utf-8"
    )


needs_git = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="git is not available or this is not a checkout",
)

PRIVATE_PATHS = [
    "data/twin.db",
    "data/media/ab12.enc",
    "data/vectors/windows.lance/data.lance",
    "data/models/embeddings/x.bin",
    "data/training/bundles/bundle.tar.zst.enc",
    "data/backups/daily.bak.enc",
    "data/reports/import-2026.md",
    "twin.db",
    "twin.db-wal",
    "dump.sqlite3",
    "exports/微信导出/conversations/1/messages.json",
    "models/run1/model.gguf",
    "models/qwen/adapter.safetensors",
    "backups/daily.bak.enc",
    "secret.enc",
    ".env",
    ".env.local",
    "config/config.yaml",
]
PUBLIC_PATHS = [
    "config/config.example.yaml",
    "config/lists/ai_phrases.txt",
    "docs/SPEC.md",
    "src/twin/cli.py",
    "tests/fixtures/synth_export.py",
    "tests/fixtures/training/yaml/5090-8b.sft.yaml",
    "scripts/privacy_scan.py",
    "training/README.md",
]


@needs_git
@pytest.mark.parametrize("path", PRIVATE_PATHS)
def test_git_ignores_every_place_that_real_data_lives(path: str) -> None:
    assert git("check-ignore", "-q", path).returncode == 0, f"{path} would be committed"


@needs_git
@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_git_does_not_ignore_what_belongs_in_the_repository(path: str) -> None:
    assert git("check-ignore", "-q", path).returncode == 1, f"{path} is ignored"


@needs_git
def test_git_tracks_nothing_from_the_private_places() -> None:
    listing = git("ls-files", "-z")
    assert listing.returncode == 0
    tracked = [name for name in listing.stdout.split("\0") if name]
    assert len(tracked) > 400
    private_roots = ("data/", "exports/", "models/", "backups/")
    private_suffixes = (".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3", ".enc", ".gguf")
    offenders = [
        name
        for name in tracked
        if name.startswith(private_roots)
        or name.endswith(private_suffixes)
        or name in {".env", "config/config.yaml"}
    ]
    assert offenders == []


def test_the_scan_finds_every_kind_of_real_data_and_never_prints_it() -> None:
    samples = {
        "wxid": f"contact {wxid()}",
        "mobile number": f"call {mobile()}",
        "ID card number": f"id {id_card_with_check('11010519491231002')}",
        "bank card number (Luhn)": f"card {card_with_luhn('411111')}",
        "API key / private key": "key " + "sk-" + "a1b2c3d4e5f6g7h8i9j0k1l2",
        "WeChat export (messages.json) fragment": export_fragment(),
    }
    for kind, text in samples.items():
        findings = scan.scan_text("notes.txt", text + "\n")
        assert [f.kind for f in findings] == [kind], kind
        assert text.split()[-1] not in findings[0].render()


def test_the_scan_runs_in_the_hook_in_the_local_check_and_in_ci() -> None:
    assert "scripts/privacy_scan.py" in (ROOT / ".pre-commit-config.yaml").read_text("utf-8")
    assert "scripts/privacy_scan.py" in (ROOT / "scripts" / "check.ps1").read_text("utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    assert "scripts/privacy_scan.py" in ci
    lint = ci[ci.index("\n  lint:") : ci.index("\n  tests:")]
    assert "uv run python scripts/privacy_scan.py" in lint  # in the job every push passes through


def test_no_tracked_file_holds_the_shape_of_real_data() -> None:
    findings = list(scan.scan_files(scan.tracked_files(ROOT), ROOT))
    assert findings == [], [f.render() for f in findings]


def test_the_test_fixtures_are_generators_and_templates_never_recorded_data() -> None:
    allowed = {".py", ".md", ".txt", ".yaml", ".xml"}
    files = [p for p in (ROOT / "tests" / "fixtures").rglob("*") if p.is_file()]
    files = [p for p in files if "__pycache__" not in p.parts]
    assert files
    for path in files:
        assert path.suffix in allowed, f"{path.relative_to(ROOT)}: a recorded file?"
        assert path.stat().st_size < 200_000, f"{path.relative_to(ROOT)} is large for a template"
    readme = (ROOT / "tests" / "fixtures" / "README.md").read_text(encoding="utf-8")
    assert "synthetic" in readme


def test_no_media_database_or_export_sits_anywhere_in_the_tests() -> None:
    suffixes = {".db", ".sqlite", ".enc", ".gguf", ".safetensors", ".mp3", ".mp4", ".silk", ".amr"}
    for path in (ROOT / "tests").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            assert path.suffix not in suffixes, path.relative_to(ROOT)
            if path.suffix == ".json":
                assert "messages" not in path.name, path.relative_to(ROOT)


# ----------------------------------------------------------------------------- 2. the logs


def test_a_sentence_of_the_conversation_is_not_in_the_log_file_at_any_level_above_debug(
    tmp_path: Path,
) -> None:
    path = configure_logging(tmp_path, level="INFO", console=False)
    try:
        log = get_logger("audit.privacy")
        log.info("received", content=BODY, text=BODY, reply=BODY, prompt=BODY, user="u1")
        log.warning("odd", body=BODY, messages=[{"role": "user", "content": BODY}])
        try:
            raise ValueError(f"cannot send {mobile()}")
        except ValueError:
            log.exception("send_failed")
        written = path.read_text(encoding="utf-8")
    finally:
        shutdown_logging()
    assert BODY not in written and mobile() not in written
    lines = [json.loads(line) for line in written.splitlines() if line]
    assert [entry["event"] for entry in lines][:2] == ["received", "odd"]


# ------------------------------------------------------------- 3. what leaves the machine

# module (relative to src/twin) -> why it may touch the network
NETWORK_MODULES = {
    "channel/ilink/channel.py": "WeChat: only the bound user (RecipientGuard)",
    "channel/ilink/connectivity.py": "doctor: a reachability probe, no payload",
    "channel/ilink/flows.py": "WeChat login and polling",
    "channel/ilink/http.py": "the one HTTP wrapper for the WeChat host",
    "channel/ilink/login.py": "WeChat QR login",
    "channel/ilink/media.py": "WeChat media: the user's own pictures, and allowed stickers out",
    "llm/deepseek.py": "DeepSeek: redact_messages before every request",
    "llm/reliability.py": "retry and circuit breaker: error types only",
    "llm/style_client.py": "style model: remote prompts are redacted, the local server is local",
    "ops/doctor.py": "doctor: reachability probes and the balance query, no chat",
    "ops/mail.py": "SMTP: the user's own address (see the outbound audit)",
    "services.py": "builds the shared HTTP transport",
    "serving/llamacpp.py": "local llama-server on this machine",
    "serving/server.py": "local llama-server on this machine",
    "serving/tunnel.py": "SSH port forward to the user's own AutoDL instance",
    "stickers/download.py": "GET of sticker pictures at the URLs of the export; sends nothing",
    "training/remote/connection.py": "AutoDL over SSH: the encrypted training package",
    "training/remote/session.py": "AutoDL over SSH: the encrypted training package",
    "training/remote/transfer.py": "AutoDL over SSH: the encrypted training package",
    "training/tokenizer.py": "GET of the public tokenizer file; sends nothing",
}
NETWORK_LIBRARIES = {"httpx", "openai", "asyncssh", "requests", "urllib", "socket", "smtplib"}


def test_the_modules_that_can_reach_the_network_are_exactly_the_reviewed_ones() -> None:
    """Chat records are read, profiled and indexed in modules that cannot send them anywhere."""
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            if any(name.split(".")[0] in NETWORK_LIBRARIES for name in names):
                found.add(path.relative_to(SRC).as_posix())
    assert found == set(NETWORK_MODULES), sorted(found ^ set(NETWORK_MODULES))


@pytest.mark.parametrize("package", ["ingest", "profile", "memory", "retrieval", "commands"])
def test_the_packages_that_handle_the_records_never_touch_the_network(package: str) -> None:
    owners = [name for name in NETWORK_MODULES if name.startswith(f"{package}/")]
    assert owners == []


def function_calls(path: str, qualified: str) -> list[str]:
    tree = ast.parse((SRC / path).read_text(encoding="utf-8"))
    owner, _, method = qualified.partition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == owner:
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef) and (
                    member.name == method
                ):
                    return [
                        ast.unparse(c.func) for c in ast.walk(member) if isinstance(c, ast.Call)
                    ]
    raise AssertionError(f"{qualified} not found in {path}")


def test_the_deepseek_client_redacts_every_message_before_the_request() -> None:
    calls = function_calls("llm/deepseek.py", "DeepSeekClient.chat")
    assert "redact_messages" in calls
    first_redact = calls.index("redact_messages")
    assert not any("create" in call for call in calls[:first_redact]), "a request before redaction"


def test_the_remote_style_prompt_is_redacted_by_default_and_the_local_one_is_not_sent_out() -> None:
    source = (SRC / "llm" / "style_client.py").read_text(encoding="utf-8")
    assert "outbound: Callable[[str], str] | None = redact_text" in source


# ------------------------------------------------------- 4. each privacy requirement has tests

PRIVACY_TESTS = {
    "R-PRIV-001": [
        "tests/unit/test_gate_scripts.py::test_sensitive_patterns_are_found_without_echoing_them",
        "tests/unit/test_gate_scripts.py::test_the_repository_itself_is_clean",
        "tests/unit/test_scan_rules.py::test_gitignore_covers_the_private_paths",
    ],
    "R-PRIV-002": [
        "tests/unit/test_llm_deepseek.py::test_personal_identifiers_never_leave_the_machine",
        "tests/unit/test_llm_style_client.py::test_vllm_redacts_the_prompt_because_it_leaves_the_machine",
        "tests/unit/test_redaction.py::test_identifiers_are_replaced_by_type_tokens",
    ],
    "R-PRIV-003": [
        "tests/unit/test_training_bundle.py::test_a_dataset_that_is_not_marked_desensitised_is_refused",
        "tests/unit/test_training_bundle.py::test_the_file_on_disk_holds_no_readable_data_and_no_plain_archive_is_left",
        "tests/unit/test_autodl_scripts.py::test_cleanup_refuses_without_confirmation",
        "tests/unit/test_training_export.py::test_a_redactor_that_misses_something_makes_the_export_fail_and_leave_nothing",
    ],
    "R-PRIV-004": [
        "tests/unit/test_encrypted_models.py::test_json_round_trip_and_ciphertext_on_disk",
        "tests/unit/test_media_store.py::test_stored_file_contains_no_plaintext",
        "tests/unit/test_ops_backup.py::test_nothing_readable_is_in_the_file",
    ],
    "R-PRIV-005": [
        "tests/unit/test_ops_purge.py::test_everything_about_her_is_gone_and_only_counts_are_reported",
        "tests/unit/test_ops_purge.py::test_a_copy_of_a_backup_cannot_be_read_after_the_purge_even_with_a_new_key_1",
        "tests/unit/test_ops_purge.py::test_there_is_no_way_to_skip_the_question",
    ],
    "R-PRIV-006": [
        "tests/unit/test_ilink_outbound.py::test_only_the_bound_user_can_be_named_as_the_recipient",
        "tests/unit/test_channel_rules.py::test_only_the_documented_endpoints_are_ever_called",
    ],
    "log without words (R-OPS-007)": [
        "tests/unit/test_logging.py::test_info_logs_never_contain_content",
        "tests/unit/test_logging.py::test_debug_logs_keep_content_but_redacted",
        "tests/unit/test_engine_e2e.py::test_an_ordinary_conversation_logs_no_words_of_it",
        "tests/unit/test_engine_failures.py::test_the_backend_failure_log_never_has_the_text_of_the_conversation",
        "tests/unit/test_scan_rules.py::test_application_code_logs_through_the_structured_logger_only",
        "tests/unit/test_scan_rules.py::test_no_print_in_production_code",
    ],
    "the bot's own words stay out of the samples (CLAUDE.md rule 7)": [
        "tests/unit/test_message_isolation.py::test_messages_and_bot_turns_are_different_tables",
        "tests/unit/test_message_isolation.py::test_the_corpus_queries_read_her_real_messages_only",
        "tests/unit/test_message_isolation.py::test_only_the_import_writes_to_the_messages_table",
        "tests/unit/test_dpo_export.py::test_the_rejected_text_is_in_the_dpo_file_and_in_no_other_file",
    ],
}


@pytest.mark.parametrize("requirement", sorted(PRIVACY_TESTS))
def test_every_privacy_requirement_names_tests_that_exist(requirement: str) -> None:
    assert PRIVACY_TESTS[requirement], requirement
    assert missing_nodes(PRIVACY_TESTS[requirement]) == [], requirement
