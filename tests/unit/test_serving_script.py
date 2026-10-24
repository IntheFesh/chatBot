"""``get_llamacpp.ps1`` and its lock file: what they do and what they refuse (R-SRV-002).

PowerShell is not available where these tests run, so the script is checked as text - it must be
plain ASCII, strict, stop at the first error, refuse to run without a valid lock, verify every
download against the lock, and put the CUDA runtime next to the CUDA build - and the lock file is
checked as data: the asset names follow the naming of the release workflow of llama.cpp and every
checksum is a SHA-256.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from twin.serving.llamacpp import CUDA_RUNTIME, WINDOWS_BASE_FILES
from twin.training import versions

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "windows" / "get_llamacpp.ps1"
LOCK = ROOT / "scripts" / "windows" / "llamacpp.lock.json"


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def lock() -> dict[str, object]:
    data = json.loads(LOCK.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_the_script_is_plain_ascii_strict_and_stops_at_the_first_error() -> None:
    script = script_text()
    script.encode("ascii")  # Windows PowerShell 5.1 reads a file without a BOM in the ANSI page
    assert "Set-StrictMode -Version Latest" in script
    assert "$ErrorActionPreference = 'Stop'" in script
    assert "[CmdletBinding()]" in script and ".SYNOPSIS" in script and ".EXAMPLE" in script
    assert script.count("{") == script.count("}") and script.count("(") == script.count(")")
    assert script.count("[") == script.count("]")
    assert "Invoke-Expression" not in script and not re.search(r"\biex\b", script)
    assert "PSVersion" in script and "Win32NT" in script  # 5.1 or newer, Windows only


def test_the_script_uses_nothing_newer_than_windows_powershell_5_1() -> None:
    script = script_text()
    for newer in ("&&", "||", " ? ", "??", "?.", "ForEach-Object -Parallel", "-AsHashtable"):
        assert newer not in script, newer


def test_the_script_refuses_to_run_without_a_valid_lock() -> None:
    script = script_text()
    assert "llamacpp.lock.json" in script
    for sentence in (
        "The lock file $LockPath is missing",
        "names no release tag",
        "has no valid SHA-256",
        "unsupported schema",
    ):
        assert sentence in script, sentence
    assert re.search(r"\^\[0-9a-fA-F\]\{64\}\$", script)  # a checksum is 64 hexadecimal digits
    assert "Read-Lock" in script and script.index("Read-Lock") < script.index("Get-GpuInfo")


def test_every_download_is_verified_and_deleted_when_it_does_not_match() -> None:
    script = script_text()
    assert "Get-FileHash" in script and "-Algorithm SHA256" in script
    assert script.count("Remove-Item -LiteralPath $target -Force") == 2  # lock and API digest
    assert "the lock file says" in script and "reports another digest" in script
    assert (
        "releases/download/$Tag/" in script
        and "api.github.com/repos/$Repository/releases/tags" in script
    )
    # the checksum is compared before anything is unpacked
    assert script.index("Get-VerifiedAsset -Asset $Parts.archive") < script.index(
        "Expand-Zip -Archive $Archive"
    )


def test_the_cuda_runtime_goes_into_the_same_folder_as_the_build() -> None:
    script = script_text()
    runtime = script.index("Get-VerifiedAsset -Asset $Parts.runtime")
    assert "Expand-Zip -Archive $Runtime -Destination $Staging" in script[runtime:]
    assert "Expand-Zip -Archive $Archive -Destination $Staging" in script


def test_a_half_finished_download_is_never_left_where_the_program_looks() -> None:
    script = script_text()
    assert (
        "'.staging-' + $Tag" in script
        and "Move-Item -LiteralPath $Staging -Destination $Final" in script
    )
    finally_block = script[script.rindex("} finally {") :]
    assert "Remove-Item -LiteralPath $leftover -Recurse -Force" in finally_block
    removals = re.findall(r"Remove-Item[^\n]*", script)
    assert all(
        any(name in line for name in ("$target", "$leftover", "$Final")) for line in removals
    ), removals  # nothing is deleted that this script did not create


def test_the_files_a_working_installation_needs_are_the_ones_the_program_checks() -> None:
    script = script_text()
    for name in WINDOWS_BASE_FILES:
        assert f"'{name}'" in script, name
    for kind, names in CUDA_RUNTIME.items():
        assert (
            f"'{kind}' = @('ggml-cuda.dll', " + ", ".join(f"'{n}'" for n in names) + ")" in script
        )


def test_the_build_choice_is_the_one_of_the_program() -> None:
    script = script_text()
    assert (
        "[Version]'13.0'" in script and "[Version]'12.4'" in script and "[Version]'7.5'" in script
    )
    assert "CUDA Version:" in script and "compute_cap" in script


def test_the_lock_names_the_release_the_converter_and_quantiser_use() -> None:
    data = lock()
    assert data["schema"] == 1 and data["repository"] == "ggml-org/llama.cpp"
    assert data["tag"] == versions.LLAMA_CPP_TAG
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(data["checked"]))


def test_the_asset_names_follow_the_naming_of_the_release_workflow() -> None:
    """``llama-<tag>-bin-win-<build>-x64.zip`` and ``cudart-llama-bin-win-cuda-<cuda>-x64.zip``
    (the runtime archive has no tag in its name: checked in release-publish.yml of b11539)."""
    data = lock()
    tag = str(data["tag"])
    assets = data["assets"]
    assert isinstance(assets, dict) and set(assets) == {"cpu", "cuda-12.4", "cuda-13.4"}
    assert assets["cpu"]["archive"]["name"] == f"llama-{tag}-bin-win-cpu-x64.zip"
    assert "runtime" not in assets["cpu"]
    for cuda in ("12.4", "13.4"):
        entry = assets[f"cuda-{cuda}"]
        assert entry["archive"]["name"] == f"llama-{tag}-bin-win-cuda-{cuda}-x64.zip"
        assert entry["runtime"]["name"] == f"cudart-llama-bin-win-cuda-{cuda}-x64.zip"


def test_every_asset_has_a_sha256_and_a_size() -> None:
    assets = lock()["assets"]
    assert isinstance(assets, dict)
    seen: set[str] = set()
    for entry in assets.values():
        for asset in entry.values():
            assert re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]), asset["name"]
            assert isinstance(asset["size"], int) and asset["size"] > 1_000_000
            seen.add(asset["sha256"])
    assert len(seen) == 5  # five different files


def test_the_downloaded_binaries_are_not_committed() -> None:
    ignored = {
        line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "tools/" in ignored
