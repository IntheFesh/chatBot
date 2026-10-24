"""Finding llama.cpp, what a complete installation holds, the command line (R-SRV-002)."""

from __future__ import annotations

from pathlib import Path

import pytest

from twin.serving.llamacpp import (
    WINDOWS_BASE_FILES,
    ServeError,
    ServerSpec,
    detect_build,
    find_install,
    inspect_install,
    loopback_endpoint,
    missing_files,
    server_name,
    version_key,
)

CPU_DLLS = ("ggml-cpu-haswell.dll", "ggml-cpu-x64.dll")


def make_folder(
    root: Path, tag: str, *, kind: str = "cpu", skip: tuple[str, ...] = (), windows: bool = True
) -> Path:
    folder = root / "tools" / "llama.cpp" / tag
    folder.mkdir(parents=True)
    names = list(WINDOWS_BASE_FILES) if windows else ["llama-server"]
    if kind == "cpu":
        names += CPU_DLLS
    else:
        runtime = "13" if kind == "cuda-13.4" else "12"
        names += [
            "ggml-cuda.dll",
            f"cudart64_{runtime}.dll",
            f"cublas64_{runtime}.dll",
            f"cublasLt64_{runtime}.dll",
        ]
    for name in names:
        if name not in skip:
            (folder / name).write_bytes(b"x")
    return folder


def test_the_newest_release_folder_wins_by_its_number_not_by_its_spelling(tmp_path: Path) -> None:
    make_folder(tmp_path, "b999")
    make_folder(tmp_path, "b11177")
    make_folder(tmp_path, "b11539")
    found = find_install(tmp_path, platform="win32")
    assert found is not None and found.version == "b11539"
    assert found.binary.name == "llama-server.exe" and found.complete and found.kind == "cpu"
    assert version_key("b11539") > version_key("b999") > version_key("nothing-numeric")


def test_a_folder_without_the_server_is_not_an_installation(tmp_path: Path) -> None:
    make_folder(tmp_path, "b11177", skip=("llama-server.exe",))
    assert find_install(tmp_path, platform="win32") is None
    assert find_install(tmp_path / "elsewhere", platform="win32") is None


def test_the_configured_binary_is_used_even_when_it_is_somewhere_else(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "b11177", kind="cuda-12.4")
    elsewhere = tmp_path / "mine" / "llama-server.exe"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"x")
    found = find_install(tmp_path, str(elsewhere), platform="win32")
    assert found is not None and found.binary == elsewhere and not found.complete
    relative = find_install(tmp_path, "tools/llama.cpp/b11177/llama-server.exe", platform="win32")
    assert relative is not None and relative.binary == folder / "llama-server.exe"
    assert find_install(tmp_path, str(tmp_path / "missing.exe"), platform="win32") is None


@pytest.mark.parametrize(
    ("kind", "expected"), [("cpu", "cpu"), ("cuda-12.4", "cuda-12.4"), ("cuda-13.4", "cuda-13.4")]
)
def test_the_build_is_told_from_the_runtime_dlls(tmp_path: Path, kind: str, expected: str) -> None:
    folder = make_folder(tmp_path, "b1", kind=kind)
    assert detect_build(folder) == expected
    assert missing_files(folder, expected) == []


def test_a_cuda_build_without_the_cudart_zip_is_incomplete_and_says_so(tmp_path: Path) -> None:
    folder = make_folder(
        tmp_path,
        "b1",
        kind="cuda-13.4",
        skip=("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"),
    )
    assert detect_build(folder) == "cuda"
    missing = missing_files(folder, "cuda")
    assert len(missing) == 1 and "cudart zip" in missing[0]
    install = find_install(tmp_path, platform="win32")
    assert install is not None and not install.complete


def test_a_cuda_build_with_only_some_runtime_dlls_names_the_missing_ones(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "b1", kind="cuda-12.4", skip=("cublasLt64_12.dll",))
    assert missing_files(folder, detect_build(folder)) == ["cublasLt64_12.dll"]


def test_a_cpu_build_needs_a_cpu_backend_dll(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "b1", kind="cpu", skip=CPU_DLLS)
    assert "ggml-cpu*.dll" in missing_files(folder, "cpu")


def test_off_windows_the_file_list_is_not_checked(tmp_path: Path) -> None:
    make_folder(tmp_path, "b1", windows=False)
    found = find_install(tmp_path, platform="linux")
    assert found is not None and found.binary.name == "llama-server" and found.complete
    assert server_name("win32") == "llama-server.exe" and server_name("linux") == "llama-server"
    assert inspect_install(found.binary, platform="linux").missing == ()


def test_the_command_line_has_the_documented_flags_and_listens_on_this_computer_only() -> None:
    spec = ServerSpec(("C:/tools/llama-server.exe",), Path("m.gguf"), 8081)
    assert spec.argv() == [
        "C:/tools/llama-server.exe",
        "-m",
        "m.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "-c",
        "4096",
        "-ngl",
        "999",
        "--parallel",
        "1",
        "--chat-template",
        "chatml",
        "--no-webui",
    ]


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.5", "example.com", "::"],  # noqa: S104 - addresses that are refused
)
def test_a_server_never_listens_on_another_address(host: str) -> None:
    with pytest.raises(ServeError, match="this computer only"):
        ServerSpec(("llama-server",), Path("m.gguf"), 8081, host=host)


def test_only_a_loopback_endpoint_can_be_managed() -> None:
    assert loopback_endpoint("http://127.0.0.1:8081") == ("127.0.0.1", 8081)
    assert loopback_endpoint("http://localhost:9000/") == ("127.0.0.1", 9000)
    with pytest.raises(ServeError, match="not on this computer"):
        loopback_endpoint("http://192.168.0.7:8081")
    with pytest.raises(ServeError, match="no port"):
        loopback_endpoint("http://127.0.0.1")
