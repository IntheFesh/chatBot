# scripts/windows/

PowerShell scripts for the Windows machine.

| Script | Round | What it does |
| --- | --- | --- |
| `install.ps1` | 12 | Installs uv if it is missing, `uv sync --frozen`, runs the `twin setup` wizard, `twin db upgrade`, registers the scheduled task with `twin service install`. `-SkipSetup` skips the wizard, `-StartNow` starts the task. |
| `uninstall.ps1` | 12 | `twin service stop`, then `twin service uninstall`. The data is not touched (`twin purge` deletes data). |
| `get_llamacpp.ps1` | 14 | Downloads the pinned llama.cpp release (`llamacpp.lock.json`), chooses the CPU, CUDA 12.4 or CUDA 13.4 build for the card (`nvidia-smi`), downloads the matching `cudart` package for a CUDA build, verifies every download against the SHA-256 of the lock file and unpacks to `tools\llama.cpp\<tag>\` (ignored by git). `-Build cpu\|cuda-12.4\|cuda-13.4` overrides the choice, `-Force` installs again. |
| `llamacpp.lock.json` | 14 | The pinned tag and the size and SHA-256 of each release asset, computed from the downloads of 2026-10-10. The script refuses to run without it; raise the version by downloading again and changing the lock and `training/autodl/versions.env` together. |

The scripts are plain ASCII (Windows PowerShell 5.1 reads a file without a byte order mark in the
ANSI code page), use `Set-StrictMode -Version Latest`, and stop at the first error.
`tests/unit/test_ops_scripts.py` checks the text of the first two and that every `twin` command they call exists; `tests/unit/test_serving_script.py` checks `get_llamacpp.ps1` and its lock file.
The task itself is defined by `twin.ops.taskscheduler.build_task_xml`; `twin service install --print-xml`
prints it. The quality-gate script `scripts/check.ps1` lives one level up.
