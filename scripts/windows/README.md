# scripts/windows/

PowerShell scripts for the Windows machine.

| Script | Round | What it does |
| --- | --- | --- |
| `install.ps1` | 12 | Installs uv if it is missing, `uv sync --frozen`, runs the `twin setup` wizard, `twin db upgrade`, registers the scheduled task with `twin service install`. `-SkipSetup` skips the wizard, `-StartNow` starts the task. |
| `uninstall.ps1` | 12 | `twin service stop`, then `twin service uninstall`. The data is not touched (`twin purge` deletes data). |
| llama.cpp download helper | 14 | Not there yet. |

Both scripts are plain ASCII (Windows PowerShell 5.1 reads a file without a byte order mark in the
ANSI code page), use `Set-StrictMode -Version Latest`, and stop at the first error.
`tests/unit/test_ops_scripts.py` checks their text and that every `twin` command they call exists.
The task itself is defined by `twin.ops.taskscheduler.build_task_xml`; `twin service install --print-xml`
prints it. The quality-gate script `scripts/check.ps1` lives one level up.
