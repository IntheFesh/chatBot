# ruff: noqa
"""Which import order makes `import torch` fail (WinError 1114)?  One fresh process per case."""

import os
import subprocess
import sys

SYSTEM = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")

CASES = {
    "torch alone": "import torch",
    "windows_toasts, then torch": "import windows_toasts; import torch",
    "torch, then windows_toasts": "import torch; import windows_toasts",
    "numpy+pyarrow+lancedb, then torch": "import numpy, pyarrow, lancedb; import torch",
    "preload system msvcp140, windows_toasts, torch": (
        "import ctypes, os;"
        f"[ctypes.WinDLL(os.path.join({SYSTEM!r}, n)) for n in ('msvcp140.dll',)];"
        "import windows_toasts; import torch"
    ),
    "preload system msvcp140 family, windows_toasts, torch": (
        "import ctypes, os;"
        f"[ctypes.WinDLL(os.path.join({SYSTEM!r}, n)) for n in "
        "('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll', "
        "'msvcp140_2.dll', 'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll')];"
        "import windows_toasts; import torch"
    ),
}

for label, code in CASES.items():
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    last = (done.stderr.strip().splitlines() or [""])[-1][:200]
    sys.stdout.write(f"CASE {label}: exit {done.returncode} {last}\n")
    sys.stdout.flush()

ps = (
    "foreach ($p in @("
    "'C:\\Windows\\System32\\msvcp140.dll',"
    "'C:\\Windows\\System32\\vcruntime140.dll',"
    "(Get-ChildItem -Recurse .venv\\Lib\\site-packages\\winrt -Filter MSVCP140.dll | Select -First 1).FullName,"
    "(Get-ChildItem -Recurse .venv\\Lib\\site-packages\\numpy.libs -Filter msvcp140*.dll | Select -First 1).FullName,"
    "(Get-ChildItem -Recurse .venv\\Lib\\site-packages\\pyarrow.libs -Filter msvcp140-*.dll | Select -First 1).FullName"
    ")) { if ($p) { '{0} {1}' -f $p, (Get-Item $p).VersionInfo.FileVersion } }"
)
done = subprocess.run(
    ["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, check=False
)
sys.stdout.write("VERSIONS\n" + done.stdout + done.stderr + "\n")
