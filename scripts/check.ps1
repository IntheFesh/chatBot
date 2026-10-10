# One-shot local quality gate (mirrors .github/workflows/ci.yml).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/check.ps1 [-Round 00]
param(
    [string]$Round = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = "1"

function Invoke-Step {
    param([string]$Name, [scriptblock]$Command)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Step "uv sync" { uv sync --frozen }
Invoke-Step "ruff check" { uv run ruff check . }
Invoke-Step "ruff format --check" { uv run ruff format --check . }
Invoke-Step "mypy (strict, src/twin)" { uv run mypy src/twin }
Invoke-Step "pytest + coverage" { uv run pytest -q -m "not live" --cov=src/twin --cov-report=json }
Invoke-Step "coverage gate" { uv run python scripts/coverage_gate.py }
Invoke-Step "privacy scan" { uv run python scripts/privacy_scan.py }
Invoke-Step "stub scan" { uv run python scripts/stub_scan.py }
Invoke-Step "decisions check" { uv run python scripts/decisions_check.py }
Invoke-Step "twin --help" { uv run twin --help }
Invoke-Step "twin doctor" { uv run twin doctor }
if ($Round -ne "") {
    Invoke-Step "trace_check --round $Round" { uv run python scripts/trace_check.py --round $Round }
} else {
    Invoke-Step "trace_check (global)" { uv run python scripts/trace_check.py }
}
Write-Host "All checks passed." -ForegroundColor Green
