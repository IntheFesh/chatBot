<#
.SYNOPSIS
    Installs wechat-twin on this Windows machine and registers it to start when you log on.

.DESCRIPTION
    Steps (each one stops the script at the first error):
      1. checks for uv and installs it with the official installer when it is missing;
      2. uv sync --frozen            (the locked environment; never an unlocked resolve);
      3. twin setup                  (consent, DeepSeek key, SMTP, target, time zone, emergency contact);
      4. twin db upgrade             (database migrations);
      5. twin service install        (the scheduled task: logon trigger, InteractiveToken, no time
                                      limit, runs on battery, one instance, restart every minute on
                                      failure; the action is <repo>\.venv\Scripts\twin.exe supervise).

    The task runs only while you are logged on: the Windows credential manager, notifications and
    the QR-code window all need your session.  For an unattended recovery after a power cut or an
    update restart you need Windows automatic logon; `twin service install` explains the risk.

    The script changes no data.  `scripts\windows\uninstall.ps1` removes the task again.

.PARAMETER SkipSetup
    Do not run the interactive `twin setup` wizard (already configured, or configuring by hand).

.PARAMETER StartNow
    Start the scheduled task right after registering it (otherwise it starts at your next logon).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\install.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipSetup,
    [switch]$StartNow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Console and child processes talk UTF-8 (Chinese output, Chinese paths).
$env:PYTHONUTF8 = '1'
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    Write-Verbose 'The console encoding could not be changed; output may look garbled.'
}

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Write-Step {
    param([string]$Text)
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Invoke-Native {
    # Runs an executable and stops the script when it exits with a non-zero code.
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$File,
        [string[]]$Arguments = @()
    )
    Write-Step $Label
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed (exit code $LASTEXITCODE)."
    }
}

function Find-Uv {
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $found) {
        return $found.Source
    }
    foreach ($candidate in @(
            (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
            (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe'))) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

if ($PSVersionTable.PSVersion -lt [Version]'5.1') {
    throw 'Windows PowerShell 5.1 or newer is required.'
}
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'install.ps1 is for Windows only.'
}
if (-not (Test-Path -LiteralPath (Join-Path $Root 'pyproject.toml'))) {
    throw "pyproject.toml was not found in $Root; run the script from a full checkout."
}
Set-Location -LiteralPath $Root

# 1. uv -----------------------------------------------------------------------------------------
Write-Step 'Checking for uv'
$Uv = Find-Uv
if ($null -eq $Uv) {
    # The official installer, run exactly as its documentation says and in a process of its own
    # (it sets its own execution policy and may call exit).
    Invoke-Native -Label 'Installing uv with the official installer (astral.sh)' -File 'powershell.exe' -Arguments @(
        '-NoProfile', '-ExecutionPolicy', 'ByPass', '-Command', 'irm https://astral.sh/uv/install.ps1 | iex'
    )
    $env:Path = (Join-Path $env:USERPROFILE '.local\bin') + ';' + $env:Path
    $Uv = Find-Uv
    if ($null -eq $Uv) {
        throw 'uv was installed but cannot be found; open a new PowerShell window and run this script again.'
    }
}
Write-Host "uv: $Uv"

# 2. the locked environment ----------------------------------------------------------------------
Invoke-Native -Label 'Installing the locked environment (uv sync --frozen)' -File $Uv -Arguments @('sync', '--frozen')

$Twin = Join-Path $Root '.venv\Scripts\twin.exe'
if (-not (Test-Path -LiteralPath $Twin)) {
    throw "$Twin was not created by uv sync."
}

# 3. the wizard ----------------------------------------------------------------------------------
if ($SkipSetup) {
    Write-Step 'Skipping the setup wizard (-SkipSetup)'
} else {
    Invoke-Native -Label 'Setup wizard (twin setup)' -File $Twin -Arguments @('setup')
}

# 4. the database --------------------------------------------------------------------------------
Invoke-Native -Label 'Database migrations (twin db upgrade)' -File $Twin -Arguments @('db', 'upgrade')

# 5. the scheduled task --------------------------------------------------------------------------
Invoke-Native -Label 'Registering the scheduled task (twin service install)' -File $Twin -Arguments @('service', 'install')

if ($StartNow) {
    Invoke-Native -Label 'Starting the scheduled task (twin service start)' -File $Twin -Arguments @('service', 'start')
}

Invoke-Native -Label 'Status (twin service status)' -File $Twin -Arguments @('service', 'status')

Write-Host ''
Write-Host 'Installed.' -ForegroundColor Green
if (-not $StartNow) {
    Write-Host 'It starts at your next logon; start it now with:  .venv\Scripts\twin.exe service start'
}
Write-Host 'Check the installation with:  .venv\Scripts\twin.exe doctor    and    .venv\Scripts\twin.exe health'
Write-Host 'Remove the task again with:   scripts\windows\uninstall.ps1  (your data stays; `twin purge` deletes data)'
