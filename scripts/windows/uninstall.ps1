<#
.SYNOPSIS
    Removes the wechat-twin scheduled task.  Your data stays where it is.

.DESCRIPTION
    Stops the supervisor and the application (the reply being sent is finished first), then removes
    the scheduled task with `twin service uninstall`.  Nothing in data\ is touched: to delete her
    data use `twin purge --all` (it asks for a typed confirmation), to remove the program delete the
    checkout folder afterwards.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\uninstall.ps1
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$env:PYTHONUTF8 = '1'
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    Write-Verbose 'The console encoding could not be changed; output may look garbled.'
}

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TaskName = 'wechat-twin'
$Twin = Join-Path $Root '.venv\Scripts\twin.exe'

function Write-Step {
    param([string]$Text)
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Invoke-Native {
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

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'uninstall.ps1 is for Windows only.'
}
Set-Location -LiteralPath $Root

if (Test-Path -LiteralPath $Twin) {
    # `service uninstall` is an exclusive command: the application must not be running.
    Invoke-Native -Label 'Stopping the application (twin service stop)' -File $Twin -Arguments @('service', 'stop')
    Invoke-Native -Label 'Removing the scheduled task (twin service uninstall)' -File $Twin -Arguments @('service', 'uninstall')
} else {
    # The environment is already gone: remove the task directly so nothing keeps starting.
    Write-Step 'The environment is missing; removing the scheduled task with schtasks'
    # schtasks writes to stderr when the task does not exist; that is an answer here, not an error.
    $ErrorActionPreference = 'Continue'
    & schtasks.exe /Query /TN $TaskName 2>&1 | Out-Null
    $registered = ($LASTEXITCODE -eq 0)
    $deleted = $false
    if ($registered) {
        & schtasks.exe /End /TN $TaskName 2>&1 | Out-Null
        & schtasks.exe /Delete /TN $TaskName /F
        $deleted = ($LASTEXITCODE -eq 0)
    }
    $ErrorActionPreference = 'Stop'
    if ($registered -and -not $deleted) {
        throw 'schtasks /Delete failed.'
    }
    if ($registered) {
        Write-Host 'scheduled task removed'
    } else {
        Write-Host 'the scheduled task was not registered'
    }
}

Write-Host ''
Write-Host 'Uninstalled.  Your data was not touched.' -ForegroundColor Green
Write-Host 'To delete her data for good:  .venv\Scripts\twin.exe purge --all'
