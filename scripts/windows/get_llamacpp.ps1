<#
.SYNOPSIS
    Downloads the pinned llama.cpp release for Windows into tools\llama.cpp\<tag>\.

.DESCRIPTION
    The local style model is served by llama-server.exe (R-SRV-002).  This script

      1. reads scripts\windows\llamacpp.lock.json: the pinned release tag and, for every build, the
         file names, sizes and SHA-256 of the release assets.  Without a valid lock the script
         refuses to run - it never skips the check;
      2. looks at the graphics card with nvidia-smi (name, driver, memory, compute capability and
         the newest CUDA version the driver supports) and picks the build:
           cuda-13.4  the driver supports CUDA 13.0+ and the card is Turing (7.5) or newer;
                      this build has native kernels for RTX 50 (Blackwell) cards;
           cuda-12.4  the driver supports CUDA 12.4+ and cuda-13.4 is not possible;
           cpu        no NVIDIA card, or a driver older than CUDA 12.4;
      3. downloads the archive and, for a CUDA build, the cudart archive of the SAME CUDA version
         (the CUDA runtime DLLs: without them ggml-cuda.dll does not load), checks the SHA-256 of
         each against the lock (and against the digest of the GitHub release API when it answers),
         and unpacks both into one folder;
      4. checks that every file a working installation needs is there, writes build.json and moves
         the folder to tools\llama.cpp\<tag>\ (a half-finished download is never left there).

    twin doctor checks the same files, twin model serve and twin run find the newest folder below
    tools\llama.cpp by themselves.  The folder is ignored by git.  Nothing outside tools\llama.cpp
    is changed.

.PARAMETER Build
    auto (default), cuda-13.4, cuda-12.4 or cpu.  Use it to override the choice, e.g. cpu on a
    computer whose card should not be used.

.PARAMETER Force
    Download and unpack again although the folder exists.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\get_llamacpp.ps1
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cuda-13.4', 'cuda-12.4', 'cpu')]
    [string]$Build = 'auto',
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$env:PYTHONUTF8 = '1'
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    Write-Verbose 'The console encoding could not be changed; output may look garbled.'
}
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
    Write-Verbose 'TLS 1.2 could not be enabled explicitly.'
}

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LockPath = Join-Path $PSScriptRoot 'llamacpp.lock.json'

# The rule for choosing a build; twin.serving.hardware.choose_build has the same numbers
# (tests/unit/test_serving_hardware.py compares them).
$Cuda13 = [Version]'13.0'
$Cuda124 = [Version]'12.4'
$Turing = [Version]'7.5'

# Files of a working installation (twin.serving.llamacpp has the same lists).
$BaseFiles = @('llama-server.exe', 'llama-server-impl.dll', 'llama.dll', 'ggml.dll', 'ggml-base.dll', 'llama-common.dll')
$RuntimeFiles = @{
    'cuda-12.4' = @('ggml-cuda.dll', 'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
    'cuda-13.4' = @('ggml-cuda.dll', 'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll')
}

function Write-Step {
    param([string]$Text)
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Test-Sha256 {
    param([string]$Value)
    return ($Value -match '^[0-9a-fA-F]{64}$')
}

function Read-Lock {
    if (-not (Test-Path -LiteralPath $LockPath)) {
        throw "The lock file $LockPath is missing. Without it the downloads cannot be checked, so the script does not run. Restore it from the repository."
    }
    $lock = Get-Content -LiteralPath $LockPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($lock.schema -ne 1) {
        throw 'The lock file has an unsupported schema.'
    }
    if ([string]::IsNullOrWhiteSpace($lock.tag) -or [string]::IsNullOrWhiteSpace($lock.repository)) {
        throw 'The lock file names no release tag; it must be filled in before the script can run.'
    }
    foreach ($name in @('cpu', 'cuda-12.4', 'cuda-13.4')) {
        if (-not ($lock.assets.PSObject.Properties.Name -contains $name)) {
            throw "The lock file has no entry for the build $name."
        }
        foreach ($part in $lock.assets.$name.PSObject.Properties) {
            $asset = $part.Value
            if (-not (Test-Sha256 $asset.sha256)) {
                throw "The lock file has no valid SHA-256 for $($asset.name); the script refuses to download without one."
            }
        }
    }
    return $lock
}

function Invoke-Smi {
    # nvidia-smi writes to stderr when it fails; that must not stop the script.
    param([string[]]$Arguments)
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $found = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($null -eq $found) {
            return $null
        }
        $output = & $found.Source @Arguments 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return $output
    } finally {
        $ErrorActionPreference = $saved
    }
}

function Get-GpuInfo {
    # name, driver, memory (MiB), compute capability (or $null), CUDA version the driver supports (or $null)
    $query = Invoke-Smi -Arguments @('--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader,nounits')
    if ($null -eq $query) {
        $query = Invoke-Smi -Arguments @('--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits')
    }
    if ($null -eq $query) {
        return $null
    }
    $best = $null
    foreach ($line in @($query)) {
        $cells = @(([string]$line).Split(',') | ForEach-Object { $_.Trim() })
        if ($cells.Count -lt 3 -or $cells[2] -notmatch '^\d+$') {
            continue
        }
        $capability = $null
        if ($cells.Count -gt 3 -and $cells[3] -match '^\d+\.\d+$') {
            $capability = [Version]$cells[3]
        }
        $card = [pscustomobject]@{
            Name = $cells[0]; Driver = $cells[1]; MemoryMiB = [int]$cells[2]; Capability = $capability; Cuda = $null
        }
        if ($null -eq $best -or $card.MemoryMiB -gt $best.MemoryMiB) {
            $best = $card
        }
    }
    if ($null -eq $best) {
        return $null
    }
    $plain = Invoke-Smi -Arguments @()
    if ($null -ne $plain) {
        $header = [regex]::Match(($plain -join "`n"), 'CUDA Version:\s*(\d+)\.(\d+)')
        if ($header.Success) {
            $best.Cuda = [Version]("$($header.Groups[1].Value).$($header.Groups[2].Value)")
        }
    }
    return $best
}

function Select-Build {
    param($Card)
    if ($null -eq $Card) {
        return @{ Kind = 'cpu'; Reason = 'no NVIDIA card was found' }
    }
    if ($null -eq $Card.Cuda) {
        return @{ Kind = 'cpu'; Reason = 'the CUDA version of the driver is unknown' }
    }
    $modern = ($null -eq $Card.Capability) -or ($Card.Capability -ge $Turing)
    if (($Card.Cuda -ge $Cuda13) -and $modern) {
        return @{ Kind = 'cuda-13.4'; Reason = 'the driver supports CUDA 13 (native kernels for RTX 50 cards)' }
    }
    if ($Card.Cuda -ge $Cuda124) {
        return @{ Kind = 'cuda-12.4'; Reason = 'the driver supports CUDA 12.4' }
    }
    return @{ Kind = 'cpu'; Reason = "the driver supports CUDA $($Card.Cuda) only" }
}

function Get-ReleaseDigests {
    # The digests the GitHub release API reports for the assets of the tag (may be unavailable).
    param([string]$Repository, [string]$Tag)
    $digests = @{}
    try {
        $headers = @{ 'User-Agent' = 'wechat-twin-get-llamacpp'; 'Accept' = 'application/vnd.github+json' }
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repository/releases/tags/$Tag" -Headers $headers -TimeoutSec 30
        foreach ($asset in $release.assets) {
            if ($asset.PSObject.Properties.Name -contains 'digest' -and -not [string]::IsNullOrWhiteSpace($asset.digest)) {
                $digests[[string]$asset.name] = ([string]$asset.digest).Replace('sha256:', '').ToLowerInvariant()
            }
        }
    } catch {
        Write-Host 'The GitHub release API did not answer; the checksums of the lock file are used alone.' -ForegroundColor Yellow
    }
    return $digests
}

function Get-VerifiedAsset {
    param($Asset, [string]$Repository, [string]$Tag, [string]$Folder, [hashtable]$Digests)
    $target = Join-Path $Folder $Asset.name
    $url = "https://github.com/$Repository/releases/download/$Tag/$($Asset.name)"
    $megabytes = [math]::Round($Asset.size / 1MB)
    Write-Host "Downloading $($Asset.name) (about $megabytes MB) ..."
    Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing
    $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Asset.sha256.ToLowerInvariant()) {
        Remove-Item -LiteralPath $target -Force
        throw "SHA-256 of $($Asset.name) is $actual, the lock file says $($Asset.sha256). The download was deleted."
    }
    if ($Digests.ContainsKey($Asset.name) -and $Digests[$Asset.name] -ne $actual) {
        Remove-Item -LiteralPath $target -Force
        throw "The GitHub release API reports another digest for $($Asset.name) than the lock file. The download was deleted."
    }
    Write-Host "  SHA-256 ok: $actual"
    return $target
}

function Expand-Zip {
    param([string]$Archive, [string]$Destination)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Archive, $Destination)
}

function Test-Installation {
    param([string]$Folder, [string]$Kind)
    $needed = @($BaseFiles)
    if ($Kind -eq 'cpu') {
        if (@(Get-ChildItem -LiteralPath $Folder -Filter 'ggml-cpu*.dll' -ErrorAction SilentlyContinue).Count -eq 0) {
            $needed += 'ggml-cpu*.dll'
        }
    } else {
        $needed += $RuntimeFiles[$Kind]
    }
    $missing = @()
    foreach ($name in $needed) {
        if (-not (Test-Path -LiteralPath (Join-Path $Folder $name))) {
            $missing += $name
        }
    }
    return $missing
}

if ($PSVersionTable.PSVersion -lt [Version]'5.1') {
    throw 'Windows PowerShell 5.1 or newer is required.'
}
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'get_llamacpp.ps1 is for Windows only.'
}

Write-Step 'Reading the lock file'
$Lock = Read-Lock
$Tag = [string]$Lock.tag
$Repository = [string]$Lock.repository
Write-Host "llama.cpp release $Tag ($Repository)"

Write-Step 'Choosing the build'
$Card = Get-GpuInfo
if ($null -eq $Card) {
    Write-Host 'No NVIDIA card found (nvidia-smi is missing or failed).'
} else {
    $capabilityText = if ($null -ne $Card.Capability) { $Card.Capability.ToString() } else { 'unknown' }
    $cudaText = if ($null -ne $Card.Cuda) { $Card.Cuda.ToString() } else { 'unknown' }
    Write-Host "Card: $($Card.Name), $($Card.MemoryMiB) MiB, driver $($Card.Driver), compute capability $capabilityText, driver supports CUDA $cudaText"
}
if ($Build -eq 'auto') {
    $Choice = Select-Build -Card $Card
    $Kind = $Choice.Kind
    Write-Host "Build: $Kind ($($Choice.Reason))"
} else {
    $Kind = $Build
    Write-Host "Build: $Kind (chosen with -Build)"
}

$ToolsDir = Join-Path (Join-Path $Root 'tools') 'llama.cpp'
$Final = Join-Path $ToolsDir $Tag
$Staging = Join-Path $ToolsDir ('.staging-' + $Tag)
$Downloads = Join-Path $ToolsDir '.downloads'

if ((Test-Path -LiteralPath (Join-Path $Final 'build.json')) -and -not $Force) {
    $Existing = Get-Content -LiteralPath (Join-Path $Final 'build.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($Existing.build -eq $Kind -and @(Test-Installation -Folder $Final -Kind $Kind).Count -eq 0) {
        Write-Host "Already installed: $Final ($Kind). Use -Force to download again." -ForegroundColor Green
        return
    }
}

New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
foreach ($leftover in @($Staging, $Downloads)) {
    if (Test-Path -LiteralPath $leftover) {
        Remove-Item -LiteralPath $leftover -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $Staging | Out-Null
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null

try {
    Write-Step 'Downloading and checking'
    $Digests = Get-ReleaseDigests -Repository $Repository -Tag $Tag
    $Parts = $Lock.assets.$Kind
    $Archive = Get-VerifiedAsset -Asset $Parts.archive -Repository $Repository -Tag $Tag -Folder $Downloads -Digests $Digests
    Write-Step 'Unpacking'
    Expand-Zip -Archive $Archive -Destination $Staging
    if ($Parts.PSObject.Properties.Name -contains 'runtime') {
        $Runtime = Get-VerifiedAsset -Asset $Parts.runtime -Repository $Repository -Tag $Tag -Folder $Downloads -Digests $Digests
        Expand-Zip -Archive $Runtime -Destination $Staging
    }

    Write-Step 'Checking the installation'
    $Missing = @(Test-Installation -Folder $Staging -Kind $Kind)
    if ($Missing.Count -gt 0) {
        throw "The unpacked folder lacks: $($Missing -join ', ')."
    }
    $Marker = [ordered]@{
        tag = $Tag
        build = $Kind
        archive = $Parts.archive.name
        archive_sha256 = $Parts.archive.sha256
        installed = (Get-Date).ToUniversalTime().ToString('o')
    }
    if ($Parts.PSObject.Properties.Name -contains 'runtime') {
        $Marker['runtime'] = $Parts.runtime.name
        $Marker['runtime_sha256'] = $Parts.runtime.sha256
    }
    $Marker | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Staging 'build.json') -Encoding UTF8

    if (Test-Path -LiteralPath $Final) {
        Remove-Item -LiteralPath $Final -Recurse -Force
    }
    Move-Item -LiteralPath $Staging -Destination $Final
} finally {
    foreach ($leftover in @($Staging, $Downloads)) {
        if (Test-Path -LiteralPath $leftover) {
            Remove-Item -LiteralPath $leftover -Recurse -Force
        }
    }
}

Write-Host ''
Write-Host "Installed llama.cpp $Tag ($Kind) in $Final" -ForegroundColor Green
Write-Host 'Next: uv run twin doctor    (checks the files)'
Write-Host '      uv run twin model recommend    (which quantisation fits the card)'
Write-Host '      uv run twin model serve <model id>    (starts the server in this window)'
