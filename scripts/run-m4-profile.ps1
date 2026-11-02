<#
.SYNOPSIS
  ExactMoE M4 metadata collection for Windows (no model weight download).
.DESCRIPTION
  Downloads only public config.json and model.safetensors.index.json from a pinned
  GLM-5.3 revision using the repository's metadata profiler.
  Produces a JSON profile and a separate, optional storage/GPU inventory.
  Does NOT download safetensors model weight shards, execute model inference,
  modify drivers, or change model files.
#>
[CmdletBinding()]
param(
    [string] $Revision = "aca966e4e02791568aa6a4ced368624b3d897f42",
    [string] $ModelId = "zai-org/GLM-5.3",
    [switch] $SkipInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $outDir = Join-Path $repoRoot "artifacts\m4"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $pythonExe = $py.Source
        $pythonFlags = @("-3")
    }
    else {
        $python = Get-Command python -ErrorAction Stop
        $pythonExe = $python.Source
        $pythonFlags = @()
    }

    Write-Host "ExactMoE M4: $ModelId @ $Revision"
    Write-Host "Only JSON metadata will be downloaded (no model weights)."
    if (-not $SkipInstall) {
        & $pythonExe @pythonFlags -m pip install -e ".[research]"
        if ($LASTEXITCODE -ne 0) { throw "Python dependency install failed." }
    }

    $jsonLines = & $pythonExe @pythonFlags -m exactmoe.cli profile --hf $ModelId --revision $Revision
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint metadata profiler failed." }
    $json = ($jsonLines -join "`n")
    $profile = $json | ConvertFrom-Json

    $profilePath = Join-Path $outDir "glm53-profile.json"
    $json | Set-Content -LiteralPath $profilePath -Encoding UTF8

    $storage = @(
        Get-CimInstance Win32_LogicalDisk -Filter "DriveType = 3" |
        Select-Object DeviceID,
            @{ Name = "free_gb"; Expression = { [math]::Round($_.FreeSpace / 1GB, 2) } },
            @{ Name = "capacity_gb"; Expression = { [math]::Round($_.Size / 1GB, 2) } }
    )
    $gpuInfo = $null
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpuInfo = @(& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null)
    }
    $systemReport = [ordered]@{
        note = "Local disk and GPU inventory only; no credentials, personal files, or model weights."
        revision = $Revision
        disks = $storage
        gpu = $gpuInfo
    }
    $systemPath = Join-Path $outDir "hardware.json"
    $systemReport | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $systemPath -Encoding UTF8

    $zipPath = Join-Path $outDir "ExactMoE-M4-Report.zip"
    Compress-Archive -LiteralPath @($profilePath, $systemPath) -DestinationPath $zipPath -Force

    Write-Host ""
    Write-Host "M4 metadata audit complete." -ForegroundColor Green
    Write-Host ("Model: " + $ModelId)
    Write-Host ("Shards: " + $profile.shard_count)
    Write-Host ("Named expert metadata complete: " + $profile.expert_layout.metadata_complete)
    Write-Host ""
    Write-Host ("Upload this report to the chat: " + $zipPath) -ForegroundColor Cyan
}
finally {
    Pop-Location
}
