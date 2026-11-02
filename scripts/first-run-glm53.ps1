<#
.SYNOPSIS
  Read-only, preflight checklist for original pinned GLM-5.3 + ExactMoE.
.DESCRIPTION
  Inspects local hardware basics, target free disk, official config/index,
  existing shards and one expert's shard plan. It never downloads model
  weights, installs packages, deletes files or changes system settings.
#>
[CmdletBinding()]
param([string] $ModelDir = "C:\Models\GLM-5.3")

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$revision = "aca966e4e02791568aa6a4ced368624b3d897f42"
$checkpointBytes = 755617140416
$recommendedGiB = 850
$minFreeGiB = [math]::Round($checkpointBytes / 1GB, 2)

Write-Host "ExactMoE GLM-5.3 preflight (READ ONLY)" -ForegroundColor Cyan
Write-Host "Pinned HF revision: $revision"
Write-Host "Model dir: $ModelDir"
Write-Host "No installation, download or deletion will occur."
Write-Host ""

$drive = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($ModelDir))
if ($drive) {
    try {
        $storage = [System.IO.DriveInfo]::new($drive)
        if ($storage.IsReady) {
            $freeGiB = [math]::Round($storage.AvailableFreeSpace / 1GB, 2)
            Write-Host "Drive free space: $freeGiB GiB (checkpoint payload: $minFreeGiB GiB)"
            if ($freeGiB -lt $recommendedGiB) {
                Write-Warning ("For the full checkpoint, aim for >= $recommendedGiB GiB free " +
                               "BEFORE starting, accounting for any already downloaded shards. " +
                               "If a download is in progress, inspect its size before deleting anything.")
            }
        }
    } catch {
        Write-Warning "Could not read target drive free space: $($_.Exception.Message)"
    }
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host ""
    Write-Host "NVIDIA GPU (name, VRAM MiB, driver):"
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
} else {
    Write-Warning "nvidia-smi not found (CPU-only metadata checks still possible)."
}

$pythonExe = $null
$pythonFlags = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command py).Source
    $pythonFlags = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command python).Source
}
if (-not $pythonExe) {
    Write-Warning "Python not found. Install Python 3.11/3.12, then run py -3 -m pip install -e '.[dev]' from repo root."
}

$index = Join-Path $ModelDir "model.safetensors.index.json"
$configFile = Join-Path $ModelDir "config.json"
if (-not (Test-Path -LiteralPath $index -PathType Leaf)) {
    Write-Warning "No official shard index at model location. For metadata only run:"
    Write-Host "hf download zai-org/GLM-5.3 config.json model.safetensors.index.json --revision $revision --local-dir `"$ModelDir`""
    Write-Host "Do NOT start a second simultaneous full download in another directory."
    Write-Host "Next step: rerun this script when model index exists."
    exit 0
}

if (Test-Path -LiteralPath $configFile -PathType Leaf) {
    try {
        $config = Get-Content -LiteralPath $configFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $block = $config.quantization_config.weight_block_size
        $valid = (
            $config.model_type -eq "glm_moe_dsa" -and
            $config.hidden_size -eq 6144 -and
            $config.num_hidden_layers -eq 78 -and
            $config.n_routed_experts -eq 256 -and
            $config.num_experts_per_tok -eq 8 -and
            $config.quantization_config.quant_method -eq "fp8" -and
            $block.Count -eq 2 -and $block[0] -eq 128 -and $block[1] -eq 128
        )
        if ($valid) {
            Write-Host "GLM-5.3 structural config: PASS (not a revision/hash verification)." -ForegroundColor Green
        } else {
            Write-Warning "GLM config deviates from expected pinned native-FP8 architecture. Stop before inference."
        }
    } catch {
        Write-Warning "Could not parse config.json: $($_.Exception.Message)"
    }
} else {
    Write-Warning "config.json missing; expert-plan can still inspect weight index."
}

if ($pythonExe) {
    Push-Location $repo
    try {
        Write-Host ""
        Write-Host "Download/checkpoint structural progress:"
        & $pythonExe @pythonFlags -m exactmoe.cli checkpoint-status $ModelDir
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "checkpoint-status failed. Check directory/index and Python dependencies."
        }
        Write-Host ""
        Write-Host "Original layer-3, expert-0 six-tensor shard plan:"
        & $pythonExe @pythonFlags -m exactmoe.cli expert-plan $ModelDir --layer 3 --expert 0
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "expert-plan failed. Check the official weight_map."
        }
        Write-Host ""
        Write-Host "PyTorch CUDA availability (optional):"
        & $pythonExe @pythonFlags -c "import importlib.util; print('torch not installed' if importlib.util.find_spec('torch') is None else 'torch installed; check CUDA separately')"
    } finally {
        Pop-Location
    }
}
Write-Host ""
Write-Host "Checkpoint bytes in place and validated metadata do NOT mean the full GLM decoder runs."
Write-Host "Next experimental gate: expert-smoke using completed official shards."
