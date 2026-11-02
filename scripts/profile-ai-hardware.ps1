<#
.SYNOPSIS
  Read-only CPU/NPU/GPU/RAM platform inventory for ExactMoE.
.DESCRIPTION
  Writes a local JSON report. This script does not install drivers, change BIOS,
  adjust power limits, pin CPU affinity, download weights, or run large models.
  Intentionally excludes usernames, system serial numbers and device IDs.
#>
[CmdletBinding()]
param(
    [string] $OutputPath = "",
    [switch] $CheckOpenVINO
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $repoRoot "artifacts\hardware\ai-platform.json"
}

# Read relevant capabilities without making any configuration changes.
$processors = @(
    Get-CimInstance Win32_Processor |
        Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
)

$memoryModules = @(
    Get-CimInstance Win32_PhysicalMemory |
        Select-Object @{Name="capacity_gib";Expression={[math]::Round($_.Capacity / 1GB,2)}},
            Speed, ConfiguredClockSpeed, Manufacturer, SMBIOSMemoryType
)
$memoryTotalGiB = [math]::Round(
    (($memoryModules | Measure-Object -Property capacity_gib -Sum).Sum), 2
)

$volumes = @(
    Get-CimInstance Win32_LogicalDisk -Filter "DriveType = 3" |
        Select-Object DeviceID,
            @{Name="total_gib";Expression={[math]::Round($_.Size/1GB,2)}},
            @{Name="free_gib";Expression={[math]::Round($_.FreeSpace/1GB,2)}}
)

$npuCandidates = @()
try {
    $npuCandidates = @(
        Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
            Where-Object { $_.FriendlyName -match '(?i)(NPU|Neural|AI Boost)' } |
            Select-Object FriendlyName, Status, Class
    )
} catch {
    # Device enumeration may be unavailable in restricted Windows environments.
}

$gpu = @()
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $lines = @(& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null)
        if ($LASTEXITCODE -eq 0) {
            foreach ($line in $lines) {
                $parts = $line -split ',', 3
                if ($parts.Count -eq 3) {
                    $gpu += [ordered]@{
                        name = $parts[0].Trim()
                        vram_mib = $parts[1].Trim()
                        driver_version = $parts[2].Trim()
                    }
                }
            }
        }
    } catch { }
}

$ovReport = [ordered]@{
    checked = [bool]$CheckOpenVINO
    status = "not_checked"
    available_devices = @()
}
if ($CheckOpenVINO) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($py -or $python) {
        $exe = if ($py) { $py.Source } else { $python.Source }
        $prefix = if ($py) { @("-3") } else { @() }
        try {
            $lines = @(& $exe @prefix -c 'import json; from openvino import Core; print(json.dumps(Core().available_devices))' 2>$null)
            if ($LASTEXITCODE -eq 0) {
                $ovReport.status = "ok"
                $ovReport.available_devices = @(($lines -join "") | ConvertFrom-Json)
            } else {
                $ovReport.status = "unavailable_or_not_installed"
            }
        } catch {
            $ovReport.status = "unavailable_or_not_installed"
        }
    } else {
        $ovReport.status = "python_not_found"
    }
}

$osVersion = (Get-CimInstance Win32_OperatingSystem).Version
$report = [ordered]@{
    schema_version = 1
    report_type = "exactmoe_hardware_capabilities"
    purpose = "Scheduling research only; no performance claims."
    windows_version = $osVersion
    processor = $processors
    physical_memory_gib = $memoryTotalGiB
    memory_modules = $memoryModules
    local_volumes = $volumes
    npu_pnp_candidates = $npuCandidates
    openvino = $ovReport
    nvidia_gpu = $gpu
    caveats = @(
        "PnP absence does not prove NPU silicon is absent; drivers/firmware may be missing.",
        "ConfiguredClockSpeed is reported by Windows and may differ from kit marketing speed.",
        "Do not assume Intel P/E CPU core index ordering.",
        "This report contains no performance measurements or GLM inference results."
    )
}
$directory = Split-Path -Parent $OutputPath
if ($directory) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$report | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $OutputPath -Encoding UTF8
Write-Host "ExactMoE hardware report written to: $OutputPath"
Write-Host "Intel CPU: $($processors[0].Name)"
Write-Host "RAM: $memoryTotalGiB GiB; detected NPU devices: $($npuCandidates.Count)"
if ($CheckOpenVINO) {
    Write-Host "OpenVINO: $($ovReport.status); devices: $($ovReport.available_devices -join ', ')"
}
Write-Host "No machine settings were modified."
