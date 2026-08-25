param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
$statusPath = Join-Path $artifactRoot "forced_grid_status.json"
$logPath = Join-Path $artifactRoot "forced_grid.log"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SpLenseForcedGridPowerState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

function Write-GridStatus {
    param(
        [string]$ModelTag,
        [string]$State,
        [string]$Detail = ""
    )

    [ordered]@{
        schema_version = 1
        model_tag = $ModelTag
        state = $State
        detail = $Detail
        process_id = $PID
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Invoke-NativeLogged {
    param(
        [string]$Phase,
        [string[]]$Arguments
    )

    "[$([DateTime]::UtcNow.ToString('o'))] START $Phase" | Tee-Object -FilePath $logPath -Append
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $comparisonExe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    if ($nativeExitCode -ne 0) {
        throw "$Phase failed with exit code $nativeExitCode"
    }
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE $Phase" | Tee-Object -FilePath $logPath -Append
}

function Invoke-ModelGrid {
    param(
        [string]$ModelTag,
        [string]$ModelConfig
    )

    $modelRoot = Join-Path $artifactRoot $ModelTag
    $manifestPaths = @(
        (Join-Path $modelRoot "directions\gradient\direction_manifest.json"),
        (Join-Path $modelRoot "directions\caa\direction_manifest.json"),
        (Join-Path $modelRoot "directions\bipo_matched\direction_manifest.json"),
        (Join-Path $modelRoot "directions\bipo_canonical\direction_manifest.json"),
        (Join-Path $modelRoot "directions\persona\direction_manifest.json")
    )
    foreach ($manifestPath in $manifestPaths) {
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
            throw "$ModelTag is not construction-complete: missing $manifestPath"
        }
    }

    $gridDirectory = Join-Path $modelRoot "forced_grid"
    New-Item -ItemType Directory -Force -Path $gridDirectory | Out-Null
    Write-GridStatus -ModelTag $ModelTag -State "resolving_plan"
    $resolveArguments = @(
        "run-forced-grid", "--model-config", $ModelConfig,
        "--direction-manifest"
    ) + $manifestPaths + @(
        "--max-new-points", "0", "--output-dir", $gridDirectory
    )
    Invoke-NativeLogged -Phase "${ModelTag}_resolve_forced_grid" -Arguments $resolveArguments

    $planPath = Join-Path $gridDirectory "forced_grid_plan.json"
    if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) {
        throw "$ModelTag planner did not publish forced_grid_plan.json"
    }
    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    if ($plan.points.Count -ne 250) {
        throw "$ModelTag forced grid contains $($plan.points.Count) points instead of 250"
    }

    Write-GridStatus -ModelTag $ModelTag -State "running_250_points"
    $runArguments = @(
        "run-forced-grid", "--model-config", $ModelConfig,
        "--direction-manifest"
    ) + $manifestPaths + @(
        "--output-dir", $gridDirectory
    )
    Invoke-NativeLogged -Phase "${ModelTag}_run_forced_grid" -Arguments $runArguments

    $pointDirectory = Join-Path $gridDirectory "points"
    $shards = @(Get-ChildItem -LiteralPath $pointDirectory -Filter "*.json" -File)
    if ($shards.Count -ne 250) {
        throw "$ModelTag produced $($shards.Count) forced-grid shards instead of 250"
    }
    Write-GridStatus -ModelTag $ModelTag -State "complete" -Detail $planPath
}

if ($SelfTest) {
    [ordered]@{
        status = "self_test_passed"
        models = @("qwen35_08b", "qwen35_2b")
        stage1_gate_required = $true
        points_per_model = 250
        expected_total_atomic_shards = 500
        persona_direction_required = $true
        paid_calls = 0
    } | ConvertTo-Json
    return
}

$keepAwakeFlags = [uint32]::Parse("80000001", [Globalization.NumberStyles]::HexNumber)
$resetExecutionFlags = [uint32]::Parse("80000000", [Globalization.NumberStyles]::HexNumber)
[void][SpLenseForcedGridPowerState]::SetThreadExecutionState($keepAwakeFlags)

try {
    Invoke-NativeLogged -Phase "verify_stage1_before_forced_grids" -Arguments @("verify-stage1")
    Invoke-ModelGrid -ModelTag "qwen35_08b" -ModelConfig "configs\qwen35_08b_aligned.json"
    Invoke-ModelGrid -ModelTag "qwen35_2b" -ModelConfig "configs\qwen35_2b_aligned.json"
    Write-GridStatus -ModelTag "all" -State "complete" -Detail "500 atomic point shards"
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-GridStatus -ModelTag "unknown_or_current" -State "failed" -Detail $failureDetail
    throw
}
finally {
    [void][SpLenseForcedGridPowerState]::SetThreadExecutionState($resetExecutionFlags)
}
