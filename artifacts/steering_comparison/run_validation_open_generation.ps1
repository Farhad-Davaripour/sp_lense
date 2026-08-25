param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$pythonExe = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
$openRoot = Join-Path $artifactRoot "validation_open"
$orchestrator = Join-Path $artifactRoot "locked_open_orchestration.py"
$evaluationValidator = Join-Path $artifactRoot "validate_locked_evaluation_artifact.py"
$preopenManifest = Join-Path $RepositoryRoot "configs\steering_comparison_preopen_lock.json"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$planPath = Join-Path $openRoot "validation_open_plan.json"
$combinedPath = Join-Path $openRoot "open_generations_all.jsonl"
$requestsPath = Join-Path $openRoot "open_judge_requests.jsonl"
$statusPath = Join-Path $artifactRoot "validation_open_generation_status.json"
$logPath = Join-Path $artifactRoot "validation_open_generation.log"
$zeroDigest = "0" * 64

function Convert-Invariant {
    param([double]$Value)
    return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Write-OpenStatus {
    param(
        [string]$State,
        [string]$Detail = ""
    )
    [ordered]@{
        schema_version = 1
        split = "validation"
        state = $State
        detail = $Detail
        process_id = $PID
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Invoke-NativeLogged {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$Phase
    )
    "[$([DateTime]::UtcNow.ToString('o'))] START $Phase" |
        Tee-Object -FilePath $logPath -Append
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    if ($nativeExitCode -ne 0) {
        throw "$Phase failed with exit code $nativeExitCode"
    }
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE $Phase" |
        Tee-Object -FilePath $logPath -Append
}

if ($SelfTest) {
    if ($zeroDigest.Length -ne 64 -or $zeroDigest.Trim("0").Length -ne 0) {
        throw "validation generation must use the all-zero pre-summary sentinel"
    }
    [ordered]@{
        status = "self_test_passed"
        split = "validation"
        preopen_gate_required = $true
        stage2_gate_used = $false
        sealed_forward_passes = 0
        calibration_summary_sha256 = $zeroDigest
        judge_calls = 0
    } | ConvertTo-Json
    return
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SpLenseValidationOpenPowerState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$keepAwakeFlags = [uint32]::Parse("80000001", [Globalization.NumberStyles]::HexNumber)
$resetExecutionFlags = [uint32]::Parse("80000000", [Globalization.NumberStyles]::HexNumber)
[void][SpLenseValidationOpenPowerState]::SetThreadExecutionState($keepAwakeFlags)

try {
    Write-OpenStatus -State "verifying_preopen"
    Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
        "verify-preopen", "--preopen-manifest", $preopenManifest
    ) -Phase "verify_preopen_before_validation_generation"
    foreach ($required in @($orchestrator, $evaluationValidator, $lockPath, $preopenManifest)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "validation-open prerequisite is missing: $required"
        }
    }

    New-Item -ItemType Directory -Force -Path $openRoot | Out-Null
    Invoke-NativeLogged -Executable $pythonExe -Arguments @(
        $orchestrator, "plan",
        "--repo-root", $RepositoryRoot,
        "--lock", $lockPath,
        "--manifest", $preopenManifest,
        "--output-dir", $openRoot,
        "--split", "validation",
        "--output", $planPath
    ) -Phase "materialize_exact_validation_open_plan"
    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    if ($plan.split -ne "validation" -or $plan.setup_count -ne $plan.setups.Count) {
        throw "validation-open plan has inconsistent split/setup count"
    }
    if ($plan.setup_count -eq 0) {
        Write-OpenStatus -State "complete" -Detail "no validation-open setup was eligible"
        return
    }

    foreach ($setup in $plan.setups) {
        $outputPath = Join-Path $RepositoryRoot $setup.generation_path
        if (Test-Path -LiteralPath $outputPath -PathType Leaf) {
            Invoke-NativeLogged -Executable $pythonExe -Arguments @(
                $evaluationValidator,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--plan", $planPath,
                "--setup-id", [string]$setup.setup_id,
                "--path", $outputPath,
                "--kind", "open"
            ) -Phase "validate_existing_validation_open_$($setup.index)"
            continue
        }
        $phase = "setup_$($setup.index)_$($setup.model_tag)_$($setup.method_id)_$($setup.track)"
        $temporaryOutput = "$outputPath.build.$PID.tmp"
        if (Test-Path -LiteralPath $temporaryOutput) {
            Remove-Item -LiteralPath $temporaryOutput -Force
        }
        Write-OpenStatus -State "generating" -Detail $phase
        Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
            "generate-open",
            "--model-config", [string]$setup.model_config,
            "--direction", (Join-Path $RepositoryRoot $setup.direction_path),
            "--track", [string]$setup.track,
            "--strength", (Convert-Invariant ([double]$setup.selected_strength)),
            "--split", "validation",
            "--preopen-manifest", $preopenManifest,
            "--calibration-summary-sha256", $zeroDigest,
            "--construction-config-sha256", [string]$setup.construction_config_sha256,
            "--output", $temporaryOutput
        ) -Phase $phase
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $evaluationValidator,
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--plan", $planPath,
            "--setup-id", [string]$setup.setup_id,
            "--path", $temporaryOutput,
            "--kind", "open"
        ) -Phase "validate_new_validation_open_$($setup.index)"
        Move-Item -LiteralPath $temporaryOutput -Destination $outputPath
        if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
            throw "$phase did not atomically publish $outputPath"
        }
    }

    Write-OpenStatus -State "combining" -Detail "$($plan.setup_count) exact setups"
    Invoke-NativeLogged -Executable $pythonExe -Arguments @(
        $orchestrator, "combine-generations",
        "--repo-root", $RepositoryRoot,
        "--plan", $planPath,
        "--output", $combinedPath
    ) -Phase "combine_validation_open_generations"
    Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
        "judge-requests", "--kind", "open",
        "--input", $combinedPath,
        "--output", $requestsPath
    ) -Phase "render_blinded_validation_open_judge_requests"
    if (-not (Test-Path -LiteralPath $requestsPath -PathType Leaf)) {
        throw "validation open judge-request renderer did not publish $requestsPath"
    }
    $requestCount = @(Get-Content -LiteralPath $requestsPath).Count
    Write-OpenStatus -State "complete" -Detail (
        "setups=$($plan.setup_count);generation_rows=$($plan.setup_count * 96);" +
        "unique_judge_requests=$requestCount;paid_calls=0"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-OpenStatus -State "failed" -Detail $failureDetail
    throw
}
finally {
    [void][SpLenseValidationOpenPowerState]::SetThreadExecutionState($resetExecutionFlags)
}
