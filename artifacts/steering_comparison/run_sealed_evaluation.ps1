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
$sealedRoot = Join-Path $artifactRoot "sealed"
$orchestrator = Join-Path $artifactRoot "locked_open_orchestration.py"
$evaluationValidator = Join-Path $artifactRoot "validate_locked_evaluation_artifact.py"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$stage2Manifest = Join-Path $RepositoryRoot "configs\steering_comparison_stage2_lock.json"
$planPath = Join-Path $sealedRoot "sealed_evaluation_plan.json"
$combinedOpenPath = Join-Path $sealedRoot "open_generations_all.jsonl"
$requestsPath = Join-Path $sealedRoot "open_judge_requests.jsonl"
$statusPath = Join-Path $artifactRoot "sealed_evaluation_status.json"
$logPath = Join-Path $artifactRoot "sealed_evaluation.log"

function Convert-Invariant {
    param([double]$Value)
    return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Write-SealedStatus {
    param(
        [string]$State,
        [string]$Detail = ""
    )
    [ordered]@{
        schema_version = 1
        split = "sealed_test"
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
    [ordered]@{
        status = "self_test_passed"
        split = "sealed_test"
        stage2_gate_required = $true
        preopen_gate_insufficient = $true
        random_open_generation = $false
        random_tbsp_evaluation = $false
        main_tbsp_evaluation = $true
        judge_calls = 0
    } | ConvertTo-Json
    return
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SpLenseSealedPowerState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$keepAwakeFlags = [uint32]::Parse("80000001", [Globalization.NumberStyles]::HexNumber)
$resetExecutionFlags = [uint32]::Parse("80000000", [Globalization.NumberStyles]::HexNumber)
[void][SpLenseSealedPowerState]::SetThreadExecutionState($keepAwakeFlags)

try {
    Write-SealedStatus -State "verifying_stage2"
    Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
        "verify-stage2", "--stage2-manifest", $stage2Manifest
    ) -Phase "verify_stage2_before_any_sealed_forward_pass"
    foreach ($required in @($orchestrator, $evaluationValidator, $lockPath, $stage2Manifest)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "sealed prerequisite is missing: $required"
        }
    }
    New-Item -ItemType Directory -Force -Path $sealedRoot | Out-Null
    Invoke-NativeLogged -Executable $pythonExe -Arguments @(
        $orchestrator, "plan",
        "--repo-root", $RepositoryRoot,
        "--lock", $lockPath,
        "--manifest", $stage2Manifest,
        "--output-dir", $sealedRoot,
        "--split", "sealed_test",
        "--output", $planPath
    ) -Phase "materialize_verified_sealed_setup_plan"
    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    if ($plan.split -ne "sealed_test" -or $plan.setup_count -ne @($plan.setups).Count) {
        throw "sealed setup plan is inconsistent"
    }
    $openCount = @($plan.setups | Where-Object { $_.open_required -eq $true }).Count
    $randomCount = @($plan.setups | Where-Object { $_.is_random_control -eq $true }).Count

    foreach ($setup in $plan.setups) {
        $forcedOutput = Join-Path $RepositoryRoot $setup.forced_path
        if (-not (Test-Path -LiteralPath $forcedOutput -PathType Leaf)) {
            $phase = "forced_$($setup.index)_$($setup.model_tag)_$($setup.method_id)"
            Write-SealedStatus -State "evaluating_forced" -Detail $phase
            $arguments = @(
                "evaluate-forced",
                "--model-config", [string]$setup.model_config,
                "--direction", (Join-Path $RepositoryRoot $setup.direction_path),
                "--track", [string]$setup.track,
                "--strength", (Convert-Invariant ([double]$setup.selected_strength)),
                "--split", "sealed_test",
                "--stage2-manifest", $stage2Manifest,
                "--calibration-summary-sha256", [string]$setup.calibration_summary_sha256,
                "--construction-config-sha256", [string]$setup.construction_config_sha256
            )
            if ($setup.tbsp_required -eq $true) {
                $arguments += @("--include-tbsp")
            }
            $temporaryForcedOutput = "$forcedOutput.build.$PID.tmp"
            if (Test-Path -LiteralPath $temporaryForcedOutput) {
                Remove-Item -LiteralPath $temporaryForcedOutput -Force
            }
            $arguments += @("--output", $temporaryForcedOutput)
            Invoke-NativeLogged -Executable $comparisonExe -Arguments $arguments -Phase $phase
            Invoke-NativeLogged -Executable $pythonExe -Arguments @(
                $evaluationValidator,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--plan", $planPath,
                "--setup-id", [string]$setup.setup_id,
                "--path", $temporaryForcedOutput,
                "--kind", "forced"
            ) -Phase "validate_new_sealed_forced_$($setup.index)"
            Move-Item -LiteralPath $temporaryForcedOutput -Destination $forcedOutput
        }
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $evaluationValidator,
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--plan", $planPath,
            "--setup-id", [string]$setup.setup_id,
            "--path", $forcedOutput,
            "--kind", "forced"
        ) -Phase "validate_sealed_forced_$($setup.index)"

        if ($setup.open_required -eq $true) {
            $openOutput = Join-Path $RepositoryRoot $setup.generation_path
            if (-not (Test-Path -LiteralPath $openOutput -PathType Leaf)) {
                $phase = "open_$($setup.index)_$($setup.model_tag)_$($setup.method_id)"
                Write-SealedStatus -State "generating_open" -Detail $phase
                $temporaryOpenOutput = "$openOutput.build.$PID.tmp"
                if (Test-Path -LiteralPath $temporaryOpenOutput) {
                    Remove-Item -LiteralPath $temporaryOpenOutput -Force
                }
                Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
                    "generate-open",
                    "--model-config", [string]$setup.model_config,
                    "--direction", (Join-Path $RepositoryRoot $setup.direction_path),
                    "--track", [string]$setup.track,
                    "--strength", (Convert-Invariant ([double]$setup.selected_strength)),
                    "--split", "sealed_test",
                    "--stage2-manifest", $stage2Manifest,
                    "--calibration-summary-sha256", [string]$setup.calibration_summary_sha256,
                    "--construction-config-sha256", [string]$setup.construction_config_sha256,
                    "--output", $temporaryOpenOutput
                ) -Phase $phase
                Invoke-NativeLogged -Executable $pythonExe -Arguments @(
                    $evaluationValidator,
                    "--repo-root", $RepositoryRoot,
                    "--lock", $lockPath,
                    "--plan", $planPath,
                    "--setup-id", [string]$setup.setup_id,
                    "--path", $temporaryOpenOutput,
                    "--kind", "open"
                ) -Phase "validate_new_sealed_open_$($setup.index)"
                Move-Item -LiteralPath $temporaryOpenOutput -Destination $openOutput
            }
            Invoke-NativeLogged -Executable $pythonExe -Arguments @(
                $evaluationValidator,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--plan", $planPath,
                "--setup-id", [string]$setup.setup_id,
                "--path", $openOutput,
                "--kind", "open"
            ) -Phase "validate_sealed_open_$($setup.index)"
        }
    }

    if ($openCount -gt 0) {
        Write-SealedStatus -State "combining_open" -Detail "$openCount exact setups"
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $orchestrator, "combine-generations",
            "--repo-root", $RepositoryRoot,
            "--plan", $planPath,
            "--output", $combinedOpenPath
        ) -Phase "combine_sealed_open_generations"
        Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
            "judge-requests", "--kind", "open",
            "--input", $combinedOpenPath,
            "--output", $requestsPath
        ) -Phase "render_blinded_sealed_open_judge_requests"
    }
    $requestCount = if (Test-Path -LiteralPath $requestsPath) {
        @(Get-Content -LiteralPath $requestsPath).Count
    }
    else {
        0
    }
    Write-SealedStatus -State "complete" -Detail (
        "setups=$($plan.setup_count);open_setups=$openCount;random_setups=$randomCount;" +
        "judge_requests=$requestCount;paid_calls=0"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-SealedStatus -State "failed" -Detail $failureDetail
    throw
}
finally {
    [void][SpLenseSealedPowerState]::SetThreadExecutionState($resetExecutionFlags)
}
