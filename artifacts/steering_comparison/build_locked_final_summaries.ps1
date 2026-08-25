param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
$openRoot = Join-Path $artifactRoot "validation_open"
$planPath = Join-Path $openRoot "validation_open_plan.json"
$recheckPath = Join-Path $artifactRoot "interpolation_rechecks_required.json"
$statusPath = Join-Path $artifactRoot "final_summary_status.json"
$logPath = Join-Path $artifactRoot "final_summary_build.log"
$coverage = @(
    [pscustomobject]@{ Method = "gradient"; Track = "matched"; ExpectedPoints = 6 },
    [pscustomobject]@{ Method = "gradient_uncorrected"; Track = "matched"; ExpectedPoints = 6 },
    [pscustomobject]@{ Method = "caa"; Track = "matched"; ExpectedPoints = 6 },
    [pscustomobject]@{ Method = "bipo"; Track = "matched"; ExpectedPoints = 6 },
    [pscustomobject]@{ Method = "persona_vector"; Track = "matched"; ExpectedPoints = 6 },
    [pscustomobject]@{ Method = "caa"; Track = "canonical"; ExpectedPoints = 96 },
    [pscustomobject]@{ Method = "bipo"; Track = "canonical"; ExpectedPoints = 4 },
    [pscustomobject]@{ Method = "persona_vector"; Track = "canonical"; ExpectedPoints = 120 }
)

function Write-FinalSummaryStatus {
    param(
        [string]$State,
        [string]$Detail = ""
    )
    [ordered]@{
        schema_version = 1
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
    "[$([DateTime]::UtcNow.ToString('o'))] START $Phase" |
        Tee-Object -FilePath $logPath -Append
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
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE $Phase" |
        Tee-Object -FilePath $logPath -Append
}

if ($SelfTest) {
    $keys = @($coverage | ForEach-Object { "$($_.Method)/$($_.Track)" })
    $pointTotal = ($coverage | Measure-Object -Property ExpectedPoints -Sum).Sum
    if (
        $coverage.Count -ne 8 -or
        (@($keys | Select-Object -Unique)).Count -ne 8 -or
        $pointTotal -ne 250
    ) {
        throw "final summaries must preserve the exact eight-cover/250-point model plan"
    }
    [ordered]@{
        status = "self_test_passed"
        summaries_per_model = 8
        summaries_total = 16
        points_per_model = 250
        open_rows_per_setup = 96
        selection_source = "committed_preopen_manifest_only"
        fallback_searches = 0
    } | ConvertTo-Json
    return
}

try {
    Invoke-NativeLogged -Phase "verify_stage1_before_final_summaries" -Arguments @(
        "verify-stage1"
    )
    foreach ($required in @($planPath, $recheckPath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "final-summary prerequisite is missing: $required"
        }
    }
    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    if ($plan.split -ne "validation" -or $plan.setup_count -ne @($plan.setups).Count) {
        throw "validation-open plan is inconsistent"
    }
    $recheckRecord = Get-Content -Raw -LiteralPath $recheckPath | ConvertFrom-Json
    if ($recheckRecord.status -notin @("none_required", "completed")) {
        throw "interpolation rechecks are not frozen"
    }
    $completedRechecks = if ($recheckRecord.status -eq "completed") {
        @($recheckRecord.completed_rechecks)
    }
    else {
        @()
    }

    $built = @()
    $assignedSetupIds = @()
    foreach ($modelTag in @("qwen35_08b", "qwen35_2b")) {
        $modelRoot = Join-Path $artifactRoot $modelTag
        $gridRoot = Join-Path $modelRoot "forced_grid"
        $gridPlanPath = Join-Path $gridRoot "forced_grid_plan.json"
        $pointDirectory = Join-Path $gridRoot "points"
        $summaryDirectory = Join-Path $modelRoot "calibration"
        $gridPlan = Get-Content -Raw -LiteralPath $gridPlanPath | ConvertFrom-Json
        if ($gridPlan.points.Count -ne 250) {
            throw "$modelTag forced-grid plan no longer has 250 points"
        }

        foreach ($item in $coverage) {
            $preopenName = "$($item.Method)_$($item.Track)_preopen.json"
            $preopenPath = Join-Path $summaryDirectory $preopenName
            if (-not (Test-Path -LiteralPath $preopenPath -PathType Leaf)) {
                throw "pre-open summary is missing: $preopenPath"
            }
            $preopenRelative = (
                "artifacts/steering_comparison/$modelTag/calibration/$preopenName"
            )
            $setupRecords = @(
                $plan.setups | Where-Object {
                    $_.calibration_summary_path -eq $preopenRelative
                }
            )
            foreach ($setup in $setupRecords) {
                $scoredPath = Join-Path $RepositoryRoot $setup.scored_path
                if (-not (Test-Path -LiteralPath $scoredPath -PathType Leaf)) {
                    throw "scored open-confirmation rows are missing: $scoredPath"
                }
                if (@(Get-Content -LiteralPath $scoredPath).Count -ne 96) {
                    throw "open-confirmation shard must contain exactly 96 rows: $scoredPath"
                }
                $assignedSetupIds += [string]$setup.setup_id
            }

            $matchingPoints = @(
                $gridPlan.points | Where-Object {
                    $_.method_id -eq $item.Method -and $_.track -eq $item.Track
                }
            )
            if ($matchingPoints.Count -ne $item.ExpectedPoints) {
                throw (
                    "$modelTag/$($item.Method)/$($item.Track) has " +
                    "$($matchingPoints.Count) points instead of $($item.ExpectedPoints)"
                )
            }
            $pointShards = @(
                $matchingPoints | ForEach-Object {
                    $path = Join-Path $pointDirectory $_.shard_name
                    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                        throw "forced-grid shard is missing: $path"
                    }
                    $path
                }
            )
            $arguments = @(
                "build-calibration-summary",
                "--mode", $item.Track,
                "--grid-plan", $gridPlanPath,
                "--point-shards"
            ) + $pointShards
            $recheck = if ($item.Track -eq "matched") {
                @(
                    $completedRechecks | Where-Object {
                        $_.model_tag -eq $modelTag -and
                        $_.method_id -eq $item.Method
                    }
                )
            }
            else {
                @()
            }
            if ($recheck.Count -gt 1) {
                throw "multiple interpolation rechecks map to one summary"
            }
            if ($recheck.Count -eq 1) {
                $arguments += @(
                    "--interpolation-recheck-rows", [string]$recheck[0].result_path
                )
            }
            if ($setupRecords.Count -gt 0) {
                $arguments += @("--open-confirmation-rows") + @(
                    $setupRecords | ForEach-Object {
                        Join-Path $RepositoryRoot $_.scored_path
                    }
                )
            }
            $finalPath = Join-Path $summaryDirectory (
                "$($item.Method)_$($item.Track)_final.json"
            )
            $arguments += @("--output", $finalPath)
            Write-FinalSummaryStatus -State "building" -Detail (
                "$modelTag/$($item.Method)/$($item.Track)"
            )
            Invoke-NativeLogged -Phase (
                "${modelTag}_$($item.Method)_$($item.Track)_final"
            ) -Arguments $arguments
            $final = Get-Content -Raw -LiteralPath $finalPath | ConvertFrom-Json
            if ($final.decision.status -eq "open_confirmation_pending") {
                throw "final summary remains open-confirmation pending: $finalPath"
            }
            if ($final.pre_open_decision_sha256 -ne (
                (Get-Content -Raw -LiteralPath $preopenPath | ConvertFrom-Json).
                    pre_open_decision_sha256
            )) {
                throw "final summary changed the frozen pre-open decision: $finalPath"
            }
            $built += [pscustomobject]@{
                model_tag = $modelTag
                method_id = $item.Method
                track = $item.Track
                path = $finalPath
                decision_status = $final.decision.status
                open_confirmation_passed = $final.decision.open_confirmation_passed
            }
        }
    }
    if ($built.Count -ne 16) {
        throw "built $($built.Count) final summaries instead of 16"
    }
    $plannedSetupIds = @($plan.setups | ForEach-Object { [string]$_.setup_id })
    if (
        $assignedSetupIds.Count -ne $plannedSetupIds.Count -or
        (@($assignedSetupIds | Sort-Object) -join "|") -ne
        (@($plannedSetupIds | Sort-Object) -join "|")
    ) {
        throw "validation-open setups were not assigned exactly once to final summaries"
    }
    [ordered]@{
        schema_version = 1
        state = "complete"
        summary_count = $built.Count
        setup_count = $plannedSetupIds.Count
        summaries = $built
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $statusPath -Encoding utf8
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-FinalSummaryStatus -State "failed" -Detail $failureDetail
    throw
}
