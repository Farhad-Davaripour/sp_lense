param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
$statusPath = Join-Path $artifactRoot "preopen_summary_status.json"
$recheckPath = Join-Path $artifactRoot "interpolation_rechecks_required.json"
$logPath = Join-Path $artifactRoot "preopen_summary_build.log"

function Write-SummaryStatus {
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

if ($SelfTest) {
    $keys = @($coverage | ForEach-Object { "$($_.Method)/$($_.Track)" })
    if ($coverage.Count -ne 8 -or (@($keys | Select-Object -Unique)).Count -ne 8) {
        throw "coverage must contain eight unique method/track summaries per model"
    }
    $pointTotal = ($coverage | Measure-Object -Property ExpectedPoints -Sum).Sum
    if ($pointTotal -ne 250) {
        throw "coverage point counts sum to $pointTotal instead of 250"
    }
    [ordered]@{
        status = "self_test_passed"
        summaries_per_model = $coverage.Count
        summaries_total = $coverage.Count * 2
        points_per_model = $pointTotal
        coverage = $keys
    } | ConvertTo-Json
    return
}

try {
    Invoke-NativeLogged -Phase "verify_stage1_before_preopen_summaries" -Arguments @(
        "verify-stage1"
    )
    $rechecks = @()
    $summaryPaths = @()
    foreach ($modelTag in @("qwen35_08b", "qwen35_2b")) {
        $modelRoot = Join-Path $artifactRoot $modelTag
        $gridRoot = Join-Path $modelRoot "forced_grid"
        $planPath = Join-Path $gridRoot "forced_grid_plan.json"
        $pointDirectory = Join-Path $gridRoot "points"
        if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) {
            throw "$modelTag lacks forced_grid_plan.json"
        }
        $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
        if ($plan.points.Count -ne 250) {
            throw "$modelTag plan contains $($plan.points.Count) points instead of 250"
        }
        $summaryDirectory = Join-Path $modelRoot "calibration"
        New-Item -ItemType Directory -Force -Path $summaryDirectory | Out-Null

        foreach ($item in $coverage) {
            $matchingPoints = @(
                $plan.points | Where-Object {
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
                        throw "missing forced-grid shard $path"
                    }
                    $path
                }
            )
            $summaryName = "$($item.Method)_$($item.Track)_preopen.json"
            $summaryPath = Join-Path $summaryDirectory $summaryName
            Write-SummaryStatus -State "building" -Detail "$modelTag/$($item.Method)/$($item.Track)"
            $arguments = @(
                "build-calibration-summary",
                "--mode", $item.Track,
                "--grid-plan", $planPath,
                "--point-shards"
            ) + $pointShards + @(
                "--pre-open-only",
                "--output", $summaryPath
            )
            Invoke-NativeLogged -Phase "${modelTag}_$($item.Method)_$($item.Track)" -Arguments $arguments
            $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
            $summaryPaths += $summaryPath
            if ($summary.pre_open_decision.status -eq "interpolation_requires_one_recheck") {
                $rechecks += [pscustomobject]@{
                    model_tag = $modelTag
                    model_id = $summary.model_id
                    method_id = $summary.method_id
                    track = $summary.track
                    summary_path = $summaryPath
                    interpolation_candidate = $summary.pre_open_decision.interpolation_candidate
                    interpolation_upper_strength = $summary.pre_open_decision.interpolation_upper_strength
                    candidate_directions = $summary.candidate_directions
                    matching_plan_points = $matchingPoints
                }
            }
        }
    }

    $record = [ordered]@{
        schema_version = 1
        status = if ($rechecks.Count -eq 0) { "none_required" } else { "required" }
        summary_count = $summaryPaths.Count
        summary_paths = $summaryPaths
        recheck_count = $rechecks.Count
        rechecks = $rechecks
    }
    $record | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $recheckPath -Encoding utf8
    if ($summaryPaths.Count -ne 16) {
        throw "built $($summaryPaths.Count) summaries instead of 16"
    }
    Write-SummaryStatus -State "complete" -Detail "$($rechecks.Count) interpolation rechecks required"
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-SummaryStatus -State "failed" -Detail $failureDetail
    throw
}
