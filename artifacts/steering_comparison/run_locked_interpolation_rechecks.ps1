param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$pythonExe = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$recheckValidator = Join-Path $artifactRoot "validate_interpolation_recheck.py"
$requestPath = Join-Path $artifactRoot "interpolation_rechecks_required.json"
$statusPath = Join-Path $artifactRoot "interpolation_recheck_status.json"
$logPath = Join-Path $artifactRoot "interpolation_recheck.log"
$zeroSha256 = "0" * 64

function Write-RecheckStatus {
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
        [string]$Executable = $comparisonExe,
        [string]$Phase,
        [string[]]$Arguments
    )

    "[$([DateTime]::UtcNow.ToString('o'))] START $Phase" | Tee-Object -FilePath $logPath -Append
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
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE $Phase" | Tee-Object -FilePath $logPath -Append
}

if ($SelfTest) {
    if ($zeroSha256.Length -ne 64 -or $zeroSha256.Trim("0").Length -ne 0) {
        throw "validation summary sentinel is invalid"
    }
    [ordered]@{
        status = "self_test_passed"
        allowed_track = "matched"
        allowed_split = "validation"
        calibration_summary_sha256 = $zeroSha256
        maximum_rechecks_per_summary = 1
    } | ConvertTo-Json
    return
}

try {
    Invoke-NativeLogged -Phase "verify_stage1_before_interpolation_rechecks" -Arguments @(
        "verify-stage1"
    )
    if (-not (Test-Path -LiteralPath $requestPath -PathType Leaf)) {
        throw "missing interpolation request record $requestPath"
    }
    $requestRecord = Get-Content -Raw -LiteralPath $requestPath | ConvertFrom-Json
    if ($requestRecord.schema_version -ne 1) {
        throw "unsupported interpolation request schema"
    }
    if ($requestRecord.status -eq "none_required") {
        Write-RecheckStatus -State "complete" -Detail "no interpolation rechecks required"
        return
    }
    if ($requestRecord.status -eq "completed") {
        $completedRecords = @($requestRecord.completed_rechecks)
        if ($completedRecords.Count -ne $requestRecord.recheck_count) {
            throw "completed interpolation record has inconsistent coverage"
        }
        foreach ($completedRecord in $completedRecords) {
            $completedPlan = Join-Path $artifactRoot (
                "$($completedRecord.model_tag)\forced_grid\forced_grid_plan.json"
            )
            Invoke-NativeLogged -Executable $pythonExe `
                -Phase "validate_completed_$($completedRecord.model_tag)_$($completedRecord.method_id)" `
                -Arguments @(
                    $recheckValidator,
                    "--repo-root", $RepositoryRoot,
                    "--lock", $lockPath,
                    "--request", $requestPath,
                    "--plan", $completedPlan,
                    "--model-tag", [string]$completedRecord.model_tag,
                    "--method-id", [string]$completedRecord.method_id,
                    "--rows", [string]$completedRecord.result_path
                )
        }
        Write-RecheckStatus -State "complete" -Detail "$($completedRecords.Count) interpolation rechecks already completed"
        return
    }
    if ($requestRecord.status -ne "required") {
        throw "interpolation request status must be required or none_required"
    }
    $requests = @($requestRecord.rechecks)
    if ($requests.Count -ne $requestRecord.recheck_count -or $requests.Count -lt 1) {
        throw "interpolation request count is inconsistent"
    }
    $identities = @(
        $requests | ForEach-Object { "$($_.model_tag)/$($_.method_id)/$($_.track)" }
    )
    if ((@($identities | Select-Object -Unique)).Count -ne $identities.Count) {
        throw "interpolation requests duplicate a model/method/track summary"
    }

    $completed = @()
    foreach ($request in $requests) {
        if ($request.track -ne "matched") {
            throw "only matched summaries may request interpolation"
        }
        $candidate = [double]$request.interpolation_candidate
        if ([double]::IsNaN($candidate) -or [double]::IsInfinity($candidate) -or $candidate -le 0) {
            throw "interpolation candidate must be positive and finite"
        }
        $candidateDirections = @($request.candidate_directions)
        if ($candidateDirections.Count -ne 1) {
            throw "matched interpolation must use exactly one candidate direction"
        }
        $matchingPoints = @($request.matching_plan_points)
        if ($matchingPoints.Count -ne 6) {
            throw "matched interpolation source must contain exactly six grid points"
        }
        $directionPaths = @($matchingPoints.direction_path | Select-Object -Unique)
        $constructionHashes = @(
            $matchingPoints.construction_config_sha256 | Select-Object -Unique
        )
        if ($directionPaths.Count -ne 1 -or $constructionHashes.Count -ne 1) {
            throw "matched interpolation source changes direction or construction identity"
        }
        $modelConfig = switch ($request.model_tag) {
            "qwen35_08b" { "configs\qwen35_08b_aligned.json" }
            "qwen35_2b" { "configs\qwen35_2b_aligned.json" }
            default { throw "unknown model tag $($request.model_tag)" }
        }
        $modelRoot = Join-Path $artifactRoot $request.model_tag
        $gridRoot = Join-Path $modelRoot "forced_grid"
        $planPath = Join-Path $gridRoot "forced_grid_plan.json"
        $pointDirectory = Join-Path $gridRoot "points"
        $methodId = [string]$request.method_id
        $outputPath = Join-Path $modelRoot "calibration\${methodId}_matched_interpolation.jsonl"
        Write-RecheckStatus -State "running" -Detail "$($request.model_tag)/$methodId"
        if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
            $temporaryOutput = "$outputPath.build.$PID.tmp"
            if (Test-Path -LiteralPath $temporaryOutput) {
                Remove-Item -LiteralPath $temporaryOutput -Force
            }
            Invoke-NativeLogged -Phase "$($request.model_tag)_${methodId}_interpolation" -Arguments @(
                "evaluate-forced",
                "--model-config", $modelConfig,
                "--direction", $directionPaths[0],
                "--track", "matched",
                "--strength", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:R}", $candidate)),
                "--split", "validation",
                "--calibration-summary-sha256", $zeroSha256,
                "--construction-config-sha256", $constructionHashes[0],
                "--output", $temporaryOutput
            )
            Invoke-NativeLogged -Executable $pythonExe `
                -Phase "$($request.model_tag)_${methodId}_validate_new_interpolation" `
                -Arguments @(
                    $recheckValidator,
                    "--repo-root", $RepositoryRoot,
                    "--lock", $lockPath,
                    "--request", $requestPath,
                    "--plan", $planPath,
                    "--model-tag", [string]$request.model_tag,
                    "--method-id", $methodId,
                    "--rows", $temporaryOutput
                )
            Move-Item -LiteralPath $temporaryOutput -Destination $outputPath
        }
        Invoke-NativeLogged -Executable $pythonExe `
            -Phase "$($request.model_tag)_${methodId}_validate_interpolation" `
            -Arguments @(
                $recheckValidator,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--request", $requestPath,
                "--plan", $planPath,
                "--model-tag", [string]$request.model_tag,
                "--method-id", $methodId,
                "--rows", $outputPath
            )

        $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
        $summaryPoints = @(
            $plan.points | Where-Object {
                $_.method_id -eq $methodId -and $_.track -eq "matched"
            }
        )
        if ($summaryPoints.Count -ne 6) {
            throw "rebuild source no longer contains six locked matched points"
        }
        $pointShards = @(
            $summaryPoints | ForEach-Object {
                $path = Join-Path $pointDirectory $_.shard_name
                if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                    throw "missing forced-grid shard $path"
                }
                $path
            }
        )
        $summaryPath = [string]$request.summary_path
        $summaryArguments = @(
            "build-calibration-summary",
            "--mode", "matched",
            "--grid-plan", $planPath,
            "--point-shards"
        ) + $pointShards + @(
            "--interpolation-recheck-rows", $outputPath,
            "--pre-open-only",
            "--output", $summaryPath
        )
        Invoke-NativeLogged -Phase "$($request.model_tag)_${methodId}_rebuild_summary" -Arguments $summaryArguments
        $rebuilt = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
        if ($rebuilt.pre_open_decision.status -eq "interpolation_requires_one_recheck") {
            throw "summary remains interpolation-pending after its one permitted recheck"
        }
        $completed += [pscustomobject]@{
            model_tag = $request.model_tag
            method_id = $methodId
            result_path = $outputPath
            summary_path = $summaryPath
            final_pre_open_status = $rebuilt.pre_open_decision.status
            selected_strength = $rebuilt.pre_open_decision.selected_strength
        }
    }

    $requestRecord.status = "completed"
    $requestRecord | Add-Member -NotePropertyName completed_rechecks -NotePropertyValue $completed -Force
    $requestRecord | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $requestPath -Encoding utf8
    Write-RecheckStatus -State "complete" -Detail "$($completed.Count) interpolation rechecks completed"
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-RecheckStatus -State "failed" -Detail $failureDetail
    throw
}
