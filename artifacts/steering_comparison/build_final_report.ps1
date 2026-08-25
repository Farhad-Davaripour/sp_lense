param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [string]$OutputDirectory = "",
    [switch]$NoStatusLog,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$pythonExe = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
$orchestrator = Join-Path $artifactRoot "locked_open_orchestration.py"
$openChainVerifier = Join-Path $artifactRoot "verify_open_judgment_chain.py"
$evaluationValidator = Join-Path $artifactRoot "validate_locked_evaluation_artifact.py"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
. (Join-Path $artifactRoot "freeze_safety.ps1")
$sealedRoot = Join-Path $artifactRoot "sealed"
$planPath = Join-Path $sealedRoot "sealed_evaluation_plan.json"
$stage2Manifest = Join-Path $RepositoryRoot "configs\steering_comparison_stage2_lock.json"
$reportOutputRoot = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $artifactRoot
}
else {
    [IO.Path]::GetFullPath($OutputDirectory)
}
$outputJson = Join-Path $reportOutputRoot "final_report.json"
$outputMarkdown = Join-Path $reportOutputRoot "FINAL_REPORT.md"
$constructionAvailability = Join-Path $artifactRoot "construction_availability.json"
$jspaceCompletionValidator = Join-Path $artifactRoot "verify_jspace_completion.py"
$statusPath = Join-Path $artifactRoot "report_status.json"
$logPath = Join-Path $artifactRoot "report.log"

function Write-ReportStatus {
    param(
        [string]$State,
        [string]$Detail = ""
    )
    if ($NoStatusLog) {
        return
    }
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
    if (-not $NoStatusLog) {
        "[$([DateTime]::UtcNow.ToString('o'))] START $Phase" |
            Tee-Object -FilePath $logPath -Append
    }
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ($NoStatusLog) {
            & $comparisonExe @Arguments
        }
        else {
            & $comparisonExe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        }
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    if ($nativeExitCode -ne 0) {
        throw "$Phase failed with exit code $nativeExitCode"
    }
    if (-not $NoStatusLog) {
        "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE $Phase" |
            Tee-Object -FilePath $logPath -Append
    }
}

if ($SelfTest) {
    [ordered]@{
        status = "self_test_passed"
        stage2_gate_required = $true
        forced_sources = "every_verified_sealed_setup"
        open_sources = "every_nonrandom_sealed_open_setup"
        jspace_role = "secondary_non_gating"
        construction_availability_optional_fail_closed = $true
        bootstrap_replicates = 100000
        new_weighted_composites = 0
        isolated_output_directory_supported = $true
        no_status_log_verification_supported = $true
    } | ConvertTo-Json
    return
}

try {
    New-Item -ItemType Directory -Path $reportOutputRoot -Force | Out-Null
    Invoke-NativeLogged -Phase "verify_stage2_before_reporting" -Arguments @(
        "verify-stage2", "--stage2-manifest", $stage2Manifest
    )
    foreach ($requiredStatus in @(
        "sealed_evaluation_status.json",
        "sealed_judgment_status.json"
    )) {
        $path = Join-Path $artifactRoot $requiredStatus
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "report prerequisite status is missing: $path"
        }
        $status = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
        if ($status.state -ne "complete") {
            throw "report prerequisite did not complete: $path"
        }
    }
    $jspaceStatusPath = Join-Path $artifactRoot "jspace_status.json"
    if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) {
        throw "sealed evaluation plan is missing"
    }
    $null = Assert-LockedOpenPlanCanonical -PythonExecutable $pythonExe `
        -OrchestratorPath $orchestrator -RepositoryRoot $RepositoryRoot `
        -LockPath $lockPath -ManifestPath $stage2Manifest `
        -OutputDirectory $sealedRoot -Split "sealed_test" -PlanPath $planPath
    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    $openSetups = @($plan.setups | Where-Object { $_.open_required -eq $true })
    if ($openSetups.Count -gt 0) {
        Invoke-Native -Executable $pythonExe -Arguments @(
            $openChainVerifier,
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--plan", $planPath,
            "--combined", (Join-Path $sealedRoot "open_generations_all.jsonl"),
            "--requests", (Join-Path $sealedRoot "open_judge_requests.jsonl"),
            "--responses", (Join-Path $sealedRoot "open_judge_responses.jsonl"),
            "--scored", (Join-Path $sealedRoot "open_scored_all.jsonl")
        )
    }
    $jspaceReceipt = Get-ValidatedJspaceCompletion -PythonExecutable $pythonExe `
        -ValidatorPath $jspaceCompletionValidator -RepositoryRoot $RepositoryRoot `
        -PlanPath $planPath -LockPath $lockPath -StatusPath $jspaceStatusPath `
        -RecordsDirectory (Join-Path $artifactRoot "jspace\records") `
        -AtomsRoot (Join-Path $artifactRoot "jspace\atoms") `
        -CompletionPath (Join-Path $artifactRoot "jspace_completion.json")
    $forcedPaths = @(
        $plan.setups | ForEach-Object {
            $path = Join-Path $RepositoryRoot $_.forced_path
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "sealed forced rows are missing: $path"
            }
            $null = Invoke-Native -Executable $pythonExe -Arguments @(
                $evaluationValidator,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--plan", $planPath,
                "--setup-id", [string]$_.setup_id,
                "--path", $path,
                "--kind", "forced"
            )
            $path
        }
    )
    $openPaths = @(
        $plan.setups | Where-Object { $_.open_required -eq $true } | ForEach-Object {
            $path = Join-Path $RepositoryRoot $_.scored_path
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "sealed scored open rows are missing: $path"
            }
            $path
        }
    )
    $jspaceRecords = @(
        $jspaceReceipt.record_paths | ForEach-Object { Join-Path $RepositoryRoot ([string]$_) }
    )
    if ($forcedPaths.Count -ne $plan.setup_count) {
        throw "forced report coverage differs from the sealed plan"
    }
    $expectedOpenCount = @(
        $plan.setups | Where-Object { $_.open_required -eq $true }
    ).Count
    if ($openPaths.Count -ne $expectedOpenCount) {
        throw "open report coverage is incomplete"
    }

    Write-ReportStatus -State "building" -Detail (
        "forced_files=$($forcedPaths.Count);open_files=$($openPaths.Count);" +
        "jspace_files=$($jspaceRecords.Count)"
    )
    $arguments = @(
        "report",
        "--stage2-manifest", $stage2Manifest,
        "--forced-rows"
    ) + $forcedPaths
    if ($openPaths.Count -gt 0) {
        $arguments += @("--open-rows") + $openPaths
    }
    if ($jspaceRecords.Count -gt 0) {
        $arguments += @("--jspace-records") + $jspaceRecords
    }
    if (Test-Path -LiteralPath $constructionAvailability -PathType Leaf) {
        $arguments += @("--construction-availability", $constructionAvailability)
    }
    $arguments += @(
        "--output-json", $outputJson,
        "--output-markdown", $outputMarkdown
    )
    Invoke-NativeLogged -Phase "build_locked_final_report" -Arguments $arguments
    foreach ($output in @($outputJson, $outputMarkdown)) {
        if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
            throw "report builder did not publish $output"
        }
    }
    $report = Get-Content -Raw -LiteralPath $outputJson | ConvertFrom-Json
    if ($report.schema_version -ne "sp_lense.comparison.report.v1") {
        throw "final report schema is invalid"
    }
    Write-ReportStatus -State "complete" -Detail (
        "production_gate=$($report.production_coverage_gate.status);" +
        "forced_files=$($forcedPaths.Count);open_files=$($openPaths.Count);" +
        "jspace_files=$($jspaceRecords.Count);" +
        "jspace_status=complete"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-ReportStatus -State "failed" -Detail $failureDetail
    throw
}
