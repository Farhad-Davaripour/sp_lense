param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [switch]$Push,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$pythonExe = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
. (Join-Path $artifactRoot "freeze_safety.ps1")
$transport = Join-Path $artifactRoot "submit_openai_judge_requests.py"
$personaValidator = Join-Path $artifactRoot "validate_persona_artifacts.py"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$preopenManifest = Join-Path $RepositoryRoot "configs\steering_comparison_preopen_lock.json"
$preopenRelative = "configs/steering_comparison_preopen_lock.json"
$artifactCommitSubject = "Freeze steering directions and forced validation artifacts"
$manifestCommitSubject = "Lock validation open-response candidates"
$statusPath = Join-Path $artifactRoot "preopen_freeze_status.json"
$volatileStatusPaths = @(Get-UniversalFreezeVolatilePaths)
$artifactInventoryRelative = "artifacts/steering_comparison/preopen_artifact_inventory.json"
$artifactInventoryPath = Join-Path $RepositoryRoot $artifactInventoryRelative

$summaryNames = @(
    "gradient_matched_preopen.json",
    "gradient_uncorrected_matched_preopen.json",
    "caa_matched_preopen.json",
    "bipo_matched_preopen.json",
    "persona_vector_matched_preopen.json",
    "caa_canonical_preopen.json",
    "bipo_canonical_preopen.json",
    "persona_vector_canonical_preopen.json"
)
$directionManifestNames = @(
    "gradient",
    "caa",
    "bipo_matched",
    "bipo_canonical",
    "persona",
    "random"
)
$summaryPaths = @(
    foreach ($modelTag in @("qwen35_08b", "qwen35_2b")) {
        foreach ($name in $summaryNames) {
            Join-Path $artifactRoot "$modelTag\calibration\$name"
        }
    }
)
$directionManifestPaths = @(
    foreach ($modelTag in @("qwen35_08b", "qwen35_2b")) {
        foreach ($name in $directionManifestNames) {
            Join-Path $artifactRoot "$modelTag\directions\$name\direction_manifest.json"
        }
    }
)
$personaJudgeVerifications = @(
    foreach ($modelTag in @("qwen35_08b", "qwen35_2b")) {
        [ordered]@{
            tag = "${modelTag}_persona"
            model_config = if ($modelTag -eq "qwen35_08b") {
                Join-Path $RepositoryRoot "configs\qwen35_08b_aligned.json"
            }
            else {
                Join-Path $RepositoryRoot "configs\qwen35_2b_aligned.json"
            }
            raw = Join-Path $artifactRoot "$modelTag\persona_raw.jsonl"
            requests = Join-Path $artifactRoot "$modelTag\persona_judge_requests.jsonl"
            responses = Join-Path $artifactRoot "$modelTag\persona_judge_responses.jsonl"
            scored = Join-Path $artifactRoot "$modelTag\persona_scored.jsonl"
            manifest = Join-Path $artifactRoot "$modelTag\directions\persona\direction_manifest.json"
            work_directory = Join-Path $artifactRoot "$modelTag\persona_judge_transport"
        }
    }
)

function Get-PreopenExpectedArtifactPaths {
    $rootNames = @(
        "ADVERSARIAL_REVIEW_CHECKLIST.md",
        "JUDGE_TRANSPORT_README.md",
        "PIPELINE_AUTOMATION_README.md",
        "build_final_report.ps1",
        "build_locked_final_summaries.ps1",
        "build_locked_preopen_summaries.ps1",
        "complete_persona_directions.ps1",
        "complete_sealed_judgments.ps1",
        "complete_validation_open.ps1",
        "freeze_final_results.ps1",
        "freeze_locked_preopen.ps1",
        "freeze_locked_stage2.ps1",
        "freeze_safety.ps1",
        "freeze_safety_selftest.ps1",
        "locked_open_orchestration.py",
        "run_jspace_secondary.ps1",
        "run_locked_forced_grids.ps1",
        "run_locked_interpolation_rechecks.ps1",
        "run_remaining_local_construction.ps1",
        "run_sealed_evaluation.ps1",
        "run_validation_open_generation.ps1",
        "set_openai_api_key.ps1",
        "submit_openai_judge_requests.py",
        "test_locked_open_orchestration.py",
        "test_validate_locked_evaluation_artifact.py",
        "test_validate_interpolation_recheck.py",
        "test_verify_open_judgment_chain.py",
        "test_submit_openai_judge_requests.py",
        "test_validate_adversarial_review.py",
        "test_validate_jspace_cache.py",
        "test_validate_jspace_record.py",
        "test_verify_jspace_completion.py",
        "test_validate_persona_artifacts.py",
        "validate_adversarial_review.py",
        "validate_jspace_cache.py",
        "validate_jspace_record.py",
        "verify_jspace_completion.py",
        "validate_locked_evaluation_artifact.py",
        "validate_interpolation_recheck.py",
        "verify_open_judgment_chain.py",
        "validate_persona_artifacts.py",
        "environment.json",
        "construction_availability.json",
        "qwen35_08b_smoke.json",
        "qwen35_2b_smoke.json",
        "local_construction_status.json",
        "local_construction.log",
        "local_construction_wrapper.stderr.log",
        "local_construction_wrapper.stdout.log",
        "local_construction_wrapper_v2.stderr.log",
        "local_construction_wrapper_v2.stdout.log",
        "persona_completion_status.json",
        "persona_completion.log",
        "forced_grid_status.json",
        "forced_grid.log",
        "preopen_summary_status.json",
        "preopen_summary_build.log",
        "interpolation_rechecks_required.json",
        "interpolation_recheck_status.json",
        "interpolation_recheck.log"
    )
    $paths = @(
        foreach ($name in $rootNames) {
            $path = Join-Path $artifactRoot $name
            if (Test-Path -LiteralPath $path -PathType Leaf) {
                ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
            }
        }
    )
    $paths += @(
        $summaryPaths | ForEach-Object {
            ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $_
        }
    )
    $paths += @(Get-DirectionManifestArtifactPaths -RepositoryRoot $RepositoryRoot `
        -ManifestPaths $directionManifestPaths)
    foreach ($modelTag in @("qwen35_08b", "qwen35_2b")) {
        $modelRoot = Join-Path $artifactRoot $modelTag
        foreach ($name in @("persona_raw.jsonl", "persona_judge_requests.jsonl", `
            "persona_judge_responses.jsonl", "persona_scored.jsonl")) {
            $path = Join-Path $modelRoot $name
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "pre-open exact artifact is missing: $path"
            }
            $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
        }
        $verification = @($personaJudgeVerifications | Where-Object { $_.tag -eq "${modelTag}_persona" })
        if ($verification.Count -ne 1) {
            throw "pre-open persona transport registry lacks $modelTag"
        }
        $paths += @(Get-LockedJudgeTransportArtifactPaths -RepositoryRoot $RepositoryRoot `
            -RequestsPath ([string]$verification[0].requests) `
            -ResponsesPath ([string]$verification[0].responses) `
            -WorkDirectory ([string]$verification[0].work_directory))
        $paths += @(Get-ForcedGridArtifactPaths -RepositoryRoot $RepositoryRoot `
            -PlanPath (Join-Path $modelRoot "forced_grid\forced_grid_plan.json") `
            -ExpectedPointCount 250)
    }
    $recheck = Get-Content -Raw -LiteralPath (
        Join-Path $artifactRoot "interpolation_rechecks_required.json"
    ) | ConvertFrom-Json
    foreach ($completed in @($recheck.completed_rechecks)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$completed.result_path)) {
            $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot `
                -Path ([string]$completed.result_path)
        }
    }
    return @($paths | Sort-Object -Unique)
}

function Write-FreezeStatus {
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

function Invoke-Native {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    if ($nativeExitCode -ne 0) {
        throw "$Executable failed with exit code $nativeExitCode"
    }
}

function Assert-PersonaCombinedCap {
    param([object[]]$Preflights)

    if ($Preflights.Count -ne 2) {
        throw "persona combined-cap check requires exactly two preflights"
    }
    foreach ($preflight in $Preflights) {
        foreach ($field in @("safe_upper_bound_usd", "user_cost_ceiling_usd")) {
            $value = [double]$preflight.$field
            if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -le 0) {
                throw "persona judge preflight has invalid $field"
            }
        }
    }
    $personaCeilings = @(
        $Preflights |
            Select-Object -ExpandProperty user_cost_ceiling_usd -Unique
    )
    if ($personaCeilings.Count -ne 1) {
        throw "persona judge preflights do not share one explicit total ceiling"
    }
    $personaSafeTotal = [double](
        ($Preflights | Measure-Object -Property safe_upper_bound_usd -Sum).Sum
    )
    if ($personaSafeTotal -gt ([double]$personaCeilings[0] + 1e-12)) {
        throw "combined persona judge safe upper bound exceeds its total ceiling"
    }
    return [pscustomobject]@{
        safe_upper_bound_usd = $personaSafeTotal
        user_cost_ceiling_usd = [double]$personaCeilings[0]
    }
}

if ($SelfTest) {
    if ($summaryPaths.Count -ne 16 -or (@($summaryPaths | Select-Object -Unique)).Count -ne 16) {
        throw "freeze must receive 16 unique calibration summaries"
    }
    if (
        $directionManifestPaths.Count -ne 12 -or
        (@($directionManifestPaths | Select-Object -Unique)).Count -ne 12
    ) {
        throw "freeze must receive 12 unique direction manifests including random controls"
    }
    $selfTestCap = Assert-PersonaCombinedCap -Preflights @(
        [pscustomobject]@{
            safe_upper_bound_usd = 0.4
            user_cost_ceiling_usd = 1.0
        },
        [pscustomobject]@{
            safe_upper_bound_usd = 0.5
            user_cost_ceiling_usd = 1.0
        }
    )
    if ([math]::Abs($selfTestCap.safe_upper_bound_usd - 0.9) -gt 1e-12) {
        throw "persona combined-cap self-test failed"
    }
    [ordered]@{
        status = "self_test_passed"
        summary_count = $summaryPaths.Count
        direction_manifest_count = $directionManifestPaths.Count
        commits = @("B_artifact_freeze", "C_preopen_manifest")
        restart_states = @(
            "new", "B_staged_uncommitted", "B_committed_C_absent", "C_file_uncommitted",
            "C_committed_unpushed", "complete"
        )
        staged_recovery_inventory = $artifactInventoryRelative
        persona_transport_verification_required = $true
        persona_combined_cap_reverification = $true
        excluded_volatile_paths = $volatileStatusPaths
        excluded_transport_lock = "**/.submission.lock"
        forbidden_future_artifact_gate = "preopen"
        excluded_rebuildable_caches = @("*.pt", "*.pth", "*.bin", "*.safetensors")
        maximum_staged_blob_bytes = Get-MaximumResearchBlobBytes
        known_rebuild_temporary_removed = "configs/steering_comparison_preopen_lock.rebuild.tmp"
        push_requested = [bool]$Push
    } | ConvertTo-Json
    return
}

try {
    Write-FreezeStatus -State "checking"
    $null = Remove-KnownFreezeRebuildTemporary -RepositoryRoot $RepositoryRoot `
        -RelativePath "configs/steering_comparison_preopen_lock.rebuild.tmp"
    Invoke-Native -Executable $comparisonExe -Arguments @("verify-stage1")
    Assert-NoForbiddenFreezeArtifacts -RepositoryRoot $RepositoryRoot `
        -ArtifactRoot $artifactRoot -Phase "preopen"
    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "main") {
        throw "pre-open freeze must run on main"
    }
    $trackedStatus = @(git status --porcelain --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "could not classify the pre-open freeze worktree"
    }
    foreach ($path in @($summaryPaths + $directionManifestPaths)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "pre-open freeze input is missing: $path"
        }
    }
    foreach ($requiredStatusPath in @(
        (Join-Path $artifactRoot "local_construction_status.json"),
        (Join-Path $artifactRoot "forced_grid_status.json"),
        (Join-Path $artifactRoot "preopen_summary_status.json"),
        (Join-Path $artifactRoot "interpolation_rechecks_required.json")
    )) {
        if (-not (Test-Path -LiteralPath $requiredStatusPath -PathType Leaf)) {
            throw "pre-open phase status is missing: $requiredStatusPath"
        }
    }
    $constructionStatus = Get-Content -Raw -LiteralPath (
        Join-Path $artifactRoot "local_construction_status.json"
    ) | ConvertFrom-Json
    if ($constructionStatus.phase -ne "local_construction" -or $constructionStatus.state -ne "complete") {
        throw "local construction is not complete"
    }
    $gridStatus = Get-Content -Raw -LiteralPath (
        Join-Path $artifactRoot "forced_grid_status.json"
    ) | ConvertFrom-Json
    if ($gridStatus.model_tag -ne "all" -or $gridStatus.state -ne "complete") {
        throw "both forced grids are not complete"
    }
    $summaryStatus = Get-Content -Raw -LiteralPath (
        Join-Path $artifactRoot "preopen_summary_status.json"
    ) | ConvertFrom-Json
    if ($summaryStatus.state -ne "complete") {
        throw "pre-open summaries are not complete"
    }
    $interpolationStatus = Get-Content -Raw -LiteralPath (
        Join-Path $artifactRoot "interpolation_rechecks_required.json"
    ) | ConvertFrom-Json
    if ($interpolationStatus.status -notin @("none_required", "completed")) {
        throw "interpolation rechecks remain pending"
    }

    Write-FreezeStatus -State "verifying_persona_judgments"
    foreach ($verification in $personaJudgeVerifications) {
        if (
            -not (Test-Path -LiteralPath $verification.requests -PathType Leaf) -or
            -not (Test-Path -LiteralPath $verification.responses -PathType Leaf) -or
            -not (Test-Path -LiteralPath $verification.work_directory -PathType Container)
        ) {
            throw "persona judge transport output is incomplete for $($verification.tag)"
        }
        Invoke-Native -Executable $pythonExe -Arguments @(
            $personaValidator,
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--model-config", [string]$verification.model_config,
            "raw", "--raw", [string]$verification.raw
        )
        $rebuiltRequests = "$($verification.requests).freeze-rebuilt.tmp"
        $rebuiltScored = "$($verification.scored).freeze-rebuilt.tmp"
        Remove-Item -LiteralPath $rebuiltRequests -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $rebuiltScored -Force -ErrorAction SilentlyContinue
        try {
            Invoke-Native -Executable $comparisonExe -Arguments @(
                "judge-requests", "--kind", "persona",
                "--input", [string]$verification.raw,
                "--output", $rebuiltRequests
            )
            if (
                (Get-FileHash -LiteralPath $rebuiltRequests -Algorithm SHA256).Hash -ne
                (Get-FileHash -LiteralPath $verification.requests -Algorithm SHA256).Hash
            ) {
                throw "persona requests differ byte-for-byte from locked regeneration"
            }
            Invoke-Native -Executable $pythonExe -Arguments @(
                $personaValidator,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--model-config", [string]$verification.model_config,
                "requests", "--raw", [string]$verification.raw,
                "--requests", [string]$verification.requests
            )
            Invoke-Native -Executable $pythonExe -Arguments @(
                $transport,
                "--requests", [string]$verification.requests,
                "--responses", [string]$verification.responses,
                "--work-dir", [string]$verification.work_directory,
                "--verify-only"
            )
            Invoke-Native -Executable $comparisonExe -Arguments @(
                "attach-judgments", "--kind", "persona",
                "--input", [string]$verification.raw,
                "--responses", [string]$verification.responses,
                "--output", $rebuiltScored
            )
            if (
                (Get-FileHash -LiteralPath $rebuiltScored -Algorithm SHA256).Hash -ne
                (Get-FileHash -LiteralPath $verification.scored -Algorithm SHA256).Hash
            ) {
                throw "persona scored rows differ byte-for-byte from exact receipt attachment"
            }
            Invoke-Native -Executable $pythonExe -Arguments @(
                $personaValidator,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--model-config", [string]$verification.model_config,
                "scored", "--raw", [string]$verification.raw,
                "--responses", [string]$verification.responses,
                "--scored", [string]$verification.scored
            )
            Invoke-Native -Executable $pythonExe -Arguments @(
                $personaValidator,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--model-config", [string]$verification.model_config,
                "manifest", "--scored", [string]$verification.scored,
                "--manifest", [string]$verification.manifest
            )
        }
        finally {
            Remove-Item -LiteralPath $rebuiltRequests -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $rebuiltScored -Force -ErrorAction SilentlyContinue
        }
    }
    $personaPreflights = @(
        foreach ($verification in $personaJudgeVerifications) {
            Get-Content -Raw -LiteralPath (
                Join-Path $verification.work_directory "cost_preflight.json"
            ) | ConvertFrom-Json
        }
    )
    $null = Assert-PersonaCombinedCap -Preflights $personaPreflights

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "failed to resolve the pre-open starting commit"
    }
    $preopenAlreadyCommitted = Test-GitPathAtCommit -RepositoryRoot $RepositoryRoot `
        -Commit $head -Path $preopenRelative
    $headSubject = ((Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "show", "-s", "--format=%s", $head
    )) -join "`n").Trim()

    if ($preopenAlreadyCommitted) {
        if ($trackedStatus.Count -ne 0) {
            throw "pre-open push recovery requires a clean worktree: $trackedStatus"
        }
        Assert-SinglePathCommit -RepositoryRoot $RepositoryRoot -Commit $head `
            -ExpectedSubject $manifestCommitSubject -ExpectedPath $preopenRelative
        $artifactCommit = (git rev-parse "$head^").Trim()
        if ($LASTEXITCODE -ne 0 -or $artifactCommit.Length -ne 40) {
            throw "failed to resolve artifact commit B from commit C"
        }
        $null = Assert-ArtifactFreezeCommit -RepositoryRoot $RepositoryRoot `
            -Commit $artifactCommit -ExpectedSubject $artifactCommitSubject -Phase "preopen" `
            -InventoryPath $artifactInventoryRelative
        $preopenCommit = $head
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot -Head $head
        Write-FreezeStatus -State "recovering_C" -Detail $preopenCommit
    }
    elseif ($headSubject -eq $artifactCommitSubject) {
        $allowedStatuses = @(
            "?? $preopenRelative", "A  $preopenRelative", " M $preopenRelative"
        )
        if (
            $trackedStatus.Count -gt 1 -or
            ($trackedStatus.Count -eq 1 -and $trackedStatus[0] -notin $allowedStatuses) -or
            ($trackedStatus.Count -eq 0 -and (Test-Path -LiteralPath $preopenManifest))
        ) {
            throw "artifact-commit recovery has unrelated or inconsistent worktree changes: $trackedStatus"
        }
        $artifactCommit = $head
        $null = Assert-ArtifactFreezeCommit -RepositoryRoot $RepositoryRoot `
            -Commit $artifactCommit -ExpectedSubject $artifactCommitSubject -Phase "preopen" `
            -InventoryPath $artifactInventoryRelative
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot -Head $head
        Write-FreezeStatus -State "recovering_B" -Detail $artifactCommit
    }
    else {
        if (Test-Path -LiteralPath $preopenManifest) {
            throw "new pre-open freeze requires no pre-open manifest"
        }
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot `
            -Head $head -RequireEqual
        $expectedPhasePaths = @(Get-PreopenExpectedArtifactPaths)
        $null = Assert-OnlyExpectedArtifactFiles -RepositoryRoot $RepositoryRoot `
            -ArtifactRoot $artifactRoot -BaseCommit $head `
            -ExpectedPhasePaths $expectedPhasePaths -InventoryPath $artifactInventoryRelative
        $stagedPaths = @(git diff --cached --name-only)
        if ($LASTEXITCODE -ne 0) {
            throw "could not inspect exact pre-open staging state"
        }
        $inventoryExists = Test-Path -LiteralPath $artifactInventoryPath -PathType Leaf
        if (-not $inventoryExists) {
            if ($stagedPaths.Count -ne 0) {
                throw "partial pre-open staging lacks its exact artifact inventory"
            }
            if ($trackedStatus.Count -ne 0) {
                throw "new pre-open freeze has unrelated worktree changes: $trackedStatus"
            }
            Write-FreezeStatus -State "staging_B"
            $inventory = New-FreezeArtifactInventory -RepositoryRoot $RepositoryRoot `
                -Phase "preopen" -BaseCommit $head -ExpectedPhasePaths $expectedPhasePaths `
                -InventoryPath $artifactInventoryRelative
        }
        else {
            Write-FreezeStatus -State "recovering_staged_B" -Detail $head
        }
        $null = Resume-ExactArtifactStaging -RepositoryRoot $RepositoryRoot `
            -InventoryPath $artifactInventoryRelative -Phase "preopen" -BaseCommit $head
        Invoke-Native -Executable "git" -Arguments @(
            "commit", "-m", $artifactCommitSubject
        )
        $artifactCommit = (git rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $artifactCommit.Length -ne 40) {
            throw "failed to resolve artifact commit B"
        }
        $null = Assert-ArtifactFreezeCommit -RepositoryRoot $RepositoryRoot `
            -Commit $artifactCommit -ExpectedSubject $artifactCommitSubject -Phase "preopen" `
            -InventoryPath $artifactInventoryRelative
    }

    $preopenArguments = @(
        "build-preopen-lock",
        "--calibration-summary"
    ) + $summaryPaths + @(
        "--direction-manifest"
    ) + $directionManifestPaths + @(
        "--output", $preopenManifest
    )
    if (-not $preopenAlreadyCommitted) {
        Write-FreezeStatus -State "building_C" -Detail $artifactCommit
        if (Test-Path -LiteralPath $preopenManifest -PathType Leaf) {
            $rebuildPath = Join-Path $RepositoryRoot "configs\steering_comparison_preopen_lock.rebuild.tmp"
            try {
                $rebuildArguments = @($preopenArguments)
                $rebuildArguments[-1] = $rebuildPath
                Invoke-Native -Executable $comparisonExe -Arguments $rebuildArguments
                if (
                    (Get-FileHash -LiteralPath $preopenManifest -Algorithm SHA256).Hash -ne
                    (Get-FileHash -LiteralPath $rebuildPath -Algorithm SHA256).Hash
                ) {
                    throw "uncommitted pre-open manifest differs from its exact canonical rebuild"
                }
            }
            finally {
                if (Test-Path -LiteralPath $rebuildPath) {
                    Remove-Item -LiteralPath $rebuildPath -Force
                }
            }
        }
        else {
            Invoke-Native -Executable $comparisonExe -Arguments $preopenArguments
        }
        Invoke-Native -Executable "git" -Arguments @(
            "add", "--", $preopenRelative
        )
        $stagedForC = @(git diff --cached --name-only)
        if ($stagedForC.Count -ne 1 -or $stagedForC[0] -ne $preopenRelative) {
            throw "commit C must contain only the pre-open manifest: $stagedForC"
        }
        Invoke-Native -Executable "git" -Arguments @(
            "commit", "-m", $manifestCommitSubject
        )
        $preopenCommit = (git rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $preopenCommit.Length -ne 40) {
            throw "failed to resolve pre-open commit C"
        }
    }
    Assert-SinglePathCommit -RepositoryRoot $RepositoryRoot -Commit $preopenCommit `
        -ExpectedSubject $manifestCommitSubject -ExpectedPath $preopenRelative
    $manifestPayload = Get-Content -Raw -LiteralPath $preopenManifest | ConvertFrom-Json
    if ([string]$manifestPayload.artifact_freeze_commit -ne $artifactCommit) {
        throw "pre-open manifest does not bind the exact artifact commit B"
    }
    Invoke-Native -Executable $comparisonExe -Arguments @(
        "verify-preopen", "--preopen-manifest", $preopenManifest
    )
    if (@(git status --porcelain --untracked-files=all).Count -ne 0) {
        throw "worktree is not clean after pre-open verification"
    }
    if ($Push) {
        Write-FreezeStatus -State "pushing" -Detail $preopenCommit
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot `
            -Head $preopenCommit
        Invoke-Native -Executable "git" -Arguments @("push", "origin", "main")
        Assert-RemoteMainEqualsHead -RepositoryRoot $RepositoryRoot -Head $preopenCommit
    }
    Write-FreezeStatus -State "complete" -Detail (
        "artifact_commit=$artifactCommit;preopen_commit=$preopenCommit;push=$([bool]$Push)"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-FreezeStatus -State "failed" -Detail $failureDetail
    throw
}
