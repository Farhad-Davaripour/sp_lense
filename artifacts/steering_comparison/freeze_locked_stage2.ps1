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
$orchestrator = Join-Path $artifactRoot "locked_open_orchestration.py"
$openChainVerifier = Join-Path $artifactRoot "verify_open_judgment_chain.py"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$preopenManifest = Join-Path $RepositoryRoot "configs\steering_comparison_preopen_lock.json"
$stage2Manifest = Join-Path $RepositoryRoot "configs\steering_comparison_stage2_lock.json"
$stage2Relative = "configs/steering_comparison_stage2_lock.json"
$artifactCommitSubject = "Freeze validation open confirmations"
$manifestCommitSubject = "Lock stage-two steering comparison artifacts"
$environmentLock = Join-Path $artifactRoot "environment.json"
$statusPath = Join-Path $artifactRoot "stage2_freeze_status.json"
$validationPlanPath = Join-Path $artifactRoot "validation_open\validation_open_plan.json"
$artifactInventoryRelative = "artifacts/steering_comparison/stage2_artifact_inventory.json"
$artifactInventoryPath = Join-Path $RepositoryRoot $artifactInventoryRelative
$volatileStatusPaths = @(Get-UniversalFreezeVolatilePaths)
$summaryNames = @(
    "gradient_matched_final.json",
    "gradient_uncorrected_matched_final.json",
    "caa_matched_final.json",
    "bipo_matched_final.json",
    "persona_vector_matched_final.json",
    "caa_canonical_final.json",
    "bipo_canonical_final.json",
    "persona_vector_canonical_final.json"
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
$validationJudgeVerification = [ordered]@{
    requests = Join-Path $artifactRoot "validation_open\open_judge_requests.jsonl"
    responses = Join-Path $artifactRoot "validation_open\open_judge_responses.jsonl"
    work_directory = Join-Path $artifactRoot "validation_open\judge_transport"
}

function Get-Stage2ExpectedArtifactPaths {
    param([Parameter(Mandatory = $true)][bool]$TransportRequired)

    $paths = @()
    foreach ($name in @(
        "validation_open_generation_status.json",
        "validation_open_generation.log",
        "validation_open_completion_status.json",
        "validation_open_completion.log",
        "final_summary_status.json",
        "final_summary_build.log"
    )) {
        $path = Join-Path $artifactRoot $name
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
        }
    }
    $plan = Get-Content -Raw -LiteralPath $validationPlanPath | ConvertFrom-Json
    $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot `
        -Path $validationPlanPath
    foreach ($setup in @($plan.setups)) {
        foreach ($field in @("generation_path", "scored_path")) {
            $path = Join-Path $RepositoryRoot ([string]$setup.$field)
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "stage-two exact validation artifact is missing: $path"
            }
            $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
        }
    }
    foreach ($name in @("open_generations_all.jsonl", "open_judge_requests.jsonl", `
        "open_judge_responses.jsonl", "open_scored_all.jsonl")) {
        $path = Join-Path $artifactRoot "validation_open\$name"
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
        }
    }
    if ($TransportRequired) {
        $paths += @(Get-LockedJudgeTransportArtifactPaths -RepositoryRoot $RepositoryRoot `
            -RequestsPath ([string]$validationJudgeVerification.requests) `
            -ResponsesPath ([string]$validationJudgeVerification.responses) `
            -WorkDirectory ([string]$validationJudgeVerification.work_directory))
    }
    $paths += @(
        $summaryPaths | ForEach-Object {
            ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $_
        }
    )
    return @($paths | Sort-Object -Unique)
}

function Write-Stage2Status {
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

if ($SelfTest) {
    if ($summaryPaths.Count -ne 16 -or (@($summaryPaths | Select-Object -Unique)).Count -ne 16) {
        throw "stage-two freeze must receive 16 unique final summaries"
    }
    if (
        $directionManifestPaths.Count -ne 12 -or
        (@($directionManifestPaths | Select-Object -Unique)).Count -ne 12
    ) {
        throw "stage-two freeze must receive 12 unique direction manifests"
    }
    [ordered]@{
        status = "self_test_passed"
        summary_count = 16
        direction_manifest_count = 12
        commits = @("D_validation_artifacts", "E_stage2_manifest")
        restart_states = @(
            "new", "D_staged_uncommitted", "D_committed_E_absent", "E_file_uncommitted",
            "E_committed_unpushed", "complete"
        )
        staged_recovery_inventory = $artifactInventoryRelative
        validation_transport_verification_required = $true
        validation_transport_requiredness_source = "locked_validation_open_plan"
        excluded_volatile_paths = $volatileStatusPaths
        excluded_transport_lock = "**/.submission.lock"
        forbidden_future_artifact_gate = "stage2"
        excluded_rebuildable_caches = @("*.pt", "*.pth", "*.bin", "*.safetensors")
        maximum_staged_blob_bytes = Get-MaximumResearchBlobBytes
        known_rebuild_temporary_removed = "configs/steering_comparison_stage2_lock.rebuild.tmp"
        push_requested = [bool]$Push
    } | ConvertTo-Json
    return
}

try {
    Write-Stage2Status -State "checking"
    $null = Remove-KnownFreezeRebuildTemporary -RepositoryRoot $RepositoryRoot `
        -RelativePath "configs/steering_comparison_stage2_lock.rebuild.tmp"
    Invoke-Native -Executable $comparisonExe -Arguments @(
        "verify-preopen", "--preopen-manifest", $preopenManifest
    )
    Assert-NoForbiddenFreezeArtifacts -RepositoryRoot $RepositoryRoot `
        -ArtifactRoot $artifactRoot -Phase "stage2"
    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "main") {
        throw "stage-two freeze must run on main"
    }
    $trackedStatus = @(git status --porcelain --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "could not classify the stage-two freeze worktree"
    }
    foreach ($path in @($summaryPaths + $directionManifestPaths + @($environmentLock))) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "stage-two freeze input is missing: $path"
        }
    }
    $completionStatus = Get-Content -Raw -LiteralPath (
        Join-Path $artifactRoot "validation_open_completion_status.json"
    ) | ConvertFrom-Json
    if ($completionStatus.state -ne "complete") {
        throw "validation open completion is not complete"
    }
    $summaryStatus = Get-Content -Raw -LiteralPath (
        Join-Path $artifactRoot "final_summary_status.json"
    ) | ConvertFrom-Json
    if ($summaryStatus.state -ne "complete" -or $summaryStatus.summary_count -ne 16) {
        throw "final calibration coverage is not complete"
    }
    $null = Assert-LockedOpenPlanCanonical -PythonExecutable $pythonExe `
        -OrchestratorPath $orchestrator -RepositoryRoot $RepositoryRoot `
        -LockPath $lockPath -ManifestPath $preopenManifest `
        -OutputDirectory (Join-Path $artifactRoot "validation_open") `
        -Split "validation" -PlanPath $validationPlanPath
    $validationRequirement = Get-LockedOpenTransportRequirement `
        -PlanPath $validationPlanPath -ExpectedSplit "validation" `
        -RequireEverySetupOpen
    $validationTransportRequired = [bool]$validationRequirement.required

    Write-Stage2Status -State "verifying_validation_judgments"
    $transportPresent = Assert-LockedJudgeTransportPresence `
        -Required $validationTransportRequired `
        -RequestsPath $validationJudgeVerification.requests `
        -ResponsesPath $validationJudgeVerification.responses `
        -WorkDirectory $validationJudgeVerification.work_directory `
        -Label "validation-open"
    if ($transportPresent) {
        Invoke-Native -Executable $pythonExe -Arguments @(
            $transport,
            "--requests", [string]$validationJudgeVerification.requests,
            "--responses", [string]$validationJudgeVerification.responses,
            "--work-dir", [string]$validationJudgeVerification.work_directory,
            "--verify-only"
        )
        Invoke-Native -Executable $pythonExe -Arguments @(
            $openChainVerifier,
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--plan", $validationPlanPath,
            "--combined", (Join-Path $artifactRoot "validation_open\open_generations_all.jsonl"),
            "--requests", [string]$validationJudgeVerification.requests,
            "--responses", [string]$validationJudgeVerification.responses,
            "--scored", (Join-Path $artifactRoot "validation_open\open_scored_all.jsonl")
        )
    }

    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "failed to resolve the stage-two starting commit"
    }
    $stage2AlreadyCommitted = Test-GitPathAtCommit -RepositoryRoot $RepositoryRoot `
        -Commit $head -Path $stage2Relative
    $headSubject = ((Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "show", "-s", "--format=%s", $head
    )) -join "`n").Trim()

    if ($stage2AlreadyCommitted) {
        if ($trackedStatus.Count -ne 0) {
            throw "stage-two push recovery requires a clean worktree: $trackedStatus"
        }
        Assert-SinglePathCommit -RepositoryRoot $RepositoryRoot -Commit $head `
            -ExpectedSubject $manifestCommitSubject -ExpectedPath $stage2Relative
        $artifactCommit = (git rev-parse "$head^").Trim()
        if ($LASTEXITCODE -ne 0 -or $artifactCommit.Length -ne 40) {
            throw "failed to resolve artifact commit D from commit E"
        }
        $null = Assert-ArtifactFreezeCommit -RepositoryRoot $RepositoryRoot `
            -Commit $artifactCommit -ExpectedSubject $artifactCommitSubject -Phase "stage2" `
            -InventoryPath $artifactInventoryRelative
        $stage2Commit = $head
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot -Head $head
        Write-Stage2Status -State "recovering_E" -Detail $stage2Commit
    }
    elseif ($headSubject -eq $artifactCommitSubject) {
        $allowedStatuses = @(
            "?? $stage2Relative", "A  $stage2Relative", " M $stage2Relative"
        )
        if (
            $trackedStatus.Count -gt 1 -or
            ($trackedStatus.Count -eq 1 -and $trackedStatus[0] -notin $allowedStatuses) -or
            ($trackedStatus.Count -eq 0 -and (Test-Path -LiteralPath $stage2Manifest))
        ) {
            throw "artifact-commit recovery has unrelated or inconsistent worktree changes: $trackedStatus"
        }
        $artifactCommit = $head
        $null = Assert-ArtifactFreezeCommit -RepositoryRoot $RepositoryRoot `
            -Commit $artifactCommit -ExpectedSubject $artifactCommitSubject -Phase "stage2" `
            -InventoryPath $artifactInventoryRelative
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot -Head $head
        Write-Stage2Status -State "recovering_D" -Detail $artifactCommit
    }
    else {
        if (Test-Path -LiteralPath $stage2Manifest) {
            throw "new stage-two freeze requires no stage-two manifest"
        }
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot `
            -Head $head -RequireEqual
        $expectedPhasePaths = @(Get-Stage2ExpectedArtifactPaths `
            -TransportRequired $validationTransportRequired)
        $null = Assert-OnlyExpectedArtifactFiles -RepositoryRoot $RepositoryRoot `
            -ArtifactRoot $artifactRoot -BaseCommit $head `
            -ExpectedPhasePaths $expectedPhasePaths -InventoryPath $artifactInventoryRelative
        $stagedPaths = @(git diff --cached --name-only)
        if ($LASTEXITCODE -ne 0) {
            throw "could not inspect exact stage-two staging state"
        }
        $inventoryExists = Test-Path -LiteralPath $artifactInventoryPath -PathType Leaf
        if (-not $inventoryExists) {
            if ($stagedPaths.Count -ne 0) {
                throw "partial stage-two staging lacks its exact artifact inventory"
            }
            if ($trackedStatus.Count -ne 0) {
                throw "new stage-two freeze has unrelated worktree changes: $trackedStatus"
            }
            Write-Stage2Status -State "staging_D"
            $inventory = New-FreezeArtifactInventory -RepositoryRoot $RepositoryRoot `
                -Phase "stage2" -BaseCommit $head -ExpectedPhasePaths $expectedPhasePaths `
                -InventoryPath $artifactInventoryRelative
        }
        else {
            Write-Stage2Status -State "recovering_staged_D" -Detail $head
        }
        $null = Resume-ExactArtifactStaging -RepositoryRoot $RepositoryRoot `
            -InventoryPath $artifactInventoryRelative -Phase "stage2" -BaseCommit $head
        Invoke-Native -Executable "git" -Arguments @(
            "commit", "-m", $artifactCommitSubject
        )
        $artifactCommit = (git rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $artifactCommit.Length -ne 40) {
            throw "failed to resolve validation artifact commit D"
        }
        $null = Assert-ArtifactFreezeCommit -RepositoryRoot $RepositoryRoot `
            -Commit $artifactCommit -ExpectedSubject $artifactCommitSubject -Phase "stage2" `
            -InventoryPath $artifactInventoryRelative
    }

    $stage2Arguments = @(
        "build-stage2-manifest",
        "--preopen-manifest", $preopenManifest,
        "--environment-lock", $environmentLock,
        "--calibration-summary"
    ) + $summaryPaths + @(
        "--direction-manifest"
    ) + $directionManifestPaths + @(
        "--output", $stage2Manifest
    )
    if (-not $stage2AlreadyCommitted) {
        Write-Stage2Status -State "building_E" -Detail $artifactCommit
        if (Test-Path -LiteralPath $stage2Manifest -PathType Leaf) {
            $rebuildPath = Join-Path $RepositoryRoot "configs\steering_comparison_stage2_lock.rebuild.tmp"
            try {
                $rebuildArguments = @($stage2Arguments)
                $rebuildArguments[-1] = $rebuildPath
                Invoke-Native -Executable $comparisonExe -Arguments $rebuildArguments
                if (
                    (Get-FileHash -LiteralPath $stage2Manifest -Algorithm SHA256).Hash -ne
                    (Get-FileHash -LiteralPath $rebuildPath -Algorithm SHA256).Hash
                ) {
                    throw "uncommitted stage-two manifest differs from its exact canonical rebuild"
                }
            }
            finally {
                if (Test-Path -LiteralPath $rebuildPath) {
                    Remove-Item -LiteralPath $rebuildPath -Force
                }
            }
        }
        else {
            Invoke-Native -Executable $comparisonExe -Arguments $stage2Arguments
        }
        Invoke-Native -Executable "git" -Arguments @("add", "--", $stage2Relative)
        $stagedForE = @(git diff --cached --name-only)
        if ($stagedForE.Count -ne 1 -or $stagedForE[0] -ne $stage2Relative) {
            throw "commit E must contain only the stage-two manifest: $stagedForE"
        }
        Invoke-Native -Executable "git" -Arguments @(
            "commit", "-m", $manifestCommitSubject
        )
        $stage2Commit = (git rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $stage2Commit.Length -ne 40) {
            throw "failed to resolve stage-two manifest commit E"
        }
    }
    Assert-SinglePathCommit -RepositoryRoot $RepositoryRoot -Commit $stage2Commit `
        -ExpectedSubject $manifestCommitSubject -ExpectedPath $stage2Relative
    $stage2Payload = Get-Content -Raw -LiteralPath $stage2Manifest | ConvertFrom-Json
    if ([string]$stage2Payload.artifact_freeze_commit -ne $artifactCommit) {
        throw "stage-two manifest does not bind the exact artifact commit D"
    }
    Invoke-Native -Executable $comparisonExe -Arguments @(
        "verify-stage2", "--stage2-manifest", $stage2Manifest
    )
    if (@(git status --porcelain --untracked-files=all).Count -ne 0) {
        throw "worktree is not clean after stage-two verification"
    }
    if ($Push) {
        Write-Stage2Status -State "pushing" -Detail $stage2Commit
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot `
            -Head $stage2Commit
        Invoke-Native -Executable "git" -Arguments @("push", "origin", "main")
        Assert-RemoteMainEqualsHead -RepositoryRoot $RepositoryRoot -Head $stage2Commit
    }
    Write-Stage2Status -State "complete" -Detail (
        "artifact_commit=$artifactCommit;stage2_commit=$stage2Commit;push=$([bool]$Push)"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-Stage2Status -State "failed" -Detail $failureDetail
    throw
}
