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
$evaluationValidator = Join-Path $artifactRoot "validate_locked_evaluation_artifact.py"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$preopenManifest = Join-Path $RepositoryRoot "configs\steering_comparison_preopen_lock.json"
$stage2Manifest = Join-Path $RepositoryRoot "configs\steering_comparison_stage2_lock.json"
$validationPlanPath = Join-Path $artifactRoot "validation_open\validation_open_plan.json"
$sealedPlanPath = Join-Path $artifactRoot "sealed\sealed_evaluation_plan.json"
$statusPath = Join-Path $artifactRoot "final_freeze_status.json"
$finalCommitSubject = "Add sealed steering comparison results and adversarial review"
$artifactInventoryRelative = "artifacts/steering_comparison/final_artifact_inventory.json"
$artifactInventoryPath = Join-Path $RepositoryRoot $artifactInventoryRelative
$adversarialReviewValidator = Join-Path $artifactRoot "validate_adversarial_review.py"
$jspaceCompletionValidator = Join-Path $artifactRoot "verify_jspace_completion.py"
$adversarialReviewCompletion = Join-Path $artifactRoot "adversarial_review_completion.json"
$script:validatedJspaceArtifactPaths = $null
$requiredOutputRelatives = @(
    "artifacts/steering_comparison/final_report.json",
    "artifacts/steering_comparison/FINAL_REPORT.md",
    "artifacts/steering_comparison/ADVERSARIAL_REVIEW.md",
    "artifacts/steering_comparison/adversarial_review_completion.json"
)
$volatileStatusPaths = @(Get-UniversalFreezeVolatilePaths)
$requiredOutputs = @(
    (Join-Path $artifactRoot "final_report.json"),
    (Join-Path $artifactRoot "FINAL_REPORT.md"),
    (Join-Path $artifactRoot "ADVERSARIAL_REVIEW.md"),
    $adversarialReviewCompletion
)
$powerShellSelfTests = @(
    "run_remaining_local_construction.ps1",
    "run_locked_forced_grids.ps1",
    "build_locked_preopen_summaries.ps1",
    "run_locked_interpolation_rechecks.ps1",
    "freeze_locked_preopen.ps1",
    "complete_persona_directions.ps1",
    "run_validation_open_generation.ps1",
    "build_locked_final_summaries.ps1",
    "complete_validation_open.ps1",
    "freeze_locked_stage2.ps1",
    "run_sealed_evaluation.ps1",
    "complete_sealed_judgments.ps1",
    "run_jspace_secondary.ps1",
    "build_final_report.ps1"
    "freeze_safety_selftest.ps1"
)
$judgeTransportVerifications = @(
    [ordered]@{
        tag = "qwen35_08b_persona"
        required = $true
        requests = Join-Path $artifactRoot "qwen35_08b\persona_judge_requests.jsonl"
        responses = Join-Path $artifactRoot "qwen35_08b\persona_judge_responses.jsonl"
        work_directory = Join-Path $artifactRoot "qwen35_08b\persona_judge_transport"
    },
    [ordered]@{
        tag = "qwen35_2b_persona"
        required = $true
        requests = Join-Path $artifactRoot "qwen35_2b\persona_judge_requests.jsonl"
        responses = Join-Path $artifactRoot "qwen35_2b\persona_judge_responses.jsonl"
        work_directory = Join-Path $artifactRoot "qwen35_2b\persona_judge_transport"
    },
    [ordered]@{
        tag = "validation_open"
        required = $false
        requests = Join-Path $artifactRoot "validation_open\open_judge_requests.jsonl"
        responses = Join-Path $artifactRoot "validation_open\open_judge_responses.jsonl"
        work_directory = Join-Path $artifactRoot "validation_open\judge_transport"
    },
    [ordered]@{
        tag = "sealed_open"
        required = $false
        requests = Join-Path $artifactRoot "sealed\open_judge_requests.jsonl"
        responses = Join-Path $artifactRoot "sealed\open_judge_responses.jsonl"
        work_directory = Join-Path $artifactRoot "sealed\judge_transport"
    }
)

function Get-FinalExpectedArtifactPaths {
    $paths = @()
    foreach ($name in @(
        "sealed_evaluation_status.json",
        "sealed_evaluation.log",
        "sealed_judgment_status.json",
        "sealed_judgment.log",
        "jspace_status.json",
        "jspace.log",
        "jspace_completion.json",
        "report_status.json",
        "report.log",
        "final_report.json",
        "FINAL_REPORT.md",
        "ADVERSARIAL_REVIEW.md",
        "adversarial_review_completion.json"
    )) {
        $path = Join-Path $artifactRoot $name
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
        }
    }
    $plan = Get-Content -Raw -LiteralPath $sealedPlanPath | ConvertFrom-Json
    $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $sealedPlanPath
    foreach ($setup in @($plan.setups)) {
        foreach ($field in @("forced_path")) {
            $path = Join-Path $RepositoryRoot ([string]$setup.$field)
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "final exact sealed artifact is missing: $path"
            }
            $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
        }
        if ($setup.open_required -eq $true) {
            foreach ($field in @("generation_path", "scored_path")) {
                $path = Join-Path $RepositoryRoot ([string]$setup.$field)
                if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                    throw "final exact sealed open artifact is missing: $path"
                }
                $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
            }
        }
    }
    foreach ($name in @("open_generations_all.jsonl", "open_judge_requests.jsonl", `
        "open_judge_responses.jsonl", "open_scored_all.jsonl")) {
        $path = Join-Path $artifactRoot "sealed\$name"
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $paths += ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $path
        }
    }
    $sealedVerification = @($judgeTransportVerifications | Where-Object { $_.tag -eq "sealed_open" })
    if ($sealedVerification.Count -ne 1) {
        throw "final transport registry lacks sealed_open"
    }
    if ([bool]$sealedVerification[0].required) {
        $paths += @(Get-LockedJudgeTransportArtifactPaths -RepositoryRoot $RepositoryRoot `
            -RequestsPath ([string]$sealedVerification[0].requests) `
            -ResponsesPath ([string]$sealedVerification[0].responses) `
            -WorkDirectory ([string]$sealedVerification[0].work_directory))
    }
    if ($null -eq $script:validatedJspaceArtifactPaths) {
        throw "final artifact inventory requested before exact J-space validation"
    }
    $paths += @($script:validatedJspaceArtifactPaths)
    return @($paths | Sort-Object -Unique)
}

function Write-FinalFreezeStatus {
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
        required_outputs = @(
            "final_report.json", "FINAL_REPORT.md", "ADVERSARIAL_REVIEW.md",
            "adversarial_review_completion.json"
        )
        stage2_reverification_required = $true
        core_tests_required = $true
        ruff_required = $true
        powershell_self_tests = $powerShellSelfTests
        judge_transport_verifications = @($judgeTransportVerifications.tag)
        open_transport_requiredness_sources = @(
            "locked_validation_open_plan", "locked_sealed_evaluation_plan"
        )
        structured_adversarial_review_items_required = 38
        persona_combined_cap_reverification = $true
        excluded_volatile_paths = $volatileStatusPaths
        excluded_transport_lock = "**/.submission.lock"
        excluded_rebuildable_caches = @("*.pt", "*.pth", "*.bin", "*.safetensors")
        maximum_staged_blob_bytes = Get-MaximumResearchBlobBytes
        deterministic_report_rebuild_required = $true
        final_push_recovery_from_exact_commit = $true
        restart_states = @("new", "final_staged_uncommitted", "final_committed_unpushed", "complete")
        staged_recovery_inventory = $artifactInventoryRelative
        push_requested = [bool]$Push
    } | ConvertTo-Json
    return
}

try {
    Write-FinalFreezeStatus -State "checking"
    Invoke-Native -Executable $comparisonExe -Arguments @(
        "verify-stage2", "--stage2-manifest", $stage2Manifest
    )
    foreach ($planSpec in @(
        [ordered]@{
            path = $validationPlanPath
            split = "validation"
            all_open = $true
            manifest = $preopenManifest
            output_directory = Join-Path $artifactRoot "validation_open"
        },
        [ordered]@{
            path = $sealedPlanPath
            split = "sealed_test"
            all_open = $false
            manifest = $stage2Manifest
            output_directory = Join-Path $artifactRoot "sealed"
        }
    )) {
        $null = Assert-LockedOpenPlanCanonical -PythonExecutable $pythonExe `
            -OrchestratorPath $orchestrator -RepositoryRoot $RepositoryRoot `
            -LockPath $lockPath -ManifestPath ([string]$planSpec.manifest) `
            -OutputDirectory ([string]$planSpec.output_directory) `
            -Split ([string]$planSpec.split) -PlanPath ([string]$planSpec.path)
        $requirementArguments = @{
            PlanPath = [string]$planSpec.path
            ExpectedSplit = [string]$planSpec.split
        }
        if ([bool]$planSpec.all_open) {
            $requirementArguments.RequireEverySetupOpen = $true
        }
        $requirement = Get-LockedOpenTransportRequirement @requirementArguments
        $required = [bool]$requirement.required
        $tag = if ($planSpec.split -eq "validation") { "validation_open" } else { "sealed_open" }
        $verification = @($judgeTransportVerifications | Where-Object { $_.tag -eq $tag })
        if ($verification.Count -ne 1) {
            throw "judge transport verification registry lacks exactly one $tag entry"
        }
        $verification[0].required = $required
        if ($required) {
            Invoke-Native -Executable $pythonExe -Arguments @(
                $openChainVerifier,
                "--repo-root", $RepositoryRoot,
                "--lock", $lockPath,
                "--plan", [string]$planSpec.path,
                "--combined", (Join-Path ([string]$planSpec.output_directory) "open_generations_all.jsonl"),
                "--requests", [string]$verification[0].requests,
                "--responses", [string]$verification[0].responses,
                "--scored", (Join-Path ([string]$planSpec.output_directory) "open_scored_all.jsonl")
            )
        }
    }
    foreach ($statusName in @(
        "sealed_evaluation_status.json",
        "sealed_judgment_status.json",
        "jspace_status.json",
        "report_status.json"
    )) {
        $path = Join-Path $artifactRoot $statusName
        $status = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
        if ($status.state -ne "complete") {
            throw "final prerequisite did not complete: $path"
        }
    }
    $sealedPlan = Get-Content -Raw -LiteralPath $sealedPlanPath | ConvertFrom-Json
    foreach ($setup in @($sealedPlan.setups)) {
        Invoke-Native -Executable $pythonExe -Arguments @(
            $evaluationValidator,
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--plan", $sealedPlanPath,
            "--setup-id", [string]$setup.setup_id,
            "--path", (Join-Path $RepositoryRoot ([string]$setup.forced_path)),
            "--kind", "forced"
        )
    }
    $jspaceReceipt = Get-ValidatedJspaceCompletion -PythonExecutable $pythonExe `
        -ValidatorPath $jspaceCompletionValidator -RepositoryRoot $RepositoryRoot `
        -PlanPath $sealedPlanPath -LockPath $lockPath `
        -StatusPath (Join-Path $artifactRoot "jspace_status.json") `
        -RecordsDirectory (Join-Path $artifactRoot "jspace\records") `
        -AtomsRoot (Join-Path $artifactRoot "jspace\atoms") `
        -CompletionPath (Join-Path $artifactRoot "jspace_completion.json")
    $script:validatedJspaceArtifactPaths = @(
        $jspaceReceipt.artifact_paths
        "artifacts/steering_comparison/jspace_completion.json"
    )
    foreach ($path in $requiredOutputs) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "final output is missing: $path"
        }
        if ((Get-Item -LiteralPath $path).Length -eq 0) {
            throw "final output is empty: $path"
        }
    }
    Invoke-Native -Executable $pythonExe -Arguments @(
        $adversarialReviewValidator,
        "--checklist", (Join-Path $artifactRoot "ADVERSARIAL_REVIEW_CHECKLIST.md"),
        "--report", $requiredOutputs[0],
        "--review", $requiredOutputs[2],
        "--completion", $adversarialReviewCompletion
    )
    $temporaryArtifacts = @(
        Get-ChildItem -LiteralPath $artifactRoot -Recurse -File -Filter "*.tmp" |
            Select-Object -ExpandProperty FullName
    )
    if ($temporaryArtifacts.Count -ne 0) {
        throw "temporary atomic-write artifacts remain: $($temporaryArtifacts -join ', ')"
    }
    $branch = (git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne "main") {
        throw "final freeze must run on main"
    }
    $trackedStatus = @(git status --porcelain --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "could not classify the final-freeze worktree"
    }
    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
        throw "failed to resolve the final-freeze starting commit"
    }
    $headSubject = ((Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "show", "-s", "--format=%s", $head
    )) -join "`n").Trim()
    $finalAlreadyCommitted = $headSubject -eq $finalCommitSubject
    if ($finalAlreadyCommitted) {
        if ($trackedStatus.Count -ne 0) {
            throw "final push recovery requires a clean worktree: $trackedStatus"
        }
        $finalCommitPaths = @(Assert-ArtifactFreezeCommit -RepositoryRoot $RepositoryRoot `
            -Commit $head -ExpectedSubject $finalCommitSubject -Phase "final" `
            -InventoryPath $artifactInventoryRelative)
        foreach ($requiredRelative in $requiredOutputRelatives) {
            if ($requiredRelative -notin $finalCommitPaths) {
                throw "final recovery commit lacks required output $requiredRelative"
            }
        }
        $commit = $head
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot -Head $head
        Write-FinalFreezeStatus -State "recovering_final_commit" -Detail $commit
    }
    else {
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot `
            -Head $head -RequireEqual
        $expectedPhasePaths = @(Get-FinalExpectedArtifactPaths)
        $null = Assert-OnlyExpectedArtifactFiles -RepositoryRoot $RepositoryRoot `
            -ArtifactRoot $artifactRoot -BaseCommit $head `
            -ExpectedPhasePaths $expectedPhasePaths -InventoryPath $artifactInventoryRelative
        $preStaged = @(git diff --cached --name-only)
        if ($LASTEXITCODE -ne 0) {
            throw "could not inspect exact final staging state"
        }
        $inventoryExists = Test-Path -LiteralPath $artifactInventoryPath -PathType Leaf
        if ($preStaged.Count -gt 0 -or $inventoryExists) {
            if (-not $inventoryExists) {
                throw "partial final staging lacks its exact artifact inventory"
            }
            $null = Resume-ExactArtifactStaging -RepositoryRoot $RepositoryRoot `
                -InventoryPath $artifactInventoryRelative -Phase "final" -BaseCommit $head
            Write-FinalFreezeStatus -State "recovering_staged_final" -Detail $head
        }
        elseif ($trackedStatus.Count -ne 0) {
            throw "new final freeze has unrelated worktree changes: $trackedStatus"
        }
    }

    Write-FinalFreezeStatus -State "testing"
    Invoke-Native -Executable $pythonExe -Arguments @("-m", "pytest", "-q")
    Invoke-Native -Executable $pythonExe -Arguments @("-m", "ruff", "check", ".")
    Invoke-Native -Executable $pythonExe -Arguments @(
        "-m", "ruff", "check",
        "artifacts/steering_comparison/locked_open_orchestration.py",
        "artifacts/steering_comparison/test_locked_open_orchestration.py",
        "artifacts/steering_comparison/validate_locked_evaluation_artifact.py",
        "artifacts/steering_comparison/test_validate_locked_evaluation_artifact.py",
        "artifacts/steering_comparison/validate_interpolation_recheck.py",
        "artifacts/steering_comparison/test_validate_interpolation_recheck.py",
        "artifacts/steering_comparison/verify_open_judgment_chain.py",
        "artifacts/steering_comparison/test_verify_open_judgment_chain.py",
        "artifacts/steering_comparison/submit_openai_judge_requests.py",
        "artifacts/steering_comparison/test_submit_openai_judge_requests.py",
        "artifacts/steering_comparison/validate_jspace_cache.py",
        "artifacts/steering_comparison/test_validate_jspace_cache.py",
        "artifacts/steering_comparison/validate_jspace_record.py",
        "artifacts/steering_comparison/test_validate_jspace_record.py",
        "artifacts/steering_comparison/verify_jspace_completion.py",
        "artifacts/steering_comparison/test_verify_jspace_completion.py",
        "artifacts/steering_comparison/validate_adversarial_review.py",
        "artifacts/steering_comparison/test_validate_adversarial_review.py"
    )
    Invoke-Native -Executable $pythonExe -Arguments @(
        "-m", "pytest", "-q",
        "artifacts/steering_comparison/test_locked_open_orchestration.py",
        "artifacts/steering_comparison/test_validate_locked_evaluation_artifact.py",
        "artifacts/steering_comparison/test_validate_interpolation_recheck.py",
        "artifacts/steering_comparison/test_verify_open_judgment_chain.py",
        "artifacts/steering_comparison/test_submit_openai_judge_requests.py",
        "artifacts/steering_comparison/test_validate_jspace_cache.py",
        "artifacts/steering_comparison/test_validate_jspace_record.py",
        "artifacts/steering_comparison/test_verify_jspace_completion.py",
        "artifacts/steering_comparison/test_validate_adversarial_review.py"
    )
    foreach ($scriptName in $powerShellSelfTests) {
        Invoke-Native -Executable "powershell.exe" -Arguments @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            (Join-Path $artifactRoot $scriptName),
            "-RepositoryRoot", $RepositoryRoot,
            "-SelfTest"
        )
    }
    foreach ($verification in $judgeTransportVerifications) {
        $transportPresent = Assert-LockedJudgeTransportPresence `
            -Required ([bool]$verification.required) `
            -RequestsPath ([string]$verification.requests) `
            -ResponsesPath ([string]$verification.responses) `
            -WorkDirectory ([string]$verification.work_directory) `
            -Label ([string]$verification.tag)
        if (-not $transportPresent) {
            continue
        }
        Invoke-Native -Executable $pythonExe -Arguments @(
            $transport,
            "--requests", [string]$verification.requests,
            "--responses", [string]$verification.responses,
            "--work-dir", [string]$verification.work_directory,
            "--verify-only"
        )
    }
    $personaPreflights = @(
        $(
            Get-Content -Raw -LiteralPath (
                Join-Path $artifactRoot "qwen35_08b\persona_judge_transport\cost_preflight.json"
            ) | ConvertFrom-Json
        )
        $(
            Get-Content -Raw -LiteralPath (
                Join-Path $artifactRoot "qwen35_2b\persona_judge_transport\cost_preflight.json"
            ) | ConvertFrom-Json
        )
    )
    $null = Assert-PersonaCombinedCap -Preflights $personaPreflights
    Invoke-Native -Executable $comparisonExe -Arguments @(
        "verify-stage2", "--stage2-manifest", $stage2Manifest
    )

    $null = Invoke-IsolatedFinalReportRebuild -RepositoryRoot $RepositoryRoot `
        -BuilderPath (Join-Path $artifactRoot "build_final_report.ps1") `
        -ExpectedJsonPath $requiredOutputs[0] `
        -ExpectedMarkdownPath $requiredOutputs[1]

    if (-not $finalAlreadyCommitted) {
        $staged = @(git diff --cached --name-only)
        if ($LASTEXITCODE -ne 0) {
            throw "could not inspect final-result staging"
        }
        $inventoryExists = Test-Path -LiteralPath $artifactInventoryPath -PathType Leaf
        if (-not $inventoryExists) {
            if ($staged.Count -ne 0) {
                throw "partial final staging lacks its exact artifact inventory"
            }
            Write-FinalFreezeStatus -State "staging"
            $inventory = New-FreezeArtifactInventory -RepositoryRoot $RepositoryRoot `
                -Phase "final" -BaseCommit $head -ExpectedPhasePaths $expectedPhasePaths `
                -InventoryPath $artifactInventoryRelative
        }
        $null = Resume-ExactArtifactStaging -RepositoryRoot $RepositoryRoot `
            -InventoryPath $artifactInventoryRelative -Phase "final" -BaseCommit $head
        Invoke-Native -Executable "git" -Arguments @(
            "commit", "-m", $finalCommitSubject
        )
        $commit = (git rev-parse HEAD).Trim()
        $finalCommitPaths = @(Assert-ArtifactFreezeCommit -RepositoryRoot $RepositoryRoot `
            -Commit $commit -ExpectedSubject $finalCommitSubject -Phase "final" `
            -InventoryPath $artifactInventoryRelative)
        foreach ($requiredRelative in $requiredOutputRelatives) {
            if ($requiredRelative -notin $finalCommitPaths) {
                throw "final result commit lacks required output $requiredRelative"
            }
        }
    }
    Invoke-Native -Executable $comparisonExe -Arguments @(
        "verify-stage2", "--stage2-manifest", $stage2Manifest
    )
    if (@(git status --porcelain --untracked-files=all).Count -ne 0) {
        throw "worktree is not clean after final result commit"
    }
    if ($Push) {
        Write-FinalFreezeStatus -State "pushing" -Detail $commit
        $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $RepositoryRoot -Head $commit
        Invoke-Native -Executable "git" -Arguments @("push", "origin", "main")
        Assert-RemoteMainEqualsHead -RepositoryRoot $RepositoryRoot -Head $commit
    }
    Write-FinalFreezeStatus -State "complete" -Detail (
        "commit=$commit;push=$([bool]$Push)"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-FinalFreezeStatus -State "failed" -Detail $failureDetail
    throw
}
