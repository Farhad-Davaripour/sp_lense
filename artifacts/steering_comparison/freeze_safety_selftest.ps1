param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $RepositoryRoot "artifacts\steering_comparison\freeze_safety.ps1")

if (-not $SelfTest) {
    throw "freeze_safety_selftest.ps1 is only a side-effect-free self-test"
}

$preopenForbidden = @(
    "artifacts/steering_comparison/validation_open/plan.json",
    "artifacts\steering_comparison\validation_open_generation_status.json",
    "artifacts/steering_comparison/sealed/results.jsonl",
    "artifacts/steering_comparison/jspace/atoms/model/atoms.pt",
    "artifacts/steering_comparison/final_report.json",
    "artifacts/steering_comparison/stage2_freeze_status.json",
    "artifacts/steering_comparison/stage2_freeze.log",
    "artifacts/steering_comparison/final_freeze_status.json",
    "artifacts/steering_comparison/final_freeze.log",
    "artifacts/steering_comparison/nested/file.tmp"
)
foreach ($path in $preopenForbidden) {
    if (-not (Test-ForbiddenFreezeArtifactPath -Path $path -Phase "preopen")) {
        throw "preopen forbidden-path self-test accepted $path"
    }
}
foreach ($path in @(
    "artifacts/steering_comparison/ADVERSARIAL_REVIEW_CHECKLIST.md",
    "artifacts/steering_comparison/build_final_report.ps1",
    "artifacts/steering_comparison/x/.submission.lock",
    "artifacts/steering_comparison/qwen35_08b/calibration/gradient_matched_preopen.json"
)) {
    if (Test-ForbiddenFreezeArtifactPath -Path $path -Phase "preopen") {
        throw "preopen forbidden-path self-test rejected $path"
    }
}
if (-not (Test-ForbiddenFreezeArtifactPath -Path (
    "artifacts/steering_comparison/SEALED/open.jsonl"
) -Phase "stage2")) {
    throw "stage2 forbidden-path self-test is not case-insensitive"
}
if (Test-ForbiddenFreezeArtifactPath -Path (
    "artifacts/steering_comparison/validation_open/open_scored_all.jsonl"
) -Phase "stage2") {
    throw "stage2 forbidden-path self-test rejected validation evidence"
}
if ((Get-MaximumResearchBlobBytes) -ne [long](95MB)) {
    throw "staged-blob ceiling self-test failed"
}
$universalVolatile = @(Get-UniversalFreezeVolatilePaths)
if (
    $universalVolatile.Count -ne 6 -or
    @($universalVolatile | Select-Object -Unique).Count -ne 6 -or
    -not (Test-UniversalFreezeVolatilePath -Path (
        "artifacts/steering_comparison/PREOPEN_FREEZE.LOG"
    ))
) {
    throw "universal freeze volatile-path registry self-test failed"
}
$cacheRejected = $false
try {
    Assert-StagedResearchFiles -RepositoryRoot $RepositoryRoot -Paths @(
        "artifacts/steering_comparison/jspace/atoms/example/atoms.pt"
    )
}
catch {
    $cacheRejected = $true
}
if (-not $cacheRejected) {
    throw "staged-cache self-test accepted a .pt artifact"
}

$temporaryBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$temporaryRepository = [IO.Path]::GetFullPath(
    (Join-Path $temporaryBase ("sp_lense_freeze_safety_" + [guid]::NewGuid().ToString("N")))
)
$temporaryRemote = $temporaryRepository + "_remote.git"
if (-not $temporaryRepository.StartsWith($temporaryBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "temporary Git fixture resolved outside the system temporary directory"
}
function Invoke-TestGit {
    param([string[]]$Arguments)
    & git -C $temporaryRepository @Arguments | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "temporary Git command failed: git $($Arguments -join ' ')"
    }
}

try {
    New-Item -ItemType Directory -Path $temporaryRepository | Out-Null
    Invoke-TestGit -Arguments @("init", "-q")
    Invoke-TestGit -Arguments @("config", "user.name", "SP Lense Freeze Test")
    Invoke-TestGit -Arguments @("config", "user.email", "freeze-test@example.invalid")
    Invoke-TestGit -Arguments @("config", "core.autocrlf", "false")
    "root`n" | Set-Content -LiteralPath (Join-Path $temporaryRepository "README.md") -Encoding utf8
    Invoke-TestGit -Arguments @("add", "README.md")
    Invoke-TestGit -Arguments @("commit", "-q", "-m", "fixture root")

    $transportFixture = Join-Path $temporaryRepository "transport_fixture"
    $transportRequests = Join-Path $transportFixture "requests.jsonl"
    $transportResponses = Join-Path $transportFixture "responses.jsonl"
    $transportWork = Join-Path $transportFixture "work"
    $missingRequiredTransportRejected = $false
    try {
        $null = Assert-LockedJudgeTransportPresence -Required $true `
            -RequestsPath $transportRequests -ResponsesPath $transportResponses `
            -WorkDirectory $transportWork -Label "self-test"
    }
    catch {
        $missingRequiredTransportRejected = $true
    }
    if (-not $missingRequiredTransportRejected) {
        throw "required missing judge transport was accepted"
    }
    New-Item -ItemType Directory -Path $transportWork -Force | Out-Null
    "{}`n" | Set-Content -LiteralPath $transportRequests -Encoding utf8
    "{}`n" | Set-Content -LiteralPath $transportResponses -Encoding utf8
    if (-not (Assert-LockedJudgeTransportPresence -Required $true `
        -RequestsPath $transportRequests -ResponsesPath $transportResponses `
        -WorkDirectory $transportWork -Label "self-test")) {
        throw "complete required judge transport was not recognized"
    }
    $unexpectedTransportRejected = $false
    try {
        $null = Assert-LockedJudgeTransportPresence -Required $false `
            -RequestsPath $transportRequests -ResponsesPath $transportResponses `
            -WorkDirectory $transportWork -Label "self-test"
    }
    catch {
        $unexpectedTransportRejected = $true
    }
    if (-not $unexpectedTransportRejected) {
        throw "judge transport was accepted when the locked plan requires none"
    }
    $validationPlanFixture = Join-Path $temporaryRepository "validation_plan.json"
    [ordered]@{
        schema_version = "sp_lense.locked_open_plan.v1"
        split = "validation"
        setup_count = 1
        setups = @([ordered]@{ open_required = $true })
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $validationPlanFixture -Encoding utf8
    $validationRequirement = Get-LockedOpenTransportRequirement `
        -PlanPath $validationPlanFixture -ExpectedSplit "validation" `
        -RequireEverySetupOpen
    if (-not $validationRequirement.required -or $validationRequirement.open_setup_count -ne 1) {
        throw "locked validation plan did not require its open judge transport"
    }
    $sealedPlanFixture = Join-Path $temporaryRepository "sealed_plan.json"
    [ordered]@{
        schema_version = "sp_lense.locked_open_plan.v1"
        split = "sealed_test"
        setup_count = 1
        setups = @([ordered]@{ open_required = $false })
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $sealedPlanFixture -Encoding utf8
    $sealedRequirement = Get-LockedOpenTransportRequirement `
        -PlanPath $sealedPlanFixture -ExpectedSplit "sealed_test"
    if ($sealedRequirement.required -or $sealedRequirement.open_setup_count -ne 0) {
        throw "locked sealed plan incorrectly required a random-only judge transport"
    }
    $planRequirednessDerived = $true

    $reportFixture = Join-Path $temporaryRepository "report_fixture"
    New-Item -ItemType Directory -Path $reportFixture -Force | Out-Null
    $expectedJson = Join-Path $reportFixture "final_report.json"
    $expectedMarkdown = Join-Path $reportFixture "FINAL_REPORT.md"
    $statusSentinel = Join-Path $reportFixture "report_status.json"
    "{`"status`":`"canonical`"}`n" | Set-Content -LiteralPath $expectedJson -Encoding utf8
    "# Canonical report`n" | Set-Content -LiteralPath $expectedMarkdown -Encoding utf8
    "{`"state`":`"must_not_change`"}`n" | Set-Content -LiteralPath $statusSentinel -Encoding utf8
    $statusHashBefore = (Get-FileHash -LiteralPath $statusSentinel -Algorithm SHA256).Hash
    $fakeBuilder = Join-Path $temporaryRepository "fake_report_builder.ps1"
    @'
param(
    [string]$RepositoryRoot,
    [string]$OutputDirectory,
    [switch]$NoStatusLog
)
$ErrorActionPreference = "Stop"
if (-not $NoStatusLog) {
    throw "isolated rebuild did not pass -NoStatusLog"
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "report_fixture\final_report.json") `
    -Destination (Join-Path $OutputDirectory "final_report.json")
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "report_fixture\FINAL_REPORT.md") `
    -Destination (Join-Path $OutputDirectory "FINAL_REPORT.md")
'@ | Set-Content -LiteralPath $fakeBuilder -Encoding utf8
    $rebuildReceipt = Invoke-IsolatedFinalReportRebuild `
        -RepositoryRoot $temporaryRepository -BuilderPath $fakeBuilder `
        -ExpectedJsonPath $expectedJson -ExpectedMarkdownPath $expectedMarkdown
    $statusHashAfter = (Get-FileHash -LiteralPath $statusSentinel -Algorithm SHA256).Hash
    if (
        $statusHashBefore -ne $statusHashAfter -or
        (Test-Path -LiteralPath $rebuildReceipt.output_directory)
    ) {
        throw "isolated final-report rebuild mutated status or retained its temporary directory"
    }
    $isolatedReportRebuildNonMutating = $true

    "artifacts/steering_comparison/**/*.pt`n" | Set-Content -LiteralPath (
        Join-Path $temporaryRepository ".gitignore"
    ) -Encoding utf8
    $baselineRelative = "artifacts/steering_comparison/baseline.json"
    New-Item -ItemType Directory -Path (
        Split-Path -Parent (Join-Path $temporaryRepository $baselineRelative)
    ) -Force | Out-Null
    "{`"baseline`":true}`n" | Set-Content -LiteralPath (
        Join-Path $temporaryRepository $baselineRelative
    ) -Encoding utf8
    Invoke-TestGit -Arguments @("add", "--", ".")
    Invoke-TestGit -Arguments @("commit", "-q", "-m", "fixture helper inputs")

    $artifactDirectory = Join-Path $temporaryRepository "artifacts\steering_comparison\model"
    New-Item -ItemType Directory -Path $artifactDirectory -Force | Out-Null
    $directionPath = Join-Path $artifactDirectory "direction.json"
    $directionRelative = "artifacts/steering_comparison/model/direction.json"
    $inventoryRelative = "artifacts/steering_comparison/preopen_artifact_inventory.json"
    "{}`n" | Set-Content -LiteralPath $directionPath -Encoding utf8
    "cache`n" | Set-Content -LiteralPath (Join-Path $artifactDirectory "atoms.pt") -Encoding utf8
    $artifactBaseCommit = (& git -C $temporaryRepository rev-parse HEAD).Trim()

    foreach ($knownTemporaryRelative in @(
        "configs/steering_comparison_preopen_lock.rebuild.tmp",
        "configs/steering_comparison_stage2_lock.rebuild.tmp"
    )) {
        $knownTemporary = Join-Path $temporaryRepository $knownTemporaryRelative
        New-Item -ItemType Directory -Path (Split-Path -Parent $knownTemporary) -Force |
            Out-Null
        "partial`n" | Set-Content -LiteralPath $knownTemporary -Encoding utf8
        if (-not (Remove-KnownFreezeRebuildTemporary -RepositoryRoot $temporaryRepository `
            -RelativePath $knownTemporaryRelative)) {
            throw "known rebuild-temporary self-test did not classify $knownTemporaryRelative"
        }
        if (Test-Path -LiteralPath $knownTemporary) {
            throw "known rebuild-temporary self-test retained $knownTemporaryRelative"
        }
    }
    $knownTemporaryRecovery = $true

    $null = Assert-OnlyExpectedArtifactFiles -RepositoryRoot $temporaryRepository `
        -ArtifactRoot (Join-Path $temporaryRepository "artifacts\steering_comparison") `
        -BaseCommit $artifactBaseCommit `
        -ExpectedPhasePaths @($baselineRelative, $directionRelative) `
        -InventoryPath $inventoryRelative
    $expectedDuplicateRejected = $false
    try {
        $null = Assert-OnlyExpectedArtifactFiles -RepositoryRoot $temporaryRepository `
            -ArtifactRoot (Join-Path $temporaryRepository "artifacts\steering_comparison") `
            -BaseCommit $artifactBaseCommit `
            -ExpectedPhasePaths @(
                $baselineRelative, $directionRelative, $directionRelative.ToUpperInvariant()
            ) -InventoryPath $inventoryRelative
    }
    catch {
        $expectedDuplicateRejected = $true
    }
    if (-not $expectedDuplicateRejected) {
        throw "expected-phase allowlist accepted an internal case collision"
    }
    $unexpectedArtifact = Join-Path $artifactDirectory "credentials.env"
    "do-not-stage`n" | Set-Content -LiteralPath $unexpectedArtifact -Encoding utf8
    $unexpectedArtifactRejected = $false
    try {
        $null = Assert-OnlyExpectedArtifactFiles -RepositoryRoot $temporaryRepository `
            -ArtifactRoot (Join-Path $temporaryRepository "artifacts\steering_comparison") `
            -BaseCommit $artifactBaseCommit `
            -ExpectedPhasePaths @($baselineRelative, $directionRelative) `
            -InventoryPath $inventoryRelative
    }
    catch {
        $unexpectedArtifactRejected = $true
    }
    Remove-Item -LiteralPath $unexpectedArtifact -Force
    if (-not $unexpectedArtifactRejected) {
        throw "exact artifact allowlist accepted an unexpected credential-like file"
    }

    $inventory = New-FreezeArtifactInventory -RepositoryRoot $temporaryRepository `
        -Phase "preopen" -BaseCommit $artifactBaseCommit `
        -ExpectedPhasePaths @($baselineRelative, $directionRelative) `
        -InventoryPath $inventoryRelative
    if (
        $baselineRelative -in @($inventory.entries.path) -or
        $directionRelative -notin @($inventory.entries.path)
    ) {
        throw "preopen overlap fixture did not inventory only the changed expected path"
    }
    Invoke-ExactArtifactStaging -RepositoryRoot $temporaryRepository `
        -Paths @($inventoryRelative)
    $null = Resume-ExactArtifactStaging -RepositoryRoot $temporaryRepository `
        -InventoryPath $inventoryRelative -Phase "preopen" -BaseCommit $artifactBaseCommit
    $stagedInventoryRecovery = $true
    $cachedPaths = @(& git -C $temporaryRepository diff --cached --name-only)
    if (
        $directionRelative -notin $cachedPaths -or
        "artifacts/steering_comparison/model/atoms.pt" -in $cachedPaths
    ) {
        throw "exact staging did not preserve cache exclusion: $cachedPaths"
    }

    $directionBytes = [IO.File]::ReadAllBytes($directionPath)
    Add-Content -LiteralPath $directionPath -Value "tamper"
    $stagedTamperRejected = $false
    try {
        $null = Assert-StagedArtifactInventory -RepositoryRoot $temporaryRepository `
            -InventoryPath $inventoryRelative -Phase "preopen" -BaseCommit $artifactBaseCommit
    }
    catch {
        $stagedTamperRejected = $true
    }
    [IO.File]::WriteAllBytes($directionPath, $directionBytes)
    if (-not $stagedTamperRejected) {
        throw "staged inventory accepted a working-file tamper"
    }

    $unexpectedStagedRelative = "artifacts/steering_comparison/model/unexpected.json"
    "{}`n" | Set-Content -LiteralPath (
        Join-Path $temporaryRepository $unexpectedStagedRelative
    ) -Encoding utf8
    Invoke-TestGit -Arguments @("add", "-f", "--", $unexpectedStagedRelative)
    $unexpectedStagedRejected = $false
    try {
        $null = Assert-StagedArtifactInventory -RepositoryRoot $temporaryRepository `
            -InventoryPath $inventoryRelative -Phase "preopen" -BaseCommit $artifactBaseCommit
    }
    catch {
        $unexpectedStagedRejected = $true
    }
    Invoke-TestGit -Arguments @("reset", "-q", "HEAD", "--", $unexpectedStagedRelative)
    Remove-Item -LiteralPath (Join-Path $temporaryRepository $unexpectedStagedRelative) -Force
    if (-not $unexpectedStagedRejected) {
        throw "staged inventory accepted an extra staged path"
    }
    $null = Assert-StagedArtifactInventory -RepositoryRoot $temporaryRepository `
        -InventoryPath $inventoryRelative -Phase "preopen" -BaseCommit $artifactBaseCommit

    Invoke-TestGit -Arguments @(
        "commit", "-q", "-m", "Freeze steering directions and forced validation artifacts"
    )
    $artifactCommit = (& git -C $temporaryRepository rev-parse HEAD).Trim()
    $null = Assert-ArtifactFreezeCommit -RepositoryRoot $temporaryRepository `
        -Commit $artifactCommit `
        -ExpectedSubject "Freeze steering directions and forced validation artifacts" `
        -Phase "preopen" -InventoryPath $inventoryRelative

    $stagedRecoveryPhases = @("preopen")
    foreach ($phase in @("stage2", "final")) {
        $phaseBase = (& git -C $temporaryRepository rev-parse HEAD).Trim()
        $phaseArtifactRelative = "artifacts/steering_comparison/${phase}_result.json"
        $phaseInventoryRelative = "artifacts/steering_comparison/${phase}_artifact_inventory.json"
        "{}`n" | Set-Content -LiteralPath (
            Join-Path $temporaryRepository $phaseArtifactRelative
        ) -Encoding utf8
        $null = Assert-OnlyExpectedArtifactFiles -RepositoryRoot $temporaryRepository `
            -ArtifactRoot (Join-Path $temporaryRepository "artifacts\steering_comparison") `
            -BaseCommit $phaseBase `
            -ExpectedPhasePaths @($baselineRelative, $phaseArtifactRelative) `
            -InventoryPath $phaseInventoryRelative
        $phaseInventory = New-FreezeArtifactInventory -RepositoryRoot $temporaryRepository `
            -Phase $phase -BaseCommit $phaseBase `
            -ExpectedPhasePaths @($baselineRelative, $phaseArtifactRelative) `
            -InventoryPath $phaseInventoryRelative
        if (
            $baselineRelative -in @($phaseInventory.entries.path) -or
            $phaseArtifactRelative -notin @($phaseInventory.entries.path)
        ) {
            throw "$phase overlap fixture did not inventory only the changed expected path"
        }
        Invoke-ExactArtifactStaging -RepositoryRoot $temporaryRepository `
            -Paths @($phaseInventoryRelative)
        $null = Resume-ExactArtifactStaging -RepositoryRoot $temporaryRepository `
            -InventoryPath $phaseInventoryRelative -Phase $phase -BaseCommit $phaseBase
        $stagedRecoveryPhases += $phase
        Invoke-TestGit -Arguments @(
            "reset", "-q", "HEAD", "--", $phaseArtifactRelative, $phaseInventoryRelative
        )
        Remove-Item -LiteralPath (
            Join-Path $temporaryRepository $phaseArtifactRelative
        ) -Force
        Remove-Item -LiteralPath (
            Join-Path $temporaryRepository $phaseInventoryRelative
        ) -Force
    }

    $configDirectory = Join-Path $temporaryRepository "configs"
    New-Item -ItemType Directory -Path $configDirectory -Force | Out-Null
    "{}`n" | Set-Content -LiteralPath (
        Join-Path $configDirectory "steering_comparison_preopen_lock.json"
    ) -Encoding utf8
    Invoke-TestGit -Arguments @("add", "configs/steering_comparison_preopen_lock.json")
    Invoke-TestGit -Arguments @("commit", "-q", "-m", "Lock validation open-response candidates")
    $manifestCommit = (& git -C $temporaryRepository rev-parse HEAD).Trim()
    Assert-SinglePathCommit -RepositoryRoot $temporaryRepository -Commit $manifestCommit `
        -ExpectedSubject "Lock validation open-response candidates" `
        -ExpectedPath "configs/steering_comparison_preopen_lock.json"

    "outside`n" | Set-Content -LiteralPath (Join-Path $temporaryRepository "outside.txt") -Encoding utf8
    Invoke-TestGit -Arguments @("add", "outside.txt")
    Invoke-TestGit -Arguments @(
        "commit", "-q", "-m", "Freeze steering directions and forced validation artifacts"
    )
    $forgedCommit = (& git -C $temporaryRepository rev-parse HEAD).Trim()
    $forgedRejected = $false
    try {
        $null = Assert-ArtifactFreezeCommit -RepositoryRoot $temporaryRepository `
            -Commit $forgedCommit `
            -ExpectedSubject "Freeze steering directions and forced validation artifacts" `
            -Phase "preopen" -InventoryPath $inventoryRelative
    }
    catch {
        $forgedRejected = $true
    }
    if (-not $forgedRejected) {
        throw "forged artifact-freeze subject with an outside path was accepted"
    }

    & git init --bare -q $temporaryRemote
    if ($LASTEXITCODE -ne 0) {
        throw "could not initialize the temporary bare origin"
    }
    Invoke-TestGit -Arguments @("branch", "-M", "main")
    Invoke-TestGit -Arguments @("remote", "add", "origin", $temporaryRemote)
    Invoke-TestGit -Arguments @("push", "-q", "-u", "origin", "main")
    $remoteHead = (& git -C $temporaryRepository rev-parse HEAD).Trim()
    Assert-RemoteMainEqualsHead -RepositoryRoot $temporaryRepository -Head $remoteHead

    "more`n" | Set-Content -LiteralPath (Join-Path $artifactDirectory "more.json") -Encoding utf8
    Invoke-TestGit -Arguments @("add", "artifacts/steering_comparison/model/more.json")
    Invoke-TestGit -Arguments @("commit", "-q", "-m", "local recovery descendant")
    $localDescendant = (& git -C $temporaryRepository rev-parse HEAD).Trim()
    $null = Assert-RemoteMainIsSafeAncestor -RepositoryRoot $temporaryRepository `
        -Head $localDescendant
}
finally {
    if (
        (Test-Path -LiteralPath $temporaryRepository) -and
        $temporaryRepository.StartsWith($temporaryBase, [StringComparison]::OrdinalIgnoreCase)
    ) {
        Remove-Item -LiteralPath $temporaryRepository -Recurse -Force
    }
    if (
        (Test-Path -LiteralPath $temporaryRemote) -and
        $temporaryRemote.StartsWith($temporaryBase, [StringComparison]::OrdinalIgnoreCase)
    ) {
        Remove-Item -LiteralPath $temporaryRemote -Recurse -Force
    }
}

[ordered]@{
    status = "self_test_passed"
    preopen_forbidden_cases = $preopenForbidden.Count
    stage2_forbidden_gate = $true
    cache_extensions_excluded = @("pt", "pth", "bin", "safetensors")
    maximum_staged_blob_bytes = Get-MaximumResearchBlobBytes
    temporary_git_commit_validation = $true
    forged_subject_outside_path_rejected = $true
    staged_binary_cache_rejected = $cacheRejected
    exact_staging_cache_exclusion = $true
    remote_history_recovery_validation = $true
    plan_required_transport_missing_rejected = $missingRequiredTransportRejected
    plan_forbidden_transport_present_rejected = $unexpectedTransportRejected
    locked_plan_requiredness_derived = $planRequirednessDerived
    isolated_final_report_rebuild_non_mutating = $isolatedReportRebuildNonMutating
    universal_freeze_volatile_paths = $universalVolatile
    unexpected_artifact_rejected = $unexpectedArtifactRejected
    exact_staged_inventory_recovery = $stagedInventoryRecovery
    staged_worktree_tamper_rejected = $stagedTamperRejected
    extra_staged_path_rejected = $unexpectedStagedRejected
    known_rebuild_temporary_recovered = $knownTemporaryRecovery
    staged_uncommitted_phase_recovery = $stagedRecoveryPhases
    tracked_expected_overlap_phases = $stagedRecoveryPhases
    expected_internal_case_collision_rejected = $expectedDuplicateRejected
} | ConvertTo-Json
