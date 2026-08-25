Set-StrictMode -Version Latest

$script:MaximumResearchBlobBytes = [long](95MB)
$script:ExcludedResearchCachePattern = "\.(pt|pth|bin|safetensors)$"
$script:ArtifactInventorySchema = "sp_lense.freeze_artifact_inventory.v1"
$script:UniversalFreezeVolatilePaths = @(
    "artifacts/steering_comparison/preopen_freeze_status.json",
    "artifacts/steering_comparison/preopen_freeze.log",
    "artifacts/steering_comparison/stage2_freeze_status.json",
    "artifacts/steering_comparison/stage2_freeze.log",
    "artifacts/steering_comparison/final_freeze_status.json",
    "artifacts/steering_comparison/final_freeze.log"
)

function ConvertTo-NormalizedResearchPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return $Path.Replace("\", "/").TrimStart("./").ToLowerInvariant()
}

function Get-RepositoryRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $root = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    )
    $fullPath = [IO.Path]::GetFullPath($Path)
    $rootPrefix = $root + [IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "path is outside the repository root: $Path"
    }
    return $fullPath.Substring($rootPrefix.Length).Replace("\", "/")
}

function Get-UniversalFreezeVolatilePaths {
    return @($script:UniversalFreezeVolatilePaths)
}

function Get-ValidatedJspaceCompletion {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExecutable,
        [Parameter(Mandatory = $true)][string]$ValidatorPath,
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$PlanPath,
        [Parameter(Mandatory = $true)][string]$LockPath,
        [Parameter(Mandatory = $true)][string]$StatusPath,
        [Parameter(Mandatory = $true)][string]$RecordsDirectory,
        [Parameter(Mandatory = $true)][string]$AtomsRoot,
        [Parameter(Mandatory = $true)][string]$CompletionPath
    )

    foreach ($requiredPath in @(
        $PythonExecutable, $ValidatorPath, $PlanPath, $LockPath, $StatusPath
    )) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "J-space completion verification input is missing: $requiredPath"
        }
    }
    $arguments = @(
        $ValidatorPath,
        "--repo-root", $RepositoryRoot,
        "--plan", $PlanPath,
        "--lock", $LockPath,
        "--status", $StatusPath,
        "--records-dir", $RecordsDirectory,
        "--atoms-root", $AtomsRoot,
        "--output", $CompletionPath
    )
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& $PythonExecutable @arguments 2>&1)
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    $outputText = @($output | ForEach-Object { $_.ToString() }) -join "`n"
    if ($nativeExitCode -ne 0) {
        throw "J-space completion verification failed: $outputText"
    }
    try {
        $receipt = $outputText | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "J-space completion verifier returned invalid JSON: $outputText"
    }
    if (
        $receipt.schema_version -ne "sp_lense.jspace_completion_receipt.v1" -or
        $receipt.status -ne "valid_complete" -or
        [int]$receipt.record_count -ne @($receipt.record_paths).Count
    ) {
        throw "J-space completion verifier returned an invalid receipt"
    }
    if (-not (Test-Path -LiteralPath $CompletionPath -PathType Leaf)) {
        throw "J-space completion verifier did not publish its deterministic receipt"
    }
    $persisted = Get-Content -Raw -LiteralPath $CompletionPath | ConvertFrom-Json
    if (
        $persisted.schema_version -ne $receipt.schema_version -or
        $persisted.plan_sha256 -ne $receipt.plan_sha256 -or
        $persisted.artifact_paths_sha256 -ne $receipt.artifact_paths_sha256 -or
        @($persisted.artifacts).Count -ne @($receipt.artifacts).Count
    ) {
        throw "persisted J-space completion receipt differs from fresh verification"
    }
    return $receipt
}

function Test-UniversalFreezeVolatilePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $normalized = ConvertTo-NormalizedResearchPath -Path $Path
    return $normalized -in @(
        $script:UniversalFreezeVolatilePaths | ForEach-Object {
            ConvertTo-NormalizedResearchPath -Path $_
        }
    )
}

function Test-RebuildableOrVolatileArtifactPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $normalized = ConvertTo-NormalizedResearchPath -Path $Path
    return (
        (Test-UniversalFreezeVolatilePath -Path $normalized) -or
        $normalized -match $script:ExcludedResearchCachePattern -or
        $normalized -match "(^|/)__pycache__/|\.pyc$|(^|/)\.submission\.lock$"
    )
}

function Test-ForbiddenFreezeArtifactPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][ValidateSet("preopen", "stage2", "final")][string]$Phase
    )

    $normalized = ConvertTo-NormalizedResearchPath -Path $Path
    if ($normalized.EndsWith(".tmp")) {
        return $true
    }
    $prefix = "artifacts/steering_comparison/"
    if (-not $normalized.StartsWith($prefix)) {
        return $false
    }
    $relative = $normalized.Substring($prefix.Length)
    if (Test-UniversalFreezeVolatilePath -Path $normalized) {
        if ($Phase -eq "preopen") {
            return $relative -match "^(stage2|final)_freeze(_status\.json|\.log)$"
        }
        if ($Phase -eq "stage2") {
            return $relative -match "^final_freeze(_status\.json|\.log)$"
        }
        return $false
    }
    $alwaysTooLate = (
        $relative -match "^sealed/" -or
        $relative -match "^sealed_(evaluation|judgment)_(status\.json|.*\.log)$" -or
        $relative -match "^jspace/" -or
        $relative -in @("jspace_status.json", "jspace.log") -or
        $relative -in @(
            "final_report.json",
            "final_report.md",
            "adversarial_review.md",
            "report_status.json",
            "report.log"
        )
    )
    if ($alwaysTooLate) {
        return $true
    }
    if ($Phase -eq "preopen") {
        return (
            $relative -match "^validation_open/" -or
            $relative -in @(
                "validation_open_generation_status.json",
                "validation_open_generation.log",
                "validation_open_completion_status.json",
                "validation_open_completion.log",
                "final_summary_status.json",
                "final_summary.log"
            )
        )
    }
    return $false
}

function Assert-NoForbiddenFreezeArtifacts {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$ArtifactRoot,
        [Parameter(Mandatory = $true)][ValidateSet("preopen", "stage2", "final")][string]$Phase
    )

    $forbidden = @(
        Get-ChildItem -LiteralPath $ArtifactRoot -Recurse -File | ForEach-Object {
            $relative = Get-RepositoryRelativePath -RepositoryRoot $RepositoryRoot `
                -Path $_.FullName
            if (Test-ForbiddenFreezeArtifactPath -Path $relative -Phase $Phase) {
                $relative.Replace("\", "/")
            }
        }
    )
    if ($forbidden.Count -ne 0) {
        throw "$Phase freeze found forbidden future/temporary artifacts: $($forbidden -join ', ')"
    }
}

function Assert-StagedResearchFiles {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string[]]$Paths
    )

    $unsafe = @()
    foreach ($path in $Paths) {
        $normalized = ConvertTo-NormalizedResearchPath -Path $path
        if ($normalized -match $script:ExcludedResearchCachePattern) {
            $unsafe += "$path (rebuildable binary cache)"
            continue
        }
        $fullPath = Join-Path $RepositoryRoot $path
        if (
            (Test-Path -LiteralPath $fullPath -PathType Leaf) -and
            (Get-Item -LiteralPath $fullPath).Length -ge $script:MaximumResearchBlobBytes
        ) {
            $unsafe += "$path (at least 95 MiB)"
        }
    }
    if ($unsafe.Count -ne 0) {
        throw "unsafe research blobs are staged: $($unsafe -join ', ')"
    }
}

function ConvertTo-ArtifactRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $root = [IO.Path]::GetFullPath($RepositoryRoot)
    $fullPath = if ([IO.Path]::IsPathRooted($Path)) {
        [IO.Path]::GetFullPath($Path)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $root $Path))
    }
    try {
        $relative = Get-RepositoryRelativePath -RepositoryRoot $root -Path $fullPath
    }
    catch {
        throw "could not resolve artifact path: $Path"
    }
    if (-not $relative.StartsWith(
        "artifacts/steering_comparison/", [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "path is outside the steering-comparison artifact root: $Path"
    }
    return $relative
}

function Get-LockedJudgeTransportArtifactPaths {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$RequestsPath,
        [Parameter(Mandatory = $true)][string]$ResponsesPath,
        [Parameter(Mandatory = $true)][string]$WorkDirectory
    )

    foreach ($path in @($RequestsPath, $ResponsesPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "judge transport inventory input is missing: $path"
        }
    }
    if (-not (Test-Path -LiteralPath $WorkDirectory -PathType Container)) {
        throw "judge transport inventory directory is missing: $WorkDirectory"
    }
    $ids = @(
        Get-Content -LiteralPath $RequestsPath | ForEach-Object {
            if ([string]::IsNullOrWhiteSpace($_)) {
                throw "judge request inventory contains a blank JSONL row"
            }
            $row = $_ | ConvertFrom-Json
            $id = [string]$row.request_id
            if ($id -notmatch "^[0-9a-f]{64}$") {
                throw "judge request inventory contains an unsafe request ID"
            }
            $id
        }
    )
    if ($ids.Count -eq 0 -or @($ids | Select-Object -Unique).Count -ne $ids.Count) {
        throw "judge request inventory IDs are empty or duplicated"
    }
    $paths = @($RequestsPath, $ResponsesPath, (Join-Path $WorkDirectory "cost_preflight.json"))
    foreach ($id in $ids) {
        $paths += @(
            (Join-Path $WorkDirectory "response_shards\$id.json"),
            (Join-Path $WorkDirectory "api_receipts\$id.json"),
            (Join-Path $WorkDirectory "submission_attempts\$id.json")
        )
    }
    foreach ($path in $paths) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "judge transport exact artifact is missing: $path"
        }
    }
    return @(
        $paths | ForEach-Object {
            ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $_
        } | Sort-Object -Unique
    )
}

function Get-DirectionManifestArtifactPaths {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string[]]$ManifestPaths
    )

    $paths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($manifestPath in $ManifestPaths) {
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
            throw "direction inventory manifest is missing: $manifestPath"
        }
        $null = $paths.Add((ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot `
            -Path $manifestPath))
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        $directions = @($manifest.directions)
        if ($directions.Count -eq 0) {
            throw "direction inventory manifest has no directions: $manifestPath"
        }
        foreach ($direction in $directions) {
            foreach ($field in @("path", "construction_config_path")) {
                $relative = ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot `
                    -Path ([string]$direction.$field)
                $fullPath = Join-Path $RepositoryRoot $relative
                if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
                    throw "direction inventory referenced file is missing: $relative"
                }
                $null = $paths.Add($relative)
            }
            $constructionPath = Join-Path $RepositoryRoot ([string]$direction.construction_config_path)
            $construction = Get-Content -Raw -LiteralPath $constructionPath | ConvertFrom-Json
            foreach ($evidence in @($construction.evidence_artifacts)) {
                $evidenceRelative = ConvertTo-ArtifactRelativePath `
                    -RepositoryRoot $RepositoryRoot -Path ([string]$evidence.path)
                $evidencePath = Join-Path $RepositoryRoot $evidenceRelative
                if (-not (Test-Path -LiteralPath $evidencePath -PathType Leaf)) {
                    throw "direction evidence artifact is missing: $evidenceRelative"
                }
                if (
                    [string]$evidence.sha256 -notmatch "^[0-9a-f]{64}$" -or
                    (Get-FileHash -LiteralPath $evidencePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
                        [string]$evidence.sha256
                ) {
                    throw "direction evidence artifact hash mismatch: $evidenceRelative"
                }
                $null = $paths.Add($evidenceRelative)
            }
        }
    }
    return @($paths | Sort-Object)
}

function Get-ForcedGridArtifactPaths {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$PlanPath,
        [Parameter(Mandatory = $true)][int]$ExpectedPointCount
    )

    if (-not (Test-Path -LiteralPath $PlanPath -PathType Leaf)) {
        throw "forced-grid inventory plan is missing: $PlanPath"
    }
    $plan = Get-Content -Raw -LiteralPath $PlanPath | ConvertFrom-Json
    $points = @($plan.points)
    if ($points.Count -ne $ExpectedPointCount) {
        throw "forced-grid inventory count differs from $ExpectedPointCount"
    }
    $pointRoot = Join-Path (Split-Path -Parent $PlanPath) "points"
    $paths = @($PlanPath)
    foreach ($point in $points) {
        $name = [string]$point.shard_name
        if (
            [string]::IsNullOrWhiteSpace($name) -or
            [IO.Path]::GetFileName($name) -ne $name -or
            -not $name.EndsWith(".json", [StringComparison]::OrdinalIgnoreCase)
        ) {
            throw "forced-grid inventory contains an unsafe shard name: $name"
        }
        $path = Join-Path $pointRoot $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "forced-grid inventory shard is missing: $path"
        }
        $paths += $path
    }
    return @(
        $paths | ForEach-Object {
            ConvertTo-ArtifactRelativePath -RepositoryRoot $RepositoryRoot -Path $_
        } | Sort-Object -Unique
    )
}

function Assert-LockedJudgeTransportPresence {
    param(
        [Parameter(Mandatory = $true)][bool]$Required,
        [Parameter(Mandatory = $true)][string]$RequestsPath,
        [Parameter(Mandatory = $true)][string]$ResponsesPath,
        [Parameter(Mandatory = $true)][string]$WorkDirectory,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $requestsExist = Test-Path -LiteralPath $RequestsPath -PathType Leaf
    $responsesExist = Test-Path -LiteralPath $ResponsesPath -PathType Leaf
    $workDirectoryExists = Test-Path -LiteralPath $WorkDirectory -PathType Container
    $observed = @($requestsExist, $responsesExist, $workDirectoryExists)
    if ($Required -and $false -in $observed) {
        throw (
            "$Label judge transport is required by its locked plan but is incomplete: " +
            "requests=$requestsExist;responses=$responsesExist;work_directory=$workDirectoryExists"
        )
    }
    if (-not $Required -and $true -in $observed) {
        throw (
            "$Label judge transport exists even though its locked plan requires no open " +
            "judgments: requests=$requestsExist;responses=$responsesExist;" +
            "work_directory=$workDirectoryExists"
        )
    }
    return $Required
}

function Get-LockedOpenTransportRequirement {
    param(
        [Parameter(Mandatory = $true)][string]$PlanPath,
        [Parameter(Mandatory = $true)][ValidateSet("validation", "sealed_test")][string]$ExpectedSplit,
        [switch]$RequireEverySetupOpen
    )

    if (-not (Test-Path -LiteralPath $PlanPath -PathType Leaf)) {
        throw "locked $ExpectedSplit open plan is missing: $PlanPath"
    }
    $plan = Get-Content -Raw -LiteralPath $PlanPath | ConvertFrom-Json
    $setups = @($plan.setups)
    if (
        $plan.schema_version -ne "sp_lense.locked_open_plan.v1" -or
        $plan.split -ne $ExpectedSplit -or
        [int]$plan.setup_count -ne $setups.Count
    ) {
        throw "locked $ExpectedSplit open plan is inconsistent"
    }
    $invalidFlags = @(
        $setups | Where-Object { $_.open_required -isnot [bool] }
    )
    if ($invalidFlags.Count -ne 0) {
        throw "locked $ExpectedSplit open plan has a non-Boolean open_required flag"
    }
    $openCount = @($setups | Where-Object { $_.open_required -eq $true }).Count
    if ($RequireEverySetupOpen -and $openCount -ne $setups.Count) {
        throw "$ExpectedSplit plan contains a setup that is not open-required"
    }
    return [pscustomobject]@{
        split = $ExpectedSplit
        setup_count = $setups.Count
        open_setup_count = $openCount
        required = $openCount -gt 0
    }
}

function Assert-LockedOpenPlanCanonical {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExecutable,
        [Parameter(Mandatory = $true)][string]$OrchestratorPath,
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$LockPath,
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$OutputDirectory,
        [Parameter(Mandatory = $true)][ValidateSet("validation", "sealed_test")][string]$Split,
        [Parameter(Mandatory = $true)][string]$PlanPath
    )

    foreach ($path in @($PythonExecutable, $OrchestratorPath, $LockPath, $ManifestPath, $PlanPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "canonical open-plan verification input is missing: $path"
        }
    }
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExecutable $OrchestratorPath verify-plan `
            --repo-root $RepositoryRoot `
            --lock $LockPath `
            --manifest $ManifestPath `
            --output-dir $OutputDirectory `
            --split $Split `
            --plan $PlanPath
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    if ($nativeExitCode -ne 0) {
        throw "canonical $Split open-plan verification failed with exit code $nativeExitCode"
    }
    return $true
}

function Invoke-IsolatedFinalReportRebuild {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$BuilderPath,
        [Parameter(Mandatory = $true)][string]$ExpectedJsonPath,
        [Parameter(Mandatory = $true)][string]$ExpectedMarkdownPath
    )

    foreach ($requiredPath in @($BuilderPath, $ExpectedJsonPath, $ExpectedMarkdownPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "isolated final-report rebuild input is missing: $requiredPath"
        }
    }
    $temporaryBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $temporaryDirectory = [IO.Path]::GetFullPath(
        (Join-Path $temporaryBase ("sp_lense_final_report_rebuild_" + [guid]::NewGuid().ToString("N")))
    )
    if (-not $temporaryDirectory.StartsWith(
        $temporaryBase, [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "isolated final-report rebuild directory escaped the system temporary directory"
    }
    try {
        New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
        $priorErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $BuilderPath `
                -RepositoryRoot $RepositoryRoot `
                -OutputDirectory $temporaryDirectory `
                -NoStatusLog
            $nativeExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $priorErrorActionPreference
        }
        if ($nativeExitCode -ne 0) {
            throw "isolated final-report rebuild failed with exit code $nativeExitCode"
        }
        $rebuiltJson = Join-Path $temporaryDirectory "final_report.json"
        $rebuiltMarkdown = Join-Path $temporaryDirectory "FINAL_REPORT.md"
        foreach ($pair in @(
            [ordered]@{ expected = $ExpectedJsonPath; rebuilt = $rebuiltJson },
            [ordered]@{ expected = $ExpectedMarkdownPath; rebuilt = $rebuiltMarkdown }
        )) {
            if (-not (Test-Path -LiteralPath $pair.rebuilt -PathType Leaf)) {
                throw "isolated final-report rebuild did not publish $($pair.rebuilt)"
            }
            if (
                (Get-FileHash -LiteralPath $pair.expected -Algorithm SHA256).Hash -ne
                (Get-FileHash -LiteralPath $pair.rebuilt -Algorithm SHA256).Hash
            ) {
                throw "final report differs from its isolated canonical rebuild: $($pair.expected)"
            }
        }
        return [pscustomobject]@{
            output_directory = $temporaryDirectory
            json_sha256 = (
                Get-FileHash -LiteralPath $rebuiltJson -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            markdown_sha256 = (
                Get-FileHash -LiteralPath $rebuiltMarkdown -Algorithm SHA256
            ).Hash.ToLowerInvariant()
        }
    }
    finally {
        if (
            (Test-Path -LiteralPath $temporaryDirectory) -and
            $temporaryDirectory.StartsWith(
                $temporaryBase, [StringComparison]::OrdinalIgnoreCase
            )
        ) {
            Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force
        }
    }
}

function Get-MaximumResearchBlobBytes {
    return $script:MaximumResearchBlobBytes
}

function Invoke-FreezeGitCapture {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $output = @(& git -C $RepositoryRoot @Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
    return @($output | ForEach-Object { [string]$_ })
}

function Get-TrackedArtifactPathsAtCommit {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Commit
    )

    return @(
        Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "ls-tree", "-r", "--name-only", $Commit, "--", "artifacts/steering_comparison"
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}

function Assert-OnlyExpectedArtifactFiles {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$ArtifactRoot,
        [Parameter(Mandatory = $true)][string]$BaseCommit,
        [Parameter(Mandatory = $true)][string[]]$ExpectedPhasePaths,
        [Parameter(Mandatory = $true)][string]$InventoryPath
    )

    $allowed = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    $declaredSets = @(
        [ordered]@{
            label = "base-commit artifact paths"
            paths = @(Get-TrackedArtifactPathsAtCommit `
                -RepositoryRoot $RepositoryRoot -Commit $BaseCommit)
        },
        [ordered]@{
            label = "expected phase paths"
            paths = @($ExpectedPhasePaths)
        },
        [ordered]@{
            label = "inventory path"
            paths = @($InventoryPath)
        }
    )
    foreach ($declaredSet in $declaredSets) {
        $withinSet = [Collections.Generic.HashSet[string]]::new(
            [StringComparer]::OrdinalIgnoreCase
        )
        foreach ($path in @($declaredSet.paths)) {
            $normalized = $path.Replace("\", "/").TrimStart("./")
            if (-not $normalized.StartsWith(
                "artifacts/steering_comparison/", [StringComparison]::OrdinalIgnoreCase
            )) {
                throw "artifact allowlist contains a non-artifact path: $path"
            }
            if (Test-RebuildableOrVolatileArtifactPath -Path $normalized) {
                throw "artifact allowlist contains a volatile/cache path: $path"
            }
            if (-not $withinSet.Add($normalized)) {
                throw "$($declaredSet.label) contains a duplicate or case-colliding path: $path"
            }
            # An expected phase path may intentionally already exist at BaseCommit.  Cross-set
            # overlaps are a normal union operation; only duplicates within one declaration are
            # malformed.
            $null = $allowed.Add($normalized)
        }
    }
    foreach ($expected in $ExpectedPhasePaths) {
        $fullPath = Join-Path $RepositoryRoot $expected
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            throw "expected phase artifact is missing: $expected"
        }
    }

    $unexpected = @(
        Get-ChildItem -LiteralPath $ArtifactRoot -Recurse -File | ForEach-Object {
            $relative = Get-RepositoryRelativePath -RepositoryRoot $RepositoryRoot `
                -Path $_.FullName
            if (
                -not (Test-RebuildableOrVolatileArtifactPath -Path $relative) -and
                -not $allowed.Contains($relative)
            ) {
                $relative
            }
        }
    )
    if ($unexpected.Count -ne 0) {
        throw "artifact tree contains files outside the exact phase allowlist: $($unexpected -join ', ')"
    }
    return @($allowed | Sort-Object)
}

function Get-FreezePathDigest {
    param([Parameter(Mandatory = $true)][string[]]$Paths)

    $joined = (@($Paths) -join "`n") + "`n"
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($joined)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace(
            "-", ""
        ).ToLowerInvariant()
    }
    finally {
        $hasher.Dispose()
    }
}

function Get-ChangedArtifactPathsFromCommit {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$BaseCommit,
        [Parameter(Mandatory = $true)][string[]]$ExpectedPhasePaths
    )

    $changed = @()
    foreach ($path in @($ExpectedPhasePaths | Sort-Object -Unique)) {
        $fullPath = Join-Path $RepositoryRoot $path
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            throw "cannot inventory missing artifact: $path"
        }
        $priorErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $baseObject = (& git -C $RepositoryRoot rev-parse "$BaseCommit`:$path" 2>$null)
            $baseExit = $LASTEXITCODE
            $workingObject = (& git -C $RepositoryRoot hash-object -- $path 2>$null)
            $workingExit = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $priorErrorActionPreference
        }
        $baseObjectLines = @($baseObject)
        $workingObjectLines = @($workingObject)
        if ($workingExit -ne 0 -or $workingObjectLines.Count -ne 1) {
            throw "could not hash expected artifact: $path"
        }
        if (
            $baseExit -ne 0 -or
            $baseObjectLines.Count -ne 1 -or
            $baseObjectLines[0].Trim() -ne $workingObjectLines[0].Trim()
        ) {
            $changed += $path.Replace("\", "/")
        }
    }
    return @($changed | Sort-Object -Unique)
}

function New-FreezeArtifactInventory {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][ValidateSet("preopen", "stage2", "final")][string]$Phase,
        [Parameter(Mandatory = $true)][string]$BaseCommit,
        [Parameter(Mandatory = $true)][string[]]$ExpectedPhasePaths,
        [Parameter(Mandatory = $true)][string]$InventoryPath
    )

    $normalizedInventory = $InventoryPath.Replace("\", "/").TrimStart("./")
    if (Test-RebuildableOrVolatileArtifactPath -Path $normalizedInventory) {
        throw "freeze inventory path cannot be volatile or rebuildable: $InventoryPath"
    }
    $paths = @(Get-ChangedArtifactPathsFromCommit -RepositoryRoot $RepositoryRoot `
        -BaseCommit $BaseCommit -ExpectedPhasePaths $ExpectedPhasePaths)
    if ($paths.Count -eq 0) {
        throw "$Phase artifact inventory has no changed paths"
    }
    $entries = @(
        foreach ($path in $paths) {
            $fullPath = Join-Path $RepositoryRoot $path
            [ordered]@{
                path = $path
                sha256 = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
                size_bytes = [long](Get-Item -LiteralPath $fullPath).Length
            }
        }
    )
    $payload = [ordered]@{
        schema_version = $script:ArtifactInventorySchema
        phase = $Phase
        base_commit = $BaseCommit
        path_count = $paths.Count
        paths_sha256 = Get-FreezePathDigest -Paths $paths
        entries = $entries
    }
    $fullInventory = Join-Path $RepositoryRoot $normalizedInventory
    $temporary = "$fullInventory.tmp"
    New-Item -ItemType Directory -Path (Split-Path -Parent $fullInventory) -Force | Out-Null
    try {
        $json = ($payload | ConvertTo-Json -Depth 10) + "`n"
        [IO.File]::WriteAllText($temporary, $json, [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporary -Destination $fullInventory -Force
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
    return $payload
}

function Invoke-ExactArtifactStaging {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string[]]$Paths
    )

    $unique = @($Paths | Sort-Object -Unique)
    if ($unique.Count -eq 0) {
        throw "exact artifact staging received no paths"
    }
    for ($offset = 0; $offset -lt $unique.Count; $offset += 48) {
        $last = [Math]::Min($offset + 47, $unique.Count - 1)
        $chunk = @($unique[$offset..$last])
        & git -C $RepositoryRoot add -f -- @chunk
        if ($LASTEXITCODE -ne 0) {
            throw "exact git add failed for artifact path chunk beginning $($chunk[0])"
        }
    }
}

function Resume-ExactArtifactStaging {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$InventoryPath,
        [Parameter(Mandatory = $true)][ValidateSet("preopen", "stage2", "final")][string]$Phase,
        [Parameter(Mandatory = $true)][string]$BaseCommit
    )

    $validated = Read-FreezeArtifactInventory -RepositoryRoot $RepositoryRoot `
        -InventoryPath $InventoryPath -Phase $Phase -BaseCommit $BaseCommit
    $expected = @((@($validated.paths) + @($InventoryPath)) | Sort-Object)
    $staged = @(
        Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "diff", "--cached", "--name-only"
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object
    )
    $unexpectedStaged = @($staged | Where-Object { $_ -notin $expected })
    if ($unexpectedStaged.Count -ne 0) {
        throw "$Phase partial staging contains paths outside the exact inventory: $unexpectedStaged"
    }

    $statusLines = @(
        Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "status", "--porcelain", "--untracked-files=all"
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $unrelatedStatus = @(
        $statusLines | Where-Object {
            $_.Length -lt 4 -or $_.Substring(3).Replace("\", "/") -notin $expected
        }
    )
    if ($unrelatedStatus.Count -ne 0) {
        throw "$Phase partial staging has unrelated status entries: $unrelatedStatus"
    }

    foreach ($entry in $validated.entries) {
        $path = [string]$entry.path
        $fullPath = Join-Path $RepositoryRoot $path
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            throw "$Phase partial staging is missing an inventoried working file: $path"
        }
        if (
            (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
                [string]$entry.sha256 -or
            [long](Get-Item -LiteralPath $fullPath).Length -ne [long]$entry.size_bytes
        ) {
            throw "$Phase partial staging working file differs from inventory metadata: $path"
        }
    }
    foreach ($path in $staged) {
        $indexObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "rev-parse", ":$path"
        )) -join ""
        $workingObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "hash-object", "--", $path
        )) -join ""
        if ($indexObject -ne $workingObject) {
            throw "$Phase partial staging index differs from its working file: $path"
        }
    }

    $missing = @($expected | Where-Object { $_ -notin $staged })
    if ($missing.Count -gt 0) {
        Invoke-ExactArtifactStaging -RepositoryRoot $RepositoryRoot -Paths $missing
    }
    return Assert-StagedArtifactInventory -RepositoryRoot $RepositoryRoot `
        -InventoryPath $InventoryPath -Phase $Phase -BaseCommit $BaseCommit
}

function Read-FreezeArtifactInventory {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$InventoryPath,
        [Parameter(Mandatory = $true)][ValidateSet("preopen", "stage2", "final")][string]$Phase,
        [Parameter(Mandatory = $true)][string]$BaseCommit
    )

    $fullPath = Join-Path $RepositoryRoot $InventoryPath
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        throw "$Phase artifact inventory is missing: $InventoryPath"
    }
    $inventory = Get-Content -Raw -LiteralPath $fullPath | ConvertFrom-Json
    $topFields = @($inventory.PSObject.Properties.Name | Sort-Object)
    $expectedTopFields = @(
        "base_commit", "entries", "path_count", "paths_sha256", "phase", "schema_version"
    ) | Sort-Object
    if (($topFields -join "|") -ne ($expectedTopFields -join "|")) {
        throw "$Phase artifact inventory has an unexpected top-level schema"
    }
    if (
        $inventory.schema_version -ne $script:ArtifactInventorySchema -or
        $inventory.phase -ne $Phase -or
        $inventory.base_commit -ne $BaseCommit
    ) {
        throw "$Phase artifact inventory identity differs from the current freeze"
    }
    $entries = @($inventory.entries)
    if ([int]$inventory.path_count -ne $entries.Count -or $entries.Count -eq 0) {
        throw "$Phase artifact inventory path count is invalid"
    }
    $paths = @()
    foreach ($entry in $entries) {
        $fields = @($entry.PSObject.Properties.Name | Sort-Object)
        if (($fields -join "|") -ne "path|sha256|size_bytes") {
            throw "$Phase artifact inventory entry has an unexpected schema"
        }
        $path = [string]$entry.path
        $normalized = $path.Replace("\", "/").TrimStart("./")
        if (
            $path -ne $normalized -or
            -not $normalized.StartsWith(
                "artifacts/steering_comparison/", [StringComparison]::OrdinalIgnoreCase
            ) -or
            (Test-RebuildableOrVolatileArtifactPath -Path $normalized)
        ) {
            throw "$Phase artifact inventory contains an unsafe path: $path"
        }
        if (
            [string]$entry.sha256 -notmatch "^[0-9a-f]{64}$" -or
            [long]$entry.size_bytes -lt 0 -or
            [long]$entry.size_bytes -ge $script:MaximumResearchBlobBytes
        ) {
            throw "$Phase artifact inventory has invalid hash/size metadata for $path"
        }
        $paths += $path
    }
    $sortedUnique = @($paths | Sort-Object -Unique)
    if (
        ($paths -join "|") -ne ($sortedUnique -join "|") -or
        $inventory.paths_sha256 -ne (Get-FreezePathDigest -Paths $paths)
    ) {
        throw "$Phase artifact inventory paths are not unique, ordered, and hash-bound"
    }
    return [pscustomobject]@{ payload = $inventory; entries = $entries; paths = $paths }
}

function Assert-StagedArtifactInventory {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$InventoryPath,
        [Parameter(Mandatory = $true)][ValidateSet("preopen", "stage2", "final")][string]$Phase,
        [Parameter(Mandatory = $true)][string]$BaseCommit
    )

    $validated = Read-FreezeArtifactInventory -RepositoryRoot $RepositoryRoot `
        -InventoryPath $InventoryPath -Phase $Phase -BaseCommit $BaseCommit
    $staged = @(
        Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "diff", "--cached", "--name-only"
        ) | Sort-Object
    )
    $expectedStaged = @((@($validated.paths) + @($InventoryPath)) | Sort-Object)
    if (($staged -join "|") -ne ($expectedStaged -join "|")) {
        throw "$Phase staged paths differ from the exact artifact inventory: $staged"
    }
    $unstaged = @(
        Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "diff", "--name-only"
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($unstaged.Count -ne 0) {
        throw "$Phase staged recovery has unstaged tracked changes: $unstaged"
    }
    $statusLines = @(
        Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "status", "--porcelain", "--untracked-files=all"
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $unrelatedStatus = @(
        $statusLines | Where-Object {
            $_.Length -lt 4 -or
            $_.Substring(0, 2) -notin @("A ", "M ") -or
            $_.Substring(3).Replace("\", "/") -notin $expectedStaged
        }
    )
    if ($unrelatedStatus.Count -ne 0) {
        throw "$Phase staged recovery has unrelated status entries: $unrelatedStatus"
    }
    foreach ($entry in $validated.entries) {
        $path = [string]$entry.path
        $fullPath = Join-Path $RepositoryRoot $path
        $indexObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "rev-parse", ":$path"
        )) -join ""
        $workingObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "hash-object", "--", $path
        )) -join ""
        if ($indexObject -ne $workingObject) {
            throw "$Phase staged artifact differs from its working file: $path"
        }
        if (
            (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
                [string]$entry.sha256 -or
            [long](Get-Item -LiteralPath $fullPath).Length -ne [long]$entry.size_bytes
        ) {
            throw "$Phase staged artifact differs from inventory metadata: $path"
        }
    }
    $inventoryIndexObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "rev-parse", ":$InventoryPath"
    )) -join ""
    $inventoryWorkingObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "hash-object", "--", $InventoryPath
    )) -join ""
    if ($inventoryIndexObject -ne $inventoryWorkingObject) {
        throw "$Phase staged inventory differs from its working file"
    }
    Assert-StagedResearchFiles -RepositoryRoot $RepositoryRoot -Paths $staged
    return $validated
}

function Assert-FreezeCommitMatchesInventory {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$InventoryPath,
        [Parameter(Mandatory = $true)][ValidateSet("preopen", "stage2", "final")][string]$Phase
    )

    $parent = ((Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "rev-parse", "$Commit^"
    )) -join "").Trim()
    $validated = Read-FreezeArtifactInventory -RepositoryRoot $RepositoryRoot `
        -InventoryPath $InventoryPath -Phase $Phase -BaseCommit $parent
    $commitPaths = @(Get-FreezeCommitPaths -RepositoryRoot $RepositoryRoot -Commit $Commit | Sort-Object)
    $expectedPaths = @((@($validated.paths) + @($InventoryPath)) | Sort-Object)
    if (($commitPaths -join "|") -ne ($expectedPaths -join "|")) {
        throw "$Phase commit paths differ from its exact artifact inventory"
    }
    foreach ($entry in $validated.entries) {
        $path = [string]$entry.path
        $fullPath = Join-Path $RepositoryRoot $path
        $commitObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "rev-parse", "$Commit`:$path"
        )) -join ""
        $workingObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "hash-object", "--", $path
        )) -join ""
        if ($commitObject -ne $workingObject) {
            throw "$Phase committed artifact differs from the checked-out file: $path"
        }
        if (
            (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
                [string]$entry.sha256 -or
            [long](Get-Item -LiteralPath $fullPath).Length -ne [long]$entry.size_bytes
        ) {
            throw "$Phase committed artifact differs from inventory metadata: $path"
        }
    }
    $inventoryCommitObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "rev-parse", "$Commit`:$InventoryPath"
    )) -join ""
    $inventoryWorkingObject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "hash-object", "--", $InventoryPath
    )) -join ""
    if ($inventoryCommitObject -ne $inventoryWorkingObject) {
        throw "$Phase committed inventory differs from the checked-out file"
    }
    return $commitPaths
}

function Remove-KnownFreezeRebuildTemporary {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][ValidateSet(
            "configs/steering_comparison_preopen_lock.rebuild.tmp",
            "configs/steering_comparison_stage2_lock.rebuild.tmp"
        )][string]$RelativePath
    )

    $root = [IO.Path]::GetFullPath($RepositoryRoot)
    $target = [IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    if (-not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "known freeze temporary escaped the repository root"
    }
    if (Test-Path -LiteralPath $target -PathType Container) {
        throw "known freeze temporary path is unexpectedly a directory: $target"
    }
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        Remove-Item -LiteralPath $target -Force
        return $true
    }
    return $false
}

function Get-FreezeCommitPaths {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Commit
    )

    return @(
        Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", $Commit
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}

function Test-GitPathAtCommit {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $output = @(Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "ls-tree", "--name-only", $Commit, "--", $Path
    ))
    return $output.Count -eq 1 -and $output[0] -eq $Path
}

function Assert-ArtifactFreezeCommit {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$ExpectedSubject,
        [Parameter(Mandatory = $true)][ValidateSet("preopen", "stage2", "final")][string]$Phase,
        [Parameter(Mandatory = $true)][string]$InventoryPath
    )

    $subject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "show", "-s", "--format=%s", $Commit
    )) -join "`n"
    if ($subject.Trim() -ne $ExpectedSubject) {
        throw "$Phase recovery commit subject is not the exact locked subject"
    }
    $paths = @(Assert-FreezeCommitMatchesInventory -RepositoryRoot $RepositoryRoot `
        -Commit $Commit -InventoryPath $InventoryPath -Phase $Phase)
    if ($paths.Count -eq 0) {
        throw "$Phase recovery commit has no changed paths"
    }
    foreach ($path in $paths) {
        $normalized = ConvertTo-NormalizedResearchPath -Path $path
        if (-not $normalized.StartsWith("artifacts/steering_comparison/")) {
            throw "$Phase recovery commit changed a non-artifact path: $path"
        }
        if (
            $normalized -match $script:ExcludedResearchCachePattern -or
            $normalized -match "(^|/)__pycache__/|\.pyc$|(^|/)\.submission\.lock$"
        ) {
            throw "$Phase recovery commit contains a forbidden cache/volatile path: $path"
        }
        if ($Phase -in @("preopen", "stage2") -and (
            Test-ForbiddenFreezeArtifactPath -Path $path -Phase $Phase
        )) {
            throw "$Phase recovery commit contains a future/temporary artifact: $path"
        }
        $sizeText = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
            "cat-file", "-s", "${Commit}:$path"
        )) -join ""
        $size = 0L
        if (-not [long]::TryParse($sizeText.Trim(), [ref]$size)) {
            throw "could not parse staged blob size for $path"
        }
        if ($size -ge $script:MaximumResearchBlobBytes) {
            throw "$Phase recovery commit contains a blob at least 95 MiB: $path"
        }
    }
    return $paths
}

function Assert-SinglePathCommit {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$ExpectedSubject,
        [Parameter(Mandatory = $true)][string]$ExpectedPath
    )

    $subject = (Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "show", "-s", "--format=%s", $Commit
    )) -join "`n"
    $paths = @(Get-FreezeCommitPaths -RepositoryRoot $RepositoryRoot -Commit $Commit)
    if ($subject.Trim() -ne $ExpectedSubject -or $paths.Count -ne 1 -or $paths[0] -ne $ExpectedPath) {
        throw "locked manifest commit must have the exact subject and only $ExpectedPath"
    }
}

function Get-RemoteMainCommit {
    param([Parameter(Mandatory = $true)][string]$RepositoryRoot)

    $lines = @(Invoke-FreezeGitCapture -RepositoryRoot $RepositoryRoot -Arguments @(
        "ls-remote", "--heads", "origin", "refs/heads/main"
    ))
    if ($lines.Count -ne 1 -or $lines[0] -notmatch "^([0-9a-f]{40})\s+refs/heads/main$") {
        throw "origin/main did not resolve to exactly one Git commit"
    }
    return $Matches[1]
}

function Assert-RemoteMainIsSafeAncestor {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Head,
        [switch]$RequireEqual
    )

    $remote = Get-RemoteMainCommit -RepositoryRoot $RepositoryRoot
    if ($RequireEqual -and $remote -ne $Head) {
        throw "origin/main differs from local HEAD before a new freeze"
    }
    if (-not $RequireEqual) {
        & git -C $RepositoryRoot merge-base --is-ancestor $remote $Head
        if ($LASTEXITCODE -ne 0) {
            throw "origin/main is not a known ancestor of the locked local history"
        }
    }
    return $remote
}

function Assert-RemoteMainEqualsHead {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Head
    )

    $remote = Get-RemoteMainCommit -RepositoryRoot $RepositoryRoot
    if ($remote -ne $Head) {
        throw "push returned but origin/main does not equal local HEAD"
    }
}
