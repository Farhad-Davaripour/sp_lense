param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [int]$AtomChunkSize = 1024,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$pythonExe = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
$orchestrator = Join-Path $artifactRoot "locked_open_orchestration.py"
. (Join-Path $artifactRoot "freeze_safety.ps1")
$cacheValidator = Join-Path $artifactRoot "validate_jspace_cache.py"
$recordValidator = Join-Path $artifactRoot "validate_jspace_record.py"
$completionValidator = Join-Path $artifactRoot "verify_jspace_completion.py"
$sealedPlanPath = Join-Path $artifactRoot "sealed\sealed_evaluation_plan.json"
$stage2Manifest = Join-Path $RepositoryRoot "configs\steering_comparison_stage2_lock.json"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$jspaceRoot = Join-Path $artifactRoot "jspace"
$statusPath = Join-Path $artifactRoot "jspace_status.json"
$completionPath = Join-Path $artifactRoot "jspace_completion.json"
$logPath = Join-Path $artifactRoot "jspace.log"
$primaryMethods = @("gradient", "caa", "bipo", "persona_vector")

function Write-JspaceStatus {
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

function Test-JspaceCache {
    param([Parameter(Mandatory = $true)][string]$ManifestPath)

    "[$([DateTime]::UtcNow.ToString('o'))] CHECK jspace_cache $ManifestPath" |
        Tee-Object -FilePath $logPath -Append
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $pythonExe $cacheValidator --manifest $ManifestPath 2>&1 |
            Tee-Object -FilePath $logPath -Append
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    return $nativeExitCode -eq 0
}

function Test-JspaceRecord {
    param(
        [Parameter(Mandatory = $true)][string]$RecordPath,
        [Parameter(Mandatory = $true)][string]$SetupId,
        [string]$AtomsManifest = ""
    )

    if (-not (Test-Path -LiteralPath $RecordPath -PathType Leaf)) {
        return $false
    }
    $arguments = @(
        $recordValidator,
        "--repo-root", $RepositoryRoot,
        "--plan", $sealedPlanPath,
        "--lock", $lockPath,
        "--setup-id", $SetupId,
        "--record", $RecordPath
    )
    if (-not [string]::IsNullOrWhiteSpace($AtomsManifest)) {
        $arguments += @("--atoms-manifest", $AtomsManifest)
    }
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $pythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    return $nativeExitCode -eq 0
}

if ($SelfTest) {
    if ($AtomChunkSize -ne 1024) {
        throw "self-test uses the documented non-scientific extraction chunk size"
    }
    [ordered]@{
        status = "self_test_passed"
        primary_methods = $primaryMethods
        diagnostic_ablation_included = $false
        random_directions_included = $false
        direction_signs = @("positive", "negative")
        k_values = @(8, 16, 25)
        cone_random_controls = 50
        non_gating = $true
        atom_cache_validated_before_skip = $true
        missing_or_invalid_atom_cache_rebuilt = $true
        rebuilt_cache_forces_record_reanalysis = $true
        skipped_records_require_full_setup_direction_layer_cache_validation = $true
        extra_record_rejection = $true
        canonical_plan_byte_comparison = $true
    } | ConvertTo-Json
    return
}

if ($AtomChunkSize -lt 1) {
    throw "-AtomChunkSize must be positive"
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SpLenseJspacePowerState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$keepAwakeFlags = [uint32]::Parse("80000001", [Globalization.NumberStyles]::HexNumber)
$resetExecutionFlags = [uint32]::Parse("80000000", [Globalization.NumberStyles]::HexNumber)
[void][SpLenseJspacePowerState]::SetThreadExecutionState($keepAwakeFlags)

try {
    Invoke-NativeLogged -Phase "verify_stage2_before_secondary_jspace" -Arguments @(
        "verify-stage2", "--stage2-manifest", $stage2Manifest
    )
    foreach ($required in @(
        $sealedPlanPath, $lockPath, $cacheValidator, $recordValidator,
        $completionValidator, $orchestrator
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "J-space prerequisite is missing: $required"
        }
    }
    $null = Assert-LockedOpenPlanCanonical -PythonExecutable $pythonExe `
        -OrchestratorPath $orchestrator -RepositoryRoot $RepositoryRoot `
        -LockPath $lockPath -ManifestPath $stage2Manifest `
        -OutputDirectory (Join-Path $artifactRoot "sealed") `
        -Split "sealed_test" -PlanPath $sealedPlanPath
    $lock = Get-Content -Raw -LiteralPath $lockPath | ConvertFrom-Json
    $plan = Get-Content -Raw -LiteralPath $sealedPlanPath | ConvertFrom-Json
    if ($plan.split -ne "sealed_test") {
        throw "J-space direction source must be the verified sealed plan"
    }
    $availableLayers = @{}
    foreach ($modelProperty in $lock.evaluation.j_space.models.PSObject.Properties) {
        $availableLayers[[string]$modelProperty.Name] = @(
            $modelProperty.Value.lens.source_layers | ForEach-Object { [int]$_ }
        )
    }

    $seen = @{}
    $directions = @()
    foreach ($setup in $plan.setups) {
        if ([string]$setup.method_id -notin $primaryMethods) {
            continue
        }
        $key = (
            "$($setup.model_id)|$($setup.method_id)|$($setup.track)|" +
            "$($setup.selected_layer)|$($setup.direction_artifact_sha256)"
        )
        if (-not $seen.ContainsKey($key)) {
            $seen[$key] = $true
            $directions += $setup
        }
    }
    if ($directions.Count -eq 0) {
        $staleRecords = @(
            Get-ChildItem -LiteralPath (Join-Path $jspaceRoot "records") `
                -Filter "*.jsonl" -File -ErrorAction SilentlyContinue
        )
        if ($staleRecords.Count -ne 0) {
            throw "J-space has records even though no primary direction is sealed-eligible"
        }
        Write-JspaceStatus -State "complete" -Detail "no primary direction was sealed-eligible"
        $receipt = Get-ValidatedJspaceCompletion -PythonExecutable $pythonExe `
            -ValidatorPath $completionValidator -RepositoryRoot $RepositoryRoot `
            -PlanPath $sealedPlanPath -LockPath $lockPath -StatusPath $statusPath `
            -RecordsDirectory (Join-Path $jspaceRoot "records") `
            -AtomsRoot (Join-Path $jspaceRoot "atoms") -CompletionPath $completionPath
        ($receipt | ConvertTo-Json -Compress) | Add-Content -LiteralPath $logPath
        return
    }

    $recordsDirectory = Join-Path $jspaceRoot "records"
    New-Item -ItemType Directory -Force -Path $recordsDirectory | Out-Null
    $recordPaths = @()
    for ($index = 0; $index -lt $directions.Count; $index++) {
        $setup = $directions[$index]
        $layer = [int]$setup.selected_layer
        $modelLayers = @($availableLayers[[string]$setup.model_id])
        $modelAtomRoot = Join-Path $jspaceRoot "atoms\$($setup.model_tag)\layer_$('{0:d2}' -f $layer)"
        $atomsPath = Join-Path $modelAtomRoot "atoms.pt"
        $labelsPath = Join-Path $modelAtomRoot "token_labels.json"
        $atomsManifest = Join-Path $modelAtomRoot "atoms_manifest.json"
        $hasLayer = $layer -in $modelLayers
        $cacheRebuilt = $false
        if ($hasLayer) {
            $cacheIsValid = Test-JspaceCache -ManifestPath $atomsManifest
            if (-not $cacheIsValid) {
                New-Item -ItemType Directory -Force -Path $modelAtomRoot | Out-Null
                Write-JspaceStatus -State "extracting_atoms" -Detail (
                    "$($setup.model_tag)/layer_$layer"
                )
                Invoke-NativeLogged -Phase (
                    "prepare_$($setup.model_tag)_layer_${layer}_atoms"
                ) -Arguments @(
                    "prepare-jspace-atoms",
                    "--model-config", [string]$setup.model_config,
                    "--layer", [string]$layer,
                    "--chunk-size", [string]$AtomChunkSize,
                    "--atoms-output", $atomsPath,
                    "--token-labels-output", $labelsPath,
                    "--manifest-output", $atomsManifest
                )
                if (-not (Test-JspaceCache -ManifestPath $atomsManifest)) {
                    throw "rebuilt J-space atom cache failed canonical validation: $atomsManifest"
                }
                $cacheRebuilt = $true
            }
        }

        $recordPath = Join-Path $recordsDirectory (
            "direction_$('{0:d3}' -f $index)_$($setup.setup_id.Substring(0,16)).jsonl"
        )
        $recordAtomsManifest = if ($hasLayer) { $atomsManifest } else { "" }
        $recordIsValid = Test-JspaceRecord -RecordPath $recordPath `
            -SetupId ([string]$setup.setup_id) -AtomsManifest $recordAtomsManifest
        if ($cacheRebuilt -or -not $recordIsValid) {
            Write-JspaceStatus -State "analyzing" -Detail (
                "$($setup.model_tag)/$($setup.method_id)/$($setup.track)/layer_$layer"
            )
            $arguments = @(
                "jspace",
                "--direction", (Join-Path $RepositoryRoot $setup.direction_path),
                "--setup", [string]$setup.track,
                "--k", "8", "16", "25",
                "--random-count", "50",
                "--random-seed", "20260824",
                "--max-working-gib", "8.0",
                "--max-dictionary-read-tib", "4.0"
            )
            if ($hasLayer) {
                $arguments += @("--atoms-manifest", $atomsManifest)
            }
            $arguments += @("--output", $recordPath)
            Invoke-NativeLogged -Phase (
                "jspace_$($setup.model_tag)_$($setup.method_id)_$($setup.track)_$layer"
            ) -Arguments $arguments
        }
        if (-not (Test-JspaceRecord -RecordPath $recordPath `
            -SetupId ([string]$setup.setup_id) -AtomsManifest $recordAtomsManifest)) {
            throw "J-space record failed exact post-run validation: $recordPath"
        }
        $recordPaths += $recordPath
    }

    $observedRecordPaths = @(
        Get-ChildItem -LiteralPath $recordsDirectory -Filter "*.jsonl" -File |
            ForEach-Object FullName | Sort-Object
    )
    $expectedRecordPaths = @($recordPaths | Sort-Object)
    if (($observedRecordPaths -join "|") -ne ($expectedRecordPaths -join "|")) {
        throw "J-space record directory contains stale or unexpected records"
    }

    Write-JspaceStatus -State "complete" -Detail (
        "primary_direction_records=$($recordPaths.Count);non_gating=true"
    )
    $receipt = Get-ValidatedJspaceCompletion -PythonExecutable $pythonExe `
        -ValidatorPath $completionValidator -RepositoryRoot $RepositoryRoot `
        -PlanPath $sealedPlanPath -LockPath $lockPath -StatusPath $statusPath `
        -RecordsDirectory $recordsDirectory -AtomsRoot (Join-Path $jspaceRoot "atoms") `
        -CompletionPath $completionPath
    ($receipt | ConvertTo-Json -Compress) | Add-Content -LiteralPath $logPath
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-JspaceStatus -State "failed" -Detail $failureDetail
    throw
}
finally {
    [void][SpLenseJspacePowerState]::SetThreadExecutionState($resetExecutionFlags)
}
