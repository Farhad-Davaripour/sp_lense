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
$personaValidator = Join-Path $artifactRoot "validate_persona_artifacts.py"
$statusPath = Join-Path $artifactRoot "local_construction_status.json"
$logPath = Join-Path $artifactRoot "local_construction.log"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SpLensePowerState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

function Write-RunStatus {
    param(
        [string]$Phase,
        [string]$State,
        [string]$Detail = ""
    )

    [ordered]@{
        schema_version = 1
        phase = $Phase
        state = $State
        detail = $Detail
        process_id = $PID
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Test-RepositoryArtifactPath {
    param([string]$RelativePath)

    if ([string]::IsNullOrWhiteSpace($RelativePath)) {
        return $false
    }
    $root = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $candidate = [IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    $prefix = $root + [IO.Path]::DirectorySeparatorChar
    return $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Test-JsonlArtifact {
    param([string]$Path)

    $rowCount = 0
    try {
        foreach ($line in [IO.File]::ReadLines($Path)) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }
            $null = $line | ConvertFrom-Json -ErrorAction Stop
            $rowCount += 1
        }
    }
    catch {
        return $false
    }
    return $rowCount -gt 0
}

function Test-DirectionManifest {
    param([string]$Path)

    try {
        $manifest = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -ErrorAction Stop
        $directions = @($manifest.directions)
        if ($directions.Count -lt 1) {
            return $false
        }
        foreach ($record in $directions) {
            foreach ($field in @(
                "path",
                "method_id",
                "layer",
                "intervention_geometry",
                "direction_float32_sha256",
                "direction_artifact_sha256",
                "metadata_sha256",
                "track",
                "construction_config_path",
                "construction_config_sha256"
            )) {
                if ($null -eq $record.$field -or [string]::IsNullOrWhiteSpace([string]$record.$field)) {
                    return $false
                }
            }
            if (
                -not (Test-RepositoryArtifactPath -RelativePath $record.path) -or
                -not (Test-RepositoryArtifactPath -RelativePath $record.construction_config_path)
            ) {
                return $false
            }
            $directionPath = Join-Path $RepositoryRoot $record.path
            $constructionPath = Join-Path $RepositoryRoot $record.construction_config_path
            if (
                -not (Test-Path -LiteralPath $directionPath -PathType Leaf) -or
                -not (Test-Path -LiteralPath $constructionPath -PathType Leaf)
            ) {
                return $false
            }
            $direction = Get-Content -LiteralPath $directionPath -Raw |
                ConvertFrom-Json -ErrorAction Stop
            $construction = Get-Content -LiteralPath $constructionPath -Raw |
                ConvertFrom-Json -ErrorAction Stop
            if (
                [string]$direction.method -ne [string]$record.method_id -or
                [int]$direction.layer -ne [int]$record.layer -or
                [string]$direction.intervention_geometry -ne [string]$record.intervention_geometry -or
                [string]$direction.direction_sha256 -ne [string]$record.direction_float32_sha256 -or
                [string]$direction.artifact_sha256 -ne [string]$record.direction_artifact_sha256 -or
                [string]$direction.metadata_sha256 -ne [string]$record.metadata_sha256 -or
                @($direction.direction).Count -ne [int]$direction.d_model -or
                [string]$construction.method_id -ne [string]$record.method_id -or
                [string]$construction.track -ne [string]$record.track -or
                [string]$construction.direction_float32_sha256 -ne [string]$record.direction_float32_sha256 -or
                [string]$construction.direction_artifact_sha256 -ne [string]$record.direction_artifact_sha256
            ) {
                return $false
            }
            $observedConstructionHash = (Get-FileHash -LiteralPath $constructionPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($observedConstructionHash -ne [string]$record.construction_config_sha256) {
                return $false
            }
            $evidenceItems = @()
            if ($construction.PSObject.Properties.Name -contains "evidence_artifacts") {
                $evidenceItems = @($construction.evidence_artifacts) |
                    Where-Object { $null -ne $_ }
            }
            foreach ($evidence in $evidenceItems) {
                if (
                    -not (Test-RepositoryArtifactPath -RelativePath $evidence.path) -or
                    -not (Test-Path -LiteralPath (Join-Path $RepositoryRoot $evidence.path) -PathType Leaf)
                ) {
                    return $false
                }
                $observedEvidenceHash = (
                    Get-FileHash -LiteralPath (Join-Path $RepositoryRoot $evidence.path) -Algorithm SHA256
                ).Hash.ToLowerInvariant()
                if ($observedEvidenceHash -ne [string]$evidence.sha256) {
                    return $false
                }
            }
        }
    }
    catch {
        return $false
    }
    return $true
}

function Test-CompletionArtifact {
    param([string]$Path)

    if (
        -not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-Item -LiteralPath $Path).Length -lt 1
    ) {
        return $false
    }
    if ([IO.Path]::GetFileName($Path) -eq "direction_manifest.json") {
        return Test-DirectionManifest -Path $Path
    }
    if ([IO.Path]::GetExtension($Path) -eq ".jsonl") {
        return Test-JsonlArtifact -Path $Path
    }
    try {
        $null = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Test-PersonaArtifact {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("raw", "requests")][string]$Kind,
        [Parameter(Mandatory = $true)][string]$ModelConfig,
        [Parameter(Mandatory = $true)][string]$RawPath,
        [string]$RequestsPath = ""
    )

    if (-not (Test-Path -LiteralPath $personaValidator -PathType Leaf)) {
        throw "persona completion validator is missing: $personaValidator"
    }
    $arguments = @(
        $personaValidator,
        "--repo-root", $RepositoryRoot,
        "--lock", $lockPath,
        "--model-config", $ModelConfig,
        $Kind,
        "--raw", $RawPath
    )
    if ($Kind -eq "requests") {
        $arguments += @("--requests", $RequestsPath)
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

function Invoke-LockedPersonaGeneration {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][string]$ModelConfig,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    if (
        (Test-Path -LiteralPath $OutputPath -PathType Leaf) -and
        (Test-PersonaArtifact -Kind "raw" -ModelConfig $ModelConfig -RawPath $OutputPath)
    ) {
        Write-RunStatus -Phase $Phase -State "already_complete" -Detail $OutputPath
        return
    }
    $temporaryPath = "$OutputPath.generating.tmp"
    Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    Write-RunStatus -Phase $Phase -State "running"
    "[$([DateTime]::UtcNow.ToString('o'))] START $Phase" | Tee-Object -FilePath $logPath -Append
    try {
        $priorErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $comparisonExe generate-persona --model-config $ModelConfig --output $temporaryPath `
                2>&1 | Tee-Object -FilePath $logPath -Append
            $nativeExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $priorErrorActionPreference
        }
        if ($nativeExitCode -ne 0) {
            throw "$Phase failed with exit code $nativeExitCode"
        }
        if (-not (Test-PersonaArtifact -Kind "raw" -ModelConfig $ModelConfig `
            -RawPath $temporaryPath)) {
            throw "$Phase did not produce the exact locked 2,000-row persona grid"
        }
        Move-Item -LiteralPath $temporaryPath -Destination $OutputPath -Force
        if (-not (Test-PersonaArtifact -Kind "raw" -ModelConfig $ModelConfig `
            -RawPath $OutputPath)) {
            throw "$Phase atomic publication failed validation"
        }
    }
    finally {
        Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    }
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE $Phase" | Tee-Object -FilePath $logPath -Append
    Write-RunStatus -Phase $Phase -State "complete" -Detail $OutputPath
}

function Invoke-LockedPersonaRequests {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][string]$ModelConfig,
        [Parameter(Mandatory = $true)][string]$RawPath,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    if (-not (Test-PersonaArtifact -Kind "raw" -ModelConfig $ModelConfig -RawPath $RawPath)) {
        throw "$Phase cannot render requests from an incomplete or stale persona grid"
    }
    if (
        (Test-Path -LiteralPath $OutputPath -PathType Leaf) -and
        (Test-PersonaArtifact -Kind "requests" -ModelConfig $ModelConfig `
            -RawPath $RawPath -RequestsPath $OutputPath)
    ) {
        Write-RunStatus -Phase $Phase -State "already_complete" -Detail $OutputPath
        return
    }
    $temporaryPath = "$OutputPath.rendering.tmp"
    Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    Write-RunStatus -Phase $Phase -State "running"
    try {
        Invoke-LockedCommand -Phase $Phase -Arguments @(
            "judge-requests", "--kind", "persona", "--input", $RawPath,
            "--output", $temporaryPath
        ) -CompletionPath $temporaryPath
        if (-not (Test-PersonaArtifact -Kind "requests" -ModelConfig $ModelConfig `
            -RawPath $RawPath -RequestsPath $temporaryPath)) {
            throw "$Phase did not render the exact locked request set"
        }
        Move-Item -LiteralPath $temporaryPath -Destination $OutputPath -Force
        if (-not (Test-PersonaArtifact -Kind "requests" -ModelConfig $ModelConfig `
            -RawPath $RawPath -RequestsPath $OutputPath)) {
            throw "$Phase atomic publication failed validation"
        }
    }
    finally {
        Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    }
    Write-RunStatus -Phase $Phase -State "complete" -Detail $OutputPath
}

function Invoke-LockedCommand {
    param(
        [string]$Phase,
        [string[]]$Arguments,
        [string]$CompletionPath
    )

    if (Test-CompletionArtifact -Path $CompletionPath) {
        Write-RunStatus -Phase $Phase -State "already_complete" -Detail $CompletionPath
        return
    }
    if (Test-Path -LiteralPath $CompletionPath) {
        throw "$Phase found an invalid or incomplete completion artifact at $CompletionPath"
    }

    Write-RunStatus -Phase $Phase -State "running"
    "[$([DateTime]::UtcNow.ToString('o'))] START $Phase" | Tee-Object -FilePath $logPath -Append
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell wraps native stderr lines as ErrorRecord objects. Model
        # loading progress and upstream warnings legitimately use stderr, so capture
        # them without converting them into terminating PowerShell exceptions. The
        # native exit code remains the authoritative success signal.
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
    if (-not (Test-CompletionArtifact -Path $CompletionPath)) {
        throw "$Phase exited successfully but did not publish a valid $CompletionPath"
    }
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE $Phase" | Tee-Object -FilePath $logPath -Append
    Write-RunStatus -Phase $Phase -State "complete" -Detail $CompletionPath
}

if ($SelfTest) {
    $knownManifests = @(
        "artifacts\steering_comparison\qwen35_08b\directions\gradient\direction_manifest.json",
        "artifacts\steering_comparison\qwen35_08b\directions\caa\direction_manifest.json",
        "artifacts\steering_comparison\qwen35_08b\directions\bipo_matched\direction_manifest.json",
        "artifacts\steering_comparison\qwen35_08b\directions\random\direction_manifest.json"
    )
    foreach ($relativePath in $knownManifests) {
        $path = Join-Path $RepositoryRoot $relativePath
        if (Test-Path -LiteralPath $path) {
            if (-not (Test-CompletionArtifact -Path $path)) {
                throw "self-test rejected known manifest $relativePath"
            }
        }
    }
    $knownJsonl = Join-Path $RepositoryRoot "data\scenarios.jsonl"
    if (-not (Test-CompletionArtifact -Path $knownJsonl)) {
        throw "self-test rejected known JSONL $knownJsonl"
    }
    $knownNonJson = Join-Path $RepositoryRoot "README.md"
    if (Test-CompletionArtifact -Path $knownNonJson) {
        throw "self-test accepted non-JSON completion marker $knownNonJson"
    }
    Write-Output "SELFTEST_OK run_remaining_local_construction"
    return
}

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED. Windows PowerShell parses the high-bit
# hexadecimal literals as signed Int32 values, so parse their UInt32 bit patterns
# explicitly. The state is reset in finally.
$keepAwakeFlags = [uint32]::Parse("80000001", [Globalization.NumberStyles]::HexNumber)
$resetExecutionFlags = [uint32]::Parse("80000000", [Globalization.NumberStyles]::HexNumber)
[void][SpLensePowerState]::SetThreadExecutionState($keepAwakeFlags)

try {
    Write-RunStatus -Phase "verify_stage1" -State "running"
    & $comparisonExe verify-stage1 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-one verification failed"
    }

    $model08Root = Join-Path $artifactRoot "qwen35_08b"
    $model2Root = Join-Path $artifactRoot "qwen35_2b"
    New-Item -ItemType Directory -Force (Join-Path $model08Root "directions") | Out-Null
    New-Item -ItemType Directory -Force (Join-Path $model2Root "directions") | Out-Null

    Invoke-LockedCommand -Phase "qwen35_08b_bipo_canonical" -Arguments @(
        "fit", "--model-config", "configs\qwen35_08b_aligned.json",
        "--method", "bipo", "--track", "canonical",
        "--output", "artifacts\steering_comparison\qwen35_08b\directions\bipo_canonical"
    ) -CompletionPath (Join-Path $model08Root "directions\bipo_canonical\direction_manifest.json")

    Invoke-LockedPersonaGeneration -Phase "qwen35_08b_persona_generation" `
        -ModelConfig "configs\qwen35_08b_aligned.json" `
        -OutputPath (Join-Path $model08Root "persona_raw.jsonl")

    Invoke-LockedPersonaRequests -Phase "qwen35_08b_persona_requests" `
        -ModelConfig "configs\qwen35_08b_aligned.json" `
        -RawPath (Join-Path $model08Root "persona_raw.jsonl") `
        -OutputPath (Join-Path $model08Root "persona_judge_requests.jsonl")

    Invoke-LockedCommand -Phase "qwen35_2b_gradient" -Arguments @(
        "fit", "--model-config", "configs\qwen35_2b_aligned.json",
        "--method", "gradient",
        "--output", "artifacts\steering_comparison\qwen35_2b\directions\gradient"
    ) -CompletionPath (Join-Path $model2Root "directions\gradient\direction_manifest.json")

    Invoke-LockedCommand -Phase "qwen35_2b_caa" -Arguments @(
        "fit", "--model-config", "configs\qwen35_2b_aligned.json",
        "--method", "caa",
        "--output", "artifacts\steering_comparison\qwen35_2b\directions\caa"
    ) -CompletionPath (Join-Path $model2Root "directions\caa\direction_manifest.json")

    Invoke-LockedCommand -Phase "qwen35_2b_bipo_matched" -Arguments @(
        "fit", "--model-config", "configs\qwen35_2b_aligned.json",
        "--method", "bipo", "--track", "matched",
        "--output", "artifacts\steering_comparison\qwen35_2b\directions\bipo_matched"
    ) -CompletionPath (Join-Path $model2Root "directions\bipo_matched\direction_manifest.json")

    Invoke-LockedCommand -Phase "qwen35_2b_bipo_canonical" -Arguments @(
        "fit", "--model-config", "configs\qwen35_2b_aligned.json",
        "--method", "bipo", "--track", "canonical",
        "--output", "artifacts\steering_comparison\qwen35_2b\directions\bipo_canonical"
    ) -CompletionPath (Join-Path $model2Root "directions\bipo_canonical\direction_manifest.json")

    Invoke-LockedCommand -Phase "qwen35_2b_random" -Arguments @(
        "fit", "--model-config", "configs\qwen35_2b_aligned.json",
        "--method", "random",
        "--output", "artifacts\steering_comparison\qwen35_2b\directions\random"
    ) -CompletionPath (Join-Path $model2Root "directions\random\direction_manifest.json")

    Invoke-LockedPersonaGeneration -Phase "qwen35_2b_persona_generation" `
        -ModelConfig "configs\qwen35_2b_aligned.json" `
        -OutputPath (Join-Path $model2Root "persona_raw.jsonl")

    Invoke-LockedPersonaRequests -Phase "qwen35_2b_persona_requests" `
        -ModelConfig "configs\qwen35_2b_aligned.json" `
        -RawPath (Join-Path $model2Root "persona_raw.jsonl") `
        -OutputPath (Join-Path $model2Root "persona_judge_requests.jsonl")

    Write-RunStatus -Phase "local_construction" -State "complete" -Detail "Persona judge requests are rendered; no hosted judge was called."
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-RunStatus -Phase "local_construction" -State "failed" -Detail $failureDetail
    throw
}
finally {
    [void][SpLensePowerState]::SetThreadExecutionState($resetExecutionFlags)
}
