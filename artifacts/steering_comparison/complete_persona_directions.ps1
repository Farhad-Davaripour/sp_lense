param(
    [string]$RepositoryRoot = "C:\Users\farha\repos\sp_lense",
    [double]$MaxCostUsd = 0,
    [double]$InputPricePerMillion = 0.40,
    [double]$OutputPricePerMillion = 1.60,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $RepositoryRoot

$pythonExe = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$comparisonExe = Join-Path $RepositoryRoot ".venv\Scripts\sp-lense-compare-steering.exe"
$artifactRoot = Join-Path $RepositoryRoot "artifacts\steering_comparison"
$transport = Join-Path $artifactRoot "submit_openai_judge_requests.py"
$personaValidator = Join-Path $artifactRoot "validate_persona_artifacts.py"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$statusPath = Join-Path $artifactRoot "persona_completion_status.json"
$logPath = Join-Path $artifactRoot "persona_completion.log"
$modelSpecs = @(
    [ordered]@{
        tag = "qwen35_08b"
        config = "configs\qwen35_08b_aligned.json"
    },
    [ordered]@{
        tag = "qwen35_2b"
        config = "configs\qwen35_2b_aligned.json"
    }
)
$lockedInputPricePerMillion = 0.40
$lockedOutputPricePerMillion = 1.60

function Convert-Invariant {
    param([double]$Value)
    return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Write-PersonaStatus {
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

function Invoke-Native {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$Phase
    )
    "[$([DateTime]::UtcNow.ToString('o'))] START $Phase" |
        Tee-Object -FilePath $logPath -Append
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
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE $Phase" |
        Tee-Object -FilePath $logPath -Append
}

function Test-PersonaArtifact {
    param(
        [Parameter(Mandatory = $true)][ValidateSet(
            "raw", "requests", "scored", "manifest"
        )][string]$Kind,
        [Parameter(Mandatory = $true)][string]$ModelConfig,
        [string]$RawPath = "",
        [string]$RequestsPath = "",
        [string]$ResponsesPath = "",
        [string]$ScoredPath = "",
        [string]$ManifestPath = ""
    )

    $arguments = @(
        $personaValidator,
        "--repo-root", $RepositoryRoot,
        "--lock", $lockPath,
        "--model-config", $ModelConfig,
        $Kind
    )
    if ($Kind -in @("raw", "requests", "scored")) {
        $arguments += @("--raw", $RawPath)
    }
    if ($Kind -eq "requests") {
        $arguments += @("--requests", $RequestsPath)
    }
    if ($Kind -eq "scored") {
        $arguments += @("--responses", $ResponsesPath, "--scored", $ScoredPath)
    }
    if ($Kind -eq "manifest") {
        $arguments += @("--scored", $ScoredPath, "--manifest", $ManifestPath)
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

function Import-OpenAIKeyForChildProcess {
    $processKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "Process")
    if (-not [string]::IsNullOrWhiteSpace($processKey)) {
        Remove-Variable -Name processKey -Force
        return
    }
    $userKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")
    if ([string]::IsNullOrWhiteSpace($userKey)) {
        throw "OPENAI_API_KEY is absent after the cost preflight; no request was sent"
    }
    try {
        [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $userKey, "Process")
    }
    finally {
        Remove-Variable -Name userKey -Force
    }
}

if (
    [double]::IsNaN($InputPricePerMillion) -or
    [double]::IsInfinity($InputPricePerMillion) -or
    [double]::IsNaN($OutputPricePerMillion) -or
    [double]::IsInfinity($OutputPricePerMillion) -or
    [math]::Abs($InputPricePerMillion - $lockedInputPricePerMillion) -gt 1e-12 -or
    [math]::Abs($OutputPricePerMillion - $lockedOutputPricePerMillion) -gt 1e-12
) {
    throw "judge prices are locked to `$0.40/M input and `$1.60/M output; no request was sent"
}

if ($SelfTest) {
    if ($modelSpecs.Count -ne 2 -or (@($modelSpecs.tag | Select-Object -Unique)).Count -ne 2) {
        throw "persona completion must cover exactly two unique locked models"
    }
    if ($MaxCostUsd -ne 0) {
        throw "self-test must not accept or spend a cost allowance"
    }
    [ordered]@{
        status = "self_test_passed"
        models = @($modelSpecs.tag)
        paid_calls = 0
        explicit_positive_total_cap_required = $true
        global_cap_check = "sum_of_conservative_per_file_upper_bounds"
        locked_input_price_per_million_usd = $lockedInputPricePerMillion
        locked_output_price_per_million_usd = $lockedOutputPricePerMillion
        request_regeneration_verification_required = $true
        post_submit_transport_verification_required = $true
        attach_kind = "persona"
        fit_method = "persona"
    } | ConvertTo-Json
    return
}

if (
    [double]::IsNaN($MaxCostUsd) -or [double]::IsInfinity($MaxCostUsd) -or
    $MaxCostUsd -le 0
) {
    throw "-MaxCostUsd must be an explicit positive finite total ceiling; no request was sent"
}
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SpLensePersonaPowerState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$keepAwakeFlags = [uint32]::Parse("80000001", [Globalization.NumberStyles]::HexNumber)
$resetExecutionFlags = [uint32]::Parse("80000000", [Globalization.NumberStyles]::HexNumber)
[void][SpLensePersonaPowerState]::SetThreadExecutionState($keepAwakeFlags)

try {
    Write-PersonaStatus -Phase "verify_stage1" -State "running"
    Invoke-Native -Executable $comparisonExe -Arguments @("verify-stage1") -Phase "verify_stage1"

    $constructionStatusPath = Join-Path $artifactRoot "local_construction_status.json"
    if (-not (Test-Path -LiteralPath $constructionStatusPath -PathType Leaf)) {
        throw "local construction status is missing; no request was sent"
    }
    $constructionStatus = Get-Content -Raw -LiteralPath $constructionStatusPath |
        ConvertFrom-Json
    if ($constructionStatus.phase -ne "local_construction" -or $constructionStatus.state -ne "complete") {
        throw "local construction is not complete; no request was sent"
    }
    foreach ($helper in @($transport, $personaValidator)) {
        if (-not (Test-Path -LiteralPath $helper -PathType Leaf)) {
            throw "locked persona helper is missing: $helper; no request was sent"
        }
    }

    $preflights = @()
    foreach ($spec in $modelSpecs) {
        $modelRoot = Join-Path $artifactRoot $spec.tag
        $requests = Join-Path $modelRoot "persona_judge_requests.jsonl"
        $responses = Join-Path $modelRoot "persona_judge_responses.jsonl"
        $workDirectory = Join-Path $modelRoot "persona_judge_transport"
        foreach ($required in @(
            (Join-Path $modelRoot "persona_raw.jsonl"),
            $requests,
            (Join-Path $RepositoryRoot $spec.config)
        )) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw "required persona input is missing: $required; no request was sent"
            }
        }
        $raw = Join-Path $modelRoot "persona_raw.jsonl"
        if (-not (Test-PersonaArtifact -Kind "raw" -ModelConfig ([string]$spec.config) `
            -RawPath $raw)) {
            throw "persona raw grid is incomplete or stale for $($spec.tag); no request was sent"
        }
        if (-not (Test-PersonaArtifact -Kind "requests" `
            -ModelConfig ([string]$spec.config) -RawPath $raw -RequestsPath $requests)) {
            throw "persona request set is incomplete or stale for $($spec.tag); no request was sent"
        }
        $rebuiltRequests = Join-Path $modelRoot "persona_judge_requests.rebuilt.tmp"
        try {
            Invoke-Native -Executable $comparisonExe -Arguments @(
                "judge-requests", "--kind", "persona",
                "--input", (Join-Path $modelRoot "persona_raw.jsonl"),
                "--output", $rebuiltRequests
            ) -Phase "$($spec.tag)_regenerate_persona_requests"
            if (
                (Get-FileHash -Algorithm SHA256 -LiteralPath $requests).Hash -ne
                (Get-FileHash -Algorithm SHA256 -LiteralPath $rebuiltRequests).Hash
            ) {
                throw "persona judge requests differ from locked regeneration: $requests"
            }
        }
        finally {
            Remove-Item -LiteralPath $rebuiltRequests -Force -ErrorAction SilentlyContinue
        }
        Write-PersonaStatus -Phase "$($spec.tag)_judge_preflight" -State "running"
        Invoke-Native -Executable $pythonExe -Arguments @(
            $transport,
            "--requests", $requests,
            "--responses", $responses,
            "--work-dir", $workDirectory,
            "--max-cost-usd", (Convert-Invariant $MaxCostUsd),
            "--input-price-per-million", (Convert-Invariant $InputPricePerMillion),
            "--output-price-per-million", (Convert-Invariant $OutputPricePerMillion),
            "--dry-run"
        ) -Phase "$($spec.tag)_judge_preflight"
        $estimate = Get-Content -Raw -LiteralPath (
            Join-Path $workDirectory "cost_preflight.json"
        ) | ConvertFrom-Json
        $preflights += [pscustomobject]@{
            spec = $spec
            requests = $requests
            responses = $responses
            work_directory = $workDirectory
            safe_upper_bound_usd = [double]$estimate.safe_upper_bound_usd
        }
    }
    $globalUpperBound = [double](
        ($preflights | Measure-Object -Property safe_upper_bound_usd -Sum).Sum
    )
    if ($globalUpperBound -gt $MaxCostUsd) {
        throw (
            "combined conservative upper bound $globalUpperBound USD exceeds the explicit " +
            "$MaxCostUsd USD total ceiling; no request was sent"
        )
    }
    Write-PersonaStatus -Phase "judge_global_preflight" -State "complete" -Detail (
        "safe_upper_bound_usd=$(Convert-Invariant $globalUpperBound);" +
        "user_ceiling_usd=$(Convert-Invariant $MaxCostUsd)"
    )

    Import-OpenAIKeyForChildProcess

    foreach ($preflight in $preflights) {
        $tag = [string]$preflight.spec.tag
        Write-PersonaStatus -Phase "${tag}_judge_submit" -State "running"
        Invoke-Native -Executable $pythonExe -Arguments @(
            $transport,
            "--requests", [string]$preflight.requests,
            "--responses", [string]$preflight.responses,
            "--work-dir", [string]$preflight.work_directory,
            "--max-cost-usd", (Convert-Invariant $MaxCostUsd),
            "--input-price-per-million", (Convert-Invariant $InputPricePerMillion),
            "--output-price-per-million", (Convert-Invariant $OutputPricePerMillion)
        ) -Phase "${tag}_judge_submit"
        if (-not (Test-Path -LiteralPath $preflight.responses -PathType Leaf)) {
            throw "judge transport did not publish $($preflight.responses)"
        }
        Invoke-Native -Executable $pythonExe -Arguments @(
            $transport,
            "--requests", [string]$preflight.requests,
            "--responses", [string]$preflight.responses,
            "--work-dir", [string]$preflight.work_directory,
            "--verify-only"
        ) -Phase "${tag}_judge_verify"
    }

    foreach ($spec in $modelSpecs) {
        $modelRoot = Join-Path $artifactRoot $spec.tag
        $raw = Join-Path $modelRoot "persona_raw.jsonl"
        $responses = Join-Path $modelRoot "persona_judge_responses.jsonl"
        $scored = Join-Path $modelRoot "persona_scored.jsonl"
        $manifest = Join-Path $modelRoot "directions\persona\direction_manifest.json"
        $scoredIsValid = (
            (Test-Path -LiteralPath $scored -PathType Leaf) -and
            (Test-PersonaArtifact -Kind "scored" -ModelConfig ([string]$spec.config) `
                -RawPath $raw -ResponsesPath $responses -ScoredPath $scored)
        )
        if (-not $scoredIsValid) {
            Write-PersonaStatus -Phase "$($spec.tag)_attach_persona" -State "running"
            $scoredTemporary = "$scored.rebuilt.tmp"
            Remove-Item -LiteralPath $scoredTemporary -Force -ErrorAction SilentlyContinue
            try {
                Invoke-Native -Executable $comparisonExe -Arguments @(
                    "attach-judgments", "--kind", "persona",
                    "--input", $raw,
                    "--responses", $responses,
                    "--output", $scoredTemporary
                ) -Phase "$($spec.tag)_attach_persona"
                if (-not (Test-PersonaArtifact -Kind "scored" `
                    -ModelConfig ([string]$spec.config) -RawPath $raw `
                    -ResponsesPath $responses -ScoredPath $scoredTemporary)) {
                    throw "rebuilt persona scored rows failed exact receipt validation"
                }
                Move-Item -LiteralPath $scoredTemporary -Destination $scored -Force
            }
            finally {
                Remove-Item -LiteralPath $scoredTemporary -Force -ErrorAction SilentlyContinue
            }
        }
        if (-not (Test-PersonaArtifact -Kind "scored" `
            -ModelConfig ([string]$spec.config) -RawPath $raw `
            -ResponsesPath $responses -ScoredPath $scored)) {
            throw "published persona scored rows failed exact validation"
        }
        $manifestIsValid = (
            (Test-Path -LiteralPath $manifest -PathType Leaf) -and
            (Test-PersonaArtifact -Kind "manifest" -ModelConfig ([string]$spec.config) `
                -ScoredPath $scored -ManifestPath $manifest)
        )
        if (-not $manifestIsValid) {
            Remove-Item -LiteralPath $manifest -Force -ErrorAction SilentlyContinue
            Write-PersonaStatus -Phase "$($spec.tag)_fit_persona" -State "running"
            Invoke-Native -Executable $comparisonExe -Arguments @(
                "fit", "--model-config", [string]$spec.config,
                "--method", "persona",
                "--persona-rollouts", $scored,
                "--output", "artifacts\steering_comparison\$($spec.tag)\directions\persona"
            ) -Phase "$($spec.tag)_fit_persona"
        }
        if (-not (Test-PersonaArtifact -Kind "manifest" `
            -ModelConfig ([string]$spec.config) -ScoredPath $scored `
            -ManifestPath $manifest)) {
            throw "persona fit did not publish a complete exact direction manifest: $manifest"
        }
    }

    Write-PersonaStatus -Phase "persona_directions" -State "complete" -Detail (
        "models=2;safe_upper_bound_usd=$(Convert-Invariant $globalUpperBound);" +
        "user_ceiling_usd=$(Convert-Invariant $MaxCostUsd)"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-PersonaStatus -Phase "persona_directions" -State "failed" -Detail $failureDetail
    throw
}
finally {
    [void][SpLensePersonaPowerState]::SetThreadExecutionState($resetExecutionFlags)
}
