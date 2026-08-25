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
$openRoot = Join-Path $artifactRoot "validation_open"
$transport = Join-Path $artifactRoot "submit_openai_judge_requests.py"
$orchestrator = Join-Path $artifactRoot "locked_open_orchestration.py"
$openChainVerifier = Join-Path $artifactRoot "verify_open_judgment_chain.py"
$summaryBuilder = Join-Path $artifactRoot "build_locked_final_summaries.ps1"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$preopenManifest = Join-Path $RepositoryRoot "configs\steering_comparison_preopen_lock.json"
$planPath = Join-Path $openRoot "validation_open_plan.json"
$generationsPath = Join-Path $openRoot "open_generations_all.jsonl"
$requestsPath = Join-Path $openRoot "open_judge_requests.jsonl"
$responsesPath = Join-Path $openRoot "open_judge_responses.jsonl"
$scoredPath = Join-Path $openRoot "open_scored_all.jsonl"
$workDirectory = Join-Path $openRoot "judge_transport"
$statusPath = Join-Path $artifactRoot "validation_open_completion_status.json"
$logPath = Join-Path $artifactRoot "validation_open_completion.log"
$lockedInputPricePerMillion = 0.40
$lockedOutputPricePerMillion = 1.60

function Convert-Invariant {
    param([double]$Value)
    return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Write-CompletionStatus {
    param(
        [string]$State,
        [string]$Detail = ""
    )
    [ordered]@{
        schema_version = 1
        split = "validation"
        state = $State
        detail = $Detail
        process_id = $PID
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Invoke-NativeLogged {
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

function Import-OpenAIKeyForChildProcess {
    $processKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "Process")
    if (-not [string]::IsNullOrWhiteSpace($processKey)) {
        Remove-Variable -Name processKey -Force
        return
    }
    $userKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")
    if ([string]::IsNullOrWhiteSpace($userKey)) {
        throw "OPENAI_API_KEY is absent after cost preflight; no request was sent"
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
    if ($MaxCostUsd -ne 0) {
        throw "self-test must not accept or spend a cost allowance"
    }
    [ordered]@{
        status = "self_test_passed"
        split = "validation"
        explicit_positive_phase_cap_required = $true
        locked_input_price_per_million_usd = $lockedInputPricePerMillion
        locked_output_price_per_million_usd = $lockedOutputPricePerMillion
        paid_calls = 0
        plan_regeneration_verification_required = $true
        combined_generation_regeneration_verification_required = $true
        request_regeneration_verification_required = $true
        post_submit_transport_verification_required = $true
        keep_awake_with_guaranteed_restore = $true
        judgment_kind = "open"
        exact_partitioning = $true
        final_summary_count = 16
        fallback_candidates = 0
    } | ConvertTo-Json
    return
}

if (
    [double]::IsNaN($MaxCostUsd) -or [double]::IsInfinity($MaxCostUsd) -or
    $MaxCostUsd -le 0
) {
    throw "-MaxCostUsd must be an explicit positive finite phase ceiling; no request was sent"
}

Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.Runtime.InteropServices;

public static class SpLenseValidationOpenIntegrity
{
    private const uint EsSystemRequired = 0x00000001;
    private const uint EsContinuous = 0x80000000;

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint SetThreadExecutionState(uint esFlags);

    public static bool KeepAwake()
    {
        return SetThreadExecutionState(EsContinuous | EsSystemRequired) != 0;
    }

    public static void RestorePowerPolicy()
    {
        SetThreadExecutionState(EsContinuous);
    }

    private static int ReadChunk(Stream stream, byte[] buffer)
    {
        int offset = 0;
        while (offset < buffer.Length)
        {
            int count = stream.Read(buffer, offset, buffer.Length - offset);
            if (count == 0)
            {
                break;
            }
            offset += count;
        }
        return offset;
    }

    public static bool FilesEqual(string leftPath, string rightPath)
    {
        FileInfo leftInfo = new FileInfo(leftPath);
        FileInfo rightInfo = new FileInfo(rightPath);
        if (!leftInfo.Exists || !rightInfo.Exists || leftInfo.Length != rightInfo.Length)
        {
            return false;
        }
        using (FileStream left = File.OpenRead(leftPath))
        using (FileStream right = File.OpenRead(rightPath))
        {
            byte[] leftBuffer = new byte[65536];
            byte[] rightBuffer = new byte[65536];
            while (true)
            {
                int leftCount = ReadChunk(left, leftBuffer);
                int rightCount = ReadChunk(right, rightBuffer);
                if (leftCount != rightCount)
                {
                    return false;
                }
                if (leftCount == 0)
                {
                    return true;
                }
                for (int index = 0; index < leftCount; index++)
                {
                    if (leftBuffer[index] != rightBuffer[index])
                    {
                        return false;
                    }
                }
            }
        }
    }
}
"@

function Assert-ExactFileMatch {
    param(
        [string]$ExpectedPath,
        [string]$RebuiltPath,
        [string]$FailureMessage
    )
    if (-not [SpLenseValidationOpenIntegrity]::FilesEqual($ExpectedPath, $RebuiltPath)) {
        throw $FailureMessage
    }
}

$keepAwakeActive = $false
try {
    if (-not [SpLenseValidationOpenIntegrity]::KeepAwake()) {
        throw "Windows refused the validation-open keep-awake request"
    }
    $keepAwakeActive = $true
    Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
        "verify-preopen",
        "--preopen-manifest", $preopenManifest
    ) -Phase "verify_preopen_before_validation_judging"
    foreach ($required in @(
        $lockPath, $preopenManifest, $planPath, $transport, $orchestrator,
        $openChainVerifier, $summaryBuilder
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "validation-open completion prerequisite is missing: $required"
        }
    }

    $rebuiltPlan = Join-Path $openRoot "validation_open_plan.rebuilt.tmp"
    try {
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $orchestrator, "plan",
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--manifest", $preopenManifest,
            "--output-dir", $openRoot,
            "--split", "validation",
            "--output", $rebuiltPlan
        ) -Phase "regenerate_validation_open_plan"
        Assert-ExactFileMatch -ExpectedPath $planPath -RebuiltPath $rebuiltPlan `
            -FailureMessage "validation-open plan differs byte-for-byte from locked regeneration"
    }
    finally {
        Remove-Item -LiteralPath $rebuiltPlan -Force -ErrorAction SilentlyContinue
    }

    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    if ($plan.split -ne "validation" -or $plan.setup_count -ne @($plan.setups).Count) {
        throw "validation-open plan is inconsistent"
    }

    if ($plan.setup_count -gt 0) {
        foreach ($required in @($generationsPath, $requestsPath)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                throw "validation-open generated input is missing: $required"
            }
        }

        $rebuiltGenerations = Join-Path $openRoot "open_generations_all.rebuilt.tmp"
        try {
            Invoke-NativeLogged -Executable $pythonExe -Arguments @(
                $orchestrator, "combine-generations",
                "--repo-root", $RepositoryRoot,
                "--plan", $planPath,
                "--output", $rebuiltGenerations
            ) -Phase "regenerate_validation_open_combined_generations"
            Assert-ExactFileMatch -ExpectedPath $generationsPath `
                -RebuiltPath $rebuiltGenerations `
                -FailureMessage (
                    "validation-open combined generations differ byte-for-byte " +
                    "from locked regeneration"
                )
        }
        finally {
            Remove-Item -LiteralPath $rebuiltGenerations -Force -ErrorAction SilentlyContinue
        }

        $rebuiltRequests = Join-Path $openRoot "open_judge_requests.rebuilt.tmp"
        try {
            Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
                "judge-requests", "--kind", "open",
                "--input", $generationsPath,
                "--output", $rebuiltRequests
            ) -Phase "regenerate_validation_open_judge_requests"
            Assert-ExactFileMatch -ExpectedPath $requestsPath -RebuiltPath $rebuiltRequests `
                -FailureMessage (
                    "validation-open judge requests differ byte-for-byte from locked regeneration"
                )
        }
        finally {
            Remove-Item -LiteralPath $rebuiltRequests -Force -ErrorAction SilentlyContinue
        }
        Write-CompletionStatus -State "judge_preflight"
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $transport,
            "--requests", $requestsPath,
            "--responses", $responsesPath,
            "--work-dir", $workDirectory,
            "--max-cost-usd", (Convert-Invariant $MaxCostUsd),
            "--input-price-per-million", (Convert-Invariant $InputPricePerMillion),
            "--output-price-per-million", (Convert-Invariant $OutputPricePerMillion),
            "--dry-run"
        ) -Phase "validation_open_judge_preflight"

        Import-OpenAIKeyForChildProcess
        Write-CompletionStatus -State "judging"
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $transport,
            "--requests", $requestsPath,
            "--responses", $responsesPath,
            "--work-dir", $workDirectory,
            "--max-cost-usd", (Convert-Invariant $MaxCostUsd),
            "--input-price-per-million", (Convert-Invariant $InputPricePerMillion),
            "--output-price-per-million", (Convert-Invariant $OutputPricePerMillion)
        ) -Phase "validation_open_judge_submit"
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $transport,
            "--requests", $requestsPath,
            "--responses", $responsesPath,
            "--work-dir", $workDirectory,
            "--verify-only"
        ) -Phase "validation_open_judge_verify"
        Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
            "attach-judgments", "--kind", "open",
            "--input", $generationsPath,
            "--responses", $responsesPath,
            "--output", $scoredPath
        ) -Phase "attach_validation_open_judgments"
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $orchestrator, "partition-scored",
            "--repo-root", $RepositoryRoot,
            "--plan", $planPath,
            "--scored", $scoredPath
        ) -Phase "partition_validation_open_judgments"
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $openChainVerifier,
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--plan", $planPath,
            "--combined", $generationsPath,
            "--requests", $requestsPath,
            "--responses", $responsesPath,
            "--scored", $scoredPath
        ) -Phase "verify_validation_open_end_to_end_chain"
    }

    Write-CompletionStatus -State "building_final_summaries"
    Invoke-NativeLogged -Executable "powershell.exe" -Arguments @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $summaryBuilder,
        "-RepositoryRoot", $RepositoryRoot
    ) -Phase "build_16_final_calibration_summaries"
    $summaryStatusPath = Join-Path $artifactRoot "final_summary_status.json"
    $summaryStatus = Get-Content -Raw -LiteralPath $summaryStatusPath | ConvertFrom-Json
    if ($summaryStatus.state -ne "complete" -or $summaryStatus.summary_count -ne 16) {
        throw "final calibration summaries did not complete exact 16-summary coverage"
    }
    $estimate = if (Test-Path -LiteralPath (Join-Path $workDirectory "cost_preflight.json")) {
        Get-Content -Raw -LiteralPath (Join-Path $workDirectory "cost_preflight.json") |
            ConvertFrom-Json
    }
    else {
        $null
    }
    Write-CompletionStatus -State "complete" -Detail (
        "setups=$($plan.setup_count);scored_rows=$($plan.setup_count * 96);" +
        "safe_upper_bound_usd=$(if($null -eq $estimate){'0'}else{$estimate.safe_upper_bound_usd});" +
        "user_ceiling_usd=$(Convert-Invariant $MaxCostUsd);final_summaries=16"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-CompletionStatus -State "failed" -Detail $failureDetail
    throw
}
finally {
    if ($keepAwakeActive) {
        [SpLenseValidationOpenIntegrity]::RestorePowerPolicy()
    }
}
