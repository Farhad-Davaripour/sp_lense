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
$sealedRoot = Join-Path $artifactRoot "sealed"
$transport = Join-Path $artifactRoot "submit_openai_judge_requests.py"
$orchestrator = Join-Path $artifactRoot "locked_open_orchestration.py"
$openChainVerifier = Join-Path $artifactRoot "verify_open_judgment_chain.py"
$lockPath = Join-Path $RepositoryRoot "configs\steering_comparison_lock.json"
$stage2Manifest = Join-Path $RepositoryRoot "configs\steering_comparison_stage2_lock.json"
$planPath = Join-Path $sealedRoot "sealed_evaluation_plan.json"
$generationsPath = Join-Path $sealedRoot "open_generations_all.jsonl"
$requestsPath = Join-Path $sealedRoot "open_judge_requests.jsonl"
$responsesPath = Join-Path $sealedRoot "open_judge_responses.jsonl"
$scoredPath = Join-Path $sealedRoot "open_scored_all.jsonl"
$workDirectory = Join-Path $sealedRoot "judge_transport"
$statusPath = Join-Path $artifactRoot "sealed_judgment_status.json"
$logPath = Join-Path $artifactRoot "sealed_judgment.log"
$lockedInputPricePerMillion = 0.40
$lockedOutputPricePerMillion = 1.60

function Convert-Invariant {
    param([double]$Value)
    return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Write-SealedJudgeStatus {
    param(
        [string]$State,
        [string]$Detail = ""
    )
    [ordered]@{
        schema_version = 1
        split = "sealed_test"
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
        split = "sealed_test"
        explicit_positive_phase_cap_required = $true
        locked_input_price_per_million_usd = $lockedInputPricePerMillion
        locked_output_price_per_million_usd = $lockedOutputPricePerMillion
        paid_calls = 0
        plan_regeneration_verification_required = $true
        combined_generation_regeneration_verification_required = $true
        request_regeneration_verification_required = $true
        post_submit_transport_verification_required = $true
        keep_awake_with_guaranteed_restore = $true
        random_open_judgments = $false
        exact_partitioning = $true
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

public static class SpLenseSealedOpenIntegrity
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
    if (-not [SpLenseSealedOpenIntegrity]::FilesEqual($ExpectedPath, $RebuiltPath)) {
        throw $FailureMessage
    }
}

$keepAwakeActive = $false
try {
    if (-not [SpLenseSealedOpenIntegrity]::KeepAwake()) {
        throw "Windows refused the sealed-open keep-awake request"
    }
    $keepAwakeActive = $true
    Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
        "verify-stage2",
        "--stage2-manifest", $stage2Manifest
    ) -Phase "verify_stage2_before_sealed_judging"
    foreach ($required in @(
        $lockPath, $stage2Manifest, $planPath, $transport, $orchestrator, $openChainVerifier
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "sealed-judgment prerequisite is missing: $required"
        }
    }

    $rebuiltPlan = Join-Path $sealedRoot "sealed_evaluation_plan.rebuilt.tmp"
    try {
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $orchestrator, "plan",
            "--repo-root", $RepositoryRoot,
            "--lock", $lockPath,
            "--manifest", $stage2Manifest,
            "--output-dir", $sealedRoot,
            "--split", "sealed_test",
            "--output", $rebuiltPlan
        ) -Phase "regenerate_sealed_open_plan"
        Assert-ExactFileMatch -ExpectedPath $planPath -RebuiltPath $rebuiltPlan `
            -FailureMessage "sealed-open plan differs byte-for-byte from locked regeneration"
    }
    finally {
        Remove-Item -LiteralPath $rebuiltPlan -Force -ErrorAction SilentlyContinue
    }

    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    if ($plan.split -ne "sealed_test" -or $plan.setup_count -ne @($plan.setups).Count) {
        throw "sealed setup plan is inconsistent"
    }
    $openSetups = @($plan.setups | Where-Object { $_.open_required -eq $true })
    if ($openSetups.Count -eq 0) {
        Write-SealedJudgeStatus -State "complete" -Detail "no sealed open setup was eligible"
        return
    }
    foreach ($required in @($generationsPath, $requestsPath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "sealed open generated input is missing: $required"
        }
    }

    $rebuiltGenerations = Join-Path $sealedRoot "open_generations_all.rebuilt.tmp"
    try {
        Invoke-NativeLogged -Executable $pythonExe -Arguments @(
            $orchestrator, "combine-generations",
            "--repo-root", $RepositoryRoot,
            "--plan", $planPath,
            "--output", $rebuiltGenerations
        ) -Phase "regenerate_sealed_open_combined_generations"
        Assert-ExactFileMatch -ExpectedPath $generationsPath `
            -RebuiltPath $rebuiltGenerations `
            -FailureMessage (
                "sealed-open combined generations differ byte-for-byte " +
                "from locked regeneration"
            )
    }
    finally {
        Remove-Item -LiteralPath $rebuiltGenerations -Force -ErrorAction SilentlyContinue
    }

    $rebuiltRequests = Join-Path $sealedRoot "open_judge_requests.rebuilt.tmp"
    try {
        Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
            "judge-requests", "--kind", "open",
            "--input", $generationsPath,
            "--output", $rebuiltRequests
        ) -Phase "regenerate_sealed_open_judge_requests"
        Assert-ExactFileMatch -ExpectedPath $requestsPath -RebuiltPath $rebuiltRequests `
            -FailureMessage (
                "sealed open judge requests differ byte-for-byte from locked regeneration"
            )
    }
    finally {
        Remove-Item -LiteralPath $rebuiltRequests -Force -ErrorAction SilentlyContinue
    }

    Write-SealedJudgeStatus -State "judge_preflight"
    Invoke-NativeLogged -Executable $pythonExe -Arguments @(
        $transport,
        "--requests", $requestsPath,
        "--responses", $responsesPath,
        "--work-dir", $workDirectory,
        "--max-cost-usd", (Convert-Invariant $MaxCostUsd),
        "--input-price-per-million", (Convert-Invariant $InputPricePerMillion),
        "--output-price-per-million", (Convert-Invariant $OutputPricePerMillion),
        "--dry-run"
    ) -Phase "sealed_open_judge_preflight"
    Import-OpenAIKeyForChildProcess

    Write-SealedJudgeStatus -State "judging"
    Invoke-NativeLogged -Executable $pythonExe -Arguments @(
        $transport,
        "--requests", $requestsPath,
        "--responses", $responsesPath,
        "--work-dir", $workDirectory,
        "--max-cost-usd", (Convert-Invariant $MaxCostUsd),
        "--input-price-per-million", (Convert-Invariant $InputPricePerMillion),
        "--output-price-per-million", (Convert-Invariant $OutputPricePerMillion)
    ) -Phase "sealed_open_judge_submit"
    Invoke-NativeLogged -Executable $pythonExe -Arguments @(
        $transport,
        "--requests", $requestsPath,
        "--responses", $responsesPath,
        "--work-dir", $workDirectory,
        "--verify-only"
    ) -Phase "sealed_open_judge_verify"
    Invoke-NativeLogged -Executable $comparisonExe -Arguments @(
        "attach-judgments", "--kind", "open",
        "--input", $generationsPath,
        "--responses", $responsesPath,
        "--output", $scoredPath
    ) -Phase "attach_sealed_open_judgments"
    Invoke-NativeLogged -Executable $pythonExe -Arguments @(
        $orchestrator, "partition-scored",
        "--repo-root", $RepositoryRoot,
        "--plan", $planPath,
        "--scored", $scoredPath
    ) -Phase "partition_sealed_open_judgments"
    Invoke-NativeLogged -Executable $pythonExe -Arguments @(
        $openChainVerifier,
        "--repo-root", $RepositoryRoot,
        "--lock", $lockPath,
        "--plan", $planPath,
        "--combined", $generationsPath,
        "--requests", $requestsPath,
        "--responses", $responsesPath,
        "--scored", $scoredPath
    ) -Phase "verify_sealed_open_end_to_end_chain"

    $estimate = Get-Content -Raw -LiteralPath (
        Join-Path $workDirectory "cost_preflight.json"
    ) | ConvertFrom-Json
    Write-SealedJudgeStatus -State "complete" -Detail (
        "open_setups=$($openSetups.Count);scored_rows=$($openSetups.Count * 96);" +
        "safe_upper_bound_usd=$($estimate.safe_upper_bound_usd);" +
        "user_ceiling_usd=$(Convert-Invariant $MaxCostUsd)"
    )
}
catch {
    $failureDetail = ($_ | Out-String).Trim()
    Write-SealedJudgeStatus -State "failed" -Detail $failureDetail
    throw
}
finally {
    if ($keepAwakeActive) {
        [SpLenseSealedOpenIntegrity]::RestorePowerPolicy()
    }
}
