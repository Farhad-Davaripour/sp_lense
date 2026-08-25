[CmdletBinding()]
param(
    [ValidateSet(
        "Verify", "Plan", "Construct", "PrepareEvaluation", "VerifyPrepared",
        "EvaluateForced", "GenerateOpen", "JudgeRequests", "AttachJudgments", "Report"
    )]
    [string]$Action = "Verify",

    [ValidateSet("qwen35_08b", "qwen35_2b")]
    [string]$ModelTag,

    [ValidateSet("matched", "canonical")]
    [string]$Track,

    [string]$SetupId,

    [string]$Responses
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$constructionRunner = Join-Path $PSScriptRoot "run_sensitivity.py"
$evaluationRunner = Join-Path $PSScriptRoot "evaluate_sensitivity.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python environment not found: $python"
}

$runner = $constructionRunner
$command = $Action.ToLowerInvariant()
if ($Action -eq "Construct") {
    if (-not $ModelTag -or -not $Track) {
        throw "Construct requires both -ModelTag and -Track"
    }
    $arguments = @($runner, $command, "--model-tag", $ModelTag, "--track", $Track)
}
elseif ($Action -in @("Verify", "Plan")) {
    if ($ModelTag -or $Track -or $SetupId -or $Responses) {
        throw "Construction-only arguments are not accepted with -Action $Action"
    }
    $arguments = @($runner, $command)
}
else {
    if ($ModelTag -or $Track) {
        throw "-ModelTag and -Track are accepted only with -Action Construct"
    }
    $runner = $evaluationRunner
    $command = switch ($Action) {
        "PrepareEvaluation" { "prepare-evaluation" }
        "VerifyPrepared" { "verify-prepared" }
        "EvaluateForced" { "evaluate-forced" }
        "GenerateOpen" { "generate-open" }
        "JudgeRequests" { "judge-requests" }
        "AttachJudgments" { "attach-judgments" }
        "Report" { "report" }
    }
    $arguments = @($runner, $command)
    if ($Action -in @("EvaluateForced", "GenerateOpen")) {
        if (-not $SetupId) {
            throw "$Action requires -SetupId from the prepared secondary plan"
        }
        $arguments += @("--setup-id", $SetupId)
    }
    elseif ($SetupId) {
        throw "-SetupId is accepted only with EvaluateForced or GenerateOpen"
    }
    if ($Action -eq "AttachJudgments") {
        if (-not $Responses) {
            throw "AttachJudgments requires -Responses"
        }
        $arguments += @("--responses", $Responses)
    }
    elseif ($Responses) {
        throw "-Responses is accepted only with AttachJudgments"
    }
}

if ($Action -ne "Construct" -and $Action -notin @("Verify", "Plan") -and -not $arguments) {
    throw "Could not resolve evaluation action"
}
elseif ($Action -ne "Construct" -and $Action -notin @("Verify", "Plan") -and ($ModelTag -or $Track)) {
    throw "-ModelTag and -Track are accepted only with -Action Construct"
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "BiPO warmup sensitivity runner failed with exit code $LASTEXITCODE"
}
