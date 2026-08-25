$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$secureApiKey = Read-Host "Paste the OpenAI API key" -AsSecureString
$secretPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureApiKey)
try {
    $plainApiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($secretPointer)
    if ([string]::IsNullOrWhiteSpace($plainApiKey)) {
        throw "The API key was empty"
    }
    [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $plainApiKey, "User")
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretPointer)
    if (Get-Variable -Name plainApiKey -ErrorAction SilentlyContinue) {
        Remove-Variable -Name plainApiKey -Force
    }
    Remove-Variable -Name secureApiKey -Force
}

Write-Host "OPENAI_API_KEY is stored in the Windows user environment. The value was not printed."
