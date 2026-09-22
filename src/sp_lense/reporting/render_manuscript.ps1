param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputPdf
)
$ErrorActionPreference = 'Stop'
$documentPath = (Resolve-Path -LiteralPath $InputDocx).Path
$pdfPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPdf)
if (-not $pdfPath.EndsWith('.pdf', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must be a PDF path'
}
$paperWord = New-Object -ComObject Word.Application
$paperWord.Visible = $false
$paperWord.DisplayAlerts = 0
$paperWord.AutomationSecurity = 3
$paperDocument = $null
try {
    $paperDocument = $paperWord.Documents.Open($documentPath, $false, $true, $false)
    $paperDocument.Repaginate()
    $paperDocument.Fields.Update() | Out-Null
    $paperDocument.ExportAsFixedFormat($pdfPath, 17)
    [PSCustomObject]@{ Pdf=$pdfPath; Pages=$paperDocument.ComputeStatistics(2); Renderer='Word '+$paperWord.Version } | ConvertTo-Json
} finally {
    if ($null -ne $paperDocument) { $paperDocument.Close(0) }
    $paperWord.Quit()
}
