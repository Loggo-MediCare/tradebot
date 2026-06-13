param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('tw', 'us')]
    [string]$Market,

    [int]$Workers = 8,
    [int]$CacheTtlHours = 2,
    [int]$Retries = 3,
    [int]$MaxDownloads = 2,
    [int]$TimeoutSeconds = 20,

    [switch]$RequireUsOpenWindow,
    [int]$UsOpenWindowBeforeMinutes = 5,
    [int]$UsOpenWindowAfterMinutes = 45,

    [switch]$DebugMissing,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = $PSScriptRoot
$ScriptPath = Join-Path $Root 'fool_dashboard.py'
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'logs\fool_dashboard'

function Write-LogMessage {
    param(
        [string]$Path,
        [string]$Message
    )

    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    Add-Content -Path $Path -Value $line -Encoding UTF8
}

function Test-IsUsOpenWindow {
    param(
        [int]$BeforeMinutes,
        [int]$AfterMinutes
    )

    $eastern = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
    $nyNowOffset = [System.TimeZoneInfo]::ConvertTime([System.DateTimeOffset]::Now, $eastern)
    $nyNow = $nyNowOffset.DateTime

    if ($nyNow.DayOfWeek -eq [System.DayOfWeek]::Saturday -or $nyNow.DayOfWeek -eq [System.DayOfWeek]::Sunday) {
        return $false
    }

    $openTime = $nyNow.Date.AddHours(9).AddMinutes(30)
    $windowStart = $openTime.AddMinutes(-1 * [Math]::Max(0, $BeforeMinutes))
    $windowEnd = $openTime.AddMinutes([Math]::Max(0, $AfterMinutes))

    return ($nyNow -ge $windowStart -and $nyNow -le $windowEnd)
}

if (-not (Test-Path $ScriptPath)) {
    throw "Cannot find $ScriptPath"
}

if (Test-Path $VenvPython) {
    $PythonPath = $VenvPython
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw 'Cannot find .venv Python or python on PATH.'
    }
    $PythonPath = $cmd.Source
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogDir ("fool_dashboard_{0}_{1}.log" -f $Market, $timestamp)

Write-LogMessage -Path $LogFile -Message ("Starting Fool Dashboard scheduled run. Market={0}" -f $Market)
Write-LogMessage -Path $LogFile -Message ("Root={0}" -f $Root)
Write-LogMessage -Path $LogFile -Message ("Python={0}" -f $PythonPath)

if ($Market -eq 'us' -and $RequireUsOpenWindow -and (-not $DryRun)) {
    if (-not (Test-IsUsOpenWindow -BeforeMinutes $UsOpenWindowBeforeMinutes -AfterMinutes $UsOpenWindowAfterMinutes)) {
        Write-LogMessage -Path $LogFile -Message 'Skipped: current time is not inside the US market-open window.'
        exit 0
    }
}

$pyArgs = @()
if ($Market -eq 'tw') {
    $pyArgs += '--tw'
} else {
    $pyArgs += '--us'
}

$pyArgs += @(
    '--workers', $Workers,
    '--cache-ttl', $CacheTtlHours,
    '--retries', $Retries,
    '--max-downloads', $MaxDownloads,
    '--timeout', $TimeoutSeconds
)

if ($DebugMissing) {
    $pyArgs += '--debug-missing'
}

$commandPreview = '& "{0}" "{1}" {2}' -f $PythonPath, $ScriptPath, ($pyArgs -join ' ')
Write-LogMessage -Path $LogFile -Message ("Command={0}" -f $commandPreview)

if ($DryRun) {
    if ($Market -eq 'us' -and $RequireUsOpenWindow) {
        Write-LogMessage -Path $LogFile -Message 'DryRun note: actual scheduled runs will skip outside the US market-open window.'
    }
    Write-LogMessage -Path $LogFile -Message 'DryRun enabled. No Python command executed.'
    exit 0
}

Push-Location $Root
try {
    $output = & $PythonPath $ScriptPath @pyArgs 2>&1
    $exitCode = $LASTEXITCODE

    foreach ($line in $output) {
        Write-Host $line
    }
    $output | Out-File -FilePath $LogFile -Encoding UTF8 -Append

    Write-LogMessage -Path $LogFile -Message ("Finished. ExitCode={0}" -f $exitCode)
    exit $exitCode
} finally {
    Pop-Location
}
