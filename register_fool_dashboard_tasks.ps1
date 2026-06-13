param(
    [string]$TaskPrefix = 'FoolDashboard',
    [string]$TwMorningTime = '08:45',
    [string]$TwAfternoonTime = '14:40',
    [string]$UsOpenDstCandidateTime = '21:35',
    [string]$UsOpenStandardCandidateTime = '22:35',
    [int]$Workers = 8,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$Root = $PSScriptRoot
$Runner = Join-Path $Root 'run_fool_dashboard_scheduled.ps1'

if (-not (Test-Path $Runner)) {
    throw "Cannot find $Runner"
}

$Weekdays = @('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
$PowerShellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

function New-FoolDashboardTask {
    param(
        [string]$Name,
        [string]$At,
        [string]$Market,
        [switch]$RequireUsOpenWindow,
        [string]$Description
    )

    $runnerArgs = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', ('"{0}"' -f $Runner),
        '-Market', $Market,
        '-Workers', $Workers
    )

    if ($RequireUsOpenWindow) {
        $runnerArgs += '-RequireUsOpenWindow'
    }

    $taskName = "{0}_{1}" -f $TaskPrefix, $Name
    $argument = $runnerArgs -join ' '

    if ($DryRun) {
        Write-Host ("DRY RUN: {0}" -f $taskName)
        Write-Host ("  Time: {0} local Windows time, weekdays" -f $At)
        Write-Host ("  Action: {0} {1}" -f $PowerShellExe, $argument)
        Write-Host ("  Description: {0}" -f $Description)
        return
    }

    $action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $argument -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Weekdays -At $At
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)
    $principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel LeastPrivilege

    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description $Description `
        -Force | Out-Null

    Write-Host ("Registered: {0} at {1}" -f $taskName, $At)
}

New-FoolDashboardTask `
    -Name 'TW_Morning' `
    -At $TwMorningTime `
    -Market 'tw' `
    -Description 'Run fool_dashboard.py --tw before Taiwan regular trading.'

New-FoolDashboardTask `
    -Name 'TW_Afternoon' `
    -At $TwAfternoonTime `
    -Market 'tw' `
    -Description 'Run fool_dashboard.py --tw after Taiwan regular and fixed-price trading.'

New-FoolDashboardTask `
    -Name 'US_Open_2135' `
    -At $UsOpenDstCandidateTime `
    -Market 'us' `
    -RequireUsOpenWindow `
    -Description 'Run fool_dashboard.py --us near US market open when Taipei time is 21:35. Runner skips outside 09:30 New York window.'

New-FoolDashboardTask `
    -Name 'US_Open_2235' `
    -At $UsOpenStandardCandidateTime `
    -Market 'us' `
    -RequireUsOpenWindow `
    -Description 'Run fool_dashboard.py --us near US market open when Taipei time is 22:35. Runner skips outside 09:30 New York window.'

if (-not $DryRun) {
    Write-Host ''
    Write-Host 'Done. Logs will be written under logs\fool_dashboard.'
    Write-Host ("Task prefix: {0}" -f $TaskPrefix)
}
