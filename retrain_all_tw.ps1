# Retrain all Taiwan stock models sequentially
# Usage: .\retrain_all_tw.ps1

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pythonPath = "C:\Users\Silvi\Projects\trading-bot\.venv\Scripts\python.exe"
$projectDir = "C:\Users\Silvi\Projects\trading-bot"
Set-Location $projectDir

$timestamp = Get-Date -Format "yyyyMMddHHmm"
$logFile = "$projectDir\retrain_log_$timestamp.txt"

$scripts = Get-ChildItem $projectDir -Filter "train_*_taiwan_improved.py" | Sort-Object Name
$total = $scripts.Count
$success = 0
$failed = @()
$i = 0

Write-Host "======================================"
Write-Host "Taiwan Stock Models - Batch Retrain"
Write-Host "======================================"
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Total scripts: $total"
Write-Host "Log: $logFile"
Write-Host "======================================"

"Retrain All TW - $timestamp | Total: $total" | Out-File $logFile -Encoding utf8

foreach ($script in $scripts) {
    $i++
    $name = $script.Name

    Write-Host ""
    Write-Host "[$i/$total] Training: $name"

    "[$i/$total] $name - $(Get-Date -Format 'HH:mm:ss')" | Out-File $logFile -Append -Encoding utf8

    try {
        & $pythonPath $script.FullName 2>&1 | Out-Null
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Host "  OK: $name"
            "  OK" | Out-File $logFile -Append -Encoding utf8
            $success++
        } else {
            Write-Host "  FAIL: $name (exit $exitCode)"
            "  FAIL (exit $exitCode)" | Out-File $logFile -Append -Encoding utf8
            $failed += $name
        }
    } catch {
        Write-Host "  ERROR: $_"
        "  ERROR: $_" | Out-File $logFile -Append -Encoding utf8
        $failed += $name
    }
}

Write-Host ""
Write-Host "======================================"
Write-Host "Done! Success: $success / $total  Failed: $($failed.Count)"
Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($failed.Count -gt 0) {
    Write-Host "Failed:"
    $failed | ForEach-Object { Write-Host "  - $_" }
}
Write-Host "======================================"

"Done: $success/$total  Failed: $($failed.Count)  End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $logFile -Append -Encoding utf8
if ($failed.Count -gt 0) {
    "Failed list:" | Out-File $logFile -Append -Encoding utf8
    $failed | ForEach-Object { "  - $_" | Out-File $logFile -Append -Encoding utf8 }
}
