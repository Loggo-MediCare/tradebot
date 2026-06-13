# run_all_local_tw_to_file.ps1

$ErrorActionPreference = "Stop"

# 啟用自動掃描訊號腳本
$env:AUTO_SCAN_SIGNAL = "1"

#（可選）指定 Python 路徑，避免用錯環境
# $python = "C:\Users\Silvi\Projects\trading-bot\.venv\Scripts\python.exe"
# & $python run_all_local_tw_fixed.py

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot "run_all_local_tw_to_excel.py"
# 生成包含日期和时间的文件名 (格式: taiwan_signals_output_202601061900.txt)
$timestamp = Get-Date -Format "yyyyMMddHHmm"
$outputFile = Join-Path $PSScriptRoot "taiwan_signals_output_$timestamp.txt"

Write-Host "Running Taiwan stock signals analysis..."
Write-Host "Output will be saved to: $outputFile"
Write-Host ""

# Run Python script and capture output with UTF-8 encoding
# Use Out-File with UTF8 encoding instead of Tee-Object -Encoding (not available in older PS versions)
& $pythonPath $scriptPath | ForEach-Object {
    Write-Host $_
    $_
} | Out-File -FilePath $outputFile -Encoding UTF8

Write-Host ""
Write-Host "Complete! Output saved to: $outputFile"

