# PowerShell script to run trading signal with proper UTF-8 encoding
# Usage: .\run_signal_utf8.ps1 <stock_code>
# Example: .\run_signal_utf8.ps1 6285

param(
    [Parameter(Mandatory=$true)]
    [string]$StockCode
)

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# Set console to UTF-8 code page
chcp 65001 > $null

# Dynamically resolve Python path
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Path
if (-not $pythonPath) {
    Write-Host "Python executable not found. Please ensure Python is installed and added to PATH." -ForegroundColor Red
    exit 1
}

$signalFile = "get_trading_signal_$StockCode.py"

if (-not (Test-Path $signalFile)) {
    Write-Host "Signal file not found: $signalFile" -ForegroundColor Red
    exit 1
}

Write-Host "Running signal for $StockCode with UTF-8 encoding..."
Write-Host ""

# Run with UTF-8 environment
& $pythonPath $signalFile

Write-Host ""
Write-Host "Complete!"
