# PowerShell script to run US/Western stock signals with proper UTF-8 encoding
# Usage: .\run_all_WESTERN_us_to_file.ps1
#
# Stock Coverage:
# ===============
# US Stocks: 60+ (NVDA, TSLA, AAPL, AMD, PLTR, SMCI, etc.)
# European: 2 (Rheinmetall)
# Hong Kong: 2 (Xiaomi, Vanke)
# Japan: 1 (SoftBank)
# South Korea: 2 (LIG Nex1, Hanwha Aerospace)
#
# Total: 67+ global stocks

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Use virtual environment Python
$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "Virtual environment not found at: $pythonPath" -ForegroundColor Red
    Write-Host "Please ensure the virtual environment is set up properly." -ForegroundColor Red
    exit 1
}

$scriptPath = Join-Path $PSScriptRoot "run_all_western.py"
# Generate output filename with timestamp (format: US_signals_output_202603031800.txt)
$timestamp = Get-Date -Format "yyyyMMddHHmm"
$outputFile = Join-Path $PSScriptRoot "US_signals_output_$timestamp.txt"

Write-Host "Running US stock signals analysis..."
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
