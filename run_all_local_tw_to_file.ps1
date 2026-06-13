# PowerShell script to run Taiwan stock signals with proper UTF-8 encoding
# Usage: .\run_all_local_tw_to_file.ps1
#
# Latest XGBoost Model Stocks Added (2026-03-02):
# ================================================
# 3563.TW    牧德           60.28%   XGBoost
# 3576.TW    聯合再生       68.15%   XGBoost
# 3615.TWO   安可           67.54%   XGBoost
# 3665.TW    貿聯-KY        52.82%   XGBoost
# 4564.TW    元翎           65.32%   XGBoost
# 4577.TWO   達航科技       51.42%   XGBoost
# 4768.TWO   晶呈科技       50.97%   XGBoost
# 4989.TW    榮科           64.24%   XGBoost
# 4991.TWO   環宇-KY        53.83%   XGBoost
# 6220.TWO   岳豐           75.60%   XGBoost ⭐ EXCELLENT
# 6230.TW    尼得科超眾     65.93%   XGBoost
# 6442.TW    光聖           50.00%   XGBoost
# 6526.TW    達發           49.28%   XGBoost
# 6789.TW    采鈺           56.77%   XGBoost
# 6830.TW    汎銓           62.50%   XGBoost
# 6877.TWO   鏵友益         70.75%   XGBoost ⭐ EXCELLENT
# 8438.TW    昶昕           53.97%   XGBoost
# 8927.TWO   北基           67.74%   XGBoost
#
# Total: 120 Taiwan stocks (17 newly added)
# Top Performers: 6220.TWO (75.60%), 6877.TWO (70.75%)

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8


# Use virtual environment Python
$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "Virtual environment not found at: $pythonPath" -ForegroundColor Red
    Write-Host "Please ensure the virtual environment is set up properly." -ForegroundColor Red
    exit 1
}

$scriptPath = Join-Path $PSScriptRoot "run_all_local_tw_to_excel.py"
# 生成包含日期和时间的文件名 (格式: taiwan_signals_output_202601061900.txt)
$timestamp = Get-Date -Format "yyyyMMddHHmm"
$outputFile = Join-Path $PSScriptRoot "taiwan_signals_output_$timestamp.txt"

Write-Host "Running Taiwan stock signals analysis..."
Write-Host "Output will be saved to: $outputFile"
Write-Host ""

# Run Python script and capture output with UTF-8 encoding
# Use Out-File with UTF8 encoding instead of Tee-Object -Encoding (not available in older PS versions)
& $pythonPath $scriptPath --txt-only | ForEach-Object {
    Write-Host $_
    $_
} | Out-File -FilePath $outputFile -Encoding UTF8

Write-Host ""
Write-Host "Complete! Output saved to: $outputFile"
