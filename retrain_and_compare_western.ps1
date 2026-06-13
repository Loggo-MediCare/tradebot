# Retrain all Western models, rerun signals, then compare with previous output
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pythonPath = "C:\Users\Silvi\Projects\trading-bot\.venv\Scripts\python.exe"
$projectDir  = "C:\Users\Silvi\Projects\trading-bot"
Set-Location $projectDir

# ── Step 1: find the most recent old signal file ──────────────────────────────
$oldFile = Get-ChildItem $projectDir -Filter "US_signals_output_*.txt" |
           Sort-Object LastWriteTime -Descending |
           Select-Object -First 1 -ExpandProperty FullName

Write-Host "Old signals file: $oldFile"

# ── Step 2: retrain all Western models ───────────────────────────────────────
$trainScripts = Get-ChildItem $projectDir -Filter "train_*.py" |
    Where-Object { $_.Name -notmatch "taiwan" } |
    Where-Object { $_.Name -match "train_(aapl|aeva|alab|amd|amkr|apld|arm|avav|avgo|bkr|crdo|etn|gev|goog|htgc|intc|mchp|mpwr|mrvl|mu|nat|nvda|nxpi|oklo|omer|on|onds|orcl|oust|pltr|qcom|rhm|rklb|rnmby|sndk|snps|stld|tsla|tsm|txn|uri|vrk|wdc)_improved" } |
    Sort-Object Name

$total = $trainScripts.Count
$success = 0; $failed = @(); $i = 0

Write-Host ""
Write-Host "======================================"
Write-Host "Step 1: Retraining $total Western models"
Write-Host "======================================"

foreach ($script in $trainScripts) {
    $i++
    Write-Host "[$i/$total] $($script.Name)"
    & $pythonPath $script.FullName 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $success++; Write-Host "  OK" }
    else { $failed += $script.Name; Write-Host "  FAIL" }
}

Write-Host ""
Write-Host "Retrain done: $success/$total  Failed: $($failed.Count)"
if ($failed.Count -gt 0) { $failed | ForEach-Object { Write-Host "  FAIL: $_" } }

# ── Step 3: run signals and save to new file ──────────────────────────────────
Write-Host ""
Write-Host "======================================"
Write-Host "Step 2: Running Western signals"
Write-Host "======================================"

$timestamp = Get-Date -Format "yyyyMMddHHmm"
$newFile = "$projectDir\US_signals_output_$timestamp.txt"

& $pythonPath "$projectDir\run_all_western.py" | ForEach-Object {
    Write-Host $_; $_
} | Out-File -FilePath $newFile -Encoding UTF8

Write-Host ""
Write-Host "New signals file: $newFile"

# ── Step 4: compare ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "======================================"
Write-Host "Step 3: Comparing signals"
Write-Host "======================================"

& $pythonPath "$projectDir\compare_signals.py" $oldFile $newFile
