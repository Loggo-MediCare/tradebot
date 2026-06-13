# Batch 2: wait for batch 1 output file to contain "All done", then start
$batch1Output = "C:\Users\Silvi\AppData\Local\Temp\claude\c--Users-Silvi-Projects-trading-bot\e4f38047-e468-43a5-b025-fce6c40bebeb\tasks\bnft9go9k.output"
$projectDir   = "C:\Users\Silvi\Projects\trading-bot"
$python       = "$projectDir\.venv\Scripts\python.exe"

Write-Host "$(Get-Date -Format 'HH:mm:ss')  等待第一批訓練完成..." -ForegroundColor Yellow

while ($true) {
    Start-Sleep -Seconds 60
    if (Test-Path $batch1Output) {
        $content = Get-Content $batch1Output -Raw -ErrorAction SilentlyContinue
        if ($content -match '\*\*\* All done \*\*\*') {
            Write-Host "$(Get-Date -Format 'HH:mm:ss')  第一批完成！啟動第二批..." -ForegroundColor Green
            break
        }
    }
    $line = (Get-Content $batch1Output -ErrorAction SilentlyContinue | Select-String "Training|DONE|FAILED" | Select-Object -Last 1)
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  第一批進行中: $line" -ForegroundColor Cyan
}

# Batch 2 tickers (3026 already in batch 1, skipped)
Set-Location $projectDir
& $python _train_both_models_tw.py `
    6515.TW 2454.TW 2383.TW 5274.TWO 6274.TWO `
    2368.TW 2308.TW 6223.TWO 3653.TW 2330.TW `
    6187.TWO 3017.TW 2360.TW 3081.TWO 3363.TWO

Write-Host "$(Get-Date -Format 'HH:mm:ss')  *** 第二批全部完成 ***" -ForegroundColor Green
