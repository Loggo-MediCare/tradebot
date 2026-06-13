$py = "C:\Users\Silvi\Projects\trading-bot\.venv\Scripts\python.exe"
$base = "C:\Users\Silvi\Projects\trading-bot"

$stocks = @(
    '1342','2241','2483','4542','6472','6585','7788','8454',
    '1773','2340','3163','3264','3265','3581','3587',
    '2892','00981A','3498','6257'
)

foreach ($t in $stocks) {
    $zip = Get-ChildItem "$base\ppo_${t}*.zip" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($zip) {
        Write-Host "[$t] already done -> $($zip.Name), skipping"
        continue
    }
    $script = "$base\train_${t}_taiwan_improved.py"
    if (-not (Test-Path $script)) {
        Write-Host "[$t] no train script, skipping"
        continue
    }
    Write-Host "[$t] starting..." -NoNewline
    $p = Start-Process -FilePath $py -ArgumentList $script `
        -RedirectStandardOutput "$base\train_${t}_output.txt" `
        -RedirectStandardError  "$base\train_${t}_err.txt" `
        -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    $zip2 = Get-ChildItem "$base\ppo_${t}*.zip" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($zip2) {
        Write-Host " done -> $($zip2.Name)"
    } else {
        Write-Host " FAILED (check train_${t}_err.txt)"
    }
}
Write-Host "All sequential training complete."
