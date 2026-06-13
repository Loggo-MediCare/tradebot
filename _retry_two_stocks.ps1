Set-Location "C:\Users\Silvi\Projects\trading-bot"
.\.venv\Scripts\python.exe _train_both_models_tw.py 1815.TWO 5483.TWO 6147.TWO 6223.TWO 6274.TWO 8358.TWO 3081.TWO *>> train_retry_two_output.txt
Write-Output "*** TWO retry done ***" >> train_retry_two_output.txt
