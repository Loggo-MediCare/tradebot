Set-Location "C:\Users\Silvi\Projects\trading-bot"
.\.venv\Scripts\python.exe _train_both_models_us.py DXCM ZS CTSH CRWD MRVL ROST HPQ SWKS NTAP RL WSM *>> train_retry_us2_output.txt
Write-Output "*** US retry2 done ***" >> train_retry_us2_output.txt
