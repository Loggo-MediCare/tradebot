Set-Location "C:\Users\Silvi\Projects\trading-bot"
.\.venv\Scripts\python.exe _train_both_models_tw.py 2357.TW 2379.TW 3013.TW 3081.TW 3413.TW 1717.TW 1815.TW 3450.TW 4958.TW 5483.TW *>> train_tw_batchA_output.txt
Write-Output "*** TW batch A done ***" >> train_tw_batchA_output.txt
