Set-Location "C:\Users\Silvi\Projects\trading-bot"
.\.venv\Scripts\python.exe _train_both_models_tw.py 6147.TW 6196.TW 6223.TW 6274.TW 6438.TW 6488.TW 6640.TW 6664.TW 8028.TW 8358.TW *>> train_tw_batchB_output.txt
Write-Output "*** TW batch B done ***" >> train_tw_batchB_output.txt
