$WshShell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop 'RunBot.lnk'
$s = $WshShell.CreateShortcut($lnk)
$s.TargetPath = 'C:\Users\Silvi\Projects\trading-bot\run_bot.bat'
$s.WorkingDirectory = 'C:\Users\Silvi\Projects\trading-bot'
$s.IconLocation = 'C:\Windows\System32\shell32.dll,1'
$s.Save()
Write-Output "Created: $lnk"
