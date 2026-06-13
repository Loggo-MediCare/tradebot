$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop 'RunBot.lnk'
if (-not (Test-Path $lnkPath)) {
    Write-Error "Shortcut not found: $lnkPath"
    exit 1
}
$wsh = New-Object -ComObject WScript.Shell
$s = $wsh.CreateShortcut($lnkPath)
$s.Hotkey = 'Ctrl+Alt+R'
$s.Save()
Write-Output "Set hotkey 'Ctrl+Alt+R' for: $lnkPath"