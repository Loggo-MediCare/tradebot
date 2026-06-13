$sourceLnk = 'C:\Users\Silvi\Projects\trading-bot\today.lnk'
$desktop = [Environment]::GetFolderPath('Desktop')
$destLnk = Join-Path $desktop 'Today.lnk'

$wsh = New-Object -ComObject WScript.Shell
if (Test-Path $sourceLnk) {
    $src = $wsh.CreateShortcut($sourceLnk)
    $target = $src.TargetPath
    $wd = $src.WorkingDirectory
    $icon = $src.IconLocation
} else {
    $target = $sourceLnk
    $wd = 'C:\Users\Silvi\Projects\trading-bot'
    $icon = 'C:\Windows\System32\shell32.dll,0'
}

$s = $wsh.CreateShortcut($destLnk)
$s.TargetPath = $target
$s.WorkingDirectory = $wd
$s.IconLocation = $icon
$s.Hotkey = 'Ctrl+Alt+D'
$s.Save()
Write-Output "Created: $destLnk -> $target with hotkey Ctrl+Alt+D"