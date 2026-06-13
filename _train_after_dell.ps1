# Wait for DELL training to finish, then train IBM SMCI ORCL
Write-Output "Waiting for DELL training to complete..."
while ($true) {
    $content = Get-Content train_dell_output.txt -Encoding Unicode -ErrorAction SilentlyContinue | Select-Object -Last 3
    if ($content -match "All US training done") {
        Write-Output "DELL done! Starting IBM SMCI ORCL..."
        break
    }
    if (!(Get-Process -Name python -ErrorAction SilentlyContinue)) {
        Write-Output "No python process — assuming DELL done"
        break
    }
    Start-Sleep -Seconds 30
}
.\.venv\Scripts\python.exe _train_both_models_us.py IBM SMCI ORCL *>> train_ibm_smci_orcl_output.txt
Write-Output "*** IBM SMCI ORCL training done ***"
