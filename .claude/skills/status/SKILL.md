# /status — 查看訓練與模型狀態

## 用法
```
/status
/status 2330
```

## 說明
顯示所有已訓練模型的狀態，或查詢特定股票的模型資訊。

## 執行步驟

### 無參數：顯示全部模型狀態
1. 列出所有 `ppo_*_improved.zip` 檔案：
   ```powershell
   Get-ChildItem "C:\Users\Silvi\Projects\trading-bot" -Filter "ppo_*_improved.zip" |
     Sort-Object LastWriteTime -Descending |
     Select-Object Name, LastWriteTime, @{N='Size(KB)';E={[math]::Round($_.Length/1KB)}}
   ```
2. 統計：
   - 總模型數量
   - 最近訓練時間
   - 模型檔案總大小

3. 對照 `run_all_local_tw.py` 的 SIGNAL_SCRIPTS 清單，找出**還沒有模型**的股票並列出

### 有參數（如 `/status 2330`）：
1. 尋找 `ppo_2330*improved.zip`
2. 顯示：
   - 模型存在 ✅ / 不存在 ❌
   - 訓練時間
   - 準確度（從 `ModelAccuracyTracker` 讀取，如果有記錄）
   - 是否有對應信號腳本 `get_trading_signal_2330.py`
