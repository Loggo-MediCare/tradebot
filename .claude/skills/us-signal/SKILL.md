# /us-signal — 美股 AI 交易信號

## 用法
```
/us-signal <股票代碼>
```

## 範例
```
/us-signal AMD
/us-signal MU
/us-signal INTC
/us-signal QCOM
/us-signal NVDA
```

## 說明
為指定美股生成 AI 交易信號，使用已訓練的 PPO 模型 + FinBERT 情緒分析。

## 執行步驟

1. 將輸入代碼轉為大寫（例如 amd → AMD）
2. 在 `C:\Users\Silvi\Projects\trading-bot\` 尋找對應腳本：
   - `get_trading_signal_<小寫代碼>.py`
3. 如果腳本存在，用 PowerShell 執行：
   ```powershell
   cd "C:\Users\Silvi\Projects\trading-bot"
   .\.venv\Scripts\python.exe get_trading_signal_<代碼>.py
   ```
4. 如果不存在：告知使用者尚未訓練此美股模型
5. 重點顯示：
   - 當前價格（美元）
   - 信號：🟢買入 / 🔴賣出 / 🟡觀望
   - AI 模型強度（0~1）
   - 建議買入比例與止損價
   - 分析師評級與目標價
   - FinBERT 市場情緒分數
