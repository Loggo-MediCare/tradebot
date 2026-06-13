# /tw-signal — 台股 AI 交易信號

## 用法
```
/tw-signal <股票代碼>
```

## 範例
```
/tw-signal 2330
/tw-signal 6147
/tw-signal 4979
```

## 說明
為指定台股生成 AI 交易信號。自動判斷 .TW 或 .TWO 後綴。

## 執行步驟

1. 解析使用者輸入的股票代碼（去除 .TW/.TWO 後綴，只取數字部分）
2. 在 `C:\Users\Silvi\Projects\trading-bot\` 尋找對應的信號腳本：
   - 優先找 `get_trading_signal_<代碼>.py`
3. 如果腳本存在：用 PowerShell 執行：
   ```powershell
   cd "C:\Users\Silvi\Projects\trading-bot"
   .\.venv\Scripts\python.exe get_trading_signal_<代碼>.py
   ```
4. 如果腳本不存在：告知使用者該股票尚未訓練，建議使用 `/train <代碼>.TW`
5. 輸出結果，重點顯示：價格、信號（買入/賣出/觀望）、AI強度、成交金額
