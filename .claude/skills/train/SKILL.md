# /train — 訓練新股票 AI 模型

## 用法
```
/train <股票代碼>
/train <代碼1> <代碼2> <代碼3>
```

## 範例
```
/train 6426.TW
/train 3131.TWO
/train 6426.TW 3189.TW 3131.TWO
```

## 說明
使用 PPO 強化學習訓練指定股票的 AI 交易模型。支援批次訓練多支股票。

## 執行步驟

1. **驗證股票代碼**：先確認 yfinance 可以取得資料
   ```powershell
   cd "C:\Users\Silvi\Projects\trading-bot"
   .\.venv\Scripts\python.exe -c "
   import yfinance as yf
   df = yf.download('<代碼>', period='5d', progress=False)
   print(len(df), 'rows')
   "
   ```
   - 若回傳 0 rows → 告知使用者代碼可能有誤或已下市，建議試試 .TWO 或 .TW

2. **檢查是否已有模型**：
   - 找 `ppo_<代碼小寫去除點>_*_improved.zip`
   - 如果已存在 → 詢問是否要重新訓練

3. **在背景執行訓練**（run_in_background: true）：
   ```powershell
   cd "C:\Users\Silvi\Projects\trading-bot"
   .\.venv\Scripts\python.exe _train_both_models_tw.py <代碼1> <代碼2>...
   ```

4. 告知使用者：訓練已在背景執行，預計 15~30 分鐘完成，完成後可用 `/tw-signal <代碼>` 查看信號
