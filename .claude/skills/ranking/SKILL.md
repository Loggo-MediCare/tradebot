# /ranking — 成交金額排行榜

## 用法
```
/ranking
/ranking tw
/ranking us
```

## 說明
從最近一次產生的信號輸出檔案中，解析所有股票的「成交量 × 價格」並排序，顯示資金流向最大的股票排行榜。

## 執行步驟

1. **執行排行榜腳本**：
   ```powershell
   cd "C:\Users\Silvi\Projects\trading-bot"
   .\.venv\Scripts\python.exe _ranking.py
   ```

2. `_ranking.py` 會自動：
   - 尋找最新的 `taiwan_signals_output_*.txt` 輸出檔案
   - 解析每支股票的價格、成交量、信號
   - 計算 成交金額 = 價格 × 成交量
   - 依成交金額由大到小排序輸出

3. **若無輸出檔**：提示使用者先執行 `run_all_local_tw_to_file.ps1`

4. **輸出格式**：
   ```
   排名  股票              價格        成交量        成交金額      信號
     1   2454.TW          NT$4,545   16,310,968   NT$74.1B  🟢 強烈買入信號
     2   3481.TW            NT$49.2   1,273,705,485  NT$62.7B  🟢 強烈買入信號
   ...
   ```
   最後顯示所有🟢買入、🔴賣出信號股票清單

## 注意
- 解析檔案同時支援 V1 格式（`当前价格: NT$x`）和 V2 box 格式（`║  價格: $x`）
- 信號分類：買入含「買入/买入/BUY」，賣出含「賣出/卖出/SELL/看空」，其餘觀望
