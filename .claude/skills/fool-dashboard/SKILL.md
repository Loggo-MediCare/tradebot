# /fool-dashboard — 傻瓜儀表板

## 用法
```
/fool-dashboard
/fool-dashboard --tw
/fool-dashboard --us
```

## 範例
```
/fool-dashboard          ← 全部掃描（台股 + 美股）
/fool-dashboard --tw     ← 只掃台股（~230支）
/fool-dashboard --us     ← 只掃美股（35支半導體/科技）
```

## 說明
執行 `fool_dashboard.py`，掃描所有股票的日/週/月 MACD 結構，每天只看三件事：

| 區塊 | 訊號 | 操作 |
|------|------|------|
| 🟢 完美多頭 / ✅ 強勢整理 | 日週月線全部正向 | 持有 / 可買 |
| 📍 MACD 收腳 + 跳空 | 日線由負轉折+跳空缺口 | 最佳進場時機 |
| 🔴 出場警示 | 週線或月線 MACD 跌破零 | 準備減碼出場 |
| 🔵 日線回調週月仍多頭 | 日線轉負但週月仍正 | 逢低分批買 |

## 執行步驟

1. **判斷使用者要掃哪個市場**：
   - 無參數 → 全部（台股＋美股）
   - `--tw` → 只掃台股
   - `--us` → 只掃美股

2. **在背景執行**（掃描約需 2~5 分鐘）：
   ```powershell
   cd "C:\Users\Silvi\Projects\trading-bot"
   .\.venv\Scripts\python.exe fool_dashboard.py --tw
   # 或
   .\.venv\Scripts\python.exe fool_dashboard.py --us
   # 或（全部）
   .\.venv\Scripts\python.exe fool_dashboard.py
   ```

3. **輸出解讀**：重點顯示以下區塊：
   - **重點股票狀態**（NVDA、AMD、TSM、台積電、鴻海等常駐顯示）
   - **🟢 完美多頭 / ✅ 強勢整理**：日週月三線全正 → 核心持倉，不動
   - **📍 MACD 收腳 + 跳空**（最重要！）：今日最佳進場候選
   - **🔴 出場警示**：週/月線翻負 → 開始減碼計畫

4. **進階選項**（如使用者要求）：
   ```powershell
   # 顯示被跳過的股票（debug用）
   .\.venv\Scripts\python.exe fool_dashboard.py --tw --debug-missing

   # 清除快取強制重新下載
   .\.venv\Scripts\python.exe fool_dashboard.py --no-cache

   # 調整並發數（電腦慢時降低）
   .\.venv\Scripts\python.exe fool_dashboard.py --workers 4 --max-downloads 1
   ```

## 傻瓜投資三原則（輸出末尾顯示）
```
1. 🟢/✅ 在列 → 繼續持有，什麼都不做
2. 📍 收腳+跳空 → 這是最好的買入時機
3. 🔴 出場警示 → 週/月線轉負才是真正出場
4. 🔵 日線回調週月仍多頭 → 可逢低分批買
```

## 技術細節
- 資料快取：`.cache_yfinance/`（預設 8 小時 TTL），避免重複下載
- 台股使用 3年 歷史資料（period='3y'），美股 2年（period='2y'）
- 多線並發下載（預設 8 workers，限制 2 concurrent downloads 避免被 Yahoo 封鎖）
- 公司名稱快取：`.cache_yfinance_names/`（30天 TTL）
