# /chart-visualizer — K線圖表視覺化

## 用法
```
/chart-visualizer <股票代碼>
/chart-visualizer <代碼> --period <期間>
```

## 範例
```
/chart-visualizer 2330
/chart-visualizer AMD
/chart-visualizer 2454 --period 6mo
/chart-visualizer 2317 --period 1y
```

## 說明
使用 `chart_visualizer.py` 的 `plot_candlestick()` 和 `SmartMoneyBot` 為指定股票繪製：
- K棒圖（含均線 SMA10/SMA30/SMA50）
- 爆量/真假突破標記
- 型態區域高亮（杯柄、W底、頭肩等）
- 籌碼掃描結果

## 執行步驟

1. **下載股價資料**：
   ```python
   import yfinance as yf
   from chart_visualizer import plot_candlestick, SmartMoneyBot
   
   # 台股加 .TW/.TWO，美股直接代碼
   ticker = '2330.TW'   # 或 'AMD'
   df = yf.Ticker(ticker).history(period='6mo')
   df.columns = [c.lower() for c in df.columns]
   ```

2. **繪製並儲存圖表**：
   ```python
   save_path = f"{代碼}_chart_viz.png"
   plot_candlestick(df, ticker, save_path=save_path)
   print(f"圖表已儲存: {save_path}")
   ```

3. **若要執行籌碼掃描**（SmartMoneyBot）：
   ```python
   bot = SmartMoneyBot()
   result = bot.scan(ticker)
   print(result)
   ```

4. 實際執行：用 PowerShell 在背景跑：
   ```powershell
   cd "C:\Users\Silvi\Projects\trading-bot"
   .\.venv\Scripts\python.exe -c "
   import yfinance as yf
   from chart_visualizer import plot_candlestick
   df = yf.Ticker('<代碼>').history(period='6mo')
   df.columns = [c.lower() for c in df.columns]
   plot_candlestick(df, '<代碼>', save_path='<代碼>_viz.png')
   print('圖表已儲存')
   "
   ```

5. 告知使用者圖表路徑，可在 VS Code 內直接預覽

## 注意
- Windows 環境下使用 Agg 後端（無 GUI），圖表輸出為 PNG 檔案
- 中文字型使用 Microsoft JhengHei（微軟正黑體）
- 期間建議：短線用 `3mo`，波段用 `6mo`，長線用 `1y`
