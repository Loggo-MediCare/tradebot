# /fundamental — 財務基本面分析 + DCF 估值

## 用法
```
/fundamental <股票代碼>
```

## 範例
```
/fundamental 2330
/fundamental AMD
/fundamental 2454
```

## 說明
抓取股票的財務報表資料，進行基本面分析並估算 DCF 內在價值，判斷目前股價是否合理。

## 執行步驟

1. **抓取財務資料**（使用 Python + yfinance）：
   ```python
   import yfinance as yf
   ticker = yf.Ticker('<代碼>.TW')  # 台股加 .TW，美股不加
   info = ticker.info
   financials = ticker.financials          # 損益表（年度）
   balance_sheet = ticker.balance_sheet   # 資產負債表
   cashflow = ticker.cashflow             # 現金流量表
   ```

2. **關鍵指標分析**：
   | 指標 | 數值 | 評估 |
   |------|------|------|
   | P/E 本益比 | x.xx | 高/低於產業平均 |
   | P/B 股價淨值比 | x.xx | 是否溢價 |
   | ROE 股東權益報酬率 | xx% | >15% 良好 |
   | 毛利率 | xx% | 趨勢分析 |
   | 自由現金流 | NT$xxx億 | 正/負 |
   | 負債比率 | xx% | <50% 健康 |
   | 殖利率 | xx% | 配息穩定性 |

3. **DCF 簡易估值**：
   - 取近 3 年平均自由現金流
   - 假設成長率：高成長股 15%，穩定股 8%，衰退股 3%
   - 折現率（WACC）：10%
   - 計算 10 年 DCF + 終值
   - 輸出：內在價值 vs 當前市價 → 高估/低估百分比

4. **輸出格式**：
   ```
   📊 <股票> 基本面分析
   ════════════════════
   當前價格: NT$xxx  |  內在價值: NT$xxx  |  安全邊際: +/-xx%
   
   💰 獲利能力         📈 成長性           🏦 財務健康
   ROE: xx%           營收成長: +xx%      負債比: xx%
   毛利率: xx%         EPS成長: +xx%      流動比: x.x
   淨利率: xx%         FCF成長: +xx%      利息保障: xx倍
   
   🎯 DCF 估值結論: [低估/合理/高估]
   ```
