#!/usr/bin/env python3
"""
MA50_slope和HTGC重要指標計算器（示例版本）
展示如何計算和解讀這些指標
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*80)
print("🎯 MA50_slope 計算方法詳解")
print("="*80)
print()

# 創建示例價格數據（模擬HTGC最近90天的價格）
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=90, freq='D')

# 模擬價格（基於$18.84附近波動）
base_price = 18.84
price_trend = np.linspace(-0.5, 0.3, 90)  # 輕微上升趨勢
noise = np.random.normal(0, 0.15, 90)  # 隨機波動
prices = pd.Series(base_price + price_trend + noise, index=dates)

print("📊 示例數據（最近10天價格）:")
print("-"*80)
for i, (date, price) in enumerate(prices.tail(10).items(), 1):
    print(f"Day {-10+i:3d}: {date.date()} - ${price:.2f}")
print()

# ==========================================
# 1. 計算MA50（50日移動平均線）
# ==========================================

print("="*80)
print("📈 步驟1: 計算MA50（50日移動平均線）")
print("="*80)
print()

ma50 = prices.rolling(window=50).mean()

print("MA50計算公式:")
print("  MA50[t] = (Price[t] + Price[t-1] + ... + Price[t-49]) / 50")
print()
print(f"最新5天的MA50:")
print("-"*80)
for date, ma_val in ma50.tail(5).items():
    price = prices[date]
    print(f"{date.date()}: ${ma_val:.2f} (當日價格: ${price:.2f})")
print()

latest_ma50 = ma50.iloc[-1]
latest_price = prices.iloc[-1]
print(f"✅ 當前MA50: ${latest_ma50:.2f}")
print(f"✅ 當前價格: ${latest_price:.2f}")
print()

# ==========================================
# 2. 計算MA50_slope（MA50斜率）
# ==========================================

print("="*80)
print("📐 步驟2: 計算MA50_slope（MA50的斜率）")
print("="*80)
print()

print("斜率計算方法:")
print("  使用最近N天的MA50值進行線性回歸")
print("  預設N=5天")
print()

# 使用最近5天的MA50計算斜率
slope_period = 5
recent_ma50 = ma50.tail(slope_period)

print(f"最近{slope_period}天的MA50值:")
print("-"*80)
for i, (date, ma_val) in enumerate(recent_ma50.items(), 1):
    print(f"Day {i}: {date.date()} - ${ma_val:.2f}")
print()

# 線性回歸計算斜率
x = np.arange(len(recent_ma50))
y = recent_ma50.values

# y = mx + b，其中m就是斜率
slope, intercept = np.polyfit(x, y, 1)

print("線性回歸公式: y = mx + b")
print(f"  斜率(m) = {slope:.6f}")
print(f"  截距(b) = {intercept:.6f}")
print()

# 計算斜率百分比
slope_pct = (slope / latest_ma50) * 100

print("斜率百分比計算:")
print(f"  slope_pct = (slope / current_ma50) × 100")
print(f"  slope_pct = ({slope:.6f} / {latest_ma50:.2f}) × 100")
print(f"  slope_pct = {slope_pct:.4f}%")
print()

print("="*80)
print(f"✅ MA50_slope = {slope:.6f}")
print(f"✅ MA50_slope_pct = {slope_pct:.4f}%")
print("="*80)
print()

# ==========================================
# 3. 解讀斜率
# ==========================================

print("="*80)
print("📊 步驟3: 解讀MA50_slope")
print("="*80)
print()

print("斜率判斷標準:")
print("-"*80)
print("  slope_pct > +0.2%  →  📈 強勢上升趨勢（強烈買入信號）")
print("  slope_pct > 0%     →  📈 溫和上升趨勢（謹慎看多）")
print("  slope_pct ≈ 0%     →  📊 盤整/橫盤（觀望）")
print("  slope_pct < -0.1%  →  📉 明顯下降趨勢（避免買入）")
print("  slope_pct < 0%     →  📉 溫和下降趨勢（謹慎看空）")
print()

# 給出判斷
if slope_pct > 0.2:
    trend = "📈 強勢上升趨勢"
    signal = "✅ 強烈買入信號"
    color = "🟢"
elif slope_pct > 0:
    trend = "📈 溫和上升趨勢"
    signal = "⚠️ 謹慎看多"
    color = "🟡"
elif slope_pct > -0.1:
    trend = "📊 盤整/橫盤"
    signal = "🟡 觀望"
    color = "🟡"
else:
    trend = "📉 下降趨勢"
    signal = "❌ 避免買入"
    color = "🔴"

print(f"當前判斷: {color}")
print(f"  趨勢狀態: {trend}")
print(f"  交易信號: {signal}")
print()

# ==========================================
# 4. 視覺化斜率
# ==========================================

print("="*80)
print("📉 步驟4: 視覺化MA50趨勢")
print("="*80)
print()

print("最近20天價格與MA50對比:")
print("-"*80)
recent_data = pd.DataFrame({
    'Price': prices.tail(20),
    'MA50': ma50.tail(20)
})

for date, row in recent_data.iterrows():
    price = row['Price']
    ma = row['MA50']
    diff = price - ma
    
    # 簡單的ASCII圖表
    if diff > 0:
        indicator = "▲" * int(abs(diff) * 2)
        print(f"{date.date()}: ${price:.2f} | MA50: ${ma:.2f} | {indicator} (+${diff:.2f})")
    else:
        indicator = "▼" * int(abs(diff) * 2)
        print(f"{date.date()}: ${price:.2f} | MA50: ${ma:.2f} | {indicator} (-${abs(diff):.2f})")

print()

# ==========================================
# 5. 實際應用示例
# ==========================================

print("="*80)
print("💡 實際應用：HTGC當前狀態分析")
print("="*80)
print()

print("根據2026-01-14的信號文件:")
print("-"*80)
print("  當前價格: $18.84")
print("  AI信號: 觀望 (WAIT)")
print("  AI強度: 0.64 / 1.00")
print()

print("為什麼AI給出觀望信號？")
print("-"*80)
print("  1. MA50_slope (7.41%重要性) → 需要確認趨勢方向")
print("  2. MACD死叉 (6.82%重要性) → 短期動能轉弱")
print("  3. 均線空頭排列 → 技術面未確認")
print()

print("如果MA50_slope轉正會怎樣？")
print("-"*80)
print("  • MA50_slope > 0.2% → 強烈買入信號")
print("  • 配合MACD金叉 → 信心大增")
print("  • 可以從20%倉位加碼至40-50%")
print()

# ==========================================
# 6. 其他重要指標計算示例
# ==========================================

print("="*80)
print("📊 其他Top 5指標計算示例")
print("="*80)
print()

# MACD計算
print("🥈 #2: MACD計算")
print("-"*80)
print("MACD = EMA(12) - EMA(26)")
print("Signal = EMA(MACD, 9)")
print("Histogram = MACD - Signal")
print()

ema12 = prices.ewm(span=12, adjust=False).mean()
ema26 = prices.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
macd_signal = macd.ewm(span=9, adjust=False).mean()
macd_hist = macd - macd_signal

latest_macd = macd.iloc[-1]
latest_macd_signal = macd_signal.iloc[-1]
latest_macd_hist = macd_hist.iloc[-1]

print(f"當前MACD: {latest_macd:.4f}")
print(f"當前Signal: {latest_macd_signal:.4f}")
print(f"當前Histogram: {latest_macd_hist:.4f}")

if latest_macd > latest_macd_signal:
    print("狀態: ✅ 金叉 (MACD > Signal) → 買入信號")
else:
    print("狀態: ❌ 死叉 (MACD < Signal) → 賣出信號")
print()

# 其他指標
print("其他指標計算方法:")
print("-"*80)
print("🥉 #3: MA_50 = 直接取MA50的值")
print("#4: MA_20 = 20日移動平均")
print("#5: OBV_MA = OBV的移動平均")
print()

ma20 = prices.rolling(window=20).mean()
print(f"當前MA20: ${ma20.iloc[-1]:.2f}")
print(f"當前MA50: ${latest_ma50:.2f}")
print()

# ==========================================
# 7. 完整的Python代碼模板
# ==========================================

print("="*80)
print("💻 完整Python代碼模板")
print("="*80)
print()

code_template = '''
def calculate_ma50_slope(prices, window=50, slope_period=5):
    """
    計算MA50斜率
    
    Parameters:
    -----------
    prices : pd.Series
        價格序列
    window : int
        移動平均窗口（預設50）
    slope_period : int
        計算斜率的天數（預設5）
    
    Returns:
    --------
    slope : float
        斜率值
    slope_pct : float
        斜率百分比
    ma50 : pd.Series
        MA50序列
    """
    # 1. 計算MA50
    ma50 = prices.rolling(window=window).mean()
    
    # 2. 取最近N天的MA50
    recent_ma50 = ma50.tail(slope_period)
    
    # 3. 線性回歸計算斜率
    x = np.arange(len(recent_ma50))
    y = recent_ma50.values
    slope = np.polyfit(x, y, 1)[0]
    
    # 4. 計算斜率百分比
    current_ma50 = ma50.iloc[-1]
    slope_pct = (slope / current_ma50) * 100
    
    return slope, slope_pct, ma50

# 使用範例
import yfinance as yf
import pandas as pd
import numpy as np

# 下載數據
ticker = yf.Ticker("HTGC")
df = ticker.history(period="6mo")
prices = df['Close']

# 計算斜率
slope, slope_pct, ma50 = calculate_ma50_slope(prices)

print(f"MA50 Slope: {slope:.6f}")
print(f"MA50 Slope %: {slope_pct:.4f}%")

# 判斷趨勢
if slope_pct > 0.2:
    print("✅ 強勢上升趨勢 - 買入信號")
elif slope_pct > 0:
    print("⚠️ 溫和上升趨勢 - 謹慎看多")
elif slope_pct > -0.1:
    print("🟡 盤整 - 觀望")
else:
    print("❌ 下降趨勢 - 避免買入")
'''

print(code_template)
print()

# ==========================================
# 8. 總結
# ==========================================

print("="*80)
print("📝 總結")
print("="*80)
print()

print("✅ 學到的知識:")
print("-"*80)
print("1. MA50_slope是MA50的變化率（斜率）")
print("2. 正斜率 = 上升趨勢，負斜率 = 下降趨勢")
print("3. 斜率越大，趨勢越強")
print("4. 對HTGC來說，MA50_slope是最重要的指標（7.41%）")
print("5. 必須配合其他指標（MACD、OBV等）確認")
print()

print("⚠️ 重要提醒:")
print("-"*80)
print("1. 不要只看單一指標")
print("2. HTGC的RSI重要性很低（4.70%），不要過度依賴")
print("3. 等待多個信號確認（至少3-4個）才大舉買入")
print("4. 使用動態止損（MA50或MA20）")
print()

print("🎯 當前HTGC操作建議:")
print("-"*80)
print("• 小倉位建倉15-20%（享受10.1%股息）")
print("• 等待MA50_slope轉正")
print("• 等待MACD金叉確認")
print("• 信號確認後加碼至40-50%")
print()

print("="*80)
print("✅ 分析完成！")
print("="*80)
