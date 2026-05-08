# -*- coding: utf-8 -*-
"""
2344.TW 最佳掛單買入價格計算器
Calculate optimal buy limit order price for 2344.TW
"""

import sys
import io
import os

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("📊 TSLA (特斯拉) 最佳掛單買入價格計算器")
print("="*70)

# =========================================================
# 1. 下載歷史數據
# =========================================================
def download_data(ticker, days=90):
    """下載歷史數據用於分析"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"\n📥 下載歷史數據分析...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass

        if df.empty:
            raise ValueError("數據為空")

        return df

    except Exception as e:
        print(f"❌ 數據下載失敗: {e}")
        return None

ticker = 'TSLA'
df = download_data(ticker, days=90)

if df is None:
    sys.exit(1)

print(f"✓ 已下載 {len(df)} 個交易日數據")

# =========================================================
# 2. 計算技術指標
# =========================================================
print(f"\n📈 計算技術指標...")

# 移動平均線
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 布林通道
sma_20 = df['Close'].rolling(window=20).mean()
std_20 = df['Close'].rolling(window=20).std()
df['BB_upper'] = sma_20 + (std_20 * 2)
df['BB_lower'] = sma_20 - (std_20 * 2)
df['BB_middle'] = sma_20

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# ATR (Average True Range) - 波動率指標
df['High_Low'] = df['High'] - df['Low']
df['High_Close'] = abs(df['High'] - df['Close'].shift())
df['Low_Close'] = abs(df['Low'] - df['Close'].shift())
df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14).mean()

df = df.dropna()

# =========================================================
# 3. 當前市場狀況
# =========================================================
latest = df.iloc[-1]
current_price = latest['Close']
current_date = df.index[-1]

print(f"\n" + "="*70)
print(f"📅 最新數據 ({current_date.strftime('%Y-%m-%d')})")
print("="*70)
print(f"當前收盤價:     ${current_price:.2f}")
print(f"今日最高:       ${latest['High']:.2f}")
print(f"今日最低:       ${latest['Low']:.2f}")
print(f"成交量:         {latest['Volume']:,.0f}")

# =========================================================
# 4. 計算支撐位和阻力位
# =========================================================
print(f"\n📊 技術分析:")
print("-" * 70)

# 近期高低點
recent_high = df['High'].iloc[-20:].max()
recent_low = df['Low'].iloc[-20:].min()

print(f"20日最高價:     ${recent_high:.2f}")
print(f"20日最低價:     ${recent_low:.2f}")

# 移動平均線
print(f"\n移動平均線:")
print(f"  SMA 5日:      ${latest['SMA_5']:.2f}")
print(f"  SMA 10日:     ${latest['SMA_10']:.2f}")
print(f"  SMA 20日:     ${latest['SMA_20']:.2f}")
print(f"  SMA 50日:     ${latest['SMA_50']:.2f}")

# 布林通道
print(f"\n布林通道 (20日, 2σ):")
print(f"  上軌:         ${latest['BB_upper']:.2f}")
print(f"  中軌:         ${latest['BB_middle']:.2f}")
print(f"  下軌:         ${latest['BB_lower']:.2f}")
print(f"  當前位置:     {((current_price - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']) * 100):.1f}%")

# RSI
print(f"\nRSI (14日):     {latest['RSI']:.2f}", end="")
if latest['RSI'] > 70:
    print(" (超買)")
elif latest['RSI'] < 30:
    print(" (超賣)")
else:
    print(" (中性)")

# MACD
print(f"\nMACD:")
print(f"  MACD:         {latest['MACD']:.3f}")
print(f"  Signal:       {latest['Signal_Line']:.3f}")
if latest['MACD'] > latest['Signal_Line']:
    print(f"  趨勢:         看多 ↗")
else:
    print(f"  趨勢:         看空 ↘")

# ATR
print(f"\nATR (14日):     ${latest['ATR']:.2f} (日均波動)")

# =========================================================
# 5. 計算建議掛單價格
# =========================================================
print(f"\n" + "="*70)
print("💡 建議掛單買入價格")
print("="*70)

# 策略 1: 基於支撐位（保守）
support_price_1 = latest['SMA_20']  # 20日均線支撐
support_price_2 = latest['BB_lower']  # 布林下軌
support_price_3 = recent_low * 1.01  # 近期低點上方1%

# 策略 2: 基於當前價格回調（積極）
pullback_3pct = current_price * 0.97  # 回調3%
pullback_5pct = current_price * 0.95  # 回調5%

# 策略 3: 基於ATR（動態）
atr_buy_price = current_price - (latest['ATR'] * 0.5)  # 當前價 - 0.5倍ATR

# 策略 4: 心理價位（整數關口）
def find_psychological_level(price, step=1.0):
    """找最近的整數關口"""
    return round(price / step) * step

psychological_price = find_psychological_level(current_price * 0.98, 1.0)

# =========================================================
# 6. 綜合建議
# =========================================================
print(f"\n策略 1️⃣ - 技術支撐位（保守，適合長期持有）:")
print(f"  20日均線支撐:  ${support_price_1:.2f} (推薦 ⭐⭐⭐)")
print(f"  布林下軌:      ${support_price_2:.2f}")
print(f"  近期低點+1%:   ${support_price_3:.2f}")

print(f"\n策略 2️⃣ - 價格回調（中等風險）:")
print(f"  回調 3%:       ${pullback_3pct:.2f} (推薦 ⭐⭐)")
print(f"  回調 5%:       ${pullback_5pct:.2f}")

print(f"\n策略 3️⃣ - ATR動態價位（適應波動）:")
print(f"  當前價-0.5ATR: ${atr_buy_price:.2f} (推薦 ⭐⭐⭐)")

print(f"\n策略 4️⃣ - 心理價位（整數關口）:")
print(f"  整數關口:      ${psychological_price:.2f}")

# =========================================================
# 7. 最終推薦價格
# =========================================================
# 綜合考慮各種因素，給出最優價格範圍
prices = [support_price_1, atr_buy_price, pullback_3pct]
recommended_price = np.median(prices)  # 取中位數

print(f"\n" + "="*70)
print(f"🎯 最終推薦")
print("="*70)
print(f"\n最佳掛單價格:   ${recommended_price:.2f}")
print(f"價格區間:       ${min(prices):.2f} - ${max(prices):.2f}")
print(f"與當前價差距:   {((recommended_price - current_price) / current_price * 100):.2f}%")

# 預期報酬計算
target_price = current_price * 1.05  # 假設目標獲利5%
potential_profit = (target_price - recommended_price) / recommended_price * 100

print(f"\n📈 預期報酬分析:")
print(f"  掛單價:       ${recommended_price:.2f}")
print(f"  目標價(+5%):  ${target_price:.2f}")
print(f"  預期獲利:     {potential_profit:.2f}%")

# =========================================================
# 8. 風險評估
# =========================================================
print(f"\n⚠️  風險評估:")

# 計算成交概率（基於歷史波動）
recent_lows = df['Low'].iloc[-20:]
prob_hit = (recent_lows <= recommended_price).sum() / len(recent_lows) * 100

print(f"  成交概率:     約 {prob_hit:.0f}% (基於過去20日低點)")

if recommended_price > latest['BB_lower']:
    print(f"  風險等級:     🟢 低風險 (高於布林下軌)")
elif recommended_price > recent_low:
    print(f"  風險等級:     🟡 中風險 (高於近期低點)")
else:
    print(f"  風險等級:     🔴 高風險 (接近或低於近期低點)")

# =========================================================
# 9. 操作建議
# =========================================================
print(f"\n" + "="*70)
print("📝 操作建議")
print("="*70)

print(f"\n1. 掛單設定:")
print(f"   掛單價格: ${recommended_price:.2f}")
print(f"   數量建議: 先掛一半資金，分批進場")
print(f"   有效期限: 建議設定 3-5 個交易日")

print(f"\n2. 多層掛單策略（更保守）:")
print(f"   第1層 (30%資金): ${pullback_3pct:.2f}")
print(f"   第2層 (40%資金): ${recommended_price:.2f}")
print(f"   第3層 (30%資金): ${support_price_2:.2f}")

print(f"\n3. 停損設定:")
stop_loss = recommended_price * 0.95
print(f"   建議停損價: ${stop_loss:.2f} (-5%)")

print(f"\n4. 獲利目標:")
take_profit_1 = recommended_price * 1.03
take_profit_2 = recommended_price * 1.05
take_profit_3 = recommended_price * 1.08
print(f"   短期目標: ${take_profit_1:.2f} (+3%)")
print(f"   中期目標: ${take_profit_2:.2f} (+5%)")
print(f"   長期目標: ${take_profit_3:.2f} (+8%)")

# =========================================================
# 10. 注意事項
# =========================================================
print(f"\n" + "="*70)
print("⚠️  重要注意事項")
print("="*70)
print(f"\n1. 此價格計算基於技術分析，不保證成交")
print(f"2. 市場可能因突發消息大幅波動")
print(f"3. 建議觀察以下確認信號:")
print(f"   - 成交量是否萎縮（表示下跌動能減弱）")
print(f"   - RSI 是否接近超賣區（<30）")
print(f"   - MACD 是否出現黃金交叉")
print(f"4. 掛單後每日檢查，根據市場調整價格")
print(f"5. 不要 All-in，保留現金應對更好機會")

print(f"\n" + "="*70)
print("✅ 分析完成！祝交易順利！")
print("="*70)
