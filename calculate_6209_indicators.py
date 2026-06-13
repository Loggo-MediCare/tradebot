"""
计算 6209.TW 的 MA20 和 ATR(14) 数值
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 下载数据
print("=" * 70)
print("📊 6209.TW (今國光) 技术指标计算")
print("=" * 70)
print("\n▶️  下载最新数据...")

ticker = "6209.TW"
df = yf.download(ticker, period='90d', progress=False)

if df.empty:
    print("❌ 数据下载失败")
    sys.exit(1)

# 如果是多级列索引，展平为单级
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"✅ 数据下载完成: {len(df)} 条记录")
print(f"\n最新数据日期: {df.index[-1].strftime('%Y-%m-%d')}")
print(f"当前价格: NT${df['Close'].iloc[-1]:.2f}")

# 计算 MA20
df['MA_20'] = df['Close'].rolling(window=20).mean()

# 计算 ATR(14)
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14).mean()

# 显示最近5天的数据
print("\n" + "=" * 70)
print("📈 最近 5 天技术指标")
print("=" * 70)
print(f"\n{'日期':<12} {'收盘价':>10} {'MA20':>10} {'ATR(14)':>10}")
print("-" * 70)

for i in range(-5, 0):
    date = df.index[i].strftime('%Y-%m-%d')
    close = df['Close'].iloc[i]
    ma20 = df['MA_20'].iloc[i]
    atr = df['ATR'].iloc[i]
    print(f"{date:<12} {close:>10.2f} {ma20:>10.2f} {atr:>10.2f}")

# 最新数值
latest_date = df.index[-1].strftime('%Y-%m-%d')
latest_close = df['Close'].iloc[-1]
latest_ma20 = df['MA_20'].iloc[-1]
latest_atr = df['ATR'].iloc[-1]

print("\n" + "=" * 70)
print("📊 最新指标数值 (2025-12-24)")
print("=" * 70)
print(f"\n当前价格:  NT${latest_close:,.2f}")
print(f"MA(20):    NT${latest_ma20:,.2f}")
print(f"ATR(14):   NT${latest_atr:,.2f}")

# 分析
price_vs_ma20 = ((latest_close - latest_ma20) / latest_ma20) * 100
print(f"\n价格 vs MA20: {price_vs_ma20:+.2f}%", end="")
if price_vs_ma20 > 0:
    print(" (价格高于MA20 - 多头)")
else:
    print(" (价格低于MA20 - 空头)")

# ATR 占价格比例
atr_pct = (latest_atr / latest_close) * 100
print(f"ATR 占价格: {atr_pct:.2f}% (波动率指标)")

# 计算其他相关指标
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['volatility'] = df['Close'].pct_change().rolling(window=20).std()

print("\n" + "=" * 70)
print("📊 其他重要指标")
print("=" * 70)
print(f"MA(50):        NT${df['MA_50'].iloc[-1]:,.2f}")
print(f"波动率(20):    {df['volatility'].iloc[-1]:.4f}")
print(f"最高价(今日):  NT${df['High'].iloc[-1]:,.2f}")
print(f"最低价(今日):  NT${df['Low'].iloc[-1]:,.2f}")
print(f"成交量(今日):  {df['Volume'].iloc[-1]:,.0f}")

# 20日平均成交量
avg_volume_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
volume_ratio = df['Volume'].iloc[-1] / avg_volume_20
print(f"20日平均量:    {avg_volume_20:,.0f}")
print(f"量比:          {volume_ratio:.2f}x")

print("\n" + "=" * 70)
