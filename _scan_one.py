# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import numpy as np
import sys

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def macd_hist(s, fast=12, slow=26, sig=9):
    m = ema(s, fast) - ema(s, slow)
    return m - m.ewm(span=sig, adjust=False).mean()

ticker = sys.argv[1] if len(sys.argv) > 1 else '3260.TWO'
name   = sys.argv[2] if len(sys.argv) > 2 else ''

data  = yf.download(ticker, period='2y',  interval='1d',  progress=False, auto_adjust=True)
wdata = yf.download(ticker, period='5y',  interval='1wk', progress=False, auto_adjust=True)
mdata = yf.download(ticker, period='10y', interval='1mo', progress=False, auto_adjust=True)

if data.empty:
    print('No data for', ticker); raise SystemExit

close   = data['Close'].squeeze()
w_close = wdata['Close'].squeeze()
m_close = mdata['Close'].squeeze()

d_hist = macd_hist(close)
w_hist = macd_hist(w_close)
m_hist = macd_hist(m_close)

price  = float(close.iloc[-1])
d_now  = float(d_hist.iloc[-1])
d_prev = float(d_hist.iloc[-2])
w_now  = float(w_hist.iloc[-1])
w_prev = float(w_hist.iloc[-2])
m_now  = float(m_hist.iloc[-1])
m_prev = float(m_hist.iloc[-2])

shrink = (d_prev - d_now) / abs(d_prev) * 100 if d_prev > 0 and d_now < d_prev else 0

label = f'{ticker} {name}'.strip()
print(f'\n{"="*55}')
print(f'  {label}  現價: ${price:.2f}')
print(f'{"="*55}')
print(f'  日 MACD hist : {d_now:+.4f}  (前日: {d_prev:+.4f})  縮短: {shrink:.1f}%')
print(f'  週 MACD hist : {w_now:+.4f}  (前週: {w_prev:+.4f})')
print(f'  月 MACD hist : {m_now:+.4f}  (前月: {m_prev:+.4f})')
print()

bull      = w_now > 0 and m_now > 0
day_neg   = d_now < 0
exit_warn = w_now < 0 or m_now < 0
foot      = d_now > 0 and d_now < d_prev and shrink >= 10
gap_up    = float(close.iloc[-1]) > float(data['Open'].squeeze().iloc[-1]) * 1.005 if foot else False

if exit_warn:
    status = '🔴 出場警示'
elif day_neg and bull:
    status = '🔵 日線回調 週月仍多頭 — 逢低留意'
elif bull:
    status = '🟢 完美多頭' if d_now > 0 else '✅ 強勢整理'
else:
    status = '⚠️ 混合訊號'

print(f'  狀態 : {status}')

if foot:
    icon = '✅ 收腳+跳空 (最強)' if gap_up else '🟡 收腳 (待明日跳空確認)'
    print(f'  訊號 : 📍 {icon}  縮短 {shrink:.1f}%')

if exit_warn:
    reasons = []
    if w_now < 0: reasons.append('週線跌破0')
    if m_now < 0: reasons.append('月線跌破0')
    print(f'  原因 : {" + ".join(reasons)}')

# Bollinger
bb_mid = close.rolling(20).mean()
bb_std = close.rolling(20).std()
bb_up  = bb_mid + 2 * bb_std
bb_lo  = bb_mid - 2 * bb_std
bb_pct = (price - float(bb_lo.iloc[-1])) / (float(bb_up.iloc[-1]) - float(bb_lo.iloc[-1])) * 100
print(f'  布林帶: 上${bb_up.iloc[-1]:.2f}  中${bb_mid.iloc[-1]:.2f}  下${bb_lo.iloc[-1]:.2f}  位置:{bb_pct:.0f}%')

# RSI
delta = close.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss
rsi   = 100 - 100 / (1 + rs)
print(f'  RSI(14): {float(rsi.iloc[-1]):.1f}')

# Recent 5 days
print()
print('  近5日收盤:')
for i in range(-5, 0):
    dt = close.index[i].strftime('%Y-%m-%d')
    c  = float(close.iloc[i])
    h  = float(d_hist.iloc[i])
    chg = (float(close.iloc[i]) - float(close.iloc[i-1])) / float(close.iloc[i-1]) * 100
    print(f'    {dt}  ${c:.2f}  ({chg:+.2f}%)  日MACD: {h:+.4f}')

print()
