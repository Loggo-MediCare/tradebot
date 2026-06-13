"""通用股票進場分析 — 傳入 ticker 即可"""
import sys, io, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.ERROR)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf, pandas as pd, numpy as np

def compute_macd(close, fast=12, slow=26, sig=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    ml = ema_f - ema_s
    ms = ml.ewm(span=sig, adjust=False).mean()
    return ml, ms, ml - ms

def resample_macd(df, freq):
    rs = df[['open','high','low','close','volume']].resample(freq).agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    ml, ms, mh = compute_macd(rs['close'])
    rs['macd'] = ml; rs['macd_hist'] = mh
    return rs

TICKER = sys.argv[1] if len(sys.argv) > 1 else '2408.TW'
NAME   = sys.argv[2] if len(sys.argv) > 2 else TICKER

# ── Download ──────────────────────────────────────────────────────────────────
df = yf.download(TICKER, period='3y', progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
df = df.rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'})
df.index = pd.to_datetime(df.index)

# ── Indicators ────────────────────────────────────────────────────────────────
c = df['close']
price  = float(c.iloc[-1])
sma10  = float(c.rolling(10).mean().iloc[-1])
sma20  = float(c.rolling(20).mean().iloc[-1])
sma50  = float(c.rolling(50).mean().iloc[-1])
sma200 = float(c.rolling(200).mean().iloc[-1])
hl = df['high'] - df['low']
hc = abs(df['high'] - df['close'].shift())
lc = abs(df['low']  - df['close'].shift())
atr = float(pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean().iloc[-1])
d = c.diff()
rs = d.where(d>0,0).rolling(14).mean() / (-d.where(d<0,0)+1e-10).rolling(14).mean()
rsi = float((100-(100/(1+rs))).iloc[-1])
dev = (price - sma20) / sma20 * 100

# ── Multi-timeframe MACD ──────────────────────────────────────────────────────
ml_d, _, mh_d = compute_macd(c)
df_w = resample_macd(df, 'W-FRI')
df_m = resample_macd(df, 'ME')
d_macd = float(ml_d.iloc[-1]); d_hist = float(mh_d.iloc[-1])
w_macd = float(df_w['macd'].iloc[-1]); w_hist = float(df_w['macd_hist'].iloc[-1])
m_macd = float(df_m['macd'].iloc[-1]); m_hist = float(df_m['macd_hist'].iloc[-1])

# ── ROI since 1y / 6m / 1m ───────────────────────────────────────────────────
def roi_since(days):
    sub = df.tail(days)
    return (sub['close'].iloc[-1] - sub['close'].iloc[0]) / sub['close'].iloc[0] * 100

roi_1m  = roi_since(21)
roi_3m  = roi_since(63)
roi_6m  = roi_since(126)
roi_1y  = roi_since(252)

# ── MACD status ───────────────────────────────────────────────────────────────
d_above = d_macd > 0; w_above = w_macd > 0; m_above = m_macd > 0
d_neg   = d_hist < 0

if d_neg and d_above and w_above and m_above:
    bull_status = "✅ 強勢整理 — 日線回調，週/月線完整"
elif not d_neg and d_above and w_above and m_above:
    bull_status = "🟢 完美多頭 — 三線全正"
elif d_above and w_above and not m_above:
    bull_status = "⚠️  月線轉負 — 長期出現裂縫"
elif not d_above:
    bull_status = "❌ 日線轉負 — 多頭動能喪失"
else:
    bull_status = "⚪ 混合訊號"

# ── Print ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  {NAME} ({TICKER}) 進場分析  ({df.index[-1].date()})")
print(f"{'='*55}")

print(f"\n📈 漲幅表現")
print(f"  近1個月: {roi_1m:+.1f}%   近3個月: {roi_3m:+.1f}%")
print(f"  近6個月: {roi_6m:+.1f}%   近1年:   {roi_1y:+.1f}%")

print(f"\n📊 技術面")
print(f"  現價:    NT${price:.1f}")
print(f"  SMA10:   NT${sma10:.1f}  ({(price-sma10)/price*100:+.1f}%)")
print(f"  SMA20:   NT${sma20:.1f}  (乖離 {dev:+.1f}%)")
print(f"  SMA50:   NT${sma50:.1f}  ({(price-sma50)/price*100:+.1f}%)")
print(f"  SMA200:  NT${sma200:.1f}  ({(price-sma200)/price*100:+.1f}%)")
print(f"  RSI:     {rsi:.1f}   ATR: NT${atr:.1f}")

print(f"\n🔍 多空結構 (日/週/月)")
print(f"  日線 MACD快線: {d_macd:+.3f}  柱狀體: {'🟢' if not d_neg else '🔴'}{d_hist:+.3f}")
print(f"  週線 MACD快線: {w_macd:+.3f}  柱狀體: {'🟢' if w_hist>=0 else '🔴'}{w_hist:+.3f}")
print(f"  月線 MACD快線: {m_macd:+.3f}  柱狀體: {'🟢' if m_hist>=0 else '🔴'}{m_hist:+.3f}")
print(f"\n  → {bull_status}")

print(f"\n{'='*55}")
print(f"  進場建議")
print(f"{'='*55}")

if not d_above:
    print(f"\n❌ 日線 MACD 已轉負，不建議進場")
    print(f"   等日線 MACD 快線重回 0 軸以上再考慮")
elif not w_above or not m_above:
    print(f"\n⚠️  週/月線出現問題，謹慎進場")
elif dev > 20:
    print(f"\n⚠️  乖離過大 ({dev:+.1f}%)，追高風險高")
    print(f"  方案A（等回調）:")
    print(f"    買點1: NT${sma10:.0f}  (SMA10, -{(price-sma10)/price*100:.1f}%)")
    print(f"    買點2: NT${sma20:.0f}  (SMA20, -{(price-sma20)/price*100:.1f}%)")
    print(f"  方案B（等MACD收腳+跳空確認後進場）")
    print(f"  方案C: 現在買30%，加碼區 NT${sma10:.0f}~NT${sma20:.0f}")
    print(f"  止損:  NT${price-2*atr:.0f}  (-2ATR = -{2*atr/price*100:.1f}%)")
elif 10 < dev <= 20:
    print(f"\n🟡 乖離偏高 ({dev:+.1f}%)，建議小量試單")
    print(f"  方案A: 買50%，回調 NT${sma10:.0f} 加碼")
    print(f"  方案B: 等MACD收腳後再進")
    print(f"  止損:  NT${price-2*atr:.0f}  (-{2*atr/price*100:.1f}%)")
else:
    print(f"\n🟢 乖離合理 ({dev:+.1f}%)，可以進場")
    print(f"  進場: NT${price:.0f}  滿倉或80%")
    print(f"  止損: NT${price-2*atr:.0f}  (-{2*atr/price*100:.1f}%)")
    print(f"  加碼: 若回測 NT${sma20:.0f} 不破再買")

print(f"\n{'='*55}")
print(f"  出場條件（不到這裡不動）")
print(f"{'='*55}")
print(f"  🔴 週線MACD快線跌破0  (現在: {w_macd:+.2f})")
print(f"  🔴 月線MACD快線跌破0  (現在: {m_macd:+.2f})")
print(f"  🔴 股價跌破SMA50       (SMA50: NT${sma50:.0f}, 差距{(price-sma50)/price*100:.0f}%)")
hold_ok = "✅ 出場條件未觸發，多頭結構完整" if (w_above and m_above and price > sma50) else "⚠️ 已觸發出場條件"
print(f"\n  → {hold_ok}")
