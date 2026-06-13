import yfinance as yf, pandas as pd, warnings, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

tk = yf.Ticker('2455.TW')

def macd_df(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    dif   = ema12 - ema26
    sig   = dif.ewm(span=9, adjust=False).mean()
    osc   = dif - sig
    return dif, sig, osc

# === 日線 ===
d = tk.history(period='6mo', interval='1d')
d.index = d.index.tz_localize(None)
dif, sig, osc = macd_df(d['Close'])

print('=== 日線 MACD 最近12根 ===')
tail = pd.DataFrame({'Close': d['Close'], 'DIF': dif, 'Signal': sig, 'OSC': osc}).tail(12)
for dt, row in tail.iterrows():
    bar = 'GREEN' if row['OSC'] > 0 else 'RED  '
    date_str = str(dt.date())
    print('  %s  Close=%8.2f  DIF=%8.3f  Sig=%8.3f  OSC=%+8.3f  [%s]' % (
        date_str, row['Close'], row['DIF'], row['Signal'], row['OSC'], bar))

# === 週線 ===
w = tk.history(period='2y', interval='1wk')
w.index = w.index.tz_localize(None)
difw, sigw, oscw = macd_df(w['Close'])

print()
print('=== 週線 MACD 最近12根 ===')
tailw = pd.DataFrame({'Close': w['Close'], 'DIF': difw, 'Signal': sigw, 'OSC': oscw}).tail(12)
for dt, row in tailw.iterrows():
    bar = 'GREEN' if row['OSC'] > 0 else 'RED  '
    date_str = str(dt.date())
    print('  %s  Close=%8.2f  DIF=%8.3f  Sig=%8.3f  OSC=%+8.3f  [%s]' % (
        date_str, row['Close'], row['DIF'], row['Signal'], row['OSC'], bar))
