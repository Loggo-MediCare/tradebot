import yfinance as yf, pandas as pd, numpy as np, warnings, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

tk = yf.Ticker('2337.TW')

def macd_line(s):
    return s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean()
def macd_hist(s):
    ml = macd_line(s); sig = ml.ewm(span=9,adjust=False).mean()
    return ml, sig, ml - sig

w = tk.history(period='2y', interval='1wk')
w.index = w.index.tz_localize(None)
ml, sig, osc = macd_hist(w['Close'])

print('=== 2337 週線 MACD 最近8根 ===')
tail = pd.DataFrame({'Close': w['Close'], 'DIF(w_macd)': ml, 'Signal': sig, 'OSC(w_hist)': osc}).tail(8)
for dt, row in tail.iterrows():
    bar = 'GREEN' if row['OSC(w_hist)'] > 0 else 'RED  '
    prev_osc = tail['OSC(w_hist)'].shift(1).loc[dt]
    flip = ' << FLIPPED' if (not np.isnan(prev_osc) and prev_osc > 0 and row['OSC(w_hist)'] < 0) else ''
    print('  %s  Close=%7.2f  DIF=%7.3f  Sig=%7.3f  OSC=%+8.3f  [%s]%s' % (
        str(dt.date()), row['Close'], row['DIF(w_macd)'], row['Signal'], row['OSC(w_hist)'], bar, flip))

print()
w_hist_last  = float(osc.iloc[-1])
w_hist_prev  = float(osc.iloc[-2])
w_macd_last  = float(ml.iloc[-1])
print('w_hist_prev=%+.3f  w_hist(latest)=%+.3f' % (w_hist_prev, w_hist_last))
print('dashboard 觸發條件 (w_hist_prev>0 and w_hist<0):', w_hist_prev > 0 and w_hist_last < 0)
print('dashboard w_macd label (the +25.75 shown):', w_macd_last)
