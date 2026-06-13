import sys, io, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.ERROR)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, yfinance as yf

def macd_line(s): return s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean()
def macd_hist(s): ml=macd_line(s); sig=ml.ewm(span=9,adjust=False).mean(); return ml, sig, ml-sig

df = yf.download('NVDA', period='2y', interval='1d', progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
c = df['Close']
print(f'NVDA rows: {len(df)}')

ml_d, sig_d, mh_d = macd_hist(c)
d_macd = float(ml_d.iloc[-1])
d_hist = float(mh_d.iloc[-1])
print(f'Daily MACD:   {d_macd:.3f}  (pos={d_macd>0})  hist={d_hist:.3f}')

cw = c.resample('W-FRI').last().dropna()
ml_w, _, mh_w = macd_hist(cw)
w_macd = float(ml_w.iloc[-1])
print(f'Weekly MACD:  {w_macd:.3f}  (pos={w_macd>0})  rows={len(cw)}')

cm = c.resample('ME').last().dropna()
ml_m, _, mh_m = macd_hist(cm)
m_macd = float(ml_m.iloc[-1])
print(f'Monthly MACD: {m_macd:.3f}  (pos={m_macd>0})  rows={len(cm)}')

bull = d_macd>0 and w_macd>0 and m_macd>0
d_neg_hist = d_hist < 0
exit_warn = (w_macd <= 0) or (m_macd <= 0)
dip_watch = (not bull) and (d_macd < 0) and (w_macd > 0) and (m_macd > 0)

print()
print(f'bull={bull}  d_neg_hist={d_neg_hist}  exit_warn={exit_warn}  dip_watch={dip_watch}')
print()
if bull and not d_neg_hist:  print('=> 🟢 完美多頭  (Section 1 - perfect)')
elif bull and d_neg_hist:    print('=> ✅ 強勢整理  (Section 1 - consol)')
elif dip_watch:              print('=> 🔵 日線轉負/週月仍多頭  (NEW Section 1b)')
elif exit_warn:              print('=> 🔴 出場警示  (Section 3)')
else:                        print('=> ⬜ 其他狀態 (會出現在 Pinned 區)')
print()
print(f'BEFORE FIX: NVDA was {"VISIBLE" if bull or exit_warn else "INVISIBLE"}')
print(f'AFTER FIX:  NVDA always visible in Pinned section + {"Section 1" if bull else "Section 1b" if dip_watch else "Pinned only"}')
