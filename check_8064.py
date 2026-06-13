import sys, io, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.ERROR)

from bull_consolidation_detector import scan_ticker, print_report
from macd_foot_detector import scan_ticker as scan_foot, print_signals

print('=' * 55)
print('  8064 東捷 — 主升段判斷')
print('=' * 55)

result, df_w, df_m = scan_ticker('8064.TWO', period='3y')
print_report(result, df_w, df_m, '8064.TWO')

print('  MACD 收腳訊號 (最近5次)')
df_foot = scan_foot('8064.TWO', period='2y', min_down_bars=2, min_shrink_pct=10)
print_signals(df_foot, '8064.TWO', last_n=5)
