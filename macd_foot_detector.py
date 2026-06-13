"""
MACD 柱狀體收腳 + 跳空缺口 Pattern Detector
==============================================
Detects:
  1. MACD 柱狀體收腳 (histogram shortening while negative)
     → histogram was negative & getting worse, now shrinking =止跌訊號
  2. 跳空缺口 (gap up after the 收腳 day)
     → next candle opens above previous high = 多頭攻勢再起
"""

import warnings, logging
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ResourceWarning)
logging.getLogger('yfinance').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


def add_macd(df, fast=12, slow=26, signal=9):
    """Compute MACD, signal line and histogram."""
    df['ema_fast']    = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow']    = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd']        = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    return df


def detect_macd_foot(df,
                     min_down_bars: int = 2,
                     min_shrink_pct: float = 10.0,
                     check_gap: bool = True) -> pd.DataFrame:
    """
    Scan every row and mark whether it qualifies as a 收腳 day.

    Parameters
    ----------
    df               : DataFrame with columns close/open/high/low/volume + MACD cols
    min_down_bars    : minimum consecutive negative-histogram bars before 收腳
    min_shrink_pct   : histogram must shrink by at least this % vs previous bar
    check_gap        : also flag whether the *next* day gaps up (跳空缺口)

    Returns
    -------
    DataFrame with extra boolean/value columns:
      macd_foot      : True on the 收腳 day
      gap_up         : True if next day opens above this day's high
      signal_strength: composite score 0-100
    """
    df = df.copy()
    n = len(df)

    df['macd_foot']      = False
    df['gap_up']         = False
    df['signal_strength']= 0.0
    df['prev_hist']      = df['macd_hist'].shift(1)
    df['shrink_pct']     = np.nan

    for i in range(max(min_down_bars, 1), n):
        hist_now  = df['macd_hist'].iloc[i]
        hist_prev = df['macd_hist'].iloc[i - 1]

        # ── Condition 1: histogram must be negative (downtrend)
        if hist_now >= 0 or hist_prev >= 0:
            continue

        # ── Condition 2: current bar SHORTER than previous (abs shrinks)
        if abs(hist_now) >= abs(hist_prev):
            continue

        # ── Condition 3: previous bars were all negative (confirmed downtrend)
        down_window = df['macd_hist'].iloc[i - min_down_bars: i]
        if not (down_window < 0).all():
            continue

        # ── Condition 4: shrinkage must be meaningful
        shrink = (abs(hist_prev) - abs(hist_now)) / (abs(hist_prev) + 1e-10) * 100
        if shrink < min_shrink_pct:
            continue

        df.at[df.index[i], 'macd_foot']   = True
        df.at[df.index[i], 'shrink_pct']  = round(shrink, 2)

        # ── Gap up on NEXT day
        if check_gap and i + 1 < n:
            next_open = df['open'].iloc[i + 1]
            this_high = df['high'].iloc[i]
            if next_open > this_high:
                df.at[df.index[i], 'gap_up'] = True

        # ── Composite signal strength (0-100)
        score = 0.0
        # shrinkage size (up to 40 pts)
        score += min(shrink / 100 * 40, 40)
        # how many consecutive down bars before (up to 20 pts)
        score += min(min_down_bars * 5, 20)
        # MACD line turning up vs signal (10 pts)
        if df['macd'].iloc[i] > df['macd'].iloc[i - 1]:
            score += 10
        # price above 10-day SMA (10 pts)
        if 'sma_10' in df.columns and df['close'].iloc[i] > df['sma_10'].iloc[i]:
            score += 10
        # volume above 20-day average (10 pts)
        if 'volume' in df.columns:
            avg_vol = df['volume'].iloc[max(0, i-20):i].mean()
            if df['volume'].iloc[i] > avg_vol:
                score += 10
        # gap-up bonus (10 pts)
        if df.at[df.index[i], 'gap_up']:
            score += 10

        df.at[df.index[i], 'signal_strength'] = round(min(score, 100), 1)

    df.drop(columns=['prev_hist'], inplace=True)
    return df


def scan_ticker(ticker: str,
                period: str = '1y',
                min_down_bars: int = 2,
                min_shrink_pct: float = 10.0) -> pd.DataFrame:
    """Download data, compute MACD, scan for 收腳 pattern."""
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw.rename(columns={
        'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'
    }).copy()
    df['sma_10'] = df['close'].rolling(10).mean()
    df = add_macd(df)
    df = df.bfill().ffill()

    df = detect_macd_foot(df, min_down_bars=min_down_bars,
                          min_shrink_pct=min_shrink_pct)
    return df


def print_signals(df: pd.DataFrame, ticker: str, last_n: int = 10):
    """Pretty-print detected signals."""
    hits = df[df['macd_foot']].copy()
    print(f"\n{'='*65}")
    print(f"  {ticker}  MACD 柱狀體收腳 訊號  (共 {len(hits)} 次)")
    print(f"{'='*65}")
    print(f"  {'日期':<12} {'收盤':>8} {'MACD柱':>10} {'縮短%':>8} {'跳空':>6} {'強度':>6}")
    print(f"  {'-'*57}")

    for date, row in hits.tail(last_n).iterrows():
        gap_flag = '✅跳空' if row['gap_up'] else '  ─  '
        star     = ' 🌟' if row['signal_strength'] >= 70 else (' ✅' if row['signal_strength'] >= 50 else '')
        print(f"  {str(date.date()):<12} "
              f"{row['close']:>8.2f} "
              f"{row['macd_hist']:>10.4f} "
              f"{row['shrink_pct']:>7.1f}% "
              f"  {gap_flag} "
              f"{row['signal_strength']:>5.0f}{star}")

    # Latest candle check
    latest = df.iloc[-1]
    print(f"\n  📅 最新一天 ({df.index[-1].date()}): "
          f"MACD柱={latest['macd_hist']:.4f}  "
          f"收腳={'✅ 是' if latest['macd_foot'] else '❌ 否'}")
    if latest['macd_foot']:
        print(f"  🔔 今日觸發收腳！強度={latest['signal_strength']}  "
              f"跳空={'✅' if latest['gap_up'] else '待明日確認'}")
    print(f"{'='*65}\n")


def get_today_signals(tickers: list,
                      period: str = '6mo',
                      min_shrink_pct: float = 10.0) -> pd.DataFrame:
    """
    Scan multiple tickers and return only those with a 收腳 signal today
    (or yesterday if market just closed).
    """
    rows = []
    for ticker in tickers:
        try:
            df = scan_ticker(ticker, period=period, min_shrink_pct=min_shrink_pct)
            last = df.iloc[-1]
            # also check second-to-last (in case today hasn't updated yet)
            prev = df.iloc[-2] if len(df) >= 2 else None

            for i, row in [(df.index[-1], last), (df.index[-2] if prev is not None else None, prev)]:
                if i is None: continue
                if row['macd_foot']:
                    rows.append({
                        'ticker':   ticker,
                        'date':     i.date(),
                        'close':    round(row['close'], 2),
                        'macd_hist':round(row['macd_hist'], 4),
                        'shrink_pct':round(row['shrink_pct'], 1),
                        'gap_up':   row['gap_up'],
                        'strength': row['signal_strength'],
                    })
                    break  # only report most recent hit per ticker
        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values('strength', ascending=False).reset_index(drop=True)
    return result


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # ── Single stock deep scan
    TICKER = 'SNDK'
    print(f"\nScanning {TICKER}...")
    df = scan_ticker(TICKER, period='2y', min_down_bars=2, min_shrink_pct=10)
    print_signals(df, TICKER, last_n=10)

    # ── Multi-stock scan for TODAY's signals
    _TWO = {'3498','3615','4533','4577','4768','4908','4991','5011',
            '6134','6187','6220','6530','6877','7805','8086','8908','8917','8927',
            '6274','1785','4749','3131','6683','3363','3081','6510',
            '8069','6223','5483','6163','7709','7717','3260','3491','5371','3105','4971',
            '8064','3163','3455','3680','4772','6788','7703','8147','8071',
            '8027','5351','7734','7751','6138','1569','1595','4951',
            '6234','6488','6207','3624','8455','8291','3577','3236','3691',
            '6204','6432','3609','3450','3581','3265',
            '5289','3587','3264','3663','6538','3580','8044','8299','3209',
            '6147'}   # 8103/5215/6449=TW not TWO
    _T = {'3449'}
    def _t(c): return f"{c}.T" if c in _T else (f"{c}.TWO" if c in _TWO else f"{c}.TW")

    TW_CODES = [
        '2330','2317','6515','2408','2308','2313','2454','2485','2337','2344',
        '2367','3481','2603','6770','3665','3017','3711','3037','2327','2382',
        '3443','2383','6442','3661','6669','6683','3231','2303','2368','2345',
        '1303','2360','2449','6443','4989','6285','3715','3563','3653','2891',
        '6239','3533','8069','6223','3363','3449','5483','6163','7709','7717',
        '3260','3491','5371','3105','4971','6187','3615','4577','4768','4991',
        '6220','6877','8927','1519','6805','6789','8021','3006','6830','2357',
        '3030','2409','2376','8210','6446','1326','8046','1605','1301','2059',
        '6781','2884','6271','2002','6526','3138','8150','1101','2890','3044',
        '4967','2451','8110','2385','4938','3576','2634','1514','4722','6472',
        '8131','6230','2363','6209','3135','6269','8438','4564','4540','8499',
        '6477','3004','4746','8222','3022','6668','2314','1314','8908','9931',
        '8917','6505','9918','2412','6274','8112','2049','1785','6531','2395',
        '4749','3131','3081','6510','3535','8064','3163','3455','2426','3583',
        '8028','3680','4772','6788','7703','8147','2404','6196','6605','6139',
        '8071','1560','6438','6449','8027','5351','4720','6176','3380','6672',
        '6213','7734','7751','2486','6138','8103','1569','1595','6108','4951',
        '1727','6234','6488','6207','6937','3189','6147','3624','8455','6924',
        '3577','8374','2359','3236','6204','3024','6432','3609','8299','3581',
        '3265','3714','2340','1773','5215','3587','3691','3264','6257','3055',
        '5289','7610','7788','2481','3023','3663','6538','3580','2355','8044',
    ]
    WATCHLIST = [_t(c) for c in TW_CODES] + [
        'NVDA','AMD','MU','SNDK','INTC','AMAT','ASML','KLAC','LRCX','TER','QCOM','TSM',
    ]
    print("\n🔍 掃描今日收腳訊號...\n")
    signals = get_today_signals(WATCHLIST, period='6mo', min_shrink_pct=10)

    if signals.empty:
        print("  今日無收腳訊號")
    else:
        print(f"  {'Ticker':<10} {'日期':<12} {'收盤':>8} {'縮短%':>8} {'跳空':>6} {'強度':>6}")
        print(f"  {'-'*56}")
        for _, row in signals.iterrows():
            gap = '✅' if row['gap_up'] else ' ─'
            star = ' 🌟' if row['strength'] >= 70 else (' ✅' if row['strength'] >= 50 else '')
            print(f"  {row['ticker']:<10} {str(row['date']):<12} "
                  f"{row['close']:>8.2f} {row['shrink_pct']:>7.1f}%  "
                  f"{gap}  {row['strength']:>5.0f}{star}")
