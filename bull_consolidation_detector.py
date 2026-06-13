"""
強勢整理偵測器 — Bull Consolidation Detector
=============================================
Pattern: 「乖離過大的強勢整理」

Conditions (all 3 timeframes must align):
  日線  — MACD histogram 由紅翻黑 (hist crosses below 0)
          MACD fast line (macd) still ABOVE zero axis ← key!
  週線  — MACD line still above 0 (mid-term uptrend intact)
  月線  — MACD line still above 0 (long-term structure intact)

This = healthy overbought pullback, NOT a trend reversal.
If weekly/monthly MACD were also negative → real bear signal.
"""

import sys, io, warnings, logging
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ResourceWarning)
logging.getLogger('yfinance').setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')


# ── MACD helper ───────────────────────────────────────────────────────────────
def compute_macd(close: pd.Series, fast=12, slow=26, sig=9):
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=sig, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def resample_and_macd(df_daily: pd.DataFrame, freq: str):
    """Resample daily OHLCV to weekly/monthly and compute MACD."""
    ohlcv = df_daily[['open','high','low','close','volume']].copy()
    ohlcv.index = pd.to_datetime(ohlcv.index)
    rs = ohlcv.resample(freq).agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
        'volume':'sum',
    }).dropna()
    macd_line, signal_line, histogram = compute_macd(rs['close'])
    rs['macd']      = macd_line
    rs['macd_sig']  = signal_line
    rs['macd_hist'] = histogram
    return rs


# ── Core detector ─────────────────────────────────────────────────────────────
def detect_bull_consolidation(df_daily: pd.DataFrame,
                               df_weekly: pd.DataFrame,
                               df_monthly: pd.DataFrame,
                               deviation_pct: float = 5.0) -> pd.DataFrame:
    """
    For each daily bar, check all 3 conditions.

    Returns df_daily with extra columns:
      bull_consol      : True when all conditions met
      day_hist_red2black: daily hist just turned negative
      day_macd_above0  : daily MACD line > 0
      wk_macd_above0   : weekly MACD line > 0 on that date
      mo_macd_above0   : monthly MACD line > 0 on that date
      deviation_pct    : price % above its 20-day SMA
      strength         : composite score 0-100
    """
    result = df_daily.copy()
    result['bull_consol']       = False
    result['day_hist_red2black'] = False
    result['day_macd_above0']   = False
    result['wk_macd_above0']    = False
    result['mo_macd_above0']    = False
    result['dev_pct']           = 0.0
    result['strength']          = 0.0

    # Pre-compute daily MACD
    ml, sl, hl = compute_macd(df_daily['close'])
    result['macd']      = ml
    result['macd_sig']  = sl
    result['macd_hist'] = hl
    result['sma20']     = df_daily['close'].rolling(20).mean()

    for i in range(1, len(result)):
        idx   = result.index[i]
        idx_p = result.index[i - 1]

        hist_now  = result['macd_hist'].iloc[i]
        hist_prev = result['macd_hist'].iloc[i - 1]
        macd_now  = result['macd'].iloc[i]

        # ── Condition A: daily histogram 由紅翻黑
        red2black = (hist_prev >= 0) and (hist_now < 0)

        # ── Condition B: daily MACD line still above zero
        day_above0 = macd_now > 0

        # ── Condition C: weekly MACD above zero on this date
        wk_row = df_weekly[df_weekly.index <= idx]
        wk_above0 = bool(wk_row['macd'].iloc[-1] > 0) if not wk_row.empty else False

        # ── Condition D: monthly MACD above zero on this date
        mo_row = df_monthly[df_monthly.index <= idx]
        mo_above0 = bool(mo_row['macd'].iloc[-1] > 0) if not mo_row.empty else False

        # ── Deviation from 20-SMA
        sma20 = result['sma20'].iloc[i]
        close = result['close'].iloc[i]
        dev   = (close - sma20) / sma20 * 100 if sma20 > 0 else 0

        result.at[idx, 'day_hist_red2black'] = red2black
        result.at[idx, 'day_macd_above0']    = day_above0
        result.at[idx, 'wk_macd_above0']     = wk_above0
        result.at[idx, 'mo_macd_above0']     = mo_above0
        result.at[idx, 'dev_pct']            = round(dev, 2)

        if red2black and day_above0 and wk_above0 and mo_above0:
            result.at[idx, 'bull_consol'] = True

            # Composite strength
            score = 40.0  # base: all 3 timeframes aligned
            # MACD line height above 0 (higher = stronger trend, up to 20 pts)
            wk_macd = wk_row['macd'].iloc[-1]
            mo_macd = mo_row['macd'].iloc[-1]
            score += min(float(wk_macd) / (abs(float(wk_macd)) + 1) * 20, 20)
            score += min(float(mo_macd) / (abs(float(mo_macd)) + 1) * 20, 20)
            # Weekly/monthly histogram direction bonus
            if len(wk_row) >= 2 and wk_row['macd_hist'].iloc[-1] > wk_row['macd_hist'].iloc[-2]:
                score += 10  # weekly hist still rising = very bullish
            if len(mo_row) >= 2 and mo_row['macd_hist'].iloc[-1] > mo_row['macd_hist'].iloc[-2]:
                score += 10  # monthly hist still rising = ultra bullish

            result.at[idx, 'strength'] = round(min(score, 100), 1)

    return result


# ── Scan single ticker ────────────────────────────────────────────────────────
def scan_ticker(ticker: str, period: str = '3y'):
    print(f"  Downloading {ticker}...", flush=True)
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw.rename(columns={
        'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'
    })
    df.index = pd.to_datetime(df.index)

    df_w = resample_and_macd(df, 'W-FRI')   # weekly
    df_m = resample_and_macd(df, 'ME')      # monthly (month-end)

    result = detect_bull_consolidation(df, df_w, df_m)
    return result, df_w, df_m


def print_report(result: pd.DataFrame, df_w: pd.DataFrame,
                 df_m: pd.DataFrame, ticker: str, last_n: int = 8):
    hits = result[result['bull_consol']].copy()

    # Current status
    latest     = result.iloc[-1]
    latest_w   = df_w.iloc[-1]
    latest_m   = df_m.iloc[-1]
    latest_date= result.index[-1].date()

    print(f"\n{'='*68}")
    print(f"  {ticker}  強勢整理偵測  (共觸發 {len(hits)} 次)")
    print(f"{'='*68}")

    # ── Timeframe status table
    print(f"\n  📊 當前多空結構 ({latest_date})")
    print(f"  {'時框':<10} {'MACD線':>10} {'柱狀體':>10} {'0軸上方':>10}")
    print(f"  {'-'*44}")

    def tf_row(name, macd_v, hist_v):
        above = '✅ 是' if macd_v > 0 else '❌ 否'
        hist_color = '🟢' if hist_v > 0 else '🔴'
        return f"  {name:<10} {macd_v:>10.4f} {hist_color}{hist_v:>9.4f}  {above}"

    print(tf_row('日線 (D)', float(latest['macd']), float(latest['macd_hist'])))
    print(tf_row('週線 (W)', float(latest_w['macd']), float(latest_w['macd_hist'])))
    print(tf_row('月線 (M)', float(latest_m['macd']), float(latest_m['macd_hist'])))

    # ── Current verdict
    d_above  = latest['macd'] > 0
    w_above  = latest_w['macd'] > 0
    m_above  = latest_m['macd'] > 0
    d_neg    = latest['macd_hist'] < 0

    print(f"\n  🔎 解讀:")
    if d_neg and d_above and w_above and m_above:
        print(f"  ✅ 強勢整理格局確認！")
        print(f"     日線柱狀體翻黑 → 短期回調")
        print(f"     日/週/月線 MACD 快線均在 0 軸之上 → 多頭結構完整")
        print(f"     這是「乖離過大的強勢整理」，非趨勢反轉")
    elif not d_neg and d_above and w_above and m_above:
        print(f"  🟢 完美多頭！日線柱狀體仍為正 (未收腳)")
        print(f"     日/週/月 MACD 全部在 0 軸上方")
    elif d_above and w_above and not m_above:
        print(f"  ⚠️  月線 MACD 已轉負 → 長期結構出現裂縫")
    elif not d_above:
        print(f"  ❌ 日線 MACD 快線跌破 0 軸 → 多頭動能喪失")
    else:
        print(f"  ⚪ 混合訊號，需進一步觀察")

    print(f"\n  乖離率 (vs SMA20): {latest['dev_pct']:+.2f}%"
          f"  {'(乖離過大，整理合理)' if latest['dev_pct'] > 5 else ''}")

    # ── Historical hits
    if not hits.empty:
        print(f"\n  📅 歷史觸發紀錄 (最近 {min(last_n, len(hits))} 次)")
        print(f"  {'日期':<12} {'收盤':>8} {'日MACD':>10} {'週MACD':>10} {'月MACD':>10} {'強度':>6}")
        print(f"  {'-'*60}")
        for date, row in hits.tail(last_n).iterrows():
            wk_val = df_w[df_w.index <= date]['macd'].iloc[-1] if not df_w[df_w.index <= date].empty else 0
            mo_val = df_m[df_m.index <= date]['macd'].iloc[-1] if not df_m[df_m.index <= date].empty else 0
            star   = ' 🌟' if row['strength'] >= 80 else (' ✅' if row['strength'] >= 60 else '')
            print(f"  {str(date.date()):<12} {row['close']:>8.2f} "
                  f"{row['macd']:>10.4f} {float(wk_val):>10.4f} {float(mo_val):>10.4f} "
                  f"{row['strength']:>5.0f}{star}")

    print(f"{'='*68}\n")


# ── Multi-ticker scan ─────────────────────────────────────────────────────────
def scan_many(tickers: list, period: str = '3y'):
    """Return summary of current bull-consolidation status for each ticker."""
    rows = []
    for ticker in tickers:
        try:
            result, df_w, df_m = scan_ticker(ticker, period=period)
            lat   = result.iloc[-1]
            lat_w = df_w.iloc[-1]
            lat_m = df_m.iloc[-1]
            hits  = result['bull_consol'].sum()

            d_neg   = bool(lat['macd_hist'] < 0)
            d_above = bool(lat['macd'] > 0)
            w_above = bool(lat_w['macd'] > 0)
            m_above = bool(lat_m['macd'] > 0)

            status = (
                '✅ 強勢整理' if (d_neg and d_above and w_above and m_above) else
                '🟢 完美多頭' if (not d_neg and d_above and w_above and m_above) else
                '⚠️  月線轉負' if (d_above and w_above and not m_above) else
                '❌ 日線轉負' if not d_above else '⚪ 觀察'
            )
            rows.append({
                'ticker': ticker,
                'close':  round(float(lat['close']), 2),
                'd_macd': round(float(lat['macd']), 4),
                'w_macd': round(float(lat_w['macd']), 4),
                'm_macd': round(float(lat_m['macd']), 4),
                'status': status,
                'hist_hits': int(hits),
            })
        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── Deep scan: single stock
    TICKER = 'SNDK'
    result, df_w, df_m = scan_ticker(TICKER, period='3y')
    print_report(result, df_w, df_m, TICKER)

    # ── Multi-stock summary (full watchlist)
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
    US_STOCKS = ['NVDA','AMD','MU','SNDK','INTC','AMAT','ASML','KLAC','LRCX','TER','QCOM','TSM']
    WATCHLIST = [_t(c) for c in TW_CODES] + US_STOCKS
    print(f"\n🔍 多空結構掃描 — {len(WATCHLIST)} 檔股票\n")
    summary = scan_many(WATCHLIST, period='3y')

    if not summary.empty:
        print(f"  {'Ticker':<10} {'收盤':>9} {'日MACD':>10} {'週MACD':>10} {'月MACD':>10}  狀態")
        print(f"  {'-'*65}")
        for _, row in summary.iterrows():
            print(f"  {row['ticker']:<10} {row['close']:>9.2f} "
                  f"{row['d_macd']:>10.4f} {row['w_macd']:>10.4f} {row['m_macd']:>10.4f}  {row['status']}")
