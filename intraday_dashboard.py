"""
當沖儀表板 (Intraday Day-Trading Dashboard)
======================================
從「日線選股器」延伸出「盤中動能掃描器」功能，監控 WATCHLIST_US / WATCHLIST_TW，
並針對 AVGO、MU、NVDA 這類容易出現 Gamma Squeeze 的股票額外計算 gamma_score。

total_score = technical_score + pattern_score + intraday_score + gamma_score + finbert_score

  - technical_score : 日線多時間框架 MACD 結構分數
  - pattern_score   : 日線型態分數 (pattern_engine.py)
  - intraday_score  : 盤中1分鐘型態分數 (pattern_intraday.py)
  - gamma_score     : Gamma Squeeze 機率分數 0-100 (gamma_squeeze_engine.py，僅 GAMMA_WATCHLIST)
  - finbert_score   : FinBERT 新聞情緒分數 (finbert_enhanced_scoring.py)

用法:
  python intraday_dashboard.py        # 美股+台股
  python intraday_dashboard.py --us   # 只看美股
  python intraday_dashboard.py --tw   # 只看台股
  python intraday_dashboard.py --workers 4
"""

import argparse
import sys
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

from pattern_intraday import get_intraday_pattern_signal
from pattern_engine import get_pattern_signal
from gamma_squeeze_engine import get_gamma_squeeze_score

try:
    from finbert_enhanced_scoring import calculate_sentiment_score
except ImportError:
    calculate_sentiment_score = None


WATCHLIST_US = ['NVDA', 'AMD', 'TSM', 'AVGO', 'MU', 'PLTR', 'SNDK', 'CRDO', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'SPCX']
WATCHLIST_TW = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2327.TW', '2330.TWO', '5269.TW', '6449.TW',
    '2303.TW', '2340.TW', '2345.TW', '2360.TW', '2402.TW', '2409.TW', '2426.TW', '2455.TW',
    '2489.TW', '3008.TW', '3450.TW', '3481.TW', '3665.TW', '3673.TW', '2492.TW',
    '3081.TWO', '3105.TWO', '3163.TWO', '3234.TWO', '3265.TWO', '3339.TWO', '3374.TWO', '3491.TWO', '3587.TWO',
]

# 容易出現 Gamma Squeeze 的股票：額外抓選擇權鏈計算 gamma_score
GAMMA_WATCHLIST = ['AVGO', 'MU', 'NVDA']


# ======================================================
# 資料抓取
# ======================================================

def _normalize_columns(df):
    """yfinance MultiIndex -> 單層小寫欄位 (open/high/low/close/volume)"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def fetch_intraday_bars(ticker, prepost=False):
    """抓取當日1分鐘K棒"""
    try:
        df = yf.Ticker(ticker).history(period='1d', interval='1m',
                                        prepost=prepost, auto_adjust=True)
        if df.empty:
            return None
        return _normalize_columns(df)
    except Exception:
        return None


def fetch_daily_bars(ticker, period='2y'):
    """抓取日線資料"""
    try:
        df = yf.Ticker(ticker).history(period=period, interval='1d',
                                        auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        return _normalize_columns(df)
    except Exception:
        return None


# ======================================================
# 評分計算
# ======================================================

def compute_technical_score(df_daily):
    """日線多時間框架 MACD 結構分數"""
    close = df_daily['close']

    def macd_line(s):
        return s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()

    def macd_hist(s):
        ml = macd_line(s)
        sig = ml.ewm(span=9, adjust=False).mean()
        return ml, ml - sig

    ml_d, mh_d = macd_hist(close)
    d_macd = float(ml_d.iloc[-1])
    d_hist = float(mh_d.iloc[-1])
    d_hist_prev = float(mh_d.iloc[-2]) if len(mh_d) >= 2 else 0.0

    cw = close.resample('W-FRI').last().dropna()
    cm = close.resample('ME').last().dropna()

    w_macd = m_macd = np.nan
    if len(cw) >= 35:
        ml_w, _ = macd_hist(cw)
        w_macd = float(ml_w.iloc[-1])
    if len(cm) >= 15:
        ml_m, _ = macd_hist(cm)
        m_macd = float(ml_m.iloc[-1])

    d_pos = d_macd > 0
    w_pos = (not np.isnan(w_macd)) and w_macd > 0
    m_pos = (not np.isnan(m_macd)) and m_macd > 0
    d_neg_hist = d_hist < 0

    score = 0
    reasons = []

    if np.isnan(w_macd) or np.isnan(m_macd):
        reasons.append("週/月線資料不足")
    elif d_neg_hist and d_pos and w_pos and m_pos:
        score += 15
        reasons.append("強勢整理 (日線收腳)")
    elif not d_neg_hist and d_pos and w_pos and m_pos:
        score += 25
        reasons.append("完美多頭 (日週月三線多)")
    elif d_pos and w_pos and not m_pos:
        score += 5
        reasons.append("月線轉負")
    elif d_pos and not w_pos:
        score -= 5
        reasons.append("週線轉負")
    elif not d_pos:
        score -= 15
        reasons.append("日線轉負")

    # MACD 收腳 (柱狀體縮減 >= 10%)
    if d_hist < 0 and d_hist_prev < 0 and abs(d_hist) < abs(d_hist_prev):
        shrink = (abs(d_hist_prev) - abs(d_hist)) / (abs(d_hist_prev) + 1e-10) * 100
        if shrink >= 10:
            score += 10
            reasons.append(f"MACD收腳 (柱縮{shrink:.0f}%)")

    return score, reasons


def compute_pattern_score(df_daily):
    """日線型態分數 (pattern_engine.py)"""
    signal = get_pattern_signal(df_daily)
    return signal['score_adjustment'], signal


def compute_intraday_score(df_1m, gex_score=None):
    """盤中1分鐘型態分數 (pattern_intraday.py)"""
    signal = get_intraday_pattern_signal(df_1m, gex_score=gex_score)
    return signal['score_adjustment'], signal


def compute_gamma_score(ticker):
    """Gamma Squeeze 機率分數 0-100 (gamma_squeeze_engine.py)"""
    try:
        result = get_gamma_squeeze_score(ticker)
        return result['score'], result
    except Exception:
        return 0, None


def compute_finbert_score(ticker):
    """FinBERT 新聞情緒分數 (finbert_enhanced_scoring.py)"""
    if calculate_sentiment_score is None:
        return 0, None
    try:
        result = calculate_sentiment_score(ticker)
        return result['score_adjustment'], result
    except Exception:
        return 0, None


# ======================================================
# 主分析流程
# ======================================================

def analyse_ticker(ticker, prepost=False):
    """分析單一標的，回傳 total_score 與各分項明細"""
    df_daily = fetch_daily_bars(ticker)
    df_1m = fetch_intraday_bars(ticker, prepost=prepost)

    if df_daily is None or df_1m is None:
        return None

    is_gamma_watch = ticker in GAMMA_WATCHLIST

    gamma_score, gamma_detail = (0, None)
    if is_gamma_watch:
        gamma_score, gamma_detail = compute_gamma_score(ticker)

    technical_score, tech_reasons = compute_technical_score(df_daily)
    pattern_score, pattern_signal = compute_pattern_score(df_daily)
    intraday_score, intraday_signal = compute_intraday_score(
        df_1m, gex_score=gamma_score if is_gamma_watch else None)
    finbert_score, finbert_result = compute_finbert_score(ticker)

    total_score = (
        technical_score
        + pattern_score
        + intraday_score
        + gamma_score
        + finbert_score
    )

    return {
        'ticker': ticker,
        'price': float(df_1m['close'].iloc[-1]),
        'is_gamma_watch': is_gamma_watch,
        'technical_score': technical_score,
        'tech_reasons': tech_reasons,
        'pattern_score': pattern_score,
        'pattern_signal': pattern_signal,
        'intraday_score': intraday_score,
        'intraday_signal': intraday_signal,
        'gamma_score': gamma_score,
        'gamma_detail': gamma_detail,
        'finbert_score': finbert_score,
        'finbert_result': finbert_result,
        'total_score': total_score,
    }


def scan_intraday(tickers, prepost=False, workers=4):
    """掃描多個標的，回傳依 total_score 排序的結果列表"""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyse_ticker, t, prepost): t for t in tickers}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda r: r['total_score'], reverse=True)
    return results


# ======================================================
# 輸出
# ======================================================

def print_intraday_dashboard(results, label):
    print("\n" + "=" * 70)
    print(f"{label}  ({len(results)} 支)")
    print("=" * 70)

    for r in results:
        tag = "  [GAMMA 觀察]" if r['is_gamma_watch'] else ""
        print(f"\n{r['ticker']:8s} ${r['price']:<10.2f} total_score: {r['total_score']:+d}{tag}")
        print(f"   technical:{r['technical_score']:+d}  pattern:{r['pattern_score']:+d}  "
              f"intraday:{r['intraday_score']:+d}  gamma:{r['gamma_score']:d}  finbert:{r['finbert_score']:+d}")

        if r['intraday_signal']['patterns']:
            print(f"   盤中型態: {', '.join(r['intraday_signal']['patterns'])}")
        if r['pattern_signal']['patterns']:
            print(f"   日線型態: {', '.join(r['pattern_signal']['patterns'])}")
        if r['tech_reasons']:
            print(f"   技術面: {', '.join(r['tech_reasons'])}")
        if r['gamma_detail'] is not None:
            print(f"   Gamma Squeeze: {r['gamma_score']}% ({r['gamma_detail']['level']})")


# ======================================================
# 主程序
# ======================================================

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="當沖儀表板")
    parser.add_argument('--us', action='store_true', help='只看美股')
    parser.add_argument('--tw', action='store_true', help='只看台股')
    parser.add_argument('--workers', type=int, default=4, help='並行數量 (預設4)')
    args = parser.parse_args()

    run_us = args.us or not args.tw
    run_tw = args.tw or not args.us

    if run_us:
        results = scan_intraday(WATCHLIST_US, prepost=True, workers=args.workers)
        print_intraday_dashboard(results, "美股當沖儀表板 (US)")

    if run_tw:
        results = scan_intraday(WATCHLIST_TW, prepost=False, workers=args.workers)
        print_intraday_dashboard(results, "台股當沖儀表板 (TW)")
