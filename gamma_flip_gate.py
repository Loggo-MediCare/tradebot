    """
大戶籌碼開關 (Gamma Flip Gate)
======================================
核心邏輯：若現價低於 Gamma Flip（造市商由正轉負 Gamma 的價位），
代表造市商在此價位下方處於「負 Gamma」（追漲追跌），
此時不論技術指標 (MACD/PPO) 多看多，一律強制沒收 BUY 訊號。

複用自 Gex_scannerv3.py 的 Gamma Flip 計算邏輯（精簡版，僅用最近到期日）。
"""

from math import log, sqrt, exp

import numpy as np
import pandas as pd
import yfinance as yf


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / 2.5066282746


def _calc_d1(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0 or K <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _calc_gamma(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
        return 0.0
    d1 = _calc_d1(S, K, T, sigma, r)
    return _norm_pdf(d1) / (S * sigma * np.sqrt(T))


def _calc_gex(oi, gamma, spot, is_call=True):
    raw = oi * gamma * 100 * spot * spot / 1_000_000
    return raw if is_call else -raw


def _fetch_nearest_chain(symbol):
    """抓取最近一個未到期的選擇權鏈"""
    ticker = yf.Ticker(symbol)
    all_expiries = ticker.options
    if not all_expiries:
        raise ValueError(f"無法取得 {symbol} 的選擇權數據")

    import datetime as _dt
    utc_now = _dt.datetime.now(_dt.timezone.utc)
    et_today = (utc_now + _dt.timedelta(hours=-4)).strftime('%Y-%m-%d')
    future = [e for e in all_expiries if e > et_today]
    target_expiry = future[0] if future else all_expiries[-1]

    chain = ticker.option_chain(target_expiry)
    return chain.calls, chain.puts, target_expiry


def get_gamma_flip(symbol, spot):
    """計算最近到期日的 Gamma Flip 價位

    回傳: (gamma_flip, found, data_quality)
    """
    calls_df, puts_df, expiry = _fetch_nearest_chain(symbol)

    expiry_dt = pd.to_datetime(expiry)
    now_dt = pd.Timestamp.now()
    T_years = max((expiry_dt - now_dt).total_seconds() / (365.25 * 24 * 3600), 0.5 / 365)

    FRESHNESS_DAYS = 5

    def safe_float(val, default=None):
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            return float(val)
        except (TypeError, ValueError):
            return default

    def safe_int(val, default=0):
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            return int(val)
        except (TypeError, ValueError):
            return default

    def is_fresh(last_trade_date_val):
        try:
            if last_trade_date_val is None:
                return False
            ts = pd.to_datetime(last_trade_date_val, utc=False)
            ts = ts.tz_localize(None) if ts.tzinfo else ts
            return (now_dt - ts).total_seconds() / 86400 <= FRESHNESS_DAYS
        except Exception:
            return False

    def bs_ncdf(x):
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (
            1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        pdf = exp(-0.5 * x * x) / 2.5066282746
        cdf = 1.0 - pdf * poly
        return cdf if x >= 0 else 1.0 - cdf

    def bs_price(S, K, T, sigma, r=0.05, is_call=True):
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
            return 0.0
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if is_call:
            return S * bs_ncdf(d1) - K * exp(-r * T) * bs_ncdf(d2)
        return K * exp(-r * T) * bs_ncdf(-d2) - S * bs_ncdf(-d1)

    def iv_bisect(price, S, K, T, r=0.05, is_call=True, tol=1e-4):
        if price <= 0 or T <= 0 or K <= 0:
            return None
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        if price < intrinsic * 0.99:
            return None
        lo, hi = 0.005, 6.0
        for _ in range(60):
            mid = (lo + hi) / 2
            est = bs_price(S, K, T, mid, r, is_call)
            if abs(est - price) < tol:
                return mid
            if est < price:
                lo = mid
            else:
                hi = mid
        v = (lo + hi) / 2
        return v if 0.01 <= v <= 5.0 else None

    gex_results = []

    def parse(row, is_call):
        K = safe_float(row.get('strike'))
        oi = safe_int(row.get('openInterest', 0))
        vol = safe_int(row.get('volume', 0))
        iv_yf = safe_float(row.get('impliedVolatility'))
        last = safe_float(row.get('lastPrice'))
        ltd = row.get('lastTradeDate', None)

        if K is None or K <= 0 or oi <= 0:
            return
        if not is_fresh(ltd):
            return

        has_yf_iv = (iv_yf is not None and 0.01 <= iv_yf <= 3.0)
        has_volume = vol > 0
        has_last = (last is not None and last > 0)

        if not has_volume and not has_yf_iv:
            return
        if not has_last:
            return

        if has_yf_iv:
            iv_use = iv_yf
        elif has_volume:
            iv_use = iv_bisect(last, spot, K, T_years, is_call=is_call)
        else:
            iv_use = None

        if iv_use is None:
            return

        gex_results.append({'strike': K, 'oi': oi, 'iv': iv_use,
                             'type': 'call' if is_call else 'put'})

    for _, row in calls_df.iterrows():
        parse(row, True)
    for _, row in puts_df.iterrows():
        parse(row, False)

    total_raw = len(calls_df) + len(puts_df)
    n_gex = len(gex_results)
    iv_ratio = n_gex / max(total_raw, 1)

    if total_raw < 6 or n_gex < 5 or iv_ratio < 0.10:
        data_quality = 'UNKNOWN'
    elif n_gex < 15:
        data_quality = 'RELAXED'
    else:
        data_quality = 'OK'

    flip_subset = [r for r in gex_results if abs(r['strike'] / spot - 1.0) < 0.30]
    if not flip_subset:
        return None, False, data_quality

    def net_gex_hypo(hypo_spot):
        total = 0.0
        for r in flip_subset:
            g = _calc_gamma(hypo_spot, r['strike'], T_years, r['iv'])
            total += _calc_gex(r['oi'], g, hypo_spot, is_call=(r['type'] == 'call'))
        return total

    scan_lo = int(spot * 0.70 // 10) * 10
    scan_hi = int(spot * 1.30 // 10) * 10 + 10
    scan_pts = list(range(max(scan_lo, 10), scan_hi + 1, 10))

    flip_intervals = []
    prev_val = net_gex_hypo(scan_pts[0])
    prev_sign = 1 if prev_val >= 0 else -1
    for i in range(1, len(scan_pts)):
        p = scan_pts[i]
        val = net_gex_hypo(p)
        sign = 1 if val >= 0 else -1
        if sign != prev_sign:
            flip_intervals.append((scan_pts[i - 1], p))
        prev_sign = sign

    precise_flips = []
    for (lo, hi) in flip_intervals:
        a, b = float(lo), float(hi)
        fa = net_gex_hypo(a)
        for _ in range(30):
            mid = (a + b) / 2.0
            fm = net_gex_hypo(mid)
            if abs(fm) < 1.0:
                break
            if (fa >= 0) == (fm >= 0):
                a, fa = mid, fm
            else:
                b = mid
        precise_flips.append((a + b) / 2.0)

    if not precise_flips:
        return None, False, data_quality

    gamma_flip = float(min(precise_flips, key=lambda x: abs(x - spot)))
    return gamma_flip, True, data_quality


def check_gamma_flip_gate(symbol, current_price):
    """大戶籌碼開關：現價跌破 Gamma Flip 時否決 BUY 訊號

    採「保守原則 (fail-closed)」：若無法取得即時選擇權數據、或數據品質不足以
    判斷 Gamma Flip，視為「無法驗證籌碼結構安全」，同樣強制沒收 BUY 訊號。
    只有在即時數據確認「現價 >= Gamma Flip」（正Gamma、造市商穩定）時，
    才放行 BUY 訊號。

    回傳 dict:
        vetoed       : bool  — True = 強制取消 BUY 訊號
        gamma_flip   : float | None
        data_quality : 'OK' | 'RELAXED' | 'UNKNOWN' | 'ERROR'
        message      : str   — 人類可讀說明
    """
    try:
        gamma_flip, found, dq = get_gamma_flip(symbol, current_price)
    except Exception as e:
        return {'vetoed': True, 'gamma_flip': None, 'data_quality': 'ERROR',
                'message': f'Gamma Flip 計算失敗，無法驗證籌碼結構（保守原則）→ 強制沒收買進訊號: {e}'}

    if not found or gamma_flip is None or dq == 'UNKNOWN':
        return {'vetoed': True, 'gamma_flip': gamma_flip, 'data_quality': dq,
                'message': f'即時選擇權數據不足，無法驗證 Gamma Flip（品質: {dq}）'
                           f'，無法確認籌碼結構安全（保守原則）→ 強制沒收買進訊號'}

    vetoed = current_price < gamma_flip
    if vetoed:
        msg = (f'現價 ${current_price:.2f} < Gamma Flip ${gamma_flip:.2f}'
               f'（負Gamma區，造市商追漲追跌）→ 強制沒收買進訊號')
    else:
        msg = (f'現價 ${current_price:.2f} >= Gamma Flip ${gamma_flip:.2f}'
               f'（正Gamma區，造市商穩定）→ 買進訊號正常放行')

    return {'vetoed': vetoed, 'gamma_flip': gamma_flip, 'data_quality': dq, 'message': msg}
