import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 安裝依賴提示
# pip install yfinance pandas numpy scipy
# ==========================================

SYMBOL = 'MU'  # 可改成任何股票代號

# ==========================================
# Black-Scholes Gamma 計算
# ==========================================
def norm_pdf(x):
    """標準常態分佈 PDF"""
    return np.exp(-0.5 * x * x) / 2.5066282746

def calc_d1(S, K, T, sigma, r=0.05):
    """計算 d1"""
    if T <= 0 or sigma <= 0 or K <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def calc_gamma(S, K, T, sigma, r=0.05):
    """計算 Gamma（造市商 delta 對股價的敏感度）"""
    if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
        return 0.0
    d1 = calc_d1(S, K, T, sigma, r)
    gamma = norm_pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def calc_gex(oi, gamma, spot, is_call=True):
    """
    GEX = OI × Gamma × 100 × Spot²
    Call 為正 GEX（造市商被迫追買）
    Put  為負 GEX（造市商被迫追賣）
    回傳值單位：百萬美元
    """
    raw = oi * gamma * 100 * spot * spot / 1_000_000
    return raw if is_call else -raw

# ==========================================
# 選擇權數據抓取
# ==========================================
def fetch_options_chain(symbol, expiry=None):
    """
    抓取 yfinance 選擇權數據
    expiry: None = 最近到期日；或傳入 'YYYY-MM-DD' 字串
    """
    ticker = yf.Ticker(symbol)
    all_expiries = ticker.options

    if not all_expiries:
        raise ValueError(f"無法取得 {symbol} 的選擇權數據，請確認股票代號正確")

    if expiry is None:
        # 永遠跳過今天及已過期的到期日
        # 理由：當天到期的合約 T≈0，Gamma 計算會失真或無有效 OI
        from datetime import timezone
        # 用美東時間判斷「今天」（UTC-4 夏令 / UTC-5 冬令）
        import datetime as _dt
        utc_now   = _dt.datetime.now(_dt.timezone.utc)
        et_offset = _dt.timedelta(hours=-4)   # 夏令 EDT；冬令改 -5
        et_today  = (utc_now + et_offset).strftime('%Y-%m-%d')

        future = [e for e in all_expiries if e > et_today]
        target_expiry = future[0] if future else all_expiries[-1]
    else:
        # 找最接近指定日期的到期日
        target_expiry = min(all_expiries, key=lambda x: abs(
            (pd.to_datetime(x) - pd.to_datetime(expiry)).days
        ))

    chain = ticker.option_chain(target_expiry)
    return chain.calls, chain.puts, target_expiry, all_expiries

def get_current_price(symbol):
    """取得即時股價"""
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    price = getattr(info, 'last_price', None) or getattr(info, 'regular_market_price', None)
    if not price:
        hist = ticker.history(period='1d')
        price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
    return float(price)

# ==========================================
# GEX 分析核心
# ==========================================
def analyze_gex(calls_df, puts_df, spot, expiry_str):
    """
    Pipeline A — OI Wall（全鏈條，高可信）
        所有 OI > 0 的合約，不需要 IV 或報價。
        輸出：Call Wall / Put Wall / OI 分佈。

    Pipeline B — GEX / IV（近期成交，中可信）
        條件：lastTradeDate 在 5 個交易日內 + (volume>0 或 yfinance IV 有效)
        IV：yfinance IV → lastPrice 反推 → 跳過（不補值）
        輸出：net GEX / Peak GEX / ATM IV / Gamma Flip
        Gamma Flip：只用 |K/spot-1|<30% 的子集 + 二分法精確定位
    """
    from math import log, sqrt, exp
    import datetime as _dt

    expiry_dt   = pd.to_datetime(expiry_str)
    now_dt      = pd.Timestamp.now(tz='UTC').tz_localize(None)                   if pd.Timestamp.now().tzinfo is None else pd.Timestamp.now()
    now_dt      = pd.Timestamp.now()          # naive local time
    expiry_naive= expiry_dt.tz_localize(None) if expiry_dt.tzinfo else expiry_dt

    # T in years using total_seconds for precision
    delta_t = expiry_naive - now_dt
    T_years = max(delta_t.total_seconds() / (365.25 * 24 * 3600), 0.5 / 365)

    FRESHNESS_DAYS = 5    # lastTradeDate 超過此天數視為陳舊

    # ── 共用工具 ───────────────────────────────────────────────────────────────
    def safe_int(val, default=0):
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            return int(val)
        except (TypeError, ValueError):
            return default

    def safe_float(val, default=None):
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            return float(val)
        except (TypeError, ValueError):
            return default

    def parse_trade_date(val):
        """回傳 pd.Timestamp 或 None"""
        try:
            if val is None: return None
            ts = pd.to_datetime(val, utc=False)
            return ts.tz_localize(None) if ts.tzinfo else ts
        except Exception:
            return None

    def is_fresh(last_trade_date_val, cutoff_days=FRESHNESS_DAYS):
        """True = 在 cutoff_days 內有成交"""
        ts = parse_trade_date(last_trade_date_val)
        if ts is None: return False
        age_days = (now_dt - ts).total_seconds() / 86400
        return age_days <= cutoff_days

    def bs_ncdf(x):
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t*(0.319381530 + t*(-0.356563782 + t*(
               1.781477937 + t*(-1.821255978 + t*1.330274429))))
        pdf  = exp(-0.5*x*x) / 2.5066282746
        cdf  = 1.0 - pdf*poly
        return cdf if x >= 0 else 1.0 - cdf

    def bs_price(S, K, T, sigma, r=0.05, is_call=True):
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0: return 0.0
        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        if is_call:
            return S*bs_ncdf(d1) - K*exp(-r*T)*bs_ncdf(d2)
        return K*exp(-r*T)*bs_ncdf(-d2) - S*bs_ncdf(-d1)

    def iv_bisect(price, S, K, T, r=0.05, is_call=True, tol=1e-4):
        if price <= 0 or T <= 0 or K <= 0: return None
        intrinsic = max(S-K, 0) if is_call else max(K-S, 0)
        if price < intrinsic*0.99: return None
        lo, hi = 0.005, 6.0
        for _ in range(60):
            mid = (lo+hi)/2
            est = bs_price(S, K, T, mid, r, is_call)
            if abs(est-price) < tol: return mid
            if est < price: lo = mid
            else:           hi = mid
        v = (lo+hi)/2
        return v if 0.01 <= v <= 5.0 else None

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE A — OI Wall（全鏈條，不需 IV）
    # ══════════════════════════════════════════════════════════════════════════
    oi_calls, oi_puts = [], []

    for _, row in calls_df.iterrows():
        K  = safe_float(row.get('strike', None))
        oi = safe_int(row.get('openInterest', 0))
        vl = safe_int(row.get('volume', 0))
        if K is None or K <= 0 or oi <= 0: continue
        if K < spot * 0.50: continue   # 極端深度 ITM
        oi_calls.append({'strike': K, 'oi': oi, 'volume': vl})

    for _, row in puts_df.iterrows():
        K  = safe_float(row.get('strike', None))
        oi = safe_int(row.get('openInterest', 0))
        vl = safe_int(row.get('volume', 0))
        if K is None or K <= 0 or oi <= 0: continue
        if K < spot * 0.25 or K > spot * 1.05: continue
        oi_puts.append({'strike': K, 'oi': oi, 'volume': vl})

    oi_calls_df = pd.DataFrame(oi_calls).sort_values('oi', ascending=False)                   if oi_calls else pd.DataFrame(columns=['strike','oi','volume'])
    oi_puts_df  = pd.DataFrame(oi_puts).sort_values('oi', ascending=False)                   if oi_puts  else pd.DataFrame(columns=['strike','oi','volume'])

    call_wall_row    = oi_calls_df.iloc[0] if not oi_calls_df.empty else None
    put_wall_row     = oi_puts_df.iloc[0]  if not oi_puts_df.empty  else None
    call_wall_strike = float(call_wall_row['strike']) if call_wall_row is not None else 0.0
    put_wall_strike  = float(put_wall_row['strike'])  if put_wall_row  is not None else 0.0
    pct_to_call_wall = (call_wall_strike - spot)/spot*100 if call_wall_strike > 0 else 0.0

    total_call_oi    = int(oi_calls_df['oi'].sum())     if not oi_calls_df.empty else 0
    total_put_oi     = int(oi_puts_df['oi'].sum())      if not oi_puts_df.empty  else 0
    total_call_vol   = int(oi_calls_df['volume'].sum()) if not oi_calls_df.empty else 0
    total_put_vol    = int(oi_puts_df['volume'].sum())  if not oi_puts_df.empty  else 0
    pc_oi_ratio      = total_put_oi  / max(total_call_oi,  1)
    pc_vol_ratio     = total_put_vol / max(total_call_vol, 1)

    near_atm_call_oi = int(oi_calls_df[
        oi_calls_df['strike'].between(spot*0.9, spot*1.1)
    ]['oi'].sum()) if not oi_calls_df.empty else 0
    atm_concentration = near_atm_call_oi / max(total_call_oi, 1)

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE B — GEX / IV（近期成交，中可信）
    # 入選條件（全部同時滿足）：
    #   1. OI > 0
    #   2. lastTradeDate 在 FRESHNESS_DAYS 內（新鮮度）
    #   3. volume > 0 OR yfinance IV 有效（非 0.00001 佔位符）
    # ══════════════════════════════════════════════════════════════════════════
    gex_results      = []
    total_raw        = 0
    valid_iv_cnt     = 0
    valid_quote_cnt  = 0
    iv_imputed_count = 0
    fresh_count      = 0     # lastTradeDate < FRESHNESS_DAYS
    vol_gt0_count    = 0     # volume > 0

    def parse_for_gex(row, is_call):
        nonlocal total_raw, valid_iv_cnt, valid_quote_cnt, iv_imputed_count
        nonlocal fresh_count, vol_gt0_count
        total_raw += 1

        K      = safe_float(row.get('strike',            None))
        oi     = safe_int  (row.get('openInterest',      0))
        vol    = safe_int  (row.get('volume',            0))
        iv_yf  = safe_float(row.get('impliedVolatility', None))
        bid    = safe_float(row.get('bid',               None))
        ask    = safe_float(row.get('ask',               None))
        last   = safe_float(row.get('lastPrice',         None))
        ltd    = row.get('lastTradeDate', None)

        # ── 1. 基本門禁 ─────────────────────────────────────────────────────
        if K is None or K <= 0: return
        if oi <= 0:             return

        # ── 2. 新鮮度過濾（lastTradeDate）────────────────────────────────────
        fresh = is_fresh(ltd, FRESHNESS_DAYS)
        if not fresh:
            return   # 超過 5 個交易日沒成交，IV 可能是 stale，跳過

        fresh_count += 1
        if vol > 0: vol_gt0_count += 1

        # ── 3. 需要「近期市場定價」的合約才納入 ──────────────────────────────
        has_yf_iv  = (iv_yf is not None and 0.01 <= iv_yf <= 3.0)
        has_volume = (vol > 0)
        has_last   = (last is not None and last > 0)

        if not has_volume and not has_yf_iv:
            return  # 無成交量 + 無有效 IV → 跳過

        if not has_last:
            return  # 連 lastPrice 都沒有，無法反推 IV

        # ── 4. IV 取得（三層，不硬補）───────────────────────────────────────
        iv_use, iv_label = None, ''

        if has_yf_iv:
            iv_use   = iv_yf
            iv_label = 'yf'
            valid_iv_cnt += 1
        elif has_volume:
            iv_imp = iv_bisect(last, spot, K, T_years, is_call=is_call)
            if iv_imp is not None:
                iv_use           = iv_imp
                iv_label         = 'imputed'
                iv_imputed_count += 1
                valid_iv_cnt     += 1

        if iv_use is None:
            return  # 無法取得任何 IV，跳過

        if bid is not None and bid > 0 and ask is not None and ask > 0:
            valid_quote_cnt += 1

        gamma   = calc_gamma(spot, K, T_years, iv_use)
        gex_val = calc_gex(oi, gamma, spot, is_call=is_call)

        gex_results.append({
            'strike':  K,
            'type':    'call' if is_call else 'put',
            'oi':      oi,
            'volume':  vol,
            'iv':      iv_use,
            'iv_src':  iv_label,
            'gamma':   gamma,
            'gex':     gex_val,
        })

    for _, row in calls_df.iterrows(): parse_for_gex(row, True)
    for _, row in puts_df.iterrows():  parse_for_gex(row, False)

    # ── 品質指標計算 ────────────────────────────────────────────────────────
    n_gex        = len(gex_results)
    n_imputed    = iv_imputed_count
    n_yf_iv      = valid_iv_cnt - iv_imputed_count
    iv_ratio     = valid_iv_cnt    / max(total_raw, 1)
    quote_ratio  = valid_quote_cnt / max(total_raw, 1)
    fresh_ratio  = fresh_count     / max(total_raw, 1)
    vol_gt0_ratio= vol_gt0_count   / max(total_raw, 1)
    imp_ratio    = n_imputed       / max(n_gex, 1)
    is_after_hours = (quote_ratio == 0.0)

    # ── 資料健康判定（最小值原則）──────────────────────────────────────────
    DATA_QUALITY  = 'OK'
    quality_notes = []

    # Layer 1 OI Wall
    if len(oi_calls) < 3 and len(oi_puts) < 3:
        DATA_QUALITY = 'UNKNOWN'
        quality_notes.append("OI 鏈條不足，Wall 不可靠")

    # Layer 2 GEX 合約數
    if n_gex < 5:
        DATA_QUALITY = 'UNKNOWN'
        quality_notes.append(f"GEX 合約數僅 {n_gex}（< 5），不可靠")
    elif n_gex < 15:
        if DATA_QUALITY == 'OK': DATA_QUALITY = 'RELAXED'
        quality_notes.append(f"GEX 合約數 {n_gex}（< 15），量值僅供參考")

    # 新鮮度替代品質指標（取代 bid/ask）
    if fresh_ratio < 0.15:
        DATA_QUALITY = 'UNKNOWN'
        quality_notes.append(f"近期成交比例 {fresh_ratio:.0%}（< 15%），數據可能陳舊")

    if vol_gt0_ratio < 0.05:
        if DATA_QUALITY == 'OK': DATA_QUALITY = 'RELAXED'
        quality_notes.append(f"今日 volume>0 比例 {vol_gt0_ratio:.0%}（< 5%），流動性低")

    # IV 覆蓋率
    if iv_ratio < 0.10:
        DATA_QUALITY = 'UNKNOWN'
        quality_notes.append(f"IV 覆蓋率 {iv_ratio:.0%}（< 10%），GEX 失真")

    # 開盤時段 bid/ask 替代判斷
    if not is_after_hours and quote_ratio < 0.15:
        DATA_QUALITY = 'UNKNOWN'
        quality_notes.append(f"開盤時段報價覆蓋率 {quote_ratio:.0%}，數據異常")

    if is_after_hours and DATA_QUALITY == 'OK':
        quality_notes.append(
            f"盤後模式：bid/ask=0 正常 | 新鮮度 {fresh_ratio:.0%} | "
            f"vol>0 {vol_gt0_ratio:.0%}"
        )

    if imp_ratio > 0.6 and DATA_QUALITY == 'OK':
        DATA_QUALITY = 'RELAXED'
        quality_notes.append(
            f"反推 IV 占 {imp_ratio:.0%}（>{60}%），GEX 方向可信、量值打折"
        )

    # ── GEX DataFrame ──────────────────────────────────────────────────────
    GEX_COLS = ['strike','type','oi','volume','iv','iv_src','gamma','gex']
    gex_df = pd.DataFrame(gex_results, columns=GEX_COLS) if gex_results \
             else pd.DataFrame(columns=GEX_COLS)

    if gex_df.empty:
        net_gex_by_strike = pd.Series(dtype=float)
        total_net_gex     = 0.0
        peak_gex_strike   = spot
        avg_atm_iv        = 0.0
    else:
        net_gex_by_strike = gex_df.groupby('strike')['gex'].sum()
        meaningful = gex_df[
            (gex_df['strike'] >= spot*0.30) &
            (gex_df['strike'] <= spot*2.00)
        ]
        total_net_gex = float(meaningful.groupby('strike')['gex'].sum().sum())

        atm_gex = net_gex_by_strike[
            (net_gex_by_strike.index >= spot*0.85) &
            (net_gex_by_strike.index <= spot*1.15)
        ]
        peak_gex_strike = float(atm_gex.abs().idxmax()) if not atm_gex.empty                           else float(net_gex_by_strike.abs().idxmax())

        # ATM IV：只用近 ATM（±5%）且 IV 有效的近期合約
        avg_atm_iv = 0.0
        if 'iv' in gex_df.columns:
            atm_calls = gex_df[
                (gex_df['type'] == 'call') &
                (gex_df['strike'] >= spot*0.95) &
                (gex_df['strike'] <= spot*1.05) &
                (gex_df['iv'] >= 0.01)
            ]
            if not atm_calls.empty:
                avg_atm_iv = float(atm_calls['iv'].mean())
            else:
                valid_iv_df = gex_df[
                    (gex_df['type'] == 'call') & (gex_df['iv'] >= 0.01)
                ]
                avg_atm_iv = float(valid_iv_df['iv'].median()) \
                             if not valid_iv_df.empty else 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # Gamma Flip 掃描（改進版）
    # 1. 只用 |K/spot - 1| < 30% 且在 gex_results 中的子集
    # 2. 先粗掃（10 元步進）找符號翻轉區間
    # 3. 再用二分法在區間內精確定位零點
    # ══════════════════════════════════════════════════════════════════════════
    gamma_flip       = None
    gamma_flip_found = False

    # 子集：只取 ATM ±30% 的合約，減少噪音 + 加速計算
    flip_subset = [r for r in gex_results
                   if abs(r['strike'] / spot - 1.0) < 0.30]

    if flip_subset:
        def net_gex_hypo(hypo_spot):
            total = 0.0
            for r in flip_subset:
                g  = calc_gamma(hypo_spot, r['strike'], T_years, r['iv'])
                gv = calc_gex(r['oi'], g, hypo_spot, is_call=(r['type']=='call'))
                total += gv
            return total

        # 粗掃：spot ±30%，10 元步進
        scan_lo  = int(spot * 0.70 // 10) * 10
        scan_hi  = int(spot * 1.30 // 10) * 10 + 10
        scan_pts = list(range(max(scan_lo, 10), scan_hi + 1, 10))

        # 找所有符號翻轉區間（可能有多個）
        flip_intervals = []
        prev_val  = net_gex_hypo(scan_pts[0])
        prev_sign = 1 if prev_val >= 0 else -1
        for i in range(1, len(scan_pts)):
            p    = scan_pts[i]
            val  = net_gex_hypo(p)
            sign = 1 if val >= 0 else -1
            if sign != prev_sign:
                flip_intervals.append((scan_pts[i-1], p))
            prev_sign = sign
            prev_val  = val

        # 二分法精確定位：在每個粗掃區間內精確找零點
        # 取最靠近 spot 的那個（通常就是最有意義的）
        precise_flips = []
        for (lo, hi) in flip_intervals:
            a, b = float(lo), float(hi)
            fa = net_gex_hypo(a)
            for _ in range(30):      # 30 次二分 = 精度約 (hi-lo)/2^30 元
                mid = (a + b) / 2.0
                fm  = net_gex_hypo(mid)
                if abs(fm) < 1.0:    # 接近零就停（GEX 單位 M，誤差 <1M 夠了）
                    break
                if (fa >= 0) == (fm >= 0):
                    a, fa = mid, fm
                else:
                    b = mid
            precise_flips.append((a + b) / 2.0)

        if precise_flips:
            # 取最靠近 spot 的 flip 點作為主要輸出
            gamma_flip       = float(min(precise_flips, key=lambda x: abs(x - spot)))
            gamma_flip_found = True

    return {
        # Layer 1 — 高可信（OI Wall）
        'oi_calls_df':       oi_calls_df,
        'oi_puts_df':        oi_puts_df,
        'call_wall':         call_wall_row,
        'put_wall':          put_wall_row,
        'call_wall_strike':  call_wall_strike,
        'put_wall_strike':   put_wall_strike,
        'pct_to_call_wall':  pct_to_call_wall,
        'total_call_oi':     total_call_oi,
        'total_put_oi':      total_put_oi,
        'total_call_vol':    total_call_vol,
        'total_put_vol':     total_put_vol,
        'pc_oi_ratio':       float(pc_oi_ratio),
        'pc_vol_ratio':      float(pc_vol_ratio),
        'atm_concentration': float(atm_concentration),
        # Layer 2 — 中可信（GEX / IV）
        'df':                gex_df,
        'calls_only':        gex_df[gex_df['type']=='call'].sort_values('oi',ascending=False)
                             if not gex_df.empty and 'type' in gex_df.columns else gex_df,
        'puts_only':         gex_df[gex_df['type']=='put'].sort_values('oi',ascending=False)
                             if not gex_df.empty and 'type' in gex_df.columns else gex_df,
        'net_gex_by_strike': net_gex_by_strike,
        'peak_gex_strike':   float(peak_gex_strike),
        'gamma_flip':        gamma_flip,
        'gamma_flip_found':  gamma_flip_found,
        'total_net_gex':     float(total_net_gex),
        'avg_atm_iv':        float(avg_atm_iv),
        # Quality metadata
        'data_quality':      DATA_QUALITY,
        'quality_notes':     quality_notes,
        'valid_contracts':   n_gex,
        'total_raw':         total_raw,
        'quote_ratio':       float(quote_ratio),
        'iv_ratio':          float(iv_ratio),
        'fresh_ratio':       float(fresh_ratio),
        'vol_gt0_ratio':     float(vol_gt0_ratio),
        'imp_ratio':         float(imp_ratio),
        'n_imputed':         n_imputed,
        'n_yf_iv':           n_yf_iv,
    }


# ==========================================
# Squeeze Score 計算
# ==========================================
def calc_squeeze_score(gex_data, spot):
    score     = 0
    reasons   = []
    warnings  = []

    pc         = gex_data['pc_oi_ratio']
    pc_vol     = gex_data['pc_vol_ratio']
    atm_conc   = gex_data['atm_concentration']
    dist_cw    = gex_data['pct_to_call_wall']
    avg_iv     = gex_data['avg_atm_iv']
    net_gex    = gex_data['total_net_gex']
    peak_gex_k = gex_data['peak_gex_strike']
    cw_strike  = gex_data['call_wall_strike']

    # ── 1. Put/Call OI 比率（越低 = Call 越擁擠）
    if pc < 0.35:
        score += 35
        reasons.append(f"P/C OI 比率 {pc:.2f} — Call 極度擁擠 ⚡")
    elif pc < 0.55:
        score += 20
        reasons.append(f"P/C OI 比率 {pc:.2f} — Call 偏重")
    elif pc > 1.2:
        score -= 10
        warnings.append(f"P/C OI 比率 {pc:.2f} — Put 偏重，Squeeze 不利")

    # ── 2. Put/Call Volume 比率
    if pc_vol < 0.35:
        score += 20
        reasons.append(f"P/C Vol 比率 {pc_vol:.2f} — 今日 Call 買盤極強")
    elif pc_vol < 0.55:
        score += 10
        reasons.append(f"P/C Vol 比率 {pc_vol:.2f} — Call 成交量偏高")

    # ── 3. ATM OI 集中度
    if atm_conc > 0.45:
        score += 20
        reasons.append(f"ATM OI 集中度 {atm_conc:.1%} — 大量 Call 堆在 ATM 附近")
    elif atm_conc > 0.30:
        score += 10
        reasons.append(f"ATM OI 集中度 {atm_conc:.1%} — Call OI 中等集中")

    # ── 4. 股價距 Call Wall 距離
    if 0 < dist_cw < 3:
        score += 20
        reasons.append(f"距 Call Wall ${cw_strike:.0f} 僅 {dist_cw:.1f}% — 即將觸發造市商追買")
    elif 0 < dist_cw < 7:
        score += 10
        reasons.append(f"距 Call Wall ${cw_strike:.0f} 僅 {dist_cw:.1f}%")
    elif dist_cw < 0:
        score += 15
        reasons.append(f"已突破 Call Wall ${cw_strike:.0f}（+{abs(dist_cw):.1f}%）— Squeeze 已啟動")

    # ── 5. 淨 GEX 方向（負 GEX = 造市商負 Gamma = 追漲追跌）
    if net_gex < -50:
        score += 15
        reasons.append(f"淨 GEX = {net_gex:.1f}M（負 Gamma）— 造市商被迫追漲")
    elif net_gex < 0:
        score += 8
        reasons.append(f"淨 GEX = {net_gex:.1f}M（輕微負 Gamma）")
    else:
        warnings.append(f"淨 GEX = {net_gex:.1f}M（正 Gamma）— 造市商有穩定作用，Squeeze 阻力大")

    # ── 6. ATM IV（高 IV = 市場預期大波動）
    if avg_iv > 0.80:
        score += 10
        reasons.append(f"ATM IV {avg_iv:.0%} — 極高，IV crush 反轉風險同樣極大")
        warnings.append(f"IV {avg_iv:.0%} 極高 — Squeeze 退潮時 IV crush 可造成快速回落")
    elif avg_iv > 0.55:
        score += 5
        reasons.append(f"ATM IV {avg_iv:.0%} — 偏高")

    # ── 7. Gamma Flip vs 股價 ─────────────────────────────────────────────────
    # 正確邏輯：
    #   Step A: 先用 net_gex(spot) 確認「現在是正還是負 Gamma」
    #   Step B: 再用 flip 點描述「何時會切換」
    #   兩者必須一致，不能同時說「正 Gamma」又說「負 Gamma 確認」
    gamma_flip       = gex_data.get('gamma_flip')
    gamma_flip_found = gex_data.get('gamma_flip_found', False)

    # Step A：現在的 Gamma 狀態由 net_gex 符號決定（唯一真相）
    currently_negative_gamma = (net_gex < 0)

    # 原則：越過 Flip 後，Gamma 狀態一定變成當前的「反面」
    # 現在正 → 越過 Flip 後變負；現在負 → 越過 Flip 後變正
    # 越過 Flip 後是當前狀態的反面：現在負→越過後轉正；現在正→越過後轉負
    after_flip_label = "正 Gamma（穩定）" if currently_negative_gamma \
                       else "負 Gamma（追漲追跌）"

    if currently_negative_gamma:
        score += 15
        reasons.append(
            f"淨 GEX {net_gex:.0f}M（負）— 造市商負 Gamma 模式，追漲追跌"
        )
        if gamma_flip_found and gamma_flip is not None:
            if gamma_flip < spot:
                dist = (spot - gamma_flip) / spot * 100
                warnings.append(
                    f"Gamma Flip ${gamma_flip:.0f} 在下方 {dist:.1f}%——"
                    f"跌破後轉{after_flip_label}"
                )
            else:
                dist = (gamma_flip - spot) / spot * 100
                warnings.append(
                    f"Gamma Flip ${gamma_flip:.0f} 在上方 {dist:.1f}%——"
                    f"突破後轉{after_flip_label}"
                )
        else:
            warnings.append("掃描範圍內無 Gamma Flip——持續負 Gamma 環境")

    else:
        # 現在是正 Gamma
        warnings.append(
            f"淨 GEX {net_gex:.0f}M（正）— 造市商正 Gamma，對價格有穩定作用"
        )
        if gamma_flip_found and gamma_flip is not None:
            if gamma_flip > spot:
                dist = (gamma_flip - spot) / spot * 100
                if dist < 5:
                    score += 8
                    reasons.append(
                        f"Gamma Flip ${gamma_flip:.0f} 在上方 {dist:.1f}%——"
                        f"突破後轉{after_flip_label}（追漲力道出現）"
                    )
                else:
                    warnings.append(
                        f"Gamma Flip ${gamma_flip:.0f} 在上方 {dist:.1f}%——"
                        f"目前仍在正 Gamma 保護範圍，突破後轉{after_flip_label}"
                    )
            else:
                # 正 Gamma + Flip 在下方：這是正常情況（例如現在的 MU）
                # 意思：若跌破 Flip，Gamma 由正轉負
                dist = (spot - gamma_flip) / spot * 100
                warnings.append(
                    f"Gamma Flip ${gamma_flip:.0f} 在下方 {dist:.1f}%——"
                    f"跌破後轉{after_flip_label}"
                )
        else:
            warnings.append("掃描範圍內無 Gamma Flip——整段皆正 Gamma，波動抑制")

    score = max(0, min(100, score))
    return score, reasons, warnings

# ==========================================
# 信號解讀
# ==========================================
def interpret_signal(score, gex_data, spot):
    """
    輸出三段式劇本——基於實際 GEX 結構，不依賴 Squeeze Score 高低。
    在報價缺失（盤後）環境下，劇本分析比單一信號更可靠。
    """
    cw       = gex_data['call_wall_strike']
    pw       = gex_data['put_wall_strike']
    pgk      = gex_data['peak_gex_strike']
    iv       = gex_data['avg_atm_iv']
    net      = gex_data['total_net_gex']
    gf       = gex_data.get('gamma_flip')
    gf_found = gex_data.get('gamma_flip_found', False)
    dq       = gex_data.get('data_quality', 'OK')

    # ── 一致性：Gamma 狀態由 net GEX 符號決定 ──────────────────────────────
    is_neg_gamma = net < 0
    gamma_label  = f"負 Gamma（淨 GEX {net:.0f}M）" if is_neg_gamma                    else f"正 Gamma（淨 GEX {net:.0f}M）"

    # ── Gamma Flip 描述（與 net_gex 一致）──────────────────────────────────
    # 原則：越過 Flip 點後，Gamma 狀態一定翻轉成「當前的反面」
    #   現在正 Gamma → 越過 Flip 後轉「負」Gamma（無論 Flip 在上方或下方）
    #   現在負 Gamma → 越過 Flip 後轉「正」Gamma（無論 Flip 在上方或下方）
    # 越過 Flip 後是當前狀態的反面
    after_flip_state = "正 Gamma（穩定）" if is_neg_gamma else "負 Gamma（追漲追跌）"

    if gf_found and gf is not None:
        if gf > spot:
            dist_pct  = (gf - spot) / spot * 100
            direction = "突破"
            location  = f"上方 {dist_pct:.1f}%"
        else:
            dist_pct  = (spot - gf) / spot * 100
            direction = "跌破"
            location  = f"下方 {dist_pct:.1f}%"
        flip_note = (
            f"Gamma Flip ${gf:.0f} 在{location}——"
            f"{direction}後轉{after_flip_state}"
        )
    else:
        if is_neg_gamma:
            flip_note = "掃描範圍內無 Gamma Flip（整段皆為負 Gamma）"
        else:
            flip_note = "掃描範圍內無 Gamma Flip（整段皆為正 Gamma）"

    # ── 品質警告前綴 ────────────────────────────────────────────────────────
    dq_warn = ""
    if dq == 'UNKNOWN':
        dq_warn = "\n  ⚠️  資料品質不足（UNKNOWN），以下劇本僅供方向性參考。\n"
    elif dq == 'RELAXED':
        dq_warn = "\n  ℹ️  數據為盤後/反推 IV，劇本方向可參考，量值僅供估算。\n"

    # ── IV 說明 ────────────────────────────────────────────────────────────
    iv_note = ""
    if iv > 0:
        iv_note = f"  ATM IV 約 {iv:.0%}（盤後反推估算，方向參考即可）\n"

    # ── 三段式劇本 ──────────────────────────────────────────────────────────
    # 以 GEX 結構（cw/pgk/pw）為骨架，不依賴 Squeeze Score
    cw_status = '已突破' if spot >= cw else f'距此 {(cw-spot)/spot*100:.1f}%，尚未確認'
    atm_band  = '內' if pgk <= spot <= cw else '外'
    put_oi    = int(gex_data['total_put_oi'])

    scenario_1 = (
        f"  📈 劇本 1 — 強勢續攻\n"
        f"     條件：有效突破並站穩 ${cw:.0f}（Call Wall）\n"
        f"     含義：${cw:.0f} 的 Call 買家全部獲利，造市商被迫大量追買對沖\n"
        f"     目標：上方無明顯 GEX 牆，動能可自我強化\n"
        f"     現況：{cw_status}"
    )
    scenario_2 = (
        f"  ↔️  劇本 2 — 震盪整理（目前最貼近）\n"
        f"     區間：${pgk:.0f}（Peak GEX 磁吸）～ ${cw:.0f}（Call Wall）\n"
        f"     含義：正 Gamma 環境下造市商在此區間買跌賣漲，自然形成均值回歸\n"
        f"     {flip_note}\n"
        f"     現況：{gamma_label}，股價 ${spot:.0f} 在磁吸帶{atm_band}"
    )
    scenario_3 = (
        f"  📉 劇本 3 — 轉弱\n"
        f"     條件：跌破 ${pgk:.0f} 且站不回來\n"
        f"     下一支撐：${pw:.0f}（Put Wall，OI {put_oi:,} 口）\n"
        f"     含義：Call 買盤退潮，IV crush 加速下跌；"
        f"Put Wall 的造市商買盤在 ${pw:.0f} 有緩衝作用"
    )

    # ── 信號標籤（仍保留給主輸出用）────────────────────────────────────────
    if score >= 70:
        label = "⚡ SQUEEZE 高風險"
    elif score >= 45:
        label = "⚠️  WARMING 中度警示"
    elif score >= 20:
        label = "➡️  NEUTRAL 結構中性"
    else:
        label = "❄️  COOLING Call 退潮"

    verdict = (
        f"{dq_warn}"
        f"  當前 Gamma 狀態：{gamma_label}\n"
        f"{iv_note}"
        "\n"
        f"{scenario_1}\n"
        "\n"
        f"{scenario_2}\n"
        "\n"
        f"{scenario_3}"
    )

    return label, verdict


# ==========================================
# 列印 Bar（ASCII 視覺化）
# ==========================================
def bar(value, max_value, width=20, char='█'):
    if max_value == 0:
        return ' ' * width
    filled = int(value / max_value * width)
    return char * filled + '░' * (width - filled)

# ==========================================
# 主輸出函數
# ==========================================
def print_gex_report(symbol, expiry_choice=None):
    print("=" * 80)
    print(f"⚡  GEX Scanner — Gamma Exposure & Squeeze Detector")
    print("=" * 80)
    print(f"掃描時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目標股票: {symbol.upper()}")

    # ── 1. 取得股價
    print(f"\n📊 取得即時股價...")
    try:
        spot = get_current_price(symbol)
        print(f"   ✅ 當前股價: ${spot:.2f}")
    except Exception as e:
        print(f"   ❌ 股價取得失敗: {e}")
        return None

    # ── 2. 取得選擇權數據
    print(f"\n📋 下載選擇權數據...")
    try:
        calls_df, puts_df, expiry_used, all_expiries = fetch_options_chain(symbol, expiry_choice)
        days_to_expiry = (pd.to_datetime(expiry_used) - pd.Timestamp.now()).days
        print(f"   ✅ 到期日: {expiry_used}（{days_to_expiry} 天後）")
        print(f"   ✅ Call 合約數: {len(calls_df)}  |  Put 合約數: {len(puts_df)}")
        print(f"\n   可用到期日（前5個）:")
        for i, exp in enumerate(all_expiries[:5]):
            d = (pd.to_datetime(exp) - pd.Timestamp.now()).days
            marker = " ◀ 當前分析" if exp == expiry_used else ""
            print(f"   [{i}] {exp}（{d}d）{marker}")
    except Exception as e:
        print(f"   ❌ 選擇權數據失敗: {e}")
        return None

    # ── 3. GEX 分析
    print(f"\n🧮 計算 Gamma Exposure...")
    try:
        gex = analyze_gex(calls_df, puts_df, spot, expiry_used)
        if gex is None:
            print("   ❌ GEX 計算失敗（無有效數據）")
            print("   💡 Debug：嘗試手動指定下一個到期日，例如：")
            print(f"      EXPIRY_CHOICE = '{all_expiries[1] if len(all_expiries) > 1 else 'None'}'")
            return None
        dq      = gex['data_quality']
        n_gex   = gex['valid_contracts']
        n_imp   = gex['n_imputed']
        n_yf    = gex['n_yf_iv']
        n_calls = len(gex['oi_calls_df'])
        n_puts  = len(gex['oi_puts_df'])
        fr      = gex['fresh_ratio']
        vr      = gex['vol_gt0_ratio']
        ir      = gex['imp_ratio']
        print(f"   ✅ [Layer 1 高可信] OI Wall：Call {n_calls} Strike / Put {n_puts} Strike")
        print(f"   ✅ [Layer 2 中可信] GEX/IV ：{n_gex} 筆近期合約")
        print(f"      yf IV {n_yf} 筆 | 反推 {n_imp} 筆（{ir:.0%}）| 新鮮度 {fr:.0%} | vol>0 {vr:.0%}")
        print(f"   📊 資料品質: {dq}  |  IV 覆蓋率: {gex['iv_ratio']:.0%}")
        for note in gex['quality_notes']:
            print(f"   ℹ️  {note}")
    except Exception as e:
        print(f"   ❌ GEX 計算錯誤: {e}")
        import traceback; traceback.print_exc()
        return None

    # ── 4. Squeeze Score
    score, reasons, warnings = calc_squeeze_score(gex, spot)
    signal_label, verdict     = interpret_signal(score, gex, spot)

    # ==========================================
    # 輸出報告
    # ==========================================
    print("\n" + "=" * 80)
    print("📐 關鍵水平總覽  [🔒 Layer 1 高可信 = OI Wall  |  ⚡ Layer 2 中可信 = GEX/Flip]")
    print("=" * 80)

    cw  = gex['call_wall_strike']
    pw  = gex['put_wall_strike']
    pgk = gex['peak_gex_strike']
    cw_oi = int(gex['call_wall']['oi']) if gex['call_wall'] is not None else 0
    pw_oi = int(gex['put_wall']['oi'])  if gex['put_wall']  is not None else 0

    gf        = gex.get('gamma_flip')
    gf_found  = gex.get('gamma_flip_found', False)
    above_cw  = "⬆ 已突破" if spot >= cw  else f"距 {((cw - spot)/spot*100):+.1f}%"
    net_sign  = "負Gamma（追漲追跌）" if gex['total_net_gex'] < 0 else "正Gamma（穩定）"

    print(f"  {'水平':<22} {'價格':>10}  {'說明'}")
    print(f"  {'─'*22} {'─'*10}  {'─'*40}")
    print(f"  {'🔴 Call Wall':<22} ${cw:>9.2f}  OI {cw_oi:,} 口  最大上方阻力  {above_cw}")
    print(f"  {'📍 當前股價':<22} ${spot:>9.2f}  ← 你在這裡  [{net_sign}]")
    if gf_found and gf is not None:
        side  = "上方" if gf > spot else "下方"
        dist  = abs((gf - spot) / spot * 100)
        # 越過 Flip 後轉成當前的反面（由 net_gex 決定，非由位置決定）
        _cur_neg   = gex['total_net_gex'] < 0
        _after     = "正Gamma（穩定）" if _cur_neg else "負Gamma（追漲追跌）"
        _side      = "上方" if gf > spot else "下方"
        _dist      = abs(gf - spot) / spot * 100
        flip_regime = f"{_side} {_dist:.1f}%  越過後轉{_after}"
        print(f"  {'⚡ Gamma Flip':<22} ${gf:>9.0f}  {side} {dist:.1f}%  {flip_regime}")
    else:
        print(f"  {'⚡ Gamma Flip':<22} {'N/A':>10}  掃描範圍內未找到翻轉點")
    print(f"  {'🟡 Peak GEX（磁吸）':<22} ${pgk:>9.2f}  OI 最大 Strike，非 Flip 點")
    print(f"  {'🟢 Put Wall':<22} ${pw:>9.2f}  OI {pw_oi:,} 口  最強下方支撐")

    print("\n" + "=" * 80)
    print("📊 選擇權結構指標  [🔒 OI 來自 Layer 1  |  ⚡ ATM IV 來自 Layer 2]")
    print("=" * 80)
    print(f"  {'P/C OI 比率':<28} {gex['pc_oi_ratio']:.3f}  "
          f"{'⚡ Call 極重' if gex['pc_oi_ratio'] < 0.4 else ('Call 偏重' if gex['pc_oi_ratio'] < 0.65 else '平衡')}")
    print(f"  {'P/C Volume 比率':<28} {gex['pc_vol_ratio']:.3f}  "
          f"{'⚡ 今日 Call 買盤極強' if gex['pc_vol_ratio'] < 0.4 else ('Call 成交量偏高' if gex['pc_vol_ratio'] < 0.65 else '平衡')}")
    print(f"  {'ATM OI 集中度':<28} {gex['atm_concentration']:.1%}  "
          f"{'高度集中在 ATM 附近' if gex['atm_concentration'] > 0.4 else '分散'}")
    print(f"  {'ATM 平均 IV':<28} {gex['avg_atm_iv']:.1%}  "
          f"{'⚠️  IV Crush 風險' if gex['avg_atm_iv'] > 0.7 else ('偏高' if gex['avg_atm_iv'] > 0.5 else '正常')}")
    print(f"  {'總淨 GEX':<28} {gex['total_net_gex']:+.1f}M USD  "
          f"{'負 Gamma（造市商追漲）' if gex['total_net_gex'] < 0 else '正 Gamma（造市商穩定）'}")
    print(f"  {'Total Call OI':<28} {gex['total_call_oi']:,} 口")
    print(f"  {'Total Put OI':<28} {gex['total_put_oi']:,} 口")
    print(f"  {'Call 成交量（今日）':<28} {gex['total_call_vol']:,} 口")
    print(f"  {'Put 成交量（今日）':<28} {gex['total_put_vol']:,} 口")

    print("\n" + "=" * 80)
    print(f"🎯 Gamma Squeeze Score: {score} / 100  │  {signal_label}")
    print("=" * 80)

    score_bar = bar(score, 100, width=40)
    print(f"\n  [{score_bar}] {score}/100\n")

    if reasons:
        print("  📌 觸發條件:")
        for r in reasons:
            print(f"      ✅ {r}")

    if warnings:
        print("\n  ⚠️  風險警示:")
        for w in warnings:
            print(f"      ⚡ {w}")

    print("\n" + "=" * 80)
    print("💡 交易結構判斷")
    print("=" * 80)
    print(f"\n  {verdict}\n")

    # ── Call Wall 明細（前15大）
    print("=" * 80)
    print(f"📞 Call OI 分佈（前15大 Strike）  到期日: {expiry_used}  [🔒 Layer 1 — 全鏈條 OI]")
    print("=" * 80)

    # Call Wall 表格：用 Pipeline A（全 OI 鏈條）
    top_calls   = gex['oi_calls_df'].head(15)
    max_call_oi = int(top_calls['oi'].max()) if not top_calls.empty else 1

    print(f"  {'Strike':>8}  {'OI（口）':>10}  {'OI Bar':<22}  {'Volume':>8}  {'Vol/OI':>6}  {'IV':>6}  {'角色'}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*22}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*12}")

    # IV lookup from Pipeline B (gex_df) keyed by strike
    _gex_iv = {}
    if not gex['df'].empty and 'iv' in gex['df'].columns:
        for _, _r in gex['df'].iterrows():
            _gex_iv[float(_r['strike'])] = float(_r['iv'])

    for _, row in top_calls.iterrows():
        k       = float(row['strike'])
        oi      = int(row['oi'])
        vol     = int(row['volume'])
        iv      = _gex_iv.get(float(row['strike']), 0.0)
        vol_oi  = f"{vol/oi:.2f}" if oi > 0 else "—"
        oi_bar  = bar(oi, max_call_oi, width=22)
        role    = ""
        if k == cw:
            role = "🔴 CALL WALL"
        elif abs(k - spot) / spot < 0.02:
            role = "📍 ATM"
        elif abs(k - pgk) / max(pgk, 1) < 0.01:
            role = "🟡 Peak GEX"
        heat    = " ⚡" if (vol / max(oi, 1)) > 2.0 else ""
        print(f"  ${k:>7.0f}  {oi:>10,}  {oi_bar}  {vol:>8,}  {vol_oi:>6}  {iv*100:>5.0f}%  {role}{heat}")

    # ── Put Wall 明細（前10大）
    print("\n" + "=" * 80)
    print(f"📉 Put OI 分佈（前10大 Strike）  到期日: {expiry_used}  [🔒 Layer 1 — 全鏈條 OI]")
    print("=" * 80)

    # Put Wall 表格：用 Pipeline A（全 OI 鏈條）
    top_puts   = gex['oi_puts_df'].head(10)
    max_put_oi = int(top_puts['oi'].max()) if not top_puts.empty else 1

    print(f"  {'Strike':>8}  {'OI（口）':>10}  {'OI Bar':<22}  {'Volume':>8}  {'IV':>6}  {'角色'}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*22}  {'─'*8}  {'─'*6}  {'─'*12}")

    for _, row in top_puts.iterrows():
        k      = float(row['strike'])
        oi     = int(row['oi'])
        vol    = int(row['volume'])
        iv     = _gex_iv.get(float(row['strike']), 0.0)
        oi_bar = bar(oi, max_put_oi, width=22)
        role   = "🟢 PUT WALL" if k == pw else ("📍 下方支撐" if k < spot else "")
        print(f"  ${k:>7.0f}  {oi:>10,}  {oi_bar}  {vol:>8,}  {iv*100:>5.0f}%  {role}")

    # ── 淨 GEX by Strike（最重要的10個）
    print("\n" + "=" * 80)
    print(f"🧲 淨 GEX 分佈（最重要10個 Strike）  [⚡ Layer 2 — {gex['data_quality']}]")
    print("=" * 80)
    print("   正值 = 造市商正 Gamma（穩定）  |  負值 = 負 Gamma（追漲追跌）")
    print("   ⚠️  GEX 量值僅供方向性參考，精確數值受 IV 品質影響")
    print()

    net_gex_s = gex['net_gex_by_strike'].sort_values(key=abs, ascending=False).head(10)
    max_abs   = float(net_gex_s.abs().max()) if not net_gex_s.empty else 1

    for strike, ng in net_gex_s.items():
        b    = bar(abs(ng), max_abs, width=20, char='▓' if ng > 0 else '▒')
        sign = '+' if ng >= 0 else ''
        role = ""
        if strike == cw:  role = "  ← Call Wall"
        if strike == pgk: role = "  ← Peak GEX (磁吸點)"
        if strike == pw:  role = "  ← Put Wall"
        print(f"  ${strike:>7.0f}  {sign}{ng:>8.1f}M  [{b}]{role}")

    # ── GEX 曲線圖（matplotlib）──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        ngbs = gex['net_gex_by_strike']

        if not ngbs.empty:
            lo_price = min(spot * 0.82, pw * 0.97 if pw else spot * 0.82)
            hi_price = max(spot * 1.18, cw * 1.02 if cw else spot * 1.18)

            mask   = (ngbs.index >= lo_price) & (ngbs.index <= hi_price)
            subset = ngbs[mask].sort_index()

            if len(subset) >= 3:
                strikes = subset.index.values.astype(float)
                values  = subset.values.astype(float)
                colors  = ['#e05c5c' if v < 0 else '#5ba85b' for v in values]
                bar_w   = max((hi_price - lo_price) / max(len(strikes), 1) * 0.75, 3)

                fig, ax = plt.subplots(figsize=(13, 5))
                fig.patch.set_facecolor('#0f1520')
                ax.set_facecolor('#0f1520')

                ax.bar(strikes, values, width=bar_w, color=colors, alpha=0.85, zorder=3)
                ax.plot(strikes, values, color='#88bbff', linewidth=1.2,
                        alpha=0.75, zorder=4, marker='o', markersize=3)
                ax.axhline(0, color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=2)

                # 負 / 正 GEX 填色
                neg_m = values < 0
                pos_m = values >= 0
                if neg_m.any():
                    ax.fill_between(strikes, values, 0, where=neg_m,
                                    color='#e05c5c', alpha=0.18, zorder=1)
                if pos_m.any():
                    ax.fill_between(strikes, values, 0, where=pos_m,
                                    color='#5ba85b', alpha=0.10, zorder=1)

                # 關鍵垂直線
                vline_defs = [
                    (spot,  '#4a9eff', f'現價 ${spot:.0f}',       2.0, 'solid'),
                    (cw,    '#e05c5c', f'Call Wall ${cw:.0f}',    1.5, 'dashed'),
                    (pw,    '#5ba85b', f'Put Wall ${pw:.0f}',     1.5, 'dashed'),
                    (pgk,   '#ffcc44', f'Peak GEX ${pgk:.0f}',   1.2, 'dotted'),
                ]
                if gf_found and gf is not None:
                    vline_defs.append(
                        (gf, '#ff88ff', f'Flip ${gf:.0f}', 1.2, 'dashdot')
                    )

                ymax = max(abs(values)) if len(values) else 1
                for (price, col, lbl, lw, ls) in vline_defs:
                    if price and lo_price <= price <= hi_price:
                        ax.axvline(price, color=col, linewidth=lw,
                                   linestyle=ls, alpha=0.85, zorder=5)
                        ax.text(price, ymax * 0.90, lbl, color=col,
                                fontsize=7, ha='center', va='top', rotation=90,
                                bbox=dict(boxstyle='round,pad=0.15',
                                          facecolor='#0f1520',
                                          edgecolor=col, alpha=0.8))

                # 標注絕對值前 6 大的 Strike
                top_idx = np.argsort(np.abs(values))[-min(6, len(values)):]
                for idx in top_idx:
                    v, k = values[idx], strikes[idx]
                    off  = ymax * 0.06
                    ax.text(k, v + (off if v >= 0 else -off),
                            f'{v:+.0f}M', color='white', fontsize=6.5,
                            ha='center', va='bottom' if v >= 0 else 'top',
                            fontweight='bold')

                ax.set_xlabel('Strike Price ($)', color='#8899aa', fontsize=9)
                ax.set_ylabel('Net GEX (M USD)', color='#8899aa', fontsize=9)
                dq_tag = gex['data_quality']
                ax.set_title(
                    f'{symbol.upper()} Net GEX by Strike  '
                    f'[到期 {expiry_used}  |  現價 ${spot:.0f}  |  品質 {dq_tag}]',
                    color='#c8d8f0', fontsize=11, fontweight='bold', pad=12
                )
                ax.tick_params(colors='#8899aa', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1e2d42')
                ax.grid(axis='y', color='#1e2d42', linewidth=0.5, alpha=0.5)
                ax.set_xlim(lo_price - 15, hi_price + 15)

                legend_items = [
                    mpatches.Patch(color='#5ba85b', alpha=0.7, label='正 GEX（造市商穩定）'),
                    mpatches.Patch(color='#e05c5c', alpha=0.7, label='負 GEX（造市商追漲追跌）'),
                ]
                ax.legend(handles=legend_items, loc='upper right',
                          facecolor='#151e2d', edgecolor='#1e2d42',
                          labelcolor='#c8d8f0', fontsize=8)

                plt.tight_layout()
                chart_path = f'{symbol.upper()}_GEX_chart.png'
                plt.savefig(chart_path, dpi=140, bbox_inches='tight',
                            facecolor='#0f1520')
                plt.close()

                n_neg = int(neg_m.sum())
                neg_desc = '孤島（1～2 個 Strike）' if n_neg <= 2 \
                           else f'連續區間（{n_neg} 個 Strike）'
                print(f"\n   📊 GEX 曲線圖已儲存：{chart_path}")
                print(f"      覆蓋 ${lo_price:.0f}～${hi_price:.0f} | "
                      f"負 GEX：{neg_desc}")
            else:
                print("\n   ℹ️  GEX 數據點不足（< 3），跳過圖表")
    except ImportError:
        print("\n   ℹ️  請安裝 matplotlib：pip install matplotlib")
    except Exception as _chart_err:
        print(f"\n   ⚠️  圖表生成失敗: {_chart_err}")

    print("\n" + "=" * 80)
    print("⚠️  風險提示")
    print("=" * 80)
    print("   • 本分析由簡化 GEX 公式計算，僅供學習與研究參考，不構成投資建議")
    print("   • GEX 計算基於 Black-Scholes 簡化假設，實際造市商部位無法從公開數據完全還原")
    print("   • 選擇權數據存在延遲（yfinance 約 15 分鐘延遲）")
    print("   • Gamma Squeeze 可快速逆轉，IV crush 下跌速度可能與上漲速度相同")
    print("   • 股市有風險，投資需謹慎")
    print("=" * 80)

    return {
        'symbol':           symbol,
        'spot':             spot,
        'expiry':           expiry_used,
        'call_wall':        cw,
        'put_wall':         pw,
        'peak_gex_strike':  pgk,
        'total_net_gex':    gex['total_net_gex'],
        'pc_oi_ratio':      gex['pc_oi_ratio'],
        'avg_atm_iv':       gex['avg_atm_iv'],
        'squeeze_score':    score,
        'signal_label':     signal_label,
    }

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # ── 參數設定 ──────────────────────────────────────────
    SYMBOL         = 'MU'      # 股票代號，可改 NVDA / AAPL / TSLA 等
    EXPIRY_CHOICE  = None      # None = 最近到期日
                               # 或指定日期字串，例如 '2026-05-30'
    # ─────────────────────────────────────────────────────

    result = print_gex_report(SYMBOL, EXPIRY_CHOICE)

    if result:
        print(f"\n✅ GEX 掃描完成!")
        print(f"\n📱 快速摘要:")
        print(f"   股票:          {result['symbol']}")
        print(f"   當前股價:      ${result['spot']:.2f}")
        print(f"   到期日:        {result['expiry']}")
        print(f"   Call Wall:     ${result['call_wall']:.0f}")
        print(f"   Put Wall:      ${result['put_wall']:.0f}")
        print(f"   Peak GEX:      ${result['peak_gex_strike']:.0f}（磁吸點）")
        print(f"   P/C OI:        {result['pc_oi_ratio']:.3f}")
        print(f"   ATM IV:        {result['avg_atm_iv']:.1%}")
        print(f"   Squeeze Score: {result['squeeze_score']} / 100")
        print(f"   信號:          {result['signal_label']}")
    else:
        print("\n❌ GEX 掃描失敗")
