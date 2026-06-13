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
        target_expiry = all_expiries[0]
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
    計算每個 Strike 的 GEX，找出：
    - Call Wall（最大 Call OI）
    - Put Wall（最大 Put OI）
    - Peak GEX Strike（Gamma 最大 = 磁吸點）
    - 淨 GEX（正 = 造市商正 Gamma，負 = 負 Gamma）
    """
    expiry_dt   = pd.to_datetime(expiry_str)
    now_dt      = pd.Timestamp.now()
    T_years     = max((expiry_dt - now_dt).days / 365, 0.001)

    results = []

    # ── Calls ──
    for _, row in calls_df.iterrows():
        K   = float(row.get('strike', 0))
        oi  = int(row.get('openInterest', 0) or 0)
        vol = int(row.get('volume', 0) or 0)
        iv  = float(row.get('impliedVolatility', 0) or 0)
        bid = float(row.get('bid', 0) or 0)
        ask = float(row.get('ask', 0) or 0)

        if K <= 0 or oi == 0:
            continue

        iv_use  = iv if iv > 0.01 else 0.5
        gamma   = calc_gamma(spot, K, T_years, iv_use)
        gex_val = calc_gex(oi, gamma, spot, is_call=True)

        results.append({
            'strike':    K,
            'type':      'call',
            'oi':        oi,
            'volume':    vol,
            'iv':        iv,
            'bid':       bid,
            'ask':       ask,
            'gamma':     gamma,
            'gex':       gex_val,
        })

    # ── Puts ──
    for _, row in puts_df.iterrows():
        K   = float(row.get('strike', 0))
        oi  = int(row.get('openInterest', 0) or 0)
        vol = int(row.get('volume', 0) or 0)
        iv  = float(row.get('impliedVolatility', 0) or 0)
        bid = float(row.get('bid', 0) or 0)
        ask = float(row.get('ask', 0) or 0)

        if K <= 0 or oi == 0:
            continue

        iv_use  = iv if iv > 0.01 else 0.5
        gamma   = calc_gamma(spot, K, T_years, iv_use)
        gex_val = calc_gex(oi, gamma, spot, is_call=False)

        results.append({
            'strike':    K,
            'type':      'put',
            'oi':        oi,
            'volume':    vol,
            'iv':        iv,
            'bid':       bid,
            'ask':       ask,
            'gamma':     gamma,
            'gex':       gex_val,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return None

    # 各 Strike 淨 GEX
    net_gex_by_strike = df.groupby('strike')['gex'].sum()

    # Call / Put 分開
    calls_only = df[df['type'] == 'call'].sort_values('oi', ascending=False)
    puts_only  = df[df['type'] == 'put'].sort_values('oi', ascending=False)

    call_wall_row  = calls_only.iloc[0] if not calls_only.empty else None
    put_wall_row   = puts_only.iloc[0]  if not puts_only.empty  else None

    # Peak GEX Strike：ATM 附近（±15%）淨 GEX 絕對值最大
    atm_mask = (df['strike'] >= spot * 0.85) & (df['strike'] <= spot * 1.15)
    atm_df   = df[atm_mask]
    if not atm_df.empty:
        peak_gex_strike = net_gex_by_strike[
            net_gex_by_strike.index.isin(atm_df['strike'])
        ].abs().idxmax()
    else:
        peak_gex_strike = net_gex_by_strike.abs().idxmax()

    # 總淨 GEX
    total_net_gex  = net_gex_by_strike.sum()
    total_call_oi  = int(calls_only['oi'].sum())
    total_put_oi   = int(puts_only['oi'].sum())
    total_call_vol = int(calls_only['volume'].sum())
    total_put_vol  = int(puts_only['volume'].sum())
    pc_oi_ratio    = total_put_oi  / (total_call_oi  + 1)
    pc_vol_ratio   = total_put_vol / (total_call_vol + 1)

    # ATM IV 平均
    atm_calls = calls_only[
        (calls_only['strike'] >= spot * 0.95) &
        (calls_only['strike'] <= spot * 1.05)
    ]
    avg_atm_iv = float(atm_calls['iv'].mean()) if not atm_calls.empty else 0

    # 近 ATM Call OI 集中度
    near_atm_call_oi = int(calls_only[
        calls_only['strike'].between(spot * 0.9, spot * 1.1)
    ]['oi'].sum())
    atm_concentration = near_atm_call_oi / (total_call_oi + 1)

    # Call Wall 距離 %
    call_wall_strike = float(call_wall_row['strike']) if call_wall_row is not None else 0
    put_wall_strike  = float(put_wall_row['strike'])  if put_wall_row  is not None else 0
    pct_to_call_wall = (call_wall_strike - spot) / spot * 100 if call_wall_strike > 0 else 0

    return {
        'df':               df,
        'calls_only':       calls_only,
        'puts_only':        puts_only,
        'net_gex_by_strike': net_gex_by_strike,
        'call_wall':        call_wall_row,
        'put_wall':         put_wall_row,
        'call_wall_strike': call_wall_strike,
        'put_wall_strike':  put_wall_strike,
        'peak_gex_strike':  float(peak_gex_strike),
        'total_net_gex':    float(total_net_gex),
        'total_call_oi':    total_call_oi,
        'total_put_oi':     total_put_oi,
        'total_call_vol':   total_call_vol,
        'total_put_vol':    total_put_vol,
        'pc_oi_ratio':      float(pc_oi_ratio),
        'pc_vol_ratio':     float(pc_vol_ratio),
        'avg_atm_iv':       float(avg_atm_iv),
        'atm_concentration': float(atm_concentration),
        'pct_to_call_wall': float(pct_to_call_wall),
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

    # ── 7. 股價 vs Peak GEX
    if spot > peak_gex_k:
        score += 5
        reasons.append(f"股價 ${spot:.2f} 已超過 Peak GEX ${peak_gex_k:.0f} — 進入負 Gamma 加速區")
    else:
        warnings.append(f"股價低於 Peak GEX ${peak_gex_k:.0f} — 仍在正 Gamma 磁吸帶內")

    score = max(0, min(100, score))
    return score, reasons, warnings

# ==========================================
# 信號解讀
# ==========================================
def interpret_signal(score, gex_data, spot):
    cw  = gex_data['call_wall_strike']
    pw  = gex_data['put_wall_strike']
    pgk = gex_data['peak_gex_strike']
    iv  = gex_data['avg_atm_iv']
    net = gex_data['total_net_gex']

    if score >= 75:
        label   = "⚡ SQUEEZE 高風險"
        verdict = (
            f"Call OI 極度擁擠，造市商處於負 Gamma 區間，Delta Hedge 追買壓力大。\n"
            f"      股價一旦突破 ${cw:.0f} 關鍵履約價，避險買盤可能自我強化形成 Gamma Squeeze。\n"
            f"      但 IV 已在 {iv:.0%} 高位——Squeeze 退潮時 IV crush + delta unwind 的下跌速度同樣驚人。\n"
            f"      操作：不追高，若已持有可設移動止損；等回落至 Peak GEX ${pgk:.0f} 附近再評估。"
        )
    elif score >= 50:
        label   = "⚠️  WARMING 中度警示"
        verdict = (
            f"Call 結構偏多但尚未達極端（P/C={gex_data['pc_oi_ratio']:.2f}）。\n"
            f"      ${cw:.0f} 若出現大量成交量突破，可能觸發造市商追加 Delta Hedge 買盤。\n"
            f"      若 Call 流量降溫（Vol/OI 下降），推力會迅速消失，回調與 IV 崩跌風險升高。\n"
            f"      操作：觀察 ${cw:.0f} 能否有效突破；止損設 Peak GEX ${pgk:.0f} 以下。"
        )
    elif score >= 25:
        label   = "➡️  NEUTRAL 結構中性"
        verdict = (
            f"P/C 比率 {gex_data['pc_oi_ratio']:.2f} 偏平衡，淨 GEX {net:.1f}M。\n"
            f"      造市商目前可能處於正 Gamma 區間，對股價有自然穩定作用。\n"
            f"      股價傾向在 Peak GEX ${pgk:.0f}（磁吸點）附近震盪。\n"
            f"      劇烈 Gamma Squeeze 概率低，Put Wall ${pw:.0f} 是主要下方支撐。"
        )
    else:
        label   = "❄️  COOLING Call 退潮"
        verdict = (
            f"P/C 比率 {gex_data['pc_oi_ratio']:.2f} 偏高，Call 買盤疲弱。\n"
            f"      若之前有 Squeeze 行情，現在可能進入退潮——造市商 Delta Unwind（賣出對沖股票）。\n"
            f"      IV crush 風險高，股價可能快速回落至 Peak GEX ${pgk:.0f} 甚至 Put Wall ${pw:.0f}。\n"
            f"      操作：不宜做多，已持有者考慮減倉。"
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
            return None
        print(f"   ✅ 分析完成，共 {len(gex['df'])} 筆有效合約")
    except Exception as e:
        print(f"   ❌ GEX 計算錯誤: {e}")
        return None

    # ── 4. Squeeze Score
    score, reasons, warnings = calc_squeeze_score(gex, spot)
    signal_label, verdict     = interpret_signal(score, gex, spot)

    # ==========================================
    # 輸出報告
    # ==========================================
    print("\n" + "=" * 80)
    print("📐 關鍵水平總覽")
    print("=" * 80)

    cw  = gex['call_wall_strike']
    pw  = gex['put_wall_strike']
    pgk = gex['peak_gex_strike']
    cw_oi = int(gex['call_wall']['oi']) if gex['call_wall'] is not None else 0
    pw_oi = int(gex['put_wall']['oi'])  if gex['put_wall']  is not None else 0

    above_cw  = "⬆ 已突破" if spot >= cw  else f"距 {((cw - spot)/spot*100):+.1f}%"
    above_pgk = "⬆ 高於磁吸點" if spot > pgk else f"距 {((pgk - spot)/spot*100):+.1f}%（磁吸引力向下）"

    print(f"  {'水平':<20} {'價格':>10}  {'說明'}")
    print(f"  {'─'*20} {'─'*10}  {'─'*35}")
    print(f"  {'🔴 Call Wall':<20} ${cw:>9.2f}  OI {cw_oi:,} 口  最大上方阻力  {above_cw}")
    print(f"  {'📍 當前股價':<20} ${spot:>9.2f}  ← 你在這裡")
    print(f"  {'🟡 Peak GEX（磁吸）':<20} ${pgk:>9.2f}  {above_pgk}")
    print(f"  {'🟢 Put Wall':<20} ${pw:>9.2f}  OI {pw_oi:,} 口  最強下方支撐")

    print("\n" + "=" * 80)
    print("📊 選擇權結構指標")
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
    print(f"📞 Call OI 分佈（前15大 Strike）  到期日: {expiry_used}")
    print("=" * 80)

    top_calls   = gex['calls_only'].head(15)
    max_call_oi = int(top_calls['oi'].max()) if not top_calls.empty else 1

    print(f"  {'Strike':>8}  {'OI（口）':>10}  {'OI Bar':<22}  {'Volume':>8}  {'Vol/OI':>6}  {'IV':>6}  {'角色'}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*22}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*12}")

    for _, row in top_calls.iterrows():
        k       = float(row['strike'])
        oi      = int(row['oi'])
        vol     = int(row['volume'])
        iv      = float(row['iv'])
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
    print(f"📉 Put OI 分佈（前10大 Strike）  到期日: {expiry_used}")
    print("=" * 80)

    top_puts   = gex['puts_only'].head(10)
    max_put_oi = int(top_puts['oi'].max()) if not top_puts.empty else 1

    print(f"  {'Strike':>8}  {'OI（口）':>10}  {'OI Bar':<22}  {'Volume':>8}  {'IV':>6}  {'角色'}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*22}  {'─'*8}  {'─'*6}  {'─'*12}")

    for _, row in top_puts.iterrows():
        k      = float(row['strike'])
        oi     = int(row['oi'])
        vol    = int(row['volume'])
        iv     = float(row['iv'])
        oi_bar = bar(oi, max_put_oi, width=22)
        role   = "🟢 PUT WALL" if k == pw else ("📍 下方支撐" if k < spot else "")
        print(f"  ${k:>7.0f}  {oi:>10,}  {oi_bar}  {vol:>8,}  {iv*100:>5.0f}%  {role}")

    # ── 淨 GEX by Strike（最重要的10個）
    print("\n" + "=" * 80)
    print("🧲 淨 GEX 分佈（最重要10個 Strike）")
    print("=" * 80)
    print("   正值 = 造市商正 Gamma（穩定）  |  負值 = 負 Gamma（追漲追跌）")
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