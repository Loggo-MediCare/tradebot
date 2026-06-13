"""
technical_indicator_score.py
============================
共用技術指標評分模組 — 供所有 PPO 信號檔 import

calculate_indicator_score(latest, df)
  -> (score: float, reasons: list[str], warnings: list[str])

評分規則:
  +3  均線多頭排列  SMA10 > SMA30 > SMA50
  +2  MACD 金叉    macd > macd_signal
  +2  RSI 超賣反彈  RSI < 35
  +2  KD 金叉      K 低位上穿 D (K < 60)
  +1  RSI 健康區    40 < RSI < 70
  +1  布林帶強勢    bb_position > 80
  -1  RSI 超買      RSI > 75
  -2  均線空頭      SMA10 < SMA30
  -2  KD 死叉      K 高位下穿 D (K > 60)

有效範圍: [-4, +11]
每 1 分 → eff_prob 調整 2%  (呼叫方自行乘以 2)
"""

def calculate_indicator_score(latest, df):
    """
    Parameters
    ----------
    latest : pd.Series  df.iloc[-1]
    df     : pd.DataFrame  含完整技術指標欄位

    Returns
    -------
    score    : float   技術指標淨得分
    reasons  : list    看漲理由（已含分數標示）
    warnings : list    看跌警告（已含分數標示）
    """
    score    = 0.0
    reasons  = []
    warnings = []

    try:
        rsi         = float(latest['rsi'])
        macd        = float(latest['macd'])
        macd_sig    = float(latest['macd_signal'])
        sma_10      = float(latest['sma_10'])
        sma_30      = float(latest['sma_30'])
        sma_50      = float(latest['sma_50'])
        K           = float(latest['K'])
        D           = float(latest['D'])
        bb_pos      = float(latest['bb_position'])
    except Exception as e:
        return 0.0, [], [f"指標讀取失敗: {e}"]

    # ── 均線排列 ──────────────────────────────────────────
    if sma_10 > sma_30 > sma_50:
        score += 3
        reasons.append(f"均線多頭排列 SMA10>SMA30>SMA50 (+3)")
    elif sma_10 < sma_30:
        score -= 2
        warnings.append(f"均線空頭 SMA10<SMA30 (-2)")

    # ── MACD ──────────────────────────────────────────────
    if macd > macd_sig:
        score += 2
        reasons.append(f"MACD 金叉 (+2)")
    # (死叉不額外扣分，由均線空頭已覆蓋)

    # ── RSI ───────────────────────────────────────────────
    if rsi < 35:
        score += 2
        reasons.append(f"RSI 超賣區 {rsi:.1f}，反彈機會 (+2)")
    elif 40 < rsi < 70:
        score += 1
        reasons.append(f"RSI 健康區 {rsi:.1f} (+1)")
    elif rsi > 75:
        score -= 1
        warnings.append(f"RSI 超買 {rsi:.1f} (-1)")

    # ── KD ────────────────────────────────────────────────
    try:
        prev_K = float(df['K'].iloc[-2])
        prev_D = float(df['D'].iloc[-2])
        # 金叉：K 從低位上穿 D
        if K > D and prev_K <= prev_D and K < 60:
            score += 2
            reasons.append(f"KD 金叉 低位上穿 K={K:.1f} (+2)")
        elif K > D and K < 50:
            score += 1
            reasons.append(f"KD 多頭低位 K={K:.1f} (+1)")
        # 死叉：K 從高位下穿 D
        elif K < D and prev_K >= prev_D and K > 60:
            score -= 2
            warnings.append(f"KD 死叉 高位下穿 K={K:.1f} (-2)")
    except Exception:
        # 若取不到前一天資料，只看當前
        if K > D and K < 60:
            score += 1
            reasons.append(f"KD 多頭 K={K:.1f} (+1)")

    # ── 布林帶 ────────────────────────────────────────────
    if bb_pos > 80:
        score += 1
        reasons.append(f"布林帶強勢區 {bb_pos:.1f}% (+1)")

    return score, reasons, warnings
