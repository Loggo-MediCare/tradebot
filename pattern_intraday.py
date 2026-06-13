"""
盤中型態識別引擎 (Intraday)
================
pattern_engine.py 的盤中版本，識別開盤初期動能型態
需搭配 1 分鐘 K 線資料使用 (df 應為當日開盤至目前為止的K棒)

支援型態:
1. OPENING_3BAR_BREAKOUT - 開盤三分鐘強勢突破 (開盤搶攻)
2. ORB                   - 開盤區間突破 (收盤價突破前5根高點)
3. VWAP_RECLAIM          - 站回VWAP (多頭轉強)
4. HIGH_OF_DAY_BREAK     - 創日內新高
5. VOLUME_SURGE          - 爆量 (RVOL > 3)
6. GAMMA_SQUEEZE_SETUP   - 選擇權逼空型態 (需傳入 gex_score，> 70 觸發)

用法:
    patterns = pattern_intraday_engine(df, gex_score=None)
    signal = get_intraday_pattern_signal(df, gex_score=None)
"""

import numpy as np
import pandas as pd


# ======================================================
# 6. Opening 3-Minute Momentum Breakout (開盤三分鐘強勢突破)
# ======================================================

def detect_opening_3bar_breakout(df):
    """
    開盤三分鐘強勢突破

    EasyLanguage 條件:
    1. 前三根1分鐘K連續上漲，且每根都收紅 (close > open)
    2. 第三根成交量 > 前兩根平均量 × 2
    3. 第三根收盤價 = 當日最高價

    Returns:
        bool
    """
    if len(df) < 3:
        return False

    open_ = df['open'].values
    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values

    # 開盤前三根K
    c1 = close[0]
    c2 = close[1]
    c3 = close[2]

    # 條件1：連三紅 (收盤價遞增)
    three_up = (
        c2 >= c1 and
        c3 >= c2
    )

    # 條件1b：每根都是紅K (收盤 > 開盤)
    three_green = (
        close[0] > open_[0]
        and close[1] > open_[1]
        and close[2] > open_[2]
    )

    # 條件2：爆量 (第三根量 > 前兩根平均量 × 2)
    avg_vol = np.mean(volume[:2])

    volume_breakout = (
        volume[2] > avg_vol * 2
    )

    # 條件3：收盤創當日新高
    close_at_high = (
        close[2] >= np.max(high[:3])
    )

    return (
        three_up
        and three_green
        and volume_breakout
        and close_at_high
    )


# ======================================================
# 2. Opening Range Breakout (開盤區間突破)
# ======================================================

def detect_orb(df):
    """
    Opening Range Breakout

    開盤區間 = 前5根K的最高點
    收盤價突破開盤區間高點 → True

    Returns:
        bool
    """
    if len(df) < 10:
        return False

    opening_high = df.iloc[:5]['high'].max()

    return (
        df['close'].iloc[-1]
        > opening_high
    )


# ======================================================
# 3. VWAP Reclaim (站回VWAP，多頭轉強)
# ======================================================

def detect_vwap_reclaim(df):
    """
    站回 VWAP

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    站回條件：前一根收盤 < VWAP，且最新一根收盤 >= VWAP (由下往上穿越)

    Returns:
        bool
    """
    if len(df) < 2:
        return False

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values

    typical_price = (high + low + close) / 3
    with np.errstate(invalid='ignore', divide='ignore'):
        vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)

    return (
        close[-2] < vwap[-2]
        and close[-1] >= vwap[-1]
    )


# ======================================================
# 4. High of Day Break (創日內新高)
# ======================================================

def detect_high_of_day_break(df):
    """
    創日內新高：最新收盤價突破截至前一根的當日最高價

    Returns:
        bool
    """
    if len(df) < 2:
        return False

    high = df['high'].values
    close = df['close'].values

    prev_hod = np.max(high[:-1])

    return close[-1] > prev_hod


# ======================================================
# 5. Volume Surge (爆量，RVOL > 3)
# ======================================================

def relative_volume(df):
    """
    RVOL (Relative Volume)

    RVOL = 今日累積成交量 / 20期平均單根成交量

    Returns:
        float
    """
    current = df['volume'].sum()
    historical = (
        df['volume']
        .rolling(20)
        .mean()
        .iloc[-1]
    )
    return current / historical


# ======================================================
# Master Intraday Pattern Engine
# ======================================================

def pattern_intraday_engine(df, gex_score=None):
    """
    主盤中型態識別引擎

    Args:
        df: 當日1分鐘K棒
        gex_score: Gamma Squeeze 機率分數 (0-100，可選)

    Returns:
        list: 檢測到的型態列表
    """
    patterns = []

    if detect_opening_3bar_breakout(df):
        patterns.append("OPENING_3BAR_BREAKOUT")

    if detect_orb(df):
        patterns.append("ORB")

    if detect_vwap_reclaim(df):
        patterns.append("VWAP_RECLAIM")

    if detect_high_of_day_break(df):
        patterns.append("HIGH_OF_DAY_BREAK")

    rvol = relative_volume(df)
    if rvol > 3:
        patterns.append("VOLUME_SURGE")

    if gex_score is not None and gex_score > 70:
        patterns.append("GAMMA_SQUEEZE_SETUP")

    return patterns


def get_intraday_pattern_signal(df, gex_score=None):
    """
    獲取盤中型態信號 (用於整合到交易系統)

    Args:
        df: 當日1分鐘K棒
        gex_score: Gamma Squeeze 機率分數 (0-100，可選)

    Returns:
        dict: {
            'patterns': list,
            'bullish': list,
            'bearish': list,
            'neutral': list,
            'score_adjustment': int,
            'signal_text': str
        }
    """
    result = {
        'patterns': [],
        'bullish': [],
        'bearish': [],
        'neutral': [],
        'score_adjustment': 0,
        'signal_text': ''
    }

    patterns = pattern_intraday_engine(df, gex_score=gex_score)
    result['patterns'] = patterns

    if not patterns:
        return result

    bullish_patterns = ['OPENING_3BAR_BREAKOUT', 'ORB', 'VWAP_RECLAIM', 'HIGH_OF_DAY_BREAK', 'GAMMA_SQUEEZE_SETUP']
    bearish_patterns = []

    for p in patterns:
        if p in bullish_patterns:
            result['bullish'].append(p)
        elif p in bearish_patterns:
            result['bearish'].append(p)
        else:
            result['neutral'].append(p)

    # 計算評分調整
    pattern_score_map = {
        'OPENING_3BAR_BREAKOUT': (10, "開盤三分鐘強勢突破"),
        'ORB':                   (10, "開盤區間突破"),
        'VWAP_RECLAIM':          (8, "站回VWAP"),
        'HIGH_OF_DAY_BREAK':     (8, "創日內新高"),
        'VOLUME_SURGE':          (0, "爆量"),
        'GAMMA_SQUEEZE_SETUP':   (15, "選擇權逼空型態"),
    }

    score = 0
    texts = []
    for p in patterns:
        adj, text = pattern_score_map[p]
        score += adj
        texts.append(text)

    result['score_adjustment'] = score
    result['signal_text'] = " + ".join(texts) if texts else ""

    return result


def format_intraday_pattern_output(signal):
    """格式化輸出"""
    if not signal['patterns']:
        return "   未檢測到明顯盤中型態"

    lines = []

    if signal['bullish']:
        lines.append(f"   看漲型態: {', '.join(signal['bullish'])}")

    if signal['bearish']:
        lines.append(f"   看跌型態: {', '.join(signal['bearish'])}")

    if signal['neutral']:
        lines.append(f"   中性型態: {', '.join(signal['neutral'])}")

    if signal['score_adjustment'] != 0:
        sign = "+" if signal['score_adjustment'] > 0 else ""
        lines.append(f"   評分調整: {sign}{signal['score_adjustment']}分")

    return "\n".join(lines)


# ======================================================
# 主程序測試
# ======================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 50)
    print("Pattern Intraday Engine Test")
    print("=" * 50)

    # 測試1: 開盤前三根1分鐘K連續收紅上漲 + 第三根爆量創當日新高
    df_3bar = pd.DataFrame({
        'open':   [100.0, 100.2, 100.5],
        'high':   [100.3, 100.6, 100.9],
        'low':    [99.9,  100.1, 100.4],
        'close':  [100.2, 100.5, 100.9],
        'volume': [1000,  1000,  3000],
    }, index=pd.date_range("2026-06-15 09:00", periods=3, freq="1min"))

    print(f"\n[1] OPENING_3BAR_BREAKOUT: {detect_opening_3bar_breakout(df_3bar)}")

    # 測試2: ORB (10根K，前5根高點=開盤區間，最後一根收盤突破)
    df_orb = pd.DataFrame({
        'open':   [100.0, 100.2, 100.5, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 101.0],
        'high':   [100.3, 100.6, 100.9, 100.5, 100.6, 100.7, 100.8, 100.9, 101.0, 101.5],
        'low':    [99.9,  100.1, 100.4, 100.1, 100.2, 100.3, 100.4, 100.5, 100.7, 100.9],
        'close':  [100.2, 100.5, 100.8, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9, 101.4],
        'volume': [1000] * 10,
    }, index=pd.date_range("2026-06-15 09:00", periods=10, freq="1min"))

    print(f"[2] ORB: {detect_orb(df_orb)}")

    # 測試3: VWAP_RECLAIM (前一根收盤跌破VWAP，最新一根收盤站回)
    df_vwap = pd.DataFrame({
        'open':   [100.0, 100.0, 100.2],
        'high':   [100.5, 100.3, 100.6],
        'low':    [99.5,  99.0,  99.8],
        'close':  [100.0, 99.0,  100.5],
        'volume': [1000,  1000,  1000],
    }, index=pd.date_range("2026-06-15 09:00", periods=3, freq="1min"))

    print(f"[3] VWAP_RECLAIM: {detect_vwap_reclaim(df_vwap)}")

    # 測試4: HIGH_OF_DAY_BREAK (最新收盤突破前面所有K的最高價)
    df_hod = pd.DataFrame({
        'open':   [100.0, 100.3, 100.5],
        'high':   [100.5, 100.8, 100.9],
        'low':    [99.8,  100.1, 100.4],
        'close':  [100.3, 100.5, 100.9],
        'volume': [1000,  1000,  1000],
    }, index=pd.date_range("2026-06-15 09:00", periods=3, freq="1min"))

    print(f"[4] HIGH_OF_DAY_BREAK: {detect_high_of_day_break(df_hod)}")

    # 測試5: VOLUME_SURGE (RVOL = 今日累積量 / 20期均量，20根K)
    df_rvol = pd.DataFrame({
        'open':   [100.0] * 20,
        'high':   [100.5] * 20,
        'low':    [99.5] * 20,
        'close':  [100.0] * 20,
        'volume': [500] * 19 + [3000],
    }, index=pd.date_range("2026-06-15 09:00", periods=20, freq="1min"))

    rvol = relative_volume(df_rvol)
    print(f"[5] RVOL: {rvol:.2f}  -> VOLUME_SURGE: {rvol > 3}")

    # 測試6: 主引擎整合測試 (ORB + HIGH_OF_DAY_BREAK + GAMMA_SQUEEZE_SETUP)
    print("\n" + "=" * 50)
    print("Master Engine Test (df_orb, gex_score=75)")
    print("=" * 50)

    patterns = pattern_intraday_engine(df_orb, gex_score=75)
    signal = get_intraday_pattern_signal(df_orb, gex_score=75)

    print(f"\nDetected patterns: {patterns}")
    print(f"\nSignal details:")
    print(format_intraday_pattern_output(signal))

    print("\n" + "=" * 50)
    print("Pattern types:")
    print("  Bullish: OPENING_3BAR_BREAKOUT / ORB / VWAP_RECLAIM / HIGH_OF_DAY_BREAK / GAMMA_SQUEEZE_SETUP")
    print("  Neutral: VOLUME_SURGE")
