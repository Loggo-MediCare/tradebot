"""
圖表型態識別引擎
================
識別多種技術分析型態

支援型態:
1. TRIANGLE - 三角收斂
2. FLAG - 旗形整理
3. W_BOTTOM - W底
4. BOX - 盤整盒型

用法:
    patterns = pattern_engine(df)
    signal = get_pattern_signal(df)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


# =========================
# Swing Points Detection
# =========================

def find_swings(series, window=3):
    """找出擺動高低點"""
    highs, lows = [], []

    for i in range(window, len(series) - window):
        block = series[i-window:i+window+1]
        if series[i] == block.max():
            highs.append((i, series[i]))
        if series[i] == block.min():
            lows.append((i, series[i]))

    return highs, lows


def calc_slope(points):
    """計算趨勢斜率"""
    if len(points) < 3:
        return None
    X = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    slope, _, _, _, _ = linregress(X, y)
    return slope


# ======================================================
# 1. Triangle Convergence (三角收斂)
# ======================================================

def detect_triangle(close, lookback=60):
    """
    三角收斂型態檢測
    - 高點下降 + 低點上升 = 收斂
    - 波動區間縮小
    """
    if len(close) < lookback:
        return False

    price = close[-lookback:]
    highs, lows = find_swings(price)

    hs = calc_slope(highs)
    ls = calc_slope(lows)

    if hs is None or ls is None:
        return False

    # 波動收斂確認 (後段區間 < 前段 60%)
    range_start = price[:10].max() - price[:10].min()
    range_end = price[-10:].max() - price[-10:].min()
    contraction = range_end < range_start * 0.6

    return hs < 0 and ls > 0 and contraction


# ======================================================
# 2. Flag Pattern (旗形整理)
# ======================================================

def detect_flag(close, lookback=40):
    """
    旗形型態檢測
    - 先有衝高 (impulse)
    - 然後小幅整理 (flag)
    - 整理幅度 < 衝高幅度 40%
    """
    if len(close) < lookback:
        return False

    price = close[-lookback:]

    # 前段為衝高，後段為整理
    impulse = price[:10]
    flag = price[10:]

    impulse_strength = impulse[-1] - impulse[0]
    flag_range = flag.max() - flag.min()

    # 衝高為正 + 整理幅度小
    return impulse_strength > 0 and flag_range < abs(impulse_strength) * 0.4


# ======================================================
# 3. W Bottom (W底)
# ======================================================

def detect_w_bottom(close, lookback=80):
    """
    W底型態檢測
    - 兩個相近低點
    - 頸線明顯高於低點
    """
    if len(close) < lookback:
        return False

    price = close[-lookback:]

    # 找最低的幾個點
    lows_idx = np.argsort(price)[:5]
    lows_idx = np.sort(lows_idx)

    if len(lows_idx) < 2:
        return False

    # 取第一和第二低點
    first_low_idx = lows_idx[0]
    second_low_idx = lows_idx[-1] if lows_idx[-1] != first_low_idx else lows_idx[1]

    first_low = price[first_low_idx]
    second_low = price[second_low_idx]

    # 兩低點之間的高點 (頸線)
    if first_low_idx < second_low_idx:
        middle_section = price[first_low_idx:second_low_idx+1]
    else:
        middle_section = price[second_low_idx:first_low_idx+1]

    if len(middle_section) < 3:
        return False

    neck = middle_section.max()

    # W底條件: 兩低點相近 (< 3%) + 頸線高於低點 8%+
    low_diff = abs(first_low - second_low) / first_low if first_low > 0 else 1
    neck_above = neck > max(first_low, second_low) * 1.05

    return low_diff < 0.05 and neck_above


# ======================================================
# 4. Box Consolidation (盤整盒型)
# ======================================================

def detect_box(close, lookback=50):
    """
    盤整盒型檢測
    - 價格在窄幅區間震盪
    - 區間幅度 < 8%
    """
    if len(close) < lookback:
        return False

    price = close[-lookback:]

    high = price.max()
    low = price.min()

    if low <= 0:
        return False

    range_pct = (high - low) / low

    return range_pct < 0.08


# ======================================================
# 5. Head and Shoulders (頭肩頂) - 額外新增
# ======================================================

def detect_head_shoulders(close, lookback=60):
    """
    頭肩頂型態檢測 (看跌)
    - 左肩 - 頭部 - 右肩
    - 頭部為最高點
    """
    if len(close) < lookback:
        return False

    price = close[-lookback:]
    highs, _ = find_swings(price, window=5)

    if len(highs) < 3:
        return False

    # 取最近三個高點
    recent_highs = highs[-3:]
    h1, h2, h3 = [h[1] for h in recent_highs]

    # 頭肩頂: 中間高點最高，兩側相近
    is_head = h2 > h1 and h2 > h3
    shoulders_similar = abs(h1 - h3) / h1 < 0.05 if h1 > 0 else False

    return is_head and shoulders_similar


# ======================================================
# Master Pattern Engine
# ======================================================

def pattern_engine(df):
    """
    主型態識別引擎

    Returns:
        list: 檢測到的型態列表
    """
    close = df['close'].values
    patterns = []

    if detect_triangle(close):
        patterns.append("TRIANGLE")

    if detect_flag(close):
        patterns.append("FLAG")

    if detect_w_bottom(close):
        patterns.append("W_BOTTOM")

    if detect_box(close):
        patterns.append("BOX")

    if detect_head_shoulders(close):
        patterns.append("HEAD_SHOULDERS")

    return patterns


def get_pattern_signal(df):
    """
    獲取型態信號 (用於整合到交易系統)

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

    patterns = pattern_engine(df)
    result['patterns'] = patterns

    if not patterns:
        return result

    # 分類型態
    bullish_patterns = ['W_BOTTOM', 'FLAG']
    bearish_patterns = ['HEAD_SHOULDERS']
    neutral_patterns = ['TRIANGLE', 'BOX']

    for p in patterns:
        if p in bullish_patterns:
            result['bullish'].append(p)
        elif p in bearish_patterns:
            result['bearish'].append(p)
        else:
            result['neutral'].append(p)

    # 計算評分調整
    score = 0
    texts = []

    if 'W_BOTTOM' in patterns:
        score += 15
        texts.append("W底成形")

    if 'FLAG' in patterns:
        score += 10
        texts.append("旗形整理")

    if 'TRIANGLE' in patterns:
        score += 5
        texts.append("三角收斂")

    if 'BOX' in patterns:
        score += 0
        texts.append("盤整盒型")

    if 'HEAD_SHOULDERS' in patterns:
        score -= 15
        texts.append("頭肩頂警示")

    result['score_adjustment'] = score
    result['signal_text'] = " + ".join(texts) if texts else ""

    return result


def format_pattern_engine_output(signal):
    """格式化輸出"""
    if not signal['patterns']:
        return "   未檢測到明顯型態"

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
    print("Pattern Engine Test")
    print("=" * 50)

    # 模擬測試數據
    np.random.seed(42)
    n = 100

    # 創建測試數據
    prices = 100 + np.cumsum(np.random.randn(n) * 2)

    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n)
    })

    # 測試
    patterns = pattern_engine(df)
    signal = get_pattern_signal(df)

    print(f"\nDetected patterns: {patterns}")
    print(f"\nSignal details:")
    print(format_pattern_engine_output(signal))

    print("\n" + "=" * 50)
    print("Pattern types:")
    print("  Bullish: W_BOTTOM, FLAG")
    print("  Bearish: HEAD_SHOULDERS")
    print("  Neutral: TRIANGLE, BOX")
