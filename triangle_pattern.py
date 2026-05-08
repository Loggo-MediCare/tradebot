"""
三角收斂型態檢測模組
====================
使用專家系統規則檢測股票三角收斂模式

功能:
1. 檢測局部高低點
2. 擬合趨勢線並檢查收斂
3. 預測突破方向
4. 視覺化 (可選)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ===============================
# 抓局部高低點
# ===============================

def find_peaks_valleys(prices, window=5):
    """
    檢測高低點: 局部最大/最小值

    Args:
        prices: 價格數列 (numpy array 或 list)
        window: 窗口大小

    Returns:
        highs: [(index, price), ...] 高點列表
        lows: [(index, price), ...] 低點列表
    """
    highs = []
    lows = []

    for i in range(window, len(prices) - window):
        local_range = prices[i-window:i+window+1]
        if prices[i] == max(local_range):
            highs.append((i, prices[i]))
        if prices[i] == min(local_range):
            lows.append((i, prices[i]))

    return highs, lows


def find_swings(series, window=3):
    """舊版函數 - 保持向下兼容"""
    highs, lows = [], []

    for i in range(window, len(series)-window):
        chunk = series[i-window:i+window+1]
        if series[i] == chunk.max():
            highs.append((i, series[i]))
        if series[i] == chunk.min():
            lows.append((i, series[i]))

    return highs, lows


# ===============================
# 三角收斂檢測 (專家系統)
# ===============================

def detect_triangle_convergence(prices, min_points=3):
    """
    擬合趨勢線並檢查收斂 (專家系統規則)

    Args:
        prices: 價格數列
        min_points: 最少需要的點數

    Returns:
        is_triangle: 是否為三角收斂
        upper_line: (slope, intercept) 上壓線參數
        lower_line: (slope, intercept) 下撐線參數
    """
    highs, lows = find_peaks_valleys(prices)

    if len(highs) < min_points or len(lows) < min_points:
        return False, None, None

    # 擬合上壓線 (高點線性回歸，預期負斜率)
    high_x = [x[0] for x in highs[-min_points:]]
    high_y = [y[1] for y in highs[-min_points:]]
    slope_h, intercept_h, _, _, _ = linregress(high_x, high_y)

    # 擬合下撐線 (低點線性回歸，預期正斜率)
    low_x = [x[0] for x in lows[-min_points:]]
    low_y = [y[1] for y in lows[-min_points:]]
    slope_l, intercept_l, _, _, _ = linregress(low_x, low_y)

    # 檢查收斂: 上線負斜、下線正斜，且預測交點在未來
    if slope_h < 0 and slope_l > 0:
        # 計算交點 x = (intercept_l - intercept_h) / (slope_h - slope_l)
        intersect_x = (intercept_l - intercept_h) / (slope_h - slope_l)
        if intersect_x > len(prices):  # 交點在未來
            return True, (slope_h, intercept_h), (slope_l, intercept_l)

    return False, None, None


def detect_triangle(df, lookback=60):
    """
    三角收斂判斷 (整合版)

    Args:
        df: DataFrame with 'close' column
        lookback: 回溯期間

    Returns:
        bool: 是否檢測到三角收斂
    """
    if len(df) < lookback:
        return False

    price = df['close'].values[-lookback:]

    # 若目前價格已是60天新高，不可能是三角收斂
    if price[-1] >= np.max(price[:-1]):
        return False

    # 方法1: 使用專家系統規則
    is_converging, upper, lower = detect_triangle_convergence(price)
    if is_converging:
        return True

    # 方法2: 舊版波動收斂確認 (備用)
    highs, lows = find_swings(price)

    if len(highs) < 3 or len(lows) < 3:
        return False

    # 計算斜率
    high_x = np.array([p[0] for p in highs]).reshape(-1, 1)
    high_y = np.array([p[1] for p in highs])
    low_x = np.array([p[0] for p in lows]).reshape(-1, 1)
    low_y = np.array([p[1] for p in lows])

    if len(high_x) >= 2 and len(low_x) >= 2:
        high_slope, _, _, _, _ = linregress(high_x.flatten(), high_y)
        low_slope, _, _, _, _ = linregress(low_x.flatten(), low_y)

        # 高點下降、低點上升 = 收斂
        if high_slope < 0 and low_slope > 0:
            # 波動收斂確認
            range_start = np.max(price[:10]) - np.min(price[:10])
            range_end = np.max(price[-10:]) - np.min(price[-10:])

            if range_end < range_start * 0.6:
                return True

    return False


# ===============================
# 突破確認
# ===============================

def triangle_breakout(df, lookback=60):
    """
    突破方向判斷

    Args:
        df: DataFrame with OHLC data
        lookback: 回溯期間

    Returns:
        str: "BREAK_UP", "BREAK_DOWN", "CONSOLIDATING", or None
    """
    if not detect_triangle(df, lookback):
        return None

    recent_high = df['high'].iloc[-lookback:-1].max()
    recent_low = df['low'].iloc[-lookback:-1].min()
    close = df['close'].iloc[-1]

    if close > recent_high:
        return "BREAK_UP"

    if close < recent_low:
        return "BREAK_DOWN"

    return "CONSOLIDATING"


def get_triangle_signal(df, lookback=60):
    """
    獲取三角收斂信號 (用於整合到交易系統)

    Returns:
        dict: {
            'detected': bool,
            'status': str,
            'signal_text': str,
            'score_adjustment': int
        }
    """
    result = {
        'detected': False,
        'status': None,
        'signal_text': '',
        'score_adjustment': 0
    }

    if not detect_triangle(df, lookback):
        return result

    result['detected'] = True
    status = triangle_breakout(df, lookback)
    result['status'] = status

    if status == "BREAK_UP":
        result['signal_text'] = "三角收斂向上突破"
        result['score_adjustment'] = 15
    elif status == "BREAK_DOWN":
        result['signal_text'] = "三角收斂向下突破"
        result['score_adjustment'] = -15
    else:
        result['signal_text'] = "三角收斂中，等待方向"
        result['score_adjustment'] = 0

    return result


def format_triangle_output(signal):
    """格式化輸出"""
    if not signal['detected']:
        return "   未檢測到三角收斂型態"

    status_emoji = {
        "BREAK_UP": "🔥",
        "BREAK_DOWN": "⚠️",
        "CONSOLIDATING": "⏳"
    }

    emoji = status_emoji.get(signal['status'], "📊")
    return f"   {emoji} {signal['signal_text']}"


# ===============================
# 主程序測試
# ===============================

if __name__ == "__main__":
    print("三角收斂型態檢測模組")
    print("=" * 50)

    # 模擬測試數據
    np.random.seed(42)
    n = 100

    # 創建收斂型態數據
    base = 100
    converging_range = np.linspace(10, 2, n)  # 範圍逐漸收窄
    prices = base + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })

    # 測試檢測
    is_triangle = detect_triangle(df)
    print(f"\n三角收斂檢測: {'是' if is_triangle else '否'}")

    if is_triangle:
        status = triangle_breakout(df)
        print(f"突破狀態: {status}")

        signal = get_triangle_signal(df)
        print(format_triangle_output(signal))
