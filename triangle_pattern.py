"""
三角收斂型態檢測模組 (修正版)
====================
修正了突破判斷邏輯，現在會依據趨勢線(Trendline)而非區間高低點來判斷突破。

功能:
1. 檢測局部高低點
2. 擬合趨勢線並檢查收斂
3. 預測突破方向 (基於趨勢線幾何突破)
4. 視覺化 (可選)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ===============================
# 抓局部高低點
# ===============================

def _deduplicate_turning_points(points, keep="high", min_separation=2):
    """移除太接近且重複的轉折點，避免平台價位造成重複計數。"""
    if not points:
        return []

    cleaned = [points[0]]
    for idx, price in points[1:]:
        last_idx, last_price = cleaned[-1]
        if idx - last_idx <= min_separation:
            replace = (keep == "high" and price >= last_price) or (keep == "low" and price <= last_price)
            if replace:
                cleaned[-1] = (idx, price)
        else:
            cleaned.append((idx, price))
    return cleaned


def find_peaks_valleys(prices, window=5, min_separation=2):
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

    # 安全檢查：數據長度必須足夠
    if len(prices) < window * 2:
        return [], []

    for i in range(window, len(prices) - window):
        local_range = prices[i-window:i+window+1]
        if prices[i] == np.max(local_range):
            highs.append((i, prices[i]))
        if prices[i] == np.min(local_range):
            lows.append((i, prices[i]))

    highs = _deduplicate_turning_points(highs, keep="high", min_separation=min_separation)
    lows = _deduplicate_turning_points(lows, keep="low", min_separation=min_separation)

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

def _evaluate_triangle(highs, lows, prices, min_points=3):
    """評估對稱/上升/下降三角收斂，返回趨勢線參數"""
    if len(highs) < min_points or len(lows) < min_points:
        return False, None, None

    # 取最近的 N 個點來擬合，避免太久以前的點干擾
    n_points = min(5, len(highs), len(lows))
    n_points = max(n_points, min_points)

    high_used = highs[-n_points:]
    low_used = lows[-n_points:]

    high_x = [x[0] for x in high_used]
    high_y = [x[1] for x in high_used]
    low_x = [x[0] for x in low_used]
    low_y = [x[1] for x in low_used]

    slope_h, intercept_h, _, _, _ = linregress(high_x, high_y)
    slope_l, intercept_l, _, _, _ = linregress(low_x, low_y)

    idx_start = min(high_x[0], low_x[0])
    idx_end = len(prices) - 1

    # 計算通道寬度
    start_gap = (slope_h * idx_start + intercept_h) - (slope_l * idx_start + intercept_l)
    end_gap = (slope_h * idx_end + intercept_h) - (slope_l * idx_end + intercept_l)

    # 1. 基本幾何檢查: 高點線必須在低點線之上
    if start_gap <= 0 or end_gap <= 0:
        return False, None, None

    # 2. 斜率檢查 (稍微收緊容差以提高精準度)
    slope_eps = max((np.max(prices) - np.min(prices)) / max(len(prices), 1) * 0.015, 1e-4)

    is_symmetric = slope_h < -slope_eps and slope_l > slope_eps
    is_ascending = abs(slope_h) <= slope_eps and slope_l > slope_eps  # 頂平底升
    is_descending = slope_h < -slope_eps and abs(slope_l) <= slope_eps  # 頂降底平
    is_wedge = (slope_h < 0 and slope_l < 0 and slope_h < slope_l) or \
               (slope_h > 0 and slope_l > 0 and slope_h < slope_l)  # 楔形

    if not (is_symmetric or is_ascending or is_descending or is_wedge):
        # 寬鬆條件：只要高點線向下、低點線向上，就算收斂
        if not (slope_h < 0 and slope_l > 0):
            return False, None, None

    # 3. 收斂核心條件: 末端寬度必須明顯小於開頭 (放寬到 0.90)
    if end_gap >= start_gap * 0.90:
        return False, None, None

    # 4. 斜率不平行情況要求交點在未來
    if is_symmetric or is_wedge:
        denom = slope_h - slope_l
        if abs(denom) <= 1e-9:
            return False, None, None
        intersect_x = (intercept_l - intercept_h) / denom
        if not (idx_end < intersect_x < idx_end + len(prices) * 2):
            return False, None, None

    return True, (slope_h, intercept_h), (slope_l, intercept_l)


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
    # 使用多組窗口，提升不同波動節奏下的檢測率
    for window in (3, 5, 7):
        highs, lows = find_peaks_valleys(prices, window=window, min_separation=max(1, window // 2))
        is_triangle, upper_line, lower_line = _evaluate_triangle(highs, lows, prices, min_points=min_points)
        if is_triangle:
            return True, upper_line, lower_line

    return False, None, None


def detect_triangle_details(df, lookback=60):
    """
    偵測三角收斂並返回詳細參數 (供 breakout 使用)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        lookback: 回溯期間

    Returns:
        is_triangle: bool - 是否檢測到三角收斂
        upper_line: (slope, intercept) or None - 上壓線參數
        lower_line: (slope, intercept) or None - 下撐線參數
    """
    # 多重lookback掃描，提升不同週期型態的檢測率
    lookback_periods = [60, 90, 120, 180]

    for lb in lookback_periods:
        if len(df) < lb:
            continue

        # 使用 high 和 low 來檢測轉折點，比 close 更準確
        price_high = df['high'].values[-lb:]
        price_low = df['low'].values[-lb:]
        price_close = df['close'].values[-lb:]

        # 嘗試不同的窗口大小尋找最佳擬合
        for window in (3, 5, 7):
            highs, _ = find_peaks_valleys(price_high, window=window, min_separation=max(1, window // 2))
            _, lows = find_peaks_valleys(price_low, window=window, min_separation=max(1, window // 2))

            # 使用 close 價格來評估趨勢線（避免極端值影響）
            is_triangle, upper, lower = _evaluate_triangle(highs, lows, price_close, min_points=3)
            if is_triangle:
                return True, upper, lower

            # Fallback: 嘗試只用 2 個點的寬鬆檢測
            is_triangle, upper, lower = _evaluate_triangle(highs, lows, price_close, min_points=2)
            if is_triangle:
                return True, upper, lower

    return False, None, None


def detect_triangle(df, lookback=60):
    """
    三角收斂判斷 (兼容舊接口)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        lookback: 回溯期間

    Returns:
        bool: 是否檢測到三角收斂
    """
    found, _, _ = detect_triangle_details(df, lookback)
    return found


# ===============================
# 突破確認 (修復版 - 基於趨勢線幾何突破)
# ===============================

def triangle_breakout(df, lookback=60):
    """
    判斷是否突破三角收斂 (使用趨勢線而非區間高低點)

    Args:
        df: DataFrame with OHLC data
        lookback: 回溯期間

    Returns:
        str: "BREAK_UP", "BREAK_DOWN", "CONSOLIDATING", or None
    """
    # 1. 獲取趨勢線參數
    found, upper_line, lower_line = detect_triangle_details(df, lookback)

    if not found:
        return None

    slope_h, intercept_h = upper_line
    slope_l, intercept_l = lower_line

    # 2. 找出實際使用的 lookback 並計算當前 K 棒位置的理論壓力/支撐價格
    # 由於我們從多個 lookback 中選擇，需要找出實際的 lookback
    for lb in [60, 90, 120, 180]:
        if len(df) < lb:
            continue
        current_idx = lb - 1

        resistance_price = slope_h * current_idx + intercept_h
        support_price = slope_l * current_idx + intercept_l

        # 驗證趨勢線是否合理（阻力線應該在支撐線上方）
        if resistance_price > support_price:
            break
    else:
        # Fallback: 使用傳入的 lookback
        current_idx = min(lookback, len(df)) - 1
        resistance_price = slope_h * current_idx + intercept_h
        support_price = slope_l * current_idx + intercept_l

    current_close = df['close'].iloc[-1]

    # 3. 判斷突破 (加入 0.5% 的緩衝區，避免假突破)
    threshold = current_close * 0.005

    if current_close > resistance_price + threshold:
        return "BREAK_UP"
    elif current_close < support_price - threshold:
        return "BREAK_DOWN"
    else:
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

    # 使用新的詳細檢測函數獲取趨勢線參數
    found, upper, lower = detect_triangle_details(df, lookback)

    if not found:
        return result

    result['detected'] = True

    # 獲取突破狀態
    status = triangle_breakout(df, lookback)
    result['status'] = status

    if status == "BREAK_UP":
        result['signal_text'] = "三角收斂向上突破 (突破壓力線)"
        result['score_adjustment'] = 20  # 提高權重，因為這是基於幾何突破
    elif status == "BREAK_DOWN":
        result['signal_text'] = "三角收斂向下突破 (跌破支撐線)"
        result['score_adjustment'] = -20
    else:
        # 計算目前價格在通道中的位置 (0~100%)
        # 找出實際使用的 lookback
        for lb in [60, 90, 120, 180]:
            if len(df) >= lb:
                current_idx = lb - 1
                break
        else:
            current_idx = min(lookback, len(df)) - 1

        res = upper[0] * current_idx + upper[1]
        sup = lower[0] * current_idx + lower[1]
        close = df['close'].iloc[-1]

        # 防止除以零
        if res - sup > 0:
            pos = (close - sup) / (res - sup) * 100
            result['signal_text'] = f"三角收斂中 (位置: {pos:.0f}%)"
        else:
            result['signal_text'] = "三角收斂末端 (即將變盤)"

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
