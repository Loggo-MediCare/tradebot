"""
結構型態分析模組
================
偵測大型技術型態：W底 (Double Bottom) 與 鍋底 (Rounding Bottom)
"""

import numpy as np
from scipy.signal import argrelextrema


def detect_structure_patterns(df, window=50):
    """
    偵測大型技術型態：W底 (Double Bottom) 與 鍋底 (Rounding Bottom)

    Args:
        df: DataFrame with 'close' column
        window: Number of days to analyze (default 50)

    Returns:
        dict with pattern detection results
    """
    # 取得最近 window 天的收盤價
    recent_data = df.iloc[-window:].copy().reset_index(drop=True)
    prices = recent_data['close'].values

    # 初始化結果
    patterns = {
        'w_bottom': False,
        'rounding_bottom': False,
        'description': [],
        'score_bonus': 0
    }

    if len(prices) < window:
        return patterns

    # ==========================================
    # 1. 偵測 W 底 (Double Bottom)
    # 邏輯：兩個明顯的低點，且第二個低點不低於第一個太多，頸線被突破
    # ==========================================
    # 尋找局部低點 (Local Minima)
    local_min_indices = argrelextrema(prices, np.less, order=5)[0]

    if len(local_min_indices) >= 2:
        # 取最後兩個低點
        low1_idx = local_min_indices[-2]
        low2_idx = local_min_indices[-1]

        low1 = prices[low1_idx]
        low2 = prices[low2_idx]

        # 兩個低點之間必須有間隔 (至少 5 天)
        if (low2_idx - low1_idx) > 5:
            # 尋找兩個低點之間的高點 (頸線)
            neckline_idx = np.argmax(prices[low1_idx:low2_idx]) + low1_idx
            neckline_price = prices[neckline_idx]
            current_price = prices[-1]

            # W底條件：
            # A. 兩個低點價格接近 (誤差 3% 內)
            bottoms_match = abs(low1 - low2) / low1 < 0.03
            # B. 現在價格是否剛突破頸線 (或在頸線附近)
            near_neckline = current_price > neckline_price * 0.98

            if bottoms_match and near_neckline:
                patterns['w_bottom'] = True
                patterns['description'].append(f"W底成形: 低點 ${low1:.1f}/${low2:.1f}, 頸線 ${neckline_price:.1f}")
                patterns['score_bonus'] = 20

    # ==========================================
    # 2. 偵測 鍋底 (Rounding Bottom / Saucer)
    # 邏輯：長期下跌 -> 平緩 -> 微幅上升 (MA排列改變)
    # ==========================================
    # 簡單判斷法：使用 MA 斜率變化
    # 前 1/3 段是跌勢，中間 1/3 盤整，後 1/3 緩漲

    if not patterns['w_bottom']:  # 如果沒有W底，才檢查鍋底
        n = len(prices)
        part1 = prices[:n//3]
        part2 = prices[n//3 : 2*n//3]
        part3 = prices[2*n//3:]

        # 計算各段線性回歸斜率
        slope1 = np.polyfit(range(len(part1)), part1, 1)[0]
        slope2 = np.polyfit(range(len(part2)), part2, 1)[0]
        slope3 = np.polyfit(range(len(part3)), part3, 1)[0]

        # 鍋底條件：先跌(負斜率) -> 平緩(接近0) -> 後漲(正斜率)
        if slope1 < -0.1 and abs(slope2) < 0.2 and slope3 > 0.1:
            patterns['rounding_bottom'] = True
            patterns['description'].append("鍋底(圓弧底)成形: 趨勢由跌轉平再轉漲")
            patterns['score_bonus'] = 15

    return patterns


def format_structure_pattern_output(patterns):
    """
    格式化結構型態分析輸出

    Args:
        patterns: dict from detect_structure_patterns()

    Returns:
        Formatted string for display
    """
    output = []

    if patterns['w_bottom']:
        output.append(f"✨ 發現型態: {patterns['description'][0]}")
        output.append("   👉 訊號: 強力看漲 (W底支撐)")
        output.append(f"   📈 評分加成: +{patterns['score_bonus']}分")
    elif patterns['rounding_bottom']:
        output.append(f"✨ 發現型態: {patterns['description'][0]}")
        output.append("   👉 訊號: 溫和看漲 (底部翻揚)")
        output.append(f"   📈 評分加成: +{patterns['score_bonus']}分")
    else:
        output.append("未發現明顯 W底 或 鍋底 型態")

    return "\n".join(output)


def get_structure_score_adjustment(patterns):
    """
    獲取結構型態評分調整

    Args:
        patterns: dict from detect_structure_patterns()

    Returns:
        Score adjustment value (positive for bullish patterns)
    """
    return patterns.get('score_bonus', 0)
