"""
真假突破檢測模組 (進階版)
==========================
使用專家系統規則分辨真突破與假突破

核心邏輯:
- 真突破: 價格突破阻力線 + 成交量放大 (量比 >= 1.5)
- 假突破: 價格突破阻力線 + 成交量萎縮 (量比 < 1.5)

進階型態檢測:
- W底突破 (W-bottom breakout)
- 杯柄突破 (Cup & Handle breakout)
- 旗形/三角突破 (Flag/Triangle breakout)
- 整理突破 (Consolidation breakout)
- 量價突破 (Volume-price breakout)
- 趨勢突破 (Trend breakout)

專家系統建議:
- 真突破 → 買入並設定止損
- 假突破 → 賣出或觀望
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Dict, List, Optional


def find_resistance_support(df, window=20):
    """
    檢測阻力/支撐位 (使用近期高點擬合阻力線)

    Args:
        df: DataFrame with 'high', 'low' columns
        window: 窗口大小

    Returns:
        resistance_line: (slope, intercept) 阻力線參數
        support_line: (slope, intercept) 支撐線參數
    """
    prices_high = df['high'].values
    prices_low = df['low'].values

    highs = []
    lows = []

    for i in range(window, len(df) - window):
        # 局部高點
        if prices_high[i] == max(prices_high[i-window:i+window+1]):
            highs.append((i, prices_high[i]))
        # 局部低點
        if prices_low[i] == min(prices_low[i-window:i+window+1]):
            lows.append((i, prices_low[i]))

    # 擬合阻力線
    if len(highs) >= 2:
        high_x = [x[0] for x in highs]
        high_y = [y[1] for y in highs]
        slope_r, intercept_r, _, _, _ = linregress(high_x, high_y)
    else:
        slope_r, intercept_r = 0, np.mean(prices_high) if len(prices_high) > 0 else 0

    # 擬合支撐線
    if len(lows) >= 2:
        low_x = [x[0] for x in lows]
        low_y = [y[1] for y in lows]
        slope_s, intercept_s, _, _, _ = linregress(low_x, low_y)
    else:
        slope_s, intercept_s = 0, np.mean(prices_low) if len(prices_low) > 0 else 0

    return (slope_r, intercept_r), (slope_s, intercept_s)


def detect_breakout(df, lookback=60, min_volume_ratio=1.5):
    """
    檢測突破並分辨真假

    Args:
        df: DataFrame with OHLCV data
        lookback: 回溯期間
        min_volume_ratio: 判定真突破的最低量比

    Returns:
        dict: {
            'detected': bool,
            'type': str ('TRUE_BREAKOUT', 'FALSE_BREAKOUT', 'BREAKDOWN', None),
            'volume_ratio': float,
            'resistance': float,
            'support': float
        }
    """
    result = {
        'detected': False,
        'type': None,
        'volume_ratio': 0,
        'resistance': 0,
        'support': 0,
        'direction': None
    }

    if len(df) < lookback + 5:
        return result

    # 取最近數據
    recent_df = df.iloc[-lookback:].copy()

    # 計算阻力/支撐線
    resistance_line, support_line = find_resistance_support(recent_df)

    # 計算當前阻力/支撐值
    x_current = len(recent_df) - 1
    resistance_value = resistance_line[0] * x_current + resistance_line[1]
    support_value = support_line[0] * x_current + support_line[1]

    result['resistance'] = resistance_value
    result['support'] = support_value

    # 當前價格和成交量
    current_close = df['close'].iloc[-1]
    current_volume = df['volume'].iloc[-1]
    prev_close = df['close'].iloc[-2]

    # 計算平均成交量 (排除今日)
    avg_volume = df['volume'].iloc[-lookback:-1].mean()
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    result['volume_ratio'] = volume_ratio

    # 前一日阻力值
    x_prev = len(recent_df) - 2
    prev_resistance = resistance_line[0] * x_prev + resistance_line[1]
    prev_support = support_line[0] * x_prev + support_line[1]

    # 檢測向上突破 (今日 Close > 阻力線，前日 Close <= 阻力線)
    if current_close > resistance_value and prev_close <= prev_resistance:
        result['detected'] = True
        result['direction'] = 'UP'
        if volume_ratio >= min_volume_ratio:
            result['type'] = 'TRUE_BREAKOUT'
        else:
            result['type'] = 'FALSE_BREAKOUT'

    # 檢測向下跌破 (今日 Close < 支撐線，前日 Close >= 支撐線)
    elif current_close < support_value and prev_close >= prev_support:
        result['detected'] = True
        result['direction'] = 'DOWN'
        if volume_ratio >= min_volume_ratio:
            result['type'] = 'TRUE_BREAKDOWN'
        else:
            result['type'] = 'FALSE_BREAKDOWN'

    return result


def get_breakout_signal(df, lookback=60):
    """
    獲取突破信號 (用於整合到交易系統)

    Returns:
        dict: {
            'detected': bool,
            'type': str,
            'signal_text': str,
            'score_adjustment': int,
            'volume_ratio': float
        }
    """
    result = {
        'detected': False,
        'type': None,
        'signal_text': '',
        'score_adjustment': 0,
        'volume_ratio': 0
    }

    breakout = detect_breakout(df, lookback)

    if not breakout['detected']:
        return result

    result['detected'] = True
    result['type'] = breakout['type']
    result['volume_ratio'] = breakout['volume_ratio']

    if breakout['type'] == 'TRUE_BREAKOUT':
        result['signal_text'] = f"放量真突破 (量比: {breakout['volume_ratio']:.1f}x)"
        result['score_adjustment'] = 15
    elif breakout['type'] == 'FALSE_BREAKOUT':
        result['signal_text'] = f"縮量假突破 (量比: {breakout['volume_ratio']:.1f}x)"
        result['score_adjustment'] = -10
    elif breakout['type'] == 'TRUE_BREAKDOWN':
        result['signal_text'] = f"放量跌破支撐 (量比: {breakout['volume_ratio']:.1f}x)"
        result['score_adjustment'] = -15
    elif breakout['type'] == 'FALSE_BREAKDOWN':
        result['signal_text'] = f"縮量跌破 (量比: {breakout['volume_ratio']:.1f}x)"
        result['score_adjustment'] = -5

    return result


def format_breakout_output(signal):
    """格式化輸出"""
    if not signal['detected']:
        return "   未檢測到突破信號"

    type_emoji = {
        'TRUE_BREAKOUT': '🚀',
        'FALSE_BREAKOUT': '⚠️',
        'TRUE_BREAKDOWN': '📉',
        'FALSE_BREAKDOWN': '⚠️'
    }

    emoji = type_emoji.get(signal['type'], '📊')
    return f"   {emoji} {signal['signal_text']}"


# ===============================================
# 進階型態檢測函數
# ===============================================

def detect_w_bottom_breakout(df: pd.DataFrame, lookback: int = 60) -> Dict:
    """
    檢測W底型態突破

    W底特徵:
    - 兩個相近的低點 (第二低點略高於第一低點)
    - 中間有反彈形成頸線
    - 突破頸線確認型態

    Args:
        df: DataFrame with OHLCV data
        lookback: 回溯期間

    Returns:
        dict: W底突破信號
    """
    result = {
        'detected': False,
        'type': 'W_BOTTOM',
        'neckline': 0,
        'low1': 0,
        'low2': 0,
        'signal_text': '',
        'score_adjustment': 0
    }

    if len(df) < lookback:
        return result

    recent_df = df.iloc[-lookback:].copy()
    lows = recent_df['low'].values
    closes = recent_df['close'].values
    highs = recent_df['high'].values

    # 尋找兩個顯著低點
    window = lookback // 4  # 用於局部最小值檢測

    local_mins = []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[max(0, i-window):min(len(lows), i+window+1)]):
            local_mins.append((i, lows[i]))

    if len(local_mins) < 2:
        return result

    # 取最近的兩個低點
    sorted_mins = sorted(local_mins, key=lambda x: x[1])[:4]  # 取4個最低點
    sorted_mins = sorted(sorted_mins, key=lambda x: x[0])  # 按時間排序

    if len(sorted_mins) < 2:
        return result

    # 選擇形成W底的兩個低點 (第二低點應在第一低點之後)
    low1_idx, low1_price = sorted_mins[0]
    low2_idx, low2_price = sorted_mins[-1]

    if low2_idx <= low1_idx:
        return result

    # 檢查兩個低點的價格相近 (差距在5%內)
    price_diff = abs(low2_price - low1_price) / low1_price
    if price_diff > 0.05:
        return result

    # 找頸線 (兩個低點之間的最高點)
    middle_highs = highs[low1_idx:low2_idx+1]
    if len(middle_highs) == 0:
        return result

    neckline = max(middle_highs)
    result['neckline'] = neckline
    result['low1'] = low1_price
    result['low2'] = low2_price

    # 檢查當前價格是否突破頸線
    current_close = closes[-1]
    if current_close > neckline:
        # 計算量比
        avg_vol = df['volume'].iloc[-lookback:-1].mean()
        current_vol = df['volume'].iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        result['detected'] = True
        result['volume_ratio'] = vol_ratio

        if vol_ratio >= 1.3:
            result['signal_text'] = f"W底放量突破頸線 ${neckline:.2f} (量比: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 20
        else:
            result['signal_text'] = f"W底突破頸線 ${neckline:.2f} (量比偏低: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 10

    return result


def detect_cup_handle_breakout(df: pd.DataFrame, cup_period: int = 40, handle_period: int = 10) -> Dict:
    """
    檢測杯柄型態突破

    杯柄特徵:
    - 杯形 (U型底部，約30-40個週期)
    - 柄形 (小幅回調，約10-15個週期)
    - 突破杯口高點

    Args:
        df: DataFrame with OHLCV data
        cup_period: 杯形期間
        handle_period: 柄形期間

    Returns:
        dict: 杯柄突破信號
    """
    result = {
        'detected': False,
        'type': 'CUP_HANDLE',
        'cup_high': 0,
        'cup_low': 0,
        'handle_low': 0,
        'signal_text': '',
        'score_adjustment': 0
    }

    total_period = cup_period + handle_period + 5
    if len(df) < total_period:
        return result

    # 杯形區域
    cup_start = -total_period
    cup_end = -handle_period - 5
    cup_data = df.iloc[cup_start:cup_end]

    cup_high = cup_data['high'].max()
    cup_low = cup_data['low'].min()

    # 柄形區域
    handle_start = -handle_period - 5
    handle_data = df.iloc[handle_start:-1]
    handle_low = handle_data['low'].min()

    result['cup_high'] = cup_high
    result['cup_low'] = cup_low
    result['handle_low'] = handle_low

    # 杯柄驗證條件
    # 1. 柄的低點必須高於杯底的95%
    if handle_low < cup_low * 0.95:
        return result

    # 2. 杯深度合理 (10%-50%)
    cup_depth = (cup_high - cup_low) / cup_high
    if cup_depth < 0.10 or cup_depth > 0.50:
        return result

    # 3. 當前價格突破杯口高點
    current_close = df['close'].iloc[-1]
    if current_close > cup_high:
        # 計算量比
        avg_vol = handle_data['volume'].mean()
        current_vol = df['volume'].iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        result['detected'] = True
        result['volume_ratio'] = vol_ratio

        if vol_ratio >= 1.3:
            result['signal_text'] = f"杯柄放量突破 ${cup_high:.2f} (量比: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 25  # 杯柄型態成功率高，加分較多
        else:
            result['signal_text'] = f"杯柄突破 ${cup_high:.2f} (量比偏低: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 12

    return result


def detect_flag_triangle_breakout(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    檢測旗形/三角收斂突破

    特徵:
    - 收斂三角形: 高點下降趨勢 + 低點上升趨勢
    - 旗形: 前期急漲後的整理
    - 突破收斂區間

    Args:
        df: DataFrame with OHLCV data
        lookback: 回溯期間

    Returns:
        dict: 旗形/三角突破信號
    """
    result = {
        'detected': False,
        'type': None,
        'pattern': None,
        'breakout_level': 0,
        'signal_text': '',
        'score_adjustment': 0
    }

    if len(df) < lookback + 10:
        return result

    recent_df = df.iloc[-lookback-1:-1].copy()  # 排除今日
    highs = recent_df['high'].values
    lows = recent_df['low'].values

    # 計算趨勢線斜率
    x = np.arange(len(highs))
    high_slope, high_intercept, _, _, _ = linregress(x, highs)
    low_slope, low_intercept, _, _, _ = linregress(x, lows)

    # 判斷型態
    is_symmetric_triangle = high_slope < 0 and low_slope > 0  # 對稱三角形
    is_ascending_triangle = abs(high_slope) < 0.01 and low_slope > 0  # 上升三角形
    is_descending_triangle = high_slope < 0 and abs(low_slope) < 0.01  # 下降三角形
    is_flag = high_slope < 0 and low_slope < 0 and abs(high_slope - low_slope) < 0.05  # 旗形

    if not any([is_symmetric_triangle, is_ascending_triangle, is_descending_triangle, is_flag]):
        return result

    # 確定型態
    if is_ascending_triangle:
        result['pattern'] = 'ASCENDING_TRIANGLE'
        breakout_level = max(highs)
    elif is_symmetric_triangle:
        result['pattern'] = 'SYMMETRIC_TRIANGLE'
        breakout_level = high_intercept + high_slope * len(highs)
    elif is_descending_triangle:
        result['pattern'] = 'DESCENDING_TRIANGLE'
        breakout_level = max(highs)
    else:
        result['pattern'] = 'FLAG'
        breakout_level = max(highs)

    result['breakout_level'] = breakout_level

    # 檢查當前價格是否突破
    current_close = df['close'].iloc[-1]
    if current_close > breakout_level:
        # 計算量比
        avg_vol = recent_df['volume'].mean()
        current_vol = df['volume'].iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        result['detected'] = True
        result['type'] = 'FLAG_TRIANGLE_BREAKOUT'
        result['volume_ratio'] = vol_ratio

        pattern_names = {
            'ASCENDING_TRIANGLE': '上升三角形',
            'SYMMETRIC_TRIANGLE': '對稱三角形',
            'DESCENDING_TRIANGLE': '下降三角形',
            'FLAG': '旗形整理'
        }
        pattern_name = pattern_names.get(result['pattern'], '收斂型態')

        if vol_ratio >= 1.3:
            result['signal_text'] = f"{pattern_name}放量突破 ${breakout_level:.2f} (量比: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 18
        else:
            result['signal_text'] = f"{pattern_name}突破 ${breakout_level:.2f} (量比偏低: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 8

    return result


def detect_consolidation_breakout(df: pd.DataFrame, consolidation_days: int = 10, max_range_pct: float = 0.08) -> Dict:
    """
    檢測整理突破 (窄幅整理後的突破)

    特徵:
    - 價格在窄幅區間內波動 (< 8%)
    - 突破整理區間上緣

    Args:
        df: DataFrame with OHLCV data
        consolidation_days: 整理天數
        max_range_pct: 最大整理幅度 (預設 8%)

    Returns:
        dict: 整理突破信號
    """
    result = {
        'detected': False,
        'type': 'CONSOLIDATION_BREAKOUT',
        'range_high': 0,
        'range_low': 0,
        'range_pct': 0,
        'signal_text': '',
        'score_adjustment': 0
    }

    if len(df) < consolidation_days + 5:
        return result

    # 整理區間 (排除今日)
    consolidation_df = df.iloc[-consolidation_days-1:-1]
    range_high = consolidation_df['high'].max()
    range_low = consolidation_df['low'].min()
    range_pct = (range_high - range_low) / range_low

    result['range_high'] = range_high
    result['range_low'] = range_low
    result['range_pct'] = range_pct

    # 檢查是否為窄幅整理
    if range_pct > max_range_pct:
        return result

    # 檢查是否突破整理區間上緣
    current_close = df['close'].iloc[-1]
    if current_close > range_high:
        # 計算量比
        avg_vol = consolidation_df['volume'].mean()
        current_vol = df['volume'].iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        result['detected'] = True
        result['volume_ratio'] = vol_ratio

        if vol_ratio >= 1.5:
            result['signal_text'] = f"窄幅整理放量突破 (整理幅度: {range_pct:.1%}, 量比: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 15
        else:
            result['signal_text'] = f"窄幅整理突破 (整理幅度: {range_pct:.1%}, 量比: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 8

    return result


def detect_trend_breakout(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    檢測趨勢突破 (收盤價突破20日高 + MA20向上)

    Args:
        df: DataFrame with OHLCV data
        lookback: 回溯期間

    Returns:
        dict: 趨勢突破信號
    """
    result = {
        'detected': False,
        'type': 'TREND_BREAKOUT',
        'high_20d': 0,
        'ma20_trend': None,
        'signal_text': '',
        'score_adjustment': 0
    }

    if len(df) < lookback + 10:
        return result

    # 計算20日高點 (排除今日)
    high_20d = df['high'].iloc[-lookback-1:-1].max()
    result['high_20d'] = high_20d

    # 計算 MA20 趨勢
    df_temp = df.copy()
    df_temp['MA20'] = df_temp['close'].rolling(window=20).mean()

    ma20_current = df_temp['MA20'].iloc[-1]
    ma20_5d_ago = df_temp['MA20'].iloc[-6]
    ma20_trend_up = ma20_current > ma20_5d_ago
    result['ma20_trend'] = 'UP' if ma20_trend_up else 'DOWN'

    # 檢查突破條件
    current_close = df['close'].iloc[-1]
    if current_close > high_20d and ma20_trend_up:
        # 計算量比
        avg_vol = df['volume'].iloc[-lookback-1:-1].mean()
        current_vol = df['volume'].iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        result['detected'] = True
        result['volume_ratio'] = vol_ratio

        if vol_ratio >= 1.5:
            result['signal_text'] = f"趨勢放量突破20日高 ${high_20d:.2f} (量比: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 18
        else:
            result['signal_text'] = f"趨勢突破20日高 ${high_20d:.2f} (量比: {vol_ratio:.1f}x)"
            result['score_adjustment'] = 10

    return result


def detect_volume_price_breakout(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    檢測量價配合突破 (突破 + 量增 + 收盤在最高點附近)

    Args:
        df: DataFrame with OHLCV data
        lookback: 回溯期間

    Returns:
        dict: 量價突破信號
    """
    result = {
        'detected': False,
        'type': 'VOLUME_PRICE_BREAKOUT',
        'close_ratio': 0,
        'signal_text': '',
        'score_adjustment': 0
    }

    if len(df) < lookback + 5:
        return result

    # 計算20日高點 (排除今日)
    high_20d = df['high'].iloc[-lookback-1:-1].max()

    # 當前K棒數據
    current_close = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    current_vol = df['volume'].iloc[-1]
    prev_vol = df['volume'].iloc[-2]

    # 計算收盤價在當日K棒中的位置 (0 = 最低, 1 = 最高)
    k_range = current_high - current_low
    if k_range > 0:
        close_ratio = (current_close - current_low) / k_range
    else:
        close_ratio = 0.5

    result['close_ratio'] = close_ratio

    # 量價配合條件
    # 1. 突破20日高
    # 2. 量增 > 20%
    # 3. 收盤在K棒上方 70%
    if (current_close > high_20d and
        current_vol > prev_vol * 1.2 and
        close_ratio > 0.7):

        avg_vol = df['volume'].iloc[-lookback-1:-1].mean()
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        result['detected'] = True
        result['volume_ratio'] = vol_ratio
        result['signal_text'] = f"量價配合突破 (收盤位置: {close_ratio:.0%}, 量比: {vol_ratio:.1f}x)"
        result['score_adjustment'] = 20

    return result


def get_advanced_breakout_signal(df: pd.DataFrame) -> Dict:
    """
    獲取進階突破信號 (整合所有型態檢測)

    Returns:
        dict: {
            'detected': bool,
            'patterns': list,  # 檢測到的型態列表
            'best_pattern': dict,  # 最佳型態
            'total_score': int,
            'signal_text': str
        }
    """
    result = {
        'detected': False,
        'patterns': [],
        'best_pattern': None,
        'total_score': 0,
        'signal_text': ''
    }

    # 執行所有型態檢測
    detectors = [
        ('W底突破', detect_w_bottom_breakout),
        ('杯柄突破', detect_cup_handle_breakout),
        ('旗形/三角突破', detect_flag_triangle_breakout),
        ('整理突破', detect_consolidation_breakout),
        ('趨勢突破', detect_trend_breakout),
        ('量價突破', detect_volume_price_breakout),
        ('基本突破', lambda df: get_breakout_signal(df)),
    ]

    detected_patterns = []

    for name, detector in detectors:
        try:
            signal = detector(df)
            if signal.get('detected'):
                signal['pattern_name'] = name
                detected_patterns.append(signal)
        except Exception:
            continue

    if not detected_patterns:
        return result

    result['detected'] = True
    result['patterns'] = detected_patterns

    # 找出最佳型態 (評分最高)
    best = max(detected_patterns, key=lambda x: x.get('score_adjustment', 0))
    result['best_pattern'] = best

    # 計算總評分 (取最高分，避免重複計算)
    result['total_score'] = best.get('score_adjustment', 0)

    # 如果有多個型態同時出現，額外加分
    if len(detected_patterns) > 1:
        result['total_score'] += 5

    # 生成信號文字
    pattern_texts = [p.get('signal_text', p.get('pattern_name', '')) for p in detected_patterns]
    result['signal_text'] = " | ".join(pattern_texts)

    return result


def format_advanced_breakout_output(signal: Dict) -> str:
    """格式化進階突破輸出"""
    if not signal.get('detected'):
        return "   未檢測到進階突破型態"

    lines = []

    for pattern in signal.get('patterns', []):
        pattern_name = pattern.get('pattern_name', '')
        signal_text = pattern.get('signal_text', '')
        score = pattern.get('score_adjustment', 0)

        emoji = '🚀' if score >= 15 else ('📈' if score >= 10 else '📊')
        lines.append(f"   {emoji} [{pattern_name}] {signal_text} (+{score}分)")

    if signal.get('total_score', 0) > 0:
        lines.append(f"   📊 總評分調整: +{signal['total_score']}分")

    return "\n".join(lines)


# ===============================
# 主程序測試
# ===============================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("真假突破檢測模組 (進階版)")
    print("=" * 60)

    # 模擬測試數據
    np.random.seed(42)
    n = 100

    # 創建帶有突破的數據
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(n) * 2)

    # 模擬放量突破
    volumes = np.random.randint(100000, 500000, n)
    volumes[-1] = int(volumes[-5:-1].mean() * 2)  # 今日放量

    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })

    # ================================
    # 1. 基本突破檢測
    # ================================
    print("\n[1] 基本突破檢測")
    print("-" * 40)
    signal = get_breakout_signal(df)

    print(f"突破檢測: {'是' if signal['detected'] else '否'}")
    if signal['detected']:
        print(f"類型: {signal['type']}")
        print(f"量比: {signal['volume_ratio']:.2f}x")
        print(format_breakout_output(signal))
        print(f"評分調整: {signal['score_adjustment']:+d}分")

    # ================================
    # 2. 進階型態檢測
    # ================================
    print("\n[2] 進階型態檢測")
    print("-" * 40)
    advanced_signal = get_advanced_breakout_signal(df)

    print(f"進階突破檢測: {'是' if advanced_signal['detected'] else '否'}")
    if advanced_signal['detected']:
        print(f"檢測到 {len(advanced_signal['patterns'])} 個型態:")
        print(format_advanced_breakout_output(advanced_signal))

    # ================================
    # 3. 個別型態測試
    # ================================
    print("\n[3] 個別型態測試")
    print("-" * 40)

    patterns = [
        ("W底突破", detect_w_bottom_breakout(df)),
        ("杯柄突破", detect_cup_handle_breakout(df)),
        ("旗形/三角", detect_flag_triangle_breakout(df)),
        ("整理突破", detect_consolidation_breakout(df)),
        ("趨勢突破", detect_trend_breakout(df)),
        ("量價突破", detect_volume_price_breakout(df)),
    ]

    for name, result in patterns:
        status = "V" if result['detected'] else "X"
        print(f"  [{status}] {name}")
        if result['detected']:
            print(f"      {result.get('signal_text', '')}")

    # ================================
    # 專家系統規則
    # ================================
    print("\n" + "=" * 60)
    print("專家系統規則:")
    print("=" * 60)
    print("基本突破:")
    print("  - 真突破 (放量 >= 1.5x): 買入並設定止損")
    print("  - 假突破 (縮量 < 1.5x): 賣出或觀望")
    print("  - 放量跌破: 減倉或停損")
    print("  - 縮量跌破: 觀望，可能為假跌破")
    print("\n進階型態:")
    print("  - W底突破: +20分 (放量) / +10分 (一般)")
    print("  - 杯柄突破: +25分 (放量) / +12分 (一般)")
    print("  - 旗形/三角: +18分 (放量) / +8分 (一般)")
    print("  - 整理突破: +15分 (放量) / +8分 (一般)")
    print("  - 趨勢突破: +18分 (放量) / +10分 (一般)")
    print("  - 量價突破: +20分 (完美配合)")
    print("=" * 60)
