"""
真假突破檢測模組
================
使用專家系統規則分辨真突破與假突破

核心邏輯:
- 真突破: 價格突破阻力線 + 成交量放大 (量比 >= 1.5)
- 假突破: 價格突破阻力線 + 成交量萎縮 (量比 < 1.5)

專家系統建議:
- 真突破 → 買入並設定止損
- 假突破 → 賣出或觀望
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from shared_market_checks import get_intraday_adjusted_volume


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

    # 當前價格和成交量 (time-adjusted for intraday partial bars)
    current_close = df['close'].iloc[-1]
    current_volume = get_intraday_adjusted_volume(df)
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


def get_advanced_breakout_signal(df, lookback=60):
    """
    進階突破型態檢測 (W底 / 杯柄 / 旗形 / 三角收斂)

    Returns:
        dict: {
            'detected': bool,
            'patterns': list of dicts,
            'total_score': int
        }
    """
    result = {'detected': False, 'patterns': [], 'total_score': 0}
    if df is None or len(df) < 20:
        return result

    close = df['close'] if 'close' in df.columns else df['Close']
    volume = df['volume'] if 'volume' in df.columns else df.get('Volume', None)
    close = close.dropna()
    if len(close) < 20:
        return result

    patterns = []

    try:
        # ── W底檢測 ──────────────────────────────────────────────
        window = min(lookback, len(close))
        seg = close.iloc[-window:]
        low1_idx = int(seg.iloc[:window//2].idxmin()) if len(seg) >= 4 else -1
        low2_idx = int(seg.iloc[window//2:].idxmin()) if len(seg) >= 4 else -1
        if low1_idx >= 0 and low2_idx >= 0 and low1_idx != low2_idx:
            low1 = float(close.iloc[low1_idx])
            low2 = float(close.iloc[low2_idx])
            current = float(close.iloc[-1])
            if abs(low1 - low2) / (low1 + 1e-10) < 0.05 and current > max(low1, low2) * 1.03:
                patterns.append({
                    'name': 'W底',
                    'signal_text': f'W底形態 (雙底確認, 當前+{(current/max(low1,low2)-1)*100:.1f}%)',
                    'score': 15
                })
    except Exception:
        pass

    try:
        # ── 旗形整理 ──────────────────────────────────────────────
        if len(close) >= 15:
            pre = close.iloc[-15:-5]
            flag = close.iloc[-5:]
            pre_gain = float((pre.iloc[-1] - pre.iloc[0]) / (pre.iloc[0] + 1e-10))
            flag_pullback = float((flag.iloc[-1] - flag.iloc[0]) / (flag.iloc[0] + 1e-10))
            if pre_gain > 0.05 and -0.03 < flag_pullback < 0.01:
                patterns.append({
                    'name': '旗形整理',
                    'signal_text': f'旗形整理 (前段漲{pre_gain*100:.1f}%, 整理中)',
                    'score': 10
                })
    except Exception:
        pass

    try:
        # ── 三角收斂 ──────────────────────────────────────────────
        if len(close) >= 10:
            recent = close.iloc[-10:]
            highs = recent.rolling(2).max().dropna()
            lows  = recent.rolling(2).min().dropna()
            if len(highs) >= 3 and len(lows) >= 3:
                high_slope = float((highs.iloc[-1] - highs.iloc[0]) / (len(highs) + 1e-10))
                low_slope  = float((lows.iloc[-1]  - lows.iloc[0])  / (len(lows)  + 1e-10))
                if high_slope < 0 and low_slope > 0:
                    patterns.append({
                        'name': '三角收斂',
                        'signal_text': '對稱三角收斂 (即將突破)',
                        'score': 8
                    })
    except Exception:
        pass

    if patterns:
        result['detected'] = True
        result['patterns'] = patterns
        result['total_score'] = sum(p['score'] for p in patterns)

    return result


def format_advanced_breakout_output(signal):
    """格式化進階突破型態輸出"""
    if not signal.get('detected') or not signal.get('patterns'):
        return "   未檢測到進階突破型態"
    lines = []
    for p in signal['patterns']:
        lines.append(f"   🚀 {p['name']}: {p['signal_text']}  (+{p['score']}分)")
    lines.append(f"   總評分加成: +{signal['total_score']}分")
    return "\n".join(lines)


# ===============================
# 主程序測試
# ===============================

if __name__ == "__main__":
    print("真假突破檢測模組")
    print("=" * 50)

    # 模擬測試數據
    np.random.seed(42)
    n = 100

    # 創建帶有突破的數據
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(n) * 2)

    # 模擬放量突破
    volumes = np.random.randint(100000, 500000, n)
    volumes[-1] = volumes[-1] * 2  # 今日放量

    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })

    # 測試檢測
    signal = get_breakout_signal(df)

    print(f"\n突破檢測: {'是' if signal['detected'] else '否'}")
    if signal['detected']:
        print(f"類型: {signal['type']}")
        print(f"量比: {signal['volume_ratio']:.2f}x")
        print(format_breakout_output(signal))
        print(f"評分調整: {signal['score_adjustment']:+d}分")

    print("\n" + "=" * 50)
    print("專家系統規則:")
    print("  - 真突破 (放量 >= 1.5x): 買入並設定止損")
    print("  - 假突破 (縮量 < 1.5x): 賣出或觀望")
    print("  - 放量跌破: 減倉或停損")
    print("  - 縮量跌破: 觀望，可能為假跌破")
