"""
爆量信號檢測模組
================
檢測法人上車的爆量信號

核心邏輯 (書籍理論):
- 爆量: 量比 > 1.5x (今日成交量 / 20日均量)
- 價格上漲: 漲幅 > 2%
- 法人上車暗示: 爆量 + 技術漂亮 (MA50 上升)

專家系統建議:
- 爆量 + 上漲 → 法人可能上車，建議買入
- 爆量 + 下跌 → 法人可能出貨，建議觀望
"""

import numpy as np
import pandas as pd
from shared_market_checks import get_intraday_adjusted_volume


def detect_volume_surge(df, volume_multiplier=1.5, price_change_threshold=0.02):
    """
    檢測爆量信號

    Args:
        df: DataFrame with OHLCV data
        volume_multiplier: 量比門檻 (預設 1.5x)
        price_change_threshold: 價格變化門檻 (預設 2%)

    Returns:
        dict: {
            'detected': bool,
            'type': str,
            'volume_ratio': float,
            'price_change': float,
            'signal_text': str
        }
    """
    result = {
        'detected': False,
        'type': None,
        'volume_ratio': 0,
        'price_change': 0,
        'signal_text': '',
        'score_adjustment': 0
    }

    if len(df) < 21:
        return result

    # 計算 20 日均量
    avg_volume = df['volume'].iloc[-21:-1].mean()

    if avg_volume <= 0:
        return result

    # 當前數據 (time-adjusted for intraday partial bars)
    current_volume = get_intraday_adjusted_volume(df)
    current_close = df['close'].iloc[-1]
    current_open = df['open'].iloc[-1]

    # 量比
    volume_ratio = current_volume / avg_volume
    result['volume_ratio'] = volume_ratio

    # 價格變化
    price_change = (current_close - current_open) / current_open if current_open > 0 else 0
    result['price_change'] = price_change

    # 檢測爆量
    is_volume_surge = volume_ratio > volume_multiplier

    if not is_volume_surge:
        return result

    result['detected'] = True

    # 分類信號
    if price_change > price_change_threshold:
        # 爆量上漲 - 法人上車
        result['type'] = 'SURGE_UP'
        result['signal_text'] = f"爆量上漲 (量比: {volume_ratio:.1f}x, 漲幅: {price_change:.1%})"
        result['score_adjustment'] = 15
    elif price_change < -price_change_threshold:
        # 爆量下跌 - 法人出貨
        result['type'] = 'SURGE_DOWN'
        result['signal_text'] = f"爆量下跌 (量比: {volume_ratio:.1f}x, 跌幅: {price_change:.1%})"
        result['score_adjustment'] = -15
    else:
        # 爆量平盤 - 換手
        result['type'] = 'SURGE_FLAT'
        result['signal_text'] = f"爆量換手 (量比: {volume_ratio:.1f}x)"
        result['score_adjustment'] = 0

    return result


def detect_continuous_volume(df, days=3, volume_multiplier=1.2):
    """
    檢測連續放量

    Args:
        df: DataFrame with OHLCV data
        days: 連續天數
        volume_multiplier: 量比門檻

    Returns:
        dict: 連續放量信號
    """
    result = {
        'detected': False,
        'days': 0,
        'avg_ratio': 0,
        'signal_text': ''
    }

    if len(df) < 21:
        return result

    # 計算每日量比
    avg_volume = df['volume'].iloc[-21:-days-1].mean()

    if avg_volume <= 0:
        return result

    # 檢查最近 N 天是否都放量
    continuous_days = 0
    total_ratio = 0

    for i in range(days):
        idx = -days + i
        daily_volume = df['volume'].iloc[idx]
        ratio = daily_volume / avg_volume

        if ratio > volume_multiplier:
            continuous_days += 1
            total_ratio += ratio
        else:
            break

    if continuous_days >= days:
        result['detected'] = True
        result['days'] = continuous_days
        result['avg_ratio'] = total_ratio / continuous_days
        result['signal_text'] = f"連續{continuous_days}天放量 (平均量比: {result['avg_ratio']:.1f}x)"

    return result


def get_volume_signal(df):
    """
    獲取完整的量能信號 (用於整合到交易系統)

    Returns:
        dict: {
            'surge': dict,  # 爆量信號
            'continuous': dict,  # 連續放量
            'score_adjustment': int,
            'signal_text': str
        }
    """
    result = {
        'surge': None,
        'continuous': None,
        'score_adjustment': 0,
        'signal_text': ''
    }

    # 檢測爆量
    surge = detect_volume_surge(df)
    result['surge'] = surge

    # 檢測連續放量
    continuous = detect_continuous_volume(df)
    result['continuous'] = continuous

    # 計算總評分
    texts = []

    if surge['detected']:
        result['score_adjustment'] += surge['score_adjustment']
        texts.append(surge['signal_text'])

    if continuous['detected']:
        # 連續放量額外加分
        result['score_adjustment'] += 5
        texts.append(continuous['signal_text'])

    result['signal_text'] = " | ".join(texts) if texts else ""

    return result


def format_volume_signal_output(signal):
    """格式化輸出"""
    lines = []

    if signal['surge'] and signal['surge']['detected']:
        surge = signal['surge']
        emoji = "🚀" if surge['type'] == 'SURGE_UP' else ("📉" if surge['type'] == 'SURGE_DOWN' else "🔄")
        lines.append(f"   {emoji} {surge['signal_text']}")

    if signal['continuous'] and signal['continuous']['detected']:
        lines.append(f"   📊 {signal['continuous']['signal_text']}")

    if not lines:
        lines.append("   量能正常，無特殊信號")

    return "\n".join(lines)


# ======================================================
# 主程序測試
# ======================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 50)
    print("Volume Surge Detector Test")
    print("=" * 50)

    # 模擬測試數據
    np.random.seed(42)
    n = 100

    # 創建帶爆量的數據
    prices = 100 + np.cumsum(np.random.randn(n) * 2)
    volumes = np.random.randint(100000, 500000, n)

    # 模擬今日爆量上漲
    prices[-1] = prices[-2] * 1.03  # 上漲 3%
    volumes[-1] = volumes[-5:-1].mean() * 2.5  # 量比 2.5x

    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })

    # 測試
    signal = get_volume_signal(df)

    print(f"\nVolume Signal:")
    print(format_volume_signal_output(signal))
    print(f"\nScore adjustment: {signal['score_adjustment']:+d}")

    print("\n" + "=" * 50)
    print("Expert System Rules:")
    print("  - Surge UP (volume > 1.5x, price > 2%): +15 pts (institutional buying)")
    print("  - Surge DOWN (volume > 1.5x, price < -2%): -15 pts (institutional selling)")
    print("  - Continuous volume (3+ days): +5 pts")
