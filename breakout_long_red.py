"""
突破長紅檢測模組
================
檢測帶量突破長紅 K 棒信號

核心邏輯:
1. 突破: 收盤價 > 近期最高價 (阻力位)
2. 長紅: K棒實體 > 平均實體 * 1.8
3. 爆量: 成交量 > 平均量 * 2.5

組合條件 = 強勢突破信號
"""

import numpy as np
import pandas as pd


def detect_breakout_long_red(df, lookback=40, body_mult=1.8, vol_mult=2.5):
    """
    檢測突破長紅信號

    Args:
        df: DataFrame with OHLCV data
        lookback: 回溯期間計算阻力位
        body_mult: K棒實體倍數門檻
        vol_mult: 成交量倍數門檻

    Returns:
        bool: 是否符合突破長紅條件
    """
    if len(df) < lookback + 1:
        return False

    close = df['close'].values
    openp = df['open'].values
    high = df['high'].values
    volume = df['volume'].values

    # 阻力位 = 近期最高價
    resistance = high[-lookback:-1].max()

    # K棒實體
    body = close[-1] - openp[-1]
    avg_body = np.mean(np.abs(close[-21:-1] - openp[-21:-1]))

    # 量能
    avg_vol = volume[-21:-1].mean()

    # 三大條件
    breakout = close[-1] > resistance
    long_body = body > avg_body * body_mult
    volume_burst = volume[-1] > avg_vol * vol_mult

    return breakout and long_body and volume_burst


def get_breakout_long_red_signal(df, lookback=40, body_mult=1.8, vol_mult=2.5):
    """
    獲取突破長紅信號 (詳細版)

    Returns:
        dict: {
            'detected': bool,
            'breakout': bool,
            'long_body': bool,
            'volume_burst': bool,
            'resistance': float,
            'body_ratio': float,
            'volume_ratio': float,
            'signal_text': str,
            'score_adjustment': int
        }
    """
    result = {
        'detected': False,
        'breakout': False,
        'long_body': False,
        'volume_burst': False,
        'resistance': 0,
        'body_ratio': 0,
        'volume_ratio': 0,
        'signal_text': '',
        'score_adjustment': 0
    }

    if len(df) < lookback + 1:
        return result

    close = df['close'].values
    openp = df['open'].values
    high = df['high'].values
    volume = df['volume'].values

    # 阻力位
    resistance = high[-lookback:-1].max()
    result['resistance'] = resistance

    # K棒實體
    body = close[-1] - openp[-1]
    avg_body = np.mean(np.abs(close[-21:-1] - openp[-21:-1]))
    body_ratio = body / avg_body if avg_body > 0 else 0
    result['body_ratio'] = body_ratio

    # 量能
    avg_vol = volume[-21:-1].mean()
    volume_ratio = volume[-1] / avg_vol if avg_vol > 0 else 0
    result['volume_ratio'] = volume_ratio

    # 三大條件
    result['breakout'] = close[-1] > resistance
    result['long_body'] = body > avg_body * body_mult
    result['volume_burst'] = volume[-1] > avg_vol * vol_mult

    # 判斷結果
    conditions_met = sum([result['breakout'], result['long_body'], result['volume_burst']])

    if conditions_met == 3:
        result['detected'] = True
        result['signal_text'] = f"突破長紅! 阻力: {resistance:.2f}, 實體: {body_ratio:.1f}x, 量能: {volume_ratio:.1f}x"
        result['score_adjustment'] = 20
    elif conditions_met == 2:
        if result['breakout'] and result['volume_burst']:
            result['signal_text'] = f"爆量突破 (實體不足), 量能: {volume_ratio:.1f}x"
            result['score_adjustment'] = 10
        elif result['breakout'] and result['long_body']:
            result['signal_text'] = f"長紅突破 (量能不足), 實體: {body_ratio:.1f}x"
            result['score_adjustment'] = 8
    elif result['breakout']:
        result['signal_text'] = f"突破阻力 {resistance:.2f} (需確認)"
        result['score_adjustment'] = 5

    return result


def scan_breakout_long_red(tickers):
    """
    掃描多檔股票找出突破長紅

    Args:
        tickers: list of ticker symbols

    Returns:
        list: 符合條件的股票列表
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed")
        return []

    results = []

    print("=" * 60)
    print("突破長紅掃描器")
    print("=" * 60)
    print(f"掃描 {len(tickers)} 檔股票...\n")

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo")

            if df.empty:
                continue

            df.columns = [c.lower() for c in df.columns]

            signal = get_breakout_long_red_signal(df)

            if signal['detected']:
                print(f"  [FOUND] {ticker}: {signal['signal_text']}")
                results.append({
                    'ticker': ticker,
                    'signal': signal
                })
            elif signal['score_adjustment'] > 0:
                print(f"  [WATCH] {ticker}: {signal['signal_text']}")

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    print(f"\n找到 {len(results)} 檔突破長紅股票")
    print("=" * 60)

    return results


def format_breakout_long_red_output(signal):
    """格式化輸出"""
    if signal['detected']:
        return f"   🚀 {signal['signal_text']}"
    elif signal['signal_text']:
        return f"   📊 {signal['signal_text']}"
    else:
        return "   未檢測到突破長紅信號"


# ======================================================
# 主程序測試
# ======================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("Breakout Long Red Detector Test")
    print("=" * 60)

    # 模擬測試數據 - 創建突破長紅場景
    np.random.seed(42)
    n = 60

    # 基礎價格
    prices = 100 + np.cumsum(np.random.randn(n) * 1)
    volumes = np.random.randint(100000, 300000, n)

    # 模擬今日突破長紅
    prices[-1] = prices[-40:-1].max() + 5  # 突破阻力
    volumes[-1] = volumes[-21:-1].mean() * 3  # 爆量

    df = pd.DataFrame({
        'open': prices * 0.98,  # 開盤價較低，形成長紅
        'high': prices * 1.01,
        'low': prices * 0.97,
        'close': prices,
        'volume': volumes
    })

    # 測試
    signal = get_breakout_long_red_signal(df)

    print(f"\nDetected: {signal['detected']}")
    print(f"Breakout: {signal['breakout']}")
    print(f"Long Body: {signal['long_body']} (ratio: {signal['body_ratio']:.2f}x)")
    print(f"Volume Burst: {signal['volume_burst']} (ratio: {signal['volume_ratio']:.2f}x)")
    print(f"\n{format_breakout_long_red_output(signal)}")
    print(f"Score adjustment: {signal['score_adjustment']:+d}")

    print("\n" + "=" * 60)
    print("Conditions for Breakout Long Red:")
    print("  1. Breakout: Close > Recent High (resistance)")
    print("  2. Long Body: Body > Avg Body * 1.8")
    print("  3. Volume Burst: Volume > Avg Volume * 2.5")
