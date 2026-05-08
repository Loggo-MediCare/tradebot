"""
回测不同RSI情景的真实表现
验证：RSI>75 在不同技术指标组合下的后续涨跌概率
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import yfinance as yf
import numpy as np

def add_technical_indicators(df):
    """添加技术指标"""
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # 成交量均线
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']

    df = df.fillna(method='bfill').fillna(method='ffill')
    return df


def analyze_rsi_scenarios(ticker='2408.TW', start_date='2020-01-01'):
    """分析RSI>75后的不同情景"""

    print("=" * 80)
    print(f"回测 {ticker} 的RSI高位情景分析")
    print("=" * 80)

    # 下载数据
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)

    # 处理MultiIndex列
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                             'Open': 'open', 'High': 'high', 'Low': 'low'})
    df = df.reset_index()
    df = add_technical_indicators(df)

    print(f"\n数据范围: {df['Date'].min()} 到 {df['Date'].max()}")
    print(f"总天数: {len(df)}")

    # 定义情景
    scenarios = {
        '情景1: RSI>75 + MACD死叉 + 缩量': {
            'condition': lambda row: (
                row['rsi'] > 75 and
                row['macd'] < row['macd_signal'] and
                row['volume_ratio'] < 0.8
            ),
            'results': {'涨': 0, '跌': 0, '平': 0, 'total': 0}
        },
        '情景2: RSI>75 + MACD金叉 + 放量': {
            'condition': lambda row: (
                row['rsi'] > 75 and
                row['macd'] > row['macd_signal'] and
                row['volume_ratio'] > 1.3 and
                row['sma_10'] > row['sma_30']
            ),
            'results': {'涨': 0, '跌': 0, '平': 0, 'total': 0}
        },
        '情景3: RSI>75 + MACD金叉 + 极度放量': {
            'condition': lambda row: (
                row['rsi'] > 75 and
                row['macd'] > row['macd_signal'] and
                row['volume_ratio'] > 2.5
            ),
            'results': {'涨': 0, '跌': 0, '平': 0, 'total': 0}
        },
        '情景4: 单纯RSI>75': {
            'condition': lambda row: row['rsi'] > 75,
            'results': {'涨': 0, '跌': 0, '平': 0, 'total': 0}
        }
    }

    # 遍历数据
    for i in range(len(df) - 3):  # 保留3天用于观察后续表现
        row = df.iloc[i]

        # 计算未来3天的表现
        future_prices = df.iloc[i+1:i+4]['close'].values
        current_price = row['close']

        if len(future_prices) < 3:
            continue

        # 判断3天后的涨跌
        price_change = ((future_prices[-1] - current_price) / current_price) * 100

        if price_change > 2:
            outcome = '涨'
        elif price_change < -2:
            outcome = '跌'
        else:
            outcome = '平'

        # 检查每个情景
        for scenario_name, scenario in scenarios.items():
            try:
                if scenario['condition'](row):
                    scenario['results'][outcome] += 1
                    scenario['results']['total'] += 1
            except:
                pass

    # 输出结果
    print("\n" + "=" * 80)
    print("📊 回测结果：RSI>75后3天的表现")
    print("=" * 80)

    for scenario_name, scenario in scenarios.items():
        results = scenario['results']
        total = results['total']

        if total == 0:
            continue

        up_pct = (results['涨'] / total) * 100
        down_pct = (results['跌'] / total) * 100
        flat_pct = (results['平'] / total) * 100

        print(f"\n{scenario_name}")
        print(f"  样本数: {total}")
        print(f"  继续涨 (>+2%): {results['涨']} 次 ({up_pct:.1f}%)")
        print(f"  回调 (<-2%):   {results['跌']} 次 ({down_pct:.1f}%)")
        print(f"  横盘 (±2%):    {results['平']} 次 ({flat_pct:.1f}%)")

        # 给出建议
        if up_pct > 50:
            print(f"  💡 建议: 持有或加仓")
        elif down_pct > 60:
            print(f"  💡 建议: 立即卖出")
        elif down_pct > 40:
            print(f"  💡 建议: 减仓50-70%")
        else:
            print(f"  💡 建议: 小幅减仓20-30%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 回测2408
    analyze_rsi_scenarios('2408.TW', '2020-01-01')

    print("\n\n")

    # 回测其他台股
    for ticker in ['2330.TW', '2317.TW']:
        print("\n")
        analyze_rsi_scenarios(ticker, '2020-01-01')
