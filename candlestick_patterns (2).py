"""
蠟燭圖型態分析模組 (CIO 嚴格審查版 v2)
========================================================
整合十字線、槌子線、吊人線等型態辨識功能
採用 CIO 嚴格標準：結合 RSI、乖離率、成交量過濾
供交易信號生成器調用
"""

import pandas as pd
import numpy as np


def analyze_candlestick_patterns(df, days=5):
    """
    分析最近N天的蠟燭圖型態 (CIO 嚴格版)

    參數:
        df: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 欄位
        days: 分析最近幾天，預設5天

    返回:
        dict: 包含型態分析結果
    """

    df = df.copy()

    # === 計算 K 線細節 ===
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']

    # 比例計算
    df['ratio_lower'] = df['lower_shadow'] / (df['body_size'] + 0.001)
    df['ratio_upper'] = df['upper_shadow'] / (df['body_size'] + 0.001)
    df['body_percent'] = df['body_size'] / (df['total_range'] + 0.001)

    # 判斷K線顏色
    df['color'] = np.where(df['close'] >= df['open'], '紅', '黑')

    # === CIO 嚴格標準：計算技術指標 ===
    # MA10 和 乖離率 (Bias)
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['bias10'] = (df['close'] - df['ma10']) / (df['ma10'] + 0.001) * 100

    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 成交量均線 (20日)
    df['vol_avg20'] = df['volume'].rolling(window=20).mean()

    # 簡單趨勢判斷 (用於兼容舊邏輯)
    df['trend'] = np.where(df['close'] > df['ma10'], '上升', '下跌')

    # 分析最近幾天
    recent_days = df.iloc[-days:]

    patterns = {
        'has_pattern': False,
        'latest_pattern': None,
        'pattern_date': None,
        'pattern_description': '',
        'bullish_signals': [],
        'bearish_signals': [],
        'detailed_analysis': []
    }

    for idx, (index, row) in enumerate(recent_days.iterrows()):
        # 獲取日期
        if 'Date' in row and row['Date'] is not None:
            date = pd.to_datetime(row['Date'])
        elif isinstance(index, pd.Timestamp):
            date = index
        else:
            try:
                date = pd.to_datetime(index)
            except:
                date = None

        day_analysis = {
            'date': date,
            'close': row['close'],
            'patterns': []
        }

        # 獲取前一天數據（用於缺口判斷）
        current_loc = df.index.get_loc(index)
        if current_loc > 0:
            prev_row = df.iloc[current_loc - 1]
        else:
            prev_row = None

        # === 1. 槌子線 (Hammer) - CIO 嚴格標準 ===
        # 條件：低檔(Bias < 0)、下影線 > 2倍實體、上影線極小、實體小
        is_hammer_strict = (
            row['bias10'] < 0 and
            row['ratio_lower'] >= 2.0 and
            row['ratio_upper'] < 0.6 and
            row['body_percent'] < 0.4
        )

        if is_hammer_strict:
            quality = "書本級" if row['ratio_lower'] >= 3.0 else "嚴格級"
            pattern_info = {
                'type': 'hammer',
                'name': '槌子線',
                'signal': 'bullish',
                'quality': quality,
                'ratio': row['ratio_lower'],
                'description': f"🔨 槌子線 ({quality}) - 底部止跌訊號"
            }
            day_analysis['patterns'].append(pattern_info)
            patterns['bullish_signals'].append('槌子線')
            patterns['has_pattern'] = True
            if idx == len(recent_days) - 1:
                patterns['latest_pattern'] = '槌子線'
                patterns['pattern_date'] = date

        # === 2. 吊人線 (Hanging Man) - CIO 嚴格標準 ===
        # 條件：高位(RSI > 60 或 Bias > 2)、下影線 > 2.5倍、上影線 < 實體一半
        is_uptrend = (row['rsi'] > 60) or (row['bias10'] > 2)
        is_hanging_strict = (
            is_uptrend and
            row['ratio_lower'] >= 2.5 and
            row['ratio_upper'] < 0.5 and
            row['body_percent'] < 0.3
        )

        if is_hanging_strict:
            pattern_info = {
                'type': 'hanging_man',
                'name': '吊人線',
                'signal': 'bearish',
                'ratio': row['ratio_lower'],
                'description': f"😵 吊人線 (嚴格級) - 高檔懸空警訊"
            }
            day_analysis['patterns'].append(pattern_info)
            patterns['bearish_signals'].append('吊人線')
            patterns['has_pattern'] = True
            if idx == len(recent_days) - 1:
                patterns['latest_pattern'] = '吊人線'
                patterns['pattern_date'] = date

        # === 3. 十字線 (Doji) - CIO 嚴格標準 ===
        # 條件：實體比例 < 10%
        if row['body_percent'] < 0.1:
            if row['upper_shadow'] > row['lower_shadow'] * 2:
                doji_type = "墓碑十字"
                signal = 'bearish'
                patterns['bearish_signals'].append('墓碑十字')
            elif row['lower_shadow'] > row['upper_shadow'] * 2:
                doji_type = "蜻蜓十字"
                signal = 'bullish'
                patterns['bullish_signals'].append('蜻蜓十字')
            else:
                doji_type = "標準十字"
                signal = 'neutral'

            pattern_info = {
                'type': 'doji',
                'name': doji_type,
                'signal': signal,
                'description': f"✨ {doji_type} - 多空平衡"
            }
            day_analysis['patterns'].append(pattern_info)
            patterns['has_pattern'] = True
            if idx == len(recent_days) - 1:
                patterns['latest_pattern'] = doji_type
                patterns['pattern_date'] = date

        # === 4. 吞噬型態 (Engulfing) ===
        if prev_row is not None:
            prev_body_top = max(prev_row['open'], prev_row['close'])
            prev_body_bottom = min(prev_row['open'], prev_row['close'])
            curr_body_top = max(row['open'], row['close'])
            curr_body_bottom = min(row['open'], row['close'])

            # 多頭吞噬
            if (row['color'] == '紅' and prev_row['color'] == '黑' and
                curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top):
                pattern_info = {
                    'type': 'bullish_engulfing',
                    'name': '多頭吞噬',
                    'signal': 'bullish',
                    'description': f"🐉 多頭吞噬 - 強力反轉訊號"
                }
                day_analysis['patterns'].append(pattern_info)
                patterns['bullish_signals'].append('多頭吞噬')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '多頭吞噬'
                    patterns['pattern_date'] = date

            # 空頭吞噬
            elif (row['color'] == '黑' and prev_row['color'] == '紅' and
                  curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top):
                pattern_info = {
                    'type': 'bearish_engulfing',
                    'name': '空頭吞噬',
                    'signal': 'bearish',
                    'description': f"🐻 空頭吞噬 - 強力反轉訊號"
                }
                day_analysis['patterns'].append(pattern_info)
                patterns['bearish_signals'].append('空頭吞噬')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '空頭吞噬'
                    patterns['pattern_date'] = date

            # === 5. 缺口 (Windows) - CIO 嚴格標準 ===
            # 使用 0.2% 緩衝避免誤判
            if row['low'] > prev_row['high'] * 1.002:
                gap_size = row['low'] - prev_row['high']
                pattern_info = {
                    'type': 'gap_up',
                    'name': '上升缺口',
                    'signal': 'bullish',
                    'gap_size': gap_size,
                    'description': f"🚀 上升缺口 (缺口: {gap_size:.2f})"
                }
                day_analysis['patterns'].append(pattern_info)
                patterns['bullish_signals'].append('上升缺口')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '上升缺口'
                    patterns['pattern_date'] = date

            elif row['high'] < prev_row['low'] * 0.998:
                gap_size = prev_row['low'] - row['high']
                pattern_info = {
                    'type': 'gap_down',
                    'name': '下降缺口',
                    'signal': 'bearish',
                    'gap_size': gap_size,
                    'description': f"📉 下降缺口 (缺口: {gap_size:.2f})"
                }
                day_analysis['patterns'].append(pattern_info)
                patterns['bearish_signals'].append('下降缺口')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '下降缺口'
                    patterns['pattern_date'] = date

        patterns['detailed_analysis'].append(day_analysis)

    # 生成綜合描述
    bullish_count = len(patterns['bullish_signals'])
    bearish_count = len(patterns['bearish_signals'])

    if bullish_count > bearish_count:
        patterns['pattern_description'] = f"最近{days}天出現{bullish_count}個看漲型態，偏多頭"
    elif bearish_count > bullish_count:
        patterns['pattern_description'] = f"最近{days}天出現{bearish_count}個看跌型態，偏空頭"
    else:
        patterns['pattern_description'] = f"最近{days}天多空型態均衡"

    return patterns


def format_pattern_output(patterns):
    """
    格式化型態分析結果輸出

    參數:
        patterns: analyze_candlestick_patterns() 的返回結果

    返回:
        str: 格式化的輸出文本
    """

    output = []
    output.append("=" * 80)
    output.append("📊 蠟燭圖型態分析 (CIO 嚴格標準)")
    output.append("=" * 80)

    if not patterns['has_pattern']:
        output.append("   😴 未發現顯著蠟燭圖型態 (盤整中)")
        return "\n".join(output)

    # 最新型態
    if patterns['latest_pattern']:
        output.append(f"   最新型態: {patterns['latest_pattern']}")
        if patterns['pattern_date'] is not None:
            try:
                if isinstance(patterns['pattern_date'], pd.Timestamp):
                    date_str = patterns['pattern_date'].strftime('%Y-%m-%d')
                else:
                    date_str = pd.to_datetime(patterns['pattern_date']).strftime('%Y-%m-%d')
                output.append(f"   出現日期: {date_str}")
            except Exception:
                try:
                    date_str = str(patterns['pattern_date'])
                    if 'Date' not in date_str and len(date_str) <= 50:
                        output.append(f"   出現日期: {date_str}")
                except:
                    pass

    # 綜合描述
    output.append(f"   {patterns['pattern_description']}")

    # 詳細列表
    if patterns['bullish_signals']:
        output.append(f"\n   看漲訊號 ({len(patterns['bullish_signals'])}):")
        for signal in set(patterns['bullish_signals']):
            count = patterns['bullish_signals'].count(signal)
            output.append(f"      • {signal} (出現{count}次)")

    if patterns['bearish_signals']:
        output.append(f"\n   看跌訊號 ({len(patterns['bearish_signals'])}):")
        for signal in set(patterns['bearish_signals']):
            count = patterns['bearish_signals'].count(signal)
            output.append(f"      • {signal} (出現{count}次)")

    # 最近型態詳情
    if patterns['detailed_analysis']:
        latest_day = patterns['detailed_analysis'][-1]
        if latest_day['patterns']:
            output.append(f"\n   今日型態:")
            for pattern in latest_day['patterns']:
                output.append(f"      {pattern['description']}")

    return "\n".join(output)


def get_pattern_score_adjustment(patterns):
    """
    根據蠟燭圖型態調整交易評分

    參數:
        patterns: analyze_candlestick_patterns() 的返回結果

    返回:
        float: 評分調整值 (-10 到 +10)
    """

    if not patterns['has_pattern']:
        return 0

    bullish_count = len(patterns['bullish_signals'])
    bearish_count = len(patterns['bearish_signals'])

    # 計算淨多空訊號
    net_signal = bullish_count - bearish_count

    # 檢查最新型態的重要性
    latest_bonus = 0
    if patterns['latest_pattern']:
        if patterns['latest_pattern'] in ['槌子線', '多頭吞噬', '蜻蜓十字', '上升缺口']:
            latest_bonus = 3
        elif patterns['latest_pattern'] in ['吊人線', '空頭吞噬', '墓碑十字', '下降缺口']:
            latest_bonus = -3

    # 計算總調整分數
    adjustment = (net_signal * 2) + latest_bonus

    # 限制在 -10 到 +10 之間
    return max(-10, min(10, adjustment))
