"""
蠟燭圖型態分析模組 (Candlestick Pattern Analysis Module)
========================================================
整合十字線、槌子線、上升趨勢中的槌子等型態辨識功能
供交易信號生成器調用
"""

import pandas as pd
import numpy as np


GRAVESTONE_MODE = "rename"
GRAVESTONE_RENAME_LABEL = "長上影十字"


def analyze_candlestick_patterns(df, days=5):
    """
    分析最近N天的蠟燭圖型態

    參數:
        df: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 欄位
        days: 分析最近幾天，預設5天

    返回:
        dict: 包含型態分析結果
    """

    # 計算實體與影線
    df = df.copy()
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']

    # 計算比例
    df['ratio_lower'] = df['lower_shadow'] / (df['body_size'] + 0.001)
    df['ratio_upper'] = df['upper_shadow'] / (df['body_size'] + 0.001)

    # 判斷K線顏色
    df['color'] = np.where(df['close'] >= df['open'], '紅', '黑')

    # 計算移動平均判斷趨勢
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma10_slope'] = df['ma10'] - df['ma10'].shift(3)
    # 上升趨勢判斷：價格在MA10之上，且MA10近3天呈上揚
    df['trend'] = np.where(
        (df['close'] > df['ma10']) & (df['ma10_slope'] > 0), '上升', '下跌'
    )

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
        # Try to get date from row['Date'] column first, fallback to index
        if 'Date' in row and row['Date'] is not None:
            date = pd.to_datetime(row['Date'])
        elif isinstance(index, pd.Timestamp):
            date = index
        else:
            # If index is not a timestamp, try to convert it
            try:
                date = pd.to_datetime(index)
            except:
                # Fallback: use None and handle later
                date = None

        day_analysis = {
            'date': date,
            'close': row['close'],
            'patterns': []
        }

        # === 1. 槌子線 (Hammer) - 修正後的嚴格判定 ===
        # 1. 下影線必須明顯長於實體 (2.5倍以上)
        # 2. 上影線必須非常短 (不超過實體的0.5倍) - 這是關鍵！
        # 3. 實體必須夠小 (佔全天波動25%以下)
        is_long_lower = row['lower_shadow'] > (row['body_size'] * 2.5)
        is_strict_lower = row['lower_shadow'] > (row['body_size'] * 3.5)  # 書本級標準更嚴格
        is_short_upper = row['upper_shadow'] < (row['body_size'] * 0.5)  # 限制上影線不能太長
        is_small_body = row['body_size'] < (row['total_range'] * 0.25)  # 縮小實體佔比

        # === 除錯：印出實際收到的數字（僅最新一天） ===
        if idx == len(recent_days) - 1:
            print(f"\n=== DEBUG 最新一天 {date} ===")
            print(f"O: {row['open']:,.0f} | H: {row['high']:,.0f} | L: {row['low']:,.0f} | C: {row['close']:,.0f}")
            print(f"Body: {row['body_size']:,.0f} | Lower shadow: {row['lower_shadow']:,.0f} | Upper shadow: {row['upper_shadow']:,.0f}")
            print(f"Total range: {row['total_range']:,.0f} (100%)")
            print(f"Body 佔比: {(row['body_size'] / row['total_range'] * 100):.1f}% | 下影線倍數: {(row['lower_shadow'] / (row['body_size'] + 0.001)):.2f}x | 上影線倍數: {(row['upper_shadow'] / (row['body_size'] + 0.001)):.2f}x")
            print(f"is_small_body: {is_small_body} (<25% ✓) | is_long_lower: {is_long_lower} (>2.5x ✓) | is_short_upper: {is_short_upper} (<0.5x ✓)")
            print(f"Trend: {row['trend']}")
            print("=" * 60)

        if is_long_lower and is_short_upper and is_small_body:
            if row['trend'] == '下跌':
                quality = "書本級" if is_strict_lower else "普通級"
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
                if idx == len(recent_days) - 1:  # 最新一天
                    patterns['latest_pattern'] = '槌子線'
                    patterns['pattern_date'] = date

            elif row['trend'] == '上升':
                pattern_info = {
                    'type': 'hanging_man',
                    'name': '上升趨勢中的槌子',
                    'signal': 'bearish',
                    'ratio': row['ratio_lower'],
                    'description': f"😵 上升趨勢中的槌子 - 高檔懸空警訊"
                }
                day_analysis['patterns'].append(pattern_info)
                patterns['bearish_signals'].append('上升趨勢中的槌子')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '上升趨勢中的槌子'
                    patterns['pattern_date'] = date

        # === 2. 十字線 (Doji) ===
        doji_threshold = row['close'] * 0.002
        if row['body_size'] <= doji_threshold:
            if row['upper_shadow'] > row['lower_shadow'] * 2:
                if GRAVESTONE_MODE == "hide":
                    doji_type = "標準十字"
                    signal = 'neutral'
                elif GRAVESTONE_MODE == "rename":
                    doji_type = GRAVESTONE_RENAME_LABEL
                    signal = 'bearish'
                    patterns['bearish_signals'].append(GRAVESTONE_RENAME_LABEL)
                else:
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

        # === 3. 吞噬型態 (Engulfing) ===
        if idx > 0:
            prev_row = recent_days.iloc[idx - 1]
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

            # === 4. 缺口 (Windows) ===
            prev_high = prev_row['high']
            prev_low = prev_row['low']

            if row['low'] > prev_high:
                gap_size = row['low'] - prev_high
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

            elif row['high'] < prev_low:
                gap_size = prev_low - row['high']
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

            # === 5. 強勢化解 (Bearish Failure) ===
            # 昨天有長上影線（不論十字或射擊之星），今天收盤化解昨日賣壓
            if prev_row['upper_shadow'] > prev_row['body_size'] * 2:
                prev_is_doji = prev_row['body_size'] <= prev_row['close'] * 0.002
                if prev_is_doji:
                    # Doji 型態：實體極小，只需收盤超過昨日收盤即算化解
                    neutralized = row['close'] > max(prev_row['open'], prev_row['close'])
                    threshold_label = f"昨日收盤 {max(prev_row['open'], prev_row['close']):.0f}"
                else:
                    # 非 Doji：需收盤突破昨日最高點
                    neutralized = row['close'] > prev_row['high']
                    threshold_label = f"昨日高點 {prev_row['high']:.0f}"
                if neutralized:
                    pattern_info = {
                        'type': 'bearish_failure',
                        'name': '強勢化解',
                        'signal': 'bullish',
                        'description': f"⚡ 強勢化解 - 今日收盤 {row['close']:.0f} 突破{threshold_label}"
                    }
                    day_analysis['patterns'].append(pattern_info)
                    patterns['bullish_signals'].append('強勢化解')
                    patterns['has_pattern'] = True
                    if idx == len(recent_days) - 1:
                        patterns['latest_pattern'] = '強勢化解'
                        patterns['pattern_date'] = date

            # === 6. 多頭強攻 (Bullish Thrust) ===
            # 今日實體 > 昨日實體 3 倍，且收盤創近5日新高
            curr_body = row['body_size']
            prev_body = prev_row['body_size']
            lookback_high = recent_days.iloc[max(0, idx - 5):idx]['high']
            recent_high = lookback_high.max() if len(lookback_high) > 0 else row['high']
            if (curr_body > prev_body * 3 and row['color'] == '紅' and row['close'] >= recent_high):
                pattern_info = {
                    'type': 'bullish_thrust',
                    'name': '多頭強攻',
                    'signal': 'bullish',
                    'description': f"🚀 多頭強攻 - 今日實體 {curr_body:.0f} 超過昨日 {prev_body:.0f} 的3倍，且收盤創近期新高"
                }
                day_analysis['patterns'].append(pattern_info)
                patterns['bullish_signals'].append('多頭強攻')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '多頭強攻'
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
    output.append("📊 蠟燭圖型態分析")
    output.append("=" * 80)

    if not patterns['has_pattern']:
        output.append("   未發現顯著蠟燭圖型態")
        return "\n".join(output)

    # 最新型態
    if patterns['latest_pattern']:
        output.append(f"   最新型態: {patterns['latest_pattern']}")
        if patterns['pattern_date'] is not None:
            try:
                # Handle both Timestamp and other date formats
                if isinstance(patterns['pattern_date'], pd.Timestamp):
                    date_str = patterns['pattern_date'].strftime('%Y-%m-%d')
                else:
                    date_str = pd.to_datetime(patterns['pattern_date']).strftime('%Y-%m-%d')
                output.append(f"   出現日期: {date_str}")
            except Exception as e:
                # If date conversion fails, try to use string representation
                try:
                    date_str = str(patterns['pattern_date'])
                    if 'Date' in date_str or len(date_str) > 50:
                        # Skip if it looks like column metadata
                        pass
                    else:
                        output.append(f"   出現日期: {date_str}")
                except:
                    pass

    # 綜合描述
    output.append(f"   {patterns['pattern_description']}")

    # 從 detailed_analysis 反查每個訊號實際出現日期
    bullish_dates_map = {}
    bearish_dates_map = {}
    for day in patterns.get('detailed_analysis', []):
        day_date = day.get('date')
        try:
            date_str = pd.to_datetime(day_date).strftime('%Y-%m-%d') if day_date is not None else None
        except Exception:
            date_str = None
        if not date_str:
            continue

        for p in day.get('patterns', []):
            name = p.get('name')
            signal = p.get('signal')
            if not name:
                continue
            if signal == 'bullish':
                bullish_dates_map.setdefault(name, []).append(date_str)
            elif signal == 'bearish':
                bearish_dates_map.setdefault(name, []).append(date_str)

    # 詳細列表
    if patterns['bullish_signals']:
        output.append(f"\n   看漲訊號 ({len(patterns['bullish_signals'])}):")
        for signal in set(patterns['bullish_signals']):
            count = patterns['bullish_signals'].count(signal)
            dates = bullish_dates_map.get(signal, [])
            date_text = f", 日期: {', '.join(dates)}" if dates else ""
            output.append(f"      • {signal} (出現{count}次{date_text})")

    if patterns['bearish_signals']:
        output.append(f"\n   看跌訊號 ({len(patterns['bearish_signals'])}):")
        for signal in set(patterns['bearish_signals']):
            count = patterns['bearish_signals'].count(signal)
            dates = bearish_dates_map.get(signal, [])
            date_text = f", 日期: {', '.join(dates)}" if dates else ""
            output.append(f"      • {signal} (出現{count}次{date_text})")

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

    # 檢查最新型態的重要性：從 signal 欄位直接判斷，避免名稱清單維護漏洞
    latest_bonus = 0
    if patterns['latest_pattern'] and patterns['detailed_analysis']:
        latest_day = patterns['detailed_analysis'][-1]
        for p in latest_day['patterns']:
            if p['name'] == patterns['latest_pattern']:
                if p['signal'] == 'bullish':
                    latest_bonus = 3
                elif p['signal'] == 'bearish':
                    latest_bonus = -3
                break

    # 計算總調整分數
    adjustment = (net_signal * 2) + latest_bonus

    # 限制在 -10 到 +10 之間
    return max(-10, min(10, adjustment))
