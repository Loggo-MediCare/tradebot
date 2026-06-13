"""
蠟燭圖型態分析模組 (Candlestick Pattern Analysis Module) - CIO 修正版
========================================================
整合十字線、槌子線、吊人線等型態辨識功能
2026-02-03 修正：強化吊人線判定邏輯、加入 RSI 與 乖離率 加權、動態成交量扣分
"""

import pandas as pd
import numpy as np


def analyze_candlestick_patterns(df, days=5):
    """
    分析最近N天的蠟燭圖型態 (CIO 修正版)

    參數:
        df: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 欄位
        days: 分析最近幾天，預設5天
    """

    # 確保資料足夠
    if len(df) < 20:
        return {'has_pattern': False, 'latest_pattern': None, 'bullish_signals': [], 'bearish_signals': [], 'detailed_analysis': []}

    df = df.copy()
    
    # 1. 計算基礎實體與影線
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']

    # 2. 計算比例
    df['ratio_lower'] = df['lower_shadow'] / (df['body_size'] + 0.001)
    df['ratio_upper'] = df['upper_shadow'] / (df['body_size'] + 0.001)
    df['color'] = np.where(df['close'] >= df['open'], '紅', '黑')

    # 3. 計算技術指標 (用於趨勢加權)
    # MA10 與 乖離率 (Bias)
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['bias_10'] = (df['close'] - df['ma10']) / df['ma10'] * 100
    
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 成交量分析
    df['avg_vol_20'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / (df['avg_vol_20'] + 1e-10)

    # 4. 趨勢判定 (CIO 建議：乖離率 或 RSI 加權)
    # 定義上升趨勢：價格在MA10之上 且 (RSI強勢 或 乖離率正向)
    df['is_uptrend'] = (df['close'] > df['ma10']) & ((df['rsi'] > 60) | (df['bias_10'] > 2))
    # 定義下跌趨勢：價格在MA10之下 且 (RSI弱勢 或 乖離率負向)
    df['is_downtrend'] = (df['close'] < df['ma10']) & ((df['rsi'] < 40) | (df['bias_10'] < -2))

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
        date = index if isinstance(index, pd.Timestamp) else pd.to_datetime(index)
        
        day_analysis = {
            'date': date,
            'close': row['close'],
            'vol_ratio': row['vol_ratio'],
            'patterns': []
        }

        # === 1. 槌子線 與 吊人線 (邏輯強化) ===
        # 共同結構：下影線長、實體小
        is_small_body = row['body_size'] < (row['total_range'] * 0.4)
        
        # A. 槌子線 (Hammer) - 底部訊號
        # 門檻：下影線 > 實體 2 倍，且上影線極小
        if (row['ratio_lower'] >= 2.0) and (row['ratio_upper'] < 0.6) and is_small_body:
            if row['is_downtrend']:
                quality = "書本級" if row['ratio_lower'] >= 3.0 else "普通級"
                pattern_info = {
                    'type': 'hammer',
                    'name': '槌子線',
                    'signal': 'bullish',
                    'vol_ratio': row['vol_ratio'],
                    'description': f"🔨 槌子線 ({quality}) - 底部止跌訊號 (下影線:{row['ratio_lower']:.1f}x)"
                }
                day_analysis['patterns'].append(pattern_info)
                patterns['bullish_signals'].append('槌子線')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '槌子線'
                    patterns['pattern_date'] = date

        # B. 吊人線 (Hanging Man) - 高檔警訊 (CIO 修正處)
        # 修正 1：下影線門檻提高至 2.5 倍
        # 修正 2：嚴格限制上影線必須小於實體一半 (ratio_upper < 0.5)
        if (row['ratio_lower'] >= 2.5) and (row['ratio_upper'] < 0.5) and is_small_body:
            # 修正 3：必須在強勢上升趨勢中 (RSI/Bias 加權)
            if row['is_uptrend']:
                pattern_info = {
                    'type': 'hanging_man',
                    'name': '吊人線',
                    'signal': 'bearish',
                    'vol_ratio': row['vol_ratio'], # 用於後續動態扣分
                    'description': f"😵 吊人線 - 高檔懸空警訊 (下影線:{row['ratio_lower']:.1f}x)"
                }
                day_analysis['patterns'].append(pattern_info)
                patterns['bearish_signals'].append('吊人線')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '吊人線'
                    patterns['pattern_date'] = date

        # === 2. 十字線 (Doji) ===
        doji_threshold = row['close'] * 0.0015
        if row['body_size'] <= doji_threshold:
            signal = 'neutral'
            _tr = row['total_range'] + 0.001
            # 墓碑十字：下影線 < 全幅5%（開收盤≈最低點），上影線長
            if row['upper_shadow'] > row['lower_shadow'] * 2 and row['lower_shadow'] < _tr * 0.05:
                doji_type = "墓碑十字"
                signal = 'bearish'
                patterns['bearish_signals'].append('墓碑十字')
            # 蜻蜓十字：上影線 < 全幅5%（開收盤≈最高點），下影線長
            elif row['lower_shadow'] > row['upper_shadow'] * 2 and row['upper_shadow'] < _tr * 0.05:
                doji_type = "蜻蜓十字"
                signal = 'bullish'
                patterns['bullish_signals'].append('蜻蜓十字')
            else:
                doji_type = "標準十字"
            
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
            # 多頭吞噬
            if (row['color'] == '紅' and prev_row['color'] == '黑' and row['close'] > prev_row['open'] and row['open'] < prev_row['close']):
                pattern_info = {'type': 'bullish_engulfing', 'name': '多頭吞噬', 'signal': 'bullish', 'description': "🐉 多頭吞噬 - 強力反轉"}
                day_analysis['patterns'].append(pattern_info)
                patterns['bullish_signals'].append('多頭吞噬')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '多頭吞噬'
                    patterns['pattern_date'] = date
            # 空頭吞噬
            elif (row['color'] == '黑' and prev_row['color'] == '紅' and row['close'] < prev_row['open'] and row['open'] > prev_row['close']):
                pattern_info = {'type': 'bearish_engulfing', 'name': '空頭吞噬', 'signal': 'bearish', 'description': "🐻 空頭吞噬 - 強力反轉"}
                day_analysis['patterns'].append(pattern_info)
                patterns['bearish_signals'].append('空頭吞噬')
                patterns['has_pattern'] = True
                if idx == len(recent_days) - 1:
                    patterns['latest_pattern'] = '空頭吞噬'
                    patterns['pattern_date'] = date

        patterns['detailed_analysis'].append(day_analysis)

    # 綜合描述
    b_count, s_count = len(patterns['bullish_signals']), len(patterns['bearish_signals'])
    patterns['pattern_description'] = f"最近{days}天出現{b_count}多{s_count}空型態"
    
    return patterns


def get_pattern_score_adjustment(patterns):
    """
    根據蠟燭圖型態調整評分 (CIO 修正版：動態成交量權重)
    """
    if not patterns['has_pattern']: return 0

    score = (len(patterns['bullish_signals']) - len(patterns['bearish_signals'])) * 2.0
    
    # 最新型態加成
    if patterns['latest_pattern']:
        latest_day = patterns['detailed_analysis'][-1]
        
        # 針對「吊人線」的動態扣分邏輯
        if patterns['latest_pattern'] == '吊人線':
            vol_ratio = latest_day.get('vol_ratio', 1.0)
            # CIO 建議：若成交量沒放大 (小於1.2倍均量)，扣分減半
            penalty = -4.0 if vol_ratio > 1.2 else -2.0
            score += penalty
        elif patterns['latest_pattern'] in ['槌子線', '多頭吞噬', '上升缺口']:
            score += 3.0
        elif patterns['latest_pattern'] in ['空頭吞噬', '墓碑十字']:
            score -= 3.0

    return max(-10, min(10, score))


def format_pattern_output(patterns):
    """ 格式化輸出 """
    output = ["=" * 40, "📊 蠟燭圖型態分析 (CIO 修正版)", "=" * 40]
    if not patterns['has_pattern']:
        output.append("   未發現顯著型態")
        return "\n".join(output)
    
    output.append(f"   最新型態: {patterns['latest_pattern']}")
    if patterns['pattern_date']:
        date_str = patterns['pattern_date'].strftime('%Y-%m-%d') if isinstance(patterns['pattern_date'], pd.Timestamp) else str(patterns['pattern_date'])
        output.append(f"   出現日期: {date_str}")
    output.append(f"   {patterns['pattern_description']}")
    
    if patterns['bearish_signals']:
        output.append(f"\n   📉 看跌訊號: {', '.join(set(patterns['bearish_signals']))}")
    if patterns['bullish_signals']:
        output.append(f"\n   📈 看漲訊號: {', '.join(set(patterns['bullish_signals']))}")
        
    return "\n".join(output)
