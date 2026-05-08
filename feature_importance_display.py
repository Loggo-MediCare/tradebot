"""
特徵重要性顯示模組
用於在交易信號中顯示關鍵技術指標的重要性
"""

import json
import os
from pathlib import Path


def load_feature_importance(symbol):
    """
    加載特徵重要性數據

    Parameters:
    -----------
    symbol : str
        股票代號 (e.g., "2330.TW", "HTGC")

    Returns:
    --------
    dict or None: 特徵重要性數據，如果文件不存在返回None
    """
    # 標準化symbol格式
    # 保持原格式：2344.TW -> 2344.TW, HTGC -> HTGC
    file_symbol = symbol
    feature_file = Path(__file__).parent / f"{file_symbol}_feature_importance.json"

    if feature_file.exists():
        try:
            with open(feature_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  無法讀取特徵重要性數據: {e}")
            return None
    else:
        return None


def get_top_features(symbol, top_n=5):
    """
    獲取最重要的N個特徵

    Parameters:
    -----------
    symbol : str
        股票代號
    top_n : int
        返回前N個重要特徵（默認5個）

    Returns:
    --------
    list of tuple: [(feature_name, importance_score), ...]
    """
    data = load_feature_importance(symbol)

    if data is None or 'feature_importance' not in data:
        return []

    # 排序並取前N個
    features = data['feature_importance']
    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)

    return sorted_features[:top_n]


def format_feature_importance_display(symbol, show_in_buy_signal=True):
    """
    格式化特徵重要性顯示

    Parameters:
    -----------
    symbol : str
        股票代號
    show_in_buy_signal : bool
        是否用於買入信號顯示（簡化版）

    Returns:
    --------
    str: 格式化的輸出文本
    """
    data = load_feature_importance(symbol)

    if data is None:
        return None

    top_features = get_top_features(symbol, top_n=5)

    if not top_features:
        return None

    # 中文名稱映射
    feature_name_map = {
        'MA50_slope': 'MA50斜率',
        'MA_50': 'MA50均線',
        'MA_20': 'MA20均線',
        'MA_200': 'MA200均線',
        'OBV': 'OBV成交量',
        'OBV_MA': 'OBV均線',
        'macd': 'MACD',
        'macd_signal': 'MACD信號線',
        'macd_hist': 'MACD柱狀圖',
        'rsi': 'RSI相對強弱',
        'ATR': 'ATR波動',
        'volatility': '波動率',
        'bb_position': '布林帶位置',
        'price_change_5d': '5日漲跌幅',
        'price_change_20d': '20日漲跌幅',
        'K': 'KD-K值',
        'D': 'KD-D值',
    }

    if show_in_buy_signal:
        # 簡化版：用於買入信號
        output = []
        output.append("\n   💎 關鍵指標 (依重要性):")
        for i, (feature, importance) in enumerate(top_features, 1):
            cn_name = feature_name_map.get(feature, feature)
            output.append(f"      {i}. {cn_name}: {importance*100:.1f}%")

        return "\n".join(output)
    else:
        # 完整版：獨立顯示
        output = []
        output.append("=" * 80)
        output.append("📊 AI模型關鍵指標分析")
        output.append("=" * 80)
        output.append(f"模型準確度: {data.get('model_accuracy', 0)*100:.1f}%")
        output.append(f"\n前5個最重要的技術指標:")
        output.append("")

        for i, (feature, importance) in enumerate(top_features, 1):
            cn_name = feature_name_map.get(feature, feature)
            # 使用條形圖顯示重要性
            bar_length = int(importance * 50)  # 最大50個字符
            bar = "█" * bar_length
            output.append(f"   {i}. {cn_name:15} {bar} {importance*100:.2f}%")

        output.append("")
        output.append("💡 說明: 這些指標對AI模型的決策影響最大")

        return "\n".join(output)


def get_feature_insight(symbol, current_indicators):
    """
    根據當前指標值和特徵重要性提供洞察

    Parameters:
    -----------
    symbol : str
        股票代號
    current_indicators : dict
        當前技術指標值 {'rsi': 75.2, 'macd': 0.5, ...}

    Returns:
    --------
    str: 洞察文本
    """
    top_features = get_top_features(symbol, top_n=3)

    if not top_features:
        return ""

    # 中文名稱映射
    feature_name_map = {
        'MA50_slope': 'MA50斜率',
        'MA_50': 'MA50均線',
        'MA_20': 'MA20均線',
        'OBV_MA': 'OBV均線',
        'macd': 'MACD',
        'macd_signal': 'MACD信號線',
        'rsi': 'RSI',
        'ATR': 'ATR',
        'volatility': '波動率',
    }

    insights = []
    insights.append("🔍 AI模型最關注的指標:")

    for i, (feature, importance) in enumerate(top_features, 1):
        cn_name = feature_name_map.get(feature, feature)
        insights.append(f"   • {cn_name} (重要性: {importance*100:.1f}%)")

    return "\n".join(insights)


# 使用範例
if __name__ == "__main__":
    # 示例1: 顯示HTGC的特徵重要性
    print(format_feature_importance_display('HTGC', show_in_buy_signal=False))
    print("\n")

    # 示例2: 買入信號簡化版
    print(format_feature_importance_display('HTGC', show_in_buy_signal=True))
    print("\n")

    # 示例3: 獲取前3個重要特徵
    print("Top 3 features for 2330.TW:")
    top_3 = get_top_features('2330.TW', top_n=3)
    for name, score in top_3:
        print(f"  {name}: {score*100:.2f}%")
