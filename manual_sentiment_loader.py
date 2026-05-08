#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动题材标注加载器
用于整合人工标注的题材新闻到交易信号系统
"""

import json
import os
from datetime import datetime

def load_manual_sentiment(symbol):
    """
    加载手动标注的市场情绪

    Args:
        symbol: 股票代码

    Returns:
        dict or None: 手动标注结果，如果不存在或已过期则返回 None
    """
    json_file = 'manual_sentiment_override.json'

    if not os.path.exists(json_file):
        return None

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if symbol not in data:
            return None

        override = data[symbol]

        # 检查是否过期
        valid_until = override.get('valid_until')
        if valid_until:
            valid_date = datetime.strptime(valid_until, '%Y-%m-%d')
            if datetime.now() > valid_date:
                print(f"   ⚠️  手动标注已过期 (有效期至 {valid_until})")
                return None

        # 返回标注结果
        return {
            'symbol': symbol,
            'sentiment_score': override['sentiment_score'],
            'sentiment_label': override['sentiment_label'],
            'score_adjustment': override['score_adjustment'],
            'news_count': override.get('news_count', 1),
            'top_news': override.get('top_news', []),
            'theme': override.get('theme', ''),
            'notes': override.get('notes', ''),
            'updated_at': override.get('updated_at', ''),
            'risk_warning': override.get('risk_warning', ''),
            'technical_override': override.get('technical_override', {})
        }

    except Exception as e:
        return None


def display_manual_sentiment(sentiment_result):
    """
    显示手动标注的题材分析结果

    Args:
        sentiment_result: load_manual_sentiment() 返回的结果
    """
    if not sentiment_result:
        return

    symbol = sentiment_result['symbol']
    theme = sentiment_result.get('theme', '')
    score = sentiment_result['sentiment_score']
    label = sentiment_result['sentiment_label']
    adjustment = sentiment_result['score_adjustment']
    news = sentiment_result.get('top_news', [])
    notes = sentiment_result.get('notes', '')
    risk = sentiment_result.get('risk_warning', '')
    tech_override = sentiment_result.get('technical_override', {})

    # 选择emoji
    if score > 0.3:
        emoji = "🚀"
    elif score > 0.15:
        emoji = "📈"
    elif score < -0.15:
        emoji = "📉"
    else:
        emoji = "➖"

    print("="*80)
    print("🏷️  人工题材标注 (Manual Theme Analysis)")
    print("="*80)
    print(f"股票代码:     {symbol}")
    print(f"题材:         {emoji} {theme}")
    print(f"情绪分数:     {score:+.3f}")
    print(f"情绪判断:     {label}")
    print(f"评分调整:     {adjustment:+d} 分")

    if news:
        print(f"\n📰 相关新闻/事件:")
        for i, item in enumerate(news, 1):
            print(f"   {i}. {item}")

    if notes:
        print(f"\n💡 分析师备注:")
        print(f"   {notes}")

    if risk:
        print(f"\n⚠️  风险提示:")
        print(f"   {risk}")

    if tech_override:
        ignore_rsi = tech_override.get('ignore_rsi_overbought', False)
        reason = tech_override.get('reason', '')
        if ignore_rsi:
            print(f"\n🔧 技术指标覆盖:")
            print(f"   ✅ 忽略RSI超买警告")
            print(f"   📌 理由: {reason}")


if __name__ == '__main__':
    # 测试
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("="*80)
    print("🧪 手动题材标注测试")
    print("="*80)

    test_symbols = ['6443.TW', '8110.TW', '2344.TW', 'OMER', '2330.TW']

    for symbol in test_symbols:
        print(f"\n{'─'*80}")
        print(f"查询: {symbol}")
        print("─"*80)

        result = load_manual_sentiment(symbol)
        if result:
            display_manual_sentiment(result)
        else:
            print(f"   ℹ️  无手动标注")
