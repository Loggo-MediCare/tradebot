#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立工具：查看任何股票的 FinBERT 新闻分析
用于检查妖股、题材股的市场情绪
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from finbert_sentiment import FinBERTAnalyzer
from finbert_enhanced_scoring import format_sentiment_output

def check_stock_news(symbol):
    """
    查看单个股票的新闻和情绪分析

    Args:
        symbol: 股票代码
    """
    print("="*80)
    print(f"📰 {symbol} 新闻与市场情绪分析")
    print("="*80)

    analyzer = FinBERTAnalyzer()
    result = analyzer.analyze_stock_sentiment(symbol, verbose=True)

    # 添加 score_adjustment 字段（为了兼容 format_sentiment_output）
    score = result['sentiment_score']
    if score > 0.3:
        result['score_adjustment'] = 20
    elif score > 0.15:
        result['score_adjustment'] = 10
    elif score > 0.05:
        result['score_adjustment'] = 5
    elif score < -0.3:
        result['score_adjustment'] = -20
    elif score < -0.15:
        result['score_adjustment'] = -10
    elif score < -0.05:
        result['score_adjustment'] = -5
    else:
        result['score_adjustment'] = 0

    # 格式化输出
    if result['news_count'] > 0:
        print("\n" + format_sentiment_output(result))

        # 显示所有新闻标题
        print(f"\n{'='*80}")
        print(f"📋 完整新闻列表 ({result['news_count']} 则)")
        print("="*80)
        for i, title in enumerate(result['top_news'], 1):
            print(f"{i:2d}. {title}")
    else:
        print("\n⚠️  未找到相关新闻")
        print("   可能原因：")
        print("   1. 该股票为台股，英文新闻较少")
        print("   2. 搜索关键字不匹配")
        print("   3. 近期无重大新闻")

    return result


def check_multiple_stocks(symbols):
    """
    批量查看多个股票的新闻

    Args:
        symbols: 股票代码列表
    """
    print("\n" + "="*80)
    print(f"🔍 批量新闻分析 - {len(symbols)} 只股票")
    print("="*80)

    results = []
    for symbol in symbols:
        print(f"\n{'─'*80}")
        result = check_stock_news(symbol)
        results.append(result)
        print()

    # 汇总报告
    print("\n" + "="*80)
    print("📊 情绪汇总报告")
    print("="*80)

    for result in results:
        symbol = result['symbol']
        sentiment = result['sentiment_label']
        score = result['sentiment_score']
        news_count = result['news_count']

        if news_count > 0:
            if score > 0.15:
                emoji = "📈"
            elif score < -0.15:
                emoji = "📉"
            else:
                emoji = "➖"

            print(f"{emoji} {symbol:10s} | 情绪: {sentiment:6s} | 分数: {score:+.3f} | 新闻: {news_count} 则")
        else:
            print(f"❌ {symbol:10s} | 无新闻数据")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='查看股票新闻与市场情绪')
    parser.add_argument('symbols', nargs='+', help='股票代码（可多个）')

    args = parser.parse_args()

    if len(args.symbols) == 1:
        # 单个股票：详细分析
        check_stock_news(args.symbols[0])
    else:
        # 多个股票：批量分析
        check_multiple_stocks(args.symbols)
