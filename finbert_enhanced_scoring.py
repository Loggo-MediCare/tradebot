#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 增强评分模块
将 FinBERT 情绪分析整合到交易信号评分系统中
"""

import sys
import io
import os
import threading
# 不要在模块中设置 sys.stdout，由主程序控制
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_FINBERT_ANALYZER = None
_FINBERT_LOCK = threading.Lock()
_SENTIMENT_CACHE = {}
_GROWTH_CACHE = {}  # ticker → calculate_growth_score_adjustment result
# 0 表示不截斷（顯示完整標題）
NEWS_TITLE_MAX_LEN = int(os.getenv("FINBERT_NEWS_TITLE_MAX_LEN", "0"))
TOP_NEWS_DISPLAY_COUNT = int(os.getenv("FINBERT_TOP_NEWS_COUNT", "3"))


def _get_finbert_analyzer():
    """延迟初始化并复用 FinBERT 分析器 (thread-safe)，避免並行載入觸發 meta-tensor 錯誤。"""
    global _FINBERT_ANALYZER
    if _FINBERT_ANALYZER is None:
        with _FINBERT_LOCK:
            if _FINBERT_ANALYZER is None:
                from finbert_sentiment import FinBERTAnalyzer
                _FINBERT_ANALYZER = FinBERTAnalyzer()
    return _FINBERT_ANALYZER


def calculate_sentiment_score(symbol, verbose=False):
    """
    计算股票的市场情绪分数
    优先使用手动标注，其次使用 FinBERT

    Args:
        symbol: 股票代码
        verbose: 是否显示详细信息

    Returns:
        dict: {
            'sentiment_score': float,  # -1.0 到 +1.0
            'sentiment_label': str,    # '正面' / '中性' / '负面'
            'news_count': int,
            'score_adjustment': int,   # 评分调整值 (-20 到 +20)
            'top_news': list
        }
    """
    # 1. 优先检查手动标注 (可选)
    try:
        from manual_sentiment_loader import load_manual_sentiment
        manual_result = load_manual_sentiment(symbol)

        if manual_result:
            if verbose:
                print(f"   ✅ 使用人工题材标注")
            return manual_result
    except (ImportError, FileNotFoundError, Exception):
        pass  # 手动标注模块不存在，继续使用 FinBERT

    # 2. 使用 FinBERT 分析
    try:
        # 命中缓存直接返回，避免同一脚本里重复抓新闻和重复输出
        if symbol in _SENTIMENT_CACHE:
            return _SENTIMENT_CACHE[symbol]

        analyzer = _get_finbert_analyzer()
        result = analyzer.analyze_stock_sentiment(symbol, verbose=verbose)

        # 将情绪分数转换为评分调整值
        sentiment_score = result['sentiment_score']

        if sentiment_score > 0.3:
            # 强烈正面 -> +20分
            score_adjustment = 20
        elif sentiment_score > 0.15:
            # 正面 -> +10分
            score_adjustment = 10
        elif sentiment_score > 0.05:
            # 轻微正面 -> +5分
            score_adjustment = 5
        elif sentiment_score < -0.3:
            # 强烈负面 -> -20分
            score_adjustment = -20
        elif sentiment_score < -0.15:
            # 负面 -> -10分
            score_adjustment = -10
        elif sentiment_score < -0.05:
            # 轻微负面 -> -5分
            score_adjustment = -5
        else:
            # 中性 -> 0分
            score_adjustment = 0

        result['score_adjustment'] = score_adjustment
        _SENTIMENT_CACHE[symbol] = result
        return result

    except ImportError:
        # FinBERT 未安装或初始化失败
        if verbose:
            print("   ⚠️  FinBERT 模块未安装，跳过情绪分析")

        return {
            'sentiment_score': 0.0,
            'sentiment_label': '未启用',
            'news_count': 0,
            'score_adjustment': 0,
            'top_news': []
        }

    except Exception as e:
        if verbose:
            print(f"   ⚠️  情绪分析失败: {e}")

        return {
            'sentiment_score': 0.0,
            'sentiment_label': '分析失败',
            'news_count': 0,
            'score_adjustment': 0,
            'top_news': []
        }


def format_sentiment_output(sentiment_result):
    """
    格式化情绪分析结果为可读输出
    支持手动标注和 FinBERT 两种来源

    Args:
        sentiment_result: calculate_sentiment_score() 返回的结果

    Returns:
        str: 格式化的输出文本
    """
    if sentiment_result['news_count'] == 0:
        return ""

    score = sentiment_result['sentiment_score']
    label = sentiment_result['sentiment_label']
    adjustment = sentiment_result['score_adjustment']
    news_count = sentiment_result['news_count']

    # 检查是否为手动标注
    is_manual = 'theme' in sentiment_result

    # 选择emoji
    if score > 0.3:
        emoji = "🚀"
    elif score > 0.15:
        emoji = "📈"
    elif score < -0.15:
        emoji = "📉"
    else:
        emoji = "➖"

    output = []
    output.append("="*80)

    if is_manual:
        output.append("🏷️  市场情绪分析 (人工题材标注)")
        output.append("="*80)
        output.append(f"题材:         {emoji} {sentiment_result.get('theme', '')}")
    else:
        output.append("🗞️  市场情绪分析 (FinBERT)")
        output.append("="*80)

    # output.append(f"新闻数量:     {news_count} 则")  # 已移除：冗餘資訊
    output.append(f"情绪分数:     {score:+.3f}")
    output.append(f"情绪判断:     {label}")
    output.append(f"评分调整:     {adjustment:+d} 分")

    # 显示热点新闻
    top_news = sentiment_result.get('top_news', [])
    if top_news:
        output.append("\n📰 热点新闻:")
        top_n = max(1, TOP_NEWS_DISPLAY_COUNT)
        for i, title in enumerate(top_news[:top_n], 1):
            if NEWS_TITLE_MAX_LEN > 0 and len(title) > NEWS_TITLE_MAX_LEN:
                short_title = title[:NEWS_TITLE_MAX_LEN] + "..."
            else:
                short_title = title
            output.append(f"   {i}. {short_title}")

    # 手动标注的额外信息
    if is_manual:
        notes = sentiment_result.get('notes', '')
        risk = sentiment_result.get('risk_warning', '')
        tech_override = sentiment_result.get('technical_override', {})

        if notes:
            output.append(f"\n💡 分析师备注:")
            output.append(f"   {notes}")

        if risk:
            output.append(f"\n⚠️  风险提示:")
            output.append(f"   {risk}")

        if tech_override.get('ignore_rsi_overbought'):
            output.append(f"\n🔧 技术指标覆盖:")
            output.append(f"   ✅ 忽略RSI超买警告")
            output.append(f"   📌 {tech_override.get('reason', '')}")

    return "\n".join(output)


def calculate_enhanced_buy_score_with_sentiment(
    rsi, macd, macd_signal, sma_10, sma_30,
    current_price, bb_upper, bb_lower,
    volume_ratio, ai_action, buy_weights,
    symbol, price_change_pct=None, rsi_prev=None, candle_direction=None,
    rsi_low_threshold=16, rsi_oversold_threshold=16
):
    """
    增强版买入评分系统 + FinBERT 情绪分析

    Args:
        (原有参数保持不变)
        symbol: 股票代码 (用于情绪分析)

    Returns:
        tuple: (buy_score, signal_override, reasons, warnings, metadata, sentiment_result)
    """
    # 1. 先获取 FinBERT 情绪分析 (需要传给enhanced_buy_score判断题材股)
    # 这里默认静默，外层脚本通常已打印过一次情绪分析结果
    sentiment_result = calculate_sentiment_score(symbol, verbose=False)

    # 2. 调用增强评分系统 (传入sentiment_score用于题材股识别)
    from enhanced_scoring_module import calculate_enhanced_buy_score

    buy_score, signal_override, reasons, warnings, metadata = calculate_enhanced_buy_score(
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        sma_10=sma_10,
        sma_30=sma_30,
        current_price=current_price,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        volume_ratio=volume_ratio,
        ai_action=ai_action,
        buy_weights=buy_weights,
        sentiment_score=sentiment_result['sentiment_score'],
        price_change_pct=price_change_pct,
        rsi_prev=rsi_prev,
        candle_direction=candle_direction,
        rsi_low_threshold=rsi_low_threshold,
        rsi_oversold_threshold=rsi_oversold_threshold
    )

    # 3. 调整评分 (仅添加情绪分数，流动性加成已在enhanced_scoring_module中处理)
    original_score = buy_score
    buy_score += sentiment_result['score_adjustment']
    buy_score = max(0, min(100, buy_score))  # 限制在 0-100

    # 4. 添加情绪相关的理由和警告
    if sentiment_result['score_adjustment'] > 0:
        reasons.append(f"📰 市场情绪{sentiment_result['sentiment_label']} ({sentiment_result['score_adjustment']:+d}分)")
    elif sentiment_result['score_adjustment'] < 0:
        warnings.append(f"📰 市场情绪{sentiment_result['sentiment_label']} ({sentiment_result['score_adjustment']:+d}分)")

    # 5. 成長因子加分 (Revenue Growth > 33% → +8; EPS Growth > 100% → +8)
    global _GROWTH_CACHE
    if symbol not in _GROWTH_CACHE:
        try:
            import yfinance as _yf
            from shared_market_checks import calculate_growth_score_adjustment
            _GROWTH_CACHE[symbol] = calculate_growth_score_adjustment(_yf, symbol)
        except Exception:
            _GROWTH_CACHE[symbol] = {"adjustment": 0, "reasons": []}
    growth = _GROWTH_CACHE[symbol]
    if growth['adjustment'] > 0:
        buy_score += growth['adjustment']
        buy_score = max(0, min(100, buy_score))
        for r in growth['reasons']:
            reasons.append(f"🌱 {r}")

    # 6. 更新元数据
    metadata['original_score'] = original_score
    metadata['sentiment_adjustment'] = sentiment_result['score_adjustment']
    metadata['growth_adjustment'] = growth['adjustment']
    metadata['final_score'] = buy_score

    return buy_score, signal_override, reasons, warnings, metadata, sentiment_result


if __name__ == '__main__':
    # 独立运行时设置 UTF-8 编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 测试
    print("="*60)
    print("🧪 FinBERT 增强评分模块测试")
    print("="*60)

    test_symbols = ['NVDA', 'OMER', 'TSM']

    for symbol in test_symbols:
        print(f"\n测试股票: {symbol}")
        result = calculate_sentiment_score(symbol, verbose=True)
        print(format_sentiment_output(result))
