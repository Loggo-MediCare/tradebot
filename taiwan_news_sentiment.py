#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台湾本地新闻情绪分析模块
整合多个台湾新闻源 + FinBERT 中英文分析
"""

import sys
import io
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 台股新闻源配置
# ==========================================
TAIWAN_NEWS_SOURCES = {
    # 鉅亨网 (cnyes.com) - 台湾最大财经网站
    'cnyes': {
        'name': '鉅亨网',
        'enabled': True,
        'url_template': 'https://news.cnyes.com/news/cat/tw_stock_{stock_code}',
        'encoding': 'utf-8'
    },

    # Google 新闻 (繁体中文)
    'google_news': {
        'name': 'Google新闻',
        'enabled': True,
        'url_template': 'https://news.google.com/rss/search?q={stock_code}+股票&hl=zh-TW&gl=TW&ceid=TW:zh-Hant',
        'encoding': 'utf-8'
    },

    # Yahoo Finance RSS (英文)
    'yahoo_rss': {
        'name': 'Yahoo Finance',
        'enabled': True,
        'url_template': 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US',
        'encoding': 'utf-8'
    }
}

# 台股代码映射（用于新闻搜索）
TAIWAN_STOCK_MAPPING = {
    '6443.TW': {
        'keywords': ['元晶', '6443', 'solar', 'SpaceX', '太阳能'],
        'priority_sources': ['google_news', 'cnyes']
    },
    '8110.TW': {
        'keywords': ['华东', '8110', 'Walton', '华邦电', '记忆体'],
        'priority_sources': ['google_news', 'cnyes']
    },
    '2344.TW': {
        'keywords': ['华邦电', '2344', 'Winbond', 'DRAM'],
        'priority_sources': ['google_news', 'cnyes']
    },
    '2330.TW': {
        'keywords': ['台积电', '2330', 'TSM', 'TSMC'],
        'priority_sources': ['yahoo_rss', 'google_news']
    },
    '2317.TW': {
        'keywords': ['鸿海', '2317', 'Hon Hai', 'Foxconn'],
        'priority_sources': ['yahoo_rss', 'google_news']
    }
}


class TaiwanNewsAnalyzer:
    """台湾新闻情绪分析器"""

    def __init__(self):
        """初始化分析器"""
        self.finbert = None
        self._load_finbert()

    def _load_finbert(self):
        """加载 FinBERT 模型"""
        try:
            from transformers import BertTokenizer, BertForSequenceClassification, pipeline

            print("🤖 正在初始化 FinBERT 模型...")
            finbert_model = BertForSequenceClassification.from_pretrained(
                'yiyanghkust/finbert-tone',
                num_labels=3
            )
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            self.finbert = pipeline(
                "sentiment-analysis",
                model=finbert_model,
                tokenizer=tokenizer,
                device=-1  # CPU
            )
            print("✅ FinBERT 模型载入完成！")
        except Exception as e:
            print(f"⚠️  FinBERT 载入失败: {e}")
            print("   将仅使用关键字匹配分析")
            self.finbert = None

    def fetch_google_news(self, stock_code, keywords):
        """
        从 Google 新闻 RSS 抓取台股新闻

        Args:
            stock_code: 股票代码（如 6443.TW）
            keywords: 搜索关键字列表

        Returns:
            list: 新闻列表 [{'title': str, 'source': str}, ...]
        """
        news_items = []
        headers = {'User-Agent': 'Mozilla/5.0'}

        # 使用第一个中文关键字搜索
        primary_keyword = keywords[0] if keywords else stock_code.replace('.TW', '')

        url = f"https://news.google.com/rss/search?q={primary_keyword}+股票&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)

                for item in root.findall('.//{http://www.w3.org/2005/Atom}entry')[:10]:
                    title_elem = item.find('{http://www.w3.org/2005/Atom}title')

                    if title_elem is not None and title_elem.text:
                        title = title_elem.text

                        # 检查是否包含相关关键字
                        if any(kw in title for kw in keywords):
                            news_items.append({
                                'title': title,
                                'source': 'Google新闻'
                            })

        except Exception as e:
            print(f"   ⚠️  Google新闻抓取失败: {e}")

        return news_items

    def fetch_yahoo_rss(self, symbol):
        """
        从 Yahoo Finance RSS 抓取新闻（英文）

        Args:
            symbol: 股票代码（如 TSM, 2330.TW）

        Returns:
            list: 新闻列表
        """
        news_items = []
        headers = {'User-Agent': 'Mozilla/5.0'}

        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)

                for item in root.findall('.//item')[:10]:
                    title_elem = item.find('title')

                    if title_elem is not None and title_elem.text:
                        news_items.append({
                            'title': title_elem.text,
                            'source': 'Yahoo Finance'
                        })

        except Exception as e:
            print(f"   ⚠️  Yahoo RSS 抓取失败: {e}")

        return news_items

    def analyze_sentiment_finbert(self, text):
        """
        使用 FinBERT 分析情绪

        Args:
            text: 新闻标题

        Returns:
            float: 情绪分数 (-1.0 到 +1.0)
        """
        if not self.finbert or not text:
            return 0.0

        try:
            # FinBERT 只支持英文，中文标题会得到不准确结果
            # 但仍可提供参考
            result = self.finbert(text[:512])[0]
            label = result['label']
            confidence = result['score']

            if label == 'Positive':
                return confidence
            elif label == 'Negative':
                return -confidence
            else:
                return 0.0

        except Exception as e:
            return 0.0

    def analyze_sentiment_keywords(self, text):
        """
        使用关键字匹配分析中文新闻情绪

        Args:
            text: 新闻标题

        Returns:
            float: 情绪分数 (-1.0 到 +1.0)
        """
        # 正面关键字
        positive_keywords = [
            '上涨', '涨停', '突破', '创新高', '大涨', '飙涨', '暴涨',
            '利多', '获利', '营收', '成长', '突破', '订单', '合作',
            '批准', 'FDA', '新药', '专利', '技术', '创新',
            'SpaceX', '马斯克', 'Elon', '特斯拉',
            '供应链', '打入', '受惠', '看好', '升级', '买进'
        ]

        # 负面关键字
        negative_keywords = [
            '下跌', '跌停', '暴跌', '重挫', '崩跌', '破底',
            '利空', '亏损', '衰退', '裁员', '停产', '召回',
            '诉讼', '罚款', '调查', '警示', '风险', '危机',
            '卖出', '看空', '降级', '减持'
        ]

        score = 0.0

        # 计算正面关键字权重
        for keyword in positive_keywords:
            if keyword in text:
                score += 0.15

        # 计算负面关键字权重
        for keyword in negative_keywords:
            if keyword in text:
                score -= 0.15

        # 限制在 -1.0 到 +1.0
        return max(-1.0, min(1.0, score))

    def analyze_stock_news(self, symbol, verbose=True):
        """
        分析台股新闻情绪

        Args:
            symbol: 股票代码（如 6443.TW）
            verbose: 是否显示详细信息

        Returns:
            dict: {
                'symbol': str,
                'news_count': int,
                'sentiment_score': float,
                'sentiment_label': str,
                'top_news': list,
                'sources': dict  # 各来源新闻数量
            }
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"📰 分析 {symbol} 的台湾新闻情绪")
            print(f"{'='*80}")

        all_news = []
        sources_count = {}

        # 1. 获取股票配置
        config = TAIWAN_STOCK_MAPPING.get(symbol, {
            'keywords': [symbol.replace('.TW', '')],
            'priority_sources': ['google_news', 'yahoo_rss']
        })

        keywords = config['keywords']
        priority_sources = config.get('priority_sources', ['google_news'])

        if verbose:
            print(f"搜索关键字: {', '.join(keywords)}")
            print(f"优先新闻源: {', '.join(priority_sources)}\n")

        # 2. 从各个新闻源抓取
        if 'google_news' in priority_sources:
            google_news = self.fetch_google_news(symbol, keywords)
            all_news.extend(google_news)
            sources_count['Google新闻'] = len(google_news)
            if verbose and google_news:
                print(f"✅ Google新闻: {len(google_news)} 则")

        if 'yahoo_rss' in priority_sources:
            yahoo_news = self.fetch_yahoo_rss(symbol)
            all_news.extend(yahoo_news)
            sources_count['Yahoo Finance'] = len(yahoo_news)
            if verbose and yahoo_news:
                print(f"✅ Yahoo Finance: {len(yahoo_news)} 则")

        if not all_news:
            if verbose:
                print("⚠️  未找到相关新闻")

            return {
                'symbol': symbol,
                'news_count': 0,
                'sentiment_score': 0.0,
                'sentiment_label': '无数据',
                'top_news': [],
                'sources': sources_count
            }

        # 3. 分析情绪
        sentiment_scores = []
        for news_item in all_news:
            title = news_item['title']

            # 根据语言选择分析方法
            if any('\u4e00' <= char <= '\u9fff' for char in title):
                # 中文标题 -> 使用关键字匹配
                score = self.analyze_sentiment_keywords(title)
            else:
                # 英文标题 -> 使用 FinBERT
                score = self.analyze_sentiment_finbert(title)

            sentiment_scores.append(score)

        # 4. 计算平均情绪
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        # 5. 情绪标签
        if avg_sentiment > 0.2:
            label = '强烈正面'
            emoji = '🚀'
        elif avg_sentiment > 0.1:
            label = '正面'
            emoji = '📈'
        elif avg_sentiment < -0.2:
            label = '强烈负面'
            emoji = '📉'
        elif avg_sentiment < -0.1:
            label = '负面'
            emoji = '📉'
        else:
            label = '中性'
            emoji = '➖'

        # 6. 获取前3条新闻
        top_news = [item['title'] for item in all_news[:3]]

        if verbose:
            print(f"\n📊 分析结果:")
            print(f"   总新闻数: {len(all_news)}")
            print(f"   情绪分数: {avg_sentiment:+.3f}")
            print(f"   情绪判断: {emoji} {label}")
            print(f"\n📰 热点新闻:")
            for i, title in enumerate(top_news, 1):
                print(f"   {i}. {title[:70]}...")

        return {
            'symbol': symbol,
            'news_count': len(all_news),
            'sentiment_score': avg_sentiment,
            'sentiment_label': label,
            'top_news': top_news,
            'sources': sources_count
        }


def test_taiwan_news():
    """测试台湾新闻分析"""
    print("="*80)
    print("🧪 台湾新闻情绪分析测试")
    print("="*80)

    analyzer = TaiwanNewsAnalyzer()

    # 测试台股列表
    test_symbols = ['6443.TW', '8110.TW', '2344.TW', '2330.TW', '2317.TW']

    results = []
    for symbol in test_symbols:
        result = analyzer.analyze_stock_news(symbol, verbose=True)
        results.append(result)

    # 汇总报告
    print("\n" + "="*80)
    print("📊 台湾市场情绪汇总")
    print("="*80)

    df = pd.DataFrame([
        {
            'Symbol': r['symbol'],
            'News': r['news_count'],
            'Sentiment': r['sentiment_score'],
            'Label': r['sentiment_label']
        }
        for r in results
    ])

    print(df.to_string(index=False))

    # 市场整体情绪
    valid_results = [r for r in results if r['news_count'] > 0]
    if valid_results:
        market_sentiment = sum(r['sentiment_score'] for r in valid_results) / len(valid_results)

        if market_sentiment > 0.1:
            market_label = "🌞 乐观"
        elif market_sentiment < -0.1:
            market_label = "🌧️  悲观"
        else:
            market_label = "☁️  中性"

        print(f"\n📌 台湾市场整体情绪: {market_label} (平均: {market_sentiment:+.3f})")


if __name__ == '__main__':
    # 独立运行时设置 UTF-8 编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    test_taiwan_news()
