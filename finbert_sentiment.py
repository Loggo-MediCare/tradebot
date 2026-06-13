#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT 情绪分析模块
用于抓取新闻并使用 FinBERT 进行情绪分析
"""

import sys
import io
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import warnings
import os
import contextlib

# 不要在模块中设置 sys.stdout，由主程序控制
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# 安全打印函数 - 自动处理 Windows 编码问题
def safe_print(text, **kwargs):
    """安全打印，避免 Windows CP950 编码错误"""
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        # 移除 emoji 和特殊字符，只保留 ASCII
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text, **kwargs)

# ==========================================
# 新闻搜索映射表
# ==========================================
SEARCH_MAPPING = {
    # 美股
    'AAPL': ('AAPL', None),
    'NVDA': ('NVDA', None),
    'GOOGL': ('GOOGL', None),
    'GOOG': ('GOOG', None),
    'MU': ('MU', None),
    'AVGO': ('AVGO', None),
    'OMER': ('OMER', None),
    'ALAB': ('ALAB', None),
    'NAT': ('NAT', None),
    'HTGC': ('HTGC', None),
    'BRK-B': ('BRK-B', None),
    'MDB': ('MDB', None),

    # 欧股
    'RHM.DE': ('RHM.DE', 'Rheinmetall'),

    # 台股 -> 搜索 ADR 或英文名
    '2330.TW': ('TSM', None),          # 台积电 -> TSM ADR
    '2317.TW': ('2317.TW', 'Hon Hai'), # 鸿海 -> Hon Hai
    '2308.TW': ('2308.TW', 'Delta'),   # 台达电
    '3711.TW': ('ASX', None),          # 日月光 -> ASE ADR
    '2337.TW': ('2337.TW', 'Macronix'), # 旺宏
    '6442.TW': ('6442.TW', 'Mega Financial'), # 兆丰金
    '3661.TW': ('3661.TW', 'Alchip'),  # 世芯
    '2360.TW': ('2360.TW', 'Chroma'),  # 致茂
    '2451.TW': ('2451.TW', 'Transcend'), # 创见
    '6269.TW': ('6269.TW', 'Taiwan Paiho'), # 台燿
    '6443.TW': ('6443.TW', 'solar'),  # 元晶 -> 搜索太阳能相关
    '8110.TW': ('8110.TW', 'Walton'),  # 华东
    '2344.TW': ('2344.TW', 'Winbond'), # 华邦电
    '2345.TW': ('2345.TW', 'Accton'),  # 智邦
    '2376.TW': ('2376.TW', 'Gigabyte'), # 技嘉
    '2385.TW': ('2385.TW', 'Chicony'), # 群光
    '2408.TW': ('2408.TW', 'Nanya'),   # 南亚科
    '2449.TW': ('2449.TW', 'Kingmax'), # 京元电子
    '3017.TW': ('3017.TW', 'Qisda'),   # 奇鋐
    '3653.TW': ('3653.TW', 'Chroma'),  # 健策
    '3715.TW': ('3715.TW', 'Epistar'), # 晶電
    '1519.TW': ('1519.TW', 'Taiwan Cement'), # 华城
    '4938.TW': ('4938.TW', 'Chipbond'), # 和鑫
    '5483.TW': ('5483.TW', 'Sino-American Silicon'), # 中美晶
    '6209.TW': ('6209.TW', 'Hiwin'),   # 上银科技
    '6515.TW': ('6515.TW', 'TMYTEK'),  # 穎崴科技
    '6669.TW': ('6669.TW', 'Wiwynn'),  # 緯穎
    '6770.TW': ('6770.TW', 'PSMC'),    # 力积电
    '6781.TW': ('6781.TW', 'AES'),     # 旭隼科技
    '6805.TW': ('6805.TW', 'Fuzetec'), # 富世达
    '7769.TW': ('7769.TW', 'Voltronic'), # 霈方
    '8131.TW': ('8131.TW', 'Weltrend'), # 福懋科
    '8210.TW': ('8210.TW', 'Chroma ATE'), # 勤誠
}

class FinBERTAnalyzer:
    """FinBERT 情绪分析器"""

    def __init__(self):
        """初始化 FinBERT 模型"""
        self.nlp_finbert = None
        self._load_model()

    def _load_model(self):
        """加载 FinBERT 模型"""
        try:
            # 禁用 Hugging Face 的自动转换和优化，加快加载
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

            # 压低 transformers/hf hub 日志等级，避免输出无关加载报告
            try:
                from transformers.utils import logging as tf_logging
                tf_logging.set_verbosity_error()
            except Exception:
                pass
            try:
                from huggingface_hub import logging as hf_logging
                hf_logging.set_verbosity_error()
            except Exception:
                pass
            try:
                from huggingface_hub.utils import disable_progress_bars
                disable_progress_bars()
            except Exception:
                pass
            
            # 某些底层库会直接打印到 stdout/stderr，这里统一静默
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                finbert = BertForSequenceClassification.from_pretrained(
                    'yiyanghkust/finbert-tone',
                    num_labels=3,
                    local_files_only=False,  # 允许下载
                    cache_dir=None  # 使用默认缓存
                )
                tokenizer = BertTokenizer.from_pretrained(
                    'yiyanghkust/finbert-tone',
                    local_files_only=False
                )
            self.nlp_finbert = pipeline(
                "sentiment-analysis",
                model=finbert,
                tokenizer=tokenizer,
                device=-1  # 使用 CPU
            )
        except KeyboardInterrupt:
            safe_print("⚠️  FinBERT 模型加载被中断")
            safe_print("   将跳过情绪分析功能，继续执行交易信号...")
            self.nlp_finbert = None
        except Exception as e:
            safe_print(f"⚠️  FinBERT 载入失败: {e}")
            safe_print("   将跳过情绪分析功能")
            self.nlp_finbert = None

    def fetch_news_via_rss(self, symbol, filter_kw=None, max_news=10):
        """
        从 Yahoo RSS 抓取新闻标题与摘要

        Args:
            symbol: 股票代码
            filter_kw: 过滤关键字
            max_news: 最多抓取新闻数量

        Returns:
            list: 新闻列表 [{'title': str, 'summary': str}, ...]
        """
        headers = {'User-Agent': 'Mozilla/5.0'}
        news_items = []

        # Yahoo Finance RSS URL
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)

                # 解析 XML <item> 标签
                for item in root.findall('.//item')[:max_news]:
                    title_elem = item.find('title')
                    desc_elem = item.find('description')

                    title = title_elem.text if title_elem is not None else ""
                    description = desc_elem.text if desc_elem is not None else ""

                    if title:
                        # 应用过滤关键字
                        if filter_kw:
                            if filter_kw.lower() not in title.lower():
                                continue

                        news_items.append({
                            'title': title,
                            'summary': description
                        })

        except Exception as e:
            safe_print(f"   ⚠️  RSS 抓取失败 ({symbol}): {e}")

        return news_items

    def get_sentiment_score(self, text):
        """
        使用 FinBERT 计算情绪分数

        Args:
            text: 文本内容

        Returns:
            float: 情绪分数 [-1.0 到 +1.0]
        """
        if not self.nlp_finbert or not text:
            return 0.0

        try:
            # FinBERT 限制 512 tokens
            result = self.nlp_finbert(text[:512])[0]
            label = result['label']
            confidence = result['score']

            if label == 'Positive':
                return confidence
            elif label == 'Negative':
                return -confidence
            else:  # Neutral
                return 0.0

        except Exception as e:
            safe_print(f"   ⚠️  情绪分析失败: {e}")
            return 0.0

    def analyze_stock_sentiment(self, symbol, verbose=True):
        """
        分析股票的市场情绪

        Args:
            symbol: 股票代码
            verbose: 是否打印详细信息

        Returns:
            dict: {
                'symbol': str,
                'news_count': int,
                'sentiment_score': float,  # -1.0 到 +1.0
                'sentiment_label': str,    # '正面' / '中性' / '负面'
                'top_news': list           # 前3条新闻标题
            }
        """
        # 获取搜索映射
        mapping = SEARCH_MAPPING.get(symbol, (symbol, None))
        search_term, filter_kw = mapping

        # 靜默前置標頭，避免批次執行時輸出過多噪音

        # 抓取新闻
        news_items = self.fetch_news_via_rss(search_term, filter_kw, max_news=10)

        if not news_items:
            if verbose:
                safe_print("   ⚠️  未找到相关新闻")
            return {
                'symbol': symbol,
                'news_count': 0,
                'sentiment_score': 0.0,
                'sentiment_label': '无数据',
                'top_news': []
            }

        # 分析每条新闻的情绪
        sentiment_scores = []
        for item in news_items:
            title = item['title']
            score = self.get_sentiment_score(title)
            sentiment_scores.append(score)

        # 计算平均情绪分数
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        # 情绪标签
        if avg_sentiment > 0.15:
            label = '正面'
            emoji = '📈'
        elif avg_sentiment < -0.15:
            label = '负面'
            emoji = '📉'
        else:
            label = '中性'
            emoji = '➖'

        # 获取前3条新闻
        top_news = [item['title'] for item in news_items[:3]]

        # Verbose output removed - 已移除冗餘輸出，統一使用 format_sentiment_output
        # if verbose:
        #     safe_print(f"\n📊 分析结果:")
        #     safe_print(f"   新闻数量: {len(news_items)}")
        #     safe_print(f"   情绪分数: {avg_sentiment:+.3f}")
        #     safe_print(f"   情绪判断: {emoji} {label}")
        #     safe_print(f"\n📰 热点新闻:")
        #     for i, title in enumerate(top_news, 1):
        #         safe_print(f"   {i}. {title[:80]}...")

        return {
            'symbol': symbol,
            'news_count': len(news_items),
            'sentiment_score': avg_sentiment,
            'sentiment_label': label,
            'top_news': top_news
        }


def test_finbert():
    """测试 FinBERT 功能"""
    safe_print("="*60)
    safe_print("🧪 FinBERT 情绪分析测试")
    safe_print("="*60)

    analyzer = FinBERTAnalyzer()

    # 测试股票列表
    test_symbols = ['NVDA', 'OMER', 'TSM', '2330.TW', 'RHM.DE']

    results = []
    for symbol in test_symbols:
        result = analyzer.analyze_stock_sentiment(symbol, verbose=True)
        results.append(result)

    # 汇总报告
    safe_print("\n" + "="*60)
    safe_print("📊 情绪分析汇总报告")
    safe_print("="*60)

    df = pd.DataFrame(results)
    safe_print(df[['symbol', 'news_count', 'sentiment_score', 'sentiment_label']])

    # 市场整体情绪
    valid_results = [r for r in results if r['news_count'] > 0]
    if valid_results:
        market_sentiment = sum(r['sentiment_score'] for r in valid_results) / len(valid_results)
        if market_sentiment > 0.1:
            market_label = "🌞 乐观"
        elif market_sentiment < -0.1:
            market_label = "🌧️  悲观"
        else:
            market_label = "☁️  中立"

        safe_print(f"\n📌 市场整体情绪: {market_label} (平均: {market_sentiment:+.3f})")


if __name__ == '__main__':
    # 独立运行时设置 UTF-8 编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    test_finbert()
