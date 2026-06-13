"""
Shared Tavily news retrieval for all get_trading_signal_*.py files.
Usage:
    from tavily_news import print_tavily_news
    print_tavily_news('2317.TW', '鴻海')      # TW stock → Chinese queries + TW domains
    print_tavily_news('NVDA', 'NVIDIA')        # US stock → English queries
"""
import urllib.request
import urllib.error
import json
from datetime import datetime

TAVILY_API_KEY  = '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'
TAVILY_ENDPOINT = 'https://api.tavily.com/search'

_TW_NEWS_DOMAINS = [
    'udn.com', 'money.udn.com', 'moneydj.com',
    'news.cnyes.com', 'cnyes.com', 'stockfeel.com.tw',
    'tw.stock.yahoo.com',
]

_BULLISH = [
    '上漲', '漲', '買進', '買入', '強', '好', '利多', '突破', '成長', '獲利', '增益', '升',
    'buy', 'bullish', 'surge', 'growth', 'profit', 'strong', 'beat', 'positive',
]
_BEARISH = [
    '下跌', '跌', '賣出', '空', '弱', '差', '利空', '跌破', '虧損', '衰退', '降',
    'sell', 'bearish', 'drop', 'loss', 'weak', 'miss', 'negative', 'decline',
]


def _fetch(query: str, max_results: int = 5,
           include_domains: list = None, search_depth: str = 'advanced') -> list:
    try:
        payload = {
            'api_key':        TAVILY_API_KEY,
            'query':          query,
            'search_depth':   search_depth,
            'max_results':    max_results,
            'include_answer': False,
            'include_images': False,
        }
        if include_domains:
            payload['include_domains'] = include_domains
        req = urllib.request.Request(
            TAVILY_ENDPOINT,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode('utf-8')).get('results', [])
    except urllib.error.HTTPError as e:
        body = ''
        try:
            body = e.read().decode('utf-8')[:200]
        except Exception:
            pass
        print(f'  ⚠️  Tavily HTTP {e.code}: {body}')
        return []
    except Exception as e:
        print(f'  ⚠️  Tavily 搜尋失敗: {e}')
        return []


def print_tavily_news(ticker: str, company_name: str, max_results: int = 5) -> float:
    """Fetch and print latest news for a stock; return a simple keyword sentiment score."""
    ticker_upper = ticker.upper()
    is_tw = '.TW' in ticker_upper or '.TWO' in ticker_upper

    if is_tw:
        stock_id = ticker.split('.')[0]
        queries  = [
            f'{company_name} {stock_id} 台股 新聞',
            f'{company_name} 股價 今日',
            f'{stock_id} {company_name} 法人',
        ]
        domains = _TW_NEWS_DOMAINS
    else:
        queries = [
            f'{company_name} {ticker} stock news',
            f'{ticker} stock price today analysis',
        ]
        domains = None

    print(f"\n{'─' * 70}")
    print(f"  📰 {ticker} {company_name} 最新新聞  "
          f"(Tavily · {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'─' * 70}")

    seen_urls: set = set()
    articles: list = []
    for q in queries:
        for art in _fetch(q, max_results=max_results,
                          include_domains=domains, search_depth='advanced'):
            url = art.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                articles.append(art)
        if len(articles) >= max_results:
            break

    if not articles:
        print('  （無搜尋結果）')
        return 0.0

    pos = neg = 0
    for art in articles[:max_results]:
        text    = ((art.get('title') or '') + ' ' + (art.get('content') or '')).lower()
        pos    += sum(1 for w in _BULLISH if w in text)
        neg    += sum(1 for w in _BEARISH if w in text)
        title   = (art.get('title') or '無標題').strip()[:72]
        url     = (art.get('url')   or '').strip()
        pub     = (art.get('published_date') or '')[:10]
        pub_str = f'  [{pub}]' if pub else ''
        print(f'\n  • {title}{pub_str}')
        if url:
            print(f'    🔗 {url}')

    total = pos + neg
    if total == 0:
        score, label = 0.0, '中性 😐'
    elif pos > neg:
        score, label = pos / total, f'偏多 📈 ({pos}正/{neg}負)'
    else:
        score, label = -(neg / total), f'偏空 📉 ({neg}負/{pos}正)'

    print(f'\n  🧠 新聞情緒簡評: {label}  (基於關鍵字統計)')
    print(f"{'─' * 70}")
    return score
