"""
Taiwan stock news tracker via Tavily REST API.
Shared module used by all get_trading_signal_XXXX.py files.
"""

import os
import urllib.request
import urllib.error
import json as _json
from datetime import datetime as _dt

# Load TAVILY_API_KEY from .env (not committed to git) if not already set
if 'TAVILY_API_KEY' not in os.environ:
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(_env_path):
        with open(_env_path, encoding='utf-8') as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip())

TAVILY_API_KEY  = os.environ.get('TAVILY_API_KEY', '')
TAVILY_ENDPOINT = 'https://api.tavily.com/search'

_TW_NEWS_DOMAINS = [
    'udn.com', 'money.udn.com',
    'moneydj.com',
    'news.cnyes.com', 'cnyes.com',
    'stockfeel.com.tw',
    'tw.stock.yahoo.com',
    'chinatimes.com',
    'technews.tw',
]

_quota_exceeded = False  # set True on first 432; prevents repeated error spam

BULLISH_WORDS = [
    '上漲','漲','買進','買入','強','好','利多','突破','成長','獲利','增益','升',
    'buy','bullish','surge','growth','profit','strong','beat','positive',
]
BEARISH_WORDS = [
    '下跌','跌','賣出','空','弱','差','利空','跌破','虧損','衰退','降',
    'sell','bearish','drop','loss','weak','miss','negative','decline',
]


def fetch_news_tavily(query: str, max_results: int = 5,
                      include_domains: list = None, search_depth: str = 'advanced') -> list:
    global _quota_exceeded
    if _quota_exceeded:
        return []
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
            data=_json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read().decode('utf-8'))
            return data.get('results', [])

    except urllib.error.HTTPError as e:
        if e.code in (432, 433):
            # 432 = invalid plan / 433 = pay-as-you-go quota exceeded
            _quota_exceeded = True
        else:
            try: body = e.read().decode('utf-8')[:200]
            except Exception: body = ''
            print(f"  Tavily HTTP {e.code}: {body}")
        return []
    except Exception as e:
        print(f"  Tavily error: {e}")
        return []


def print_tavily_news_tw(code: str, name: str, max_results: int = 5) -> float:
    """
    Fetch and print the latest Taiwan stock news for a given code+name.
    Returns a simple keyword-based sentiment score (-1.0 to +1.0).
    """
    queries = [
        f'{name} {code} 台股 新聞',
        f'{name} 股價 今日',
        f'{code} {name} 法人',
    ]

    print(f"\n{'─'*70}")
    print(f"  📰 {code} {name} 最新新聞  (Tavily · {_dt.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'─'*70}")

    seen_urls = set()
    all_articles = []

    for q in queries:
        if _quota_exceeded:
            break
        for art in fetch_news_tavily(q, max_results=max_results,
                                     include_domains=_TW_NEWS_DOMAINS,
                                     search_depth='advanced'):
            url = art.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(art)
        if len(all_articles) >= max_results:
            break

    if _quota_exceeded and not all_articles:
        print("  ⚠️  Tavily 配額已達上限，新聞暫時無法取得")
        print(f"{'─'*70}")
        return 0.0

    if not all_articles:
        print("  （無搜尋結果）")
        return 0.0

    pos = neg = 0
    for art in all_articles[:max_results]:
        title   = (art.get('title')   or '').lower()
        content = (art.get('content') or '').lower()
        text    = title + ' ' + content
        pos += sum(1 for w in BULLISH_WORDS if w in text)
        neg += sum(1 for w in BEARISH_WORDS if w in text)

        t   = (art.get('title') or '無標題').strip()[:72]
        url = (art.get('url')   or '').strip()
        pub = (art.get('published_date') or '')[:10]
        pub_str = f'  [{pub}]' if pub else ''
        print(f"\n  • {t}{pub_str}")
        if url:
            print(f"    🔗 {url}")

    total = pos + neg
    if total == 0:
        sentiment_score = 0.0
        sentiment_label = '中性 😐'
    elif pos > neg:
        sentiment_score = pos / total
        sentiment_label = f'偏多 📈 ({pos}正/{neg}負)'
    else:
        sentiment_score = -(neg / total)
        sentiment_label = f'偏空 📉 ({neg}負/{pos}正)'

    print(f"\n  🧠 新聞情緒簡評: {sentiment_label}  (基於關鍵字統計)")
    print(f"{'─'*70}")
    return sentiment_score
