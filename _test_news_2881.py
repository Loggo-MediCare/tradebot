# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import urllib.request, json

TAVILY_API_KEY  = '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'
TAVILY_ENDPOINT = 'https://api.tavily.com/search'

_TW_NEWS_DOMAINS = [
    'udn.com', 'money.udn.com',
    'moneydj.com',
    'news.cnyes.com', 'cnyes.com',
    'stockfeel.com.tw',
    'tw.stock.yahoo.com',
]

queries = [
    '富邦金控 2881 台股 新聞',
    '富邦金 股價 今日',
    '2881 富邦 金控 法人',
]

seen_urls = set()
all_articles = []

for q in queries:
    payload = {
        'api_key': TAVILY_API_KEY,
        'query': q,
        'search_depth': 'advanced',
        'max_results': 5,
        'include_answer': False,
        'include_images': False,
        'include_domains': _TW_NEWS_DOMAINS,
    }
    req = urllib.request.Request(
        TAVILY_ENDPOINT,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        for art in data.get('results', []):
            url = art.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(art)
    if len(all_articles) >= 5:
        break

print(f"\n找到 {len(all_articles)} 則新聞:\n")
for art in all_articles[:5]:
    t   = (art.get('title') or '').strip()[:80]
    url = (art.get('url') or '').strip()
    pub = (art.get('published_date') or '')[:10]
    print(f"  • {t}  [{pub}]")
    print(f"    {url}")
    print()
