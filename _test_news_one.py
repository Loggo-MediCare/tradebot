# -*- coding: utf-8 -*-
import sys, io, urllib.request, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TAVILY_API_KEY  = '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'
TAVILY_ENDPOINT = 'https://api.tavily.com/search'

_TW_NEWS_DOMAINS = [
    'udn.com', 'money.udn.com', 'moneydj.com',
    'news.cnyes.com', 'cnyes.com', 'stockfeel.com.tw',
    'tw.stock.yahoo.com',
]

code = sys.argv[1] if len(sys.argv) > 1 else '2356'
name = sys.argv[2] if len(sys.argv) > 2 else '英業達'

queries = [
    f'{name} {code} 台股 新聞',
    f'{name} 股價 今日',
    f'{code} {name} 法人',
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

BULLISH = ['上漲','漲','買進','強','好','利多','突破','成長','獲利','增益','升','漲停','新高']
BEARISH = ['下跌','跌','賣出','空','弱','差','利空','跌破','虧損','衰退','降','虧']

pos = neg = 0
print(f"\n找到 {len(all_articles)} 則新聞:\n")
for art in all_articles[:5]:
    t   = (art.get('title') or '').strip()[:80]
    url = (art.get('url') or '').strip()
    pub = (art.get('published_date') or '')[:10]
    text = t.lower() + (art.get('content') or '').lower()
    pos += sum(1 for w in BULLISH if w in text)
    neg += sum(1 for w in BEARISH if w in text)
    print(f"  • {t}  [{pub}]")
    print(f"    {url}")
    print()

total = pos + neg
if total == 0:   label = '中性 😐'
elif pos > neg:  label = f'偏多 📈 ({pos}正/{neg}負)'
else:            label = f'偏空 📉 ({neg}負/{pos}正)'
print(f"新聞情緒: {label}")
