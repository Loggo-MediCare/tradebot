"""
傻瓜儀表板 — Fool's Investment Dashboard
==========================================
每天只看三件事：

  🟢 哪些股票今天多頭結構完整？  → 持有 / 可買
  📍 哪些股票今天 MACD 收腳+跳空？→ 最佳進場時機
  🔴 哪些股票週/月線開始轉負？   → 準備出場

Usage:
  python fool_dashboard.py           # 全部掃描
  python fool_dashboard.py --tw      # 只掃台股
  python fool_dashboard.py --us      # 只掃美股
"""
import sys, io, os, re, json, warnings, logging, argparse, time, random, threading
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ResourceWarning)
logging.getLogger('yfinance').setLevel(logging.ERROR)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── yfinance acceleration & stability (cache + retry + rate-limit) ───────────
CACHE_DIR = os.path.join(os.getcwd(), '.cache_yfinance')
os.makedirs(CACHE_DIR, exist_ok=True)

# Company name cache (for printing: "2330.TW 台積電")
NAME_CACHE_DIR = os.path.join(os.getcwd(), '.cache_yfinance_names')
os.makedirs(NAME_CACHE_DIR, exist_ok=True)

# Optional local alias file (you can put Chinese names here)
# Example ticker_aliases.json:
# {
#   "2330.TW": "台積電",
#   "MU": "Micron"
# }
ALIASES_FILE = os.path.join(os.getcwd(), 'ticker_aliases.json')

# Defaults (can be overridden by CLI args)
USE_CACHE = True
CACHE_TTL_HOURS = 8
NAME_CACHE_TTL_DAYS = 30
MAX_RETRIES = 3
MAX_CONCURRENT_DOWNLOADS = 2
MAX_CONCURRENT_INFO = 1
DOWNLOAD_TIMEOUT = 20  # seconds per request

# Minimum MACD-line magnitude (as fraction of price) to count as a real break.
# Values inside ±0.3% of price are treated as noise / borderline.
MACD_LINE_MIN_PCT = 0.003

# Concurrency limiters (reduce Yahoo throttling)
DOWNLOAD_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_DOWNLOADS)
INFO_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_INFO)

# ── Tavily 新聞搜尋設定 ──────────────────────────────────────────────────────
TAVILY_API_KEY   = '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'
TAVILY_ENDPOINT  = 'https://api.tavily.com/search'
NEWS_ENABLED     = False   # 透過 --news 旗標開啟

# In-memory caches
_TICKER_ALIASES = {}
_NAME_MEMO = {}

# Name display controls
# - 'mixed': alias -> cache -> yahoo
# - 'yahoo': cache -> yahoo (ignore aliases)
# - 'alias': alias only
NAME_SOURCE = 'mixed'
NAME_MAXLEN = 10


def _safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))


def _cache_path(ticker: str, period: str, auto_adjust: bool = True) -> str:
    tag = f"{_safe_name(ticker)}__{_safe_name(period)}__adj{int(auto_adjust)}"
    return os.path.join(CACHE_DIR, f"{tag}.pkl")


def _is_cache_fresh(path: str, ttl_hours: float) -> bool:
    try:
        age = time.time() - os.path.getmtime(path)
        return age <= float(ttl_hours) * 3600
    except Exception:
        return False


def _name_cache_path(ticker: str) -> str:
    return os.path.join(NAME_CACHE_DIR, f"{_safe_name(ticker)}.json")


def _load_aliases_once():
    global _TICKER_ALIASES
    if _TICKER_ALIASES:
        return

    # Built-ins (edit/add freely in ticker_aliases.json)
    built_in = {
        # ── 台股持倉 ──────────────────────────────────────────────────
        '2330.TW': '台積電',
        '2317.TW': '鴻海',
        '2454.TW': '聯發科',
        '2344.TW': '華邦電',
        '2308.TW': '台達電',
        '2345.TW': '智邦',
        '2337.TW': '旺宏',
        '2881.TW': '富邦金',
        '2891.TW': '中信金',
        '3037.TW': '欣興',
        '3443.TW': '創意',
        '3711.TW': '日月光投控',
        '6442.TW': '光聖',
        '6531.TW': '愛普',
        '8046.TW': '南電',
        # ── 美股持倉 ──────────────────────────────────────────────────
        'NVDA': 'NVIDIA',
        'MU':   'Micron',
        'SNDK': 'SanDisk',
        'LITE': 'Lumentum',
        'GOOGL':'Google A',
        'GOOG': 'Google C',
        'AMZN': 'Amazon',
        'TSM':  'TSMC',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
    }
    # normalize built-ins to uppercase keys
    for k, v in built_in.items():
        _TICKER_ALIASES[str(k).upper()] = str(v)

    if os.path.exists(ALIASES_FILE):
        try:
            with open(ALIASES_FILE, 'r', encoding='utf-8') as f:
                user_aliases = json.load(f)
            if isinstance(user_aliases, dict):
                for k, v in user_aliases.items():
                    if k and v:
                        _TICKER_ALIASES[str(k).upper()] = str(v)
        except Exception:
            pass


def get_company_name(ticker: str, max_retries: int = 2) -> str:
    """Resolve company name for display, with cache + retry.

    NAME_SOURCE behavior:
      - mixed: alias -> memo/disk -> yahoo
      - yahoo: memo/disk -> yahoo (ignore aliases)
      - alias: alias only

    NOTE: yfinance name lookup uses an extra endpoint. We intentionally call
    this only for items that are printed (small subset), not for all tickers.
    """
    if not ticker:
        return ''

    tkey = str(ticker).upper()

    # 1) aliases (optional)
    if NAME_SOURCE in ('mixed', 'alias'):
        _load_aliases_once()
        if tkey in _TICKER_ALIASES:
            return _TICKER_ALIASES[tkey]
        if NAME_SOURCE == 'alias':
            return ''

    # 2) memory
    if tkey in _NAME_MEMO:
        return _NAME_MEMO[tkey]

    # 3) disk cache
    cache_file = _name_cache_path(tkey)
    if USE_CACHE and os.path.exists(cache_file):
        try:
            age = time.time() - os.path.getmtime(cache_file)
            if age <= float(NAME_CACHE_TTL_DAYS) * 86400:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                name = str(data.get('name') or '').strip()
                if name:
                    _NAME_MEMO[tkey] = name
                    return name
        except Exception:
            pass

    # 4) yfinance fetch (rate-limited)
    name = ''
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            with INFO_SEMAPHORE:
                yt = yf.Ticker(ticker)
                info = None
                try:
                    info = yt.get_info()
                except Exception:
                    info = getattr(yt, 'info', None)

                if not isinstance(info, dict):
                    raise RuntimeError('No info dict')

                # Prefer English-like fields from Yahoo
                name = (info.get('shortName') or info.get('longName') or info.get('displayName') or '').strip()
                time.sleep(0.15 + random.random() * 0.25)

            if name:
                break
            raise RuntimeError('Empty name')

        except Exception:
            backoff = min(10.0, (0.6 * (2 ** (attempt - 1))) + random.random())
            time.sleep(backoff)

    name = name or ''
    _NAME_MEMO[tkey] = name

    if USE_CACHE and name:
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'ticker': ticker, 'name': name, 'ts': int(time.time())}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return name


def download_with_cache(ticker: str, period: str, ttl_hours: float, max_retries: int):
    """Download via yfinance with:
    - Local pickle cache (TTL)
    - Retry with exponential backoff + jitter
    - Concurrency limit via semaphore (avoid yfinance throttling)

    Returns a DataFrame (may be empty on failure).
    """
    cache_file = _cache_path(ticker, period, auto_adjust=True)

    if USE_CACHE and _is_cache_fresh(cache_file, ttl_hours):
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            # Corrupted cache -> remove and re-download
            try:
                os.remove(cache_file)
            except Exception:
                pass

    last_exc = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            # Limit concurrent downloads to reduce rate-limit / connection issues
            with DOWNLOAD_SEMAPHORE:
                df = yf.download(
                    ticker,
                    period=period,
                    interval='1d',
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                    timeout=int(DOWNLOAD_TIMEOUT),
                )

                # Fallback path: sometimes download() returns empty under throttling.
                # history() can succeed when download() fails.
                if df is None or df.empty:
                    try:
                        df = yf.Ticker(ticker).history(
                            period=period,
                            interval='1d',
                            auto_adjust=True,
                        )
                    except Exception:
                        pass
                # Small pacing even on success
                time.sleep(0.15 + random.random() * 0.25)

            if df is None or df.empty:
                raise RuntimeError('Empty data from yfinance (download/history)')

            if USE_CACHE:
                try:
                    df.to_pickle(cache_file)
                except Exception:
                    pass
            return df

        except Exception as e:
            last_exc = e
            # Backoff: 0.8s, 1.6s, 3.2s ... + jitter
            backoff = min(30.0, (1.2 * (2 ** (attempt - 1))) + random.random())
            time.sleep(backoff)

    # final fallback
    return pd.DataFrame()

# ── Tavily 新聞搜尋 ────────────────────────────────────────────────────────────
def fetch_news_tavily(query: str, max_results: int = 5,
                      api_key: str = TAVILY_API_KEY) -> list:
    """
    呼叫 Tavily REST API 搜尋最新新聞。
    回傳 list of dict: {'title', 'url', 'content', 'published_date'}
    """
    import urllib.request
    import urllib.error

    try:
        payload = json.dumps({
            'api_key':        api_key,
            'query':          query,
            'search_depth':   'basic',
            'max_results':    max_results,
            'include_answer': False,
            'include_images': False,
            'topic':          'news',
        }).encode('utf-8')

        req = urllib.request.Request(
            TAVILY_ENDPOINT,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('results', [])

    except urllib.error.HTTPError as e:
        body = ''
        try: body = e.read().decode('utf-8')[:200]
        except Exception: pass
        print(f"  ⚠️  Tavily HTTP {e.code}: {body}")
        return []
    except Exception as e:
        print(f"  ⚠️  Tavily 搜尋失敗: {e}")
        return []


def _fmt_news_item(a: dict, indent: str = '    ', title_max: int = 70) -> str:
    """格式化單則新聞為可讀字串。"""
    title   = (a.get('title') or '無標題').strip()[:title_max]
    url     = (a.get('url')   or '').strip()
    pub     = (a.get('published_date') or '')[:10]
    pub_str = f'  [{pub}]' if pub else ''
    lines   = [f'{indent}• {title}{pub_str}']
    if url:
        lines.append(f'{indent}  🔗 {url}')
    return '\n'.join(lines)


def print_news_section(results: list, label: str = '台股',
                       market_top: int = 5, stock_top_n: int = 4,
                       stock_news_n: int = 3):
    """
    搜尋並顯示台股最新新聞。
    market_top  : 大盤新聞條數
    stock_top_n : 最多顯示幾支個股新聞
    stock_news_n: 每支個股顯示幾則新聞
    """
    today_str = datetime.now().strftime('%Y年%m月')
    print(f"\n{'═'*65}")
    print(f"  📰 最新台股新聞  (Tavily 即時搜尋 · {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'═'*65}")

    # ── 1. 大盤 / 市場整體新聞 ─────────────────────────────────────────────
    mkt_query = f'台股 加權指數 大盤 今日行情 {today_str}'
    print(f"\n  🌐 大盤動態  (查詢: {mkt_query})")
    print(f"{'─'*65}")
    mkt_news = fetch_news_tavily(mkt_query, max_results=market_top)
    if mkt_news:
        for a in mkt_news[:market_top]:
            print(_fmt_news_item(a))
    else:
        print('    （無搜尋結果）')

    # ── 2. 出場警示個股新聞（最需注意）─────────────────────────────────────
    exit_stocks = [r for r in results if r.get('exit_warn')]
    if exit_stocks and stock_top_n > 0:
        n = min(len(exit_stocks), stock_top_n)
        print(f"\n  🔴 出場警示個股新聞  (前 {n} 支)")
        print(f"{'─'*65}")
        for r in exit_stocks[:n]:
            ticker = r['ticker']
            code   = ticker.split('.')[0]
            nm     = get_company_name(ticker, max_retries=1)
            sym    = f'{code} {nm}'.strip() if nm else code
            q      = f'{code} {nm} 股票 新聞' if nm else f'{code} 台股 新聞'
            articles = fetch_news_tavily(q, max_results=stock_news_n)
            print(f'\n  🔴 {sym}  ({r.get("exit_reason","")})  ── ${r["price"]:.2f}')
            if articles:
                for a in articles[:stock_news_n]:
                    print(_fmt_news_item(a))
            else:
                print('    （無搜尋結果）')

    # ── 3. MACD 收腳進場訊號個股新聞 ─────────────────────────────────────
    foot_stocks = [r for r in results if r.get('macd_foot')]
    if foot_stocks and stock_top_n > 0:
        n = min(len(foot_stocks), stock_top_n)
        print(f"\n  📍 進場訊號個股新聞  (前 {n} 支)")
        print(f"{'─'*65}")
        for r in foot_stocks[:n]:
            ticker = r['ticker']
            code   = ticker.split('.')[0]
            nm     = get_company_name(ticker, max_retries=1)
            sym    = f'{code} {nm}'.strip() if nm else code
            icon   = '✅ 跳空+收腳' if r.get('gap_up') else '🟡 收腳'
            q      = f'{code} {nm} 股票 新聞' if nm else f'{code} 台股 新聞'
            articles = fetch_news_tavily(q, max_results=stock_news_n)
            print(f'\n  {icon}  {sym}  縮短:{r["shrink"]:.0f}%  ── ${r["price"]:.2f}')
            if articles:
                for a in articles[:stock_news_n]:
                    print(_fmt_news_item(a))
            else:
                print('    （無搜尋結果）')

    # ── 4. 指定關鍵字額外搜尋（可擴充）──────────────────────────────────────
    # 例: 半導體、AI、台積電
    extra_queries = [
        f'Taiwan semiconductor AI chip supply chain {today_str}',
        f'TSMC Taiwan chip stocks Nvidia {today_str}',
    ]
    print(f"\n  🔍 產業熱點新聞")
    print(f"{'─'*65}")
    for q in extra_queries:
        print(f"\n  查詢: {q}")
        arts = fetch_news_tavily(q, max_results=3)
        if arts:
            for a in arts[:3]:
                print(_fmt_news_item(a))
        else:
            print('    （無搜尋結果）')

    print(f"\n{'═'*65}\n")


# ── Ticker lists ──────────────────────────────────────────────────────────────
_TWO = {'3498','3615','4533','4577','4768','4908','4991','5011','6134','6187',
        '6220','6530','6877','7805','8086','8908','8917','8927','6274','1785',
        '4749','3131','6683','3363','3081','6510','8069','6223','5483','6163',
        '7709','7717','3260','3491','5371','3105','4971','8064','3163','3455',
        '3680','4772','6788','7703','8147','8071','8027','5351','7734','7751',
        '6138','1569','1595','4951','6234','6488','6207','3624','8455','8291',
        '3577','3236','3691','6204','6432','3609','3450','3581','3265',
        '5289','3587','3264','3663','6538','3580','8044','8299','3209','6147'}
_T = {'3449'}

def _t(code):
    if code in _T:   return f"{code}.T"
    if code in _TWO: return f"{code}.TWO"
    return f"{code}.TW"

TW_CODES = [
    '2330','2317','6515','2408','2308','2313','2454','2485','2337','2344',
    '2367','3481','2603','6770','3665','3017','3711','3037','2327','2382',
    '3443','2383','6442','3661','6669','6683','3231','2303','2368','2345',
    '1303','2360','2449','6443','4989','6285','3715','3563','3653','2891','2881',
    '6239','3533','8069','6223','3363','3449','5483','6163','7709','7717',
    '3260','3491','5371','3105','4971','6187','3615','4577','4768','4991',
    '6220','6877','8927','1519','6805','6789','8021','3006','6830','2357',
    '3030','2409','2376','8210','6446','1326','8046','1605','1301','2059',
    '6781','2884','6271','2002','6526','3138','8150','1101','2890','3044',
    '4967','2451','8110','2385','4938','3576','2634','1514','4722','6472',
    '8131','6230','2363','6209','3135','6269','8438','4564','4540','8499',
    '6477','3004','4746','8222','3022','6668','2314','1314','8908','9931',
    '8917','6505','9918','2412','6274','8112','2049','1785','6531','2395',
    '4749','3131','3081','6510','3535','8064','3163','3455','2426','3583',
    '8028','3680','4772','6788','7703','8147','2404','6196','6605','6139',
    '8071','1560','6438','6449','8027','5351','4720','6176','3380','6672',
    '6213','7734','7751','2486','6138','8103','1569','1595','6108','4951',
    '1727','6234','6488','6207','6937','3189','6147','3624','8455','6924',
    '3577','8374','2359','3236','6204','3024','6432','3609','8299','3581',
    '3265','3714','2340','1773','5215','3587','3691','3264','6257','3055',
    '5289','7610','7788','2481','3023','3663','6538','3580','2355','8044',
    '00981a'
]
TW_TICKERS = [_t(c) for c in TW_CODES]

US_TICKERS = [
    'NVDA','AMD','MU','SNDK','INTC','AMAT','ASML','KLAC','LRCX','TER',
    'QCOM','TSM','PLTR','ARM','MRVL','NXPI','SNPS','MPWR','TXN','BKR',
    'URI','ON','GEV','STLD','SMCI','AVGO','META','MSFT','AAPL','AMZN',
    'GOOGL','GOOG','TSLA','VRT','CRDO','ALAB','DELL',
    # ── 持倉額外補充 ──
    'LITE',
    # ── 新增股票 ──
    'JPM','SPY','XLE','BRK-A','BRK-B','MCD','LIN','RDDT','KO',
    'DIS','CRM','LULU','XOP','SMH','SHOP','COIN','RSP',
]


# ── Core analysis per ticker ──────────────────────────────────────────────────
def analyse(ticker, period='1y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, return_reason: bool = False):
    try:
        raw = download_with_cache(ticker, period=period, ttl_hours=ttl_hours, max_retries=max_retries)
        if raw.empty:
            return (None, 'empty data') if return_reason else None
        if len(raw) < 60:
            return (None, f'not enough rows (<60): {len(raw)}') if return_reason else None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        df = raw.rename(columns={'Close':'c','Open':'o','High':'h','Low':'l','Volume':'v'})
        df.index = pd.to_datetime(df.index)

        # yfinance can occasionally return duplicated columns (e.g., Close + Adj Close)
        # which makes df['c'] become a DataFrame. Normalize them to Series to avoid
        # "TypeError: cannot convert the series to <class 'float'>".
        def _series(col: str) -> pd.Series:
            x = df[col]
            if isinstance(x, pd.DataFrame):
                x = x.iloc[:, 0]
            return x

        c = _series('c')
        o = _series('o') if 'o' in df.columns else None
        h = _series('h') if 'h' in df.columns else None

        # ── Daily MACD ───────────────────────────────────────────────────────
        def macd_line(s):
            return (s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean())
        def macd_hist(s):
            ml = macd_line(s); sig = ml.ewm(span=9,adjust=False).mean()
            return ml, sig, ml - sig

        ml_d, sig_d, mh_d = macd_hist(c)
        d_macd = float(ml_d.iloc[-1])
        d_hist = float(mh_d.iloc[-1])
        d_hist_prev = float(mh_d.iloc[-2]) if len(mh_d) >= 2 else 0

        # ── Weekly MACD ──────────────────────────────────────────────────────
        cw = c.resample('W-FRI').last().dropna()
        if len(cw) >= 35:
            ml_w, sig_w, mh_w = macd_hist(cw)
            w_macd = float(ml_w.iloc[-1])
            w_macd_prev = float(ml_w.iloc[-2]) if len(ml_w) >= 2 else np.nan
            w_hist = float(mh_w.iloc[-1])
            w_hist_prev = float(mh_w.iloc[-2]) if len(mh_w) >= 2 else np.nan
        else:
            # Not enough weekly history -> treat as unknown, not bullish by default
            w_macd = np.nan
            w_macd_prev = np.nan
            w_hist = np.nan
            w_hist_prev = np.nan

        # ── Monthly MACD ─────────────────────────────────────────────────────
        cm = c.resample('ME').last().dropna()
        if len(cm) >= 15:
            ml_m, sig_m, mh_m = macd_hist(cm)
            m_macd = float(ml_m.iloc[-1])
            m_hist = float(mh_m.iloc[-1])
        else:
            # Not enough monthly history -> treat as unknown
            m_macd = np.nan
            m_hist = np.nan

        # ── Status ───────────────────────────────────────────────────────────
        d_pos = d_macd > 0
        w_pos = (not np.isnan(w_macd)) and (w_macd > 0)
        m_pos = (not np.isnan(m_macd)) and (m_macd > 0)
        d_neg_hist = d_hist < 0

        # If weekly/monthly data is insufficient, don't incorrectly label as bull
        if np.isnan(w_macd) or np.isnan(m_macd):
            status = '⚪ 資料不足'
        elif d_neg_hist and d_pos and w_pos and m_pos:
            status = '✅ 強勢整理'
        elif not d_neg_hist and d_pos and w_pos and m_pos:
            status = '🟢 完美多頭'
        elif d_pos and w_pos and not m_pos:
            status = '⚠️  月線轉負'
        elif d_pos and not w_pos:
            status = '⚠️  週線轉負'
        elif not d_pos:
            status = '❌ 日線轉負'
        else:
            status = '⚪ 觀察'

        # ── Pre-compute confirmation inputs ──────────────────────────────────
        v_series = _series('v') if 'v' in df.columns else None
        vol_ratio = 1.0
        if v_series is not None and len(v_series) >= 20:
            vol_ma20 = float(v_series.rolling(20).mean().iloc[-1])
            if vol_ma20 > 0:
                vol_ratio = float(v_series.iloc[-1]) / vol_ma20

        sma10 = float(c.rolling(10).mean().iloc[-1])
        sma20 = float(c.rolling(20).mean().iloc[-1])
        _last_close = float(c.dropna().iloc[-1]) if len(c.dropna()) else np.nan
        price_above_ma = (not np.isnan(_last_close)) and _last_close > max(sma10, sma20)

        d_hist_2 = float(mh_d.iloc[-3]) if len(mh_d) >= 3 else np.nan

        # ── MACD 收腳 detection ──────────────────────────────────────────────
        macd_foot = False; gap_up = False; shrink = 0.0
        foot_confirmed = False; foot_score = 0; foot_tags = []
        if d_hist < 0 and d_hist_prev < 0 and abs(d_hist) < abs(d_hist_prev):
            shrink = (abs(d_hist_prev) - abs(d_hist)) / (abs(d_hist_prev) + 1e-10) * 100
            if shrink >= 10:
                macd_foot = True
                # gap up: today open > yesterday high
                if len(df) >= 2:
                    if o is not None and h is not None:
                        gap_up = float(o.iloc[-1]) > float(h.iloc[-2])

                # ── Three-condition confirmation ──────────────────────────
                # A. Price: 站回均線帶 or 跳空
                cond_A = gap_up or price_above_ma
                # B. Momentum: meaningful multi-day shrinkage (core condition)
                multi_shrink = (not np.isnan(d_hist_2)
                                and d_hist_2 < 0
                                and d_hist_prev < d_hist_2
                                and d_hist < d_hist_prev)
                cond_B = shrink >= 20 or multi_shrink
                # C. Volume: breakout needs volume surge
                cond_C = vol_ratio >= 1.3

                if cond_A: foot_tags.append('A價格✅')
                if cond_B: foot_tags.append('B動能✅')
                if cond_C: foot_tags.append(f'C量能✅({vol_ratio:.1f}x)')
                foot_score = sum([cond_A, cond_B, cond_C])
                # Confirmed: at least 2 conditions met, and B (momentum) must be one
                foot_confirmed = foot_score >= 2 and cond_B

        # price must be known before exit checks (needed for MACD_LINE_MIN_PCT normalisation)
        c_valid = c.dropna()
        price = float(c_valid.iloc[-1]) if len(c_valid) else np.nan

        # ── Warning: weekly/monthly turning negative ──────────────────────────
        exit_warn = False
        exit_reasons = []

        # Weekly histogram flips from >0 to <0
        if (not np.isnan(w_hist_prev)) and (not np.isnan(w_hist)):
            if w_hist_prev > 0 and w_hist < 0:
                exit_warn = True
                exit_reasons.append('週線柱狀體翻黑')

        # Weekly MACD below 0 — fire only when magnitude is meaningful OR just crossed
        w_just_crossed = (not np.isnan(w_macd_prev)) and w_macd_prev > 0 and w_macd <= 0
        if not np.isnan(w_macd) and w_macd <= 0 and price > 0:
            w_significant = abs(w_macd) / price > MACD_LINE_MIN_PCT
            if w_significant or w_just_crossed:
                exit_warn = True
                exit_reasons.append('週線MACD跌破0')

        # Monthly MACD below 0 — magnitude gate only (monthly moves slowly, no single-week flip)
        if not np.isnan(m_macd) and m_macd <= 0 and price > 0:
            if abs(m_macd) / price > MACD_LINE_MIN_PCT:
                exit_warn = True
                exit_reasons.append('月線MACD跌破0')

        exit_reason = '+'.join(exit_reasons)

        # Position reduction range — reuses w_just_crossed computed above
        w_borderline = (not np.isnan(w_macd) and price > 0
                        and abs(w_macd) / price < MACD_LINE_MIN_PCT)

        if '月線MACD跌破0' in exit_reasons and '週線MACD跌破0' in exit_reasons:
            reduce_pct = '減 50%~80%'       # 週+月雙空：中期趨勢反轉
        elif '週線MACD跌破0' in exit_reasons:
            if w_borderline and w_just_crossed:
                reduce_pct = '減 0~20% ⚠️ 邊緣觀察'   # 剛跨零且數值近0，容易翻正
            elif w_just_crossed:
                reduce_pct = '減 15%~25% 首次跌破'      # 首次跌破但幅度有意義
            else:
                reduce_pct = '減 30%~60%'               # 已確認在0軸以下多週
        elif '週線柱狀體翻黑' in exit_reasons:
            reduce_pct = '減 20%~35%'       # 僅週柱翻黑：風險控管
        else:
            reduce_pct = ''
        result = {
            'ticker':    ticker,
            'price':     price,
            'status':    status,
            'bull':      d_pos and w_pos and m_pos,
            'perfect':   not d_neg_hist and d_pos and w_pos and m_pos,
            'consol':    d_neg_hist and d_pos and w_pos and m_pos,
            'macd_foot':      macd_foot,
            'foot_confirmed': foot_confirmed,
            'foot_score':     foot_score,
            'foot_tags':      foot_tags,
            'gap_up':         gap_up,
            'shrink':         round(shrink, 1),
            'vol_ratio':      round(vol_ratio, 2),
            'exit_warn': exit_warn,
            'exit_reason': exit_reason,
            'reduce_pct': reduce_pct,
            'd_macd':    round(d_macd, 3),
            'w_macd':    round(w_macd, 3),
            'm_macd':    round(m_macd, 3),
        }
        return (result, '') if return_reason else result
    except Exception as e:
        if return_reason:
            msg = f"{type(e).__name__}: {str(e)[:120]}" if str(e) else type(e).__name__
            return None, msg
        return None


def scan(tickers, workers=8, period='1y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, debug_missing: bool = False):
    results = []
    missing = []  # list of {'ticker':..., 'reason':...}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(analyse, t, period, ttl_hours, max_retries, debug_missing): t
            for t in tickers
        }
        done = 0
        for f in as_completed(futs):
            done += 1
            print(f"\r  掃描進度: {done}/{len(tickers)}  ", end='', flush=True)
            out = f.result()

            if debug_missing:
                r, reason = out
                if r:
                    results.append(r)
                else:
                    missing.append({'ticker': futs[f], 'reason': reason or 'unknown'})
            else:
                r = out
                if r:
                    results.append(r)

    print()
    return (results, missing) if debug_missing else results


def print_dashboard(results, label):
    def fmt_macd(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return 'NA'
            return f"{float(x):+.2f}"
        except Exception:
            return 'NA'

    def fmt_sym(ticker: str) -> str:
        nm = get_company_name(ticker, max_retries=2)
        if nm and NAME_MAXLEN and len(nm) > int(NAME_MAXLEN):
            nm = nm[: int(NAME_MAXLEN)]
        return f"{ticker} {nm}".strip()

    # ── 持倉股：始終 pin 在儀表板最上方 ──────────────────────────────────
    PINNED_US  = [
        'SNDK', 'NVDA', 'MU', 'LITE',
        'GOOGL', 'GOOG', 'AMZN',
    ]
    PINNED_TW  = [
        '2330.TW', '2317.TW', '2308.TW',   # 台積電、鴻海、台達電
        '2454.TW', '2345.TW', '2337.TW',   # 聯發科、智邦、旺宏
        '2881.TW', '2891.TW',               # 富邦金、中信金
        '3037.TW', '3443.TW', '3711.TW',   # 欣興、創意、日月光
        '6442.TW', '6531.TW', '8046.TW',   # 光聖、愛普、南電
    ]

    bull    = [r for r in results if r['bull']]
    perfect = [r for r in results if r['perfect']]
    consol  = [r for r in results if r['consol']]
    # NEW: daily MACD turned negative but weekly+monthly still bullish → dip watch
    dip_watch = [r for r in results
                 if not r['bull']
                 and r['d_macd'] < 0          # daily turned negative
                 and not np.isnan(r['w_macd']) and r['w_macd'] > 0   # weekly still positive
                 and not np.isnan(r['m_macd']) and r['m_macd'] > 0]  # monthly still positive
    foot    = [r for r in results if r['macd_foot']]
    exit_   = [r for r in results if r['exit_warn']]

    # Build pinned lookup
    result_map = {r['ticker']: r for r in results}

    MAX_ROWS = 25   # increased from 15

    print(f"\n{'═'*65}")
    print(f"  🎯 傻瓜儀表板 — {label}  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"  掃描 {len(results)} 支股票")
    print(f"{'═'*65}")

    # ── Section 0: Pinned key stocks ─────────────────────────────────────────
    pinned_candidates = PINNED_US if label == '美股' else (PINNED_TW if label == '台股' else PINNED_US + PINNED_TW)
    pinned_found = [result_map[t] for t in pinned_candidates if t in result_map]
    if pinned_found:
        print(f"\n{'─'*65}")
        print(f"  📌 重點股票狀態  ({len(pinned_found)} 支)  — 始終顯示")
        print(f"{'─'*65}")
        for r in pinned_found:
            sym  = fmt_sym(r['ticker'])
            sta  = r['status']
            icon = '🟢' if '完美' in sta else ('✅' if '整理' in sta else ('🔴' if '出場' in r.get('exit_reason','') or r['exit_warn'] else ('⚠️' if '轉負' in sta else '⬜')))
            print(f"  {icon} {sym:<20} ${r['price']:>8.2f}  {sta:<12}  "
                  f"日MACD:{fmt_macd(r['d_macd'])}  週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}")

    # ── Section 1: Bull structure ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  🟢 多頭結構完整  ({len(bull)} 支)  →  持有 / 可買")
    print(f"{'─'*65}")
    if perfect:
        print(f"\n  🟢 完美多頭 ({len(perfect)} 支) — 日線柱狀體仍正:")
        for r in sorted(perfect, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"月MACD:{fmt_macd(r['m_macd'])}  週MACD:{fmt_macd(r['w_macd'])}")
    if consol:
        print(f"\n  ✅ 強勢整理 ({len(consol)} 支) — 日線回調但結構完整:")
        for r in sorted(consol, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"月MACD:{fmt_macd(r['m_macd'])}  週MACD:{fmt_macd(r['w_macd'])}")

    # ── Section 1b: Dip watch (日線轉負, 週月線仍多頭) ─────────────────────────
    if dip_watch:
        print(f"\n  🔵 日線回調、週月仍多頭 ({len(dip_watch)} 支) — 逢低留意:")
        for r in sorted(dip_watch, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"日MACD:{fmt_macd(r['d_macd'])}  週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}")

    # ── Section 2: MACD foot + gap ───────────────────────────────────────────
    confirmed_foot = [r for r in foot if r.get('foot_confirmed')]
    unconfirmed_foot = [r for r in foot if not r.get('foot_confirmed')]
    print(f"\n{'─'*65}")
    print(f"  📍 MACD 收腳訊號  ({len(foot)} 支)  →  最佳進場時機")
    print(f"  (確認標準: B動能必中 + 至少再中一項)")
    print(f"{'─'*65}")
    if confirmed_foot:
        print(f"\n  🟢 強勢確認 ({len(confirmed_foot)} 支) ← A+B+C 至少兩中且含B:")
        for r in sorted(confirmed_foot, key=lambda x: x['foot_score'], reverse=True):
            sym = fmt_sym(r['ticker'])
            tags = ' '.join(r.get('foot_tags', []))
            gap_tag = ' ✅跳空' if r['gap_up'] else ''
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"縮{r['shrink']:.0f}%{gap_tag}  [{tags}]  {r['status']}")
    if unconfirmed_foot:
        print(f"\n  🟡 待確認 ({len(unconfirmed_foot)} 支) — 條件不足，觀察次日:")
        for r in sorted(unconfirmed_foot, key=lambda x: x['shrink'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            tags = ' '.join(r.get('foot_tags', [])) or '—'
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"縮{r['shrink']:.0f}%  [{tags}]  {r['status']}")
    if not foot:
        print(f"    今日無收腳訊號")

    # ── Section 3: Exit warnings ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  🔴 出場警示  ({len(exit_)} 支)  →  準備減碼")
    print(f"{'─'*65}")
    if exit_:
        for r in sorted(exit_, key=lambda x: x['w_macd'] if not np.isnan(x['w_macd']) else float('inf'))[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            reduce_tag = f"  【{r['reduce_pct']}】" if r.get('reduce_pct') else ''
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"⚠️  {r['exit_reason']}{reduce_tag}  "
                  f"週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}")
    else:
        print(f"    目前無出場警示 — 持倉安全")

    print(f"\n{'═'*65}")
    print(f"  📌 傻瓜投資三原則:")
    print(f"  1. 🟢/✅ 在列 → 繼續持有，什麼都不做")
    print(f"  2. 📍 收腳+跳空 → 這是最好的買入時機")
    print(f"  3. 🔴 出場警示 → 週/月線轉負才是真正出場")
    print(f"  4. 🔵 日線回調週月仍多頭 → 可逢低分批買")
    print(f"{'═'*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tw', action='store_true', help='Only TW stocks')
    parser.add_argument('--us', action='store_true', help='Only US stocks')
    parser.add_argument('--workers', type=int, default=8)

    # yfinance stability controls
    parser.add_argument('--no-cache', action='store_true', help='Disable yfinance cache')
    parser.add_argument('--cache-ttl', type=float, default=8.0, help='Cache TTL in hours (default: 8)')
    parser.add_argument('--retries', type=int, default=3, help='Download retries (default: 3)')
    parser.add_argument('--max-downloads', type=int, default=2, help='Max concurrent downloads (default: 2)')
    parser.add_argument('--timeout', type=int, default=20, help='yfinance request timeout seconds (default: 20)')
    parser.add_argument('--max-info', type=int, default=1, help='Max concurrent yfinance info calls for names (default: 1)')
    parser.add_argument('--name-source', choices=['mixed','yahoo','alias'], default='mixed',
                        help="Company name source: mixed(alias->yahoo), yahoo(English from Yahoo only), alias(only ticker_aliases.json)")
    parser.add_argument('--name-len', type=int, default=10, help='Max company name length to display (default: 10)')

    parser.add_argument('--debug-missing', action='store_true', help='Show tickers that were skipped (and why)')
    parser.add_argument('--missing-limit', type=int, default=50, help='Max missing tickers to print (default: 50)')

    # ── Tavily 新聞 ───────────────────────────────────────────────────────────
    parser.add_argument('--news',            action='store_true', help='掃描結束後搜尋 Tavily 台股最新新聞')
    parser.add_argument('--news-market',     type=int, default=5, help='大盤新聞條數 (default: 5)')
    parser.add_argument('--news-stocks',     type=int, default=4, help='個股新聞顯示幾支 (default: 4)')
    parser.add_argument('--news-per-stock',  type=int, default=3, help='每支個股顯示幾則新聞 (default: 3)')
    parser.add_argument('--news-only',       action='store_true', help='只顯示新聞，跳過技術分析儀表板')

    args = parser.parse_args()

    # Apply globals
    NEWS_ENABLED = args.news or args.news_only
    USE_CACHE = (not args.no_cache)
    CACHE_TTL_HOURS = float(args.cache_ttl)
    MAX_RETRIES = int(args.retries)
    MAX_CONCURRENT_DOWNLOADS = max(1, int(args.max_downloads))
    DOWNLOAD_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    DOWNLOAD_TIMEOUT = max(5, int(args.timeout))

    MAX_CONCURRENT_INFO = max(1, int(args.max_info))
    INFO_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_INFO)

    NAME_SOURCE = str(args.name_source)
    NAME_MAXLEN = int(args.name_len)

    def _print_missing(missing_list):
        if not args.debug_missing:
            return
        if not missing_list:
            print("\n（Debug）沒有被跳過的 ticker。")
            return

        lim = max(1, int(args.missing_limit))
        print("\n" + "─" * 65)
        print(f"（Debug）被跳過 / 失敗的 ticker：{len(missing_list)} 支（顯示前 {min(lim, len(missing_list))} 支）")
        print("─" * 65)
        for item in missing_list[:lim]:
            print(f"  {item['ticker']:<12}  {item['reason']}")

    # ── 新聞-only 模式：直接搜尋，不做技術分析 ─────────────────────────────
    if args.news_only:
        print_news_section(
            [],
            label='台股',
            market_top=args.news_market,
            stock_top_n=0,
            stock_news_n=args.news_per_stock,
        )
        sys.exit(0)

    if args.tw:
        print(f"\n  掃描 {len(TW_TICKERS)} 支台股...", flush=True)
        out = scan(
            TW_TICKERS,
            workers=args.workers,
            period='3y',
            ttl_hours=CACHE_TTL_HOURS,
            max_retries=MAX_RETRIES,
            debug_missing=args.debug_missing,
        )
        if args.debug_missing:
            r, missing = out
        else:
            r, missing = out, []
        print_dashboard(r, '台股')
        _print_missing(missing)
        if NEWS_ENABLED:
            print_news_section(r, label='台股',
                               market_top=args.news_market,
                               stock_top_n=args.news_stocks,
                               stock_news_n=args.news_per_stock)

    elif args.us:
        print(f"\n  掃描 {len(US_TICKERS)} 支美股...", flush=True)
        out = scan(
            US_TICKERS,
            workers=args.workers,
            period='2y',
            ttl_hours=CACHE_TTL_HOURS,
            max_retries=MAX_RETRIES,
            debug_missing=args.debug_missing,
        )
        if args.debug_missing:
            r, missing = out
        else:
            r, missing = out, []
        print_dashboard(r, '美股')
        _print_missing(missing)
        if NEWS_ENABLED:
            print_news_section(r, label='美股',
                               market_top=args.news_market,
                               stock_top_n=args.news_stocks,
                               stock_news_n=args.news_per_stock)

    else:
        print(f"\n  掃描 {len(US_TICKERS)} 支美股...", flush=True)
        out_us = scan(
            US_TICKERS,
            workers=args.workers,
            period='2y',
            ttl_hours=CACHE_TTL_HOURS,
            max_retries=MAX_RETRIES,
            debug_missing=args.debug_missing,
        )
        if args.debug_missing:
            r_us, missing_us = out_us
        else:
            r_us, missing_us = out_us, []
        print_dashboard(r_us, '美股 Western')
        _print_missing(missing_us)

        print(f"\n  掃描 {len(TW_TICKERS)} 支台股...", flush=True)
        out_tw = scan(
            TW_TICKERS,
            workers=args.workers,
            period='3y',
            ttl_hours=CACHE_TTL_HOURS,
            max_retries=MAX_RETRIES,
            debug_missing=args.debug_missing,
        )
        if args.debug_missing:
            r_tw, missing_tw = out_tw
        else:
            r_tw, missing_tw = out_tw, []
        print_dashboard(r_tw, '台股')
        _print_missing(missing_tw)
        if NEWS_ENABLED:
            # 合併台股+美股結果一起找相關個股新聞
            print_news_section(r_tw + r_us, label='台股+美股',
                               market_top=args.news_market,
                               stock_top_n=args.news_stocks,
                               stock_news_n=args.news_per_stock)
