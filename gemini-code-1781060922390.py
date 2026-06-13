"""
傻瓜儀表板 — Fool's Investment Dashboard (CIO Buy-side Enhanced Edition)
======================================================================
從買方（Buy-side）機構視角出發，重新定錨賣方評級與風險邊界：

  🟢 多頭結構與基準錨（Baseline Anchor）監控
  📍 MACD 短期轉折與量價確認（動態進場）
  🔴 機構下檔協定（Downside Protocol）與結構性轉負警示
  📊 美光（MU）專屬：CIO 假設監控儀表板（Assumption Watchlist）

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

# ── yfinance acceleration & stability ────────────────────────────────────────
CACHE_DIR = os.path.join(os.getcwd(), '.cache_yfinance')
os.makedirs(CACHE_DIR, exist_ok=True)

NAME_CACHE_DIR = os.path.join(os.getcwd(), '.cache_yfinance_names')
os.makedirs(NAME_CACHE_DIR, exist_ok=True)

ALIASES_FILE = os.path.join(os.getcwd(), 'ticker_aliases.json')

USE_CACHE = True
CACHE_TTL_HOURS = 8
NAME_CACHE_TTL_DAYS = 30
MAX_RETRIES = 3
MAX_CONCURRENT_DOWNLOADS = 2
MAX_CONCURRENT_INFO = 1
DOWNLOAD_TIMEOUT = 20 

MACD_LINE_MIN_PCT = 0.003

DOWNLOAD_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_DOWNLOADS)
INFO_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_INFO)

# ── Tavily 新聞搜尋設定 ──────────────────────────────────────────────────────
TAVILY_API_KEY   = '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'
TAVILY_ENDPOINT  = 'https://api.tavily.com/search'
NEWS_ENABLED     = False

_TICKER_ALIASES = {}
_NAME_MEMO = {}

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

    built_in = {
        '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '2344.TW': '華邦電',
        '2308.TW': '台達電', '2345.TW': '智邦', '2337.TW': '旺宏', '2881.TW': '富邦金',
        '2891.TW': '中信金', '3037.TW': '欣興', '3443.TW': '創意', '3711.TW': '日月光投控',
        '6442.TW': '光聖', '6531.TW': '愛普', '8046.TW': '南電',
        'NVDA': 'NVIDIA', 'MU': 'Micron', 'SNDK': 'SanDisk', 'LITE': 'Lumentum',
        'GOOGL':'Google A', 'GOOG': 'Google C', 'AMZN': 'Amazon', 'TSM': 'TSMC',
        'AAPL': 'Apple', 'MSFT': 'Microsoft',
    }
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
    if not ticker: return ''
    tkey = str(ticker).upper()

    if NAME_SOURCE in ('mixed', 'alias'):
        _load_aliases_once()
        if tkey in _TICKER_ALIASES: return _TICKER_ALIASES[tkey]
        if NAME_SOURCE == 'alias': return ''

    if tkey in _NAME_MEMO: return _NAME_MEMO[tkey]

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

    name = ''
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            with INFO_SEMAPHORE:
                yt = yf.Ticker(ticker)
                info = None
                try: info = yt.get_info()
                except Exception: info = getattr(yt, 'info', None)

                if not isinstance(info, dict): raise RuntimeError('No info dict')
                name = (info.get('shortName') or info.get('longName') or info.get('displayName') or '').strip()
                time.sleep(0.15 + random.random() * 0.25)
            if name: break
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
    cache_file = _cache_path(ticker, period, auto_adjust=True)
    if USE_CACHE and _is_cache_fresh(cache_file, ttl_hours):
        try: return pd.read_pickle(cache_file)
        except Exception:
            try: os.remove(cache_file)
            except Exception: pass

    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            with DOWNLOAD_SEMAPHORE:
                df = yf.download(ticker, period=period, interval='1d', progress=False, auto_adjust=True, threads=False, timeout=int(DOWNLOAD_TIMEOUT))
                if df is None or df.empty:
                    try: df = yf.Ticker(ticker).history(period=period, interval='1d', auto_adjust=True)
                    except Exception: pass
                time.sleep(0.15 + random.random() * 0.25)

            if df is None or df.empty: raise RuntimeError('Empty data from yfinance')
            if USE_CACHE:
                try: df.to_pickle(cache_file)
                except Exception: pass
            return df
        except Exception:
            backoff = min(30.0, (1.2 * (2 ** (attempt - 1))) + random.random())
            time.sleep(backoff)
    return pd.DataFrame()


# ── Tavily 新聞搜尋 ────────────────────────────────────────────────────────────
def fetch_news_tavily(query: str, max_results: int = 5, api_key: str = TAVILY_API_KEY) -> list:
    import urllib.request, urllib.error
    try:
        payload = json.dumps({
            'api_key': api_key, 'query': query, 'search_depth': 'basic',
            'max_results': max_results, 'include_answer': False, 'include_images': False, 'topic': 'news',
        }).encode('utf-8')
        req = urllib.request.Request(TAVILY_ENDPOINT, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('results', [])
    except Exception:
        return []


def _fmt_news_item(a: dict, indent: str = '    ', title_max: int = 70) -> str:
    title = (a.get('title') or '無標題').strip()[:title_max]
    url = (a.get('url') or '').strip()
    pub = (a.get('published_date') or '')[:10]
    pub_str = f'  [{pub}]' if pub else ''
    lines = [f'{indent}• {title}{pub_str}']
    if url: lines.append(f'{indent}  🔗 {url}')
    return '\n'.join(lines)


def print_news_section(results: list, label: str = '台股', market_top: int = 5, stock_top_n: int = 4, stock_news_n: int = 3):
    today_str = datetime.now().strftime('%Y年%m月')
    print(f"\n{'═'*65}")
    print(f"  📰 最新市場情報與動態  (Tavily 即時監控 · {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'═'*65}")

    mkt_query = f'{label} 大盤 指數 今日行情 總經趨勢 {today_str}'
    print(f"\n  🌐 大盤與總經觀點  (查詢: {mkt_query})")
    print(f"{'─'*65}")
    mkt_news = fetch_news_tavily(mkt_query, max_results=market_top)
    if mkt_news:
        for a in mkt_news[:market_top]: print(_fmt_news_item(a))
    else:
        print('    （無搜尋結果）')

    exit_stocks = [r for r in results if r.get('exit_warn')]
    if exit_stocks and stock_top_n > 0:
        n = min(len(exit_stocks), stock_top_n)
        print(f"\n  🔴 觸發下檔協定（Downside Protocol）個股新聞  (前 {n} 支)")
        print(f"{'─'*65}")
        for r in exit_stocks[:n]:
            ticker = r['ticker']
            code = ticker.split('.')[0]
            nm = get_company_name(ticker, max_retries=1)
            sym = f'{code} {nm}'.strip() if nm else code
            q = f'{code} {nm} 財報 供應鏈 負面 新聞' if nm else f'{code} 新聞'
            articles = fetch_news_tavily(q, max_results=stock_news_n)
            print(f'\n  🔴 {sym}  ({r.get("exit_reason","")})  ── ${r["price"]:.2f}')
            if articles:
                for a in articles[:stock_news_n]: print(_fmt_news_item(a))
            else:
                print('    （無搜尋結果）')


# ── Ticker 列表定義 ────────────────────────────────────────────────────────────
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
    'GOOGL','GOOG','TSLA','VRT','CRDO','ALAB','DELL','LITE',
    'JPM','SPY','XLE','BRK-A','BRK-B','MCD','LIN','RDDT','KO',
    'DIS','CRM','LULU','XOP','SMH','SHOP','COIN','RSP',
]


# ── Core Analysis per Ticker ──────────────────────────────────────────────────
def analyse(ticker, period='1y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, return_reason: bool = False):
    try:
        raw = download_with_cache(ticker, period=period, ttl_hours=ttl_hours, max_retries=max_retries)
        if raw.empty: return (None, 'empty data') if return_reason else None
        if len(raw) < 60: return (None, f'not enough rows (<60): {len(raw)}') if return_reason else None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        df = raw.rename(columns={'Close':'c','Open':'o','High':'h','Low':'l','Volume':'v'})
        df.index = pd.to_datetime(df.index)

        def _series(col: str) -> pd.Series:
            x = df[col]
            if isinstance(x, pd.DataFrame): x = x.iloc[:, 0]
            return x

        c = _series('c')
        o = _series('o') if 'o' in df.columns else None
        h = _series('h') if 'h' in df.columns else None

        # ── Daily MACD ───────────────────────────────────────────────────────
        def macd_hist(s):
            ml = s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean()
            sig = ml.ewm(span=9,adjust=False).mean()
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
            w_macd = w_macd_prev = w_hist = w_hist_prev = np.nan

        # ── Monthly MACD ─────────────────────────────────────────────────────
        cm = c.resample('ME').last().dropna()
        if len(cm) >= 15:
            ml_m, sig_m, mh_m = macd_hist(cm)
            m_macd = float(ml_m.iloc[-1])
        else:
            m_macd = np.nan

        # ── Regime Status Definition ─────────────────────────────────────────
        d_pos = d_macd > 0
        w_pos = (not np.isnan(w_macd)) and (w_macd > 0)
        m_pos = (not np.isnan(m_macd)) and (m_macd > 0)
        d_neg_hist = d_hist < 0

        if np.isnan(w_macd) or np.isnan(m_macd): status = '⚪ 資料不足'
        elif d_neg_hist and d_pos and w_pos and m_pos: status = '✅ 基準錨·強勢整理'
        elif not d_neg_hist and d_pos and w_pos and m_pos: status = '🟢 基準錨·完美多頭'
        elif d_pos and w_pos and not m_pos: status = '⚠️  結構警示(月線負)'
        elif d_pos and not w_pos: status = '⚠️  結構警示(週線負)'
        elif not d_pos: status = '❌ 下檔觸發(日線負)'
        else: status = '⚪ 觀察資產'

        # ── Technical Confirmation Signals ──────────────────────────────────
        v_series = _series('v') if 'v' in df.columns else None
        vol_ratio = 1.0
        if v_series is not None and len(v_series) >= 20:
            vol_ma20 = float(v_series.rolling(20).mean().iloc[-1])
            if vol_ma20 > 0: vol_ratio = float(v_series.iloc[-1]) / vol_ma20

        sma10 = float(c.rolling(10).mean().iloc[-1])
        sma20 = float(c.rolling(20).mean().iloc[-1])
        _last_close = float(c.dropna().iloc[-1]) if len(c.dropna()) else np.nan
        price_above_ma = (not np.isnan(_last_close)) and _last_close > max(sma10, sma20)

        d_hist_2 = float(mh_d.iloc[-3]) if len(mh_d) >= 3 else np.nan

        macd_foot = False; gap_up = False; shrink = 0.0
        foot_confirmed = False; foot_score = 0; foot_tags = []
        if d_hist < 0 and d_hist_prev < 0 and abs(d_hist) < abs(d_hist_prev):
            shrink = (abs(d_hist_prev) - abs(d_hist)) / (abs(d_hist_prev) + 1e-10) * 100
            if shrink >= 10:
                macd_foot = True
                if len(df) >= 2 and o is not None and h is not None:
                    gap_up = float(o.iloc[-1]) > float(h.iloc[-2])

                cond_A = gap_up or price_above_ma
                cond_B = shrink >= 20 or (not np.isnan(d_hist_2) and d_hist_2 < 0 and d_hist_prev < d_hist_2 and d_hist < d_hist_prev)
                cond_C = vol_ratio >= 1.3

                if cond_A: foot_tags.append('A價格✅')
                if cond_B: foot_tags.append('B動能✅')
                if cond_C: foot_tags.append(f'C量能✅({vol_ratio:.1f}x)')
                foot_score = sum([cond_A, cond_B, cond_C])
                foot_confirmed = foot_score >= 2 and cond_B

        price = float(c.dropna().iloc[-1]) if len(c.dropna()) else np.nan

        # ── Downside Protocol Gate ───────────────────────────────────────────
        exit_warn = False
        exit_reasons = []

        if (not np.isnan(w_hist_prev)) and (not np.isnan(w_hist)) and w_hist_prev > 0 and w_hist < 0:
            exit_warn = True
            exit_reasons.append('週柱翻黑')

        w_just_crossed = (not np.isnan(w_macd_prev)) and w_macd_prev > 0 and w_macd <= 0
        if not np.isnan(w_macd) and w_macd <= 0 and price > 0:
            if (abs(w_macd) / price > MACD_LINE_MIN_PCT) or w_just_crossed:
                exit_warn = True
                exit_reasons.append('週線破0')

        if not np.isnan(m_macd) and m_macd <= 0 and price > 0:
            if abs(m_macd) / price > MACD_LINE_MIN_PCT:
                exit_warn = True
                exit_reasons.append('月線破0')

        exit_reason = '+'.join(exit_reasons)
        w_borderline = (not np.isnan(w_macd) and price > 0 and abs(w_macd) / price < MACD_LINE_MIN_PCT)

        if '月線破0' in exit_reasons and '週線破0' in exit_reasons:
            reduce_pct = '體制破壞：回收 50%~80% 曝險'
        elif '週線破0' in exit_reasons:
            if w_borderline and w_just_crossed: reduce_pct = '邊緣觀察：暫減 0%~20%'
            elif w_just_crossed: reduce_pct = '首次跌破：減碼 15%~25%'
            else: reduce_pct = '趨勢下行：縮減 30%~60% 權重'
        elif '週柱翻黑' in exit_reasons:
            reduce_pct = '風險防範：微調 20%~35%'
        else:
            reduce_pct = ''

        return ({
            'ticker': ticker, 'price': price, 'status': status,
            'bull': d_pos and w_pos and m_pos, 'perfect': not d_neg_hist and d_pos and w_pos and m_pos, 'consol': d_neg_hist and d_pos and w_pos and m_pos,
            'macd_foot': macd_foot, 'foot_confirmed': foot_confirmed, 'foot_score': foot_score, 'foot_tags': foot_tags,
            'gap_up': gap_up, 'shrink': round(shrink, 1), 'vol_ratio': round(vol_ratio, 2),
            'exit_warn': exit_warn, 'exit_reason': exit_reason, 'reduce_pct': reduce_pct,
            'd_macd': round(d_macd, 3), 'w_macd': round(w_macd, 3), 'm_macd': round(m_macd, 3),
        }, '') if return_reason else {
            'ticker': ticker, 'price': price, 'status': status,
            'bull': d_pos and w_pos and m_pos, 'perfect': not d_neg_hist and d_pos and w_pos and m_pos, 'consol': d_neg_hist and d_pos and w_pos and m_pos,
            'macd_foot': macd_foot, 'foot_confirmed': foot_confirmed, 'foot_score': foot_score, 'foot_tags': foot_tags,
            'gap_up': gap_up, 'shrink': round(shrink, 1), 'vol_ratio': round(vol_ratio, 2),
            'exit_warn': exit_warn, 'exit_reason': exit_reason, 'reduce_pct': reduce_pct,
            'd_macd': round(d_macd, 3), 'w_macd': round(w_macd, 3), 'm_macd': round(m_macd, 3),
        }
    except Exception as e:
        if return_reason: return None, f"{type(e).__name__}: {str(e)[:120]}"
        return None


def scan(tickers, workers=8, period='1y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, debug_missing: bool = False):
    results = []; missing = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyse, t, period, ttl_hours, max_retries, debug_missing): t for t in tickers}
        done = 0
        for f in as_completed(futs):
            done += 1
            print(f"\r  📊 買方量化掃描進度: {done}/{len(tickers)}  ", end='', flush=True)
            out = f.result()
            if debug_missing:
                r, reason = out
                if r: results.append(r)
                else: missing.append({'ticker': futs[f], 'reason': reason or 'unknown'})
            else:
                if out: results.append(out)
    print()
    return (results, missing) if debug_missing else results


# ── CIO 專屬：美光科技（MU）基本面假設監控儀表板 ─────────────────────────────
def print_mu_assumption_watchlist():
    print(f"\n{'═'*65}")
    print(f"  📊 核心資產監控：美光 (MU) 買方假設驗證清單 (Assumption Watchlist)")
    print(f"  [ regime 判定標準：不對單一賣方評分定錨，僅追蹤底層核心因子 ]")
    print(f"{'═'*65}")
    
    print("\n  A. 結構性需求（Demand）因子")
    print("    • HBM/伺服器 DRAM 報價 (ASP) 走勢：")
    print("      👉 [監控點] 連續兩季低於公司 Guideline 或市場預期下緣 ➔ 判定 AI 體制轉換置信度下降。")
    print("    • 3–5 年期長約（LTA）產能覆蓋率：")
    print("      👉 [監控點] 覆蓋率持續上升意謂自由現金流能見度定錨；若停滯或大幅下滑 ➔ 回歸大宗商品循環。")
    
    print("\n  B. 結構性供給（Supply）防禦線")
    print("    • 三大原廠（SK Hynix / 三星）資本支出與擴產節奏：")
    print("      👉 [監控點] 先進 HBM 產線若出現產能過剩前移 ➔ 極端牛市（Bull Case）容錯率將瞬間歸零。")
    print("    • 先進製程（1-alpha / 1-beta）及高階封裝良率：")
    print("      👉 [監控點] 良率異常為「跳變風險」（Step-risk），將直接導致毛利率與 EPS 預估模型失效。")
    
    print("\n  C. 財務落地與變現效率（Monetization）")
    print("    • 毛利率新平台支撐力：")
    print("      👉 [監控點] 觀察非 GAAP 毛利率是否穩固在 75%～80%+ 區間。此平台為高估值的防禦底線。")
    print("    • CapEx ➔ FCF 轉換效率：")
    print("      👉 [監控點] 高額資本支出不必然等同利多，投委會將嚴格審查其是否轉化為高度可預測之每股盈餘。")
    
    print(f"{'─'*65}")
    print("  💡 賣方行為買方解讀提示 (Sell-side Behavior Translation):")
    print("    - 瑞銀 (UBS) / Cantor ($1,500~$1,625) ➔ 容錯率極低之單點極端押注，僅供「上行壓力測試（Upside Stress Test）」之情境用途。")
    print("    - 富國 (Wells Fargo) 的暴力上調 ➔ 視為「共識擴散」與賣方追價（Catch-up），警惕交易擁擠度（Crowding Risk）與回撤肥尾風險。")
    print(f"{'═'*65}\n")


def print_dashboard(results, label):
    def fmt_macd(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return 'NA'
        return f"{float(x):+.2f}"

    def fmt_sym(ticker: str) -> str:
        nm = get_company_name(ticker, max_retries=2)
        if nm and NAME_MAXLEN and len(nm) > int(NAME_MAXLEN): nm = nm[: int(NAME_MAXLEN)]
        return f"{ticker} {nm}".strip()

    PINNED_US = ['SNDK', 'NVDA', 'MU', 'LITE', 'GOOGL', 'GOOG', 'AMZN']
    PINNED_TW = ['2330.TW', '2317.TW', '2308.TW', '2454.TW', '2345.TW', '2337.TW', '2881.TW', '2891.TW', '3037.TW', '3443.TW', '3711.TW', '6442.TW', '6531.TW', '8046.TW']

    bull = [r for r in results if r['bull']]
    perfect = [r for r in results if r['perfect']]
    consol = [r for r in results if r['consol']]
    dip_watch = [r for r in results if not r['bull'] and r['d_macd'] < 0 and not np.isnan(r['w_macd']) and r['w_macd'] > 0 and not np.isnan(r['m_macd']) and r['m_macd'] > 0]
    foot = [r for r in results if r['macd_foot']]
    exit_ = [r for r in results if r['exit_warn']]

    result_map = {r['ticker']: r for r in results}
    MAX_ROWS = 25

    print(f"\n{'═'*65}")
    print(f"  🎯 CIO 買方資產管理儀表板 — {label}  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"  總計審查: {len(results)} 支標的資產")
    print(f"{'═'*65}")

    pinned_candidates = PINNED_US if label == '美股' else (PINNED_TW if label == '台股' else PINNED_US + PINNED_TW)
    pinned_found = [result_map[t] for t in pinned_candidates if t in result_map]
    if pinned_found:
        print(f"\n{'─'*65}")
        print(f"  📌 核心戰略持倉狀態監控  ({len(pinned_found)} 支)  — 始終錨定顯示")
        print(f"{'─'*65}")
        for r in pinned_found:
            sym = fmt_sym(r['ticker'])
            sta = r['status']
            icon = '🟢' if '完美' in sta else ('✅' if '整理' in sta else ('🔴' if r['exit_warn'] else ('⚠️' if '轉負' in sta else '⬜')))
            print(f"  {icon} {sym:<20} ${r['price']:>8.2f}  {sta:<14}  日MACD:{fmt_macd(r['d_macd'])}  週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}")

    print(f"\n{'─'*65}")
    print(f"  🟢 多頭體制完整資產區（Baseline Anchor） ({len(bull)} 支) ➔ 建議：維持權重 / 穩健配置")
    print(f"{'─'*65}")
    if perfect:
        print(f"\n  🟢 完美多頭範疇 ({len(perfect)} 支) — 日級別動能向上:")
        for r in sorted(perfect, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  月MACD:{fmt_macd(r['m_macd'])}  週MACD:{fmt_macd(r['w_macd'])}")
    if consol:
        print(f"\n  ✅ 基準錨·強勢整理 ({len(consol)} 支) — 日級別回調但長週期體制未變:")
        for r in sorted(consol, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  月MACD:{fmt_macd(r['m_macd'])}  週MACD:{fmt_macd(r['w_macd'])}")

    if dip_watch:
        print(f"\n  🔵 日線回調、週月多頭 ({len(dip_watch)} 支) ➔ 戰術性左側留意點:")
        for r in sorted(dip_watch, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  日:{fmt_macd(r['d_macd'])}  週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}")

    confirmed_foot = [r for r in foot if r.get('foot_confirmed')]
    unconfirmed_foot = [r for r in foot if not r.get('foot_confirmed')]
    print(f"\n{'─'*65}")
    print(f"  📍 買方量價確認進場訊號  ({len(foot)} 支) ➔ 戰術右側增碼時機")
    print(f"  [ 嚴格標準：B動能因子必中 + A/C 至少任一滿足 ]")
    print(f"{'─'*65}")
    if confirmed_foot:
        print(f"\n  🟢 買方高確信確認 ({len(confirmed_foot)} 支) [動能與量價結構共振]:")
        for r in sorted(confirmed_foot, key=lambda x: x['foot_score'], reverse=True):
            sym = fmt_sym(r['ticker'])
            tags = ' '.join(r.get('foot_tags', []))
            gap_tag = ' ✅跳空' if r['gap_up'] else ''
            print(f"    {sym:<22} ${r['price']:>8.2f}  縮{r['shrink']:.0f}%{gap_tag}  [{tags}]  {r['status']}")
    if unconfirmed_foot:
        print(f"\n  🟡 待驗證右側訊號 ({len(unconfirmed_foot)} 支) — 容錯防禦不足，靜待次日確認:")
        for r in sorted(unconfirmed_foot, key=lambda x: x['shrink'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            tags = ' '.join(r.get('foot_tags', [])) or '—'
            print(f"    {sym:<22} ${r['price']:>8.2f}  縮{r['shrink']:.0f}%  [{tags}]  {r['status']}")

    print(f"\n{'─'*65}")
    print(f"  🔴 觸發機構下檔協定（Downside Protocol Warning） ({len(exit_)} 支)")
    print(f"{'─'*65}")
    if exit_:
        for r in sorted(exit_, key=lambda x: x['w_macd'] if not np.isnan(x['w_macd']) else float('inf'))[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            reduce_tag = f"  【{r['reduce_pct']}】" if r.get('reduce_pct') else ''
            print(f"    {sym:<22} ${r['price']:>8.2f}  ⚠️  {r['exit_reason']}{reduce_tag}  週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}")
    else:
        print(f"    目前未觸發任何資產風控下檔協定 — 核心持倉體制安全")

    # ── 美股特殊處理：如果是掃描美股或全掃，在儀表板尾端強制輸出美光專屬 Assumption Watchlist
    if label in ('美股', '美股 Western', '台股+美股'):
        print_mu_assumption_watchlist()

    print(f"\n{'═'*65}")
    print(f"  📜 投資委員會執行決議總結 (CIO Investment Committee Executive Summary):")
    print(f"  1. 對於高波動肥尾資產（如美光 MU 44x P/E），我們將賣方評分（Score）視為雜訊較高的「弱訊號」。")
    print(f"  2. 決策權重完全取決於底層結構性假設是否透明、可監控且具備高容錯率。")
    print(f"  3. 依據情境用途（Baseline Anchor 基準配置 vs Stress Test 壓力測試）動態決定資本配置大小與退出邊界。")
    print(f"  4. [下檔執行標準] 不以單一目標價作界線，若核心長約、毛利率平台或供給端擴產速度發生關鍵體制破壞（Regime Break），")
    print(f"     則立即啟動 Downside Protocol 進行強制降級與權重收回。")
    print(f"{'═'*65}\n")


# ── Main 入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tw', action='store_true', help='Only TW stocks')
    parser.add_argument('--us', action='store_true', help='Only US stocks')
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    parser.add_argument('--cache-ttl', type=float, default=8.0, help='Cache TTL in hours')
    parser.add_argument('--retries', type=int, default=3, help='Retries')
    parser.add_argument('--max-downloads', type=int, default=2, help='Max concurrent downloads')
    parser.add_argument('--timeout', type=int, default=20, help='Timeout seconds')
    parser.add_argument('--max-info', type=int, default=1, help='Max info calls')
    parser.add_argument('--name-source', choices=['mixed','yahoo','alias'], default='mixed')
    parser.add_argument('--name-len', type=int, default=10, help='Max name len')

    parser.add_argument('--debug-missing', action='store_true', help='Show missing')
    parser.add_argument('--missing-limit', type=int, default=50)

    parser.add_argument('--news', action='store_true', help='Tavily news scan')
    parser.add_argument('--news-market', type=int, default=5)
    parser.add_argument('--news-stocks', type=int, default=4)
    parser.add_argument('--news-per-stock', type=int, default=3)
    parser.add_argument('--news-only', action='store_true')

    args = parser.parse_args()

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
        if not args.debug_missing: return
        if not missing_list: return
        lim = max(1, int(args.missing_limit))
        print(f"\n  ⚠️  被跳過 / 失敗的資產標的：{len(missing_list)} 支")
        for item in missing_list[:lim]:
            print(f"    {item['ticker']:<12}  {item['reason']}")

    if args.news_only:
        print_news_section([], label='台股', market_top=args.news_market, stock_top_n=0, stock_news_n=args.news_per_stock)
        sys.exit(0)

    if args.tw:
        print(f"\n  🚀 開始執行 {len(TW_TICKERS)} 支台股配置模型審查...", flush=True)
        out = scan(TW_TICKERS, workers=args.workers, period='3y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, debug_missing=args.debug_missing)
        r, missing = out if args.debug_missing else (out, [])
        print_dashboard(r, '台股')
        _print_missing(missing)
        if NEWS_ENABLED: print_news_section(r, label='台股', market_top=args.news_market, stock_top_n=args.news_stocks, stock_news_n=args.news_per_stock)

    elif args.us:
        print(f"\n  🚀 開始執行 {len(US_TICKERS)} 支美股配置模型審查...", flush=True)
        out = scan(US_TICKERS, workers=args.workers, period='2y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, debug_missing=args.debug_missing)
        r, missing = out if args.debug_missing else (out, [])
        print_dashboard(r, '美股')
        _print_missing(missing)
        if NEWS_ENABLED: print_news_section(r, label='美股', market_top=args.news_market, stock_top_n=args.news_stocks, stock_news_n=args.news_per_stock)

    else:
        print(f"\n  🚀 開始執行 {len(US_TICKERS)} 支美股配置模型審查...", flush=True)
        out_us = scan(US_TICKERS, workers=args.workers, period='2y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, debug_missing=args.debug_missing)
        r_us, missing_us = out_us if args.debug_missing else (out_us, [])
        print_dashboard(r_us, '美股 Western')
        _print_missing(missing_us)

        print(f"\n  🚀 開始執行 {len(TW_TICKERS)} 支台股配置模型審查...", flush=True)
        out_tw = scan(TW_TICKERS, workers=args.workers, period='3y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, debug_missing=args.debug_missing)
        r_tw, missing_tw = out_tw if args.debug_missing else (out_tw, [])
        print_dashboard(r_tw, '台股')
        _print_missing(missing_tw)
        if NEWS_ENABLED: print_news_section(r_tw + r_us, label='台股+美股', market_top=args.news_market, stock_top_n=args.news_stocks, stock_news_n=args.news_per_stock)