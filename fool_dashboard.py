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
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except (AttributeError, io.UnsupportedOperation):
    pass
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── yfinance acceleration & stability (cache + retry + rate-limit) ───────────
CACHE_DIR = os.path.join(os.getcwd(), '.cache_yfinance')
os.makedirs(CACHE_DIR, exist_ok=True)

# Company name cache (for printing: "2330.TW 台積電")
NAME_CACHE_DIR = os.path.join(os.getcwd(), '.cache_yfinance_names')

# Earnings growth cache (30-day TTL; used to weight reduce recommendations)
EARNINGS_CACHE_DIR = os.path.join(os.getcwd(), '.cache_earnings')
os.makedirs(EARNINGS_CACHE_DIR, exist_ok=True)
os.makedirs(NAME_CACHE_DIR, exist_ok=True)

# Pre/post-market price cache (15-min TTL; only meaningful for US tickers
# during US pre-market ~16:00-21:30 / post-market ~04:00-08:00 Taiwan time)
PREMARKET_CACHE_DIR = os.path.join(os.getcwd(), '.cache_premarket')
os.makedirs(PREMARKET_CACHE_DIR, exist_ok=True)
PREMARKET_CACHE_TTL_MIN = 15

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

# 週線柱狀體翻黑門檻（兩個條件同時需通過）：
#   Gate A 相對門檻：|新負柱| >= 前正柱 * RATIO  → 濾掉「翻黑幅度太小」
#   Gate B 絕對門檻：|新負柱| >= 週收盤價 * PCT   → 濾掉「0 軸附近微幅雜訊」
# 設 0 可關閉任一門檻（還原原始行為）
HIST_FLIP_MIN_RATIO = 0.3    # 30% of previous positive bar
HIST_FLIP_MIN_PCT   = 0.002  # 0.2% of weekly close price
# 週/月線 MACD (DIF line) 跌破 0 的最小幅度門檻
# 例: 玉山金 close=34, 門檻=0.068；需 |w_macd| >= 此值才觸發
# 設 0 可關閉（還原原始行為）
MACD_LINE_MIN_PCT   = 0.002  # 0.2% of close price

# 減碼幅度基準（由 exit_reason 決定）
# earnings modifier 再乘上這個基準決定最終建議
_REDUCE_BASE = {
    'both':  (50, 80),   # 週+月 MACD < 0
    'week':  (30, 60),   # 週 MACD < 0 only
    'hist':  (20, 35),   # 週線柱狀體翻黑 only
    'month': (20, 40),   # 月 MACD < 0 only (週仍正)
}

# 個股手動覆寫（最高優先；只在特殊基本面理由時使用）
REDUCE_OVERRIDES: dict[str, str] = {}

NAME_CACHE_TTL_DAYS = 30
MAX_RETRIES = 3
MAX_CONCURRENT_DOWNLOADS = 2
MAX_CONCURRENT_INFO = 1
DOWNLOAD_TIMEOUT = 20  # seconds per request

# Concurrency limiters (reduce Yahoo throttling)
DOWNLOAD_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_DOWNLOADS)
INFO_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_INFO)

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


def _fetch_earnings_modifier(ticker: str, ttl_days: int = 30) -> float:
    """
    Fetch earningsGrowth (YoY) via yfinance and map to a reduce-size multiplier:
      growth > 20%   → 0.70  (strong earnings, soften technical signal)
      growth 10-20%  → 0.85
      growth 0-10%   → 1.00  (neutral)
      growth -10~0%  → 1.15
      growth < -10%  → 1.30  (declining, amplify signal)
    Falls back to 1.0 if data unavailable (e.g. most Taiwan stocks).
    Cached 30 days so repeated scans are instant.
    """
    path = os.path.join(EARNINGS_CACHE_DIR, f"{_safe_name(ticker)}.json")
    if os.path.exists(path):
        try:
            if time.time() - os.path.getmtime(path) < ttl_days * 86400:
                with open(path) as f:
                    return float(json.load(f).get('modifier', 1.0))
        except Exception:
            pass

    modifier = 1.0
    try:
        with INFO_SEMAPHORE:
            info = yf.Ticker(ticker).info
        eg = info.get('earningsGrowth')
        if eg is None:
            eg = info.get('revenueGrowth')
        if eg is not None:
            eg = float(eg)
            if eg > 0.20:
                modifier = 0.70
            elif eg > 0.10:
                modifier = 0.85
            elif eg < -0.10:
                modifier = 1.30
            elif eg < 0:
                modifier = 1.15
    except Exception:
        pass

    try:
        with open(path, 'w') as f:
            json.dump({'modifier': modifier, 'ticker': ticker}, f)
    except Exception:
        pass
    return modifier


def _fetch_premarket(ticker: str, ttl_minutes: float = PREMARKET_CACHE_TTL_MIN):
    """
    Fetch pre-market (or post-market, as fallback) price via yfinance .info and
    compute the % change vs the previous close.
    Returns {'price': float, 'pct': float, 'session': 'pre'|'post'} or None if
    no pre/post-market quote is currently available (i.e. outside those windows).
    Cached briefly (15min) since pre/post-market prices move continuously.
    """
    path = os.path.join(PREMARKET_CACHE_DIR, f"{_safe_name(ticker)}.json")
    if os.path.exists(path):
        try:
            if time.time() - os.path.getmtime(path) < ttl_minutes * 60:
                with open(path) as f:
                    data = json.load(f)
                return data if data.get('price') is not None else None
        except Exception:
            pass

    out = None
    try:
        with INFO_SEMAPHORE:
            info = yf.Ticker(ticker).info
        prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
        pm_price = info.get('preMarketPrice')
        session = 'pre'
        if pm_price is None:
            pm_price = info.get('postMarketPrice')
            session = 'post'
        if pm_price is not None and prev_close:
            pct = (float(pm_price) - float(prev_close)) / float(prev_close) * 100
            out = {'price': float(pm_price), 'pct': pct, 'session': session}
    except Exception:
        pass

    try:
        with open(path, 'w') as f:
            json.dump(out if out is not None else {'price': None}, f)
    except Exception:
        pass
    return out


def _load_aliases_once():
    global _TICKER_ALIASES
    if _TICKER_ALIASES:
        return

    # Built-ins (edit/add freely in ticker_aliases.json)
    built_in = {
        '2330.TW': '台積電',
        '2317.TW': '鴻海',
        '2454.TW': '聯發科',
        '2344.TW': '華邦電',
        'NVDA': 'NVIDIA',
        'MU': 'Micron',
        'TSM': 'TSMC',
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

# ── Ticker lists ──────────────────────────────────────────────────────────────
_TWO = {'3498','3615','4533','4577','4768','4908','4991','5011','6134','6187',
        '6220','6530','6877','7805','8086','8908','8917','8927','6274','1785',
        '4749','3131','6683','3363','3081','6510','8069','6223','5483','6163',
        '7709','7717','3260','3491','5371','3105','4971','8064','3163','3455',
        '3680','4772','6788','7703','8147','8071','8027','5351','7734','7751',
        '6138','1569','1595','4951','6234','6488','6207','3624','8455','8291',
        '3577','3236','3691','6204','6432','3609','3450','3581','3265',
        '5289','3587','3264','3663','6538','3580','8044','8299','3209','6147',
        '1815','8358'}
_T = {'3449'}

def _t(code):
    if code in _T:   return f"{code}.T"
    if code in _TWO: return f"{code}.TWO"
    return f"{code}.TW"

TW_CODES = [
    '2330','2317','6515','2408','2308','2313','2454','2485','2337','2344',
    '2367','3481','2603','6770','3665','3017','3711','3037','2327','2382',
    '3443','2383','6442','3661','6669','6683','3231','2303','2368','2345',
    '1303','2360','2449','6443','4989','6285','3715','3563','3653','2891',
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
    '1815','8358',
]
TW_TICKERS = [_t(c) for c in TW_CODES]

US_TICKERS = [
    'NVDA','AMD','MU','SNDK','INTC','AMAT','ASML','KLAC','LRCX','TER',
    'QCOM','TSM','PLTR','ARM','MRVL','NXPI','SNPS','MPWR','TXN','BKR',
    'URI','ON','GEV','STLD','SMCI','AVGO','META','MSFT','AAPL','AMZN',
    'GOOGL','TSLA','VRT','CRDO','ALAB','MDB','AAOI','DOCN',
    'BRK-B','BRK-A','LULU','COIN','CRM','MCD','SHOP','RDDT','DIS','KO','LIN','LITE',
    'WDC','PANW','ADI','STX','CRWD',
]

# Key stocks always pinned in summary (never hidden by category filters)
PINNED_US = ['NVDA', 'AMD', 'TSM', 'AVGO', 'MU', 'PLTR', 'SNDK', 'CRDO', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'TSLA']
PINNED_TW = ['2330.TW', '2317.TW', '2454.TW', '2308.TW', '2327.TW', '2330.TWO']


# ── Core analysis per ticker ──────────────────────────────────────────────────
def analyse(ticker, period='1y', ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES, return_reason: bool = False):
    try:
        raw = download_with_cache(ticker, period=period, ttl_hours=ttl_hours, max_retries=max_retries)
        if raw.empty:
            return (None, 'empty data') if return_reason else None
        if len(raw) < 60:
            return (None, f'not enough rows (<60): {len(raw)}') if return_reason else None
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance sometimes batches multiple tickers into one response.
            # Filter to only the requested ticker's columns before dropping the level.
            level1 = raw.columns.get_level_values(1)
            if ticker in level1:
                raw = raw.loc[:, level1 == ticker]
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
        lw = df['l'].resample('W-FRI').min().dropna()   # weekly low
        if len(cw) >= 35:
            ml_w, sig_w, mh_w = macd_hist(cw)
            w_macd      = float(ml_w.iloc[-1])
            w_macd_prev = float(ml_w.iloc[-2]) if len(ml_w) >= 2 else np.nan
            w_hist      = float(mh_w.iloc[-1])
            w_hist_prev = float(mh_w.iloc[-2]) if len(mh_w) >= 2 else np.nan
            w_close     = float(cw.iloc[-1])
            w_ma20      = float(cw.rolling(20).mean().iloc[-1])
            w_prev_low  = float(lw.iloc[-2]) if len(lw) >= 2 else np.nan
        else:
            # Not enough weekly history -> treat as unknown, not bullish by default
            w_macd = np.nan; w_macd_prev = np.nan
            w_hist = np.nan; w_hist_prev = np.nan
            w_close = np.nan; w_ma20 = np.nan; w_prev_low = np.nan

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

        # ── MACD 收腳 detection ──────────────────────────────────────────────
        macd_foot = False; gap_up = False; shrink = 0.0
        if d_hist < 0 and d_hist_prev < 0 and abs(d_hist) < abs(d_hist_prev):
            shrink = (abs(d_hist_prev) - abs(d_hist)) / (abs(d_hist_prev) + 1e-10) * 100
            if shrink >= 10:
                macd_foot = True
                # gap up: today open > yesterday high
                if len(df) >= 2:
                    if o is not None and h is not None:
                        gap_up = float(o.iloc[-1]) > float(h.iloc[-2])

        # ── Warning: weekly/monthly turning negative ──────────────────────────
        exit_warn = False
        exit_reasons = []

        # Weekly histogram flips from >0 to <0  — three-layer filter:
        #   A. relative: |new bar| >= prev_bar * HIST_FLIP_MIN_RATIO  (no tiny flips)
        #   B. absolute: |new bar| >= w_close * HIST_FLIP_MIN_PCT     (no zero-noise)
        #   C. price structure: close < 20W-MA  OR  close < prev-week low
        if (not np.isnan(w_hist_prev)) and (not np.isnan(w_hist)):
            abs_hist   = abs(w_hist)
            gate_cross = w_hist_prev > 0 and w_hist < 0
            gate_rel   = abs_hist >= w_hist_prev * HIST_FLIP_MIN_RATIO
            gate_abs   = abs_hist >= w_close * HIST_FLIP_MIN_PCT
            gate_price = ((not np.isnan(w_ma20)    and w_close < w_ma20) or
                          (not np.isnan(w_prev_low) and w_close < w_prev_low))
            if gate_cross and gate_rel and gate_abs and gate_price:
                exit_warn = True
                exit_reasons.append('週線柱狀體翻黑')

        # Weekly MACD below 0 — with minimum magnitude filter (avoids near-zero noise)
        # e.g. 玉山金 close=34, threshold=0.068; w_macd=-0.02 → NOT triggered
        if not np.isnan(w_macd) and not np.isnan(w_close):
            _w_thresh = w_close * MACD_LINE_MIN_PCT
            if w_macd < 0 and abs(w_macd) >= _w_thresh:
                exit_warn = True
                # Distinguish just-crossed (前週仍正) vs persistent (已持續負)
                if not np.isnan(w_macd_prev) and w_macd_prev > 0:
                    exit_reasons.append('週線MACD剛跌破0')
                else:
                    exit_reasons.append('週線MACD跌破0')

        # Monthly MACD below 0 — same minimum magnitude filter
        if not np.isnan(m_macd) and not np.isnan(w_close):
            _m_thresh = w_close * MACD_LINE_MIN_PCT
            if m_macd < 0 and abs(m_macd) >= _m_thresh:
                exit_warn = True
                exit_reasons.append('月線MACD跌破0')

        exit_reason = '+'.join(exit_reasons)

        price = float(c.dropna().iloc[-1])
        result = {
            'ticker':    ticker,
            'price':     price,
            'status':    status,
            'bull':      d_pos and w_pos and m_pos,
            'perfect':   not d_neg_hist and d_pos and w_pos and m_pos,
            'consol':    d_neg_hist and d_pos and w_pos and m_pos,
            'macd_foot': macd_foot,
            'gap_up':    gap_up,
            'shrink':    round(shrink, 1),
            'exit_warn': exit_warn,
            'exit_reason': exit_reason,
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


def enrich_earnings(results: list) -> None:
    """Fetch earningsGrowth for exit-warning stocks and attach earnings_mod to result dicts.
    Called once after scan(); uses 30-day cache so repeated runs are instant."""
    warn_results = [r for r in results if r.get('exit_warn')]
    if not warn_results:
        return
    with ThreadPoolExecutor(max_workers=4) as ex:
        mods = dict(zip(
            [r['ticker'] for r in warn_results],
            ex.map(lambda r: _fetch_earnings_modifier(r['ticker']), warn_results)
        ))
    for r in results:
        r['earnings_mod'] = mods.get(r['ticker'], 1.0)


def enrich_premarket(results: list, label: str) -> None:
    """For US scans, fetch pre/post-market price for every ticker and tag whether
    it's consistent with (✅一致) or diverges from (⚠️背離) the relevant trend.
    For exit-warning rows the trigger (週線/月線MACD跌破0 or 週線柱狀體翻黑) is
    always bearish, so a pre/post-market decline confirms it (✅一致); for all
    other rows the reference is today's daily MACD direction.
    No-op for TW scans (pre/post-market quotes don't apply). 15-min cache keeps
    repeated runs cheap, but a cold run still costs one .info call per ticker."""
    if '美股' not in label:
        return
    with ThreadPoolExecutor(max_workers=4) as ex:
        data = dict(zip(
            [r['ticker'] for r in results],
            ex.map(lambda r: _fetch_premarket(r['ticker']), results)
        ))
    for r in results:
        pm = data.get(r['ticker'])
        if not pm:
            continue
        trend_pos = False if r.get('exit_warn') else r['d_macd'] > 0
        if pm['pct'] > 0.05:
            tag = '✅一致' if trend_pos else '⚠️背離'
        elif pm['pct'] < -0.05:
            tag = '✅一致' if not trend_pos else '⚠️背離'
        else:
            tag = '➖持平'
        r['premarket'] = {**pm, 'tag': tag}


def _reduce_label(exit_reason: str, ticker: str = '', earnings_mod: float = 1.0) -> str:
    """Return a suggested position-reduction range.
    Manual REDUCE_OVERRIDES take highest priority; otherwise uses
    base range × earnings_mod (strong earnings → soften, declining → amplify)."""
    if ticker and ticker in REDUCE_OVERRIDES:
        return REDUCE_OVERRIDES[ticker]
    hw = '週線MACD' in exit_reason
    hm = '月線MACD' in exit_reason
    hb = '週線柱狀體翻黑' in exit_reason
    if hw and hm:
        lo, hi = _REDUCE_BASE['both']
    elif hw:
        lo, hi = _REDUCE_BASE['week']
    elif hb:
        lo, hi = _REDUCE_BASE['hist']
    elif hm:
        lo, hi = _REDUCE_BASE['month']
    else:
        return ''
    # Apply earnings modifier: round to nearest 5%
    lo = max(5,   round(lo * earnings_mod / 5) * 5)
    hi = min(100, round(hi * earnings_mod / 5) * 5)
    # Append icon so user can see the adjustment
    note = ''
    if earnings_mod <= 0.75:
        note = ' 📈強獲利'
    elif earnings_mod >= 1.25:
        note = ' 📉衰退'
    return f'【減 {lo}%~{hi}%{note}】'


def fmt_macd(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 'NA'
        return f"{float(x):+.2f}"
    except Exception:
        return 'NA'


def print_dashboard(results, label):
    def fmt_sym(ticker: str) -> str:
        nm = get_company_name(ticker, max_retries=2)
        if nm and NAME_MAXLEN and len(nm) > int(NAME_MAXLEN):
            nm = nm[: int(NAME_MAXLEN)]
        return f"{ticker} {nm}".strip()

    def pm_str(r) -> str:
        pm = r.get('premarket')
        if not pm:
            return ''
        sess = '盤前' if pm['session'] == 'pre' else '盤後'
        return f"  {sess}:${pm['price']:.2f}({pm['pct']:+.1f}%){pm['tag']}"

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
    gap     = [r for r in results if r['macd_foot'] and r['gap_up']]
    exit_   = [r for r in results if r['exit_warn']]

    # Build pinned lookup
    result_map = {r['ticker']: r for r in results}

    MAX_ROWS = 50

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
                  f"日MACD:{fmt_macd(r['d_macd'])}  週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}{pm_str(r)}")

    # ── Section 1: Bull structure ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  🟢 多頭結構完整  ({len(bull)} 支)  →  持有 / 可買")
    print(f"{'─'*65}")
    if perfect:
        print(f"\n  🟢 完美多頭 ({len(perfect)} 支) — 日線柱狀體仍正:")
        for r in sorted(perfect, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"月MACD:{fmt_macd(r['m_macd'])}  週MACD:{fmt_macd(r['w_macd'])}{pm_str(r)}")
    if consol:
        print(f"\n  ✅ 強勢整理 ({len(consol)} 支) — 日線回調但結構完整:")
        for r in sorted(consol, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"月MACD:{fmt_macd(r['m_macd'])}  週MACD:{fmt_macd(r['w_macd'])}{pm_str(r)}")

    # ── Section 1b: Dip watch (日線轉負, 週月線仍多頭) ─────────────────────────
    if dip_watch:
        print(f"\n  🔵 日線回調、週月仍多頭 ({len(dip_watch)} 支) — 逢低留意:")
        for r in sorted(dip_watch, key=lambda x: x['m_macd'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"日MACD:{fmt_macd(r['d_macd'])}  週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}{pm_str(r)}")

    # ── Section 2: MACD foot + gap ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  📍 MACD 收腳訊號  ({len(foot)} 支)  →  最佳進場時機")
    print(f"{'─'*65}")
    if gap:
        print(f"\n  ✅ 收腳 + 跳空 ({len(gap)} 支) ← 最強訊號:")
        for r in sorted(gap, key=lambda x: x['shrink'], reverse=True):
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"縮短:{r['shrink']:.0f}%  ✅跳空  {r['status']}{pm_str(r)}")
    foot_only = [r for r in foot if not r['gap_up']]
    if foot_only:
        print(f"\n  🟡 收腳 (待確認) ({len(foot_only)} 支):")
        for r in sorted(foot_only, key=lambda x: x['shrink'], reverse=True)[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"縮短:{r['shrink']:.0f}%  待明日跳空確認  {r['status']}{pm_str(r)}")
    if not foot:
        print(f"    今日無收腳訊號")

    # ── Section 3: Exit warnings ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  🔴 出場警示  ({len(exit_)} 支)  →  準備減碼")
    print(f"{'─'*65}")
    if exit_:
        for r in sorted(exit_, key=lambda x: x['w_macd'])[:MAX_ROWS]:
            sym = fmt_sym(r['ticker'])
            rlabel = _reduce_label(r['exit_reason'], r['ticker'], r.get('earnings_mod', 1.0))
            rlabel_str = f'  {rlabel}' if rlabel else ''
            print(f"    {sym:<22} ${r['price']:>8.2f}  "
                  f"⚠️  {r['exit_reason']}{rlabel_str}  "
                  f"週:{fmt_macd(r['w_macd'])}  月:{fmt_macd(r['m_macd'])}{pm_str(r)}")
    else:
        print(f"    目前無出場警示 — 持倉安全")

    print(f"\n{'═'*65}")
    print(f"  📌 傻瓜投資三原則:")
    print(f"  1. 🟢/✅ 在列 → 繼續持有，什麼都不做")
    print(f"  2. 📍 收腳+跳空 → 這是最好的買入時機")
    print(f"  3. 🔴 出場警示 → 週/月線轉負才是真正出場")
    print(f"  4. 🔵 日線回調週月仍多頭 → 可逢低分批買")
    print(f"{'═'*65}\n")


def export_to_excel(results_by_label: dict, out_path: str = None):
    """Export scan results to an Excel workbook, one sheet per dashboard
    section (重點股票 / 多頭結構 / 日線回調 / MACD收腳 / 出場警示 / 全部) per market.

    results_by_label: e.g. {'美股': r_us, '台股': r_tw}
    Returns the path written, or None if openpyxl is unavailable.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  ⚠️  未安裝 openpyxl，略過 Excel 匯出（pip install openpyxl）")
        return None

    if out_path is None:
        out_path = f"fool_dashboard_{datetime.now().strftime('%Y%m%d')}.xlsx"

    columns = ['代號', '名稱', '價格', '狀態', '日MACD', '週MACD', '月MACD',
               '收腳縮短%', '跳空', '出場理由', '減碼建議', '盤前盤後']

    def to_row(r):
        pm = r.get('premarket')
        pm_text = ''
        if pm:
            sess = '盤前' if pm['session'] == 'pre' else '盤後'
            pm_text = f"{sess} {pm['price']:.2f} ({pm['pct']:+.1f}%) {pm['tag']}"
        return {
            '代號':      r['ticker'],
            '名稱':      get_company_name(r['ticker'], max_retries=1),
            '價格':      r['price'],
            '狀態':      r['status'],
            '日MACD':    r['d_macd'],
            '週MACD':    r['w_macd'],
            '月MACD':    r['m_macd'],
            '收腳縮短%':  r['shrink'],
            '跳空':      '是' if r.get('gap_up') else '',
            '出場理由':   r.get('exit_reason', ''),
            '減碼建議':   _reduce_label(r.get('exit_reason', ''), r['ticker'], r.get('earnings_mod', 1.0)),
            '盤前盤後':   pm_text,
        }

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for label, results in results_by_label.items():
            result_map = {r['ticker']: r for r in results}
            pinned_candidates = (PINNED_US if label.startswith('美股')
                                  else PINNED_TW if label == '台股'
                                  else PINNED_US + PINNED_TW)
            pinned    = [result_map[t] for t in pinned_candidates if t in result_map]
            dip_watch = [r for r in results
                         if not r['bull']
                         and r['d_macd'] < 0
                         and not np.isnan(r['w_macd']) and r['w_macd'] > 0
                         and not np.isnan(r['m_macd']) and r['m_macd'] > 0]

            sections = {
                '重點股票': pinned,
                '多頭結構': [r for r in results if r['perfect'] or r['consol']],
                '日線回調': dip_watch,
                'MACD收腳': [r for r in results if r['macd_foot']],
                '出場警示': [r for r in results if r['exit_warn']],
                '全部':    results,
            }
            for section, rows in sections.items():
                df = pd.DataFrame([to_row(r) for r in rows], columns=columns)
                sheet_name = f"{label}_{section}"[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  💾 已匯出 Excel: {out_path}")
    return out_path


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

    parser.add_argument('--excel', action='store_true', help='Export results to an Excel file (one sheet per section)')
    parser.add_argument('--excel-out', type=str, default=None,
                         help='Excel output path (default: fool_dashboard_<date>.xlsx)')

    args = parser.parse_args()

    # Apply globals
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
        enrich_earnings(r)
        print_dashboard(r, '台股')
        _print_missing(missing)
        if args.excel:
            export_to_excel({'台股': r}, args.excel_out)

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
        enrich_earnings(r)
        enrich_premarket(r, '美股')
        print_dashboard(r, '美股')
        _print_missing(missing)
        if args.excel:
            export_to_excel({'美股': r}, args.excel_out)

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
        enrich_earnings(r_us)
        enrich_premarket(r_us, '美股 Western')
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
        enrich_earnings(r_tw)
        print_dashboard(r_tw, '台股')
        _print_missing(missing_tw)

        if args.excel:
            export_to_excel({'美股': r_us, '台股': r_tw}, args.excel_out)
