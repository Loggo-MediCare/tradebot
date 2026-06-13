"""
MU Regime 假設監控儀表板 — Assumption Watchlist Dashboard
==========================================================
核心原則：不信看起來很準的分數，只信「可被監控與拆解的假設」。

  🟡 弱訊號區    → 外部評分 (82/98) / 賣方目標價 = 歷史 regime 的投影，僅供參考
  📋 假設監控    → Demand / Supply / Monetization 三組 KPI（觸發條件 → 風險動作）
  🔴 Downside   → regime break 條件式退出（倉位開關），不用單一目標價當風險線
  🧮 Sanity     → CIO 式估值檢核（EPS / 本益比 / 估值容忍度前提）

定位說明：
  - 「score」從結論因子降級為「弱訊號」：82/98 這種分數是歷史 regime 投影，
    不能被一句「那分數怎麼算的？」打穿。
  - 賣方大幅調升目標價 = catch-up（追趕市場）+ crowding（共識擴散/擁擠交易），
    重點不是誰更懂產業，而是它怎麼改變部位擁擠度與回撤分布。
  - 1,500~1,625 這類 bull-case target 的功能是「壓力測試上限 (upside stress test)」，
    不是 baseline；直接影響倉位大小、風險預算與再平衡規則。

Usage:
  python mu_regime_dashboard.py                     # 預設掃 MU
  python mu_regime_dashboard.py --ticker MU
  python mu_regime_dashboard.py --state state.json  # 載入人工監控值（ASP/長約等）
  python mu_regime_dashboard.py --no-cache
"""
import sys, io, os, re, json, warnings, logging, argparse, time, random, threading
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ResourceWarning)
logging.getLogger('yfinance').setLevel(logging.ERROR)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime

# ── yfinance acceleration & stability (cache + retry + rate-limit) ───────────
CACHE_DIR = os.path.join(os.getcwd(), '.cache_regime')
os.makedirs(CACHE_DIR, exist_ok=True)

USE_CACHE        = True
CACHE_TTL_HOURS  = 8
MAX_RETRIES      = 3
DOWNLOAD_TIMEOUT = 20
INFO_SEMAPHORE   = threading.Semaphore(1)

# ── 弱訊號區設定（賣方評分 / 目標價，全部視為噪音較高的輔助訊號）────────────
SELLSIDE_WEAK_SIGNALS = {
    # 外部評分：歷史 regime 的投影，不是結論因子
    'external_scores': [
        {'source': '外部量化評分 A', 'score': '82/98',
         'note': '歷史 regime 投影；regime change 期間參考價值下降'},
    ],
    # 大幅調升目標價：解讀為 catch-up + crowding，而非「更準」
    'price_targets': [
        {'source': 'UBS',    'target': 1500.0,
         'note': 'catch-up：追趕市場；依賴單點假設、容錯率低'},
        {'source': 'Cantor', 'target': 1625.0,
         'note': 'crowding：共識擴散/擁擠交易；影響回撤分布'},
    ],
    # bull-case target 的「正確用途」
    'bull_case_role': '壓力測試上限 (upside stress test)，非 baseline 錨點',
}

# ── CIO Sanity Check 預設參數（可被即時資料覆蓋）─────────────────────────────
FALLBACK_TTM_EPS = 21.46     # 近四季 EPS（抓不到即時資料時的退路值）
FALLBACK_PE      = 44.0      # 對應本益比
GM_PLATFORM_FLOOR = 0.50     # 毛利率「新平台」下緣假設（低於此 = 偏離平台）
GM_BREAK_QUARTERS = 2        # 連續 N 季偏離 → 視為 regime 假設受損

# ── 人工監控值（無法從行情 API 自動取得的 KPI，由 state.json 提供）──────────
# 範例 state.json：
# {
#   "asp_below_lower_bound_quarters": 0,     ← HBM/伺服器DRAM ASP 連續低於指引下緣的季數
#   "contract_coverage_trend": "up",         ← 長約覆蓋率趨勢: up / flat / down
#   "peer_capex_pull_in": false,             ← 同業擴產/資本開支節奏是否前移
#   "yield_anomaly": false                   ← 先進製程/封裝良率異常訊號
# }
STATE_FILE = os.path.join(os.getcwd(), 'state.json')
DEFAULT_STATE = {
    'asp_below_lower_bound_quarters': None,
    'contract_coverage_trend': None,
    'peer_capex_pull_in': None,
    'yield_anomaly': None,
}

# 狀態符號
OK, WARN, BREAK, NA = '✅', '🟡', '🔴', '⚪'


def _safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))


def _cache_path(tag: str) -> str:
    return os.path.join(CACHE_DIR, f"{_safe_name(tag)}.json")


def _is_cache_fresh(path: str, ttl_hours: float) -> bool:
    try:
        age = time.time() - os.path.getmtime(path)
        return age <= float(ttl_hours) * 3600
    except Exception:
        return False


def load_state(path: str) -> dict:
    """載入人工監控值（ASP / 長約覆蓋率 / 同業擴產 / 良率），缺檔則全部 N/A。"""
    state = dict(DEFAULT_STATE)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                user = json.load(f)
            if isinstance(user, dict):
                for k in state:
                    if k in user:
                        state[k] = user[k]
        except Exception as e:
            print(f"  ⚠️  state.json 讀取失敗，改用 N/A：{e}")
    return state


# ── yfinance 基本面抓取（cache + retry + backoff）────────────────────────────
def fetch_fundamentals(ticker: str, ttl_hours: float = CACHE_TTL_HOURS,
                       max_retries: int = MAX_RETRIES) -> dict:
    """抓取價格 / EPS / 毛利率 / CapEx / FCF 季度序列。

    回傳 dict（任何欄位都可能是 None / 空 list，呼叫端必須容錯）：
      price, trailing_eps, pe,
      gm_quarters:   [(期別, 毛利率), ...] 最新在前
      capex_quarters:[(期別, capex), ...]
      fcf_quarters:  [(期別, fcf), ...]
    """
    cache_file = _cache_path(f"{ticker}__fundamentals")
    if USE_CACHE and _is_cache_fresh(cache_file, ttl_hours):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            try: os.remove(cache_file)
            except Exception: pass

    out = {'price': None, 'trailing_eps': None, 'pe': None,
           'gm_quarters': [], 'capex_quarters': [], 'fcf_quarters': []}

    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            with INFO_SEMAPHORE:
                yt = yf.Ticker(ticker)

                # 價格
                try:
                    hist = yt.history(period='5d', interval='1d', auto_adjust=True)
                    if hist is not None and not hist.empty:
                        out['price'] = float(hist['Close'].iloc[-1])
                except Exception:
                    pass

                # EPS / PE
                try:
                    info = yt.get_info()
                except Exception:
                    info = getattr(yt, 'info', None)
                if isinstance(info, dict):
                    eps = info.get('trailingEps')
                    if eps:
                        out['trailing_eps'] = float(eps)
                    if out['price'] and out['trailing_eps'] and out['trailing_eps'] > 0:
                        out['pe'] = out['price'] / out['trailing_eps']

                # 季度損益 → 毛利率
                qis = None
                for attr in ('quarterly_income_stmt', 'quarterly_financials'):
                    try:
                        qis = getattr(yt, attr)
                        if qis is not None and not qis.empty:
                            break
                    except Exception:
                        qis = None
                if qis is not None and not qis.empty:
                    rev = gp = None
                    for r in ('Total Revenue', 'TotalRevenue'):
                        if r in qis.index: rev = qis.loc[r]; break
                    for g in ('Gross Profit', 'GrossProfit'):
                        if g in qis.index: gp = qis.loc[g]; break
                    if rev is not None and gp is not None:
                        for col in qis.columns:
                            try:
                                r_, g_ = float(rev[col]), float(gp[col])
                                if r_ > 0:
                                    out['gm_quarters'].append(
                                        (str(pd.Timestamp(col).date()), g_ / r_))
                            except Exception:
                                continue

                # 季度現金流 → CapEx / FCF
                try:
                    qcf = yt.quarterly_cashflow
                except Exception:
                    qcf = None
                if qcf is not None and not qcf.empty:
                    for row_key, out_key in (('Capital Expenditure', 'capex_quarters'),
                                             ('Free Cash Flow',     'fcf_quarters')):
                        if row_key in qcf.index:
                            s = qcf.loc[row_key]
                            for col in qcf.columns:
                                try:
                                    out[out_key].append(
                                        (str(pd.Timestamp(col).date()), float(s[col])))
                                except Exception:
                                    continue

                time.sleep(0.15 + random.random() * 0.25)

            if USE_CACHE:
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(out, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            return out

        except Exception:
            backoff = min(30.0, (1.2 * (2 ** (attempt - 1))) + random.random())
            time.sleep(backoff)

    return out


# ── Assumption Watchlist：每個 KPI = 觸發條件 → 風險動作 ─────────────────────
def build_watchlist(fund: dict, state: dict) -> list:
    """產生 6 個可追蹤 KPI。回傳 list of dict:
       {cat, name, status, value, trigger, action}
    status ∈ {OK, WARN, BREAK, NA}；BREAK = regime 假設受損，連動 Downside Protocol。
    """
    kpis = []

    # ── A. 結構性需求 (Demand) ────────────────────────────────────────────
    q = state.get('asp_below_lower_bound_quarters')
    if q is None:
        st, val = NA, '未提供 (state.json)'
    elif int(q) >= 2:
        st, val = BREAK, f'連續 {q} 季低於指引下緣'
    elif int(q) == 1:
        st, val = WARN, '已有 1 季低於指引下緣'
    else:
        st, val = OK, '位於指引區間內'
    kpis.append({'cat': 'A. 需求', 'name': 'HBM/伺服器DRAM ASP 走勢',
                 'status': st, 'value': val,
                 'trigger': '連續兩季低於公司指引/市場預期下緣',
                 'action': 'regime change 可信度降級 → baseline 權重下調'})

    trend = state.get('contract_coverage_trend')
    trend_map = {'up':   (OK,    '覆蓋率上升＝現金流能見度上升'),
                 'flat': (WARN,  '覆蓋率停滯＝回到舊循環機率增加'),
                 'down': (BREAK, '覆蓋率下滑＝回到舊循環機率增加')}
    st, val = trend_map.get(str(trend).lower() if trend else '', (NA, '未提供 (state.json)'))
    kpis.append({'cat': 'A. 需求', 'name': '長約覆蓋率 (3–5 年協議出貨/產能)',
                 'status': st, 'value': val,
                 'trigger': '覆蓋率停滯或下滑',
                 'action': '現金流能見度假設降級 → 重新檢視倉位'})

    # ── B. 結構性供給 (Supply) ────────────────────────────────────────────
    pull = state.get('peer_capex_pull_in')
    if pull is None:
        st, val = NA, '未提供 (state.json)'
    elif pull:
        st, val = BREAK, '供給曲線前移，bull-case 容錯率瞬間變差'
    else:
        st, val = OK, '擴產節奏未前移'
    kpis.append({'cat': 'B. 供給', 'name': '同業擴產/CapEx 節奏（先進 DRAM/HBM）',
                 'status': st, 'value': val,
                 'trigger': '供給曲線前移（擴產提前）',
                 'action': 'bull-case 權重回收至情境性曝險'})

    yld = state.get('yield_anomaly')
    if yld is None:
        st, val = NA, '未提供 (state.json)'
    elif yld:
        st, val = BREAK, '良率異常＝毛利率/EPS 的跳變風險'
    else:
        st, val = OK, '無異常訊號'
    kpis.append({'cat': 'B. 供給', 'name': '先進製程/封裝良率異常',
                 'status': st, 'value': val,
                 'trigger': '任何良率異常訊號（良率不是慢變數）',
                 'action': '視為跳變風險 → 立即重估毛利率與 EPS 彈性'})

    # ── C. 財務落地 (Monetization) ────────────────────────────────────────
    gm = fund.get('gm_quarters') or []
    if gm:
        recent = gm[:GM_BREAK_QUARTERS + 2]
        below = 0
        for _, g in recent:
            if g < GM_PLATFORM_FLOOR: below += 1
            else: break
        latest_q, latest_gm = gm[0]
        if below >= GM_BREAK_QUARTERS:
            st = BREAK
        elif below == 1:
            st = WARN
        else:
            st = OK
        val = f'{latest_q} 毛利率 {latest_gm*100:.1f}%（平台下緣 {GM_PLATFORM_FLOOR*100:.0f}%）'
    else:
        st, val = NA, '無季度財報資料'
    kpis.append({'cat': 'C. 落地', 'name': '毛利率維持「新平台」高檔區間',
                 'status': st, 'value': val,
                 'trigger': f'連續 {GM_BREAK_QUARTERS} 季低於新平台假設下緣',
                 'action': '估值容忍度假設受損 → baseline 權重降級'})

    capex = fund.get('capex_quarters') or []
    fcf   = fund.get('fcf_quarters') or []
    if len(capex) >= 2 and len(fcf) >= 2:
        capex_up = abs(capex[0][1]) > abs(capex[1][1])
        fcf_down = fcf[0][1] < fcf[1][1]
        if capex_up and fcf_down:
            st, val = WARN, 'CapEx 走高、FCF 走低（高 CapEx 未換到更可預測的 FCF）'
        else:
            st, val = OK, 'CapEx → 產出/獲利轉換效率未見惡化'
    else:
        st, val = NA, '現金流資料不足'
    kpis.append({'cat': 'C. 落地', 'name': 'CapEx → 產出/獲利轉換效率',
                 'status': st, 'value': val,
                 'trigger': 'CapEx 走高但 EPS/FCF 可預測性未提升',
                 'action': 'CapEx 走高 ≠ 利多 → 重審資本配置假設'})

    return kpis


# ── Downside Protocol：regime break 條件式退出（倉位開關）────────────────────
def evaluate_downside(kpis: list) -> dict:
    breaks = [k for k in kpis if k['status'] == BREAK]
    warns  = [k for k in kpis if k['status'] == WARN]
    nas    = [k for k in kpis if k['status'] == NA]

    if breaks:
        verdict, icon = 'BASELINE 權重降級；bull-case 權重回收至情境性曝險', BREAK
    elif warns:
        verdict, icon = '維持 baseline，但提高監控頻率（假設出現裂縫）', WARN
    else:
        verdict, icon = '維持 baseline 權重（regime 假設成立中）', OK
    return {'breaks': breaks, 'warns': warns, 'nas': nas,
            'verdict': verdict, 'icon': icon}


# ── Dashboard 輸出 ────────────────────────────────────────────────────────────
def fmt_status(k: dict) -> str:
    return f"{k['status']} {k['name']}"


def print_dashboard(ticker: str, fund: dict, kpis: list, downside: dict):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    price = fund.get('price')
    eps   = fund.get('trailing_eps') or FALLBACK_TTM_EPS
    pe    = fund.get('pe') or (price / eps if (price and eps) else FALLBACK_PE)

    print(f"\n{'═'*65}")
    print(f"  🧭 {ticker} Regime 假設監控儀表板  ({now})")
    print(f"{'═'*65}")

    # ── Section 1: 弱訊號區（score 降級，不是結論因子）──────────────────────
    print(f"\n{'─'*65}")
    print(f"  🟡 弱訊號區  →  外部評分/目標價 = 歷史 regime 投影，僅供參考")
    print(f"{'─'*65}")
    for s in SELLSIDE_WEAK_SIGNALS['external_scores']:
        print(f"    {s['source']:<14} 評分 {s['score']:<8}  ({s['note']})")
    for t in SELLSIDE_WEAK_SIGNALS['price_targets']:
        up = ''
        if price:
            up = f"  距現價 {((t['target']/price)-1)*100:+.0f}%"
        print(f"    {t['source']:<14} 目標 ${t['target']:>7,.0f}{up}  ({t['note']})")
    print(f"    ↳ 用途定位：{SELLSIDE_WEAK_SIGNALS['bull_case_role']}")
    print(f"    ↳ 23% 上檔 ≠ 安全邊際，只是常態波動的一部分")

    # ── Section 2: Assumption Watchlist ─────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  📋 可監控假設清單 (Assumption Watchlist)  →  KPI = 觸發條件 → 風險動作")
    print(f"{'─'*65}")
    cur_cat = None
    for k in kpis:
        if k['cat'] != cur_cat:
            cur_cat = k['cat']
            print(f"\n  ◆ {cur_cat}")
        print(f"    {fmt_status(k)}")
        print(f"        現況   : {k['value']}")
        print(f"        觸發   : {k['trigger']}")
        print(f"        動作   : {k['action']}")

    # ── Section 3: Downside Protocol（不是 target price，是條件式退出）──────
    print(f"\n{'─'*65}")
    print(f"  🔴 Downside Protocol  →  以「regime 假設是否維持成立」作為倉位開關")
    print(f"{'─'*65}")
    print(f"    退出條件（任一成立即降級 baseline 權重）：")
    print(f"      (i)   長約覆蓋率下滑或未再擴張")
    print(f"      (ii)  ASP/毛利率連續偏離新平台假設")
    print(f"      (iii) 供給端擴產提前 → 供需缺口收斂速度加快")
    print(f"\n    {downside['icon']} 判定：{downside['verdict']}")
    if downside['breaks']:
        for k in downside['breaks']:
            print(f"      🔴 已觸發：{k['name']} — {k['value']}")
    if downside['warns']:
        for k in downside['warns']:
            print(f"      🟡 警戒中：{k['name']} — {k['value']}")
    if downside['nas']:
        names = '、'.join(k['name'] for k in downside['nas'])
        print(f"      ⚪ 待補資料：{names}（請更新 state.json）")
    print(f"\n    ↳ 我們不是在賭方向，而是在賭「假設是否持續成立」。")

    # ── Section 4: CIO 式 Sanity Check ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  🧮 CIO Sanity Check  →  用最少數字、講最硬的話")
    print(f"{'─'*65}")
    p_str = f"${price:,.2f}" if price else 'N/A'
    print(f"    現價 {p_str}   近四季 EPS ≈ {eps:.2f}   本益比 ≈ {pe:.0f}x")
    print(f"\n    Baseline 成立前提：")
    print(f"      市場相信 EPS 可持續性還會上修，且 multiple 不因供給釋放快速再定價")
    print(f"    Bull-case 成立前提：")
    print(f"      EPS 上修 + 「供需缺口延續年限」被市場重新錨定（1年 → 2–3年），")
    print(f"      否則 multiple 很難再無痛擴張")
    print(f"    對賣方目標價的定性：")
    print(f"      它們不是「更準」，而是「更依賴單點假設、容錯率更低」")

    # ── Section 5: 投委會結語（權重分配機制，不是哲學宣言）──────────────────
    print(f"\n{'═'*65}")
    print(f"  📌 投委會結語：")
    print(f"  我們將賣方評分視為噪音較高的輔助訊號；")
    print(f"  決策權重取決於假設是否【透明、可監控、且具容錯率】，")
    print(f"  並以情境用途（baseline vs stress test）決定資本配置大小與風險邊界。")
    print(f"{'═'*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='MU', help='標的代號 (default: MU)')
    parser.add_argument('--state',  type=str, default=STATE_FILE,
                        help='人工監控值 JSON（ASP/長約/同業擴產/良率）')

    # yfinance stability controls
    parser.add_argument('--no-cache',  action='store_true', help='Disable yfinance cache')
    parser.add_argument('--cache-ttl', type=float, default=8.0, help='Cache TTL in hours (default: 8)')
    parser.add_argument('--retries',   type=int, default=3, help='Download retries (default: 3)')
    parser.add_argument('--timeout',   type=int, default=20, help='yfinance timeout seconds (default: 20)')

    args = parser.parse_args()

    USE_CACHE        = (not args.no_cache)
    CACHE_TTL_HOURS  = float(args.cache_ttl)
    MAX_RETRIES      = int(args.retries)
    DOWNLOAD_TIMEOUT = max(5, int(args.timeout))

    ticker = args.ticker.upper()
    print(f"\n  抓取 {ticker} 基本面資料...", flush=True)
    fund  = fetch_fundamentals(ticker, ttl_hours=CACHE_TTL_HOURS, max_retries=MAX_RETRIES)
    state = load_state(args.state)
    kpis  = build_watchlist(fund, state)
    downside = evaluate_downside(kpis)
    print_dashboard(ticker, fund, kpis, downside)
