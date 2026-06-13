"""
_patch_finbert_tavily_v2.py
Handle the 64 files skipped by the first patcher:
  Case A  — DQN/Keras files (have TICKER_NAME = '...')      → add FinBERT + Tavily before footer
  Case B  — English-FinBERT files (have "No news available") → add Tavily after no-news block
  Case C  — Remaining minimal PPO / other files              → add FinBERT + Tavily before footer

Run:
    python _patch_finbert_tavily_v2.py         # patch
    python _patch_finbert_tavily_v2.py --dry   # preview
"""
import re
import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DRY_RUN  = '--dry' in sys.argv
BASE_DIR = Path(__file__).parent
ALIASES  = json.loads((BASE_DIR / 'ticker_aliases.json').read_text(encoding='utf-8'))

US_NAMES = {
    'NVDA': 'NVIDIA', 'AMD': 'AMD', 'MU': 'Micron', 'SNDK': 'SanDisk',
    'INTC': 'Intel', 'AVGO': 'Broadcom', 'AAPL': 'Apple', 'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet', 'GOOG': 'Alphabet', 'TSLA': 'Tesla', 'AMZN': 'Amazon',
    'META': 'Meta', 'QCOM': 'Qualcomm', 'TXN': 'Texas Instruments',
    'AMAT': 'Applied Materials', 'LRCX': 'Lam Research', 'KLAC': 'KLA',
    'ASML': 'ASML', 'ARM': 'Arm Holdings', 'MRVL': 'Marvell', 'NXPI': 'NXP',
    'ON': 'ON Semiconductor', 'CRDO': 'Credo Technology', 'ALAB': 'Astera Labs',
    'PLTR': 'Palantir', 'TSM': '台積電ADR', 'UMC': '聯電ADR', 'ASX': '日月光ADR',
    'SMCI': 'Super Micro', 'GEV': 'GE Vernova', 'VRT': 'Vertiv',
    'URI': 'United Rentals', 'STLD': 'Steel Dynamics', 'MPWR': 'Monolithic Power',
    'BKR': 'Baker Hughes', 'SNPS': 'Synopsys', 'TER': 'Teradyne',
    'ORCL': 'Oracle', 'IONQ': 'IonQ', 'OKLO': 'Oklo', 'RKLB': 'Rocket Lab',
    'APLD': 'Applied Digital', 'AVAV': 'AeroVironment', 'AEVA': 'Aeva',
    'ONDS': 'Ondas', 'NAT': 'Nordic American Tankers', 'HTGC': 'Hercules Capital',
    'ETN': 'Eaton', 'STX': 'Seagate', 'WDC': 'Western Digital',
    'AMKR': 'Amkor', 'MCHP': 'Microchip', 'OMER': 'Omeros',
    'OUST': 'Ouster', 'RHM': 'Rheinmetall', 'RNMBY': 'Renault', 'VRK': 'Veracyte',
}

# ── regex helpers ─────────────────────────────────────────────────────────────

TICKER_CONST_RE    = re.compile(r"^TICKER\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
TICKER_NAME_RE     = re.compile(r"^TICKER_NAME\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)

# DQN/minimal footer anchors (we insert BEFORE these)
FOOTER_PATTERNS = [
    '    print("\\n" + "=" * 80)\n    print("本信号由 AI 模型生成，仅供参考，不构成投资建议")',
    "    print('\\n' + '=' * 80)\n    print('本信号由 AI 模型生成，仅供参考，不构成投资建议')",
    '    print("\\n" + "=" * 60)\n    print("⚠️ 风险提示: 本信号仅供参考,不构成投资建议")',
    "    print('\\n' + '=' * 60)\n    print('⚠️ 风险提示: 本信号仅供参考,不构成投资建议')",
    '    print("\\n" + "=" * 80)\n    print("⚠️  风险提示")',
    "    print('\\n' + '=' * 80)\n    print('⚠️  风险提示')",
]

# English-FinBERT else-block end (no sentiment_label key)
ENG_ELSE_RE = re.compile(
    r"        sentiment_result = \{'sentiment_score': 0\.0, 'news_count': 0\}"
)

# ── company-name lookup ───────────────────────────────────────────────────────

def lookup_company(ticker: str, ticker_name: str = '') -> str:
    if ticker_name and ticker_name != ticker:
        return ticker_name
    return (ALIASES.get(ticker)
            or ALIASES.get(ticker.upper())
            or US_NAMES.get(ticker.upper(), ticker))


# ── block builders ────────────────────────────────────────────────────────────

def finbert_block(ticker: str) -> str:
    return (
        '\n    # ── FinBERT 情緒分析 ────────────────────────────────────────────────────\n'
        '    print("\\n" + "=" * 80)\n'
        '    print("📰 市场情绪分析 (FinBERT NLP Engine)")\n'
        '    print("=" * 80)\n'
        '    from finbert_enhanced_scoring import calculate_sentiment_score, format_sentiment_output\n'
        f"    sentiment_result = calculate_sentiment_score('{ticker}', verbose=False)\n"
        '    if sentiment_result and sentiment_result[\'news_count\'] > 0:\n'
        '        print(format_sentiment_output(sentiment_result))\n'
        '    else:\n'
        '        print("⚠️  未找到相关新闻，情绪分析不可用")\n'
        "        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}\n"
    )


def tavily_block(ticker: str, company: str) -> str:
    return (
        '\n    # ── Tavily 即時新聞 ─────────────────────────────────────────────────────\n'
        '    print("\\n" + "=" * 80)\n'
        f'    print("🌐 {company} ({ticker}) 即時新聞  (Tavily REST API)")\n'
        '    print("=" * 80)\n'
        f"    print_tavily_news('{ticker}', '{company}', max_results=5)\n"
    )


def tavily_import_line() -> str:
    return 'from tavily_news import print_tavily_news\n'


# ── insert-position helpers ───────────────────────────────────────────────────

def find_footer_pos(text: str):
    """Return the char position of the first matching footer print."""
    for pat in FOOTER_PATTERNS:
        idx = text.find(pat)
        if idx != -1:
            return idx
    return None


def find_import_end(text: str) -> int:
    """Return pos just after 'from chart_visualizer import' or similar anchor."""
    for anchor in ('from chart_visualizer import',
                   'from model_accuracy_tracker import',
                   'from finbert_enhanced_scoring import',
                   'from dynamic_signal_weights import',
                   'import warnings'):
        idx = text.find(anchor)
        if idx != -1:
            return text.find('\n', idx) + 1
    return -1


# ── per-file patcher ──────────────────────────────────────────────────────────

def patch_case_b(text: str, ticker: str, company: str) -> str:
    """English-FinBERT: insert Tavily after 'No news available' else-block."""
    m = ENG_ELSE_RE.search(text)
    if not m:
        return None
    call_pos   = m.end()
    import_pos = find_import_end(text)
    if import_pos < 0:
        return None
    return (
        text[:import_pos]
        + tavily_import_line()
        + text[import_pos:call_pos]
        + tavily_block(ticker, company)
        + text[call_pos:]
    )


def patch_case_ac(text: str, ticker: str, company: str) -> str:
    """DQN / minimal: insert FinBERT + Tavily before the risk-warning footer."""
    footer_pos = find_footer_pos(text)
    if footer_pos is None:
        return None
    import_pos = find_import_end(text)
    if import_pos < 0:
        return None
    insertion = finbert_block(ticker) + tavily_block(ticker, company)
    return (
        text[:import_pos]
        + tavily_import_line()
        + text[import_pos:footer_pos]
        + insertion
        + text[footer_pos:]
    )


# ── main loop ─────────────────────────────────────────────────────────────────

patched = skipped = errors = 0

for fpath in sorted(BASE_DIR.glob('get_trading_signal_*.py')):
    text = fpath.read_text(encoding='utf-8')

    # Already handled
    if 'tavily_news' in text or 'print_tavily_news' in text or 'TAVILY_API_KEY' in text:
        skipped += 1
        continue

    # Also skip if already has active FinBERT (handled by first patcher)
    if 'calculate_sentiment_score' in text and '未找到相关新闻，情绪分析不可用' in text:
        skipped += 1
        continue

    # ── extract ticker ────────────────────────────────────────────────────────
    m_ticker = TICKER_CONST_RE.search(text)
    if not m_ticker:
        print(f'  SKIP {fpath.name}: no TICKER constant found')
        skipped += 1
        continue
    ticker = m_ticker.group(1)

    m_name = TICKER_NAME_RE.search(text)
    ticker_name = m_name.group(1) if m_name else ''
    company = lookup_company(ticker, ticker_name)

    # ── detect case and patch ─────────────────────────────────────────────────
    if 'No news available' in text:
        new_text = patch_case_b(text, ticker, company)
        case = 'B (English FinBERT)'
    else:
        new_text = patch_case_ac(text, ticker, company)
        case = 'A/C (DQN / minimal)'

    if new_text is None:
        print(f'  SKIP {fpath.name}: insertion point not found  [{case}]')
        skipped += 1
        continue

    if DRY_RUN:
        print(f'  DRY  {fpath.name}  {case}  ticker={ticker}  company={company}')
        patched += 1
        continue

    try:
        fpath.write_text(new_text, encoding='utf-8')
        print(f'  OK   {fpath.name}  ({ticker} / {company})')
        patched += 1
    except Exception as e:
        print(f'  ERR  {fpath.name}: {e}')
        errors += 1

print(f'\n{"DRY RUN — " if DRY_RUN else ""}Done: {patched} patched, {skipped} skipped, {errors} errors')
