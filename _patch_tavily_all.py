"""
_patch_tavily_all.py
Inject Tavily news into every get_trading_signal_*.py that still uses FinBERT only.

Run from the trading-bot directory:
    python _patch_tavily_all.py           # actually patches files
    python _patch_tavily_all.py --dry     # preview only
"""
import re
import sys
import io
import json
from pathlib import Path

# Force UTF-8 stdout so Chinese chars don't crash the console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DRY_RUN = '--dry' in sys.argv

BASE_DIR     = Path(__file__).parent
ALIASES_FILE = BASE_DIR / 'ticker_aliases.json'

# Supplemental names for US tickers not in ticker_aliases.json
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
    'OUST': 'Ouster', 'RHM': 'Rheinmetall', 'RNMBY': 'Renault',
    'VRK': 'Veracyte', 'GOOGL': 'Alphabet',
}

# ── regex patterns ────────────────────────────────────────────────────────────

# Matches the active (non-comment-line) calculate_sentiment_score call
# The line starts with exactly 4 spaces (inside a function body)
TICKER_RE = re.compile(
    r'^    sentiment_result = calculate_sentiment_score\([\'"]([^\'"]+)[\'"]',
    re.MULTILINE,
)

# Matches the else-block's closing line
ELSE_END_RE = re.compile(
    r"        sentiment_result = \{'sentiment_score': 0\.0, 'news_count': 0, 'sentiment_label': '中性'\}"
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _is_commented(text: str, match_start: int) -> bool:
    line_start = text.rfind('\n', 0, match_start) + 1
    return text[line_start:match_start].strip().startswith('#')


def get_active_ticker(text: str):
    for m in TICKER_RE.finditer(text):
        if not _is_commented(text, m.start()):
            return m.group(1)
    return None


def get_last_active_else_end(text: str):
    pos = None
    for m in ELSE_END_RE.finditer(text):
        if not _is_commented(text, m.start()):
            pos = m.end()
    return pos


def get_import_insert_pos(text: str):
    for anchor in ('from finbert_enhanced_scoring import',
                   'from dynamic_signal_weights import',
                   'from candlestick_patterns import'):
        idx = text.find(anchor)
        if idx != -1:
            return text.find('\n', idx) + 1
    return None

# ── main ──────────────────────────────────────────────────────────────────────

aliases = json.loads(ALIASES_FILE.read_text(encoding='utf-8'))

files = sorted(BASE_DIR.glob('get_trading_signal_*.py'))

patched = skipped = errors = 0

for fpath in files:
    text = fpath.read_text(encoding='utf-8')

    # Already has Tavily (inline or via module)
    if 'tavily_news' in text or 'print_tavily_news' in text or 'TAVILY_API_KEY' in text:
        skipped += 1
        continue

    ticker = get_active_ticker(text)
    if not ticker:
        print(f'  SKIP {fpath.name}: no active FinBERT section')
        skipped += 1
        continue

    company = (aliases.get(ticker)
               or aliases.get(ticker.upper())
               or US_NAMES.get(ticker.upper(), ticker))

    call_pos = get_last_active_else_end(text)
    if call_pos is None:
        print(f'  SKIP {fpath.name}: else-block not found')
        skipped += 1
        continue

    import_pos = get_import_insert_pos(text)
    if import_pos is None:
        print(f'  SKIP {fpath.name}: import anchor not found')
        skipped += 1
        continue

    import_line = 'from tavily_news import print_tavily_news\n'
    call_block = (
        '\n\n    # ── Tavily 即時新聞 ─────────────────────────────────────────────────────\n'
        '    print("\\n" + "=" * 80)\n'
        f'    print("🌐 {company} ({ticker}) 即時新聞  (Tavily REST API)")\n'
        '    print("=" * 80)\n'
        f"    print_tavily_news('{ticker}', '{company}', max_results=5)\n"
    )

    if DRY_RUN:
        print(f'  DRY  {fpath.name}  ticker={ticker}  company={company}')
        patched += 1
        continue

    # Both positions are in the original `text`; import_pos < call_pos always
    new_text = (
        text[:import_pos]
        + import_line
        + text[import_pos:call_pos]
        + call_block
        + text[call_pos:]
    )

    try:
        fpath.write_text(new_text, encoding='utf-8')
        print(f'  OK   {fpath.name}  ({ticker} / {company})')
        patched += 1
    except Exception as e:
        print(f'  ERR  {fpath.name}: {e}')
        errors += 1

print(f'\n{"DRY RUN — " if DRY_RUN else ""}Done: {patched} patched, {skipped} skipped, {errors} errors')
