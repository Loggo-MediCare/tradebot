"""
Batch-patches all get_trading_signal_*.py files to apply growth-based sell
protection before the sell score is finalised.

Changes injected into each file:
  1. Adds `calculate_growth_score_adjustment` to the `shared_market_checks` import.
  2. Inserts a sell-score reduction block immediately before the line:
         adjusted_strength = min(sell_score / 100, 1.0)

Safe to re-run: files already patched are detected and skipped.
"""

import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Patterns ──────────────────────────────────────────────────────────────────
IMPORT_PATTERN = re.compile(
    r"(from shared_market_checks import\b[^\n]+)"
)
TICKER_PATTERN = re.compile(
    r"evaluate_fundamentals_for_sell\s*\(\s*yf\s*,\s*['\"]([A-Za-z0-9_.^-]+)['\"]"
)
STRENGTH_LINE = "        adjusted_strength = min(sell_score / 100, 1.0)"
ALREADY_PATCHED_MARKER = "calculate_growth_score_adjustment"

GROWTH_BLOCK_TEMPLATE = """\
        # 🌱 基本面成長保護: Revenue Growth > 33% 或 EPS Growth > 100% 各降低賣出強度 8 分
        _growth = calculate_growth_score_adjustment(yf, '{ticker}')
        if _growth['adjustment'] > 0 and sell_score > 0:
            sell_score = max(0, sell_score - _growth['adjustment'])
            for _gr in _growth['reasons']:
                reasons.append(f'🌱 {{_gr}}')
"""


def patch_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()

    # Skip if already patched
    if ALREADY_PATCHED_MARKER in src:
        return "SKIPPED (already patched)"

    # Must have the finalisation line
    if STRENGTH_LINE not in src:
        return "SKIPPED (no adjusted_strength line)"

    # Extract ticker
    m = TICKER_PATTERN.search(src)
    if not m:
        return "SKIPPED (ticker not found)"
    ticker = m.group(1)

    # 1. Update import line
    def add_to_import(match):
        line = match.group(1)
        if "calculate_growth_score_adjustment" in line:
            return line
        return line.rstrip() + ", calculate_growth_score_adjustment"

    new_src, n_import = IMPORT_PATTERN.subn(add_to_import, src, count=1)
    if n_import == 0:
        return "SKIPPED (import line not found)"

    # 2. Insert growth block before the strength line
    growth_block = GROWTH_BLOCK_TEMPLATE.format(ticker=ticker)
    new_src = new_src.replace(
        STRENGTH_LINE,
        growth_block + STRENGTH_LINE,
        1,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_src)
    return f"PATCHED (ticker={ticker})"


def main():
    files = [f for f in os.listdir(SCRIPT_DIR) if f.startswith("get_trading_signal_") and f.endswith(".py")]
    files.sort()

    patched = skipped = errors = 0
    for fname in files:
        path = os.path.join(SCRIPT_DIR, fname)
        try:
            result = patch_file(path)
            status = result.split()[0]
            if status == "PATCHED":
                patched += 1
            else:
                skipped += 1
            print(f"  {fname}: {result}")
        except Exception as e:
            errors += 1
            print(f"  {fname}: ERROR — {e}")

    print(f"\nDone: {patched} patched, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
