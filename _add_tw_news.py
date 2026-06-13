"""
Adds tw_news_tracker import + call to all Taiwan stock signal files.
Run once; skips files that already have it.
"""
import glob
import re
import os
import sys

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_code_name(filepath):
    """Extract (code, name, ticker) from file content."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    content = raw.decode('utf-8', errors='replace')

    # ticker from TICKER constant or yf.download
    ticker = None
    m = re.search(r"TICKER\s*=\s*'([^']+)'", content)
    if not m:
        m = re.search(r'TICKER\s*=\s*"([^"]+)"', content)
    if m:
        ticker = m.group(1)
    else:
        m = re.search(r"yf\.download\('([^']+)'", content)
        if not m:
            m = re.search(r'yf\.download\("([^"]+)"', content)
        if m:
            ticker = m.group(1)

    if not ticker:
        return None, None, None

    # code from filename
    basename = os.path.basename(filepath)
    code = basename.replace('get_trading_signal_', '').replace('.py', '')

    # name from docstring line 2: "XXXX.TWO (Name) ..."
    name = code  # fallback
    first_lines = content[:300]
    n = re.search(r'\(([^\x00-\x7F\s][^\)]{1,20})\)', first_lines)
    if n:
        name = n.group(1).strip()

    return code, name, ticker


def build_import_line():
    return "from tw_news_tracker import print_tavily_news_tw\n"


def build_call_block(code, name):
    return (
        "\n"
        "    # ── Tavily 即時新聞 ───────────────────────────────────────────────────────\n"
        f"    print('\\n' + '=' * 80)\n"
        f"    print('🌐 {code} {name} 即時新聞  (Tavily REST API)')\n"
        f"    print('=' * 80)\n"
        f"    print_tavily_news_tw('{code}', '{name}', max_results=5)\n"
    )


# ── insertion helpers ─────────────────────────────────────────────────────────

def insert_import(content, import_line):
    """Insert import after the last 'from X import' or 'import X' block."""
    # Find position after last import line in the top-level imports
    lines = content.split('\n')
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('from ') or stripped.startswith('import '):
            last_import_idx = i

    if last_import_idx == -1:
        return import_line + content

    lines.insert(last_import_idx + 1, import_line.rstrip('\n'))
    return '\n'.join(lines)


def find_insertion_point_in_main(content):
    """
    Find where to insert the Tavily news call in the main function body.
    Strategy (in priority order):
      1. After FinBERT/sentiment block (pattern: sentiment_result = {...})
      2. Before '# 蠟燭圖' section
      3. After technical-indicator print section (before 'return {')
      4. Just before 'return {' as fallback
    """
    lines = content.split('\n')

    # Priority 1: line after the sentiment_result neutral fallback assignment
    for i, line in enumerate(lines):
        if ("sentiment_result = {'sentiment_score': 0.0" in line or
                "sentiment_result = {" in line and 'sentiment_score' in line and '0.0' in line):
            return i + 1  # insert after this line

    # Priority 2: line before candlestick pattern comment
    for i, line in enumerate(lines):
        if '蠟燭圖' in line or 'candlestick' in line.lower():
            return i  # insert before this

    # Priority 3: before '交易信號' print section
    for i, line in enumerate(lines):
        if '交易信號' in line and 'print' in line:
            return i

    # Priority 4: before 'return {'
    for i, line in enumerate(lines):
        if line.strip().startswith('return {') or line.strip() == 'return {':
            return i

    return -1


def process_file(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    content = raw.decode('utf-8', errors='replace')

    # Skip if already patched
    if 'print_tavily_news_tw' in content or 'tw_news_tracker' in content:
        return 'already'

    code, name, ticker = extract_code_name(filepath)
    if not code:
        return 'no-ticker'

    # Skip western/non-TW stocks (should not happen but guard)
    if not re.match(r'^\d', code) and not re.match(r'^\d', code):
        return 'skip'

    # 1. Add import
    import_line = build_import_line()
    if import_line.strip() not in content:
        content = insert_import(content, import_line)

    # 2. Add call block
    call = build_call_block(code, name)
    idx = find_insertion_point_in_main(content)
    if idx == -1:
        return 'no-insertion-point'

    lines = content.split('\n')
    # Determine indentation from surrounding lines
    indent = '    '
    # Insert call lines (already indented in build_call_block)
    call_lines = call.split('\n')
    for j, cl in enumerate(call_lines):
        lines.insert(idx + j, cl)

    new_content = '\n'.join(lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return f'ok:{code}:{name}'


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    files = sorted(glob.glob('get_trading_signal_*.py'))
    tw_files = []
    for f in files:
        code = f.replace('get_trading_signal_', '').replace('.py', '')
        # Only numeric codes → Taiwan stocks
        if re.match(r'^\d{4}', code) and '(' not in code:
            tw_files.append(f)

    print(f"Processing {len(tw_files)} Taiwan stock files...")

    stats = {'ok': 0, 'already': 0, 'error': 0}
    for f in tw_files:
        try:
            result = process_file(f)
            if result == 'already':
                stats['already'] += 1
                print(f"  SKIP (already): {f}")
            elif result.startswith('ok:'):
                stats['ok'] += 1
                parts = result.split(':')
                print(f"  OK  {parts[1]} ({parts[2]}): {f}")
            else:
                stats['error'] += 1
                print(f"  WARN ({result}): {f}")
        except Exception as e:
            stats['error'] += 1
            print(f"  ERROR: {f} -> {e}")

    print(f"\nDone: {stats['ok']} patched, {stats['already']} already done, {stats['error']} issues")


if __name__ == '__main__':
    main()
