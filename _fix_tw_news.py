"""
Fix the tw_news_tracker import that was incorrectly inserted mid-function.
1. Remove the bad import line (and associated call block if duplicated).
2. Re-insert import at true module level (no indentation).
3. Re-insert call block at the correct place inside the function.
Run once.
"""
import glob
import re
import os
import sys
import io
import py_compile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

IMPORT_LINE = 'from tw_news_tracker import print_tavily_news_tw'
CALL_MARKER = '# ── Tavily 即時新聞 ─'


def extract_code_name(filepath):
    raw = open(filepath, 'rb').read()
    content = raw.decode('utf-8', errors='replace')

    # code from filename
    basename = os.path.basename(filepath)
    code = basename.replace('get_trading_signal_', '').replace('.py', '')

    # ticker
    ticker = None
    m = re.search(r"TICKER\s*=\s*'([^']+)'", content)
    if not m: m = re.search(r'TICKER\s*=\s*"([^"]+)"', content)
    if m:
        ticker = m.group(1)
    else:
        m = re.search(r"yf\.download\('([^']+)'", content)
        if not m: m = re.search(r'yf\.download\("([^"]+)"', content)
        if m: ticker = m.group(1)

    # name from docstring
    name = code
    n = re.search(r'\(([^\x00-\x7F\s][^\)]{1,20})\)', content[:300])
    if n:
        name = n.group(1).strip()

    return code, name


def build_call_lines(code, name, indent='    '):
    return [
        '',
        f'{indent}# ── Tavily 即時新聞 ───────────────────────────────────────────────────────',
        f"{indent}print('\\n' + '=' * 80)",
        f"{indent}print('\U0001f310 {code} {name} 即時新語  (Tavily REST API)')",
        f"{indent}print('=' * 80)",
        f"{indent}print_tavily_news_tw('{code}', '{name}', max_results=5)",
    ]


def fix_file(filepath):
    raw = open(filepath, 'rb').read()
    content = raw.decode('utf-8', errors='replace')
    lines = content.split('\n')

    code, name = extract_code_name(filepath)

    # ── Step 1: Remove ALL occurrences of the import line ──
    new_lines = [l for l in lines if l.strip() != IMPORT_LINE]

    # ── Step 2: Remove old call block (if any) ──
    cleaned = []
    i = 0
    while i < len(new_lines):
        l = new_lines[i]
        if CALL_MARKER in l:
            # skip this line + next 4 (print + print + print + print_tavily_news_tw call)
            i += 5
            continue
        cleaned.append(l)
        i += 1
    new_lines = cleaned

    # ── Step 3: Insert import at true module level ──
    # Find last non-indented import line
    last_module_import = -1
    for idx, line in enumerate(new_lines):
        stripped = line.strip()
        if not line.startswith(' ') and not line.startswith('\t'):
            if stripped.startswith('from ') or stripped.startswith('import '):
                last_module_import = idx

    if last_module_import == -1:
        # Fallback: insert after os.chdir line
        for idx, line in enumerate(new_lines):
            if 'os.chdir' in line:
                last_module_import = idx
                break

    insert_pos = last_module_import + 1
    new_lines.insert(insert_pos, IMPORT_LINE)

    # ── Step 4: Insert call block inside function ──
    # Find insertion point: after sentiment fallback, before candlestick section
    insertion_idx = -1
    for idx, line in enumerate(new_lines):
        if ("sentiment_result = {'sentiment_score': 0.0" in line or
                ("sentiment_result = {" in line and 'sentiment_score' in line and '0.0' in line)):
            insertion_idx = idx + 1
            break

    if insertion_idx == -1:
        for idx, line in enumerate(new_lines):
            if '蠟燭圖' in line or 'candlestick' in line.lower():
                insertion_idx = idx
                break

    if insertion_idx == -1:
        for idx, line in enumerate(new_lines):
            if line.strip().startswith('return {') or line.strip() == 'return {':
                insertion_idx = idx
                break

    if insertion_idx == -1:
        return 'no-insertion-point'

    # Detect indentation from surrounding lines
    indent = '    '
    for check_idx in [insertion_idx - 1, insertion_idx]:
        if 0 <= check_idx < len(new_lines):
            l = new_lines[check_idx]
            if l and l[0] == ' ':
                indent = ' ' * (len(l) - len(l.lstrip()))
                break

    call_block = build_call_lines(code, name, indent)
    for j, cl in enumerate(call_block):
        new_lines.insert(insertion_idx + j, cl)

    new_content = '\n'.join(new_lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # Verify syntax
    try:
        py_compile.compile(filepath, doraise=True)
        return f'ok:{code}:{name}'
    except py_compile.PyCompileError as e:
        return f'syntax-error:{e}'


def main():
    files = sorted(glob.glob('get_trading_signal_*.py'))
    tw_files = [f for f in files
                if re.match(r'^\d{4}', f.replace('get_trading_signal_', '').replace('.py', ''))
                and '(' not in f]

    print(f"Fixing {len(tw_files)} files...")
    stats = {'ok': 0, 'error': 0}
    errors = []

    for f in tw_files:
        result = fix_file(f)
        if result.startswith('ok:'):
            stats['ok'] += 1
        else:
            stats['error'] += 1
            errors.append((f, result))

    print(f"\nDone: {stats['ok']} fixed OK, {stats['error']} errors")
    for e in errors[:20]:
        print(f"  ERROR: {e[0]} -> {e[1][:100]}")


if __name__ == '__main__':
    main()
