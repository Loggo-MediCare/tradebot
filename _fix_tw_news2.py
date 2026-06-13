"""
Clean fix for tw_news_tracker injection.
Strategy:
  1. Strip ALL previously inserted tw_news_tracker code (import + call blocks).
  2. Add import at module level (after first real import block).
  3. Append a standalone `if __name__ == '__main__': print_tavily_news_tw(...)` at end of file.
     This is the safest approach — it never touches function bodies.
"""
import glob
import re
import os
import sys
import io
import py_compile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

IMPORT_LINE  = 'from tw_news_tracker import print_tavily_news_tw'
CALL_MARKERS = [
    '# ── Tavily 即時新聞',
    'print_tavily_news_tw(',
    "print('\U0001f310",           # 🌐 globe emoji line
    "print('\\n' + '=' * 80)\n",  # too generic, handled below
]


def extract_code_name(filepath):
    raw = open(filepath, 'rb').read()
    content = raw.decode('utf-8', errors='replace')
    basename = os.path.basename(filepath)
    code = basename.replace('get_trading_signal_', '').replace('.py', '')

    # name from docstring line 2: "XXXX.TW (Name) ..."
    name = code
    n = re.search(r'\(([^\x00-\x7F\s][^\)]{1,20})\)', content[:400])
    if n:
        cand = n.group(1).strip()
        # Exclude replacement chars
        if '�' not in cand and len(cand) >= 2:
            name = cand

    # Also try NAME constant
    nm = re.search(r"NAME\s*=\s*['\"]([^'\"]+)['\"]", content)
    if nm:
        name = nm.group(1)

    return code, name


def strip_all_news_code(lines):
    """Remove ALL lines related to tw_news_tracker insertions."""
    cleaned = []
    skip_next = 0
    for i, line in enumerate(lines):
        if skip_next > 0:
            skip_next -= 1
            continue
        stripped = line.strip()
        # Remove import line
        if stripped == IMPORT_LINE:
            continue
        # Remove call block marker + next 5 lines
        if '# ── Tavily 即時新聞' in line:
            skip_next = 5
            continue
        # Remove stray print_tavily_news_tw calls
        if 'print_tavily_news_tw(' in line and 'def ' not in line and 'import' not in line:
            continue
        cleaned.append(line)
    return cleaned


def insert_import_at_module_level(lines):
    """Insert import after last non-indented 'from/import' line."""
    last_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        # Must be non-indented (module level)
        if line and line[0] not in (' ', '\t', '#', '"', "'"):
            if s.startswith('from ') or s.startswith('import '):
                last_idx = i
    if last_idx == -1:
        # Fallback: after os.chdir
        for i, line in enumerate(lines):
            if 'os.chdir' in line:
                last_idx = i
                break
    if last_idx == -1:
        last_idx = 0
    lines.insert(last_idx + 1, IMPORT_LINE)
    return lines


def append_news_block(lines, code, name):
    """Append a second __main__ block at end of file."""
    block = [
        '',
        '# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────',
        "if __name__ == '__main__':",
        f"    print('\\n' + '=' * 80)",
        f"    print('\U0001f310 {code} {name} 即時新聞  (Tavily REST API)')",
        f"    print('=' * 80)",
        f"    print_tavily_news_tw('{code}', '{name}', max_results=5)",
    ]
    lines.extend(block)
    return lines


def process_file(filepath):
    raw = open(filepath, 'rb').read()
    content = raw.decode('utf-8', errors='replace')
    lines = content.split('\n')

    code, name = extract_code_name(filepath)

    # 1. Strip all existing tw_news_tracker code
    lines = strip_all_news_code(lines)

    # 2. Add import at module level
    if IMPORT_LINE not in '\n'.join(lines):
        lines = insert_import_at_module_level(lines)

    # 3. Append news block at end
    lines = append_news_block(lines, code, name)

    new_content = '\n'.join(lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # Verify
    try:
        py_compile.compile(filepath, doraise=True)
        return f'ok:{code}:{name}'
    except py_compile.PyCompileError as e:
        return f'syntax-error:{str(e)[:100]}'


def main():
    files = sorted(glob.glob('get_trading_signal_*.py'))
    tw_files = [f for f in files
                if re.match(r'^\d{4}', f.replace('get_trading_signal_', '').replace('.py', ''))
                and '(' not in f]

    print(f"Processing {len(tw_files)} files...")
    ok = err = 0
    errors = []

    for f in tw_files:
        try:
            result = process_file(f)
            if result.startswith('ok:'):
                ok += 1
            else:
                err += 1
                errors.append((f, result))
        except Exception as e:
            err += 1
            errors.append((f, str(e)[:100]))

    print(f"\nDone: {ok} OK, {err} errors")
    for f, e in errors[:30]:
        print(f"  ERR: {f}: {e}")


if __name__ == '__main__':
    main()
