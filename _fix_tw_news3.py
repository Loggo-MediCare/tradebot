"""
Definitive fix for tw_news_tracker injection across all TW signal files.
Key improvements:
  - Strips carriage returns (CRLF -> LF) to fix accumulated \\r\\r issues
  - Removes ALL tw_news_tracker insertions cleanly
  - Adds import at true module level
  - Appends second __main__ block at end of file
  - Writes with clean LF line endings
"""
import glob
import re
import os
import sys
import io
import ast

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

IMPORT_LINE = 'from tw_news_tracker import print_tavily_news_tw'


def extract_code_name(content, filepath):
    basename = os.path.basename(filepath)
    code = basename.replace('get_trading_signal_', '').replace('.py', '')

    # NAME constant takes priority
    nm = re.search(r"NAME\s*=\s*['\"]([^'\"]+)['\"]", content)
    if nm:
        return code, nm.group(1)

    # Docstring line 2: "XXXX.TW (Name) ..."
    n = re.search(r'\(([^\x00-\x7F\s][^\)]{1,20})\)', content[:400])
    if n:
        cand = n.group(1).strip()
        if '?' not in cand and len(cand) >= 2:
            return code, cand

    return code, code  # fallback


def strip_news_code(lines):
    """Remove ALL tw_news_tracker related lines, including call blocks."""
    clean = []
    skip = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        i += 1

        if skip > 0:
            skip -= 1
            continue

        # Remove import
        if s == IMPORT_LINE:
            continue

        # Remove call block: marker + next 4 lines
        if '# ── Tavily 即時新聞' in line:
            skip = 4
            continue

        # Remove stray print_tavily_news_tw calls
        if 'print_tavily_news_tw(' in line and 'def ' not in line and 'import' not in line:
            continue

        # Remove stray globe print lines from old blocks
        if "print('\U0001f310" in line or 'print("\U0001f310' in line:
            continue

        clean.append(line)
    return clean


def insert_import(lines):
    """Insert import after last module-level (non-indented) from/import line."""
    last_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s and line[0] not in (' ', '\t', '#', '"', "'", '\r'):
            if s.startswith('from ') or s.startswith('import '):
                last_idx = i

    if last_idx == -1:
        for i, line in enumerate(lines):
            if 'os.chdir' in line:
                last_idx = i
                break

    insert_at = last_idx + 1 if last_idx >= 0 else 0
    lines.insert(insert_at, IMPORT_LINE)
    return lines


def append_news_block(lines, code, name):
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


def check_syntax(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def process_file(filepath):
    raw = open(filepath, 'rb').read()
    # Normalise to LF, strip accumulated \r
    content = raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')

    code, name = extract_code_name(content, filepath)

    lines = content.split('\n')

    # Strip all existing news code
    lines = strip_news_code(lines)

    # Add import at module level
    if not any(l.strip() == IMPORT_LINE for l in lines):
        lines = insert_import(lines)

    # Append news block
    lines = append_news_block(lines, code, name)

    new_content = '\n'.join(lines)

    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)

    ok, err = check_syntax(filepath)
    if ok:
        return f'ok:{code}:{name}'
    else:
        return f'syntax-error:{err[:120]}'


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
            errors.append((f, repr(e)[:100]))

    print(f"\nDone: {ok} OK, {err} errors")
    for f, e in errors[:30]:
        print(f"  ERR: {f}: {e}")


if __name__ == '__main__':
    main()
