"""
Final comprehensive fix:
1. Read file with proper encoding (handle BOM, CRLF)
2. Collapse excessive blank lines (max 2 consecutive)
3. Strip ALL tw_news_tracker code
4. Verify clean syntax; if not clean, skip adding news code
5. Add import at module level + append __main__ news block
"""
import glob
import re
import os
import sys
import io
import ast

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

IMPORT_LINE = 'from tw_news_tracker import print_tavily_news_tw'
GLOBE = '\U0001f310'


def read_file(filepath):
    """Read file with BOM handling and CRLF normalization."""
    raw = open(filepath, 'rb').read()
    # Strip UTF-8 BOM if present
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    content = raw.decode('utf-8', errors='replace')
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    return content


def collapse_blank_lines(lines, max_consecutive=2):
    """Replace runs of blank lines longer than max_consecutive with max_consecutive."""
    result = []
    blank_count = 0
    for line in lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= max_consecutive:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return result


def strip_news_code(lines):
    """Remove ALL tw_news_tracker related lines."""
    clean = []
    skip = 0
    for line in lines:
        if skip > 0:
            skip -= 1
            continue
        s = line.strip()
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
        # Remove stray globe print lines
        if GLOBE in line and 'print(' in line:
            continue
        # Remove the second __main__ block we add (to allow re-running safely)
        if s == "if __name__ == '__main__':":
            # peek: if next non-blank line has print_tavily_news_tw, skip whole block
            pass  # handled separately below
        clean.append(line)

    # Also remove appended __main__ blocks with news (check from end)
    # Find the last '# ── Tavily 即時新聞' section and remove from there
    result = []
    i = len(clean)
    # Walk from end, find last news-related __main__
    last_news_main = -1
    for idx in range(len(clean) - 1, -1, -1):
        if '# ── Tavily 即時新聞' in clean[idx]:
            last_news_main = idx
            break
    if last_news_main >= 0:
        result = clean[:last_news_main]
        # Strip trailing blank lines
        while result and not result[-1].strip():
            result.pop()
    else:
        result = clean

    return result


def extract_code_name(content, filepath):
    basename = os.path.basename(filepath)
    code = basename.replace('get_trading_signal_', '').replace('.py', '')
    # NAME constant takes priority
    nm = re.search(r"NAME\s*=\s*['\"]([^'\"]+)['\"]", content)
    if nm:
        return code, nm.group(1)
    # Docstring line 2
    n = re.search(r'\(([^\x00-\x7F\s][^\)]{1,20})\)', content[:400])
    if n:
        cand = n.group(1).strip()
        if '?' not in cand and '�' not in cand and len(cand) >= 2:
            return code, cand
    return code, code


def insert_import(lines):
    """Insert import after last module-level (non-indented) from/import line."""
    last_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if line and line[0] not in (' ', '\t', '#', '"', "'"):
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
        f"    print('{GLOBE} {code} {name} 即時新聞  (Tavily REST API)')",
        f"    print('=' * 80)",
        f"    print_tavily_news_tw('{code}', '{name}', max_results=5)",
    ]
    lines.extend(block)
    return lines


def check_syntax(source):
    try:
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f'{e.msg} at line {e.lineno}'


def process_file(filepath):
    content = read_file(filepath)
    code, name = extract_code_name(content, filepath)
    lines = content.split('\n')

    # Collapse excessive blank lines first
    lines = collapse_blank_lines(lines, max_consecutive=2)

    # Strip all existing news code
    lines = strip_news_code(lines)

    # Check syntax without news code
    base_source = '\n'.join(lines)
    ok, err = check_syntax(base_source)
    if not ok:
        # Write at least a clean version without news code
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(base_source)
        return f'pre-existing-error:{code}:{err}'

    # Add import at module level
    if not any(l.strip() == IMPORT_LINE for l in lines):
        lines = insert_import(lines)

    # Append news block
    lines = append_news_block(lines, code, name)

    new_content = '\n'.join(lines)
    ok2, err2 = check_syntax(new_content)

    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)

    if ok2:
        return f'ok:{code}:{name}'
    else:
        return f'final-error:{code}:{err2}'


def main():
    files = sorted(glob.glob('get_trading_signal_*.py'))
    tw_files = [f for f in files
                if re.match(r'^\d{4}', f.replace('get_trading_signal_', '').replace('.py', ''))
                and '(' not in f]

    print(f'Processing {len(tw_files)} files...')
    stats = {'ok': 0, 'pre_err': 0, 'final_err': 0}
    errors = []
    pre_errors = []

    for f in tw_files:
        try:
            result = process_file(f)
            if result.startswith('ok:'):
                stats['ok'] += 1
            elif result.startswith('pre-existing-error:'):
                stats['pre_err'] += 1
                pre_errors.append((f, result[19:]))
            else:
                stats['final_err'] += 1
                errors.append((f, result))
        except Exception as e:
            stats['final_err'] += 1
            errors.append((f, repr(e)[:100]))

    print(f'\nResults: {stats["ok"]} OK, '
          f'{stats["pre_err"]} pre-existing errors (news skipped), '
          f'{stats["final_err"]} unexpected errors')

    if pre_errors:
        print(f'\nPre-existing syntax errors ({len(pre_errors)}):')
        for f, e in pre_errors[:20]:
            print(f'  {f}: {e}')

    if errors:
        print(f'\nUnexpected errors ({len(errors)}):')
        for f, e in errors[:20]:
            print(f'  {f}: {e}')


if __name__ == '__main__':
    main()
