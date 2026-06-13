"""
Targeted fixes for the 73 remaining broken files:
 A) 'unexpected indent' → restore missing 'return {' before orphaned dict keys
 B) 'invalid syntax' → remove blank lines inside backslash continuations
 C) BOM → already handled by _fix_tw_news_final.py (strips BOM)
After fixing each file, re-runs the news tracker insertion.
"""
import glob, re, os, sys, io, ast

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

IMPORT_LINE = 'from tw_news_tracker import print_tavily_news_tw'
GLOBE = '\U0001f310'
BS = chr(92)


def read_file(fp):
    raw = open(fp, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    return raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')


def check_syntax(src):
    try:
        ast.parse(src)
        return True, None
    except SyntaxError as e:
        return False, (e.msg, e.lineno)


def extract_code_name(content, filepath):
    basename = os.path.basename(filepath)
    code = basename.replace('get_trading_signal_', '').replace('.py', '')
    nm = re.search(r"NAME\s*=\s*['\"]([^'\"]+)['\"]", content)
    if nm:
        return code, nm.group(1)
    n = re.search(r'\(([^\x00-\x7F\s][^\)]{1,20})\)', content[:400])
    if n:
        cand = n.group(1).strip()
        if '?' not in cand and '?' not in cand and len(cand) >= 2:
            return code, cand
    return code, code


def fix_unexpected_indent(lines, err_lineno):
    """
    Insert 'return {' before the first orphaned dict-key line.
    Orphaned dict key = line with 8 spaces indent and starts with ' (like 'ticker':).
    """
    # Find first line that looks like an orphaned dict key around err_lineno
    for i in range(max(0, err_lineno - 5), min(len(lines), err_lineno + 2)):
        l = lines[i]
        s = l.strip()
        if (l.startswith('        ') and  # 8-space indent
                (s.startswith("'") or s.startswith('"')) and
                ':' in s):
            # Insert return { before this block
            lines.insert(i, '    return {')
            # Find the matching closing } (look ahead)
            j = i + 1
            brace_depth = 1
            while j < len(lines) and brace_depth > 0:
                cl = lines[j]
                brace_depth += cl.count('{') - cl.count('}')
                j += 1
            # If no closing }, add one
            if brace_depth > 0:
                lines.insert(j, '    }')
            return lines
    return lines


def fix_backslash_blank(lines):
    """Remove blank lines that follow a backslash continuation."""
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        result.append(line)
        # If this line ends with backslash, remove following blank lines
        if line.rstrip('\n').rstrip('\r').endswith(BS):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1  # skip blank lines
            i = j  # jump to first non-blank
        else:
            i += 1
    return result


def insert_news(content, code, name):
    lines = content.split('\n')
    # Add import
    if IMPORT_LINE not in content:
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
        lines.insert(last_idx + 1 if last_idx >= 0 else 0, IMPORT_LINE)

    # Append news block
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
    return '\n'.join(lines)


def process_file(fp):
    content = read_file(fp)
    code, name = extract_code_name(content, fp)

    ok, err = check_syntax(content)
    if ok:
        # Already fixed — add news if missing
        if IMPORT_LINE not in content and 'print_tavily_news_tw' not in content:
            new = insert_news(content, code, name)
            with open(fp, 'w', encoding='utf-8', newline='\n') as f:
                f.write(new)
        return 'already-ok'

    msg, lineno = err
    lines = content.split('\n')

    if 'unexpected indent' in msg:
        lines = fix_unexpected_indent(lines, lineno - 1)
    elif 'invalid syntax' in msg or 'backslash' in msg.lower():
        lines = fix_backslash_blank(lines)
    else:
        return f'unknown-error:{msg}'

    fixed = '\n'.join(lines)
    ok2, err2 = check_syntax(fixed)
    if ok2:
        new = insert_news(fixed, code, name)
        ok3, err3 = check_syntax(new)
        if ok3:
            with open(fp, 'w', encoding='utf-8', newline='\n') as f:
                f.write(new)
            return f'fixed+news:{code}:{name}'
        else:
            with open(fp, 'w', encoding='utf-8', newline='\n') as f:
                f.write(fixed)
            return f'fixed-no-news:{err3}'
    else:
        with open(fp, 'w', encoding='utf-8', newline='\n') as f:
            f.write(fixed)
        return f'still-broken:{err2}'


def main():
    # Only process files that had syntax errors
    files = sorted(glob.glob('get_trading_signal_*.py'))
    tw_files = [f for f in files
                if re.match(r'^\d{4}', f.replace('get_trading_signal_', '').replace('.py', ''))
                and '(' not in f]

    print(f'Checking {len(tw_files)} files for remaining errors...')
    need_fix = []
    for f in tw_files:
        content = read_file(f)
        ok, _ = check_syntax(content)
        if not ok:
            need_fix.append(f)

    print(f'{len(need_fix)} files need fixing.')
    stats = {}
    for f in need_fix:
        result = process_file(f)
        key = result.split(':')[0]
        stats[key] = stats.get(key, 0) + 1
        if key not in ('fixed+news', 'already-ok'):
            print(f'  {key}: {f}: {result}')

    print(f'\nResults: {stats}')


if __name__ == '__main__':
    main()
