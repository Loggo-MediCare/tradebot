import glob, re, sys
sys.stdout.reconfigure(encoding='utf-8')

txt = open('run_all_local_tw_to_excel.py', encoding='utf-8', errors='ignore').read()
listed_files = set(re.findall(r"'file':\s*'(get_trading_signal_[^']+\.py)'", txt))

tw_not_listed = sorted([f for f in glob.glob('get_trading_signal_*.py')
    if f not in listed_files
    and re.match(r'get_trading_signal_\d{4,5}\.py$', f)
    and '(2)' not in f and '(3)' not in f and '(4)' not in f
    and 'bak' not in f and 'backup' not in f.lower()])

print(f"Count: {len(tw_not_listed)}")
results = []
for f in tw_not_listed:
    code = re.search(r'get_trading_signal_(\w+)\.py', f).group(1)
    try:
        s = open(f, encoding='utf-8', errors='ignore').read()[:600]
        # Pattern: code (name) in header
        m = re.search(code + r'\s*[\(\（]([^\)\）\n]{2,15})[\)\）]', s)
        if not m:
            m = re.search(r"NAME\s*=\s*'([^']{1,20})'", s)
        if not m:
            m = re.search(r'NAME\s*=\s*"([^"]{1,20})"', s)
        name = m.group(1).strip() if m else ''
        if not name or name in ('台股', '交易', '信号'):
            ch = re.findall(r'[一-鿿㐀-䶿]{2,10}', s[:400])
            for c in ch:
                if c not in ('台股', '交易', '信号', '生成', '使用', '训练', '模型', '策略', '技术', '指标'):
                    name = c
                    break
        if not name:
            name = code
    except:
        name = code
    results.append((code, name))

for code, name in results:
    print(f"  {{'file': 'get_trading_signal_{code}.py', 'name': '{code} {name}'}},")
