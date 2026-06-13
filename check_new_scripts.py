import ast, re, os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

scripts = ['6147','3624','8455','6924','3577','8374','2359','3236','6204',
           '3024','6432','3609','8299','3581','3265','3714','2340','1773']
for code in scripts:
    fname = f'get_trading_signal_{code}.py'
    content = open(fname, encoding='utf-8').read()
    try:
        ast.parse(content)
        syn = 'OK'
    except SyntaxError as e:
        syn = f'ERR line {e.lineno}'
    m = re.search(r'model_filename\s*=\s*["\']([^"\']+)["\']', content)
    model = m.group(1) if m else '?'
    exists = 'OK' if os.path.exists(f'{model}.zip') else 'MISSING'
    print(f'{code}  {syn}  model={model}  zip={exists}')
