import subprocess, sys

codes = ['7610','7788','1717','3013','3413','1425','2442','3293','3360','3374',
         '3630','3690','3706','3707','4167','4541','4973','5274','6104','6265',
         '6423','6485','6603','6829','6949','7728','8038','8074','8271','8292',
         '8377','8431','8450']

for c in codes:
    f = f'get_trading_signal_{c}.py'
    try:
        p = subprocess.run([sys.executable, f], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=120)
        out = p.stdout + p.stderr
        lines = [l for l in out.splitlines() if l.strip()]
        # find last traceback-relevant line
        last = lines[-1] if lines else '(no output)'
        # try to find the exception type line
        exc_line = ''
        for l in reversed(lines):
            if any(k in l for k in ['Error', 'Exception', 'error']):
                exc_line = l
                break
        print(f'{c}: rc={p.returncode} | exc={exc_line[:160]}')
    except subprocess.TimeoutExpired:
        print(f'{c}: TIMEOUT')
    except Exception as e:
        print(f'{c}: RUNNER_ERROR {e}')
