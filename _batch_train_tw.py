"""Sequential batch trainer - writes each ticker to a temp .py file to avoid -c crash."""
import sys, io, subprocess, os, re, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# All unique TW stocks requested across all sessions (excludes already-trained: 1717, 2344, 2442, 3135)
TICKERS = sorted(set([
    # batch 1
    '3665','5475','6223','6531','6683','3481','6166','6446','3167','6217','4919','9933',
    '6443','6457','3209','3583','4142','6592','2912','2303','2492','3498','6831','4966',
    # batch 2
    '6344','6426','6574','2395','2409',
    # batch 3
    '6263','5351','8043','6834','7734','7769','2404','7703','6944','6139','3563','3402','6691',
    # batch 4
    '2356','2382','8021','5386','8112','4958','6584','6187','3189','4900','3037','2337',
    '1605','2383','2327','2059','6770','2810','3443','2344','3163','3234','3450','6442',
    '4979','2417','6658','8027','4542','6234','6727','1582','6903','8039','3576',
    # batch 5
    '3455','7709','6163','2412','2880','7744','2892','2645','2884','2886','4938',
    '2485','3090',
    # batch 6 (latest request)
    '6781','3211','2301','7828','8064','4540','2451','6805','3231','2883','2867',
    '3017','2345','2308','3711','2330','8046','2360','2454','3081','6683','2455',
    '4979','2891','9907','2317','3680','1727','6861','7610','6658',
    # extra from current session
    '3090','3008','2891','4743','5489','5536','6613','3563','7610','6861','5284',
    '2745','2731','6612','6894','4931','2308',
    # new batches
    '1582','2568','3023','3663','3746','6209','6443','6538',
    '3535','8044','8455','3580','8291','3167',
    '3491','6980','3138','2313','6285','3481','2408','2344','2454','2337',
    '6138','3264','6147','6257','8064','3455','3450','4979','2426','4966',
    '3581','3265','3535','3714','2340','3587','8028','3680','1773',
    '2355','2481','2568','3023','3037','3663','3746','5351','6217','6531',
    '3491','3138','6285','3481','6980','2313',
    '1582','3535','8044','8455','3580','3090','8291','3167',
    '3147','6538','2568','3746','3663','3023',

]))

SCRIPT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl_trading_improved_anti_overfit.py')
PYTHON   = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(SCRIPT, 'r', encoding='utf-8') as f:
    template = f.read()

total  = len(TICKERS)
done   = []
failed = []

for i, code in enumerate(TICKERS, 1):
    ticker = f'{code}.TW'
    model_file = os.path.join(BASE_DIR, f'{ticker}_improved_anti_overfit_model.keras')

    if os.path.exists(model_file):
        print(f'[{i}/{total}] {ticker} already trained — skip', flush=True)
        done.append(ticker)
        continue

    print(f'\n{"="*60}', flush=True)
    print(f'[{i}/{total}] Training {ticker}', flush=True)
    print(f'{"="*60}', flush=True)

    content = re.sub(r"ticker\s*=\s*'[^']*'", f"ticker = '{ticker}'", template)

    # Write to temp file to avoid -c argument crash
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                     encoding='utf-8', dir=BASE_DIR)
    tmp.write(content)
    tmp.close()

    try:
        result = subprocess.run(
            [PYTHON, tmp.name],
            cwd=BASE_DIR,
            timeout=3600,  # 1 hour per stock max
        )
        if result.returncode == 0:
            print(f'[{i}/{total}] {ticker} DONE', flush=True)
            done.append(ticker)
            # Auto-register signal script
            try:
                subprocess.run([PYTHON, os.path.join(BASE_DIR, '_auto_register_trained_models.py')],
                               cwd=BASE_DIR, timeout=60)
            except Exception as _e:
                print(f'  Auto-register error: {_e}', flush=True)
            # Run RF→PPO variant comparison (A, B, C)
            try:
                print(f'  Running RF→PPO variant comparison for {ticker}...', flush=True)
                subprocess.run([PYTHON, os.path.join(BASE_DIR, '_compare_rf_ppo_variants.py'), ticker],
                               cwd=BASE_DIR, timeout=7200)
            except Exception as _e:
                print(f'  RF comparison error: {_e}', flush=True)
        else:
            print(f'[{i}/{total}] {ticker} FAILED (exit {result.returncode})', flush=True)
            failed.append(ticker)
    except subprocess.TimeoutExpired:
        print(f'[{i}/{total}] {ticker} TIMEOUT', flush=True)
        failed.append(ticker)
    finally:
        os.unlink(tmp.name)

    import time; time.sleep(3)  # brief pause between stocks to avoid DNS rate limiting

print(f'\n{"="*60}', flush=True)
print(f'Batch complete: {len(done)} done, {len(failed)} failed', flush=True)
if failed:
    print(f'Failed: {", ".join(failed)}', flush=True)
