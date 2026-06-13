"""
Batch train Hybrid RF→PPO for ALL Taiwan stocks.
Skips stocks already trained (hybrid model file exists).
Records ROI + Sharpe ratio per stock.
"""
import os, sys, io, re, subprocess, tempfile, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable

# Priority stocks — trained first
PRIORITY = [
    '2330.TW','2327.TW','2313.TW','2408.TW','2308.TW','2454.TW',
    '2303.TW','3481.TW','2344.TW','2337.TW','3037.TW','3189.TW',
    '3711.TW','2317.TW','6285.TW','2481.TW','3491.TWO','3443.TW',
    '2382.TW','2376.TW','2345.TW','6223.TWO','2360.TW',
]

# Get remaining TW stocks from run_all_local_tw.py
with open(os.path.join(BASE_DIR, 'run_all_local_tw.py'), encoding='utf-8') as f:
    content = f.read()

import re as _re
codes = _re.findall(r"get_trading_signal_(\d+)\.py", content)
codes = sorted(set(codes))

def get_ticker(code):
    if os.path.exists(os.path.join(BASE_DIR, f'ppo_{code}_two_improved.zip')):
        return f'{code}.TWO'
    return f'{code}.TW'

priority_set = {t.split('.')[0] for t in PRIORITY}
rest = [get_ticker(c) for c in codes if c not in priority_set]

# Priority first, then the rest
TICKERS = PRIORITY + rest
total   = len(TICKERS)
done    = []
failed  = []
skipped = []

print(f'Batch Hybrid RF→PPO Training — {total} TW stocks', flush=True)
print(f'Steps: 150,000 | 3 years data | out-of-sample Sharpe', flush=True)
print('='*60, flush=True)

for i, ticker in enumerate(TICKERS, 1):
    code = ticker.split('.')[0]

    # Skip if already trained
    model_file = os.path.join(BASE_DIR, f'hybrid_{ticker.lower().replace(".", "_")}_ppo.zip')
    if os.path.exists(model_file):
        print(f'[{i}/{total}] {ticker} already trained — skip', flush=True)
        skipped.append(ticker)
        continue

    print(f'\n{"="*60}', flush=True)
    print(f'[{i}/{total}] Hybrid RF→PPO: {ticker}', flush=True)
    print(f'{"="*60}', flush=True)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                     encoding='utf-8', dir=BASE_DIR)
    tmp.write(f"import sys; sys.argv=['_train_hybrid_rf_ppo.py','{ticker}']\n")
    tmp.write(open(os.path.join(BASE_DIR, '_train_hybrid_rf_ppo.py'), encoding='utf-8').read())
    tmp.close()

    try:
        r = subprocess.run([PYTHON, tmp.name], cwd=BASE_DIR, timeout=7200)
        if r.returncode == 0:
            print(f'[{i}/{total}] {ticker} DONE', flush=True)
            done.append(ticker)
        else:
            print(f'[{i}/{total}] {ticker} FAILED (exit {r.returncode})', flush=True)
            failed.append(ticker)
    except subprocess.TimeoutExpired:
        print(f'[{i}/{total}] {ticker} TIMEOUT', flush=True)
        failed.append(ticker)
    finally:
        os.unlink(tmp.name)

    time.sleep(3)

# Final summary
print(f'\n{"="*60}', flush=True)
print(f'Hybrid RF→PPO Batch Complete', flush=True)
print(f'Done: {len(done)}  Failed: {len(failed)}  Skipped: {len(skipped)}', flush=True)
if failed:
    print(f'Failed: {", ".join(failed)}', flush=True)

# Show ROI table
print(f'\n{"="*60}', flush=True)
print(f'ROI + Sharpe Summary', flush=True)
print(f'{"="*60}', flush=True)
import json, glob
rows = []
for f in glob.glob(os.path.join(BASE_DIR, 'model_accuracy_*_Hybrid.json')):
    sym = os.path.basename(f).replace('model_accuracy_','').replace('_Hybrid.json','').replace('_','.')
    with open(f, encoding='utf-8') as fh:
        d = json.load(fh)
    ba = d.get('backtest_accuracy', 0) or 0
    wr = d.get('win_rate', 0) or 0
    sr = d.get('sharpe_ratio')
    from model_accuracy_tracker import ModelAccuracyTracker
    roi_est = (ba - 50) * 2
    rows.append((sym, roi_est, sr, wr))

rows.sort(key=lambda x: x[1], reverse=True)
print(f'{"Stock":<20} {"Est. ROI":>10} {"Sharpe":>8} {"Win Rate":>9}')
print('-'*50)
for sym, roi, sr, wr in rows:
    sr_s = f'{sr:.3f}' if sr else '-'
    print(f'{sym:<20} {roi:>+9.1f}% {sr_s:>8} {wr:>8.1f}%')
