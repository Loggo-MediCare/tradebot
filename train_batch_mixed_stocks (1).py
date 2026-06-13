"""
Batch train mixed Taiwan & US stocks (11 stocks)
Taiwan: 2408南亞科, 2308台達電, 2337旺宏, 3006晶豪科, 6442 EZconn, 3017奇鋐, 2344華邦電
US: LITE Lumentum, UMC, OMER Omeros, ONDS Ondas
"""
import subprocess
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Taiwan stocks use train_{ticker}_taiwan_improved.py
# US stocks use train_{ticker}_us_improved.py
STOCKS = [
    # Taiwan stocks
    ('2408', '南亞科 Nanya - DRAM', 'taiwan'),
    ('2308', '台達電 Delta - Power', 'taiwan'),
    ('2337', '旺宏 Macronix - Flash', 'taiwan'),
    ('3006', '晶豪科 Elite - IC Design', 'taiwan'),
    ('6442', 'EZconn Corp - Connectors', 'taiwan'),
    ('3017', '奇鋐 AURAS - Thermal', 'taiwan'),
    ('2344', '華邦電 Winbond - Flash', 'taiwan'),
    # US stocks
    ('lite', 'Lumentum - Optical', 'us'),
    ('umc', 'United Microelectronics - Semi', 'us'),
    ('omer', 'Omeros - Biotech', 'us'),
    ('onds', 'Ondas Holdings - Wireless', 'us'),
]

def train_stock(ticker, name, market):
    if market == 'taiwan':
        script = f"train_{ticker}_taiwan_improved.py"
        display = f"{ticker}.TW"
    else:
        script = f"train_{ticker}_us_improved.py"
        display = ticker.upper()

    print(f"\n{'='*80}")
    print(f"Training: {display} - {name}")
    print(f"{'='*80}", flush=True)

    try:
        process = subprocess.Popen(
            [sys.executable, '-u', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1
        )
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait(timeout=900)
        if process.returncode == 0:
            print(f"SUCCESS: {display}")
            return True
        else:
            print(f"FAILED: {display} (exit code: {process.returncode})")
            return False
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"TIMEOUT: {display}")
        return False
    except Exception as e:
        print(f"ERROR: {display} - {e}")
        return False

if __name__ == "__main__":
    start = datetime.now()
    print("=" * 80)
    print(f"Training {len(STOCKS)} Mixed TW/US Stocks")
    print(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    success, failed = 0, []
    for i, (ticker, name, market) in enumerate(STOCKS, 1):
        print(f"\nProgress: [{i}/{len(STOCKS)}]")
        if train_stock(ticker, name, market):
            success += 1
        else:
            failed.append(ticker)

    print("\n" + "=" * 80)
    print(f"COMPLETE - Time: {datetime.now() - start}")
    print(f"Success: {success}/{len(STOCKS)}")
    if failed: print(f"Failed: {', '.join(failed)}")
