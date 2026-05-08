"""
Batch train AI & Power stocks (11 stocks)
- AI: 2454聯發科, 5269祥碩, 3034聯詠, 2330台積電, 2317鴻海, 2382廣達, 2308台達電
- Power/BBU: 1519華城, 6781 AES-KY
- Airlines: 2610華航, 2618長榮航
"""
import subprocess
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

STOCKS = [
    ('2454', '聯發科 MediaTek'),
    ('5269', '祥碩 ASMedia'),
    ('1519', '華城電機'),
    ('3034', '聯詠 Novatek'),
    ('2610', '華航 China Airlines'),
    ('2618', '長榮航 EVA Air'),
    ('2330', '台積電 TSMC'),
    ('2317', '鴻海 Foxconn'),
    ('2382', '廣達 Quanta'),
    ('2308', '台達電 Delta'),
    ('6781', 'AES-KY BBU'),
]

def train_stock(ticker, name):
    script_name = f"train_{ticker}_taiwan_improved.py"
    print(f"\n{'='*80}")
    print(f"Training: {ticker}.TW - {name}")
    print(f"{'='*80}", flush=True)

    try:
        # Use Popen with streaming to avoid buffer issues
        process = subprocess.Popen(
            [sys.executable, '-u', script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1
        )

        # Stream output line by line
        for line in process.stdout:
            print(line, end='', flush=True)

        process.wait(timeout=900)

        if process.returncode == 0:
            print(f"SUCCESS: {ticker}.TW")
            return True
        else:
            print(f"FAILED: {ticker}.TW (exit code: {process.returncode})")
            return False

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"TIMEOUT: {ticker}.TW")
        return False
    except Exception as e:
        print(f"ERROR: {ticker}.TW - {e}")
        return False

if __name__ == "__main__":
    start_time = datetime.now()
    print("=" * 80)
    print(f"Training {len(STOCKS)} AI/Power/Airline Stocks")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    success = 0
    failed = []

    for i, (ticker, name) in enumerate(STOCKS, 1):
        print(f"\nProgress: [{i}/{len(STOCKS)}]")
        if train_stock(ticker, name):
            success += 1
        else:
            failed.append(ticker)

    end_time = datetime.now()
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Time: {end_time - start_time}")
    print(f"Success: {success}/{len(STOCKS)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
