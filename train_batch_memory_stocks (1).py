"""
Batch train Memory Sector stocks (11 stocks)
- Upstream DRAM: 2408南亞科, 2344華邦電, 6770力積電
- Midstream SSD/Modules: 8299群聯, 5289宜鼎, 2451創見, 3260威剛, 3081聯亞
- Reference: 2330台積電
- Shipping: 2603長榮, 2609陽明
"""
import subprocess
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

STOCKS = [
    # Upstream DRAM
    ('2408', '南亞科 Nanya - DRAM'),
    ('2344', '華邦電 Winbond - Flash'),
    ('6770', '力積電 Powerchip - Foundry'),
    # Midstream SSD/Modules
    ('8299', '群聯 Phison - SSD Controller'),
    ('5289', '宜鼎 Innodisk - Industrial SSD'),
    ('2451', '創見 Transcend - Memory Modules'),
    ('3260', '威剛 ADATA - Consumer Memory'),
    ('3081', '聯亞 Apacer - Memory Modules'),
    # Reference
    ('2330', '台積電 TSMC'),
    # Shipping
    ('2603', '長榮 Evergreen - Shipping'),
    ('2609', '陽明 Yang Ming - Shipping'),
]

def train_stock(ticker, name):
    script = f"train_{ticker}_taiwan_improved.py"
    print(f"\n{'='*80}")
    print(f"Training: {ticker}.TW - {name}")
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
            print(f"SUCCESS: {ticker}")
            return True
        else:
            print(f"FAILED: {ticker}")
            return False
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"TIMEOUT: {ticker}")
        return False
    except Exception as e:
        print(f"ERROR: {ticker} - {e}")
        return False

if __name__ == "__main__":
    start = datetime.now()
    print("=" * 80)
    print(f"Training {len(STOCKS)} Memory Sector Stocks")
    print(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    success, failed = 0, []
    for i, (ticker, name) in enumerate(STOCKS, 1):
        print(f"\nProgress: [{i}/{len(STOCKS)}]")
        if train_stock(ticker, name):
            success += 1
        else:
            failed.append(ticker)

    print("\n" + "=" * 80)
    print(f"COMPLETE - Time: {datetime.now() - start}")
    print(f"Success: {success}/{len(STOCKS)}")
    if failed: print(f"Failed: {', '.join(failed)}")
