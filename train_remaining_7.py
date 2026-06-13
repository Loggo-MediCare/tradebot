"""
Batch train the 7 remaining watchlist stocks
"""
import subprocess
import sys
import io
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 7 remaining stocks
STOCKS = ['1326', '3533', '2059', '1514', '6805', '8046', '1519']

def train_stock(ticker):
    """Train single stock"""
    script_name = f"train_{ticker}_taiwan_improved.py"
    print(f"\n{'='*80}")
    print(f"Training: {ticker}.TW")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600,
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode == 0:
            print(f"SUCCESS: {ticker}.TW")
            return True
        else:
            print(f"FAILED: {ticker}.TW")
            print(result.stderr[:300])
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {ticker}.TW")
        return False
    except Exception as e:
        print(f"ERROR: {ticker}.TW - {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("Batch Train 7 Remaining Watchlist Stocks")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stocks: {', '.join(STOCKS)}")
    print("="*80)

    success = []
    failed = []

    for i, ticker in enumerate(STOCKS, 1):
        print(f"\nProgress: [{i}/{len(STOCKS)}]")
        if train_stock(ticker):
            success.append(ticker)
        else:
            failed.append(ticker)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Success: {len(success)}/{len(STOCKS)}")
    print(f"Failed: {len(failed)}")

    if success:
        print(f"\nSuccessfully trained:")
        for s in success:
            print(f"  - {s}.TW")

    if failed:
        print(f"\nFailed:")
        for s in failed:
            print(f"  - {s}.TW")

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
