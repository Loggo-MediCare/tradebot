"""
Batch train new Taiwan stocks
"""
import subprocess
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

NEW_TW_STOCKS = [
    '2884', '6603', '4540', '6477', '3030', '1815', '1101', '8377',
    '2313', '8292', '8042', '3690', '4541', '2303', '3481', '2363',
    '3360', '3022', '3630', '2634', '3004', '8222', '6829', '6668',
    '2645', '8033', '5371'
]

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
            print(f"SUCCESS: {ticker}.TW trained!")
            return True
        else:
            print(f"FAILED: {ticker}.TW")
            print(result.stderr[:500])
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {ticker}.TW")
        return False
    except Exception as e:
        print(f"ERROR: {ticker}.TW - {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print(f"Batch Training {len(NEW_TW_STOCKS)} New Taiwan Stocks")
    print("="*80)

    success = 0
    failed = []

    for i, ticker in enumerate(NEW_TW_STOCKS, 1):
        print(f"\nProgress: [{i}/{len(NEW_TW_STOCKS)}]")
        if train_stock(ticker):
            success += 1
        else:
            failed.append(ticker)

    print("\n" + "="*80)
    print("BATCH TRAINING COMPLETE")
    print("="*80)
    print(f"Success: {success}/{len(NEW_TW_STOCKS)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed stocks: {', '.join(failed)}")
