"""
Batch train 23 Taiwan stocks
"""
import subprocess
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 23 Taiwan stocks to train
TW_STOCKS = [
    '2367', '2313', '1471', '8046', '3037', '2431', '2484', '3308',
    '3645', '7788', '6155', '2492', '3432', '2478', '2327', '6282',
    '6862', '2308', '6133', '3715', '3092', '4989', '3044'
]

def train_stock(ticker):
    """Train single stock"""
    script_name = f"train_{ticker}_taiwan_improved.py"
    print(f"\n{'='*80}")
    print(f"Training: {ticker}.TW")
    print(f"Script: {script_name}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=900,  # 15 minutes timeout
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode == 0:
            print(f"SUCCESS: {ticker}.TW trained!")
            return True
        else:
            print(f"FAILED: {ticker}.TW")
            if result.stderr:
                print(result.stderr[:500])
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {ticker}.TW")
        return False
    except Exception as e:
        print(f"ERROR: {ticker}.TW - {e}")
        return False

if __name__ == "__main__":
    start_time = datetime.now()
    print("=" * 80)
    print(f"Batch Training {len(TW_STOCKS)} Taiwan Stocks")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    success = 0
    failed = []

    for i, ticker in enumerate(TW_STOCKS, 1):
        print(f"\nProgress: [{i}/{len(TW_STOCKS)}]")
        if train_stock(ticker):
            success += 1
        else:
            failed.append(ticker)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("BATCH TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {duration}")
    print(f"Success: {success}/{len(TW_STOCKS)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed stocks: {', '.join(failed)}")
