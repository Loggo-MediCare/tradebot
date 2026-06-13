"""
Batch train specific Taiwan stocks
"""
import subprocess
import sys
import io
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Custom stock list (normalize ticker format)
STOCKS_TO_TRAIN = [
    '2344', '2408', '2478', '2515', '3060',
    '3138', '3163', '3189', '3234', '3236',
    '3363', '3430', '3481', '3543', '3576',
    '3615', '4722', '4764', '4927', '5386',
    '6187', '6861', '8046'
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
            timeout=600,  # 10 min timeout
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode == 0:
            print(f"SUCCESS: {ticker}.TW trained!")
            return True
        else:
            print(f"FAILED: {ticker}.TW training failed:")
            print(result.stderr[:500])
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {ticker}.TW (>10 min)")
        return False
    except FileNotFoundError:
        print(f"SCRIPT NOT FOUND: {script_name}")
        return False
    except Exception as e:
        print(f"ERROR: {ticker}.TW - {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("Batch Train Custom Taiwan Stocks")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total stocks: {len(STOCKS_TO_TRAIN)}")
    print("="*80)

    success_count = 0
    failed_stocks = []

    for i, ticker in enumerate(STOCKS_TO_TRAIN, 1):
        print(f"\nProgress: [{i}/{len(STOCKS_TO_TRAIN)}]")

        if train_stock(ticker):
            success_count += 1
        else:
            failed_stocks.append(ticker)

    # Final summary
    print("\n" + "="*80)
    print("Batch Training Complete!")
    print("="*80)
    print(f"Success: {success_count}/{len(STOCKS_TO_TRAIN)}")
    print(f"Failed: {len(failed_stocks)}")

    if failed_stocks:
        print(f"\nFailed stocks:")
        for stock in failed_stocks:
            print(f"   - {stock}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
