"""
Batch retrain all Taiwan stocks (updated to 2025-07-31 data)
"""
import subprocess
import sys
import io
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Taiwan stocks list
TW_STOCKS = [
    '1303', '1519', '1605', '2308', '2330', '2337', '2344', '2360', '2368',
    '2408', '2449', '2451', '2454', '3017', '3443', '3653', '3661', '3711',
    '3715', '4746', '4938', '6187', '6209', '6269', '6442', '6443', '6515',
    '6770', '6781', '6805', '8110', '8131', '8210'
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
    except Exception as e:
        print(f"ERROR: {ticker}.TW - {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("Batch Retrain All Taiwan Stocks")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total stocks: {len(TW_STOCKS)}")
    print("="*80)

    success_count = 0
    failed_stocks = []

    for i, ticker in enumerate(TW_STOCKS, 1):
        print(f"\nProgress: [{i}/{len(TW_STOCKS)}]")

        if train_stock(ticker):
            success_count += 1
        else:
            failed_stocks.append(ticker)

    # Final summary
    print("\n" + "="*80)
    print("Batch Training Complete!")
    print("="*80)
    print(f"Success: {success_count}/{len(TW_STOCKS)}")
    print(f"Failed: {len(failed_stocks)}")

    if failed_stocks:
        print(f"\nFailed stocks:")
        for stock in failed_stocks:
            print(f"   - {stock}.TW")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
