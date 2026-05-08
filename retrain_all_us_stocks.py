"""
Batch retrain all US stocks (updated to 2025-07-31 data)
"""
import subprocess
import sys
import io
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# US stocks list
US_STOCKS = [
    'aapl', 'aeva', 'alab', 'amkr', 'avgo', 'goog', 'htgc', 'mu',
    'nat', 'nvda', 'nxpi', 'omer', 'onds', 'pltr', 'tsla'
]

def train_stock(ticker):
    """Train single stock"""
    script_name = f"train_{ticker}_improved.py"
    print(f"\n{'='*80}")
    print(f"Training: {ticker.upper()}")
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
            print(f"SUCCESS: {ticker.upper()} trained!")
            return True
        else:
            print(f"FAILED: {ticker.upper()} training failed:")
            print(result.stderr[:500])
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {ticker.upper()} (>10 min)")
        return False
    except Exception as e:
        print(f"ERROR: {ticker.upper()} - {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("Batch Retrain All US Stocks")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total stocks: {len(US_STOCKS)}")
    print("="*80)

    success_count = 0
    failed_stocks = []

    for i, ticker in enumerate(US_STOCKS, 1):
        print(f"\nProgress: [{i}/{len(US_STOCKS)}]")

        if train_stock(ticker):
            success_count += 1
        else:
            failed_stocks.append(ticker)

    # Final summary
    print("\n" + "="*80)
    print("Batch Training Complete!")
    print("="*80)
    print(f"Success: {success_count}/{len(US_STOCKS)}")
    print(f"Failed: {len(failed_stocks)}")

    if failed_stocks:
        print(f"\nFailed stocks:")
        for stock in failed_stocks:
            print(f"   - {stock.upper()}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
