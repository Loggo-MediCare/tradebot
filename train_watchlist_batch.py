"""
Batch train watchlist stocks
"""
import subprocess
import sys
import io
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Watchlist stocks - Taiwan stocks only (US stocks need different handling)
WATCHLIST_STOCKS = [
    '8150', '1326', '8046', '1301', '1519', '3533',
    '3017', '8021', '2059', '3653', '1514', '2317',
    '2383', '6805'
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
            print(f"✅ SUCCESS: {ticker}.TW trained!")
            return True
        else:
            print(f"❌ FAILED: {ticker}.TW training failed:")
            # Print first 500 chars of error
            error_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
            print(error_msg)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {ticker}.TW (>10 min)")
        return False
    except FileNotFoundError:
        print(f"📁 SCRIPT NOT FOUND: {script_name}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {ticker}.TW - {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("🚀 Batch Train Watchlist Stocks")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total stocks: {len(WATCHLIST_STOCKS)}")
    print("="*80)

    success_count = 0
    failed_stocks = []
    success_stocks = []

    for i, ticker in enumerate(WATCHLIST_STOCKS, 1):
        print(f"\n📊 Progress: [{i}/{len(WATCHLIST_STOCKS)}]")

        if train_stock(ticker):
            success_count += 1
            success_stocks.append(ticker)
        else:
            failed_stocks.append(ticker)

    # Final summary
    print("\n" + "="*80)
    print("✨ Batch Training Complete!")
    print("="*80)
    print(f"✅ Success: {success_count}/{len(WATCHLIST_STOCKS)}")
    print(f"❌ Failed: {len(failed_stocks)}")

    if success_stocks:
        print(f"\n✅ Successfully trained stocks:")
        for stock in success_stocks:
            print(f"   ✓ {stock}.TW")

    if failed_stocks:
        print(f"\n❌ Failed stocks:")
        for stock in failed_stocks:
            print(f"   ✗ {stock}.TW")

    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
