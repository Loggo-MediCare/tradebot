"""
Retrain stocks missing feature importance files
"""
import subprocess
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Taiwan stocks missing feature importance
TW_STOCKS = [
    '1101', '1303', '1605', '2303', '2313', '2363', '2368', '2382',
    '2383', '2409', '2454', '2634', '2884', '3004', '3006', '3022',
    '3030', '3037', '3135', '3231', '3443', '3481', '4540', '4746',
    '6239', '6477', '6668', '6669', '6805', '8150', '8222'
]

# US stocks missing feature importance
US_STOCKS = ['aeva', 'alab', 'oklo', 'onds', 'sndk']

def retrain_stock(ticker, is_taiwan=True):
    """Retrain single stock"""
    if is_taiwan:
        script_name = f"train_{ticker}_taiwan_improved.py"
        display_name = f"{ticker}.TW"
    else:
        script_name = f"train_{ticker}_improved.py"
        display_name = ticker.upper()

    print(f"\n{'='*80}")
    print(f"Retraining: {display_name}")
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
            print(f"✓ SUCCESS: {display_name}")
            return True
        else:
            print(f"✗ FAILED: {display_name}")
            print(result.stderr[:300])
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱ TIMEOUT: {display_name}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {display_name} - {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("Batch Retrain for Feature Importance Files")
    print("="*80)

    total = len(TW_STOCKS) + len(US_STOCKS)
    print(f"Taiwan stocks: {len(TW_STOCKS)}")
    print(f"US stocks: {len(US_STOCKS)}")
    print(f"Total: {total}")

    success = 0
    failed = []

    # Retrain Taiwan stocks
    for i, ticker in enumerate(TW_STOCKS, 1):
        print(f"\n[{i}/{total}] Processing Taiwan stock...")
        if retrain_stock(ticker, is_taiwan=True):
            success += 1
        else:
            failed.append(f"{ticker}.TW")

    # Retrain US stocks
    offset = len(TW_STOCKS)
    for i, ticker in enumerate(US_STOCKS, 1):
        print(f"\n[{offset + i}/{total}] Processing US stock...")
        if retrain_stock(ticker, is_taiwan=False):
            success += 1
        else:
            failed.append(ticker.upper())

    print("\n" + "="*80)
    print("BATCH RETRAIN COMPLETE")
    print("="*80)
    print(f"Success: {success}/{total}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed stocks: {', '.join(failed)}")
