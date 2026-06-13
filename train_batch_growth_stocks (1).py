"""
Batch train Growth/Value stocks (8 stocks)
- US: NVDA (Nvidia), MU (Micron)
- TW Growth: 2059川湖, 3665貿聯-KY, 6958筑生科技
- TW Defensive: 8462柏文, 6835青新創, 8341日友
"""
import subprocess
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

STOCKS = [
    # US Stocks
    ('NVDA', 'Nvidia', 'train_nvda_improved.py'),
    ('MU', 'Micron', 'train_mu_improved.py'),
    # TW Growth Stocks
    ('2059', '川湖 King Slide - AI Server Rails', 'train_2059_taiwan_improved.py'),
    ('3665', '貿聯-KY BizLink - Nvidia Supplier', 'train_3665_taiwan_improved.py'),
    ('6958', '筑生科技 - TSMC/Micron Smart Factory', 'train_6958_taiwan_improved.py'),
    # TW Defensive/Value Stocks
    ('8462', '柏文 Fitness Factory', 'train_8462_taiwan_improved.py'),
    ('6835', '青新創 Qingxin - Waste Management', 'train_6835_taiwan_improved.py'),
    ('8341', '日友 Cleanaway', 'train_8341_taiwan_improved.py'),
]

def train_stock(ticker, name, script):
    print(f"\n{'='*80}")
    print(f"Training: {ticker} - {name}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=900,
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode == 0:
            print(f"SUCCESS: {ticker}")
            return True
        else:
            print(f"FAILED: {ticker}")
            if result.stderr:
                print(result.stderr[:500])
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {ticker}")
        return False
    except Exception as e:
        print(f"ERROR: {ticker} - {e}")
        return False

if __name__ == "__main__":
    start_time = datetime.now()
    print("=" * 80)
    print(f"Training {len(STOCKS)} Growth/Value Stocks")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    success = 0
    failed = []

    for i, (ticker, name, script) in enumerate(STOCKS, 1):
        print(f"\nProgress: [{i}/{len(STOCKS)}]")
        if train_stock(ticker, name, script):
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
