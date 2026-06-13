"""
Batch Training Script for Taiwan Stocks (Batch 6)
Trains: 6415, 2740, 7788, 5245, 6510, 8088, 3221, 7610, 9103, 6588, 7777, 6265
"""
import subprocess
import sys
import os

# Taiwan stocks to train
TW_STOCKS = [
    '6415', '2740', '7788', '5245', '6510', '8088',
    '3221', '7610', '9103', '6588', '7777', '6265'
]

def check_and_fix_script(ticker):
    """Check if training script exists and fix pandas compatibility"""
    script_file = f'train_{ticker}_taiwan_improved.py'

    if not os.path.exists(script_file):
        return False, "Script not found"

    # Fix pandas compatibility
    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix fillna syntax if needed
        if "fillna(method='bfill')" in content or "fillna(method='ffill')" in content:
            content = content.replace(
                "df = df.fillna(method='bfill').fillna(method='ffill')",
                "df = df.bfill().ffill()"
            )
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed and ready"

        return True, "Ready"
    except Exception as e:
        return False, f"Error: {e}"

def train_stock(ticker):
    """Train a single Taiwan stock"""
    print(f"\n{'='*70}")
    print(f"Processing {ticker}.TW...")
    print(f"{'='*70}")

    # Check if script exists and fix it
    script_file = f'train_{ticker}_taiwan_improved.py'
    exists, status = check_and_fix_script(ticker)

    if not exists:
        print(f"[SKIP] {script_file} not found - {status}")
        return False

    print(f"[INFO] {status}")

    # Run training
    try:
        print(f"[INFO] Starting training for {ticker}.TW...")
        result = subprocess.run(
            [sys.executable, script_file],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
            encoding='utf-8'
        )

        if result.returncode == 0:
            print(f"[OK] {ticker}.TW training completed successfully")
            # Extract model filename from output
            if 'ppo_' in result.stdout:
                for line in result.stdout.split('\n'):
                    if '模型已保存' in line or 'ppo_' in line:
                        print(f"[INFO] {line.strip()}")
                        break
            return True
        else:
            print(f"[ERROR] {ticker}.TW training failed")
            # Print last 500 chars of error
            if result.stderr:
                print(result.stderr[-500:])
            return False

    except subprocess.TimeoutExpired:
        print(f"[ERROR] {ticker}.TW training timed out (>10 min)")
        return False
    except Exception as e:
        print(f"[ERROR] Error training {ticker}.TW: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("Batch Training for Taiwan Stocks (Batch 6)")
    print("="*70)
    print(f"Stocks to train: {', '.join(TW_STOCKS)}")
    print(f"Total: {len(TW_STOCKS)} stocks")
    print("="*70)

    results = {}
    skipped = []

    for ticker in TW_STOCKS:
        result = train_stock(ticker)
        if result is False and f'train_{ticker}_taiwan_improved.py' not in os.listdir('.'):
            skipped.append(ticker)
        else:
            results[ticker] = result

    # Summary
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    successful = [t for t, s in results.items() if s]
    failed = [t for t, s in results.items() if not s]

    print(f"[OK] Successful: {len(successful)}/{len(TW_STOCKS)}")
    if successful:
        for ticker in successful:
            print(f"   - {ticker}.TW")

    if failed:
        print(f"\n[ERROR] Failed: {len(failed)}/{len(TW_STOCKS)}")
        for ticker in failed:
            print(f"   - {ticker}.TW")

    if skipped:
        print(f"\n[SKIP] Skipped (no script): {len(skipped)}/{len(TW_STOCKS)}")
        for ticker in skipped:
            print(f"   - {ticker}.TW (need to create train_{ticker}_taiwan_improved.py)")

    print("\n" + "="*70)
    print(f"Total processed: {len(results)}/{len(TW_STOCKS)}")
    print("="*70)
