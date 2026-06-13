"""
Fix headers that still have wrong stock ticker and company name
"""
import re
from pathlib import Path

def fix_header(file_path):
    """Fix header with wrong ticker and company name"""

    # Extract the actual stock ticker from filename
    match = re.match(r'get_trading_signal_(\w+)\.py', file_path.name)
    if not match:
        return False

    actual_ticker = match.group(1)

    # Don't fix 6442 - that's the correct file for 光聖
    if actual_ticker == '6442':
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern: 台股 6442 (光聖) AI 交易信号生成器
    # Replace with: 台股 {actual_ticker} AI 交易信号生成器
    pattern = r'台股 6442 \(光聖\) AI 交易信号生成器'

    if re.search(pattern, content):
        # Determine suffix for display
        if actual_ticker.isdigit():
            replacement = f'台股 {actual_ticker} AI 交易信号生成器'
        else:
            replacement = f'{actual_ticker} AI 交易信号生成器'

        content = re.sub(pattern, replacement, content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all signal file headers"""
    print("=" * 80)
    print("Fixing Headers with Wrong Stock Ticker")
    print("=" * 80)

    signal_files = sorted(Path('.').glob('get_trading_signal_*.py'))

    fixed_count = 0
    skipped_count = 0

    for file_path in signal_files:
        try:
            if fix_header(file_path):
                print(f"[OK] Fixed {file_path.name}")
                fixed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Fixed: {fixed_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Total: {len(signal_files)} files")
    print("=" * 80)

if __name__ == "__main__":
    main()
