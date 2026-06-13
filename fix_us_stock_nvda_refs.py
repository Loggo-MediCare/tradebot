"""
Fix hardcoded NVDA/NVIDIA references in US stock signal files
"""
import re
from pathlib import Path

def fix_nvda_references(file_path):
    """Replace hardcoded NVDA/NVIDIA references with correct ticker"""

    # Extract actual stock ticker from filename (will be lowercase)
    match = re.match(r'get_trading_signal_(\w+)\.py', file_path.name)
    if not match:
        return False, "Could not extract ticker from filename"

    actual_ticker = match.group(1).upper()  # Convert to uppercase for display

    # Skip if this is actually the NVDA file
    if actual_ticker == 'NVDA':
        return False, "This is the correct NVDA file"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if file has any NVDA/NVIDIA references
    if 'NVDA' not in content and 'NVIDIA' not in content:
        return False, "No NVDA/NVIDIA references found"

    changes_made = []

    # Fix header: 美股 NVDA (NVIDIA) AI 交易信号生成器
    pattern1 = r'美股 NVDA \(NVIDIA\) AI 交易信号生成器'
    if re.search(pattern1, content):
        replacement = f'美股 {actual_ticker} AI 交易信号生成器'
        content = re.sub(pattern1, replacement, content)
        changes_made.append("header")

    # Fix example output in docstring: 股票: NVDA (NVIDIA)
    pattern2 = r'股票: NVDA \(NVIDIA\)'
    if re.search(pattern2, content):
        replacement = f'股票: {actual_ticker}'
        content = re.sub(pattern2, replacement, content)
        changes_made.append("docstring example")

    # Fix print statement: 美股 NVDA (NVIDIA) AI 交易信号生成器
    pattern3 = r'print\("🤖 美股 NVDA \(NVIDIA\) AI 交易信号生成器"\)'
    if re.search(pattern3, content):
        replacement = f'print("🤖 美股 {actual_ticker} AI 交易信号生成器")'
        content = re.sub(pattern3, replacement, content)
        changes_made.append("print statement")

    # Fix summary output: 股票: {result['symbol']} (NVIDIA)
    pattern4 = r"print\(f\"   股票: \{result\['symbol'\]\} \(NVIDIA\)\"\)"
    if re.search(pattern4, content):
        replacement = f"print(f\"   股票: {{result['symbol']}}\")"
        content = re.sub(pattern4, replacement, content)
        changes_made.append("summary output")

    # Save if changes were made
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, f"Fixed: {', '.join(changes_made)}"

    # Check if there are still NVDA/NVIDIA references
    if 'NVDA' in content or 'NVIDIA' in content:
        return False, f"WARNING: Still has NVDA/NVIDIA references"

    return False, "No changes needed"

def main():
    """Fix all US stock signal files with NVDA/NVIDIA references"""
    print("=" * 80)
    print("Fixing Hardcoded NVDA/NVIDIA References")
    print("=" * 80)

    # Only check US stock files (alphabetic tickers without dots or numbers)
    all_files = list(Path('.').glob('get_trading_signal_*.py'))
    signal_files = []
    for f in all_files:
        # Extract ticker from filename: get_trading_signal_TICKER.py
        ticker = f.stem.replace('get_trading_signal_', '')
        # US stocks are alphabetic (not numeric) and don't contain dots
        # Skip special files like nvda_finbert
        if ticker.isalpha() and '.' not in ticker and '_' not in ticker:
            signal_files.append(f)

    fixed_count = 0
    warning_count = 0
    skipped_count = 0

    for file_path in signal_files:
        try:
            fixed, message = fix_nvda_references(file_path)
            if fixed:
                print(f"[OK] {file_path.name}: {message}")
                fixed_count += 1
            elif "WARNING" in message:
                print(f"[WARN] {file_path.name}: {message}")
                warning_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Fixed: {fixed_count} files")
    print(f"Warnings: {warning_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Total checked: {len(signal_files)} US stock files")
    print("=" * 80)

if __name__ == "__main__":
    main()
