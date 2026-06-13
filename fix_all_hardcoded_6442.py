"""
Fix all hardcoded 6442 references in signal files
This includes print statements, model paths, and other references
"""
import re
from pathlib import Path

def fix_hardcoded_6442(file_path):
    """Replace all hardcoded 6442 references with the correct ticker"""

    # Extract actual stock ticker from filename
    match = re.match(r'get_trading_signal_(\w+)\.py', file_path.name)
    if not match:
        return False, "Could not extract ticker from filename"

    actual_ticker = match.group(1)

    # Skip if this is actually the 6442 file
    if actual_ticker == '6442':
        return False, "This is the correct 6442 file"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if file has any 6442 references
    if '6442' not in content:
        return False, "No 6442 references found"

    changes_made = []

    # Fix print statement: 台股 6442  AI 交易信号生成器
    pattern1 = r'print\("🤖 台股 6442  AI 交易信号生成器"\)'
    if re.search(pattern1, content):
        replacement = f'print("🤖 台股 {actual_ticker} AI 交易信号生成器")'
        content = re.sub(pattern1, replacement, content)
        changes_made.append("print statement")

    # Fix model path: ppo_6442_tw_improved
    pattern2 = r'ppo_6442_tw_improved'
    if re.search(pattern2, content):
        # Determine suffix based on actual ticker
        if actual_ticker.upper().endswith('HK'):
            suffix = 'hk'
        elif any(actual_ticker.endswith(s) for s in ['TWO', 'two']):
            suffix = 'two'
        else:
            suffix = 'tw'

        replacement = f'ppo_{actual_ticker.lower()}_{suffix}_improved'
        content = re.sub(pattern2, replacement, content)
        changes_made.append("model path")

    # Save if changes were made
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, f"Fixed: {', '.join(changes_made)}"

    # Check if there are still other 6442 references
    if '6442' in content:
        return False, f"WARNING: Still has 6442 references after fixes"

    return False, "No changes needed"

def main():
    """Fix all signal files with hardcoded 6442 references"""
    print("=" * 80)
    print("Fixing Hardcoded 6442 References")
    print("=" * 80)

    signal_files = sorted(Path('.').glob('get_trading_signal_*.py'))

    fixed_count = 0
    warning_count = 0
    skipped_count = 0

    for file_path in signal_files:
        try:
            fixed, message = fix_hardcoded_6442(file_path)
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
    print(f"Total: {len(signal_files)} files")
    print("=" * 80)

if __name__ == "__main__":
    main()
