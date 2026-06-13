"""
Fix SMA 200 issue in all get_trading*.py files
Adds sma_200 calculation and increases data download period to 300 days
"""
import os
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_sma200_and_period(file_path):
    """Fix SMA 200 and data period in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = []

        # Fix 1: Add sma_200 if missing
        if "df['sma_200']" not in content and "def add_technical_indicators" in content:
            # Find and add sma_200 after sma_50
            pattern = r"(    df\['sma_50'\] = df\['close'\]\.rolling\(50\)\.mean\(\))"
            replacement = r"\1\n    df['sma_200'] = df['close'].rolling(200).mean()  # 添加200日均线"

            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content, count=1)
                changes_made.append("Added sma_200")

        # Fix 2: Change period to 300d if it's less
        period_patterns = [
            (r"period='60d'", "period='300d'  # 改为300天以计算200日均线"),
            (r"period='90d'", "period='300d'  # 改为300天以计算200日均线"),
            (r"period='120d'", "period='300d'  # 改为300天以计算200日均线"),
            (r"period='180d'", "period='300d'  # 改为300天以计算200日均线"),
        ]

        for old_period, new_period in period_patterns:
            if re.search(old_period, content):
                content = re.sub(old_period, new_period, content)
                changes_made.append(f"Updated period to 300d")
                break

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, ", ".join(changes_made) if changes_made else "Fixed"
        else:
            return False, "No changes needed"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Fixing SMA 200 and data period in all get_trading*.py files")
    print("=" * 80)

    # Find all get_trading*.py files
    current_dir = Path('.')
    files = list(current_dir.glob('get_trading_signal_*.py'))

    if not files:
        print("[ERROR] No get_trading_signal_*.py files found")
        return

    print(f"Found {len(files)} files to process\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for file_path in sorted(files):
        print(f"Processing {file_path.name}...", end=" ")
        success, message = fix_sma200_and_period(file_path)

        if success:
            print(f"[OK] {message}")
            success_count += 1
        else:
            if "No changes" in message:
                print(f"[SKIP] {message}")
                skip_count += 1
            else:
                print(f"[ERROR] {message}")
                error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files:    {len(files)}")
    print(f"Fixed:          {success_count}")
    print(f"Already OK:     {skip_count}")
    print(f"Errors:         {error_count}")
    print("=" * 80)

    if success_count > 0:
        print(f"\n[OK] Successfully fixed {success_count} files!")
    else:
        print("\n[INFO] All files already OK or encountered errors")

if __name__ == "__main__":
    main()
