"""
Add sma_200 calculation to all get_trading_signal_*.py files
"""
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def add_sma_200(file_path):
    """Add sma_200 calculation to add_technical_indicators function"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Check if sma_200 already exists
        if "df['sma_200']" in content:
            return False, "Already has sma_200"

        # Add sma_200 after sma_50
        pattern = r"(df\['sma_50'\] = df\['close'\]\.rolling\(50\)\.mean\(\))"
        replacement = r"\1\n    df['sma_200'] = df['close'].rolling(200).mean()  # 添加200日均线"

        content = re.sub(pattern, replacement, content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Added sma_200"
        else:
            return False, "Pattern not found"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Adding sma_200 calculation to all get_trading_signal_*.py files")
    print("=" * 80)

    # Find all get_trading*.py files
    current_dir = Path('.')
    files = list(current_dir.glob('get_trading_signal_*.py'))

    if not files:
        print("[ERROR] No get_trading_signal_*.py files found")
        return

    print(f"Found {len(files)} files to update\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for file_path in sorted(files):
        print(f"Updating {file_path.name}...", end=" ")
        success, message = add_sma_200(file_path)

        if success:
            print(f"[OK] {message}")
            success_count += 1
        else:
            if "Already has" in message:
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
    print(f"Updated:        {success_count}")
    print(f"Already OK:     {skip_count}")
    print(f"Errors:         {error_count}")
    print("=" * 80)

    if success_count > 0:
        print(f"\n[OK] Successfully updated {success_count} files!")
    else:
        print("\n[INFO] All files already have sma_200")

if __name__ == "__main__":
    main()
