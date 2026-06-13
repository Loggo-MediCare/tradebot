"""
Fix indentation errors in get_trading*.py files
"""
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_indentation(file_path):
    """Fix indentation error in add_technical_indicators"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix the indentation error where bfill().ffill() has extra indent
        # Pattern: find the incorrectly indented line after calculate_obv
        pattern = r'(    df = calculate_obv\(df\)\s+\n)\s+(    df = df\.bfill\(\)\.ffill\(\))'

        if re.search(pattern, content):
            # Replace with correct indentation
            content = re.sub(pattern, r'\1\2', content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed indentation"
        else:
            return False, "No indentation errors"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Fixing indentation errors in get_trading*.py files")
    print("=" * 80)

    # Find all get_trading*.py files
    current_dir = Path('.')
    files = list(current_dir.glob('get_trading_signal_*.py'))

    if not files:
        print("[ERROR] No get_trading_signal_*.py files found")
        return

    print(f"Found {len(files)} files to check\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for file_path in sorted(files):
        print(f"Checking {file_path.name}...", end=" ")
        success, message = fix_indentation(file_path)

        if success:
            print(f"[OK] {message}")
            success_count += 1
        else:
            if "No indentation" in message:
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
        print("\n[INFO] All files have correct indentation")

if __name__ == "__main__":
    main()
