"""
Fix broken comment in yfinance download call
"""
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_comment_syntax(file_path):
    """Fix broken comment in yf.download call"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix the broken comment in yf.download call
        # Pattern: period='300d', progress=False  # comment, auto_adjust=True)
        # Should be: period='300d', progress=False, auto_adjust=True)  # comment
        pattern = r"period='300d',\s*progress=False\s*#[^,\n]+,\s*auto_adjust=True\)"
        replacement = r"period='300d', progress=False, auto_adjust=True)  # 改为300天以计算200日均线"

        content = re.sub(pattern, replacement, content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed comment syntax"
        else:
            return False, "No comment issues"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Fixing broken comments in get_trading*.py files")
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
        success, message = fix_comment_syntax(file_path)

        if success:
            print(f"[OK] {message}")
            success_count += 1
        else:
            if "No comment" in message:
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
        print("\n[INFO] All files have correct comment syntax")

if __name__ == "__main__":
    main()
