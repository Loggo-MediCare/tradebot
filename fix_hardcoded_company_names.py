"""
Remove hardcoded company name "(光聖)" from signal files
This was left over from the 6442 template
"""
import re
from pathlib import Path

def fix_company_name(file_path):
    """Remove hardcoded company name from signal file"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to find: 股票: {result['symbol']} (光聖)
    # Replace with: 股票: {result['symbol']}
    pattern = r"(股票: \{result\['symbol'\]\}) \(光聖\)"
    replacement = r"\1"

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all signal files with hardcoded company name"""
    print("=" * 80)
    print("Removing Hardcoded Company Names")
    print("=" * 80)

    signal_files = sorted(Path('.').glob('get_trading_signal_*.py'))

    fixed_count = 0
    skipped_count = 0

    for file_path in signal_files:
        try:
            if fix_company_name(file_path):
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
    print(f"Skipped: {skipped_count} files (no hardcoded name found)")
    print(f"Total: {len(signal_files)} files")
    print("=" * 80)

if __name__ == "__main__":
    main()
