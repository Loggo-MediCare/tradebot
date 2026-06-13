"""
Fix broken string literals in explosion detection code
"""
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_broken_strings(file_path):
    """Fix broken string in explosion detection"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix the first broken print statement: print("\n🚀 主升段爆发行情侦测!")
        # This is currently broken as: print("\n (newline here) 🚀 主升段爆发行情侦测!")
        content = re.sub(
            r'print\("\s*\n🚀 主升段爆发行情侦测!"\)',
            r'print("\\n🚀 主升段爆发行情侦测!")',
            content
        )

        # Fix the second broken print statement: print("\n📌 操作策略:")
        # This is currently broken as: print("\n (newline here) 📌 操作策略:")
        content = re.sub(
            r'print\("\s*\n📌 操作策略:"\)',
            r'print("\\n📌 操作策略:")',
            content
        )

        # Fix the third broken print statement in summary: print("\n" + "=" * 80)
        # This is currently broken as: print("\n (newline here) " + "=" * 80)
        content = re.sub(
            r'print\("\s*\n" \+ "=" \* 80\)',
            r'print("\\n" + "=" * 80)',
            content
        )

        # Fix broken f-string: print(f"\n🔥 爆发行情数据:")
        # This is currently broken as: print(f"\n (newline here) 🔥 爆发行情数据:")
        content = re.sub(
            r'print\(f"\s*\n🔥 爆发行情数据:"\)',
            r'print(f"\\n🔥 爆发行情数据:")',
            content
        )

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed broken strings"
        else:
            return False, "No broken strings"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Fixing broken strings in get_trading*.py files")
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
        success, message = fix_broken_strings(file_path)

        if success:
            print(f"[OK] {message}")
            success_count += 1
        else:
            if "No broken" in message:
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
        print("\n[INFO] All files have correct strings")

if __name__ == "__main__":
    main()
