"""
Fix indentation after skip_sell_scoring check
"""
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_sell_scoring_indent(file_path):
    """Fix indentation after skip_sell_scoring check"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_lines = lines.copy()
        modified = False

        # Find the line with "if not skip_sell_scoring:"
        for i, line in enumerate(lines):
            if 'if not skip_sell_scoring:' in line and i + 1 < len(lines):
                # Check if next line starts with "        signal = " (8 spaces instead of 12)
                next_line = lines[i + 1]
                if next_line.startswith('        signal = "卖出 (SELL)"'):
                    # Need to fix indentation for all lines in this block
                    # until we hit a line that's not part of the sell scoring block
                    j = i + 1
                    while j < len(lines):
                        current_line = lines[j]
                        # Stop if we hit another elif/else at same or lower indent level
                        if current_line.strip() and (
                            current_line.startswith('    elif ') or
                            current_line.startswith('    else:') or
                            current_line.startswith('    # ') and '=' not in current_line or
                            (current_line.startswith('        #') and '评分' in current_line)
                        ):
                            break

                        # Add 4 spaces of indentation if line starts with 8 spaces
                        if current_line.startswith('        ') and not current_line.startswith('            '):
                            lines[j] = '    ' + current_line
                            modified = True
                        elif current_line.startswith('            ') and not current_line.startswith('                '):
                            # Also indent lines that are already at 12 spaces to 16 spaces
                            lines[j] = '    ' + current_line
                            modified = True

                        j += 1
                    break

        # Only write if content changed
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True, "Fixed indentation"
        else:
            return False, "No indentation issues"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Fixing sell_scoring indentation in get_trading*.py files")
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
        success, message = fix_sell_scoring_indent(file_path)

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
