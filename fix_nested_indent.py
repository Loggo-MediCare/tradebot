"""
Fix missing indentation in nested if/else blocks
"""
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_nested_indent(file_path):
    """Fix missing indentation in nested if/else blocks"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_lines = lines.copy()
        modified = False

        # Pattern 1: Fix indentation after "if sell_score == 0:" in nested context
        for i in range(len(lines) - 2):
            line = lines[i]
            next_line = lines[i + 1]

            # Check for "if sell_score == 0:" or "if sell_score > 0:" followed by unindented print
            if ('if sell_score == 0:' in line or 'if sell_score > 0:' in line) and i + 1 < len(lines):
                # Count indent of the if statement
                if_indent = len(line) - len(line.lstrip())
                next_indent = len(next_line) - len(next_line.lstrip())

                # If next line is print and has same or less indent, need to add indent
                if next_line.strip().startswith('print(') and next_indent <= if_indent:
                    lines[i + 1] = ' ' * (if_indent + 4) + next_line.lstrip()
                    modified = True

            # Check for "else:" followed by unindented print
            if line.strip() == 'else:' and i + 1 < len(lines):
                else_indent = len(line) - len(line.lstrip())
                next_indent = len(next_line) - len(next_line.lstrip())

                if next_line.strip().startswith('print(') and next_indent <= else_indent:
                    lines[i + 1] = ' ' * (else_indent + 4) + next_line.lstrip()
                    modified = True

            # Check for "for ... in ..." followed by unindented print
            if 'for ' in line and ' in ' in line and ':' in line and i + 1 < len(lines):
                for_indent = len(line) - len(line.lstrip())
                next_indent = len(next_line) - len(next_line.lstrip())

                if next_line.strip().startswith('print(') and next_indent <= for_indent:
                    lines[i + 1] = ' ' * (for_indent + 4) + next_line.lstrip()
                    modified = True

            # Check for "try:" followed by unindented code
            if line.strip() == 'try:' and i + 1 < len(lines):
                try_indent = len(line) - len(line.lstrip())
                next_indent = len(next_line) - len(next_line.lstrip())

                if next_line.strip() and next_indent <= try_indent and not next_line.strip().startswith('#'):
                    lines[i + 1] = ' ' * (try_indent + 4) + next_line.lstrip()
                    modified = True

        # Only write if content changed
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True, "Fixed nested indentation"
        else:
            return False, "No nested indentation issues"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Fixing nested indentation in get_trading*.py files")
    print("=" * 80)

    # Only fix the 4 files with errors
    error_files = [
        'get_trading_signal_2449.py',
        'get_trading_signal_6443.py',
        'get_trading_signal_8110.py',
        'get_trading_signal_pltr.py'
    ]

    print(f"Found {len(error_files)} files with errors to fix\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    current_dir = Path('.')

    for filename in error_files:
        file_path = current_dir / filename
        if not file_path.exists():
            print(f"Checking {filename}... [ERROR] File not found")
            error_count += 1
            continue

        print(f"Checking {filename}...", end=" ")
        success, message = fix_nested_indent(file_path)

        if success:
            print(f"[OK] {message}")
            success_count += 1
        else:
            if "No nested" in message:
                print(f"[SKIP] {message}")
                skip_count += 1
            else:
                print(f"[ERROR] {message}")
                error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files:    {len(error_files)}")
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
