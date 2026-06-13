"""
Fix all remaining indentation issues by adding indent after if/elif/else/for statements
"""
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_all_indents(file_path):
    """Fix all indentation issues in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_lines = lines.copy()
        modified = False

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()

            # Check if this line ends with a colon (if/elif/else/for/while/try/except/with/def/class)
            if stripped and (stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'with ')) or
                           ('def ' in stripped and ':' in stripped) or ('class ' in stripped and ':' in stripped)):
                if line.rstrip().endswith(':'):
                    # Get the indent level of this control structure
                    control_indent = len(line) - len(line.lstrip())

                    # Check the next non-empty line
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1

                    if j < len(lines):
                        next_line = lines[j]
                        next_indent = len(next_line) - len(next_line.lstrip())

                        # If next line has same or less indent than control structure, it needs fixing
                        if next_line.strip() and next_indent <= control_indent:
                            # Add 4 spaces of indentation
                            lines[j] = ' ' * (control_indent + 4) + next_line.lstrip()
                            modified = True

            i += 1

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
    print("Fixing all remaining indentation issues")
    print("=" * 80)

    # Fix the 2 files with errors
    error_files = [
        'get_trading_signal_6443.py',
        'get_trading_signal_pltr.py'
    ]

    print(f"Found {len(error_files)} files to fix\n")

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

        print(f"Fixing {filename}...", end=" ")

        # Run the fix multiple times to catch nested issues
        total_fixed = False
        for attempt in range(5):  # Max 5 passes
            success, message = fix_all_indents(file_path)
            if success:
                total_fixed = True
            else:
                break  # No more changes needed

        if total_fixed:
            print(f"[OK] Fixed indentation")
            success_count += 1
        else:
            print(f"[SKIP] No issues found")
            skip_count += 1

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
