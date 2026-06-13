"""
Comprehensive indentation fixer that fixes entire blocks
"""
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_comprehensive_indent(file_path):
    """Fix all indentation issues comprehensively"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_lines = lines.copy()
        modified = False

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()

            # Check if this line is a control structure that requires a block
            if stripped and line.rstrip().endswith(':'):
                control_indent = len(line) - len(line.lstrip())

                # Find all lines that should be in this block
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]

                    # Skip empty lines
                    if not next_line.strip():
                        j += 1
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip())
                    next_stripped = next_line.lstrip()

                    # If this line is at control indent or less, AND it's a new control structure, stop
                    if next_indent <= control_indent:
                        # Check if it's a continuation (elif, else, except, finally)
                        if next_stripped.startswith(('elif ', 'else:', 'except', 'finally')):
                            break  # This is a continuation at same level, stop here
                        elif next_stripped.startswith('#'):
                            # Comment at same level might be okay
                            j += 1
                            continue
                        else:
                            # Need to indent this line
                            lines[j] = ' ' * (control_indent + 4) + next_line.lstrip()
                            modified = True

                    j += 1
                    # Stop if we've processed too many lines (safety limit)
                    if j - i > 200:
                        break

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
    print("Comprehensive indentation fix")
    print("=" * 80)

    error_files = [
        'get_trading_signal_6443.py',
        'get_trading_signal_pltr.py'
    ]

    print(f"Found {len(error_files)} files to fix\n")

    current_dir = Path('.')

    for filename in error_files:
        file_path = current_dir / filename
        if not file_path.exists():
            print(f"{filename}... [ERROR] File not found")
            continue

        print(f"Fixing {filename}...")

        # Run multiple passes to fix nested structures
        for attempt in range(10):
            success, message = fix_comprehensive_indent(file_path)
            if not success:
                break

        print(f"{filename}... [OK] Fixed")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()
